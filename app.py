"""
Streamlit Interactive Dashboard for ArXiv Chip Architecture Paper Tracker.
"""

import os
import re
import sys
import io
import zipfile
import datetime
import streamlit as st
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.database import Database, Paper
from src.ingest import ArXivCrawler, SyncResult
from src.ai_filter import AIFilter
from src.embedding_similarity import EmbeddingSimilarityMatcher
from src.pdf_parser import PDFProcessor
from src.deep_synthesis import DeepSynthesizer


load_dotenv()

# Configuration
DB_PATH = os.getenv('DB_PATH', './data/papers.db')
PDF_DIR = os.getenv('PDF_DIR', './pdfs')
MIN_SCORE_DEFAULT = 7
PAPER_DONE_MARKER_PREFIX = "<!-- PAPER_DONE:"
PAPER_DONE_MARKER_SUFFIX = " -->"
FALLBACK_PAPER_FIGURES_CACHE: dict[str, list[dict]] = {}
_embedding_matcher: EmbeddingSimilarityMatcher | None = None
SEMANTIC_HINT_GROUPS = {
    'architecture': (
        '架构', '组件', '模块', '拓扑', '目录', '客户端', '系统设计', '体系结构',
        'architecture', 'architectural', 'component', 'components', 'topology', 'client', 'directory', 'overview'
    ),
    'protocol': (
        '协议', '状态机', '一致性', '状态流转', '共享状态',
        'protocol', 'coherence', 'state machine', 'state transition', 'directory state'
    ),
    'experiment': (
        '实验', '评估', '测试', '性能', '吞吐', '时延', '延迟', '带宽', '速度提升',
        'experiment', 'evaluation', 'benchmark', 'latency', 'bandwidth', 'throughput', 'iops', 'speedup', 'performance'
    ),
    'setup': (
        '实验设置', '平台', '环境', '配置', '仿真', '模拟',
        'setup', 'environment', 'configuration', 'emulation'
    ),
    'memory': (
        '缓存', '页', '页缓存', '远程内存', '共享内存', 'dram', 'cxl',
        'cache', 'page', 'memory', 'dram', 'cxl', 'remote memory'
    ),
}


def init_session_state():
    """Initialize session state variables."""
    if 'selected_papers' not in st.session_state:
        st.session_state.selected_papers = set()
    if 'sync_running' not in st.session_state:
        st.session_state.sync_running = False
    if 'ai_process_running' not in st.session_state:
        st.session_state.ai_process_running = False
    if 'synthesis_result' not in st.session_state:
        st.session_state.synthesis_result = None
    if 'synthesis_entries' not in st.session_state:
        st.session_state.synthesis_entries = []
    if 'generating_synthesis' not in st.session_state:
        st.session_state.generating_synthesis = False
    if 'saved_synthesis_paper_ids' not in st.session_state:
        st.session_state.saved_synthesis_paper_ids = set()
    if 'paper_figures_cache' not in st.session_state:
        st.session_state.paper_figures_cache = {}


def get_paper_figures_cache() -> dict[str, list[dict]]:
    """Return the figure cache, even when code runs outside a Streamlit session."""
    try:
        if 'paper_figures_cache' not in st.session_state:
            st.session_state.paper_figures_cache = {}
        return st.session_state.paper_figures_cache
    except Exception:
        return FALLBACK_PAPER_FIGURES_CACHE


def get_embedding_matcher() -> EmbeddingSimilarityMatcher:
    """Create or reuse the embedding matcher."""
    global _embedding_matcher
    if _embedding_matcher is None:
        _embedding_matcher = EmbeddingSimilarityMatcher()
    return _embedding_matcher


def get_db():
    """Get database connection."""
    if 'db' not in st.session_state:
        st.session_state.db = Database(DB_PATH)
    return st.session_state.db


def format_authors(authors: str, max_len: int = 45) -> str:
    """Format authors for display - shorter to fit metadata in one line."""
    if len(authors) <= max_len:
        return authors
    return authors[:max_len] + "..."


def get_paper_done_marker(paper_id: str) -> str:
    """Create an invisible marker so resume logic can count completed papers exactly."""
    return f"{PAPER_DONE_MARKER_PREFIX}{paper_id}{PAPER_DONE_MARKER_SUFFIX}"


def get_completed_paper_ids(content: str) -> set[str]:
    """Extract completed paper IDs from the persisted partial synthesis."""
    completed_paper_ids = set()
    for line in content.splitlines():
        if line.startswith(PAPER_DONE_MARKER_PREFIX) and line.endswith(PAPER_DONE_MARKER_SUFFIX):
            completed_paper_ids.add(line[len(PAPER_DONE_MARKER_PREFIX):-len(PAPER_DONE_MARKER_SUFFIX)])
    return completed_paper_ids


def normalize_synthesis_entry(entry: str) -> str:
    """Normalize a single synthesis entry for standalone display."""
    normalized_entry = entry.strip()
    while normalized_entry.endswith("---"):
        normalized_entry = normalized_entry[:-3].rstrip()

    lines = normalized_entry.splitlines()
    if lines and re.match(r'^##\s+\d+\.\s+', lines[0]):
        lines[0] = re.sub(r'^##\s+\d+\.\s+', '## ', lines[0], count=1)
    return "\n".join(lines).strip()


def parse_synthesis_entries(content: str) -> list[str]:
    """Parse stored synthesis content into standalone per-paper entries."""
    if not content:
        return []

    if PAPER_DONE_MARKER_PREFIX in content:
        entries = []
        current_lines = []
        capturing = False

        for line in content.splitlines():
            if line.startswith(PAPER_DONE_MARKER_PREFIX) and line.endswith(PAPER_DONE_MARKER_SUFFIX):
                if current_lines:
                    entries.append(normalize_synthesis_entry("\n".join(current_lines)))
                current_lines = []
                capturing = True
                continue

            if capturing:
                current_lines.append(line)

        if current_lines:
            entries.append(normalize_synthesis_entry("\n".join(current_lines)))

        return [entry for entry in entries if entry]

    heading_pattern = re.compile(r'(?m)^##\s+(?:\d+\.\s+)?[^\n]+\n\n\*\*作者\*\*:')
    matches = list(heading_pattern.finditer(content))
    if not matches:
        return [normalize_synthesis_entry(content)] if content.strip() else []

    entries = []
    for index, match in enumerate(matches):
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        entry = normalize_synthesis_entry(content[start:end])
        if entry:
            entries.append(entry)
    return entries


def get_synthesis_entry_title(entry: str) -> str:
    """Extract the paper title from a single synthesis entry."""
    first_line = entry.strip().splitlines()[0] if entry.strip() else ""
    if first_line.startswith("## "):
        return first_line[3:].strip()
    return "深度推文"


def build_synthesis_entry(paper: Paper, body: str, include_arxiv_link: bool = False) -> str:
    """Build the markdown entry for a single paper."""
    entry = f"## {paper.title}\n\n"
    entry += f"**作者**: {paper.authors}\n\n"
    entry += f"评分: **{paper.ai_score}/10** 标签: {paper.ai_tags}\n\n"
    if include_arxiv_link:
        entry += f"ArXiv: [{paper.id}]({paper.pdf_url})\n\n"
    else:
        entry += f"原文链接: {paper.pdf_url}\n\n"
    entry += "---\n\n"
    entry += body.strip()
    return entry


def extract_paper_id_from_entry(entry: str) -> str | None:
    """Extract the paper ID from a synthesis entry."""
    arxiv_link_match = re.search(r'ArXiv:\s*\[([^\]]+)\]\(', entry)
    if arxiv_link_match:
        return arxiv_link_match.group(1).strip()

    pdf_link_match = re.search(r'https?://[^\s)]+/(?:pdf|abs)/([A-Za-z0-9._-]+)', entry)
    if not pdf_link_match:
        return None

    paper_id = pdf_link_match.group(1).strip().rstrip('/')
    if paper_id.lower().endswith('.pdf'):
        paper_id = paper_id[:-4]
    return paper_id or None


def get_entry_figure_refs(entry: str) -> list[str]:
    """Return unique figure references in first-appearance order."""
    figure_refs = []
    seen_refs = set()
    for match in re.finditer(r'\[(图\d+)\]', entry):
        figure_key = match.group(1)
        if figure_key not in seen_refs:
            seen_refs.add(figure_key)
            figure_refs.append(figure_key)
    return figure_refs


def is_content_block(block: str) -> bool:
    """Return whether a markdown block is substantive content for figure placement."""
    stripped = block.strip()
    if not stripped:
        return False
    if stripped.startswith('## '):
        return False
    if stripped.startswith('**作者**'):
        return False
    if stripped.startswith('评分:'):
        return False
    if stripped.startswith('ArXiv:'):
        return False
    if stripped.startswith('原文链接:'):
        return False
    if stripped == '---':
        return False
    return len(stripped) >= 12


def build_block_embedding_text(block: str) -> str:
    """Prepare a paragraph for embedding similarity scoring."""
    return f"推文段落：{block.strip()}"


def build_figure_embedding_text(figure: dict) -> str:
    """Prepare a figure caption for embedding similarity scoring."""
    return f"论文图注：{figure['caption']}"


def extract_similarity_tokens(text: str) -> set[str]:
    """Extract lightweight keyword tokens for fallback matching."""
    normalized = text.lower()
    english_tokens = set(re.findall(r'[a-z][a-z0-9_+-]{1,}', normalized))
    numeric_tokens = set(re.findall(r'\b\d+(?:\.\d+)?(?:x|kb|mb|gb|tb|ns|us|ms|s|%)?\b', normalized))
    acronym_tokens = {
        token for token in english_tokens
        if token.isupper() or token in {'cxl', 'dram', 'nvme', 'fio', 'iops'}
    }
    return english_tokens | numeric_tokens | acronym_tokens


def get_semantic_hints(text: str) -> set[str]:
    """Map text to coarse semantic labels for fallback matching."""
    normalized = text.lower()
    hints = set()
    for label, keywords in SEMANTIC_HINT_GROUPS.items():
        for keyword in keywords:
            if keyword in text or keyword in normalized:
                hints.add(label)
                break
    return hints


def score_block_figure_fallback(block: str, figure: dict) -> float:
    """Fallback score when embeddings are unavailable."""
    if not is_content_block(block):
        return -1.0

    block_tokens = extract_similarity_tokens(block)
    caption_tokens = extract_similarity_tokens(figure['caption'])
    token_overlap = block_tokens & caption_tokens

    block_hints = get_semantic_hints(block)
    caption_hints = get_semantic_hints(figure['caption'])
    hint_overlap = block_hints & caption_hints

    score = 0.0
    score += len(token_overlap) * 2.2
    score += len(hint_overlap) * 4.5

    if block.strip().startswith('###'):
        score -= 1.0
    if len(block) > 120:
        score += 0.4
    return score


def plan_entry_figures(entry: str, figures: list[dict]) -> tuple[list[str], dict[int, list[dict]], list[dict]]:
    """Plan figure placement for an entry using explicit refs first, then embedding similarity."""
    blocks = [block.strip() for block in re.split(r'\n{2,}', entry.strip()) if block.strip()]
    assignments = {index: [] for index in range(len(blocks))}
    figure_map = {figure['figure_key']: figure for figure in figures}
    assigned_figure_keys = set()

    for index, block in enumerate(blocks):
        for figure_key in get_entry_figure_refs(block):
            if figure_key in figure_map and figure_key not in assigned_figure_keys:
                assignments[index].append(figure_map[figure_key])
                assigned_figure_keys.add(figure_key)

    candidate_indices = [index for index, block in enumerate(blocks) if is_content_block(block)]
    remaining_figures = [
        figure for figure in figures
        if figure['figure_key'] not in assigned_figure_keys
    ]
    unmatched_figures = []

    if not candidate_indices or not remaining_figures:
        unmatched_figures.extend(remaining_figures)
        return blocks, assignments, unmatched_figures

    matcher = get_embedding_matcher()
    similarity_matrix = None
    if matcher.is_configured():
        block_texts = [build_block_embedding_text(blocks[index]) for index in candidate_indices]
        figure_texts = [build_figure_embedding_text(figure) for figure in remaining_figures]
        similarity_matrix = matcher.similarity_matrix(block_texts, figure_texts)

    if similarity_matrix is None:
        for figure in remaining_figures:
            best_index = None
            best_score = 0.0
            for index in candidate_indices:
                score = score_block_figure_fallback(blocks[index], figure)
                if score > best_score:
                    best_index = index
                    best_score = score

            if best_index is not None and best_score >= 4.5:
                assignments[best_index].append(figure)
            else:
                unmatched_figures.append(figure)
        return blocks, assignments, unmatched_figures

    scored_assignments: dict[int, list[tuple[float, dict]]] = {index: [] for index in range(len(blocks))}
    for figure_position, figure in enumerate(remaining_figures):
        best_candidate_position = None
        best_score = float('-inf')
        for candidate_position, candidate_index in enumerate(candidate_indices):
            score = similarity_matrix[candidate_position][figure_position]
            if score > best_score:
                best_candidate_position = candidate_index
                best_score = score

        if best_candidate_position is not None and best_score >= matcher.similarity_threshold:
            scored_assignments[best_candidate_position].append((best_score, figure))
            assigned_figure_keys.add(figure['figure_key'])
        else:
            unmatched_figures.append(figure)

    for block_index, scored_figures in scored_assignments.items():
        if not scored_figures:
            continue
        scored_figures.sort(key=lambda item: item[0], reverse=True)
        assignments[block_index].extend(figure for _, figure in scored_figures)

    return blocks, assignments, unmatched_figures


def make_safe_package_name(value: str) -> str:
    """Create a filesystem-safe name for markdown package files."""
    safe_name = re.sub(r'[^\w\u4e00-\u9fff.-]+', '-', value).strip('-.')
    return safe_name or 'synthesis'


def build_entry_markdown_with_images(entry: str, figures: list[dict], assets_dir: str) -> tuple[str, list[dict]]:
    """Insert relative image links into markdown using the same placement plan as the UI."""
    blocks, assignments, unmatched_figures = plan_entry_figures(entry, figures)
    output_lines = []
    included_figures = []

    for index, block in enumerate(blocks):
        output_lines.append(block)
        output_lines.append('')
        for figure in assignments.get(index, []):
            image_name = os.path.basename(figure['image_path'])
            relative_path = f"{assets_dir}/{image_name}"
            output_lines.append(f"![{figure['figure_key']} - {figure['caption']}]({relative_path})")
            output_lines.append('')
            output_lines.append(f"> {figure['figure_key']} · 第{figure['page']}页 · {figure['caption']}")
            output_lines.append('')
            included_figures.append(figure)

    if unmatched_figures:
        output_lines.append('## 配图补充')
        output_lines.append('')
        for figure in unmatched_figures:
            image_name = os.path.basename(figure['image_path'])
            relative_path = f"{assets_dir}/{image_name}"
            output_lines.append(f"![{figure['figure_key']} - {figure['caption']}]({relative_path})")
            output_lines.append('')
            output_lines.append(f"> {figure['figure_key']} · 第{figure['page']}页 · {figure['caption']}")
            output_lines.append('')
            included_figures.append(figure)

    return '\n'.join(output_lines).strip() + '\n', included_figures


def build_markdown_package(entries: list[str], db: Database, pdf_processor: PDFProcessor) -> bytes:
    """Create a zip package with per-entry markdown and related images."""
    package_buffer = io.BytesIO()
    readme_lines = ['# ArXiv 深度推文 Markdown 包', '', '包含内容：', '']
    combined_entries = []
    written_assets = set()

    with zipfile.ZipFile(package_buffer, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
        for index, entry in enumerate(entries, 1):
            paper_id = extract_paper_id_from_entry(entry) or f'entry-{index:02d}'
            title = get_synthesis_entry_title(entry)
            figures = get_paper_figures(paper_id, db, pdf_processor)
            assets_dir = f"assets/{make_safe_package_name(paper_id)}"
            markdown_text, used_figures = build_entry_markdown_with_images(entry, figures, assets_dir)
            markdown_name = f"{index:02d}-{make_safe_package_name(paper_id)}.md"

            archive.writestr(markdown_name, markdown_text)
            combined_entries.append(markdown_text)
            readme_lines.append(f"- {markdown_name} : {title}")

            for figure in used_figures:
                asset_name = f"{assets_dir}/{os.path.basename(figure['image_path'])}"
                if asset_name in written_assets:
                    continue
                if not os.path.exists(figure['image_path']):
                    continue
                with open(figure['image_path'], 'rb') as image_file:
                    archive.writestr(asset_name, image_file.read())
                written_assets.add(asset_name)

        archive.writestr('README.md', '\n'.join(readme_lines).strip() + '\n')
        archive.writestr('all.md', '\n\n---\n\n'.join(combined_entries).strip() + '\n')

    package_buffer.seek(0)
    return package_buffer.getvalue()


def get_paper_figures(paper_id: str | None, db: Database, pdf_processor: PDFProcessor) -> list[dict]:
    """Load extracted figures for a paper with a small session cache."""
    if not paper_id:
        return []

    figure_cache = get_paper_figures_cache()
    cached_figures = figure_cache.get(paper_id)
    if cached_figures is not None:
        return cached_figures

    paper = db.get_paper(paper_id)
    if paper is None:
        figure_cache[paper_id] = []
        return []

    figures = pdf_processor.extract_figures(paper)
    figure_cache[paper_id] = figures
    return figures


def render_synthesis_entry(entry: str, figures: list[dict]) -> None:
    """Render a synthesis entry and insert related figures near their references."""
    blocks, assignments, remaining_figures = plan_entry_figures(entry, figures)

    for index, block in enumerate(blocks):
        st.markdown(block)
        for figure in assignments.get(index, []):
            st.image(
                figure['image_path'],
                caption=f"{figure['figure_key']} · 第{figure['page']}页 · {figure['caption']}",
                use_container_width=True,
            )

    if remaining_figures:
        with st.expander("🖼️ 论文配图", expanded=False):
            for figure in remaining_figures:
                st.image(
                    figure['image_path'],
                    caption=f"{figure['figure_key']} · 第{figure['page']}页 · {figure['caption']}",
                    use_container_width=True,
                )


def get_selected_paper_ids_in_order(papers: list[Paper]) -> list[str]:
    """Sync checkbox widget state into session state and return a stable paper order."""
    visible_paper_ids = {paper.id for paper in papers}
    ordered_selected_ids = []

    for paper in papers:
        checkbox_key = f"select_{paper.id}"
        default_selected = paper.id in st.session_state.selected_papers
        if st.session_state.get(checkbox_key, default_selected):
            ordered_selected_ids.append(paper.id)

    hidden_selected_ids = sorted(
        paper_id
        for paper_id in st.session_state.selected_papers
        if paper_id not in visible_paper_ids
    )

    all_selected_ids = ordered_selected_ids + hidden_selected_ids
    st.session_state.selected_papers = set(all_selected_ids)
    return all_selected_ids


def main():
    st.set_page_config(
        page_title="ArXiv 芯片架构与 EDA 前沿追踪",
        page_icon="📚",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for better styling
    st.markdown("""
    <style>
    /* Improve container spacing - paper card styling */
    .stContainer {
        padding-top: 0.8rem;
        padding-bottom: 0.8rem;
        border-radius: 8px;
        border: 1px solid #e5e7eb;
        margin-bottom: 0.5rem;
        background-color: rgba(255, 255, 255, 0.02);
        transition: all 0.2s;
    }

    .stContainer:hover {
        border-color: #9ca3af;
        background-color: rgba(255, 255, 255, 0.04);
    }

    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .stContainer {
            border-color: #374151;
        }
        .stContainer:hover {
            border-color: #6b7280;
        }
    }

    /* Tag pills */
    .tag-pill {
        display: inline-block;
        background: rgba(99, 102, 241, 0.12);
        color: #a5b4fc;
        border: 1px solid rgba(99, 102, 241, 0.28);
        border-radius: 100px;
        padding: 1px 9px;
        font-size: 0.73rem;
        white-space: nowrap;
        margin: 0 2px;
        vertical-align: middle;
    }

    /* Score badges */
    .score-badge {
        display: inline-block;
        border-radius: 100px;
        padding: 1px 10px;
        font-size: 0.78rem;
        font-weight: 700;
        vertical-align: middle;
        letter-spacing: 0.01em;
    }
    .score-s9  { background: rgba(239,68,68,0.18);  color: #fca5a5; border: 1px solid rgba(239,68,68,0.35); }
    .score-s7  { background: rgba(249,115,22,0.18); color: #fdba74; border: 1px solid rgba(249,115,22,0.35); }
    .score-s5  { background: rgba(234,179,8,0.18);  color: #fde047; border: 1px solid rgba(234,179,8,0.35); }
    .score-low { background: rgba(99,102,241,0.14); color: #a5b4fc; border: 1px solid rgba(99,102,241,0.3); }
    .score-na  { background: rgba(156,163,175,0.14); color: #9ca3af; border: 1px solid rgba(156,163,175,0.3); }

    /* Metadata row */
    .meta-row {
        display: flex;
        align-items: center;
        flex-wrap: wrap;
        gap: 6px;
        margin-bottom: 0.45rem;
        font-size: 0.85rem;
        line-height: 1.6;
    }
    .meta-authors {
        color: #9ca3af;
        font-size: 0.82rem;
    }
    .meta-sep { color: #4b5563; font-size: 0.9rem; }

    /* Action column: stack checkbox + star button tightly (scoped to paper cards only) */
    .stContainer [data-testid="column"]:last-child {
        display: flex;
        flex-direction: column;
        align-items: center;
        padding-top: 0.3rem;
    }
    .stContainer [data-testid="column"]:last-child > div {
        margin-bottom: 0 !important;
    }
    .stContainer [data-testid="column"]:last-child .stButton > button {
        padding: 0.15rem 0.5rem;
        min-width: 2.4rem;
        font-size: 1rem;
    }

    /* Improve heading spacing */
    h1 {
        margin-bottom: 1.5rem;
    }

    h3 {
        margin-top: 0.1rem !important;
        margin-bottom: 0.5rem !important;
    }

    /* Improve divider spacing */
    hr {
        margin: 1.5rem 0;
    }

    /* Remove problematic global button CSS that squishes everything */
    </style>
    """, unsafe_allow_html=True)

    init_session_state()
    db = get_db()
    ai_filter = AIFilter()
    pdf_processor = PDFProcessor(PDF_DIR)
    synthesizer = DeepSynthesizer()

    # Sidebar
    with st.sidebar:
        st.title("🎯 ArXiv 追踪系统")
        st.markdown("*芯片架构与 EDA 前沿论文 AI 精选*")

        st.divider()

        # Stats
        stats = db.get_stats()
        st.subheader("📊 统计信息")
        # Use metric cards for better visual appeal
        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.metric("总论文数", stats['total_papers'])
            st.metric("待处理", stats['unprocessed_papers'])
            st.metric("已收藏", stats['starred_papers'])
        with col_stat2:
            st.metric("已AI评分", stats['processed_papers'])
            st.metric("今日新增", stats['today_added'])
            if stats['average_score']:
                st.metric("平均分", f"{stats['average_score']}")

        st.divider()

        # Manual sync button
        st.subheader("🔄 数据同步")
        crawler = ArXivCrawler(db)
        synced_today = crawler.check_todays_sync_done()
        if synced_today:
            st.success("✅ 今日已完成同步")
        else:
            st.warning("⚠️ 今日尚未同步")

        last_sync = db.get_last_sync_time()
        if last_sync:
            try:
                last_sync_dt = datetime.datetime.fromisoformat(last_sync)
                st.caption(f"上次同步: {last_sync_dt.strftime('%Y-%m-%d %H:%M')}")
            except:
                st.caption(f"上次同步: {last_sync}")

        active_categories = ', '.join(crawler.categories)
        st.caption(f"追踪分类: {active_categories}")

        if not synced_today:
            if st.button("🚀 补跑今日同步", disabled=st.session_state.sync_running, use_container_width=True):
                st.session_state.sync_running = True
                with st.spinner("正在执行今日补同步..."):
                    result: SyncResult = crawler.sync()
                    if result.success:
                        st.success(f"补同步完成！新增 {result.papers_added} 篇，更新 {result.papers_updated} 篇")
                    else:
                        st.error(f"补同步失败: {result.error_message}")
                st.session_state.sync_running = False
                st.rerun()

        if st.button("🔄 立即同步最新论文", disabled=st.session_state.sync_running, use_container_width=True):
            st.session_state.sync_running = True
            with st.spinner("正在同步 ArXiv... 这可能需要几分钟时间"):
                result: SyncResult = crawler.sync()
                if result.success:
                    st.success(f"同步完成！新增 {result.papers_added} 篇，更新 {result.papers_updated} 篇")
                else:
                    st.error(f"同步失败: {result.error_message}")
            st.session_state.sync_running = False
            st.rerun()

        st.divider()

        # AI processing
        st.subheader("🤖 AI 评分")
        pending = db.count_unprocessed()
        st.caption(f"待处理: {pending} 篇")

        if not ai_filter.is_configured():
            st.warning("⚠️ API 密钥未配置，请检查 .env 文件")

        batch_size = st.slider("批次处理数量", 1, 50, 10)

        if st.button("🤖 处理待评分论文", disabled=st.session_state.ai_process_running or not ai_filter.is_configured(), use_container_width=True):
            st.session_state.ai_process_running = True
            processed = 0
            progress_bar = st.progress(0)
            status_text = st.empty()

            for i in range(min(batch_size, pending)):
                status_text.text(f"正在处理第 {i+1}/{min(batch_size, pending)} 篇...")
                processed_batch = ai_filter.process_next_batch(db, batch_size=1, delay_seconds=1.5)
                processed += processed_batch
                progress_bar.progress((i + 1) / min(batch_size, pending))

            status_text.text(f"完成！共处理 {processed} 篇论文")
            st.session_state.ai_process_running = False
            st.rerun()

        st.divider()

        # PDF storage info
        pdf_stats = pdf_processor.get_storage_stats()
        st.subheader("📄 PDF 存储")
        st.caption(f"已下载: {pdf_stats['file_count']} 个文件")
        st.caption(f"占用空间: {pdf_stats['total_size_mb']:.1f} MB")

        st.divider()

        # Saved syntheses history
        st.subheader("📜 历史生成记录")
        synth_count = db.count_syntheses()
        if synth_count > 0:
            st.caption(f"共保存了 {synth_count} 条记录")
            recent = db.get_recent_syntheses(limit=50)
            
            # Map ID to summary
            synth_options = {}
            for synth in recent:
                paper_ids = synth['paper_ids'] or []
                lead_id = paper_ids[0] if paper_ids else '未知'
                try:
                    created_at = datetime.datetime.fromisoformat(synth['created_at']).strftime('%m-%d %H:%M')
                except ValueError:
                    created_at = synth['created_at'][:16].replace('T', ' ')
                paper_count = len(paper_ids)
                if paper_count > 1:
                    summary_text = f"{created_at} | {lead_id} 等{paper_count}篇"
                else:
                    summary_text = f"{created_at} | {lead_id}"
                synth_options[synth['id']] = summary_text

            selected_synth_id = st.selectbox(
                "选择历史记录", 
                options=list(synth_options.keys()), 
                format_func=lambda x: synth_options[x],
                label_visibility="collapsed"
            )
            
            if selected_synth_id:
                selected_synth = next((s for s in recent if s['id'] == selected_synth_id), None)
                if selected_synth:
                    col1, col2, col3 = st.columns(3, vertical_alignment="center")
                    with col1:
                        if st.button("📂", key="load_history", use_container_width=True, help="在主界面加载此推文"):
                            st.session_state.synthesis_result = selected_synth['content']
                            st.session_state.synthesis_entries = parse_synthesis_entries(selected_synth['content'])
                            st.rerun()
                    with col2:
                        st.download_button(
                            label="⬇️",
                            data=selected_synth['content'],
                            file_name=f"arxiv-synthesis-{selected_synth['id']}.md",
                            mime="text/markdown",
                            key="dl_history",
                            use_container_width=True,
                            help="下载Markdown文件"
                        )
                    with col3:
                        if st.button("🗑️", key="del_history", use_container_width=True, help="删除此记录"):
                            db.delete_synthesis(selected_synth['id'])
                            st.rerun()
        else:
            st.caption("暂无保存的生成记录")

    # Main content
    st.title("📚 ArXiv 芯片架构与 EDA 前沿论文")

    # Filters - better balanced proportions
    st.subheader("🔍 筛选条件")
    col1, col2, col3, col4 = st.columns([1.2, 1.5, 1.2, 2.1], vertical_alignment="center")
    with col1:
        filter_today = st.checkbox("仅今日新增", value=False)
    with col2:
        filter_high_score = st.checkbox("仅高分必读 (≥7分)", value=True)
    with col3:
        filter_starred = st.checkbox("仅已收藏", value=False)
    with col4:
        search_query = st.text_input("搜索标题/作者/标签", placeholder="输入关键词搜索...")

    # Build filter parameters
    min_score = 7 if filter_high_score else None
    only_today = filter_today
    only_starred = filter_starred
    search = search_query if search_query else None

    # Get filtered papers
    total_filtered = db.count_filtered(
        only_today=only_today,
        min_score=min_score,
        only_starred=only_starred,
        search=search
    )
    papers = db.get_filtered_papers(
        only_today=only_today,
        min_score=min_score,
        only_starred=only_starred,
        search=search,
        limit=100
    )

    st.write(f"**找到 {total_filtered} 篇论文**，显示前 100 篇")

    # Pre-fetch all recent syntheses once to avoid N+1 queries inside the paper loop
    _recent_synths = db.get_recent_syntheses(limit=20) if db.count_syntheses() > 0 else []
    synth_by_paper: dict = {}
    for _s in _recent_synths:
        for _pid in _s['paper_ids']:
            synth_by_paper.setdefault(_pid, []).append(_s)

    if not papers:
        st.info("没有找到符合条件的论文。请先点击侧边栏的「立即同步最新论文」获取数据。")
        return

    # Display papers in a table-like format with checkboxes
    st.divider()

    # Action bar for selected papers - TOP so you don't need to scroll to bottom
    selected_paper_ids = get_selected_paper_ids_in_order(papers)
    selected_count = len(selected_paper_ids)
    col_a, col_b = st.columns([3, 1], vertical_alignment="center")
    with col_a:
        st.markdown(f"##### 已选择 **{selected_count}** 篇论文")
    with col_b:
        if selected_count > 0 and not st.session_state.generating_synthesis:
            if st.button("📝 生成深度推文", use_container_width=True):
                if 'partial_synthesis' in st.session_state:
                    del st.session_state.partial_synthesis
                st.session_state.synthesis_entries = []
                st.session_state.synthesis_result = None
                st.session_state.saved_synthesis_paper_ids = set()
                st.session_state.synthesis_queue = selected_paper_ids
                st.session_state.generating_synthesis = True
                st.rerun()
        elif st.session_state.generating_synthesis:
            st.button("📝 正在生成中...", disabled=True, use_container_width=True)

    # Show warning once
    if st.session_state.generating_synthesis:
        st.warning("⚠️ 生成过程中请勿刷新或重复点击，刷新会中断生成，但是已完成内容会保存可断点续传。")

    # Actual generation runs when generating_synthesis is True - TOP so you see progress/result immediately
    if st.session_state.generating_synthesis:
        queued_paper_ids = st.session_state.get('synthesis_queue') or selected_paper_ids
        selected_paper_objs = [p for pid in queued_paper_ids if (p := db.get_paper(pid))]
        progress_bar = st.progress(0)
        status_text = st.empty()
        completed_paper_ids = set()
        saved_paper_ids = set(st.session_state.saved_synthesis_paper_ids)
        synthesis_entries = []
        # Resume from partial result in session if we have it (interrupted by refresh)
        if 'partial_synthesis' in st.session_state:
            synthesis_result = st.session_state.partial_synthesis
            completed_paper_ids = get_completed_paper_ids(synthesis_result)
            completed_paper_ids.update(saved_paper_ids)
            synthesis_entries = parse_synthesis_entries(synthesis_result)
            if completed_paper_ids:
                st.info(f"恢复之前生成到一半的结果，已完成 {len(completed_paper_ids)}/{len(selected_paper_objs)} 篇...")
        else:
            synthesis_result = f"# ArXiv 芯片架构前沿精选\n\n"
            synthesis_result += f"生成时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            synthesis_result += "---\n\n"
            st.session_state.partial_synthesis = synthesis_result

        for i, paper in enumerate(selected_paper_objs, 1):
            # Skip papers already completed in a previous partial run.
            if paper.id in completed_paper_ids:
                progress_bar.progress(i / len(selected_paper_objs))
                continue

            progress = i / len(selected_paper_objs)
            status_text.text(f"正在处理第 {i}/{len(selected_paper_objs)} 篇: {paper.title[:50]}...")
            progress_bar.progress(progress)
            paper_marker = get_paper_done_marker(paper.id)
            entry_body = ""
            include_arxiv_link = False

            # Get full text from PDF
            full_text = pdf_processor.get_or_download(paper)
            if full_text is None:
                entry_body = "PDF 下载失败，无法生成深度分析。"
            else:
                figures = pdf_processor.extract_figures(paper)
                get_paper_figures_cache()[paper.id] = figures
                synthesis = synthesizer.synthesize_paper(paper, full_text, figures=figures)
                if synthesis is None:
                    entry_body = "AI 生成失败。"
                else:
                    entry_body = synthesis
                    include_arxiv_link = True

            entry_content = build_synthesis_entry(
                paper,
                entry_body,
                include_arxiv_link=include_arxiv_link,
            )

            db.save_synthesis([paper.id], entry_content)
            saved_paper_ids.add(paper.id)
            st.session_state.saved_synthesis_paper_ids = saved_paper_ids

            synthesis_result += f"{paper_marker}\n"
            synthesis_result += entry_content
            synthesis_result += "\n\n---\n\n"
            synthesis_entries.append(entry_content)
            completed_paper_ids.add(paper.id)
            st.session_state.partial_synthesis = synthesis_result

        progress_bar.progress(1.0)
        status_text.empty()
        st.session_state.synthesis_result = synthesis_result
        st.session_state.synthesis_entries = synthesis_entries
        # Clear partial result
        if 'partial_synthesis' in st.session_state:
            del st.session_state.partial_synthesis
        st.success(f"生成完成！已逐篇保存到数据库，共处理 {len(selected_paper_objs)} 篇论文")
        # Reset generating state
        st.session_state.generating_synthesis = False
        if 'synthesis_queue' in st.session_state:
            del st.session_state.synthesis_queue
        st.session_state.saved_synthesis_paper_ids = set()
        # Auto-offer download immediately so it's not lost on refresh
        filename = f"arxiv-selection-{datetime.datetime.now().strftime('%Y%m%d')}.md"
        st.download_button(
            label="⬇️ 立即下载 Markdown",
            data=synthesis_result,
            file_name=filename,
            mime="text/markdown"
        )

    # Display existing synthesis results as separate entries.
    if st.session_state.synthesis_entries:
        st.subheader("📄 生成的深度推文")
        for index, entry in enumerate(st.session_state.synthesis_entries, 1):
            entry_title = get_synthesis_entry_title(entry)
            entry_paper_id = extract_paper_id_from_entry(entry)
            entry_figures = get_paper_figures(entry_paper_id, db, pdf_processor)
            with st.expander(entry_title, expanded=index == 1):
                render_synthesis_entry(entry, entry_figures)
                entry_package = build_markdown_package([entry], db, pdf_processor)
                st.download_button(
                    label="⬇️ 下载当前 Markdown",
                    data=entry,
                    file_name=f"arxiv-synthesis-{index}-{datetime.datetime.now().strftime('%Y%m%d')}.md",
                    mime="text/markdown",
                    key=f"download_entry_{index}"
                )
                st.download_button(
                    label="🗂️ 下载当前 Markdown 包",
                    data=entry_package,
                    file_name=f"arxiv-synthesis-{index}-{datetime.datetime.now().strftime('%Y%m%d')}.zip",
                    mime="application/zip",
                    key=f"download_entry_package_{index}"
                )

        action_col1, action_col2 = st.columns([1, 3])
        with action_col1:
            if st.button("🗑️ 清空结果"):
                st.session_state.synthesis_result = None
                st.session_state.synthesis_entries = []
                st.rerun()
        with action_col2:
            all_package = build_markdown_package(st.session_state.synthesis_entries, db, pdf_processor)
            package_col1, package_col2 = st.columns(2)
            with package_col1:
                st.download_button(
                    label="🗂️ 下载 Markdown 打包",
                    data=all_package,
                    file_name=f"arxiv-selection-package-{datetime.datetime.now().strftime('%Y%m%d')}.zip",
                    mime="application/zip",
                    key="download_markdown_package"
                )
            with package_col2:
                if len(st.session_state.synthesis_entries) > 1 and st.session_state.synthesis_result:
                    st.download_button(
                        label="⬇️ 下载全部 Markdown",
                        data=st.session_state.synthesis_result,
                        file_name=f"arxiv-selection-{datetime.datetime.now().strftime('%Y%m%d')}.md",
                        mime="text/markdown",
                        key="download_all_entries"
                    )

        st.divider()

    # Paper list
    st.divider()
    for paper in papers:
        with st.container():
            # Pre-fetched synth_by_paper mapping is built outside this loop (see above)
            col1, col2 = st.columns([20, 1], vertical_alignment="center")

            with col1:
                star_icon = "⭐ " if paper.is_starred else ""
                title_line = f"{star_icon}[{paper.id}] {paper.title}"
                st.markdown(f"### {title_line}")

                # Score badge HTML
                if paper.ai_score is None:
                    score_html = '<span class="score-badge score-na">未评分</span>'
                elif paper.ai_score >= 9:
                    score_html = f'<span class="score-badge score-s9">{paper.ai_score}/10</span>'
                elif paper.ai_score >= 7:
                    score_html = f'<span class="score-badge score-s7">{paper.ai_score}/10</span>'
                elif paper.ai_score >= 5:
                    score_html = f'<span class="score-badge score-s5">{paper.ai_score}/10</span>'
                else:
                    score_html = f'<span class="score-badge score-low">{paper.ai_score}/10</span>'

                # Tag pills
                if paper.ai_tags:
                    raw_tags = [t.strip() for t in paper.ai_tags.split(',') if t.strip()][:4]
                else:
                    raw_tags = [t.strip() for t in paper.categories.split() if t.strip()][:3]
                tags_html = ''.join(f'<span class="tag-pill">{t}</span>' for t in raw_tags)

                # Metadata row
                meta_html = (
                    f'<div class="meta-row">'
                    f'<span class="meta-authors">✍️ {format_authors(paper.authors)}</span>'
                    f'<span class="meta-sep">·</span>'
                    f'{tags_html}'
                    f'<span class="meta-sep">·</span>'
                    f'{score_html}'
                    f'</div>'
                )
                st.markdown(meta_html, unsafe_allow_html=True)

                # Check if this paper is in any saved synthesis
                found = synth_by_paper.get(paper.id, [])
                has_saved = bool(found)

                # Build expanders
                if paper.ai_reason or has_saved:
                    with st.expander("💡 AI 推荐理由" + (" / 📜 已生成推文" if has_saved else ""), expanded=False):
                        if paper.ai_reason:
                            st.write(paper.ai_reason)
                        if has_saved:
                            st.divider()
                            st.write("**📜 已保存的深度推文:**")
                            for synth in found:
                                dt = synth['created_at'][:19]
                                paper_count = len(synth['paper_ids'])
                                synth_label = dt if paper_count <= 1 else f"{dt} ({paper_count} 篇)"
                                col_synth1, col_synth2 = st.columns([3, 1], vertical_alignment="center")
                                with col_synth1:
                                    st.text(synth_label)
                                with col_synth2:
                                    if st.button("📂 加载", key=f"load_synth_{synth['id']}_{paper.id}"):
                                        st.session_state.synthesis_result = synth['content']
                                        st.session_state.synthesis_entries = parse_synthesis_entries(synth['content'])
                                        st.rerun()

                with st.expander("📝 摘要", expanded=False):
                    st.write(paper.abstract)

                st.write(f"📅 发表: {paper.published[:10]} | 🔗 [PDF链接]({paper.pdf_url})")

            with col2:
                checkbox_key = f"select_{paper.id}"
                if checkbox_key not in st.session_state:
                    st.session_state[checkbox_key] = paper.id in st.session_state.selected_papers
                st.checkbox("选择", key=checkbox_key)
                if st.button("⭐" if not paper.is_starred else "💫", key=f"star_{paper.id}"):
                    db.toggle_starred(paper.id)
                    st.rerun()



if __name__ == "__main__":
    main()
