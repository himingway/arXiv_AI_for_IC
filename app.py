"""
Streamlit Interactive Dashboard for ArXiv Chip Architecture Paper Tracker.
"""

import os
import sys
import datetime
import streamlit as st
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.database import Database, Paper
from src.ingest import ArXivCrawler, SyncResult
from src.ai_filter import AIFilter
from src.pdf_parser import PDFProcessor
from src.deep_synthesis import DeepSynthesizer


load_dotenv()

# Configuration
DB_PATH = os.getenv('DB_PATH', './data/papers.db')
PDF_DIR = os.getenv('PDF_DIR', './pdfs')
MIN_SCORE_DEFAULT = 7


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

    # Paper list (process selection first)
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
                                col_synth1, col_synth2 = st.columns([3, 1], vertical_alignment="center")
                                with col_synth1:
                                    st.text(f"{dt} ({len(synth['paper_ids'])} 篇)")
                                with col_synth2:
                                    if st.button("📂 加载", key=f"load_synth_{synth['id']}_{paper.id}"):
                                        st.session_state.synthesis_result = synth['content']
                                        st.rerun()

                with st.expander("📝 摘要", expanded=False):
                    st.write(paper.abstract)

                st.write(f"📅 发表: {paper.published[:10]} | 🔗 [PDF链接]({paper.pdf_url})")

            with col2:
                is_selected = paper.id in st.session_state.selected_papers
                if st.checkbox("选择", key=f"select_{paper.id}", value=is_selected):
                    st.session_state.selected_papers.add(paper.id)
                else:
                    if paper.id in st.session_state.selected_papers:
                        st.session_state.selected_papers.remove(paper.id)
                if st.button("⭐" if not paper.is_starred else "💫", key=f"star_{paper.id}"):
                    db.toggle_starred(paper.id)
                    st.rerun()

    st.divider()

    # Action bar for selected papers (after processing selection)
    selected_count = len(st.session_state.selected_papers)
    col_a, col_b = st.columns([4, 1], vertical_alignment="center")
    with col_a:
        st.markdown(f"##### 已选择 **{selected_count}** 篇论文")
    with col_b:
        if selected_count > 0:
            if st.button("📝 生成深度推文", use_container_width=True):
                selected_paper_objs = [p for pid in st.session_state.selected_papers if (p := db.get_paper(pid))]
                progress_bar = st.progress(0)
                status_text = st.empty()
                synthesis_result = f"# ArXiv 芯片架构前沿精选\n\n"
                synthesis_result += f"生成时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
                synthesis_result += "---\n\n"

                for i, paper in enumerate(selected_paper_objs, 1):
                    progress = i / len(selected_paper_objs)
                    status_text.text(f"正在处理第 {i}/{len(selected_paper_objs)} 篇: {paper.title[:50]}...")
                    progress_bar.progress(progress)

                    # Get full text from PDF
                    full_text = pdf_processor.get_or_download(paper)
                    if full_text is None:
                        synthesis_result += f"## {i}. {paper.title}\n\n"
                        synthesis_result += f"**作者**: {paper.authors}\n\n"
                        synthesis_result += f"评分: **{paper.ai_score}/10** 标签: {paper.ai_tags}\n\n"
                        synthesis_result += f"PDF 下载失败，无法生成深度分析。\n\n"
                        synthesis_result += f"原文链接: {paper.pdf_url}\n\n"
                        synthesis_result += "---\n\n"
                        continue

                    synthesis = synthesizer.synthesize_paper(paper, full_text)
                    if synthesis is None:
                        synthesis_result += f"## {i}. {paper.title}\n\n"
                        synthesis_result += f"**作者**: {paper.authors}\n\n"
                        synthesis_result += f"评分: **{paper.ai_score}/10** 标签: {paper.ai_tags}\n\n"
                        synthesis_result += f"AI 生成失败。\n\n"
                        synthesis_result += f"原文链接: {paper.pdf_url}\n\n"
                        synthesis_result += "---\n\n"
                        continue

                    synthesis_result += f"## {i}. {paper.title}\n\n"
                    synthesis_result += f"**作者**: {paper.authors}\n\n"
                    synthesis_result += f"评分: **{paper.ai_score}/10** 标签: {paper.ai_tags}\n\n"
                    synthesis_result += f"ArXiv: [{paper.id}]({paper.pdf_url})\n\n"
                    synthesis_result += "---\n\n"
                    synthesis_result += synthesis
                    synthesis_result += "\n\n---\n\n"

                progress_bar.progress(1.0)
                status_text.empty()
                st.session_state.synthesis_result = synthesis_result
                # Save to database
                selected_ids = [p.id for p in selected_paper_objs]
                db.save_synthesis(selected_ids, synthesis_result)
                st.success(f"生成完成！已保存到数据库，共处理 {len(selected_paper_objs)} 篇论文")
                # Auto-offer download immediately so it's not lost on refresh
                filename = f"arxiv-selection-{datetime.datetime.now().strftime('%Y%m%d')}.md"
                st.download_button(
                    label="⬇️ 立即下载 Markdown",
                    data=synthesis_result,
                    file_name=filename,
                    mime="text/markdown"
                )
        else:
            st.button("📝 生成深度推文", disabled=True, use_container_width=True)

    if st.session_state.synthesis_result:
        with st.expander("📄 生成的深度推文（点击展开/收起）", expanded=True):
            st.markdown(st.session_state.synthesis_result)
            # Add download button
            st.download_button(
                label="⬇️ 下载 Markdown 文件",
                data=st.session_state.synthesis_result,
                file_name=f"arxiv-selection-{datetime.datetime.now().strftime('%Y%m%d')}.md",
                mime="text/markdown"
            )
            if st.button("🗑️ 清空结果"):
                st.session_state.synthesis_result = None
                st.rerun()

            st.divider()


if __name__ == "__main__":
    main()
