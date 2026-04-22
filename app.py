"""
Streamlit Interactive Dashboard for ArXiv Chip Architecture Paper Tracker.
"""

import os
import sys
import datetime
import streamlit as st
import pandas as pd
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


def format_authors(authors: str, max_len: int = 60) -> str:
    """Format authors for display."""
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

    init_session_state()
    db = get_db()
    ai_filter = AIFilter()
    pdf_processor = PDFProcessor(PDF_DIR)
    synthesizer = DeepSynthesizer()

    # Sidebar
    with st.sidebar:
        st.title("🎯 ArXiv 追踪系统")
        st.markdown("**芯片架构与 EDA 前沿论文精选**")

        st.divider()

        # Stats
        stats = db.get_stats()
        st.subheader("📊 统计信息")
        st.write(f"总论文数: **{stats['total_papers']}**")
        st.write(f"已AI评分: **{stats['processed_papers']}**")
        st.write(f"待处理: **{stats['unprocessed_papers']}**")
        st.write(f"今日新增: **{stats['today_added']}**")
        st.write(f"已收藏: **{stats['starred_papers']}**")
        if stats['average_score']:
            st.write(f"平均分: **{stats['average_score']}**")

        st.divider()

        # Manual sync button
        st.subheader("🔄 数据同步")
        last_sync = db.get_last_sync_time()
        if last_sync:
            try:
                last_sync_dt = datetime.datetime.fromisoformat(last_sync)
                st.write(f"上次同步: {last_sync_dt.strftime('%Y-%m-%d %H:%M')}")
            except:
                st.write(f"上次同步: {last_sync}")

        if st.button("🔄 立即同步最新论文", disabled=st.session_state.sync_running, use_container_width=True):
            st.session_state.sync_running = True
            with st.spinner("正在同步 ArXiv... 这可能需要几分钟时间"):
                crawler = ArXivCrawler(db)
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
        st.write(f"待处理: {pending} 篇")

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
                processed += processed_batch[0] if isinstance(processed_batch, tuple) else processed_batch
                progress_bar.progress((i + 1) / min(batch_size, pending))

            status_text.text(f"完成！共处理 {processed} 篇论文")
            st.session_state.ai_process_running = False
            st.rerun()

        st.divider()

        # PDF storage info
        pdf_stats = pdf_processor.get_storage_stats()
        st.subheader("📄 PDF 存储")
        st.write(f"已下载: {pdf_stats['file_count']} 文件")
        st.write(f"占用空间: {pdf_stats['total_size_mb']} MB")

    # Main content
    st.title("📚 ArXiv 芯片架构与 EDA 前沿论文")

    # Filters
    col1, col2, col3, col4 = st.columns([1, 1, 1, 2])
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

    if not papers:
        st.info("没有找到符合条件的论文。请先点击侧边栏的「立即同步最新论文」获取数据。")
        return

    # Display papers in a table-like format with checkboxes
    st.divider()

    # Action bar for selected papers
    selected_count = len(st.session_state.selected_papers)
    col_a, col_b = st.columns([3, 1])
    with col_a:
        st.write(f"已选择 {selected_count} 篇论文")
    with col_b:
        if selected_count > 0:
            if st.button("📝 生成深度推文", use_container_width=True):
                with st.spinner("正在生成深度推文... 这需要几分钟时间"):
                    selected_paper_objs = [db.get_paper(pid) for pid in st.session_state.selected_papers if db.get_paper(pid)]
                    synthesis_result = synthesizer.synthesize_multiple(selected_paper_objs, pdf_processor)
                    st.session_state.synthesis_result = synthesis_result
                    st.success("生成完成！")
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

    # Paper list
    for paper in papers:
        with st.container():
            col1, col2 = st.columns([20, 1])

            with col1:
                # Score badge color
                if paper.ai_score is None:
                    score_badge = "⚪ 未评分"
                elif paper.ai_score >= 9:
                    score_badge = f"🔴 **{paper.ai_score}/10**"
                elif paper.ai_score >= 7:
                    score_badge =f"🟠 **{paper.ai_score}/10**"
                elif paper.ai_score >= 5:
                    score_badge =f"🟡 **{paper.ai_score}/10**"
                else:
                    score_badge =f"🔵 **{paper.ai_score}/10**"

                star_icon = "⭐ " if paper.is_starred else ""
                title_line = f"{star_icon}[{paper.id}] {paper.title}"
                st.markdown(f"### {title_line}")

                col_authors, col_tags, col_score = st.columns([3, 2, 1])
                with col_authors:
                    st.write(f"**作者:** {format_authors(paper.authors)}")
                with col_tags:
                    if paper.ai_tags:
                        st.write(f"**标签:** {paper.ai_tags}")
                    else:
                        st.write("**分类:** {paper.categories}")
                with col_score:
                    st.markdown(score_badge)

                if paper.ai_reason:
                    with st.expander("💡 AI 推荐理由", expanded=False):
                        st.write(paper.ai_reason)

                with st.expander("📝 摘要", expanded=False):
                    st.write(paper.abstract)

                st.write(f"📅 发表: {paper.published[:10]} | 🔗 [PDF链接]({paper.pdf_url})")

            with col2:
                # Checkbox for selection
                is_selected = paper.id in st.session_state.selected_papers
                if st.checkbox("选择", key=f"select_{paper.id}", value=is_selected):
                    st.session_state.selected_papers.add(paper.id)
                else:
                    if paper.id in st.session_state.selected_papers:
                        st.session_state.selected_papers.remove(paper.id)

                # Star toggle
                if st.button("⭐" if not paper.is_starred else "💫", key=f"star_{paper.id}"):
                    new_status = db.toggle_starred(paper.id)
                    st.rerun()

            st.divider()


if __name__ == "__main__":
    main()
