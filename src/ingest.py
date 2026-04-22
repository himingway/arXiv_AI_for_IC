"""
Data Ingestion module for ArXiv paper crawler.
Fetches papers from specified categories and stores in database.
"""

import arxiv
import datetime
import time
from typing import List, Tuple, Iterator
from dataclasses import dataclass

from .database import Database, Paper


@dataclass
class SyncResult:
    """Result of a sync operation."""
    papers_added: int
    papers_updated: int
    success: bool
    error_message: str = ""


class ArXivCrawler:
    """Crawler for fetching papers from ArXiv."""

    # Target categories for our domain
    DEFAULT_CATEGORIES = ['cs.AR', 'cs.DC', 'cs.ET', 'cs.AI']

    def __init__(self, db: Database, categories: List[str] = None, max_results: int = 1000):
        self.db = db
        self.categories = categories or self.DEFAULT_CATEGORIES
        self.max_results = max_results
        self.client = arxiv.Client(
            page_size=100,
            delay_seconds=3,
            num_retries=5
        )

    def _build_search_query(self) -> str:
        """Build the search query for multiple categories."""
        category_queries = [f"cat:{cat}" for cat in self.categories]
        return " OR ".join(category_queries)

    def _paper_from_result(self, result: arxiv.Result) -> Paper:
        """Convert arxiv.Result to our Paper dataclass."""
        # Extract short ID (arxiv uses different formats)
        paper_id = result.entry_id.split('/')[-1]
        # Remove version suffix if present (e.g., v1 -> keep just the ID)
        if paper_id.endswith(('v1', 'v2', 'v3', 'v4', 'v5')):
            paper_id = paper_id.rsplit('v', 1)[0]

        return Paper(
            id=paper_id,
            title=result.title.strip(),
            authors=', '.join([author.name for author in result.authors]),
            abstract=result.summary.strip(),
            categories=' '.join(result.categories),
            published=result.published.isoformat() if result.published else "",
            updated=result.updated.isoformat() if result.updated else "",
            pdf_url=result.pdf_url,
            entry_id=result.entry_id,
            ai_processed=False,
            ai_score=None,
            ai_reason=None,
            ai_tags=None,
            is_starred=False,
            created_at=datetime.datetime.now().isoformat(),
            processed_at=None
        )

    def sync(self, max_papers: int = None) -> SyncResult:
        """
        Synchronize papers from ArXiv to database.
        Returns count of new papers and updated papers.
        """
        search_query = self._build_search_query()
        search = arxiv.Search(
            query=search_query,
            max_results=max_papers or self.max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )

        papers_added = 0
        papers_updated = 0

        try:
            results: Iterator[arxiv.Result] = self.client.results(search)

            for result in results:
                paper = self._paper_from_result(result)
                is_new = self.db.insert_or_update_paper(paper)
                if is_new:
                    papers_added += 1
                else:
                    papers_updated += 1

                # Rate limiting
                time.sleep(0.5)

            self.db.log_sync(papers_added, papers_updated, 'success')
            return SyncResult(
                papers_added=papers_added,
                papers_updated=papers_updated,
                success=True
            )

        except Exception as e:
            error_msg = str(e)
            self.db.log_sync(papers_added, papers_updated, 'error', error_msg)
            return SyncResult(
                papers_added=papers_added,
                papers_updated=papers_updated,
                success=False,
                error_message=error_msg
            )

    def check_todays_sync_done(self) -> bool:
        """Check if we've already synced today."""
        last_sync = self.db.get_last_sync_time()
        if not last_sync:
            return False

        try:
            last_sync_dt = datetime.datetime.fromisoformat(last_sync)
            today = datetime.datetime.now().date()
            return last_sync_dt.date() == today
        except Exception:
            return False


def run_sync(db_path: str, categories: List[str] = None, max_results: int = 1000) -> SyncResult:
    """Helper function to run sync from command line."""
    db = Database(db_path)
    crawler = ArXivCrawler(db, categories=categories, max_results=max_results)
    return crawler.sync()
