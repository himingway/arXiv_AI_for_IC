"""
Data Ingestion module for ArXiv paper crawler.
Fetches papers from specified categories and stores in database.
Supports parallel fetching for faster speed.
"""

import os
import re
import arxiv
import datetime
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Iterator, Optional
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

from .database import Paper
from .database import Database

# Load environment variables from project root
project_root = Path(__file__).parent.parent
load_dotenv(project_root / '.env')


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

    @classmethod
    def get_categories_from_env(cls) -> List[str]:
        """Load target categories from ARXIV_CATEGORIES env (comma-separated)."""
        raw = os.getenv('ARXIV_CATEGORIES', '')
        if not raw.strip():
            return cls.DEFAULT_CATEGORIES
        categories = [item.strip() for item in raw.split(',') if item.strip()]
        return categories or cls.DEFAULT_CATEGORIES

    def __init__(self, db: Database, categories: List[str] = None, max_results: int = None):
        self.db = db
        self.categories = categories or self.get_categories_from_env()
        # Read MAX_RESULTS from environment if not explicitly given
        if max_results is None:
            max_results_env = os.getenv('MAX_RESULTS', '')
            if max_results_env.strip():
                try:
                    self.max_results = int(max_results_env)
                except ValueError:
                    self.max_results = 100
            else:
                self.max_results = 100
        else:
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
        # Remove version suffix if present (e.g., v1, v12 -> keep just the ID)
        paper_id = re.sub(r'v\d+$', '', paper_id)

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
        Uses sequential fetching to comply with ArXiv rate limits (1 request per 3 seconds, single connection).
        Optimized: stops early when hitting many existing papers (since results are sorted by date descending).
        """
        search_query = self._build_search_query()
        max_papers = max_papers or self.max_results
        search = arxiv.Search(
            query=search_query,
            max_results=max_papers,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )

        papers_added = 0
        papers_updated = 0
        processed_count = 0
        stop_threshold = 20  # Stop when the last 20 papers were all already in database
        # Sliding window of recent existence status: True = existing (updated), False = new
        recent = []

        try:
            results: Iterator[arxiv.Result] = self.client.results(search)
            print(f"Fetching papers from ArXiv (max {max_papers})...\n")
            print("Complying with ArXiv rate limits: 1 request per 3 seconds, single connection.\n")

            for result in results:
                # Convert and insert directly - process sequentially
                paper_id = result.entry_id.split('/')[-1]
                paper_id = re.sub(r'v\d+$', '', paper_id)

                processed_count += 1

                # Convert result to our Paper format directly - no need for second pass
                paper = self._paper_from_result(result)

                # Insert or update
                if self.db.insert_or_update_paper(paper):
                    papers_added += 1
                    is_existing = False
                else:
                    papers_updated += 1
                    is_existing = True

                # Update sliding window: keep only the last stop_threshold entries
                recent.append(is_existing)
                if len(recent) > stop_threshold:
                    recent.pop(0)

                # Progress
                if processed_count % 10 == 0:
                    consecutive_existing = sum(recent)
                    print(f"Processed {processed_count} | Added {papers_added} new papers | "
                          f"Consecutive existing (last {stop_threshold}): {sum(recent)}")

                # Early stop conditions (ALL must be true):
                # 1. We have at least stop_threshold recent papers
                # 2. All of the last stop_threshold papers are already existing in database
                # 3. AND: We have already added at least one new paper in this sync
                #
                # This guarantees correctness for all scenarios:
                # - Normal daily sync: if all newest papers already in DB, early stop (after first new found + then 20 existing)
                # - Interrupted sync restart: if all papers so far are existing but some later are not,
                #   papers_added = 0 so no early stop → will continue processing until max to find any missing ones
                # - Only stop when we're sure all remaining older papers are already in database
                if len(recent) == stop_threshold and all(recent) and papers_added > 0:
                    print(f"\nEarly stop: after adding {papers_added} new papers, the last {stop_threshold} consecutive "
                          f"papers are all already in database. All remaining older papers are already processed.")
                    break

                # Comply with ArXiv rate limit: one request every three seconds
                # We are already on single connection, this ensures rate compliance
                if processed_count < max_papers:  # No need to sleep after last request
                    time.sleep(3.0)

            print(f"\nSync complete! Total processed: {processed_count}, added: {papers_added}, updated: {papers_updated}")
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


def run_sync(db_path: str, categories: List[str] = None, max_results: int = None) -> SyncResult:
    """Helper function to run sync from command line."""
    db = Database(db_path)
    # If max_results not explicitly given from command line, let ArXivCrawler
    # read it from MAX_RESULTS environment variable (fallback to 100 if not set)
    crawler = ArXivCrawler(db, categories=categories, max_results=max_results)
    return crawler.sync()
