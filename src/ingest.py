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

    def __init__(self, db: Database, categories: List[str] = None, max_results: int = 100):
        self.db = db
        self.categories = categories or self.get_categories_from_env()
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

    def sync(self, max_papers: int = None, max_workers: int = 3) -> SyncResult:
        """
        Synchronize papers from ArXiv to database.
        Uses parallel fetching for faster speed.
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
        consecutive_existing = 0
        stop_threshold = 20  # Stop after 20 consecutive existing papers (all newer already processed)

        # First pass: quickly scan to find which papers are new
        # We just get the IDs from the search (much faster)
        paper_ids_to_fetch = []

        try:
            results: Iterator[arxiv.Result] = self.client.results(search)
            print(f"Scanning for new papers (max {max_papers})...\n")

            for result in results:
                paper_id = result.entry_id.split('/')[-1]
                paper_id = re.sub(r'v\d+$', '', paper_id)

                processed_count += 1
                existing = self.db.get_paper(paper_id)

                if existing is not None:
                    papers_updated += 1
                    consecutive_existing += 1
                else:
                    paper_ids_to_fetch.append(paper_id)
                    consecutive_existing = 0
                    # We'll fetch details in parallel

                # Progress
                if processed_count % 50 == 0:
                    print(f"Scanned {processed_count} | Found {len(paper_ids_to_fetch)} new papers | "
                          f"Consecutive existing: {consecutive_existing}")

                # Early stop
                if consecutive_existing >= stop_threshold and len(paper_ids_to_fetch) == 0:
                    print(f"\nEarly stop: {stop_threshold} consecutive existing papers and no new papers found. "
                          f"All newer papers already processed.")
                    break

                # Light rate limiting during scanning
                time.sleep(0.05)

            print(f"\nScan complete. Scanned {processed_count} papers, found {len(paper_ids_to_fetch)} new papers to fetch.")

            # Second pass: parallel fetch details for new papers
            if len(paper_ids_to_fetch) > 0:
                print(f"Fetching {len(paper_ids_to_fetch)} new papers in parallel (max {max_workers} workers)...")

                def fetch_paper(paper_id: str, max_retries: int = 3) -> Optional[Paper]:
                    """Fetch a single paper by ID with retries on failure."""
                    for retry in range(max_retries):
                        try:
                            search = arxiv.Search(id_list=[paper_id])
                            paper_result = next(self.client.results(search))
                            paper = self._paper_from_result(paper_result)
                            time.sleep(1.0)  # Rate limiting per request
                            return paper
                        except Exception as e:
                            if "429" in str(e):
                                # Rate limit hit, backoff
                                backoff = 5 * (retry + 1)
                                print(f"  [{paper_id}] Rate limit 429, retry {retry+1}/{max_retries} after {backoff}s...")
                                time.sleep(backoff)
                            else:
                                print(f"  [{paper_id}] Failed (attempt {retry+1}/{max_retries}): {e}")
                                if retry < max_retries - 1:
                                    time.sleep(3)
                    print(f"  [{paper_id}] All {max_retries} retries failed, skipping.")
                    return None

                papers_added = 0
                # Process in batches to avoid overwhelming ArXiv
                batch_size = max_workers * 5
                for batch_start in range(0, len(paper_ids_to_fetch), batch_size):
                    batch_ids = paper_ids_to_fetch[batch_start:batch_start + batch_size]

                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = [executor.submit(fetch_paper, pid) for pid in batch_ids]
                        for i, future in enumerate(as_completed(futures), 1):
                            paper = future.result()
                            if paper is not None:
                                if self.db.insert_or_update_paper(paper):
                                    papers_added += 1
                            completed_total = batch_start + i
                            if completed_total % 10 == 0:
                                print(f"  Fetched {completed_total}/{len(paper_ids_to_fetch)}")

                    # Pause between batches
                    if batch_start + batch_size < len(paper_ids_to_fetch):
                        print(f"  Pausing for 5 seconds to avoid rate limit...")
                        time.sleep(5)

            print(f"\nSync complete! Total scanned: {processed_count}, added: {papers_added}, updated: {papers_updated}")
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
    if max_results is None:
        max_results = 100  # Default
    crawler = ArXivCrawler(db, categories=categories, max_results=max_results)
    return crawler.sync()
