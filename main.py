#!/usr/bin/env python3
"""
ArXiv Chip Architecture & EDA Frontier Tracking System
Main entry point for CLI operations.

Usage:
  python main.py sync          # Manual sync of papers
  python main.py process       # Process unprocessed papers with AI
  python main.py scheduler     # Start daily scheduler
  python main.py stats         # Show database statistics
"""

import sys
import os
from dotenv import load_dotenv

from src.database import Database
from src.ingest import run_sync
from src.ai_filter import process_all_unprocessed
from src.scheduler import run_scheduler


load_dotenv()
DB_PATH = os.getenv('DB_PATH', './data/papers.db')


def show_stats():
    """Show database statistics."""
    db = Database(DB_PATH)
    stats = db.get_stats()

    print("=== Database Statistics ===")
    print(f"Total papers:       {stats['total_papers']}")
    print(f"Processed by AI:    {stats['processed_papers']}")
    print(f"Pending processing: {stats['unprocessed_papers']}")
    print(f"Starred papers:     {stats['starred_papers']}")
    print(f"Added today:        {stats['today_added']}")
    if stats['average_score']:
        print(f"Average AI score:   {stats['average_score']}")
    print()
    print("Recent syncs:")
    for sync in stats['recent_syncs'][:5]:
        print(f"  {sync['sync_time'][:19]} - {sync['papers_added']} added - {sync['status']}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]

    if cmd == 'sync':
        print("Starting sync from ArXiv...")
        result = run_sync(DB_PATH)
        if result.success:
            print(f"Sync completed: added {result.papers_added} papers, updated {result.papers_updated}")
        else:
            print(f"Sync failed: {result.error_message}")

    elif cmd == 'process':
        print("Processing unprocessed papers with AI...")
        db = Database(DB_PATH)
        total = process_all_unprocessed(db, batch_size=10, delay=2.0)
        print(f"Processing complete. Total {total} papers processed.")

    elif cmd == 'scheduler':
        run_scheduler()

    elif cmd == 'stats':
        show_stats()

    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
