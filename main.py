#!/usr/bin/env python3
"""
ArXiv Chip Architecture & EDA Frontier Tracking System
Main entry point for CLI operations.

Usage:
  python main.py sync          # Manual sync of papers (default 100 max)
  python main.py sync N        # Sync with N max results (e.g. python main.py sync 500)
  python main.py process       # Process unprocessed papers with AI
  python main.py scheduler     # Start daily scheduler
  python main.py stats         # Show database statistics
  python main.py debug         # Debug: print current LLM configuration
  python main.py clear         # Clear database (requires confirmation)
  python main.py clear --pdf   # Clear database AND all downloaded PDFs
"""

import sys
import os
import glob
from dotenv import load_dotenv

from src.database import Database
from src.ingest import run_sync
from src.ai_filter import process_all_unprocessed
from src.scheduler import run_scheduler


load_dotenv()
DB_PATH = os.getenv('DB_PATH', './data/papers.db')
PDF_DIR = os.getenv('PDF_DIR', './pdfs')


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


def clear_database(clear_pdf: bool = False):
    """Clear the database (and optionally PDFs). Requires confirmation."""
    db_path = DB_PATH
    pdf_dir = PDF_DIR

    if not os.path.exists(db_path):
        print(f"Database not found at {db_path}")
        return

    print("WARNING: This will DELETE ALL data in the database!")
    print(f"Database path: {db_path}")
    if clear_pdf:
        print(f"This will ALSO DELETE ALL downloaded PDFs in {pdf_dir}")

    print("\nType 'YES' to confirm, anything else to cancel:")
    try:
        response = input().strip()
    except (EOFError, KeyboardInterrupt):
        print("\nCancelled.")
        return

    if response != 'YES':
        print("Cancelled.")
        return

    # Delete database
    try:
        os.remove(db_path)
        print(f"✓ Deleted database: {db_path}")
    except Exception as e:
        print(f"Error deleting database: {e}")

    # Delete PDFs if requested
    if clear_pdf and os.path.exists(pdf_dir):
        pdf_files = glob.glob(os.path.join(pdf_dir, '*.pdf'))
        deleted = 0
        for pdf_file in pdf_files:
            try:
                os.remove(pdf_file)
                deleted += 1
            except Exception:
                pass
        print(f"✓ Deleted {deleted} PDF files from {pdf_dir}")

    print("\nDone. Next sync will create a fresh database.")


def debug_config():
    """Debug: print current configuration."""
    import os
    from pathlib import Path
    from dotenv import load_dotenv

    # Load .env
    project_root = Path(__file__).parent
    env_path = project_root / '.env'
    print(f"Loading .env from: {env_path}")
    print(f"File exists: {env_path.exists()}")

    if env_path.exists():
        load_dotenv(env_path)

    print("\n=== Current Configuration ===")
    print(f"LLM_PROVIDER: {os.getenv('LLM_PROVIDER')}")
    print(f"BASE_URL: {os.getenv('BASE_URL')}")
    print(f"ANTHROPIC_BASE_URL: {os.getenv('ANTHROPIC_BASE_URL')}")
    api_key = os.getenv('API_KEY', '')
    if api_key:
        # Show only first 8 and last 4 chars for safety
        if len(api_key) > 12:
            masked = api_key[:8] + '*' * (len(api_key) - 12) + api_key[-4:]
        else:
            masked = '*' * len(api_key)
        print(f"API_KEY: {masked} (length: {len(api_key)})")
    else:
        print(f"API_KEY: <empty>")
    print(f"LLM_MODEL: {os.getenv('LLM_MODEL')}")
    print(f"TEMPERATURE: {os.getenv('TEMPERATURE')}")
    print(f"MAX_TOKENS_SCORING: {os.getenv('MAX_TOKENS_SCORING')}")
    print(f"MAX_TOKENS_SYNTHESIS: {os.getenv('MAX_TOKENS_SYNTHESIS')}")
    print(f"DB_PATH: {os.getenv('DB_PATH')}")
    print(f"PDF_DIR: {os.getenv('PDF_DIR')}")
    print("==============================")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]

    if cmd == 'sync':
        # Check if max results given in args
        max_results = None
        if len(sys.argv) >= 3:
            try:
                max_results = int(sys.argv[2])
            except ValueError:
                pass
        print(f"Starting sync from ArXiv... max_results={max_results or '(default 100)'}")
        result = run_sync(DB_PATH, max_results=max_results)
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

    elif cmd == 'clear':
        clear_database(clear_pdf='--pdf' in sys.argv)

    elif cmd == 'debug':
        debug_config()

    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
