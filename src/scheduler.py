"""
Scheduler module for daily automatic sync.
Runs daily at 8:00 Beijing time (UTC 00:00).
"""

import os
import time
import datetime
from typing import Optional
from dotenv import load_dotenv
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from .database import Database
from .ingest import ArXivCrawler, SyncResult
from .ai_filter import AIFilter


load_dotenv()

DB_PATH = os.getenv('DB_PATH', './data/papers.db')


def sync_job():
    """Daily sync job."""
    print(f"\n=== Starting daily sync at {datetime.datetime.now().isoformat()} ===")
    db = Database(DB_PATH)
    crawler = ArXivCrawler(db)
    result = crawler.sync()

    if result.success:
        print(f"Daily sync completed: added {result.papers_added}, updated {result.papers_updated}")

        # Auto-process newly added papers if AI is configured
        ai_filter = AIFilter()
        if ai_filter.is_configured():
            pending = db.count_unprocessed()
            if pending > 0:
                print(f"Auto-processing {min(pending, 10)} pending papers...")
                processed = ai_filter.process_next_batch(db, batch_size=min(pending, 10))
                print(f"Processed {processed} papers")
    else:
        print(f"Daily sync failed: {result.error_message}")

    print(f"=== Daily sync finished ===")


def run_scheduler():
    """Start the blocking scheduler."""
    # Run at 8:00 AM Beijing time (UTC+8)
    # Which is 00:00 UTC
    scheduler = BlockingScheduler(timezone='UTC')
    # Add job: every day at 00:00 UTC = 08:00 Beijing
    scheduler.add_job(
        sync_job,
        CronTrigger(hour=0, minute=0),
        id='daily_sync',
        replace_existing=True
    )

    print(f"Scheduler started. Will run daily sync at 08:00 Beijing time (00:00 UTC)")

    # Check if we need to run on startup if not synced today
    db = Database(DB_PATH)
    crawler = ArXivCrawler(db)
    if not crawler.check_todays_sync_done():
        print("No sync today, running initial sync...")
        sync_job()

    # Calculate next run time before starting
    job = scheduler.get_job('daily_sync')
    next_run = job.trigger.get_next_fire_time(None, datetime.datetime.now(datetime.timezone.utc))
    print(f"Next run will be at: {next_run}")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("Scheduler stopped")


if __name__ == "__main__":
    run_scheduler()
