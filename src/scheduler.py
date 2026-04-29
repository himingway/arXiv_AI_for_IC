"""
Scheduler module for daily automatic sync.
Runs daily at 8:00 Beijing time (UTC 00:00).
"""

import os
import time
import datetime
import logging
from typing import Optional
from dotenv import load_dotenv
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from .database import Database
from .ingest import ArXivCrawler, SyncResult
from .ai_filter import AIFilter

logger = logging.getLogger(__name__)

load_dotenv()

DB_PATH = os.getenv('DB_PATH', './data/papers.db')


def sync_job():
    """Daily sync job."""
    logger.info(f"Starting daily sync at {datetime.datetime.now().isoformat()}")
    db = Database(DB_PATH)
    crawler = ArXivCrawler(db)
    result = crawler.sync()

    if result.success:
        logger.info(f"Daily sync completed: added {result.papers_added}, updated {result.papers_updated}")

        ai_filter = AIFilter()
        if ai_filter.is_configured():
            pending = db.count_unprocessed()
            if pending > 0:
                logger.info(f"Auto-processing {min(pending, 10)} pending papers...")
                processed = ai_filter.process_next_batch(db, batch_size=min(pending, 10))
                logger.info(f"Processed {processed} papers")
    else:
        logger.error(f"Daily sync failed: {result.error_message}")

    logger.info("Daily sync finished")


def run_scheduler():
    """Start the blocking scheduler."""
    from .logging_config import setup_logging
    setup_logging()

    scheduler = BlockingScheduler(timezone='UTC')
    scheduler.add_job(
        sync_job,
        CronTrigger(hour=0, minute=0),
        id='daily_sync',
        replace_existing=True
    )

    logger.info("Scheduler started. Will run daily sync at 08:00 Beijing time (00:00 UTC)")

    db = Database(DB_PATH)
    crawler = ArXivCrawler(db)
    if not crawler.check_todays_sync_done():
        logger.info("No sync today, running initial sync...")
        sync_job()

    job = scheduler.get_job('daily_sync')
    next_run = job.trigger.get_next_fire_time(None, datetime.datetime.now(datetime.timezone.utc))
    logger.info(f"Next run will be at: {next_run}")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Scheduler stopped")


if __name__ == "__main__":
    run_scheduler()
