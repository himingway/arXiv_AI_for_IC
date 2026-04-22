"""
Database module for ArXiv paper tracking system.
Handles SQLite operations for storing paper metadata and AI analysis results.
"""

import sqlite3
import datetime
from contextlib import contextmanager
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
import json


@dataclass
class Paper:
    """Represents an ArXiv paper with metadata and AI analysis."""
    id: str  # ArXiv ID
    title: str
    authors: str
    abstract: str
    categories: str
    published: str
    updated: str
    pdf_url: str
    entry_id: str
    ai_processed: bool = False
    ai_score: Optional[int] = None
    ai_reason: Optional[str] = None
    ai_tags: Optional[str] = None
    is_starred: bool = False
    created_at: str = None
    processed_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'title': self.title,
            'authors': self.authors,
            'abstract': self.abstract,
            'categories': self.categories,
            'published': self.published,
            'updated': self.updated,
            'pdf_url': self.pdf_url,
            'entry_id': self.entry_id,
            'ai_processed': self.ai_processed,
            'ai_score': self.ai_score,
            'ai_reason': self.ai_reason,
            'ai_tags': self.ai_tags,
            'is_starred': self.is_starred,
            'created_at': self.created_at,
            'processed_at': self.processed_at,
        }

    @classmethod
    def from_row(cls, row: Tuple) -> 'Paper':
        return cls(
            id=row[0],
            title=row[1],
            authors=row[2],
            abstract=row[3],
            categories=row[4],
            published=row[5],
            updated=row[6],
            pdf_url=row[7],
            entry_id=row[8],
            ai_processed=bool(row[9]),
            ai_score=row[10],
            ai_reason=row[11],
            ai_tags=row[12],
            is_starred=bool(row[13]),
            created_at=row[14],
            processed_at=row[15],
        )


class Database:
    """SQLite database wrapper for paper storage."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _connect(self):
        """Context manager that yields a connection and guarantees close."""
        conn = self._get_connection()
        try:
            yield conn
        finally:
            conn.close()

    def _init_db(self):
        """Create tables if they don't exist."""
        with self._connect() as conn:
            cursor = conn.cursor()

            # Main papers table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS papers (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    authors TEXT NOT NULL,
                    abstract TEXT NOT NULL,
                    categories TEXT NOT NULL,
                    published TEXT NOT NULL,
                    updated TEXT NOT NULL,
                    pdf_url TEXT NOT NULL,
                    entry_id TEXT NOT NULL,
                    ai_processed INTEGER DEFAULT 0,
                    ai_score INTEGER,
                    ai_reason TEXT,
                    ai_tags TEXT,
                    is_starred INTEGER DEFAULT 0,
                    created_at TEXT NOT NULL,
                    processed_at TEXT
                )
            ''')

            # Sync tracking table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS sync_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    sync_time TEXT NOT NULL,
                    papers_added INTEGER DEFAULT 0,
                    papers_updated INTEGER DEFAULT 0,
                    status TEXT NOT NULL,
                    error_message TEXT
                )
            ''')

            # Generated syntheses table (store generated deep articles)
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS syntheses (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL,
                    paper_ids TEXT NOT NULL,
                    content TEXT NOT NULL
                )
            ''')

            conn.commit()

    def insert_or_update_paper(self, paper: Paper) -> bool:
        """Insert a new paper or update existing one. Returns True if inserted as new."""
        if paper.created_at is None:
            paper.created_at = datetime.datetime.now().isoformat()

        with self._connect() as conn:
            cursor = conn.cursor()
            # Use INSERT OR IGNORE to avoid TOCTOU race in parallel ingestion
            cursor.execute('''
                INSERT OR IGNORE INTO papers (
                    id, title, authors, abstract, categories, published, updated,
                    pdf_url, entry_id, ai_processed, ai_score, ai_reason, ai_tags,
                    is_starred, created_at, processed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                paper.id, paper.title, paper.authors, paper.abstract, paper.categories,
                paper.published, paper.updated, paper.pdf_url, paper.entry_id,
                paper.ai_processed, paper.ai_score, paper.ai_reason, paper.ai_tags,
                paper.is_starred, paper.created_at, paper.processed_at
            ))
            is_new = cursor.rowcount > 0
            if not is_new:
                # Update existing paper but preserve AI data and starred status
                cursor.execute('''
                    UPDATE papers
                    SET title = ?, authors = ?, abstract = ?, categories = ?,
                        published = ?, updated = ?, pdf_url = ?, entry_id = ?
                    WHERE id = ?
                ''', (
                    paper.title, paper.authors, paper.abstract, paper.categories,
                    paper.published, paper.updated, paper.pdf_url, paper.entry_id,
                    paper.id
                ))
            conn.commit()
            return is_new

    def update_ai_analysis(self, paper_id: str, score: int, reason: str, tags: str):
        """Update paper with AI analysis results."""
        processed_at = datetime.datetime.now().isoformat()
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE papers
                SET ai_processed = 1, ai_score = ?, ai_reason = ?, ai_tags = ?, processed_at = ?
                WHERE id = ?
            ''', (score, reason, tags, processed_at, paper_id))
            conn.commit()

    def toggle_starred(self, paper_id: str) -> bool:
        """Toggle starred status. Returns new status."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT is_starred FROM papers WHERE id = ?', (paper_id,))
            result = cursor.fetchone()
            if result is None:
                return False
            current = bool(result[0])
            new_status = not current
            cursor.execute('UPDATE papers SET is_starred = ? WHERE id = ?', (int(new_status), paper_id))
            conn.commit()
            return new_status

    def get_paper(self, paper_id: str) -> Optional[Paper]:
        """Get a single paper by ID."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM papers WHERE id = ?', (paper_id,))
            row = cursor.fetchone()
            if row:
                return Paper.from_row(tuple(row))
            return None

    def get_unprocessed_papers(self, limit: Optional[int] = None) -> List[Paper]:
        """Get papers that haven't been processed by AI."""
        with self._connect() as conn:
            cursor = conn.cursor()
            if limit is not None:
                cursor.execute(
                    'SELECT * FROM papers WHERE ai_processed = 0 ORDER BY created_at ASC LIMIT ?',
                    (limit,)
                )
            else:
                cursor.execute('SELECT * FROM papers WHERE ai_processed = 0 ORDER BY created_at ASC')
            rows = cursor.fetchall()
            return [Paper.from_row(tuple(row)) for row in rows]

    def count_unprocessed(self) -> int:
        """Count unprocessed papers."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM papers WHERE ai_processed = 0')
            return cursor.fetchone()[0]

    def get_all_papers(self, limit: int = 1000, offset: int = 0) -> List[Paper]:
        """Get all papers sorted by AI score descending."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM papers
                ORDER BY ai_score DESC NULLS LAST, published DESC
                LIMIT ? OFFSET ?
            ''', (limit, offset))
            rows = cursor.fetchall()
            return [Paper.from_row(tuple(row)) for row in rows]

    def get_filtered_papers(self,
                            only_today: bool = False,
                            min_score: Optional[int] = None,
                            only_starred: bool = False,
                            search: Optional[str] = None,
                            limit: int = 1000,
                            offset: int = 0) -> List[Paper]:
        """Get filtered papers based on criteria."""
        conditions = []
        params = []

        if only_today:
            today = datetime.datetime.now().date().isoformat()
            conditions.append("DATE(created_at) = DATE(?)")
            params.append(today)

        if min_score is not None:
            conditions.append("ai_score >= ?")
            params.append(min_score)

        if only_starred:
            conditions.append("is_starred = 1")

        if search:
            search_term = f"%{search}%"
            conditions.append("(title LIKE ? OR authors LIKE ? OR abstract LIKE ? OR ai_tags LIKE ?)")
            params.extend([search_term, search_term, search_term, search_term])

        query = "SELECT * FROM papers"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY ai_score DESC NULLS LAST, published DESC"
        query += " LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [Paper.from_row(tuple(row)) for row in rows]

    def count_filtered(self,
                       only_today: bool = False,
                       min_score: Optional[int] = None,
                       only_starred: bool = False,
                       search: Optional[str] = None) -> int:
        """Count filtered papers."""
        conditions = []
        params = []

        if only_today:
            today = datetime.datetime.now().date().isoformat()
            conditions.append("DATE(created_at) = DATE(?)")
            params.append(today)

        if min_score is not None:
            conditions.append("ai_score >= ?")
            params.append(min_score)

        if only_starred:
            conditions.append("is_starred = 1")

        if search:
            search_term = f"%{search}%"
            conditions.append("(title LIKE ? OR authors LIKE ? OR abstract LIKE ? OR ai_tags LIKE ?)")
            params.extend([search_term, search_term, search_term, search_term])

        query = "SELECT COUNT(*) FROM papers"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return cursor.fetchone()[0]

    def get_total_count(self) -> int:
        """Get total number of papers."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM papers')
            return cursor.fetchone()[0]

    def get_last_sync_time(self) -> Optional[str]:
        """Get the time of the last successful sync."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT sync_time FROM sync_log
                WHERE status = 'success'
                ORDER BY id DESC
                LIMIT 1
            ''')
            result = cursor.fetchone()
            return result[0] if result else None

    def log_sync(self, papers_added: int, papers_updated: int, status: str, error_message: Optional[str] = None):
        """Log a sync operation."""
        sync_time = datetime.datetime.now().isoformat()
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO sync_log (sync_time, papers_added, papers_updated, status, error_message)
                VALUES (?, ?, ?, ?, ?)
            ''', (sync_time, papers_added, papers_updated, status, error_message))
            conn.commit()

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        with self._connect() as conn:
            cursor = conn.cursor()

            cursor.execute('SELECT COUNT(*) FROM papers')
            total = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(*) FROM papers WHERE ai_processed = 1')
            processed = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(*) FROM papers WHERE is_starred = 1')
            starred = cursor.fetchone()[0]

            cursor.execute('SELECT COUNT(*) FROM papers WHERE DATE(created_at) = DATE(?)',
                           (datetime.datetime.now().isoformat(),))
            today_added = cursor.fetchone()[0]

            cursor.execute('SELECT AVG(ai_score) FROM papers WHERE ai_processed = 1')
            avg_score = cursor.fetchone()[0]

            cursor.execute('''
                SELECT sync_time, papers_added, status FROM sync_log
                ORDER BY id DESC LIMIT 5
            ''')
            recent_syncs = cursor.fetchall()

        return {
            'total_papers': total,
            'processed_papers': processed,
            'unprocessed_papers': total - processed,
            'starred_papers': starred,
            'today_added': today_added,
            'average_score': round(avg_score, 2) if avg_score else None,
            'recent_syncs': [dict(sync) for sync in recent_syncs]
        }


    # Saved synthesis methods
    def save_synthesis(self, paper_ids: List[str], content: str) -> int:
        """Save a generated synthesis to database. Returns new synthesis ID."""
        created_at = datetime.datetime.now().isoformat()
        paper_ids_str = ','.join(paper_ids)
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO syntheses (created_at, paper_ids, content)
                VALUES (?, ?, ?)
            ''', (created_at, paper_ids_str, content))
            conn.commit()
            return cursor.lastrowid

    def get_recent_syntheses(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent generated syntheses, newest first."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, created_at, paper_ids, content
                FROM syntheses
                ORDER BY id DESC
                LIMIT ?
            ''', (limit,))
            rows = cursor.fetchall()
        result = []
        for row in rows:
            paper_ids = row[2].split(',') if row[2] else []
            result.append({
                'id': row[0],
                'created_at': row[1],
                'paper_ids': paper_ids,
                'content': row[3],
            })
        return result

    def get_synthesis(self, synthesis_id: int) -> Optional[Dict[str, Any]]:
        """Get a specific synthesis by ID."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT id, created_at, paper_ids, content
                FROM syntheses
                WHERE id = ?
            ''', (synthesis_id,))
            row = cursor.fetchone()
        if not row:
            return None
        paper_ids = row[2].split(',') if row[2] else []
        return {
            'id': row[0],
            'created_at': row[1],
            'paper_ids': paper_ids,
            'content': row[3],
        }

    def delete_synthesis(self, synthesis_id: int) -> bool:
        """Delete a synthesis."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM syntheses WHERE id = ?', (synthesis_id,))
            conn.commit()
            return cursor.rowcount > 0

    def count_syntheses(self) -> int:
        """Count total saved syntheses."""
        with self._connect() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT COUNT(*) FROM syntheses')
            return cursor.fetchone()[0]
