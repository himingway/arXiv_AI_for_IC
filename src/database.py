"""
Database module for ArXiv paper tracking system.
Handles SQLite operations for storing paper metadata and AI analysis results.
"""

import sqlite3
import datetime
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

    def _init_db(self):
        """Create tables if they don't exist."""
        conn = self._get_connection()
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
                processed_at TEXT,
                UNIQUE(id)
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

        conn.commit()
        conn.close()

    def insert_or_update_paper(self, paper: Paper) -> bool:
        """Insert a new paper or update existing one. Returns True if inserted as new."""
        conn = self._get_connection()
        cursor = conn.cursor()

        # Check if exists
        cursor.execute('SELECT id FROM papers WHERE id = ?', (paper.id,))
        exists = cursor.fetchone() is not None

        if paper.created_at is None:
            paper.created_at = datetime.datetime.now().isoformat()

        if exists:
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
        else:
            cursor.execute('''
                INSERT INTO papers (
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

        conn.commit()
        conn.close()
        return not exists

    def update_ai_analysis(self, paper_id: str, score: int, reason: str, tags: str):
        """Update paper with AI analysis results."""
        conn = self._get_connection()
        cursor = conn.cursor()
        processed_at = datetime.datetime.now().isoformat()
        cursor.execute('''
            UPDATE papers
            SET ai_processed = 1, ai_score = ?, ai_reason = ?, ai_tags = ?, processed_at = ?
            WHERE id = ?
        ''', (score, reason, tags, processed_at, paper_id))
        conn.commit()
        conn.close()

    def toggle_starred(self, paper_id: str) -> bool:
        """Toggle starred status. Returns new status."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT is_starred FROM papers WHERE id = ?', (paper_id,))
        result = cursor.fetchone()
        if result is None:
            conn.close()
            return False

        current = bool(result[0])
        new_status = not current
        cursor.execute('UPDATE papers SET is_starred = ? WHERE id = ?', (int(new_status), paper_id))
        conn.commit()
        conn.close()
        return new_status

    def get_paper(self, paper_id: str) -> Optional[Paper]:
        """Get a single paper by ID."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM papers WHERE id = ?', (paper_id,))
        row = cursor.fetchone()
        conn.close()
        if row:
            return Paper.from_row(tuple(row))
        return None

    def get_unprocessed_papers(self) -> List[Paper]:
        """Get all papers that haven't been processed by AI."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM papers WHERE ai_processed = 0 ORDER BY created_at ASC')
        rows = cursor.fetchall()
        conn.close()
        return [Paper.from_row(tuple(row)) for row in rows]

    def count_unprocessed(self) -> int:
        """Count unprocessed papers."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM papers WHERE ai_processed = 0')
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def get_all_papers(self, limit: int = 1000, offset: int = 0) -> List[Paper]:
        """Get all papers sorted by AI score descending."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM papers
            ORDER BY ai_score DESC NULLS LAST, published DESC
            LIMIT ? OFFSET ?
        ''', (limit, offset))
        rows = cursor.fetchall()
        conn.close()
        return [Paper.from_row(tuple(row)) for row in rows]

    def get_filtered_papers(self,
                            only_today: bool = False,
                            min_score: Optional[int] = None,
                            only_starred: bool = False,
                            search: Optional[str] = None,
                            limit: int = 1000,
                            offset: int = 0) -> List[Paper]:
        """Get filtered papers based on criteria."""
        conn = self._get_connection()
        cursor = conn.cursor()

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

        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        return [Paper.from_row(tuple(row)) for row in rows]

    def count_filtered(self,
                       only_today: bool = False,
                       min_score: Optional[int] = None,
                       only_starred: bool = False,
                       search: Optional[str] = None) -> int:
        """Count filtered papers."""
        conn = self._get_connection()
        cursor = conn.cursor()

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

        cursor.execute(query, params)
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def get_total_count(self) -> int:
        """Get total number of papers."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM papers')
        count = cursor.fetchone()[0]
        conn.close()
        return count

    def get_last_sync_time(self) -> Optional[str]:
        """Get the time of the last successful sync."""
        conn = self._get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT sync_time FROM sync_log
            WHERE status = 'success'
            ORDER BY id DESC
            LIMIT 1
        ''')
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else None

    def log_sync(self, papers_added: int, papers_updated: int, status: str, error_message: Optional[str] = None):
        """Log a sync operation."""
        conn = self._get_connection()
        cursor = conn.cursor()
        sync_time = datetime.datetime.now().isoformat()
        cursor.execute('''
            INSERT INTO sync_log (sync_time, papers_added, papers_updated, status, error_message)
            VALUES (?, ?, ?, ?, ?)
        ''', (sync_time, papers_added, papers_updated, status, error_message))
        conn.commit()
        conn.close()

    def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        conn = self._get_connection()
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

        conn.close()

        return {
            'total_papers': total,
            'processed_papers': processed,
            'unprocessed_papers': total - processed,
            'starred_papers': starred,
            'today_added': today_added,
            'average_score': round(avg_score, 2) if avg_score else None,
            'recent_syncs': [dict(sync) for sync in recent_syncs]
        }
