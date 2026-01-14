"""Database service for storing research data using SQLite."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import aiosqlite

from .config import Settings, get_settings


# SQL schema for creating tables
SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS research_sessions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL UNIQUE,
    started_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    ended_at TEXT NULL
);

CREATE INDEX IF NOT EXISTS idx_session_id ON research_sessions(session_id);

CREATE TABLE IF NOT EXISTS research_findings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    query TEXT NOT NULL,
    title TEXT NOT NULL,
    snippet TEXT,
    url TEXT,
    source TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_session_findings ON research_findings(session_id);
CREATE INDEX IF NOT EXISTS idx_query ON research_findings(query);

CREATE TABLE IF NOT EXISTS validated_findings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    finding_id INTEGER NOT NULL,
    session_id TEXT NOT NULL,
    status TEXT NOT NULL CHECK(status IN ('VERIFIED', 'PARTIALLY_VERIFIED', 'UNVERIFIED', 'CONTRADICTED')),
    confidence REAL NOT NULL,
    reason TEXT,
    sources TEXT,
    validated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (finding_id) REFERENCES research_findings(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_session_validated ON validated_findings(session_id);
CREATE INDEX IF NOT EXISTS idx_status ON validated_findings(status);

CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    findings_count INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    filename TEXT
);

CREATE INDEX IF NOT EXISTS idx_session_reports ON reports(session_id);
CREATE INDEX IF NOT EXISTS idx_created_at ON reports(created_at);

CREATE TABLE IF NOT EXISTS conversation_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_session_history ON conversation_history(session_id);
"""


class DatabaseService:
    """Async SQLite database service for research agents."""

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self._db_path = self.settings.database_path
        self._connection: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        """Create database connection."""
        if self._connection is not None:
            return

        # Ensure data directory exists
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._connection = await aiosqlite.connect(self._db_path)
        self._connection.row_factory = aiosqlite.Row
        # Enable foreign keys
        await self._connection.execute("PRAGMA foreign_keys = ON")

    async def close(self) -> None:
        """Close database connection."""
        if self._connection:
            await self._connection.close()
            self._connection = None

    async def initialize_schema(self) -> None:
        """Create database tables if they don't exist."""
        if not self._connection:
            await self.connect()

        await self._connection.executescript(SCHEMA_SQL)
        await self._connection.commit()

    # Session management
    async def create_session(self, session_id: str) -> None:
        """Create a new research session."""
        await self._connection.execute(
            "INSERT INTO research_sessions (session_id) VALUES (?)",
            (session_id,),
        )
        await self._connection.commit()

    async def end_session(self, session_id: str) -> None:
        """Mark a session as ended."""
        await self._connection.execute(
            "UPDATE research_sessions SET ended_at = ? WHERE session_id = ?",
            (datetime.now().isoformat(), session_id),
        )
        await self._connection.commit()

    # Research findings
    async def save_finding(
        self,
        session_id: str,
        query: str,
        title: str,
        snippet: str = "",
        url: str = "",
        source: str = "",
    ) -> int:
        """Save a research finding and return its ID."""
        cursor = await self._connection.execute(
            """INSERT INTO research_findings
               (session_id, query, title, snippet, url, source)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, query, title, snippet, url, source),
        )
        await self._connection.commit()
        return cursor.lastrowid

    async def get_findings_by_session(self, session_id: str) -> list[dict[str, Any]]:
        """Get all findings for a session."""
        cursor = await self._connection.execute(
            """SELECT * FROM research_findings
               WHERE session_id = ? ORDER BY created_at""",
            (session_id,),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_pending_findings(self, session_id: str) -> list[dict[str, Any]]:
        """Get findings that haven't been validated yet."""
        cursor = await self._connection.execute(
            """SELECT rf.* FROM research_findings rf
               LEFT JOIN validated_findings vf ON rf.id = vf.finding_id
               WHERE rf.session_id = ? AND vf.id IS NULL
               ORDER BY rf.created_at""",
            (session_id,),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    # Validation
    async def save_validation(
        self,
        finding_id: int,
        session_id: str,
        status: str,
        confidence: float,
        reason: str = "",
        sources: list[str] | None = None,
    ) -> int:
        """Save a validation result."""
        cursor = await self._connection.execute(
            """INSERT INTO validated_findings
               (finding_id, session_id, status, confidence, reason, sources)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                finding_id,
                session_id,
                status,
                confidence,
                reason,
                json.dumps(sources or []),
            ),
        )
        await self._connection.commit()
        return cursor.lastrowid

    async def get_validated_findings(self, session_id: str) -> list[dict[str, Any]]:
        """Get all validated findings for a session."""
        cursor = await self._connection.execute(
            """SELECT rf.*, vf.status, vf.confidence, vf.reason, vf.sources, vf.validated_at
               FROM research_findings rf
               JOIN validated_findings vf ON rf.id = vf.finding_id
               WHERE rf.session_id = ? AND vf.status IN ('VERIFIED', 'PARTIALLY_VERIFIED')
               ORDER BY vf.confidence DESC""",
            (session_id,),
        )
        rows = await cursor.fetchall()
        results = []
        for row in rows:
            d = dict(row)
            if d.get("sources"):
                d["sources"] = json.loads(d["sources"])
            results.append(d)
        return results

    async def get_removed_findings(self, session_id: str) -> list[dict[str, Any]]:
        """Get findings that were removed during validation."""
        cursor = await self._connection.execute(
            """SELECT rf.*, vf.status, vf.confidence, vf.reason
               FROM research_findings rf
               JOIN validated_findings vf ON rf.id = vf.finding_id
               WHERE rf.session_id = ? AND vf.status IN ('UNVERIFIED', 'CONTRADICTED')
               ORDER BY vf.validated_at""",
            (session_id,),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    # Reports
    async def save_report(
        self,
        session_id: str,
        title: str,
        content: str,
        findings_count: int = 0,
        filename: str | None = None,
    ) -> int:
        """Save a report."""
        cursor = await self._connection.execute(
            """INSERT INTO reports
               (session_id, title, content, findings_count, filename)
               VALUES (?, ?, ?, ?, ?)""",
            (session_id, title, content, findings_count, filename),
        )
        await self._connection.commit()
        return cursor.lastrowid

    async def get_reports_by_session(self, session_id: str) -> list[dict[str, Any]]:
        """Get all reports for a session."""
        cursor = await self._connection.execute(
            """SELECT * FROM reports
               WHERE session_id = ? ORDER BY created_at DESC""",
            (session_id,),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_all_reports(self, limit: int = 50) -> list[dict[str, Any]]:
        """Get recent reports across all sessions."""
        cursor = await self._connection.execute(
            """SELECT * FROM reports
               ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def get_report_by_id(self, report_id: int) -> dict[str, Any] | None:
        """Get a specific report by ID."""
        cursor = await self._connection.execute(
            "SELECT * FROM reports WHERE id = ?",
            (report_id,),
        )
        row = await cursor.fetchone()
        return dict(row) if row else None

    # Conversation history
    async def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
    ) -> int:
        """Save a conversation message."""
        cursor = await self._connection.execute(
            """INSERT INTO conversation_history
               (session_id, role, content) VALUES (?, ?, ?)""",
            (session_id, role, content),
        )
        await self._connection.commit()
        return cursor.lastrowid

    async def get_conversation_history(self, session_id: str) -> list[dict[str, Any]]:
        """Get conversation history for a session."""
        cursor = await self._connection.execute(
            """SELECT role, content, created_at FROM conversation_history
               WHERE session_id = ? ORDER BY created_at""",
            (session_id,),
        )
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    # Statistics
    async def get_session_stats(self, session_id: str) -> dict[str, Any]:
        """Get statistics for a session."""
        # Count findings
        cursor = await self._connection.execute(
            "SELECT COUNT(*) as count FROM research_findings WHERE session_id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
        findings_count = row["count"]

        # Count validated
        cursor = await self._connection.execute(
            """SELECT COUNT(*) as count FROM validated_findings
               WHERE session_id = ? AND status IN ('VERIFIED', 'PARTIALLY_VERIFIED')""",
            (session_id,),
        )
        row = await cursor.fetchone()
        validated_count = row["count"]

        # Count reports
        cursor = await self._connection.execute(
            "SELECT COUNT(*) as count FROM reports WHERE session_id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
        reports_count = row["count"]

        return {
            "findings_count": findings_count,
            "validated_count": validated_count,
            "reports_count": reports_count,
        }


# Singleton instance
_db_service: DatabaseService | None = None


def get_database() -> DatabaseService:
    """Get the database service singleton."""
    global _db_service
    if _db_service is None:
        _db_service = DatabaseService()
    return _db_service
