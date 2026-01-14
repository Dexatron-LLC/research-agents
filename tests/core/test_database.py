"""Tests for the database module."""

from pathlib import Path

import pytest

from research_agents.core.config import Settings
from research_agents.core.database import DatabaseService, get_database


class TestDatabaseService:
    """Tests for the DatabaseService class."""

    async def test_connect_creates_directory(self, temp_dir: Path):
        """Test that connect creates the data directory if it doesn't exist."""
        db_path = temp_dir / "subdir" / "test.db"
        settings = Settings(database_path=db_path)
        db = DatabaseService(settings)

        await db.connect()

        assert db_path.parent.exists()
        await db.close()

    async def test_initialize_schema_creates_tables(self, test_database: DatabaseService):
        """Test that schema initialization creates all tables."""
        # Tables should already be created by the fixture
        cursor = await test_database._connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        rows = await cursor.fetchall()
        table_names = {row["name"] for row in rows}

        expected_tables = {
            "research_sessions",
            "research_findings",
            "validated_findings",
            "reports",
            "conversation_history",
        }
        assert expected_tables.issubset(table_names)

    async def test_create_session(self, test_database: DatabaseService):
        """Test creating a research session."""
        session_id = "test-session-001"
        await test_database.create_session(session_id)

        cursor = await test_database._connection.execute(
            "SELECT session_id FROM research_sessions WHERE session_id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
        assert row is not None
        assert row["session_id"] == session_id

    async def test_end_session(self, test_database: DatabaseService):
        """Test ending a research session."""
        session_id = "test-session-002"
        await test_database.create_session(session_id)
        await test_database.end_session(session_id)

        cursor = await test_database._connection.execute(
            "SELECT ended_at FROM research_sessions WHERE session_id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
        assert row["ended_at"] is not None

    async def test_save_finding(self, test_database: DatabaseService):
        """Test saving a research finding."""
        session_id = "test-session-003"
        await test_database.create_session(session_id)

        finding_id = await test_database.save_finding(
            session_id=session_id,
            query="test query",
            title="Test Finding",
            snippet="Test snippet content",
            url="https://example.com",
            source="web_search",
        )

        assert finding_id is not None
        assert finding_id > 0

    async def test_get_findings_by_session(self, test_database: DatabaseService):
        """Test retrieving findings by session."""
        session_id = "test-session-004"
        await test_database.create_session(session_id)

        # Save multiple findings
        for i in range(3):
            await test_database.save_finding(
                session_id=session_id,
                query="query",
                title=f"Finding {i}",
                snippet=f"Snippet {i}",
            )

        findings = await test_database.get_findings_by_session(session_id)

        assert len(findings) == 3
        assert findings[0]["title"] == "Finding 0"
        assert findings[2]["title"] == "Finding 2"

    async def test_save_validation(self, test_database: DatabaseService):
        """Test saving a validation result."""
        session_id = "test-session-005"
        await test_database.create_session(session_id)

        finding_id = await test_database.save_finding(
            session_id=session_id,
            query="query",
            title="Test",
            snippet="Snippet",
        )

        validation_id = await test_database.save_validation(
            finding_id=finding_id,
            session_id=session_id,
            status="VERIFIED",
            confidence=0.85,
            reason="Confirmed by sources",
            sources=["https://wikipedia.org"],
        )

        assert validation_id is not None
        assert validation_id > 0

    async def test_get_validated_findings(self, test_database: DatabaseService):
        """Test retrieving validated findings."""
        session_id = "test-session-006"
        await test_database.create_session(session_id)

        # Save and validate findings
        finding_id = await test_database.save_finding(
            session_id=session_id,
            query="query",
            title="Verified Finding",
            snippet="Content",
        )
        await test_database.save_validation(
            finding_id=finding_id,
            session_id=session_id,
            status="VERIFIED",
            confidence=0.9,
            sources=["https://source.com"],
        )

        validated = await test_database.get_validated_findings(session_id)

        assert len(validated) == 1
        assert validated[0]["title"] == "Verified Finding"
        assert validated[0]["status"] == "VERIFIED"
        assert validated[0]["confidence"] == 0.9
        assert validated[0]["sources"] == ["https://source.com"]

    async def test_get_removed_findings(self, test_database: DatabaseService):
        """Test retrieving removed findings."""
        session_id = "test-session-007"
        await test_database.create_session(session_id)

        finding_id = await test_database.save_finding(
            session_id=session_id,
            query="query",
            title="Unverified Finding",
            snippet="Content",
        )
        await test_database.save_validation(
            finding_id=finding_id,
            session_id=session_id,
            status="UNVERIFIED",
            confidence=0.2,
            reason="No sources found",
        )

        removed = await test_database.get_removed_findings(session_id)

        assert len(removed) == 1
        assert removed[0]["title"] == "Unverified Finding"
        assert removed[0]["status"] == "UNVERIFIED"

    async def test_save_report(self, test_database: DatabaseService):
        """Test saving a report."""
        session_id = "test-session-008"
        await test_database.create_session(session_id)

        report_id = await test_database.save_report(
            session_id=session_id,
            title="Test Report",
            content="# Test Report\n\nContent here.",
            findings_count=5,
            filename="test_report.md",
        )

        assert report_id is not None
        assert report_id > 0

    async def test_get_reports_by_session(self, test_database: DatabaseService):
        """Test retrieving reports by session."""
        session_id = "test-session-009"
        await test_database.create_session(session_id)

        await test_database.save_report(
            session_id=session_id,
            title="Report 1",
            content="Content 1",
        )
        await test_database.save_report(
            session_id=session_id,
            title="Report 2",
            content="Content 2",
        )

        reports = await test_database.get_reports_by_session(session_id)

        assert len(reports) == 2

    async def test_get_all_reports(self, test_database: DatabaseService):
        """Test retrieving all reports."""
        # Create reports in different sessions
        for i in range(3):
            session_id = f"test-session-all-{i}"
            await test_database.create_session(session_id)
            await test_database.save_report(
                session_id=session_id,
                title=f"Report {i}",
                content=f"Content {i}",
            )

        reports = await test_database.get_all_reports(limit=10)
        assert len(reports) >= 3

    async def test_get_report_by_id(self, test_database: DatabaseService):
        """Test retrieving a specific report by ID."""
        session_id = "test-session-010"
        await test_database.create_session(session_id)

        report_id = await test_database.save_report(
            session_id=session_id,
            title="Specific Report",
            content="Specific content",
        )

        report = await test_database.get_report_by_id(report_id)

        assert report is not None
        assert report["title"] == "Specific Report"
        assert report["content"] == "Specific content"

    async def test_get_report_by_id_not_found(self, test_database: DatabaseService):
        """Test retrieving nonexistent report."""
        report = await test_database.get_report_by_id(99999)
        assert report is None

    async def test_save_message(self, test_database: DatabaseService):
        """Test saving a conversation message."""
        session_id = "test-session-011"
        await test_database.create_session(session_id)

        message_id = await test_database.save_message(
            session_id=session_id,
            role="user",
            content="Hello, this is a test message.",
        )

        assert message_id is not None
        assert message_id > 0

    async def test_get_conversation_history(self, test_database: DatabaseService):
        """Test retrieving conversation history."""
        session_id = "test-session-012"
        await test_database.create_session(session_id)

        await test_database.save_message(session_id, "user", "Hello")
        await test_database.save_message(session_id, "assistant", "Hi there!")
        await test_database.save_message(session_id, "user", "How are you?")

        history = await test_database.get_conversation_history(session_id)

        assert len(history) == 3
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "Hello"
        assert history[1]["role"] == "assistant"

    async def test_get_session_stats(self, test_database: DatabaseService):
        """Test retrieving session statistics."""
        session_id = "test-session-013"
        await test_database.create_session(session_id)

        # Add some data
        finding_id = await test_database.save_finding(
            session_id=session_id,
            query="query",
            title="Finding",
            snippet="Content",
        )
        await test_database.save_validation(
            finding_id=finding_id,
            session_id=session_id,
            status="VERIFIED",
            confidence=0.8,
        )
        await test_database.save_report(
            session_id=session_id,
            title="Report",
            content="Content",
        )

        stats = await test_database.get_session_stats(session_id)

        assert stats["findings_count"] == 1
        assert stats["validated_count"] == 1
        assert stats["reports_count"] == 1

    async def test_pending_findings(self, test_database: DatabaseService):
        """Test getting pending (unvalidated) findings."""
        session_id = "test-session-014"
        await test_database.create_session(session_id)

        # Add two findings
        finding1_id = await test_database.save_finding(
            session_id=session_id,
            query="query",
            title="Pending Finding",
            snippet="Content",
        )
        finding2_id = await test_database.save_finding(
            session_id=session_id,
            query="query",
            title="Validated Finding",
            snippet="Content",
        )

        # Validate only one
        await test_database.save_validation(
            finding_id=finding2_id,
            session_id=session_id,
            status="VERIFIED",
            confidence=0.9,
        )

        pending = await test_database.get_pending_findings(session_id)

        assert len(pending) == 1
        assert pending[0]["title"] == "Pending Finding"


class TestGetDatabase:
    """Tests for the get_database function."""

    def test_returns_database_service(self):
        """Test get_database returns a DatabaseService instance."""
        db = get_database()
        assert isinstance(db, DatabaseService)

    def test_returns_singleton(self):
        """Test get_database returns the same instance."""
        db1 = get_database()
        db2 = get_database()
        assert db1 is db2
