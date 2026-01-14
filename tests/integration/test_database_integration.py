"""Integration tests for database operations."""

import pytest

from research_agents.core.database import DatabaseService


class TestDatabaseSessionLifecycle:
    """Test complete session lifecycle in the database."""

    async def test_complete_session_lifecycle(self, integration_database: DatabaseService):
        """Test creating, using, and ending a session."""
        session_id = "lifecycle-test-session"

        # Create session
        await integration_database.create_session(session_id)

        # Add findings
        finding_ids = []
        for i in range(3):
            fid = await integration_database.save_finding(
                session_id=session_id,
                query="test query",
                title=f"Finding {i}",
                snippet=f"Snippet content {i}",
                url=f"https://example.com/{i}",
                source="test",
            )
            finding_ids.append(fid)

        # Validate findings
        for i, fid in enumerate(finding_ids):
            status = "VERIFIED" if i < 2 else "UNVERIFIED"
            await integration_database.save_validation(
                finding_id=fid,
                session_id=session_id,
                status=status,
                confidence=0.8 if status == "VERIFIED" else 0.2,
                reason=f"Test reason {i}",
                sources=["https://source.com"] if status == "VERIFIED" else [],
            )

        # Save report
        await integration_database.save_report(
            session_id=session_id,
            title="Test Report",
            content="# Test Report\n\nContent here.",
            findings_count=2,
        )

        # Save conversation
        await integration_database.save_message(session_id, "user", "Search for test")
        await integration_database.save_message(session_id, "assistant", "Here are the results")

        # End session
        await integration_database.end_session(session_id)

        # Verify everything was saved correctly
        stats = await integration_database.get_session_stats(session_id)
        assert stats["findings_count"] == 3
        assert stats["validated_count"] == 2  # Only VERIFIED count
        assert stats["reports_count"] == 1

        validated = await integration_database.get_validated_findings(session_id)
        assert len(validated) == 2

        removed = await integration_database.get_removed_findings(session_id)
        assert len(removed) == 1

        history = await integration_database.get_conversation_history(session_id)
        assert len(history) == 2


class TestDatabaseDataIntegrity:
    """Test data integrity across database operations."""

    async def test_findings_validation_relationship(self, integration_database: DatabaseService):
        """Test that validation records correctly reference findings."""
        session_id = "integrity-test-session"
        await integration_database.create_session(session_id)

        # Create finding
        finding_id = await integration_database.save_finding(
            session_id=session_id,
            query="integrity test",
            title="Test Finding",
            snippet="Test content",
        )

        # Create validation
        validation_id = await integration_database.save_validation(
            finding_id=finding_id,
            session_id=session_id,
            status="VERIFIED",
            confidence=0.95,
            reason="Verified by multiple sources",
            sources=["https://source1.com", "https://source2.com"],
        )

        # Retrieve and verify
        validated = await integration_database.get_validated_findings(session_id)
        assert len(validated) == 1
        assert validated[0]["title"] == "Test Finding"
        assert validated[0]["status"] == "VERIFIED"
        assert validated[0]["confidence"] == 0.95
        assert len(validated[0]["sources"]) == 2

    async def test_pending_findings_tracking(self, integration_database: DatabaseService):
        """Test tracking findings that haven't been validated."""
        session_id = "pending-test-session"
        await integration_database.create_session(session_id)

        # Create 5 findings
        finding_ids = []
        for i in range(5):
            fid = await integration_database.save_finding(
                session_id=session_id,
                query="pending test",
                title=f"Finding {i}",
                snippet=f"Content {i}",
            )
            finding_ids.append(fid)

        # Initially all should be pending
        pending = await integration_database.get_pending_findings(session_id)
        assert len(pending) == 5

        # Validate 3 of them
        for fid in finding_ids[:3]:
            await integration_database.save_validation(
                finding_id=fid,
                session_id=session_id,
                status="VERIFIED",
                confidence=0.8,
            )

        # Now only 2 should be pending
        pending = await integration_database.get_pending_findings(session_id)
        assert len(pending) == 2

    async def test_multiple_reports_per_session(self, integration_database: DatabaseService):
        """Test saving and retrieving multiple reports in a session."""
        session_id = "multi-report-session"
        await integration_database.create_session(session_id)

        # Save multiple reports
        report_ids = []
        for i in range(3):
            rid = await integration_database.save_report(
                session_id=session_id,
                title=f"Report {i}",
                content=f"# Report {i}\n\nContent for report {i}.",
                findings_count=i + 1,
            )
            report_ids.append(rid)

        # Retrieve by session
        reports = await integration_database.get_reports_by_session(session_id)
        assert len(reports) == 3

        # Verify all reports are present (order may vary due to timing)
        report_titles = {r["title"] for r in reports}
        assert report_titles == {"Report 0", "Report 1", "Report 2"}

        # Retrieve by ID - this should be consistent
        for i, rid in enumerate(report_ids):
            report = await integration_database.get_report_by_id(rid)
            assert report is not None
            assert report["title"] == f"Report {i}"


class TestDatabaseIsolation:
    """Test that sessions are properly isolated."""

    async def test_session_data_isolation(self, integration_database: DatabaseService):
        """Test that data from different sessions don't mix."""
        session1 = "isolation-session-1"
        session2 = "isolation-session-2"

        await integration_database.create_session(session1)
        await integration_database.create_session(session2)

        # Add data to session 1
        await integration_database.save_finding(
            session_id=session1,
            query="session 1 query",
            title="Session 1 Finding",
            snippet="Session 1 content",
        )
        await integration_database.save_report(
            session_id=session1,
            title="Session 1 Report",
            content="Content 1",
        )

        # Add different data to session 2
        for i in range(3):
            await integration_database.save_finding(
                session_id=session2,
                query="session 2 query",
                title=f"Session 2 Finding {i}",
                snippet=f"Session 2 content {i}",
            )

        # Verify isolation
        stats1 = await integration_database.get_session_stats(session1)
        stats2 = await integration_database.get_session_stats(session2)

        assert stats1["findings_count"] == 1
        assert stats1["reports_count"] == 1

        assert stats2["findings_count"] == 3
        assert stats2["reports_count"] == 0

        # Verify findings are isolated
        findings1 = await integration_database.get_findings_by_session(session1)
        findings2 = await integration_database.get_findings_by_session(session2)

        assert len(findings1) == 1
        assert findings1[0]["title"] == "Session 1 Finding"

        assert len(findings2) == 3
        assert all("Session 2" in f["title"] for f in findings2)


class TestDatabaseConversationHistory:
    """Test conversation history integration."""

    async def test_conversation_ordering(self, integration_database: DatabaseService):
        """Test that conversation messages maintain order."""
        session_id = "conversation-test"
        await integration_database.create_session(session_id)

        messages = [
            ("user", "First message"),
            ("assistant", "First response"),
            ("user", "Second message"),
            ("assistant", "Second response"),
            ("user", "Third message"),
            ("assistant", "Third response"),
        ]

        for role, content in messages:
            await integration_database.save_message(session_id, role, content)

        history = await integration_database.get_conversation_history(session_id)

        assert len(history) == 6
        for i, (role, content) in enumerate(messages):
            assert history[i]["role"] == role
            assert history[i]["content"] == content

    async def test_conversation_with_different_roles(self, integration_database: DatabaseService):
        """Test saving messages with all role types."""
        session_id = "role-test"
        await integration_database.create_session(session_id)

        await integration_database.save_message(session_id, "system", "System prompt")
        await integration_database.save_message(session_id, "user", "User query")
        await integration_database.save_message(session_id, "assistant", "Assistant response")

        history = await integration_database.get_conversation_history(session_id)

        roles = [h["role"] for h in history]
        assert roles == ["system", "user", "assistant"]
