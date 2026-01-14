"""Tests for the web search tool."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from research_agents.tools.web_search import WebSearchTool, SearchResult


class TestSearchResult:
    """Tests for the SearchResult dataclass."""

    def test_create_search_result(self):
        """Test creating a SearchResult."""
        result = SearchResult(
            title="Test Title",
            url="https://example.com",
            snippet="Test snippet content",
        )

        assert result.title == "Test Title"
        assert result.url == "https://example.com"
        assert result.snippet == "Test snippet content"

    def test_search_result_equality(self):
        """Test SearchResult equality."""
        result1 = SearchResult("Title", "https://url.com", "Snippet")
        result2 = SearchResult("Title", "https://url.com", "Snippet")

        assert result1 == result2

    def test_search_result_repr(self):
        """Test SearchResult string representation."""
        result = SearchResult("Title", "https://url.com", "Snippet")
        repr_str = repr(result)

        assert "SearchResult" in repr_str
        assert "Title" in repr_str


class TestWebSearchToolInit:
    """Tests for WebSearchTool initialization."""

    def test_init(self, mock_ddgs):
        """Test tool initialization."""
        tool = WebSearchTool()

        assert tool._browser is None
        assert tool._page is None
        assert tool._playwright is None
        assert tool._ddgs is not None

    def test_init_creates_ddgs_client(self, mock_ddgs):
        """Test that DDGS client is created on init."""
        tool = WebSearchTool()
        assert tool._ddgs is not None


class TestWebSearchToolSearch:
    """Tests for the search method."""

    def test_search_returns_results(self, mock_ddgs):
        """Test that search returns SearchResult objects."""
        tool = WebSearchTool()
        results = tool.search("test query")

        assert len(results) == 2
        assert all(isinstance(r, SearchResult) for r in results)

    def test_search_result_content(self, mock_ddgs):
        """Test search result content."""
        tool = WebSearchTool()
        results = tool.search("test query")

        assert results[0].title == "Mock Result 1"
        assert results[0].url == "https://mock1.com"
        assert results[0].snippet == "Mock snippet 1"

    def test_search_with_num_results(self, mock_ddgs):
        """Test search with custom number of results."""
        tool = WebSearchTool()
        tool.search("query", num_results=3)

        mock_ddgs.text.assert_called_with("query", max_results=3)

    def test_search_default_num_results(self, mock_ddgs):
        """Test search with default number of results."""
        tool = WebSearchTool()
        tool.search("query")

        mock_ddgs.text.assert_called_with("query", max_results=10)

    def test_search_empty_results(self, mock_ddgs):
        """Test search when no results are returned."""
        mock_ddgs.text.return_value = []

        tool = WebSearchTool()
        results = tool.search("obscure query")

        assert results == []

    def test_search_handles_missing_fields(self, mock_ddgs):
        """Test search handles results with missing fields."""
        mock_ddgs.text.return_value = [
            {"title": "Only Title"},  # Missing href and body
            {"href": "https://only-url.com"},  # Missing title and body
        ]

        tool = WebSearchTool()
        results = tool.search("query")

        assert len(results) == 2
        assert results[0].title == "Only Title"
        assert results[0].url == ""
        assert results[0].snippet == ""
        assert results[1].title == ""
        assert results[1].url == "https://only-url.com"


class TestWebSearchToolBrowser:
    """Tests for browser-related functionality."""

    async def test_close_without_browser(self, mock_ddgs):
        """Test closing when browser was never opened."""
        tool = WebSearchTool()
        await tool.close()

        # Should not raise any exceptions
        assert tool._browser is None
        assert tool._playwright is None

    async def test_close_with_browser(self, mock_ddgs):
        """Test closing with active browser."""
        tool = WebSearchTool()

        # Mock the browser components
        mock_browser = AsyncMock()
        mock_playwright = AsyncMock()

        tool._browser = mock_browser
        tool._playwright = mock_playwright
        tool._page = MagicMock()

        await tool.close()

        mock_browser.close.assert_called_once()
        mock_playwright.stop.assert_called_once()
        assert tool._browser is None
        assert tool._playwright is None
        assert tool._page is None


class TestWebSearchToolFetchPage:
    """Tests for the fetch_page method."""

    async def test_fetch_page(self, mock_ddgs):
        """Test fetching a page."""
        tool = WebSearchTool()

        # Mock the browser page
        mock_page = AsyncMock()
        mock_page.title.return_value = "Test Page Title"
        mock_page.inner_text.return_value = "Page content here"
        mock_page.content.return_value = "<html><body>HTML content</body></html>"
        mock_page.is_closed.return_value = False

        # Mock _get_page to return our mock page
        tool._page = mock_page

        with patch.object(tool, "_get_page", return_value=mock_page):
            result = await tool.fetch_page("https://example.com")

        assert result["url"] == "https://example.com"
        assert result["title"] == "Test Page Title"
        assert "content" in result
        assert "html" in result

    async def test_fetch_page_truncates_content(self, mock_ddgs):
        """Test that fetch_page truncates long content."""
        tool = WebSearchTool()

        mock_page = AsyncMock()
        mock_page.title.return_value = "Title"
        mock_page.inner_text.return_value = "x" * 20000
        mock_page.content.return_value = "y" * 20000
        mock_page.is_closed.return_value = False

        with patch.object(tool, "_get_page", return_value=mock_page):
            result = await tool.fetch_page("https://example.com", max_content_length=100)

        assert len(result["content"]) == 100
        assert len(result["html"]) == 100


class TestWebSearchToolInteractions:
    """Tests for page interaction methods."""

    async def test_screenshot(self, mock_ddgs):
        """Test taking a screenshot."""
        tool = WebSearchTool()

        mock_page = AsyncMock()
        mock_page.screenshot.return_value = b"PNG image data"
        mock_page.is_closed.return_value = False

        with patch.object(tool, "_get_page", return_value=mock_page):
            result = await tool.screenshot()

        assert result == b"PNG image data"
        mock_page.screenshot.assert_called_once()

    async def test_screenshot_with_url(self, mock_ddgs):
        """Test screenshot after navigating to URL."""
        tool = WebSearchTool()

        mock_page = AsyncMock()
        mock_page.screenshot.return_value = b"PNG data"
        mock_page.is_closed.return_value = False

        with patch.object(tool, "_get_page", return_value=mock_page):
            await tool.screenshot(url="https://example.com")

        mock_page.goto.assert_called_once_with("https://example.com")

    async def test_click(self, mock_ddgs):
        """Test clicking an element."""
        tool = WebSearchTool()

        mock_page = AsyncMock()
        mock_page.is_closed.return_value = False

        with patch.object(tool, "_get_page", return_value=mock_page):
            await tool.click("#button")

        mock_page.click.assert_called_once_with("#button")

    async def test_fill(self, mock_ddgs):
        """Test filling a form field."""
        tool = WebSearchTool()

        mock_page = AsyncMock()
        mock_page.is_closed.return_value = False

        with patch.object(tool, "_get_page", return_value=mock_page):
            await tool.fill("#input", "test value")

        mock_page.fill.assert_called_once_with("#input", "test value")

    async def test_get_element_text(self, mock_ddgs):
        """Test getting element text."""
        tool = WebSearchTool()

        mock_page = AsyncMock()
        mock_page.inner_text.return_value = "Element text content"
        mock_page.is_closed.return_value = False

        with patch.object(tool, "_get_page", return_value=mock_page):
            result = await tool.get_element_text("#element")

        assert result == "Element text content"
        mock_page.inner_text.assert_called_once_with("#element")
