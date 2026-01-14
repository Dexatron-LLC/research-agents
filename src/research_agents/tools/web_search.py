"""Web search tool for agents using DuckDuckGo and Playwright."""

from dataclasses import dataclass
from typing import Any

from ddgs import DDGS
from playwright.async_api import async_playwright, Browser, Page


@dataclass
class SearchResult:
    """A single search result."""
    title: str
    url: str
    snippet: str


class WebSearchTool:
    """Tool for performing web searches and fetching pages."""

    def __init__(self):
        self._browser: Browser | None = None
        self._page: Page | None = None
        self._playwright = None
        self._ddgs = DDGS()

    async def _get_page(self) -> Page:
        """Get or create a browser page for fetching content."""
        if self._page is None or self._page.is_closed():
            if self._playwright is None:
                self._playwright = await async_playwright().start()
            if self._browser is None:
                self._browser = await self._playwright.chromium.launch(headless=True)
            self._page = await self._browser.new_page()
        return self._page

    async def close(self) -> None:
        """Clean up browser resources."""
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None
        self._page = None

    def search(self, query: str, num_results: int = 10) -> list[SearchResult]:
        """Perform a web search using DuckDuckGo.

        Args:
            query: The search query
            num_results: Maximum number of results to return

        Returns:
            List of search results
        """
        results: list[SearchResult] = []

        for r in self._ddgs.text(query, max_results=num_results):
            results.append(SearchResult(
                title=r.get("title", ""),
                url=r.get("href", ""),
                snippet=r.get("body", ""),
            ))

        return results

    async def fetch_page(self, url: str, max_content_length: int = 10000) -> dict[str, Any]:
        """Fetch and parse a web page using Playwright.

        Args:
            url: The URL to fetch
            max_content_length: Maximum content length to return

        Returns:
            Dictionary with page content and metadata
        """
        page = await self._get_page()
        await page.goto(url)

        title = await page.title()
        content = await page.inner_text("body")
        html = await page.content()

        return {
            "url": url,
            "title": title,
            "content": content[:max_content_length],
            "html": html[:max_content_length],
        }

    async def screenshot(self, url: str | None = None) -> bytes:
        """Take a screenshot of the current or specified page.

        Args:
            url: Optional URL to navigate to before screenshot

        Returns:
            Screenshot as PNG bytes
        """
        page = await self._get_page()
        if url:
            await page.goto(url)
        return await page.screenshot()

    async def click(self, selector: str) -> None:
        """Click an element on the current page.

        Args:
            selector: CSS selector of element to click
        """
        page = await self._get_page()
        await page.click(selector)

    async def fill(self, selector: str, value: str) -> None:
        """Fill a form field on the current page.

        Args:
            selector: CSS selector of input field
            value: Text to fill in
        """
        page = await self._get_page()
        await page.fill(selector, value)

    async def get_element_text(self, selector: str) -> str:
        """Get text content of an element.

        Args:
            selector: CSS selector of element

        Returns:
            Text content of the element
        """
        page = await self._get_page()
        return await page.inner_text(selector)
