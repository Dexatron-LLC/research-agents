"""Report agent that synthesizes information into markdown reports."""

from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from ..core.base_agent import BaseAgent
from ..core.config import Settings, get_settings


class ReportAgent(BaseAgent):
    """Agent specialized in compiling information into cohesive markdown reports."""

    def __init__(self, settings: Settings | None = None):
        super().__init__(name="report")
        self.settings = settings or get_settings()
        self.reports_dir = self.settings.reports_dir
        self.current_report: str | None = None
        self.current_title: str | None = None

    def _get_report_system_prompt(self) -> str:
        """Get system prompt for report generation, combining YAML definition with specific guidelines."""
        base_prompt = super().get_system_prompt()

        guidelines = """
## Report Writing Guidelines
- Use clear, professional language
- Organize information logically with appropriate headings
- Use markdown formatting effectively (headers, lists, emphasis, code blocks where appropriate)
- Include a summary or executive overview at the beginning
- Cite sources when provided
- Highlight key findings and insights
- Be concise but comprehensive

Output only the markdown report content, nothing else."""

        return f"{base_prompt}\n{guidelines}"

    async def _call_llm(self, prompt: str) -> str:
        """Call the LLM API to generate report content."""
        if not self.settings.api_key:
            return "Error: ANTHROPIC_API_KEY not configured. Please set it in .env or environment."

        async with httpx.AsyncClient() as client:
            response = await client.post(
                self.settings.get_api_url("/v1/messages"),
                headers={
                    "x-api-key": self.settings.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.settings.default_model,
                    "max_tokens": 4096,
                    "system": self._get_report_system_prompt(),
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=self.settings.request_timeout * 2,  # Reports may take longer
            )

            if response.status_code != 200:
                return f"Error: API returned status {response.status_code}: {response.text}"

            data = response.json()
            return data["content"][0]["text"]

    async def compile_report(
        self,
        title: str,
        information: list[dict[str, Any]],
        additional_instructions: str | None = None,
    ) -> str:
        """Compile multiple pieces of information into a markdown report.

        Args:
            title: The report title
            information: List of information dicts with 'source', 'content', and optional 'metadata'
            additional_instructions: Optional extra instructions for report formatting

        Returns:
            The generated markdown report
        """
        info_sections = []
        for i, info in enumerate(information, 1):
            source = info.get("source", f"Source {i}")
            content = info.get("content", "")
            metadata = info.get("metadata", {})

            section = f"### Source: {source}\n"
            if metadata:
                section += f"Metadata: {metadata}\n"
            section += f"\n{content}\n"
            info_sections.append(section)

        prompt = f"""Create a comprehensive markdown report with the following title: "{title}"

Based on the following information sources:

{'---'.join(info_sections)}

{f'Additional instructions: {additional_instructions}' if additional_instructions else ''}

Generate a well-structured markdown report that synthesizes all this information."""

        report = await self._call_llm(prompt)
        self.current_report = report
        self.current_title = title
        return report

    async def save_report(self, filename: str | None = None) -> Path:
        """Save the current report to disk.

        Args:
            filename: Optional filename (without extension). If not provided, generates from title.

        Returns:
            Path to the saved report file
        """
        if not self.current_report:
            raise ValueError("No report to save. Generate a report first using compile_report().")

        self.reports_dir.mkdir(parents=True, exist_ok=True)

        if not filename:
            safe_title = "".join(c if c.isalnum() or c in " -_" else "_" for c in (self.current_title or "report"))
            safe_title = safe_title.replace(" ", "_").lower()
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{safe_title}_{timestamp}"

        filepath = self.reports_dir / f"{filename}.md"
        filepath.write_text(self.current_report, encoding="utf-8")
        return filepath

    def list_reports(self) -> list[Path]:
        """List all saved reports.

        Returns:
            List of paths to saved report files
        """
        if not self.reports_dir.exists():
            return []
        return sorted(self.reports_dir.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True)

    def load_report(self, filename: str) -> str:
        """Load a previously saved report.

        Args:
            filename: The report filename (with or without .md extension)

        Returns:
            The report content
        """
        if not filename.endswith(".md"):
            filename = f"{filename}.md"

        filepath = self.reports_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Report not found: {filepath}")

        content = filepath.read_text(encoding="utf-8")
        self.current_report = content
        self.current_title = filepath.stem
        return content

    async def execute(self, task: str, context: dict[str, Any] | None = None) -> dict[str, Any]:
        """Execute a report task.

        Args:
            task: Task description (used as report title if no context provided)
            context: Should contain 'information' list and optionally 'title', 'save', 'filename'

        Returns:
            Dictionary containing the report and metadata
        """
        context = context or {}

        # Handle save request
        if context.get("save") and self.current_report:
            filepath = await self.save_report(context.get("filename"))
            return {
                "status": "saved",
                "filepath": str(filepath),
                "report": self.current_report,
            }

        # Handle list request
        if context.get("list"):
            reports = self.list_reports()
            return {
                "status": "completed",
                "reports": [str(p) for p in reports],
            }

        # Handle load request
        if context.get("load"):
            content = self.load_report(context["load"])
            return {
                "status": "loaded",
                "report": content,
            }

        # Generate new report
        information = context.get("information", [])
        if not information:
            # If no structured information, treat the task as the content
            information = [{"source": "User Input", "content": task}]

        title = context.get("title", task)
        instructions = context.get("instructions")

        report = await self.compile_report(title, information, instructions)

        result = {
            "status": "completed",
            "title": title,
            "report": report,
        }

        # Auto-save if requested
        if context.get("save"):
            filepath = await self.save_report(context.get("filename"))
            result["filepath"] = str(filepath)
            result["status"] = "saved"

        return result
