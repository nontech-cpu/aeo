from html import escape
from pathlib import Path
from jinja2 import Environment, FileSystemLoader, select_autoescape
from app.schemas.report_schema import FullAuditReport
from app.services.browser_agent.browser_context import BrowserContextManager

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates"
_jinja_env = Environment(
    loader=FileSystemLoader(str(TEMPLATE_DIR)),
    autoescape=select_autoescape(["html", "xml"]),
)

class PDFReportGenerator:
    """
    Industry-grade Executive PDF Engine using Playwright Chromium Print-CSS.
    Renders clean semantic HTML (real tables, formatted lists, crisp typography, zero spacing artifacts)
    with automatic running page numbers.
    """

    @classmethod
    async def generate_pdf_bytes_async(cls, report: FullAuditReport) -> bytes:
        template = _jinja_env.get_template("report_pdf.html")
        rendered_html = template.render(report=report)

        header_html = (
            "<div style='width:100%; font-size:7.5pt; color:#64748b; padding:0 14mm; "
            "display:flex; justify-content:space-between; font-family:sans-serif;'>"
            f"<span>ShapeBytesAEO Audit &mdash; {escape(report.brand_name)}</span>"
            f"<span>{escape(report.website_url)}</span>"
            "</div>"
        )
        footer_html = (
            "<div style='width:100%; font-size:7.5pt; color:#64748b; padding:0 14mm; "
            "display:flex; justify-content:space-between; font-family:sans-serif;'>"
            f"<span>Generated {escape(report.timestamp)}</span>"
            "<span>Page <span class='pageNumber'></span> of <span class='totalPages'></span></span>"
            "</div>"
        )

        return await BrowserContextManager.render_html_to_pdf(
            html_content=rendered_html,
            header_html=header_html,
            footer_html=footer_html,
        )
