"""MkDocs hook: generate scholar-ready PDFs for blog posts with citation metadata.

Activated by setting SCHOLARLY_PDF=1 in the environment. Skipped during normal
``mkdocs serve`` to keep the dev loop fast. In CI, add the env var before the
build step and ensure Playwright browsers are installed (``playwright install chromium``).
"""

import logging
import os
from pathlib import Path

log = logging.getLogger("mkdocs.hooks.scholarly_pdf")

PRINT_CSS = """
/* Hide site chrome */
.md-header, .md-footer, .md-sidebar, .md-tabs, .md-top,
.md-source, .toc-toggle, .landing-nav, .md-search,
.md-header--shadow, .headerlink,
[data-md-component="navigation"], [data-md-component="toc"],
[data-md-component="header"], .blog-subscribe,
.md-typeset .md-content__button {
    display: none !important;
}

/* Full-width content */
.md-content { margin: 0 !important; }
.md-content__inner { margin: 0 auto !important; max-width: 7in !important; }
.md-main__inner { margin: 0 !important; }

/* Academic styling */
.blog-post-title { text-align: center !important; }
.blog-post-byline { text-align: center !important; margin-bottom: 1.5em !important; }
.blog-post-authors a { color: #000 !important; text-decoration: none !important; }

img { max-width: 100% !important; }
"""


def on_post_build(config, **kwargs):
    """Generate a ``paper.pdf`` alongside each blog post that has citation_authors metadata."""
    if not os.environ.get("SCHOLARLY_PDF"):
        log.info("SCHOLARLY_PDF not set — skipping PDF generation")
        return

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        log.warning("playwright not installed — skipping PDF generation")
        return

    site_dir = Path(config["site_dir"])
    targets = [
        f
        for f in sorted(site_dir.glob("blog/**/index.html"))
        if 'name="citation_title"' in f.read_text(encoding="utf-8")
    ]

    if not targets:
        return

    log.info("Generating scholarly PDFs for %d page(s)", len(targets))

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()

            for html_file in targets:
                pdf_path = html_file.parent / "paper.pdf"
                log.info("  %s", pdf_path.relative_to(site_dir))

                page.goto(html_file.as_uri(), wait_until="networkidle")
                page.add_style_tag(content=PRINT_CSS)
                page.pdf(
                    path=str(pdf_path),
                    format="Letter",
                    margin={"top": "0.75in", "right": "0.75in", "bottom": "0.75in", "left": "0.75in"},
                    print_background=True,
                )

            browser.close()
    except Exception:
        log.exception("PDF generation failed (is 'playwright install chromium' done?)")
        return

    log.info("Done — generated %d PDF(s)", len(targets))
