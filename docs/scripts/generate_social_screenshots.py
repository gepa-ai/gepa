#!/usr/bin/env python3
"""
Generate social media preview screenshots from built documentation.
Only runs in CI (check for CI environment variable).
Replaces auto-generated social cards with real page screenshots.
"""

import os
import sys
from pathlib import Path

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("Warning: playwright not installed, skipping social screenshots")
    sys.exit(0)


def get_pages_to_screenshot() -> list[tuple[str, str]]:
    """Return list of (html_file, output_path) tuples for key pages.

    Paths are relative to site_dir (no site/ prefix — generate_screenshots
    prepends site_dir automatically).
    """
    return [
        ("index.html", "assets/social/home.png"),
        ("guides/use-cases/index.html", "assets/social/showcase.png"),
        ("about/index.html", "assets/social/about.png"),
        ("blog/index.html", "assets/social/blog.png"),
        ("guides/index.html", "assets/social/guides.png"),
        ("api/index.html", "assets/social/api.png"),
        ("tutorials/index.html", "assets/social/tutorials.png"),
    ]


def generate_screenshots(site_dir: str = "site") -> None:
    """Generate screenshots for pages and save as OG images."""
    print("🖼️  Generating social preview screenshots...")

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(args=["--no-sandbox"])

            for html_file, output_path in get_pages_to_screenshot():
                full_path = Path(site_dir) / html_file

                if not full_path.exists():
                    print(f"  ⊘ Skipping {html_file} (not found)")
                    continue

                try:
                    # Create output directory
                    output_file = Path(site_dir) / output_path
                    output_file.parent.mkdir(parents=True, exist_ok=True)

                    # Render page at 1200x630 (standard OG image size)
                    page = browser.new_page(viewport={"width": 1200, "height": 630})
                    page.goto(f"file://{full_path.resolve()}", wait_until="networkidle")

                    # Take screenshot
                    page.screenshot(path=str(output_file), full_page=False)
                    page.close()

                    print(f"  ✓ Generated: {output_path}")

                except Exception as e:
                    print(f"  ✗ Failed to screenshot {html_file}: {e}")
                    continue

            browser.close()
            print("✓ Social preview screenshots generated")

    except Exception as e:
        print(f"Error generating screenshots: {e}")
        sys.exit(1)


def update_og_tags(site_dir: str = "site") -> None:
    """Update OG and Twitter image tags in HTML files to point to generated screenshots."""
    import re

    print("🔗 Updating OG image tags...")

    # Map of HTML file to OG image path (relative to site root)
    og_updates = {
        "index.html": "/assets/social/home.png",
        "guides/use-cases/index.html": "/assets/social/showcase.png",
        "about/index.html": "/assets/social/about.png",
        "blog/index.html": "/assets/social/blog.png",
        "guides/index.html": "/assets/social/guides.png",
        "api/index.html": "/assets/social/api.png",
        "tutorials/index.html": "/assets/social/tutorials.png",
    }

    for html_file, og_image_path in og_updates.items():
        full_path = Path(site_dir) / html_file
        screenshot_path = Path(site_dir) / og_image_path.lstrip("/")

        if not full_path.exists():
            continue

        # Only update if the screenshot was actually generated
        if not screenshot_path.exists():
            print(f"  ⊘ Skipping {html_file} (screenshot not found)")
            continue

        try:
            content = full_path.read_text(encoding="utf-8")

            # Replace og:image
            if 'property="og:image"' in content:
                content = re.sub(
                    r'<meta property="og:image" content="[^"]*"',
                    f'<meta property="og:image" content="{og_image_path}"',
                    content,
                )
            else:
                og_tag = f'\n    <meta property="og:image" content="{og_image_path}">'
                content = content.replace("</title>", f"</title>{og_tag}", 1)

            # Replace twitter:image
            if 'property="twitter:image"' in content:
                content = re.sub(
                    r'<meta property="twitter:image" content="[^"]*"',
                    f'<meta property="twitter:image" content="{og_image_path}"',
                    content,
                )
            elif 'name="twitter:image"' in content:
                content = re.sub(
                    r'<meta name="twitter:image" content="[^"]*"',
                    f'<meta name="twitter:image" content="{og_image_path}"',
                    content,
                )

            full_path.write_text(content, encoding="utf-8")
            print(f"  ✓ Updated: {html_file}")

        except Exception as e:
            print(f"  ✗ Failed to update {html_file}: {e}")


if __name__ == "__main__":
    # Only run in CI
    if not os.getenv("CI"):
        print("Not in CI environment, skipping social screenshot generation")
        sys.exit(0)

    generate_screenshots()
    update_og_tags()
