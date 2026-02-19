#!/usr/bin/env python3
"""
Generate social media preview screenshots from built documentation.
Only runs in CI (check for CI environment variable).
Replaces auto-generated social cards with real page screenshots.
"""

import os
import sys
from pathlib import Path
from typing import Optional

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("Warning: playwright not installed, skipping social screenshots")
    sys.exit(0)


def get_pages_to_screenshot() -> list[tuple[str, str]]:
    """Return list of (html_file, output_path) tuples for key pages."""
    return [
        ("site/index.html", "site/assets/social/home.png"),
        ("site/guides/use-cases/index.html", "site/assets/social/showcase.png"),
        ("site/about/index.html", "site/assets/social/about.png"),
        ("site/blog/index.html", "site/assets/social/blog.png"),
        ("site/guides/index.html", "site/assets/social/guides.png"),
        ("site/api/index.html", "site/assets/social/api.png"),
        ("site/tutorials/index.html", "site/assets/social/tutorials.png"),
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
    """Update OG image tags in HTML files to point to generated screenshots."""
    print("🔗 Updating OG image tags...")

    # Map of HTML file to OG image URL
    og_updates = {
        "index.html": "/assets/social/home.png",
        "guides/use-cases/index.html": "/assets/social/showcase.png",
        "about/index.html": "/assets/social/about.png",
        "blog/index.html": "/assets/social/blog.png",
        "guides/index.html": "/assets/social/guides.png",
        "api/index.html": "/assets/social/api.png",
        "tutorials/index.html": "/assets/social/tutorials.png",
    }

    for html_file, og_image_url in og_updates.items():
        full_path = Path(site_dir) / html_file

        if not full_path.exists():
            continue

        try:
            content = full_path.read_text(encoding="utf-8")

            # Replace or add og:image tag
            if 'property="og:image"' in content:
                # Replace existing og:image tag with new URL
                import re
                pattern = r'<meta property="og:image" content="[^"]*"'
                replacement = f'<meta property="og:image" content="{og_image_url}"'
                content = re.sub(pattern, replacement, content)
            else:
                # Add og:image tag after og:title
                og_image_tag = f'\n    <meta property="og:image" content="{og_image_url}">'
                content = content.replace("</title>", f"</title>{og_image_tag}", 1)

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
