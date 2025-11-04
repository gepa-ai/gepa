#!/usr/bin/env python3
"""
Check OpenRouter API rate limits for your key.

Usage:
    export OPENROUTER_API_KEY=your_key_here
    python scripts/check_openrouter_limits.py
"""

import json
import os
import sys

import requests


def check_rate_limits():
    """Query OpenRouter API for rate limit information."""
    api_key = os.environ.get("OPENROUTER_API_KEY")

    if not api_key:
        print("❌ OPENROUTER_API_KEY environment variable not set", file=sys.stderr)
        print("\nUsage:")
        print("  export OPENROUTER_API_KEY=your_key_here")
        print("  python scripts/check_openrouter_limits.py")
        sys.exit(1)

    print("🔍 Checking OpenRouter rate limits...")
    print(f"Key: {api_key[:8]}...{api_key[-4:]}\n")

    try:
        response = requests.get(
            url="https://openrouter.ai/api/v1/key",
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=10,
        )

        if response.status_code == 200:
            data = response.json()
            print("✅ API Key Info:")
            print(json.dumps(data, indent=2))

            # Parse key information if available
            if "data" in data:
                key_data = data["data"]
                print("\n📊 Summary:")

                if "label" in key_data:
                    print(f"   Label: {key_data['label']}")

                if "limit" in key_data:
                    print(f"   Rate Limit: {key_data['limit']} requests/minute")

                if "usage" in key_data:
                    print(f"   Current Usage: {key_data['usage']} requests")

                if "is_free_tier" in key_data:
                    tier = "Free Tier" if key_data["is_free_tier"] else "Paid Tier"
                    print(f"   Tier: {tier}")

        elif response.status_code == 401:
            print("❌ Invalid API key", file=sys.stderr)
            sys.exit(1)

        else:
            print(f"❌ Error: HTTP {response.status_code}", file=sys.stderr)
            print(response.text)
            sys.exit(1)

    except requests.exceptions.RequestException as e:
        print(f"❌ Request failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    check_rate_limits()
