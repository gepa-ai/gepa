#!/usr/bin/env python3
"""
Test script to verify the evolution dashboard loads and works correctly.
"""

import json
import time
from pathlib import Path
import subprocess
import sys

def check_server_running():
    """Check if HTTP server is running on port 8000."""
    try:
        import urllib.request
        urllib.request.urlopen('http://localhost:8000/', timeout=2)
        return True
    except:
        return False

def check_data_files():
    """Check if required data files exist."""
    current_json = Path('.turbo_gepa/evolution/current.json')

    if not current_json.exists():
        print("❌ current.json not found")
        return False

    with open(current_json) as f:
        current = json.load(f)

    run_id = current.get('run_id')
    if not run_id:
        print("❌ No run_id in current.json")
        return False

    run_json = Path(f'.turbo_gepa/evolution/{run_id}.json')
    if not run_json.exists():
        print(f"❌ Run file {run_json} not found")
        return False

    with open(run_json) as f:
        run_data = json.load(f)

    # Validate structure
    if 'lineage' not in run_data:
        print("❌ No lineage in run data")
        return False

    if 'evolution_stats' not in run_data:
        print("❌ No evolution_stats in run data")
        return False

    lineage_count = len(run_data['lineage'])
    parent_children = run_data['evolution_stats'].get('parent_children', {})
    edge_count = sum(len(children) for children in parent_children.values())

    print(f"✅ Data files valid")
    print(f"   Run ID: {run_id}")
    print(f"   Nodes: {lineage_count}")
    print(f"   Edges: {edge_count}")

    return True

def main():
    print("🔍 Testing TurboGEPA Evolution Dashboard\n")

    # Check if we're in the right directory
    if not Path('.turbo_gepa').exists():
        print("❌ Not in project root (no .turbo_gepa/ directory)")
        sys.exit(1)

    # Check data files
    if not check_data_files():
        sys.exit(1)

    # Check if server is running
    if check_server_running():
        print("✅ HTTP server running on http://localhost:8000")
    else:
        print("⚠️  HTTP server not running")
        print("   Start it with: python3 -m http.server 8000")
        sys.exit(1)

    # Try to fetch the data via HTTP
    try:
        import urllib.request
        import urllib.error

        # Test current.json endpoint
        response = urllib.request.urlopen('http://localhost:8000/.turbo_gepa/evolution/current.json')
        current = json.loads(response.read())
        run_id = current['run_id']

        # Test run data endpoint
        response = urllib.request.urlopen(f'http://localhost:8000/.turbo_gepa/evolution/{run_id}.json')
        run_data = json.loads(response.read())

        print("✅ Server can access data files via HTTP")

        # Test HTML file
        response = urllib.request.urlopen('http://localhost:8000/scripts/evolution_live_v2.html')
        html = response.read().decode('utf-8')

        if 'TurboGEPA Evolution' in html and 'cytoscape' in html:
            print("✅ Dashboard HTML served correctly")
        else:
            print("❌ Dashboard HTML seems incomplete")
            sys.exit(1)

    except urllib.error.URLError as e:
        print(f"❌ HTTP fetch error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

    print("\n✅ All checks passed!")
    print("\n🌐 Open dashboard at: http://localhost:8000/scripts/evolution_live_v2.html")

if __name__ == '__main__':
    main()
