#!/usr/bin/env python3
"""Debug the quality chart using Playwright."""

import asyncio
import sys
from playwright.async_api import async_playwright

async def test_chart():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        page = await browser.new_page()

        # Enable console logging
        page.on("console", lambda msg: print(f"[Browser Console] {msg.type}: {msg.text}"))
        page.on("pageerror", lambda err: print(f"[Browser Error] {err}"))

        print("Loading dashboard...")
        await page.goto("http://localhost:8000/scripts/evolution_live_v2.html")

        # Wait for page to load
        await page.wait_for_timeout(3000)

        print("\nChecking if chart canvas exists...")
        chart_exists = await page.locator("#qualityChart").count()
        print(f"Chart canvas found: {chart_exists > 0}")

        print("\nEvaluating chart state...")
        chart_state = await page.evaluate("""() => {
            return {
                chartExists: window.qualityChart !== null && window.qualityChart !== undefined,
                currentRunId: window.currentRunId,
                lastSeenEval: window.lastSeenEval,
                dataPoints: window.qualityChart ? window.qualityChart.data.datasets[0].data.length : 0,
                chartData: window.qualityChart ? window.qualityChart.data.datasets[0].data : []
            };
        }""")

        print(f"\nChart State:")
        print(f"  Chart initialized: {chart_state['chartExists']}")
        print(f"  Current run ID: {chart_state['currentRunId']}")
        print(f"  Last seen eval: {chart_state['lastSeenEval']}")
        print(f"  Data points: {chart_state['dataPoints']}")
        print(f"  Chart data: {chart_state['chartData']}")

        print("\nWaiting for data updates (10 seconds)...")
        await page.wait_for_timeout(10000)

        # Check again
        chart_state = await page.evaluate("""() => {
            return {
                dataPoints: window.qualityChart ? window.qualityChart.data.datasets[0].data.length : 0,
                chartData: window.qualityChart ? window.qualityChart.data.datasets[0].data : []
            };
        }""")

        print(f"\nAfter waiting:")
        print(f"  Data points: {chart_state['dataPoints']}")
        print(f"  Chart data: {chart_state['chartData']}")

        if chart_state['dataPoints'] == 0:
            print("\n❌ No data in chart! Investigating...")

            # Check if fetch is working
            fetch_test = await page.evaluate("""async () => {
                try {
                    const resp = await fetch('/.turbo_gepa/evolution/current.json');
                    const current = await resp.json();
                    const dataResp = await fetch(`/.turbo_gepa/evolution/${current.run_id}.json`);
                    const data = await dataResp.json();
                    return {
                        success: true,
                        runId: data.run_id,
                        timelineLength: data.timeline ? data.timeline.length : 0,
                        timeline: data.timeline
                    };
                } catch (e) {
                    return { success: false, error: e.toString() };
                }
            }""")

            print(f"\nFetch test: {fetch_test}")

        else:
            print("\n✅ Chart has data!")

        print("\nKeeping browser open for inspection. Press Ctrl+C to exit...")
        try:
            await page.wait_for_timeout(300000)  # 5 minutes
        except KeyboardInterrupt:
            print("\nExiting...")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(test_chart())
