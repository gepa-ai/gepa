"""Frontier-CS judge client and C++ code extraction."""

from __future__ import annotations

import re
import time
from typing import Any

import requests


def extract_cpp_code(response_text: str) -> str:
    """Extract C++ code from LLM response, handling ```cpp code blocks."""
    match = re.search(r"```\s*cpp\s*\n(.*?)```", response_text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return response_text.strip()


class FrontierCSJudgeClient:
    """Client for the Frontier-CS judge API (LightCPVerifier)."""

    def __init__(self, judge_url: str = "http://localhost:8081"):
        self.judge_url = judge_url.rstrip("/")
        self.session = requests.Session()

    def submit_solution(self, pid: str, code: str) -> str | None:
        try:
            files = {"code": ("solution.cpp", code)}
            data = {"pid": pid, "lang": "cpp"}
            response = self.session.post(
                f"{self.judge_url}/submit",
                files=files,
                data=data,
                timeout=60,
            )
            response.raise_for_status()
            return response.json().get("sid")
        except requests.RequestException:
            return None

    def get_result(self, sid: str, poll_interval: int = 2, timeout_seconds: int = 300) -> dict[str, Any] | None:
        """Poll for judge result until done or error."""
        start = time.time()
        while time.time() - start < timeout_seconds:
            try:
                response = self.session.get(f"{self.judge_url}/result/{sid}", timeout=10)
                if response.status_code == 404:
                    time.sleep(poll_interval)
                    continue
                response.raise_for_status()
                result = response.json()
                if result.get("status") in ("done", "error"):
                    return result
            except requests.RequestException:
                pass
            time.sleep(poll_interval)
        return {"status": "error", "error": "TIMEOUT", "score": 0}
