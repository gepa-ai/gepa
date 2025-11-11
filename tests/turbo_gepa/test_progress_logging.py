import json
import time

from turbo_gepa.logging.progress import ProgressReporter, ProgressSnapshot
from turbo_gepa.logging.logger import LogLevel, LoggerProtocol


class _CollectingLogger(LoggerProtocol):
    def __init__(self) -> None:
        self.messages: list[tuple[str, LogLevel]] = []

    def log(self, message: str, level: LogLevel = LogLevel.INFO):
        self.messages.append((message, level))


def test_progress_reporter_emits_human_and_structured_logs():
    logger = _CollectingLogger()
    reporter = ProgressReporter(logger)

    now = time.time()
    snapshot = ProgressSnapshot(
        timestamp=now,
        elapsed=5.0,
        run_id="abc123",
        round_index=1,
        evaluations=10,
        pareto_size=2,
        best_quality=0.6,
        best_quality_shard=1.0,
        best_prompt_snippet="Best prompt snippet",
        queue_size=3,
        inflight_candidates=2,
        inflight_examples=2,
        target_quality=0.8,
        target_reached=False,
        stop_reason=None,
    )

    reporter(snapshot)

    # First logged line should be the human-readable summary with speed/delta info
    text_msg, text_level = logger.messages[0]
    assert "speed=" in text_msg
    assert "Δ=+0.600" in text_msg
    assert "target=0.80" in text_msg
    assert text_level == LogLevel.WARNING

    # Last log is the structured JSON payload
    structured_msg, _ = logger.messages[-1]
    structured = json.loads(structured_msg)
    assert structured["event"] == "progress"
    assert structured["run_id"] == "abc123"
    assert structured["best_quality_delta"] == snapshot.best_quality
    assert structured["eval_rate_per_sec"] > 0

    # Calling again with the same round shouldn't emit new logs
    prior_len = len(logger.messages)
    reporter(snapshot)
    assert len(logger.messages) == prior_len

