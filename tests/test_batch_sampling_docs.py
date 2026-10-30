"""Keep the offline batch-sampling guide example executable and accurate."""

import re
from pathlib import Path


def test_offline_coverage_example(capsys):
    guide = Path(__file__).resolve().parents[1] / "docs/docs/guides/batch-sampling.md"
    text = guide.read_text(encoding="utf-8")
    examples = [
        block
        for block in re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)
        if block.startswith("# Offline coverage example\n")
    ]
    assert len(examples) == 1
    exec(compile(examples[0], str(guide), "exec"), {})
    expected = re.search(r"Expected output:\n\n```text\n(.*?)```", text, flags=re.DOTALL)
    assert expected is not None
    assert capsys.readouterr().out == expected.group(1)
