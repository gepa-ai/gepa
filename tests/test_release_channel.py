"""Release-channel policy: which tags may publish, and where."""

import importlib.util
import subprocess
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / ".github/workflows/build_utils/release_channel.py"
_WORKFLOW = _ROOT / ".github/workflows/build_and_release.yml"


def _load_release_channel():
    spec = importlib.util.spec_from_file_location("gepa_release_channel", _SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_release_channel = _load_release_channel()


def test_stable_patch_versions_are_stable():
    assert _release_channel.classify_release("0.1.5") == "stable"
    assert _release_channel.classify_release("0.1.10") == "stable"
    assert _release_channel.classify_release("1.0.0") == "stable"


def test_canonical_prereleases_are_prerelease():
    for version in ("0.1.5rc1", "0.1.5rc2", "0.1.5a1", "0.1.5b2", "0.1.5rc0", "0.1.5a0"):
        assert _release_channel.classify_release(version) == "prerelease"


def test_dev_post_and_noncanonical_tags_are_invalid():
    for version in (
        "0.1.5.dev0",
        "0.1.5.dev1",
        "0.1.5rc1.dev0",
        "0.1.5.post1",
        "0.1.5rc1.post1",
        "0.1.5.rc1",
        "0.1.5-rc1",
        "0.1.5alpha1",
        "0.1.5beta1",
        "0.1.5preview1",
        "0.1.5c1",
        "0.1",
        "0.1.5.0",
        "01.2.3",
        "v0.1.5",
        "1!0.1.5",
        "0.1.5+local",
        "",
    ):
        assert _release_channel.classify_release(version) == "invalid", version


def test_next_dev_version_is_the_next_patch():
    assert _release_channel.next_dev_version("0.1.5") == "0.1.6.dev0"
    assert _release_channel.next_dev_version("0.1.9") == "0.1.10.dev0"


def test_next_dev_version_refuses_prerelease_and_dev():
    for version in ("0.1.5rc1", "0.1.5.dev0", "0.1.5a1"):
        try:
            _release_channel.next_dev_version(version)
        except ValueError:
            continue
        raise AssertionError(f"next_dev_version accepted {version}")


def test_marked_version_must_follow_the_marker_with_no_spaces(tmp_path: Path):
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('#replace_package_version_marker\nversion="0.1.6.dev0"\n')
    _release_channel.require_marked_version(str(pyproject), "0.1.6.dev0")

    spaced = tmp_path / "spaced.toml"
    spaced.write_text('#replace_package_version_marker\nversion = "0.1.6.dev0"\n')
    try:
        _release_channel.require_marked_version(str(spaced), "0.1.6.dev0")
    except ValueError:
        return
    raise AssertionError("marker check accepted spaces around '='")


def test_cli_classify_and_next_dev():
    stable = subprocess.run(
        [sys.executable, str(_SCRIPT), "classify", "0.1.5"], check=True, capture_output=True, text=True
    )
    assert stable.stdout.strip() == "stable"

    rc = subprocess.run(
        [sys.executable, str(_SCRIPT), "classify", "0.1.5rc1"], check=True, capture_output=True, text=True
    )
    assert rc.stdout.strip() == "prerelease"

    dev = subprocess.run(
        [sys.executable, str(_SCRIPT), "classify", "0.1.5.dev0"], check=True, capture_output=True, text=True
    )
    assert dev.stdout.strip() == "invalid"

    nxt = subprocess.run(
        [sys.executable, str(_SCRIPT), "next-dev", "0.1.5"], check=True, capture_output=True, text=True
    )
    assert nxt.stdout.strip() == "0.1.6.dev0"

    refused = subprocess.run([sys.executable, str(_SCRIPT), "next-dev", "0.1.5rc1"], capture_output=True, text=True)
    assert refused.returncode != 0


def _job_block(name: str) -> str:
    """Return one job's YAML, from its key through the line before the next job."""
    lines = _WORKFLOW.read_text().splitlines(keepends=True)
    start = next(index for index, line in enumerate(lines) if line.startswith(f"  {name}:"))
    block = [lines[start]]
    for line in lines[start + 1 :]:
        if line.startswith("  ") and not line.startswith("   ") and line.rstrip("\n").endswith(":"):
            break
        block.append(line)
    return "".join(block)


def test_workflow_publishes_prereleases_to_testpypi_only():
    workflow = _WORKFLOW.read_text()
    pypi = _job_block("build-and-publish-pypi")
    test_pypi = _job_block("build-and-publish-test-pypi")

    assert "needs.extract-tag.outputs.channel == 'stable'" in pypi
    assert "pypa/gh-action-pypi-publish@release/v1" in pypi
    assert "repository-url: https://test.pypi.org/legacy/" not in pypi
    assert "next-dev" in pypi
    assert "check-marker" in pypi

    assert "needs.extract-tag.outputs.channel == 'stable'" in test_pypi
    assert "needs.extract-tag.outputs.channel == 'prerelease'" in test_pypi
    assert "repository-url: https://test.pypi.org/legacy/" in test_pypi
    assert 'if [ "$RELEASE_CHANNEL" = "prerelease" ]' in test_pypi
    assert "test_version.py" in test_pypi
    # The alpha-bump helper is the stable-only fallback, not the RC path.
    assert 'if [ "$RELEASE_CHANNEL" = "stable" ]' in test_pypi

    assert "release_channel.py classify" in workflow
    assert "is not on origin/main" in workflow
    assert "refs/heads/main" in workflow
