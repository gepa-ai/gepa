#!/usr/bin/env python3
"""Classify a GEPA release tag and compute the post-release dev version.

The publish workflow accepts two canonical PEP 440 forms and nothing else:

- stable: ``X.Y.Z`` (for example ``0.1.5``)
- prerelease: ``X.Y.ZrcN``, ``X.Y.ZaN``, or ``X.Y.ZbN`` (for example ``0.1.5rc1``)

Dev versions (``X.Y.Z.devN``), posts, locals, epochs, and non-canonical
spellings such as ``0.1.5.rc1`` or ``0.1.5alpha1`` are ``invalid``. Invalid
tags must not be published.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# No leading zeros. Three numeric components, then nothing (stable) or a
# canonical pre-release segment. ``rc`` is matched before the single-letter
# forms so it is not confused with a truncated token.
_STABLE = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
_PRERELEASE = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(rc|a|b)(0|[1-9][0-9]*)$")


def classify_release(version: str) -> str:
    """Return ``stable``, ``prerelease``, or ``invalid``."""
    if _STABLE.fullmatch(version):
        return "stable"
    if _PRERELEASE.fullmatch(version):
        return "prerelease"
    return "invalid"


def next_dev_version(stable_version: str) -> str:
    """Return the next patch ``.dev0`` after a stable ``X.Y.Z`` release.

    ``0.1.5`` becomes ``0.1.6.dev0``. Refuses every non-stable version so a
    pre-release or dev tag cannot advance ``main``.
    """
    if classify_release(stable_version) != "stable":
        raise ValueError(f"{stable_version} is not a stable X.Y.Z version")
    major, minor, micro = stable_version.split(".")
    return f"{major}.{minor}.{int(micro) + 1}.dev0"


def require_marked_version(path: str, version: str) -> None:
    """Require ``version="..."`` on the line after ``#replace_package_version_marker``.

    The publish workflow edits that line with sed and depends on no spaces
    around ``=``.
    """
    expected = f'version="{version}"'
    lines = Path(path).read_text().splitlines()
    for index, line in enumerate(lines):
        if line.strip() == "#replace_package_version_marker":
            if index + 1 >= len(lines) or lines[index + 1] != expected:
                found = lines[index + 1] if index + 1 < len(lines) else "<eof>"
                raise ValueError(f"expected {expected!r} on the line after the version marker, found {found!r}")
            return
    raise ValueError(f"missing #replace_package_version_marker in {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Classify GEPA release tags.")
    sub = parser.add_subparsers(dest="command", required=True)

    classify = sub.add_parser("classify", help="Print stable, prerelease, or invalid")
    classify.add_argument("version")

    nxt = sub.add_parser("next-dev", help="Print the next patch .dev0 after a stable version")
    nxt.add_argument("version")

    check = sub.add_parser("check-marker", help="Check the pyproject version marker line")
    check.add_argument("path")
    check.add_argument("version")

    args = parser.parse_args(argv)
    try:
        if args.command == "classify":
            print(classify_release(args.version))
        elif args.command == "next-dev":
            print(next_dev_version(args.version))
        else:
            require_marked_version(args.path, args.version)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
