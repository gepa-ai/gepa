# Releasing GEPA

GEPA stays on the **0.1.x** line for now. `main` is always the next development version, a release candidate is a tag you can try on TestPyPI, and only a stable `vX.Y.Z` tag is published to PyPI.

Publishing is entirely tag-triggered by [`.github/workflows/build_and_release.yml`](.github/workflows/build_and_release.yml). Nothing in this repo cuts a PyPI release on a schedule. GEPA does not keep a changelog.

## Channels

| Channel | Where it lives | Version | Publishes to |
| --- | --- | --- | --- |
| Dev | `main`, between stables | `0.1.5.dev0`, then `0.1.6.dev0`, … | Nowhere |
| RC | A pre-release tag on `main`, such as `v0.1.5rc1` | The tag, exactly | TestPyPI only |
| Stable | A `vX.Y.Z` tag on `main`, such as `v0.1.5` | The tag, exactly | TestPyPI, then PyPI |

Between stables, `main` carries a PEP 440 `.dev0` version. After the 0.1.4 release, that version is `0.1.5.dev0`. Dev versions are not released: a `vX.Y.Z.devN` tag is rejected.

The version in `pyproject.toml` sits on the line immediately after `#replace_package_version_marker`, written as `version="..."` with no spaces around `=`. The publish workflow finds it by that marker. Please leave both in place.

## Cutting a release candidate

From an up-to-date `main` (the tagged commit has to already be on `main`):

```bash
git checkout main
git pull
git tag v0.1.5rc1
git push origin v0.1.5rc1
```

The tag is a canonical PEP 440 pre-release: `vX.Y.ZrcN`, `vX.Y.ZaN`, or `vX.Y.ZbN`. The workflow runs CI, publishes that exact version to TestPyPI, and stops. It does not upload to PyPI, and it does not change the version on `main`.

If that version is already on TestPyPI, the job fails. Tag the next candidate (`v0.1.5rc2`) rather than republishing.

## Cutting a stable release

Stable releases are meant to happen about monthly, and to be skipped when the month was quiet. A person decides and pushes the tag. There is no scheduled workflow that publishes to PyPI.

```bash
git checkout main
git pull
git tag v0.1.5
git push origin v0.1.5
```

A stable tag is exactly `vX.Y.Z`: three numeric components, no `rc`, `a`, `b`, or `dev`. It must point at a commit that is already on `main`.

The workflow then:

1. Runs CI.
2. Rehearses on TestPyPI, then publishes the **exact** tag version to PyPI. If that version already exists on PyPI, the PyPI job fails.
3. Merges a `release-X.Y.Z` branch that carries the frozen version into `main`, bumps `main` to the next patch `.dev0` (`0.1.5` becomes `0.1.6.dev0`, and `uv.lock` is refreshed), and pushes those commits together.

If TestPyPI already has that exact stable version, the rehearsal falls back to [`.github/workflows/build_utils/test_version.py`](.github/workflows/build_utils/test_version.py), which selects a newer pre-release version for TestPyPI only. PyPI still receives the exact tag version.

The dev bump is committed locally and only then pushed with the release merge. If that step fails after PyPI has already accepted the package, `main` is left as it was and `release-X.Y.Z` holds the frozen version. Set `main` to the next `.dev0` by hand before further work, keeping the version marker and refreshing `uv.lock`.

## What does not publish

- An ordinary push to `main`, including the `.dev0` version.
- A tag that is not a stable `vX.Y.Z` or a `vX.Y.ZrcN` / `vX.Y.ZaN` / `vX.Y.ZbN` pre-release. That includes dev tags, post-releases, and non-canonical spellings such as `v0.1.5.rc1`.
- A pre-release tag. The PyPI job is skipped entirely.
- A tag that was not pushed from `main`, or whose commit is not already on `main`.
