# Publishing TurboGEPA to PyPI

This guide explains how to publish TurboGEPA to PyPI so users can install it with `pip install turbo-gepa`.

## Prerequisites

1. **PyPI Account**: Create account at https://pypi.org/account/register/
2. **API Token**: Generate at https://pypi.org/manage/account/token/
3. **Install build tools**:
   ```bash
   pip install build twine
   ```

## Steps to Publish

### 1. Update Version Number

Edit `pyproject.toml`:
```toml
version="0.1.0"  # Increment for each release
```

Follow [Semantic Versioning](https://semver.org/):
- **0.1.0** → Initial release
- **0.1.1** → Bug fixes
- **0.2.0** → New features (backward compatible)
- **1.0.0** → Stable release

### 2. Clean Previous Builds

```bash
rm -rf dist/ build/ *.egg-info/
```

### 3. Build Distribution Packages

```bash
python -m build
```

This creates:
- `dist/turbo_gepa-0.1.0-py3-none-any.whl` (wheel)
- `dist/turbo-gepa-0.1.0.tar.gz` (source)

### 4. Test with TestPyPI (Optional but Recommended)

Upload to test repository first:

```bash
python -m twine upload --repository testpypi dist/*
```

Test installation:
```bash
pip install --index-url https://test.pypi.org/simple/ turbo-gepa
```

### 5. Upload to PyPI

```bash
python -m twine upload dist/*
```

You'll be prompted for:
- **Username**: `__token__`
- **Password**: Your PyPI API token (starts with `pypi-`)

### 6. Verify Installation

```bash
pip install turbo-gepa
python -c "import turbo_gepa; print('Success!')"
```

## Automation with GitHub Actions

Create `.github/workflows/publish.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'

      - name: Install build tools
        run: |
          python -m pip install --upgrade pip
          pip install build twine

      - name: Build package
        run: python -m build

      - name: Publish to PyPI
        env:
          TWINE_USERNAME: __token__
          TWINE_PASSWORD: ${{ secrets.PYPI_API_TOKEN }}
        run: python -m twine upload dist/*
```

**Setup**:
1. Add PyPI API token to GitHub Secrets: Settings → Secrets → New repository secret
2. Name it `PYPI_API_TOKEN`
3. Create a GitHub release to trigger automatic publishing

## Release Checklist

Before each release:

- [ ] Update version in `pyproject.toml`
- [ ] Update `CHANGELOG.md` with changes
- [ ] Run tests: `pytest tests/`
- [ ] Update documentation if needed
- [ ] Create git tag: `git tag v0.1.0`
- [ ] Push tag: `git push origin v0.1.0`
- [ ] Build and upload to PyPI
- [ ] Create GitHub Release with release notes
- [ ] Verify installation: `pip install turbo-gepa`

## Troubleshooting

### "File already exists" Error

PyPI doesn't allow re-uploading the same version. You must:
1. Increment version number
2. Rebuild: `python -m build`
3. Upload new version

### Import Errors After Installation

Check package structure:
```bash
pip show turbo-gepa
```

Ensure `src/turbo_gepa/__init__.py` exists and imports are correct.

### Authentication Failed

- Verify API token is correct
- Use `__token__` as username (not your PyPI username)
- Token must have "Upload packages" permission

## Resources

- [PyPI Packaging Guide](https://packaging.python.org/tutorials/packaging-projects/)
- [Twine Documentation](https://twine.readthedocs.io/)
- [Semantic Versioning](https://semver.org/)
