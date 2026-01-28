# GEPA Documentation

This directory contains the MkDocs documentation for GEPA.

## Building the Documentation

### Prerequisites

Install the documentation dependencies using uv:

```bash
cd docs
uv pip install -r requirements.txt
```

### Generate API Documentation

Before building, generate the API reference pages:

```bash
uv run python scripts/generate_api_docs.py
```

### Local Development

To serve the documentation locally with live reloading:

```bash
uv run mkdocs serve
```

Then visit http://localhost:8000

### Building for Production

To build the static site:

```bash
uv run mkdocs build
```

The output will be in the `site/` directory.

## Structure

```
docs/
├── docs/                    # Documentation source files
│   ├── index.md            # Home page
│   ├── api/                # Auto-generated API reference
│   ├── guides/             # User guides
│   └── tutorials/          # Tutorial notebooks
├── scripts/
│   └── generate_api_docs.py # API doc generator
├── mkdocs.yml              # MkDocs configuration
└── requirements.txt        # Python dependencies
```

## Adding Content

### Adding a New Guide

1. Create a new `.md` file in `docs/guides/`
2. Add it to the `nav` section in `mkdocs.yml`

### Adding a Tutorial Notebook

1. Copy the `.ipynb` file to `docs/tutorials/`
2. Add it to the `nav` section in `mkdocs.yml`

### Adding API Documentation

1. Add the new module/class to `scripts/generate_api_docs.py`
2. Run `python scripts/generate_api_docs.py`
3. Add the new page to the `nav` section in `mkdocs.yml`

## Deployment

Documentation is automatically built and deployed on push to main via GitHub Actions.
