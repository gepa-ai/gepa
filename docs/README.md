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

The script auto-generates documentation for all items in `API_MAPPING`, including:
- **Core**: optimize, GEPAAdapter, EvaluationBatch, GEPAResult, GEPACallback, DataLoader, GEPAState, EvaluationCache
- **Callbacks**: All event types (OptimizationStartEvent, IterationEndEvent, etc.) and CompositeCallback
- **Stop Conditions**: All stopper classes
- **Adapters**: DefaultAdapter, DSPyAdapter, RAGAdapter, MCPAdapter, etc.
- **Proposers**: CandidateProposal, ReflectiveMutationProposer, MergeProposer, etc.
- **Logging**: LoggerProtocol, StdOutLogger, Logger, ExperimentTracker, create_experiment_tracker
- **Strategies**: BatchSampler, CandidateSelector, ComponentSelector, EvaluationPolicy variants

### Validating API Documentation

To ensure all API items can be imported:

```bash
uv run python scripts/generate_api_docs.py --validate
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
│   │   ├── core/           # Core classes and functions
│   │   ├── callbacks/      # Callback system and events
│   │   ├── stop_conditions/# Stop condition classes
│   │   ├── adapters/       # Adapter implementations
│   │   ├── proposers/      # Proposer classes
│   │   ├── logging/        # Logging utilities
│   │   └── strategies/     # Strategy classes
│   ├── guides/             # User guides
│   │   ├── quickstart.md   # Getting started guide
│   │   ├── adapters.md     # Creating custom adapters
│   │   ├── callbacks.md    # Using the callback system
│   │   └── contributing.md # Contributing guide
│   └── tutorials/          # Tutorial notebooks
├── scripts/
│   └── generate_api_docs.py # API doc generator (automated)
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

The API documentation is auto-generated from `scripts/generate_api_docs.py`:

1. Add the new module/class to the `API_MAPPING` dictionary in `scripts/generate_api_docs.py`
2. Run `uv run python scripts/generate_api_docs.py` to regenerate all API docs
3. Add the new page to the `nav` section in `mkdocs.yml`

**Note**: The `API_MAPPING` in `generate_api_docs.py` is the source of truth for API documentation. 
The script auto-generates both the markdown files and the index content.

## Automation Features

The `generate_api_docs.py` script provides several automation features:

| Command | Description |
|---------|-------------|
| `python scripts/generate_api_docs.py` | Generate all API documentation |
| `python scripts/generate_api_docs.py --validate` | Validate all API imports work |
| `python scripts/generate_api_docs.py --print-nav` | Print nav structure for mkdocs.yml |

## Deployment

Documentation is automatically built and deployed to GitHub Pages on push to main via GitHub Actions.

### Troubleshooting

**Build fails with import errors:**
- Ensure all GEPA dependencies are installed
- Check that `src/gepa` is importable
- Run `--validate` to check specific imports

**Pages not updating:**
- Check the Actions tab for failed deployments
- Verify GitHub Pages is set to "GitHub Actions" source

**Local build works but CI fails:**
- CI installs from `pyproject.toml`, not editable mode
- Ensure all imports work without editable install

**API docs out of sync with nav:**
- Regenerate API docs: `python scripts/generate_api_docs.py`
- Validate the mapping: `python scripts/generate_api_docs.py --validate`
