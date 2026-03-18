# Repository Guidelines

## Project Structure & Module Organization
`neuralforecast/` contains the Python package: core orchestration lives in `core.py`, auto-tuning helpers in `auto.py`, shared utilities in `common/`, losses in `losses/`, and model implementations in `models/`. Tests mirror the package layout under `tests/` (`tests/test_models/`, `tests/test_common/`, etc.). Documentation sources live in `docs/` and `nbs/`, while reproducible research and benchmarks live in `experiments/`. Small maintenance scripts are kept in `scripts/`.

## Build, Test, and Development Commands
- `make devenv` — install all extras with `uv` and set up pre-commit hooks.
- `uv sync --group dev --torch-backend auto` — create/update the recommended dev environment.
- `uv run pytest` — run the full test suite with coverage.
- `uv run pytest tests/test_models/test_nhits.py` — run a focused test file while iterating.
- `pre-commit run --all-files` — run Ruff, mypy, and repository hooks before pushing.
- `make all_docs` — regenerate API and notebook-based docs.
- `make preview_docs` — preview Mintlify docs locally.

## Coding Style & Naming Conventions
Target Python 3.10+ and follow existing package style: 4-space indentation, snake_case for functions/modules, PascalCase for classes, and short imperative method names. Keep lines within 88 characters to match Ruff/Black defaults. Lint with Ruff and type-check with mypy via pre-commit. Preserve surrounding formatting in edited files; avoid unrelated whitespace-only churn. Use Google-style docstrings for public APIs so docs can be regenerated from source.

## Testing Guidelines
Pytest is the test runner, and the repository enforces at least 80% coverage (`--cov-fail-under=80`). Add regression tests with every behavior change. Place tests next to the affected area using `test_*.py` naming, for example `tests/test_losses/test_pytorch.py` or `tests/test_models/test_patchtst.py`. Prefer targeted runs during development, then finish with `uv run pytest`.

## Commit & Pull Request Guidelines
Recent history favors short, imperative commit subjects, often prefixed with `[FIX]`, `[FEAT]`, or `[CHORE]` (for example, `[FIX] Deprecate the losses.numpy module`). Keep each PR focused, link the relevant issue, explain the problem and solution, and include tests for functional changes. Do not mix style-only edits with behavior changes. If you change docs, notebooks, or public APIs, update the generated docs and mention that in the PR.
