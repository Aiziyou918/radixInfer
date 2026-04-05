# Contributing to radixInfer

Thanks for contributing to `radixInfer`.

## Getting Started

Use Python 3.10+ and work from the repository root.

```bash
conda activate radixInfer
export PYTHONPATH=python
pip install -e .[dev]
```

If you do not use conda, any equivalent virtual environment is fine.

## Development Workflow

1. Create a focused branch for your change.
2. Keep changes scoped to a single feature, bug fix, or documentation update.
3. Run the relevant tests before opening a pull request.
4. Update documentation when user-facing behavior, APIs, CLI flags, or benchmark methodology change.

## Code Style

- Keep changes small and easy to review.
- Prefer explicit, readable code over clever shortcuts.
- Preserve existing module boundaries: `api`, `transport`, `runtime`, `cache`, `engine`, and `models`.
- Avoid mixing unrelated refactors into feature or bug-fix PRs.
- Do not commit local benchmark outputs from `bench/results/` or runtime logs from `bench/logs/`.

## Testing

Run the default suite:

```bash
pytest -q
```

Run a specific module when iterating locally:

```bash
pytest tests/test_api_helpers.py -v
pytest tests/test_prefix_store.py -v
```

If your change affects HTTP behavior, scheduling, cache behavior, or benchmark tooling, include the validation you ran in the pull request description.

## Documentation Expectations

Please update the relevant docs when changing:

- CLI flags or startup behavior
- public HTTP endpoints or request/response shapes
- supported model/runtime behavior
- benchmark methodology or published benchmark charts

Documentation entry points:

- `README.md`
- `README.zh-CN.md`
- `docs/api-guide.md`
- `docs/architecture.md`
- `docs/development.md`
- `bench/README.md`

## Pull Requests

A good pull request should include:

- a clear summary of the problem and the change
- the affected subsystem(s)
- any compatibility or migration notes
- the tests or benchmark commands you ran

If the change is incomplete or intentionally scoped down, state that explicitly.
