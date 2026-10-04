# Project Guide: DSI Production

This repo contains the course materials of DSI Production. The course is described in the repo's README.md file. This file describes the standards and requirements of the repo.

---

## Commands

Always use `uv run python` — bare `python` misses the project venv.

```bash
uv run python -m pytest              # run full test suite
uv run python -m pytest -m unit      # one layer: unit | pipeline | experiment
uv run python -m pytest 05_src/tests/credit/test_data.py -v  # single file
uv run ruff check <files>            # lint
uv run pyrefly check <files>         # type check
```

---

## Code Style

- No comments unless the WHY is non-obvious
- No unused imports — ruff F401 is enforced
- Docstrings: resolution-order lists and policy sections are welcome; avoid restating what the code already says

---

## Testing

- The test suite lives in `05_src/tests/`, mirroring `credit/`, `stock_prices/`, and `utils/`. 
- It runs locally against the real course data (gitignored under `05_src/data/`) and a throwaway MLflow store — no Docker, no CI. 
- `05_src/tests/readme.md` is the student-facing guide to setup and layers.

### How it was set up

1. **Scoped in a brainstorm, then planned** (`docs/plans/2026-10-03-1928-feat-src-test-suite-plan.md`). Decisions: real data over fixtures, missing data **fails** (never skips), MLflow isolated in a temp store, experiments run through their real entry points with minimal settings (2 folds, 1–3 trials), bugs the tests expose are fixed in the same PR, tests double as teaching material.
2. **Harness first** (`tests/harness.py`, `tests/conftest.py`, `tests/layer_markers.py`, `[tool.pytest.ini_options]` in `pyproject.toml`). Order is load-bearing: set `LOG_DIR` → import `credit.experiment` (it runs `load_dotenv(override=True)` + `mlflow.set_tracking_uri`) → re-apply `LOG_DIR` and redirect MLflow to SQLite in the temp root → `chdir` into `05_src` in a session fixture, never at import time.
3. **Unit tests per module**, then the stock-price **pipeline** tests, then **experiment** tests for `run_cv` and all four `exp__*` scripts. Each behaviour fix was written test-first and its test was observed failing before the fix.
4. **Simplify pass, multi-reviewer code review** (including a cross-model pass), then PR.

### Rules for new tests

- Exactly one layer marker per test (`unit`, `pipeline`, `experiment`); collection aborts otherwise.
- Assert exact values computed independently (`pytest.approx`, a pandas reference, the best child run's params) — not `> 0` / non-empty.
- Request data through the `credit_csv` / `price_csv_dir` fixtures; write outputs only to `tmp_path`.
- Use `tests/helpers.py`: `unique_name()` for MLflow experiment/model names (the store is shared per session), `experiment_runs()`, `assert_valid_probabilities()`.
- New `mlflow.sklearn.log_model` calls pass `skops_trusted_types=SKOPS_TRUSTED_TYPES`.
- After changing import-time code, run the **full** suite — collection-order bugs only show up there.

`docs/solutions/` — documented solutions to past problems (bugs, best practices, workflow patterns), organized by category with YAML frontmatter (`module`, `tags`, `problem_type`). The testing setup, its gotchas, and how they were diagnosed are in `docs/solutions/best-practices/testing-ml-course-code-against-real-data.md`.

---

## Readme Files

Every subdirectory under `05_src/` has a `readme.md`. The top-level `src/readme.md` describes the overall system architecture and provides an end-to-end worked example. These files are a human-maintained source of truth for each module's design intent and public interface.

**Use them to inform your work.** Read a module's readme before exploring its source. When planning changes, check whether any readme content will become stale.

### What each readme must contain

- Description of the module's purpose and its main classes
- Public method and function signatures with parameter tables
- At least one fully runnable code example per major class

### Readme review is part of every PR

When a PR adds or changes a public class, method, field, or behaviour, the corresponding `readme.md` must be updated in the same PR. Check:

- New classes or functions are documented
- Changed field defaults, types, or constraints are reflected
- Changed validation rules or error messages are reflected
- Examples still run correctly against the updated code
- `src/readme.md` is still accurate if the module's role or main interfaces changed

Flag a missing or stale readme update the same way you flag a missing test.

---

## PR Review Patterns

Review should include but are not limited to:

- Missing test coverage
- Inconsistency between a model validator's definition and the lower-level function it guards
- Duplication of duties
- Smoke tests (`assert value > 0`) where precision tests (`assert value == pytest.approx(expected)`) are possible
- Readme not updated to reflect new or changed public API
