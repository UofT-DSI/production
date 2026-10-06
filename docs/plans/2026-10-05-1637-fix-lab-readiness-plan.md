---
title: Lab Readiness - Plan
type: fix
date: 2026-10-05
artifact_contract: ce-unified-plan/v1
product_contract_source: ce-plan-bootstrap
execution: code
---

# Lab Readiness - Plan

## Goal Capsule

**Objective:** Students can run the labs in sequence and trust the examples and instructions they use to learn from them.

**Means:** Correct the seven reviewed issues in the existing notebooks, Docker notes, and path helper (KTD1, KTD2).

**Authority:** The course objective of human readable, simple code governs implementation choices. The current branch and its data pipeline define the expected paths and column names.

**Execution:** Update the teaching materials locally; verify the changed lessons with focused checks and notebook runs where the course datasets and services are available.

## Product Contract

### Summary

Repair the broken distribution shift lesson and the misleading model comparison, then align the Docker and setup explanations with the repository's actual behavior.

### Problem Frame

The course is about to be taught again. Two examples can fail against this repository's current data and setup, and one model comparison can silently produce misleading results. Several explanations also leave students with the wrong operational expectation.

### Requirements

**Runnable lessons**

- R1. The distribution shift notebook loads the same environment and return column that the preceding data engineering lesson produces.
- R2. The two pipeline configurations remain distinct through fitting and cross-validation.
- R3. Cross-validation used for model selection excludes the held-out test set; the test set is used only for final evaluation.
- R4. The explainability notebook makes clear whether the loaded fitted model is reused or deliberately retrained.

**Accurate setup instructions**

- R5. Docker instructions describe what `docker compose down -v` actually removes in this stack.
- R6. Docker Desktop installation instructions allow both Apple Silicon and Intel Mac users to choose the right download.
- R7. Notebook setup states that the kernel's working directory must be `01_materials/labs` for relative paths to work.

### Scope Boundaries

The work covers the seven review findings in `01_materials/labs`. Do not reorganize the lessons, optimize their examples for speed, or refactor production modules outside the lab materials. Keep the code direct enough for students to follow.

## Planning Contract

### Key Technical Decisions

- KTD1. **Use existing data conventions.** Load the labs `.env` and use `Returns`, matching `02_data_engineering.ipynb` and `05_src/stock_prices/data_manager.py`; do not add a compatibility rename for an obsolete saved output.
- KTD2. **Keep model selection visible.** Make two independent pipeline objects and show their parameter values before comparing results. Run cross-validation on the non-test portion, then show one final held-out evaluation of the selected candidate.
- KTD3. **Reuse the persisted fitted model in explainability.** The preceding hyperparameter notebook saves `best_estimator_`; remove the unexplained fit call unless execution reveals a specific reason to retrain, in which case explain that reason next to the call.
- KTD4. **Document the working-directory prerequisite.** Keep `update_path.py` short and state the directory assumption near the first notebook setup instructions. Do not add path-discovery machinery when the other notebook data paths also assume that directory.

## Implementation Units

### U1. Repair the distribution shift lesson

- **Goal:** Make the notebook run against features produced by the current data engineering lesson.
- **Requirements:** R1.
- **Dependencies:** None.
- **Files:** `01_materials/labs/07_distribution_shifts.ipynb`.
- **Approach:** Use the same environment loading pattern as the other labs. Read the current return column consistently in summaries and KS comparisons. Clear stale cell outputs that show the obsolete lowercase column, then rerun the lesson.
- **Patterns to follow:** `01_materials/labs/02_data_engineering.ipynb` and `05_src/stock_prices/data_manager.py`.
- **Test scenarios:** With fresh features from the data engineering lesson and a kernel started in `01_materials/labs`, run every cell from top to bottom; all five yearly summaries and four KS comparisons must complete without missing-path or missing-column errors.
- **Verification:** Confirm the notebook outputs reflect the current `Returns` column and no saved output masks an old run.

### U2. Correct model comparison and explainability

- **Goal:** Ensure the pipeline lesson compares the intended models without test leakage, and the explainability lesson handles its saved model plainly.
- **Requirements:** R2, R3, R4.
- **Dependencies:** None.
- **Files:** `01_materials/labs/03b_pipeline.ipynb`, `01_materials/labs/06_explainability.ipynb`.
- **Approach:** Follow KTD2 and KTD3. Keep the two parameter settings and data split easy to inspect. Replace the open-ended refit comment with a short explanation of the chosen behavior.
- **Patterns to follow:** `01_materials/labs/05_hyperparams.ipynb` keeps its held-out test set outside `GridSearchCV` and saves a fitted best estimator.
- **Test scenarios:** Run the pipeline notebook sequentially and confirm the two compared estimators retain different `C` values. Confirm model-selection calls receive no held-out test rows, then evaluate the selected estimator once on that set. Load the saved artifact in the explainability notebook and produce the plots without an unexplained refit.
- **Verification:** Inspect the displayed parameters and metrics as well as successful cell execution; a successful run alone would not detect the aliasing or leakage.

### U3. Correct Docker guidance

- **Goal:** Make setup and cleanup instructions match the Compose stack.
- **Requirements:** R5, R6.
- **Dependencies:** None.
- **Files:** `01_materials/labs/04_0_docker.md`.
- **Approach:** Explain that `down` removes containers and `-v` removes named volumes, while the stack's host folders remain. Replace the Mac-only ARM instruction with architecture-specific selection guidance. Keep this in plain language and link to the existing stack README if more cleanup detail is useful.
- **Patterns to follow:** `05_src/experiment_tracking/docker-compose.yml` and `05_src/experiment_tracking/readme.md`.
- **Test scenarios:** Read the instructions beside the Compose mounts and confirm that a student who wants to stop services, retain data, or clear data can tell which action has which effect. Confirm both Mac architectures have a download choice.
- **Verification:** Manual documentation check; no Docker operation is needed to prove the wording change.

### U4. State the notebook launch directory

- **Goal:** Make the path helper's assumption clear at the point students first use it.
- **Requirements:** R7.
- **Dependencies:** None.
- **Files:** `01_materials/labs/01_setup.ipynb`, `01_materials/labs/update_path.py` (inspect; change only if the comment or name needs clarification).
- **Approach:** State that notebooks should be launched with `01_materials/labs` as the kernel working directory because `update_path.py` and data paths resolve relative to it. Preserve the helper's simple implementation.
- **Test scenarios:** From a fresh kernel in `01_materials/labs`, run the setup cells and confirm `utils.logger` imports. Confirm the written instruction identifies the directory needed when launch defaults differ.
- **Verification:** The first notebook's instruction and helper behavior agree.

## Verification Contract

Run a static syntax pass over all eight notebook code cells after removing IPython magics. Check local notebook links and inspect saved outputs for stale values. Where course datasets are available, execute U1 and U2 notebooks from a fresh kernel in their documented working directory. Review the final diff to ensure it contains only the teaching-material changes above.

## Definition of Done

- All seven review findings are addressed in the named files.
- Sequential runs demonstrate the corrected data and model behavior where inputs are available; unavailable datasets or services are recorded as verification limits.
- The examples remain readable without extra abstractions, and no abandoned trial code or stale outputs remain.
