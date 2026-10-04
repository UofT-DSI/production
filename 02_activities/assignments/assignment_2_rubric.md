# Assignment 2 Rubric — Model pipelines on the Forest Fires data

Markdown version of [`assignment_2_rubric_clean.xlsx`](./assignment_2_rubric_clean.xlsx), for agents and people assessing submissions. Criterion text comes from the spreadsheet, with two typos fixed ("fatures", "Grides"). "Look for" notes come from the instructions in [`assignment_2.ipynb`](./assignment_2.ipynb).

## Submission under assessment

- **Artifact:** the student's completed `02_activities/assignments/assignment_2.ipynb`, submitted on a branch named `assignment-2`. The notebook should be the only file changed in the pull request.
- **Task:** predict burned `area` in the UCI Forest Fires dataset by building four model pipelines from two preprocessors and two regressors, tuning them with grid search and cross-validation, pickling the best model, and explaining it with SHAP.
- The notebook says its instructions are minimum requirements. Do not penalise extra steps that improve the work.

## Scoring

- Score each criterion from **0 to 1** (spreadsheet column "Score (0-1)").
- There are **15 criteria**, all equally weighted.
- **Final score = sum of criterion scores / 15**, a value between 0 and 1.
- A criterion with several bullet points is still **one** criterion. Score it as a whole, giving partial credit when some of its bullets are missing.

## Criteria

| ID | Section | Criterion | Look for |
|----|---------|-----------|----------|
| A2-01 | Prep Data | Splits the data into features and target (X and Y). | `X` holds every column except `area`; `Y` is `area`. |
| A2-02 | Prep Data | Creates a training and test set; uses a reasonable split; sets the random state to 42. | `train_test_split` with a sensible `test_size` (for example 0.2–0.3) and `random_state=42`. |
| A2-03 | Column Transformer | Creates at least two different preprocessing steps. Generally these should be column transformers, but a combination of pipelines and transformers can also be submitted. | `preproc1` and `preproc2`, normally `ColumnTransformer`s that differ only in a non-linear transformation of the numeric variables. |
| A2-04 | Column Transformer | The preprocessing pipelines treat month and day as categorical variables. | `month` and `day` are one-hot encoded, not scaled as numbers. |
| A2-05 | Column Transformer | One hot encoding explicitly treats unseen values. | `OneHotEncoder(handle_unknown='ignore')` or an equivalent explicit setting. |
| A2-06 | Column Transformer | A non-linear transformation is applied in at least two pipelines. | `preproc2` applies a non-linear transform such as `PowerTransformer`, a log/`log1p` `FunctionTransformer`, or `QuantileTransformer`, and is used in two of the four pipelines. |
| A2-07 | Model pipeline | Creates four pipelines in total. | A = preproc1 + baseline, B = preproc2 + baseline, C = preproc1 + advanced, D = preproc2 + advanced, each with steps named `preprocessing` and `regressor`. |
| A2-08 | Model pipeline | Two models are linear (Linear Regression, Lasso, Ridge, etc.); another model is more complex (ensembles, MLP, etc.). | Both baseline pipelines use a linear or simple model; both advanced pipelines use a more complex model. The notebook also accepts KNN as a baseline and suggests a tree-based advanced model so SHAP runs quickly. |
| A2-09 | Model pipeline | Pickles pipeline and saves. Ideally uses a context manager (`with` statement). | The best pipeline is saved with `pickle.dump` (or `joblib.dump`), ideally inside `with open(...)`. |
| A2-10 | Cross-validation | Performs cross validation using 5 folds. | `GridSearchCV(..., cv=5)` or an explicit 5-fold CV. |
| A2-11 | Cross-validation | Selects an ERROR metric (as opposed to R2 or variance explained). | `scoring` is RMSE, MAE or max error (for example `neg_root_mean_squared_error`), not R², explained variance or another correlation metric. |
| A2-12 | Cross-validation | Selects model based on optimal metric. Generally uses GridSearchCV and obtains the best params or best pipeline directly. | The best model is chosen by the error metric, e.g. from `best_score_` / `best_estimator_` / `best_params_` compared across the four searches. The notebook's "Which model has the best performance?" is answered. |
| A2-13 | Cross-validation | Grids include diversity with at least four cases per grid. Hyperparameters are reasonable (learning rate, regularization params, etc.). | Each of the four grids tunes at least one hyperparameter over at least four values or combinations, with sensible ranges. |
| A2-14 | SHAP | Applies SHAP to obtain local and global explanations. Displays beeswarm and waterfall plots. | SHAP on the best model: a waterfall plot for one test-set observation (local) and a beeswarm plot over the training set (global). |
| A2-15 | SHAP | Interprets SHAP values and identifies variables with low importance. | A written answer naming the most and least important features, which features to remove and why, and how to test whether removing them helps. |

## Assessment output template

```markdown
## Assignment 2 assessment — <student / PR link>

| ID | Score (0-1) | Evidence (cell or line) | Note |
|----|-------------|-------------------------|------|
| A2-01 | | | |
| ... | | | |
| A2-15 | | | |

**Total:** <sum> / 15 = <final score, 0-1>
**Summary:** <2-3 sentences of feedback for the student>
```
