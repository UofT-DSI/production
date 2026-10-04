# Assignment 1 Rubric — Working with parquet files

Markdown version of [`assignment_1_rubric_clean.xlsx`](./assignment_1_rubric_clean.xlsx), for agents and people assessing submissions. Criterion text is copied from the spreadsheet. "Look for" notes come from the instructions in [`assignment_1.ipynb`](./assignment_1.ipynb).

## Submission under assessment

- **Artifact:** the student's completed `02_activities/assignments/assignment_1.ipynb`, submitted on a branch named `assignment-1`. The notebook should be the only file changed in the pull request.
- **Task:** load stock price parquet files (from the `PRICE_DATA` environment variable) with Dask, compute per-ticker features, convert to pandas, add a 10-day moving average of returns, and comment on the Dask vs pandas choice.

## Scoring

- Score each criterion from **0 to 1** (spreadsheet column "Score (0-1)").
- There are **12 criteria**, all equally weighted.
- **Final score = sum of criterion scores / 12**, a value between 0 and 1.
- The notebook shows point values (11 pts total). Grading uses the rubric's equal weights, not those point values.

## Criteria

| ID | Section | Criterion | Look for |
|----|---------|-----------|----------|
| A1-01 | Prerequisites | Student loads environment | `load_dotenv()` from `dotenv` (or an equivalent call) before the environment variables are used. |
| A1-02 | Load data | Imports dask.dataframe (other imports are optional) | `import dask.dataframe as dd` or an equivalent import. |
| A1-03 | Load data | Loads environment variable | Reads `PRICE_DATA`, for example `os.getenv('PRICE_DATA')`. |
| A1-04 | Load data | Produces glob | Uses `glob` to list all parquet files under `PRICE_DATA` (typically recursive, e.g. `**/*.parquet`). |
| A1-05 | Calculate features | Uses groupby('ticker') | Features are computed per ticker with `groupby('ticker')`. |
| A1-06 | Calculate features | Calculates returns | `returns = Close / Close_lag_1 - 1`. Lags of `Close` and `Adj_Close` are created first (the notebook asks for both). |
| A1-07 | Calculate features | Calculates hi-low range | `hi_lo_range = High - Low`. |
| A1-08 | Calculate features | Performs all of the tasks in dask and not pandas. | Lags, returns and range are computed on a Dask dataframe and assigned to `dd_feat`, with no `compute()` or conversion to pandas before this step. |
| A1-09 | Moving average | Converts to pandas df using compute() | The Dask dataframe is converted with `.compute()`. |
| A1-10 | Moving average | Adds moving average | A 10-day moving average of `returns`, e.g. `.rolling(10).mean()`, is added as a new column. |
| A1-11 | Moving average | Uses groupby('ticker'), not necessary to use group_keys. | The moving average is computed within each ticker using `groupby('ticker')`. Do not deduct for leaving out `group_keys`. |
| A1-12 | Comments | Assess comments (guide provided in response notebook). | A written answer to both questions: was converting to pandas necessary for the moving average, and would Dask have been better, and why. |

### Note for A1-12

The rubric points to a grading guide in the "response notebook". That notebook is not in this repository. Score this criterion with that guide if you have it. Without it, award credit for a reasoned answer to both questions, and state in the assessment that no reference guide was used.

## Assessment output template

```markdown
## Assignment 1 assessment — <student / PR link>

| ID | Score (0-1) | Evidence (cell or line) | Note |
|----|-------------|-------------------------|------|
| A1-01 | | | |
| ... | | | |
| A1-12 | | | |

**Total:** <sum> / 12 = <final score, 0-1>
**Summary:** <2-3 sentences of feedback for the student>
```
