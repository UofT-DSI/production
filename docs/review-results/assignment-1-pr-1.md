# Assignment 1 review — [PR #1](https://github.com/calderonjesus/production/pull/1)

Reviewed commit: `2212a0927a322f7a162fd15f1d516fe64191cf78`

The notebook loads the environment and parquet files, keeps the feature work in Dask, and calculates the 10-row moving average within each ticker after converting to pandas. Sorting each ticker by `Date` before making lags is a good choice. Three expressions in the feature cell need correction before the resulting values match the assignment.

| ID | Rubric criterion | Score (0–1) | Evidence and feedback |
| --- | --- | ---: | --- |
| A1-01 | Load environment | 1 | Cell 2 calls `load_dotenv()` before reading `PRICE_DATA`. |
| A1-02 | Import Dask dataframe | 1 | Cell 3 imports `dask.dataframe as dd`. |
| A1-03 | Load environment variable | 1 | Cell 5 reads `os.getenv('PRICE_DATA')`. |
| A1-04 | Produce parquet glob | 1 | Cell 5 recursively finds `*.parquet` beneath `PRICE_DATA`. |
| A1-05 | Group by ticker | 1 | Cell 7 uses `dd_px.groupby('ticker', group_keys=False)`. |
| A1-06 | Calculate returns | 0 | Cell 7 sorts by `Date`, but `shift(-1)` takes the **next** close, while a one-day lag needs `shift(1)`. It also computes `Close / Close_lag_1` without subtracting 1. Apply `shift(1)` to both `Close` and `Adj Close`, then use `Close / Close_lag_1 - 1`. For a close sequence of 100, 110, 121, the return on the second date should be `0.10`. |
| A1-07 | Calculate high–low range | 0 | Cell 7 uses `Low - High`, reversing the requested range. Use `High - Low`; a high of 115 and low of 100 should give 15. |
| A1-08 | Complete features in Dask | 1 | Cell 7 assigns the lag, return, and range expressions to `dd_feat` before any `.compute()`. The method meets this criterion, although the expressions above need correction. |
| A1-09 | Convert to pandas | 1 | Cell 9 calls `dd_feat.compute()`. |
| A1-10 | Add 10-day moving average | 1 | Cell 9 uses `.rolling(10).mean()` on the `returns` column. The method is right; recompute it after fixing the return values in cell 7. |
| A1-11 | Group moving average by ticker | 1 | Cell 9 groups by the `ticker` index level before the rolling calculation. |
| A1-12 | Explain pandas versus Dask | 1 | The added markdown answers both questions and explains the memory and shuffle tradeoff. The response notebook referenced by the rubric was unavailable, so this score uses the submitted explanation. |

**Total: 10 / 12 = 0.8333.** The rubric gives each of its 12 criteria equal weight; the notebook's point labels are separate.

For a reference, compare the lag and return calculation in [DataManager.create_features](https://github.com/UofT-DSI/production/blob/main/05_src/stock_prices/data_manager.py) and the **Load** and **Transform** sections of the [session 02 data engineering lab](https://github.com/UofT-DSI/production/blob/main/01_materials/labs/02_data_engineering.ipynb). After editing cell 7, run the notebook with `PRICE_DATA` configured and inspect one ticker's first few dates. Check that the first lag and return are missing, the next return uses the preceding close, the range is nonnegative when `High >= Low`, and the moving average reflects the corrected returns.

**Verification:** I compared the PR diff with the assignment notebook, the spreadsheet rubric, the session 02 lab, and the stock price implementation. An independent three-row pandas calculation confirmed the direction and sign errors. I did not execute the submitted notebook end to end because the required `PRICE_DATA` input was not available in the PR.
