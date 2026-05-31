"""
Ingest and featurize stock price data for the DSI production project.

Reads four environment variables (all required at runtime, no defaults):
  PRICE_DATA      — root directory for partitioned parquet output
  PRICE_CSV_DATA  — root directory containing raw CSV files
  FEATURES_DATA   — output path for the featurized parquet dataset
  TICKERS         — path to the tickers file (reserved for future use)
"""

import argparse
import os
import random
from glob import glob

import dask.dataframe as dd
import pandas as pd
from dotenv import load_dotenv

from utils.logger import get_logger

load_dotenv()

_logs = get_logger(__name__)

PRICE_DATA = os.getenv('PRICE_DATA')
PRICE_CSV_DATA = os.getenv('PRICE_CSV_DATA')
FEATURES_DATA = os.getenv('FEATURES_DATA')
TICKERS = os.getenv('TICKERS')


class DataManager:
    """Download, partition, and featurize stock price data.

    The typical workflow has two phases:

    **Ingest** — read raw CSVs, partition by ticker/year, write parquet:
        ``process_all_files()``  or  ``process_sample_files()``

    **Featurize** — load partitioned parquet, compute lagged returns, write
        a single feature dataset:
        ``featurize()``

    Parameters
    ----------
    csv_dir : str | None
        Directory tree containing raw CSV files (one file per ticker).
        Defaults to the ``PRICE_CSV_DATA`` env var.
    price_dir : str | None
        Root directory for partitioned parquet output.
        Defaults to the ``PRICE_DATA`` env var.
    features_path : str | None
        Output path for the featurized parquet dataset.
        Defaults to the ``FEATURES_DATA`` env var.
    tickers_file : str | None
        Path to the tickers file (reserved for future use).
        Defaults to the ``TICKERS`` env var.
    n_sample : int
        Number of tickers to sample during ``process_sample_files()``.
        If larger than the available file count the full set is used.
    random_state : int | None
        Seed for the random sampler; ``None`` produces non-deterministic results.
    """

    def __init__(
        self,
        csv_dir: str | None = PRICE_CSV_DATA,
        price_dir: str | None = PRICE_DATA,
        features_path: str | None = FEATURES_DATA,
        tickers_file: str | None = TICKERS,
        n_sample: int = 30,
        random_state: int | None = None,
    ) -> None:
        self.price_csv_dir = csv_dir
        self.price_dir = price_dir
        self.price_dd: dd.DataFrame | None = None
        self.features: dd.DataFrame | None = None
        self.features_path = features_path
        self.tickers_file = tickers_file
        self.n_sample = n_sample
        self.random_state = random_state

    def process_all_files(self) -> None:
        """Ingest every CSV found under ``price_csv_dir``.

        Calls ``get_file_list()`` then ``get_data_and_save_by_year()``.
        """
        _logs.info('Processing all tickers')
        self.get_file_list()
        self.get_data_and_save_by_year()

    def process_sample_files(
        self,
        n_sample: int | None = None,
        random_state: int | None = None,
    ) -> None:
        """Ingest a random sample of CSVs found under ``price_csv_dir``.

        Parameters
        ----------
        n_sample : int | None
            Override the instance ``n_sample`` for this call.
        random_state : int | None
            Override the instance ``random_state`` for this call.
        """
        if n_sample is not None:
            self.n_sample = n_sample
        if random_state is not None:
            self.random_state = random_state
        _logs.info('Processing sample of tickers')
        self.get_file_list()
        self.select_sample()
        self.get_data_and_save_by_year()

    def get_file_list(self) -> None:
        """Populate ``self.file_list`` from all CSVs under ``price_csv_dir``."""
        _logs.info(f'Getting file list from {self.price_csv_dir}')
        self.file_list = glob(os.path.join(self.price_csv_dir, "**/*.csv"), recursive=True)
        _logs.info(f'Found {len(self.file_list)} files in {self.price_csv_dir}')

    def select_sample(self) -> None:
        """Down-sample ``self.file_list`` to ``self.n_sample`` entries in place.

        If ``n_sample`` exceeds the available file count the full list is kept.
        """
        _logs.info('Selecting sample of files')
        if self.n_sample is None or self.n_sample > len(self.file_list):
            self.n_sample = len(self.file_list)
        else:
            random.seed(self.random_state)
            self.file_list = random.sample(self.file_list, self.n_sample)
        _logs.info(f'Selected {len(self.file_list)} files')

    def get_data_and_save_by_year(self) -> None:
        """Read each file in ``self.file_list`` and delegate to ``save_by_year``."""
        for s_file in self.file_list:
            _logs.info(f'Processing file {s_file}')
            ticker_dt = DataManager.get_stock_price_data(s_file)
            _logs.debug(f'Columns in ticker data {ticker_dt.columns}')
            DataManager.save_by_year(ticker_dt, self.price_dir)

    @staticmethod
    def save_by_year(price_dt: pd.DataFrame, out_dir: str) -> None:
        """Partition ``price_dt`` by ticker and year, writing one parquet dataset each.

        Output layout: ``<out_dir>/<TICKER>/<TICKER>_<YEAR>/``

        Parameters
        ----------
        price_dt : pd.DataFrame
            DataFrame with at minimum ``ticker`` (str) and ``Date`` (datetime) columns.
        out_dir : str
            Root directory under which ticker/year subdirectories are created.
        """
        _logs.info('Saving data by year')
        for ticker in price_dt['ticker'].unique():
            _logs.info(f'Processing ticker: {ticker}')
            ticker_dt = price_dt[price_dt['ticker'] == ticker]
            ticker_dt = ticker_dt.assign(Year=ticker_dt.Date.dt.year)
            for yr in ticker_dt['Year'].unique():
                _logs.info(f'Processing year {yr} for ticker {ticker}.')
                yr_dd = dd.from_pandas(ticker_dt[ticker_dt['Year'] == yr], 2)
                yr_path = os.path.join(out_dir, ticker, f"{ticker}_{yr}")
                os.makedirs(os.path.dirname(yr_path), exist_ok=True)
                _logs.info(f'Writing data to path: {yr_path}')
                yr_dd.to_parquet(yr_path, engine="pyarrow")

    @staticmethod
    def get_stock_price_data(stock_file: str) -> pd.DataFrame:
        """Read a single CSV and return a DataFrame with ticker metadata attached.

        Adds two columns derived from the filename:
        - ``source`` — bare filename (e.g. ``AAPL.csv``)
        - ``ticker`` — filename without extension (e.g. ``AAPL``)

        Parameters
        ----------
        stock_file : str
            Absolute or relative path to the CSV file.

        Returns
        -------
        pd.DataFrame
            Raw price data with ``Date`` parsed as datetime and ticker columns added.
        """
        _logs.info(f"Reading file: {stock_file}")
        price_dt = pd.read_csv(stock_file).assign(
            source=os.path.basename(stock_file),
            ticker=os.path.basename(stock_file).replace('.csv', ''),
            Date=lambda x: pd.to_datetime(x['Date'])
        )
        return price_dt

    def featurize(self) -> None:
        """Load partitioned parquet, compute lagged returns, and save features.

        Calls ``load_prices()``, ``create_features()``, and ``save_features()``
        in sequence.
        """
        _logs.info('Creating features data.')
        self.load_prices()
        self.create_features()
        self.save_features()

    def load_prices(self) -> None:
        """Read all parquet files under ``price_dir`` into ``self.price_dd``."""
        _logs.info(f'Loading price data from {self.price_dir}')
        parquet_files = glob(os.path.join(self.price_dir, "**/*.parquet"), recursive=True)
        self.price_dd = dd.read_parquet(parquet_files).set_index("ticker")

    def create_features(self) -> None:
        """Compute ``Close_lag_1`` and ``Returns`` and store in ``self.features``.

        ``Returns`` is defined as ``Close / Close_lag_1 - 1`` (simple daily return).
        Sorting within each ticker group ensures the lag is applied chronologically.
        """
        _logs.debug(f'Columns in price data {self.price_dd.columns}')
        price_dd = self.price_dd
        features = (
            price_dd
                .groupby('ticker', group_keys=False)
                .apply(
                    lambda x: x.sort_values('Date', ascending=True)
                            .assign(Close_lag_1=x['Close'].shift(1)),
                    meta=pd.DataFrame(data={
                            'Date': 'datetime64[ns]',
                            'Open': 'f8',
                            'High': 'f8',
                            'Low': 'f8',
                            'Close': 'f8',
                            'Adj Close': 'f8',
                            'Volume': 'i8',
                            'source': 'object',
                            'Year': 'int32',
                            'Close_lag_1': 'f8'},
                        index=pd.Index([], dtype=pd.StringDtype(), name='ticker'))
                ))
        self.features = features.assign(
            Returns=lambda x: x['Close'] / x['Close_lag_1'] - 1
        )

    def save_features(self) -> None:
        """Write ``self.features`` to parquet at ``self.features_path``."""
        _logs.info(f'Saving features to {self.features_path}')
        _logs.debug(f'Features columns {self.features.columns}')
        self.features.to_parquet(
            self.features_path,
            overwrite=True,
            write_index=True,
            schema={
                'Date': 'timestamp[ns]',
                'Open': 'float64',
                'High': 'float64',
                'Low': 'float64',
                'Close': 'float64',
                'Adj Close': 'float64',
                'Volume': 'int64',
                'source': 'string',
                'Year': 'int32',
                'Close_lag_1': 'float64',
                'Returns': 'float64',
                'ticker': 'large_string'
            })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download and process stock price data.')
    parser.add_argument('--action', default='all', help='Action to perform. Options: all, ingest, featurize')
    parser.add_argument('--n_sample', type=int, default=None, help='Number of tickers to sample. If None, all tickers are used.')
    parser.add_argument('--random_state', type=int, default=None, help='Random state for sampling tickers.')
    args = parser.parse_args()
    _logs.info(f'Starting from command line with args {args}')

    dm = DataManager(n_sample=args.n_sample, random_state=args.random_state)
    if args.action in ('all', 'ingest'):
        _logs.info('Downloading data.')
        if args.n_sample is not None:
            dm.process_sample_files(n_sample=args.n_sample, random_state=args.random_state)
        else:
            dm.process_all_files()
    if args.action in ('all', 'featurize'):
        _logs.info('Featurizing data.')
        dm.featurize()
