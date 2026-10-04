"""
Pipeline tests for ``DataManager``: real price CSVs → partitioned parquet → features.

Every run writes into ``tmp_path``; the real ``05_src/data`` folders are only read.
"""

import random
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from stock_prices.data_manager import DataManager

pytestmark = pytest.mark.pipeline

FEATURE_DTYPES = {
    'Open': 'float64',
    'High': 'float64',
    'Low': 'float64',
    'Close': 'float64',
    'Adj Close': 'float64',
    'Volume': 'int64',
    'Year': 'int32',
    'Close_lag_1': 'float64',
    'Returns': 'float64',
}


def _sample_tickers(price_csv_dir: Path, n: int, seed: int = 0) -> list[Path]:
    files = sorted(price_csv_dir.glob('*.csv'))
    return random.Random(seed).sample(files, n)


def _expected_features(csv_files: list[Path]) -> pd.DataFrame:
    """Lag and returns computed directly with pandas, per ticker in date order."""
    frames = []
    for csv in csv_files:
        df = pd.read_csv(csv, parse_dates=['Date']).sort_values('Date')
        df['ticker'] = csv.stem
        df['Close_lag_1'] = df['Close'].shift(1)
        df['Returns'] = df['Close'] / df['Close_lag_1'] - 1
        frames.append(df)
    return pd.concat(frames)


def _run_pipeline(csv_dir: Path, out_dir: Path) -> pd.DataFrame:
    """Ingest every CSV in ``csv_dir`` into ``out_dir/prices``, featurize, and read the features back."""
    dm = DataManager(
        csv_dir=str(csv_dir),
        price_dir=str(out_dir / 'prices'),
        features_path=str(out_dir / 'features'),
    )
    dm.process_all_files()
    dm.featurize()
    return pd.read_parquet(out_dir / 'features').reset_index()


def _assert_lags_match(features: pd.DataFrame, expected: pd.DataFrame) -> None:
    got = features.sort_values(['ticker', 'Date']).reset_index(drop=True)
    want = expected.sort_values(['ticker', 'Date']).reset_index(drop=True)
    assert got['ticker'].tolist() == want['ticker'].tolist()
    assert got['Date'].tolist() == want['Date'].tolist()
    np.testing.assert_allclose(got['Close_lag_1'], want['Close_lag_1'], equal_nan=True)
    np.testing.assert_allclose(got['Returns'], want['Returns'], equal_nan=True)


@pytest.fixture(scope='module')
def three_tickers(price_csv_dir, tmp_path_factory) -> list[Path]:
    csv_dir = tmp_path_factory.mktemp('csv')
    for csv in _sample_tickers(price_csv_dir, 3):
        shutil.copy(csv, csv_dir / csv.name)
    return sorted(csv_dir.glob('*.csv'))


@pytest.fixture(scope='module')
def pipeline_run(three_tickers, tmp_path_factory) -> tuple[Path, pd.DataFrame]:
    """One pipeline run over the three sampled tickers: ``(output dir, features)``."""
    out_dir = tmp_path_factory.mktemp('pipeline')
    return out_dir, _run_pipeline(three_tickers[0].parent, out_dir)


@pytest.fixture(scope='module')
def expected_features(three_tickers) -> pd.DataFrame:
    return _expected_features(three_tickers)


def test_ingest_then_featurize_preserves_rows_and_schema(three_tickers, pipeline_run, expected_features):
    _, features = pipeline_run

    assert len(features) == len(expected_features)
    assert set(features['ticker']) == {p.stem for p in three_tickers}
    for column, dtype in FEATURE_DTYPES.items():
        assert str(features[column].dtype) == dtype, column
    assert pd.api.types.is_datetime64_any_dtype(features['Date'])
    assert set(features['source']) == {p.name for p in three_tickers}


def test_partitions_are_written_per_ticker_and_year(three_tickers, pipeline_run):
    out_dir, _ = pipeline_run

    for csv in three_tickers:
        years = pd.DatetimeIndex(pd.read_csv(csv)['Date']).year.unique()
        written = sorted(p.name for p in (out_dir / 'prices' / csv.stem).iterdir())
        assert written == sorted(f'{csv.stem}_{y}' for y in years)


def test_features_match_pandas_reference(pipeline_run, expected_features):
    _, features = pipeline_run
    _assert_lags_match(features, expected_features)


def test_lag_is_chronological_when_input_rows_are_shuffled(three_tickers, expected_features, tmp_path):
    csv_dir = tmp_path / 'shuffled'
    csv_dir.mkdir()
    for csv in three_tickers:
        pd.read_csv(csv).sample(frac=1.0, random_state=0).to_csv(csv_dir / csv.name, index=False)

    features = _run_pipeline(csv_dir, tmp_path)

    _assert_lags_match(features, expected_features)
    first_rows = features.sort_values('Date').groupby('ticker').head(1)
    assert first_rows['Close_lag_1'].isna().all()
