"""
Unit tests for the ``DataManager`` ingest helpers.
"""

import pandas as pd
import pytest
from stock_prices.data_manager import DataManager

pytestmark = pytest.mark.unit

PRICE_HEADER = 'Date,Open,High,Low,Close,Adj Close,Volume\n'


def _write_csv(path, rows: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(PRICE_HEADER + ''.join(f'{r}\n' for r in rows))


def test_get_stock_price_data_adds_ticker_and_source_and_parses_dates(tmp_path):
    csv = tmp_path / 'AAPL.csv'
    _write_csv(csv, ['2020-01-02,1,2,0.5,1.5,1.4,100', '2020-01-03,1.5,2,1,1.8,1.7,200'])

    df = DataManager.get_stock_price_data(str(csv))

    assert df['ticker'].tolist() == ['AAPL', 'AAPL']
    assert df['source'].tolist() == ['AAPL.csv', 'AAPL.csv']
    assert pd.api.types.is_datetime64_any_dtype(df['Date'])
    assert df['Date'].tolist() == [pd.Timestamp('2020-01-02'), pd.Timestamp('2020-01-03')]
    assert df['Close'].tolist() == pytest.approx([1.5, 1.8])


def test_get_file_list_finds_csvs_recursively(tmp_path):
    for name in ['a/X.csv', 'b/c/Y.csv', 'Z.csv']:
        _write_csv(tmp_path / name, [])
    (tmp_path / 'notes.txt').write_text('ignore me')

    dm = DataManager(csv_dir=str(tmp_path))
    dm.get_file_list()

    assert sorted(p.replace('\\', '/').rsplit('/', 1)[1] for p in dm.file_list) == ['X.csv', 'Y.csv', 'Z.csv']


def test_select_sample_is_deterministic_for_fixed_seed():
    files = [f'T{i}.csv' for i in range(20)]
    picks = []
    for _ in range(2):
        dm = DataManager(n_sample=5, random_state=11)
        dm.file_list = list(files)
        dm.select_sample()
        picks.append(dm.file_list)
    assert picks[0] == picks[1]
    assert len(picks[0]) == 5
    assert set(picks[0]) <= set(files)


def test_select_sample_keeps_every_file_when_n_sample_exceeds_count():
    files = ['A.csv', 'B.csv', 'C.csv']
    dm = DataManager(n_sample=10, random_state=0)
    dm.file_list = list(files)
    dm.select_sample()
    assert dm.file_list == files
    assert dm.n_sample == 3


def test_save_by_year_writes_one_dataset_per_ticker_year(tmp_path):
    frame = pd.DataFrame({
        'ticker': ['AAA', 'AAA', 'AAA', 'BBB'],
        'Date': pd.to_datetime(['2019-12-30', '2019-12-31', '2020-01-02', '2020-01-02']),
        'Close': [1.0, 2.0, 3.0, 10.0],
    })

    DataManager.save_by_year(frame, str(tmp_path))

    written = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.glob('*/*') if p.is_dir())
    assert written == ['AAA/AAA_2019', 'AAA/AAA_2020', 'BBB/BBB_2020']
    aaa_2019 = pd.read_parquet(tmp_path / 'AAA' / 'AAA_2019')
    assert aaa_2019['Close'].tolist() == [1.0, 2.0]
    assert set(aaa_2019['Year']) == {2019}
