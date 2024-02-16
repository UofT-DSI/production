import dask.dataframe as dd
import pandas as pd
import numpy as np
import yfinance as yf
from dotenv import load_dotenv
from datetime import datetime
from glob import glob
import os

from logger import get_logger

load_dotenv()

_logs = get_logger(__name__)

logger = get_logger(__name__)
PRICE_DATA = os.getenv('PRICE_DATA')
FEATURES_DATA = os.getenv('FEATURES_DATA')
TICKERS = os.getenv('TICKERS')

class DataManager():
    def __init__(self, 
                 start_date = "2000-01-01",
                 end_date = datetime.now().strftime("%Y-%m-%d"),
                 price_dir = PRICE_DATA, 
                 features_path = FEATURES_DATA, 
                 tickers_file = TICKERS):
        self.price_dir = price_dir
        self.price_dd = None
        self.features = None
        self.features_path = features_path
        self.tickers_file = tickers_file
        self.start_date = start_date
        self.end_date = end_date

    def download_all(self):
        _logs.info(f'Getting price data for all tickers.')
        self.get_tickers()
        self.process_all_tickers()


    def get_tickers(self):
        _logs.info(f'Getting tickers from {self.tickers_file}')
        tickers = pd.read_csv(self.tickers_file)
        self.tickers = tickers['ticker'].unique()


    def process_all_tickers(self):
        _logs.info(f'Processing all tickers')
        for ticker in self.tickers:
            self.get_data_and_save_by_year(ticker, self.start_date, self.end_date, self.price_dir)

    @staticmethod
    def get_data_and_save_by_year(ticker, start_date, end_date, outpath):
        _logs.info(f'Processing ticker {ticker}')
        ticker_dt = DataManager.get_stock_price_data(ticker, start_date, end_date)
        DataManager.save_by_year(ticker, outpath, ticker_dt)

    @staticmethod
    def save_by_year(ticker, outpath, ticker_dt):
        _logs.info(f'Saving data for {ticker} by year') 
        if ticker_dt.shape[0] > 0:
            os.makedirs(os.path.join(outpath, ticker), exist_ok = True)
            ticker_dt = ticker_dt.reset_index()
            _logs.debug(f'ticker_dt.columns {ticker_dt.columns}')
            ticker_dt = (ticker_dt
                        .assign(ticker = ticker)
                        .assign(year = ticker_dt['Date'].dt.year))
            for yr in ticker_dt.year.unique():
                yr_dt = ticker_dt[ticker_dt.year == yr]
                yr_path = os.path.join(outpath, ticker, f"{ticker}_{yr}.parquet")
                yr_dt.to_parquet(yr_path, engine = "pyarrow")
        else:
            _logs.warning(f'No data found for {ticker}')

    @staticmethod
    def get_stock_price_data(ticker, start_date, end_date):
        _logs.info(f'Getting stock price data for {ticker} from {start_date} to {end_date}')
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        return stock_data


    def featurize(self):
        _logs.info(f'Creating features data.')
        self.load_prices()
        self.create_features()
        self.save_features()


    def load_prices(self):
        _logs.info(f'Loading price data from {self.price_dir}')
        parquet_files = glob(os.path.join(self.price_dir, "**/*.parquet"))
        self.price_dd = dd.read_parquet(parquet_files).set_index("ticker")

    def create_features(self):
        _logs.info(f'Creating features')
        price_dd = self.price_dd
        features = (price_dd
                   .groupby('ticker', group_keys=False)
                   .apply(
                        lambda x: x.assign(Close_lag_1 = x['Close'].shift(1))
                    ).assign(
                        log_returns = lambda x: np.log(x['Close']/x['Close_lag_1']), 
                        returns = lambda x: x['Close']/x['Close_lag_1'] - 1
                    ).assign(
                        positive_return = lambda x: (x['returns'] > 0)*1
                    ).set_index("ticker"))
        self.features = features

    def save_features(self):
        _logs.info(f'Saving features to {self.features_path}')
        self.features.to_parquet(
            self.features_path, 
            write_index = True, 
            overwrite = True, 
            partition_on='ticker')
