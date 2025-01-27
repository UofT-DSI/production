import dask.dataframe as dd
import pandas as pd
import numpy as np
import yfinance as yf
from dotenv import load_dotenv
from datetime import datetime
from glob import glob
import os
import argparse
from logger import get_logger

load_dotenv()

_logs = get_logger(__name__)

PRICE_DATA = os.getenv('PRICE_DATA')
FEATURES_DATA = os.getenv('FEATURES_DATA')
TICKERS = os.getenv('TICKERS')

class DataManager():
    '''
    A class to download and process stock price data and create features.
    '''
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
        '''
        Downloads price data for all tickers and saves it to the price_dir.
        '''
        _logs.info(f'Getting price data for all tickers.')
        self.get_tickers()
        self.process_all_tickers()


    def get_tickers(self):
        '''
        Loads ticker from tickers_file.
        '''
        _logs.info(f'Getting tickers from {self.tickers_file}')
        tickers = pd.read_csv(self.tickers_file)
        self.tickers = tickers.drop_duplicates(subset=['ticker'])


    def process_all_tickers(self):
        _logs.info(f'Processing all tickers')
        ticker_list = self.tickers['ticker'].unique().tolist()
        self.get_data_and_save_by_year(ticker_list, 
                                        self.start_date, 
                                        self.end_date, 
                                        self.price_dir)

    @staticmethod
    def get_data_and_save_by_year(tickers, start_date, end_date, outpath, sector = None, subsector = None):
        '''
        Gets individual data for a ticker, partitions by year and saves. (Wrapper)
        '''

        _logs.info(f'Processing ticker {tickers}')
        ticker_dt = DataManager.get_stock_price_data(tickers, start_date, end_date)
        _logs.debug(f'ticker_dt columns {ticker_dt.columns}')
        DataManager.save_by_year(ticker_dt, outpath)

    @staticmethod
    def save_by_year(ticker_dt, outpath):
        '''
        Partition by year and save.
        '''
        _logs.info(f'Saving data by year') 
        ticker_dt = ticker_dt.assign(Year = ticker_dt['Date'].dt.year)
        for ticker in ticker_dt['Ticker'].unique().tolist():
            _logs.info(f'Saving data for {ticker}')
            os.makedirs(os.path.join(outpath, ticker), exist_ok = True)
            t_dt = ticker_dt[ticker_dt['Ticker'] == ticker]
            for yr in t_dt['Year'].unique():
                yr_dd = (dd
                         .from_pandas(t_dt[t_dt['Year'] == yr], npartitions=1)
                         .set_index('Ticker'))
                yr_path = os.path.join(outpath, ticker, f"{ticker}_{yr}")
                yr_dd.to_parquet(yr_path, overwrite = True)


    @staticmethod
    def get_stock_price_data(tickers, start_date, end_date):
        '''
        Download stock prices for a given ticker and date range.
        '''

        _logs.info(f'Getting stock price data for {tickers} from {start_date} to {end_date}')
        yfinance_dt = yf.download(tickers, start=start_date, end=end_date)
        price_dt = (yfinance_dt
           .stack(future_stack=True)
           .reset_index()
           .sort_values(['Ticker', 'Date']))
        
        return price_dt


    def featurize(self):
        '''
        Create futures and target data.
        '''
        _logs.info(f'Creating features data.')
        self.load_prices()
        self.create_features()
        self.save_features()


    def load_prices(self):
        '''
        Give a set of parquet files, load them into a dask dataframe.
        '''
        _logs.info(f'Loading price data from {self.price_dir}')
        parquet_files = glob(os.path.join(self.price_dir, "**/*.parquet"),
                             recursive = True)
        self.price_dd = dd.read_parquet(parquet_files).set_index("Ticker")

    def create_features(self):
        '''
        Create features from price data.
        '''
        _logs.info(f'Creating features')
        _logs.debug(f'Columns in price data {self.price_dd.columns}')
        price_dd = self.price_dd
        features = (price_dd.groupby('Ticker', group_keys=False)
                            .apply(
                                lambda x: x.assign(
                                    Close_lag_1 = x['Close'].shift(1))
                            ))
        self.features = features

    def create_target(self, target_name = 'positive_return', target_window = 1):
        '''
        Create target variable.
        '''

        _logs.info(f'Creating target')
        self.features = (self.features.groupby('Ticker', group_keys=False).apply(
                        lambda x: x.sort_values('Date').assign(
                            target = lambda x: x[target_name].shift(-target_window)
                        )))

    def save_features(self):
        '''
        Save to parquet.
        '''
        _logs.info(f'Saving features to {self.features_path}')
        _logs.debug(f'Features columns {self.features.columns}')
        self.features.to_parquet(
            self.features_path, 
            write_index = True, 
            overwrite = True)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download and process stock price data.')
    parser.add_argument('--start_date', type=str, default="2000-01-01", help='Start date for downloading data.')
    parser.add_argument('--end_date', type=str, default=datetime.now().strftime("%Y-%m-%d"), help='End date for downloading data.')
    parser.add_argument('--action', default='all', help='Action to perform. Options: all, download, featurize')
    args = parser.parse_args()
    _logs.info(f'Starting from command line with args {args}')
    
    dm = DataManager(start_date = args.start_date, end_date = args.end_date)
    if args.action in ('all', 'download'):
        _logs.info('Downloading data.')
        dm.download_all()
    if args.action in ('all', 'featurize'):
        _logs.info('Featurizing data.')
        dm.featurize()
