import dask.dataframe as dd
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime
from glob import glob
import random
import os
import argparse
from utils.logger import get_logger

load_dotenv()

_logs = get_logger(__name__)

PRICE_DATA = os.getenv('PRICE_DATA')
PRICE_CSV_DATA = os.getenv('PRICE_CSV_DATA')
FEATURES_DATA = os.getenv('FEATURES_DATA')
TICKERS = os.getenv('TICKERS')

class DataManager():
    '''
    A class to download and process stock price data and create features.
    '''
    def __init__(self, 
                 csv_dir = PRICE_CSV_DATA,
                 price_dir = PRICE_DATA, 
                 features_path = FEATURES_DATA, 
                 tickers_file = TICKERS, 
                 n_sample = 30,
                 random_state = None):
        self.price_csv_dir = csv_dir
        self.price_dir = price_dir
        self.price_dd = None
        self.features = None
        self.features_path = features_path
        self.tickers_file = tickers_file
        self.n_sample = n_sample
        self.random_state = random_state


    def process_all_files(self):
        '''
        Process all files in the price directory.
        '''
        _logs.info(f'Processing all tickers')
        self.get_file_list()
        self.get_data_and_save_by_year()

    def process_sample_files(self, n_sample = None, random_state = None):
        '''
        Process a sample of files.
        '''
        _logs.info(f'Processing sample of tickers')
        self.get_file_list()
        self.select_sample()
        self.get_data_and_save_by_year()
    
    
    
    def get_file_list(self):
        '''
        Get a list of all files in the price directory.
        '''
        _logs.info(f'Getting file list from {self.price_csv_dir}')
        self.file_list = glob(os.path.join(self.price_csv_dir, "**/*.csv"), recursive=True)
        _logs.info(f'Found {len(self.file_list)} files in {self.price_dir}')

 

    def select_sample(self):
        '''
        Select a sample of files from the file list.
        '''
        _logs.info(f'Selecting sample of files')
        if self.n_sample is None or self.n_sample > len(self.file_list):
            self.n_sample = len(self.file_list)
        else:
            random.seed(self.random_state)
            self.file_list = random.sample(self.file_list, self.n_sample)
        _logs.info(f'Selected {len(self.file_list)} files')


    
    
    def get_data_and_save_by_year(self):
        '''
        Gets data for a tickers, partitions by year and saves. (Wrapper)
        '''
        for s_file in self.file_list:
            _logs.info(f'Processing file {s_file}')
            ticker_dt = DataManager.get_stock_price_data(s_file)
            _logs.debug(f'Columns in ticker data {ticker_dt.columns}')
            DataManager.save_by_year(ticker_dt, self.price_dir)


    @staticmethod
    def save_by_year(price_dt, out_dir):
        '''
        Partition by ticker/year and save parquet.
        '''
        _logs.info(f'Saving data by year') 
        for ticker in price_dt['ticker'].unique():
            _logs.info(f'Processing ticker: {ticker}')
            ticker_dt = price_dt[price_dt['ticker'] == ticker]
            ticker_dt = ticker_dt.assign(Year = ticker_dt.Date.dt.year)
            for yr in ticker_dt['Year'].unique():
                _logs.info(f'Processing year {yr} for ticker {ticker}.')
                yr_dd = dd.from_pandas(ticker_dt[ticker_dt['Year'] == yr],2)
                yr_path = os.path.join(out_dir, ticker, f"{ticker}_{yr}")
                os.makedirs(os.path.dirname(yr_path), exist_ok=True)
                _logs.info(f'Writing data to path: {yr_path}')
                yr_dd.to_parquet(yr_path, engine = "pyarrow")


    @staticmethod
    def get_stock_price_data(stock_file):
        '''
        Download stock prices for a given ticker and date range.
        '''
        
        _logs.info(f"Reading file: {stock_file}")
        price_dt = pd.read_csv(stock_file).assign(
            source = os.path.basename(stock_file),
            ticker = os.path.basename(stock_file).replace('.csv', ''),
            Date = lambda x: pd.to_datetime(x['Date'])
        )
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
        self.price_dd = dd.read_parquet(parquet_files).set_index("ticker")

    def create_features(self):
        '''
        Create features from price data.
        '''
        _logs.info(f'Creating features')
        _logs.debug(f'Columns in price data {self.price_dd.columns}')
        price_dd = self.price_dd
        features = (
            price_dd
                .groupby('ticker', group_keys=False)
                .apply(
                    lambda x: x.sort_values('Date', ascending = True)
                            .assign(Close_lag_1 = x['Close'].shift(1)), 
                    meta = pd.DataFrame(data ={'Date': 'datetime64[ns]',
                            'Open': 'f8',
                            'High': 'f8',
                            'Low': 'f8',
                            'Close': 'f8',
                            'Adj Close': 'f8',
                            'Volume': 'i8',
                            'source': 'object',
                            'Year': 'int32',
                            'Close_lag_1': 'f8'},
                            index = pd.Index([], dtype=pd.StringDtype(), name='ticker'))
            ))
        dd_returns = features.assign(
            Returns = lambda x: x['Close']/x['Close_lag_1'] - 1
        )
        self.features = dd_returns


    def save_features(self):
        '''
        Save to parquet.
        '''
        _logs.info(f'Saving features to {self.features_path}')
        _logs.debug(f'Features columns {self.features.columns}')
        self.features.to_parquet(FEATURES_DATA, 
                   overwrite = True,
                   write_index = True,
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
