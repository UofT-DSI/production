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
        for k, stock_info in self.tickers.iterrows():
            self.get_data_and_save_by_year(stock_info['ticker'], 
                                           self.start_date, 
                                           self.end_date, 
                                           self.price_dir, 
                                           sector = stock_info['GICS Sector'], 
                                           subsector = stock_info['GICS Sub-Industry'])

    @staticmethod
    def get_data_and_save_by_year(ticker, start_date, end_date, outpath, sector = None, subsector = None):
        '''
        Gets individual data for a ticker, partitions by year and saves. (Wrapper)
        '''

        _logs.info(f'Processing ticker {ticker}')
        ticker_dt = DataManager.get_stock_price_data(ticker, start_date, end_date)
        _logs.debug(f'ticker_dt columns {ticker_dt.columns}')
        DataManager.save_by_year(ticker, outpath, ticker_dt, sector, subsector)

    @staticmethod
    def save_by_year(ticker, outpath, ticker_dt, sector = None, subsector = None):
        '''
        Partition by year and save.
        '''
        _logs.info(f'Saving data for {ticker} by year') 
        if ticker_dt.shape[0] > 0:
            os.makedirs(os.path.join(outpath, ticker), exist_ok = True)
            ticker_dt = ticker_dt.reset_index()
            _logs.debug(f'ticker_dt.columns {ticker_dt.columns}')
            ticker_dt = (ticker_dt
                        .assign(ticker = ticker,
                                sector = sector,
                                subsector = subsector)
                        .assign(year = ticker_dt['Date'].dt.year))
            
            for yr in ticker_dt.year.unique():
                yr_dd = (dd
                         .from_pandas(ticker_dt[ticker_dt.year == yr], npartitions=1)
                         .set_index('ticker'))
                yr_path = os.path.join(outpath, ticker, f"{ticker}_{yr}.parquet")
                yr_dd.to_parquet(yr_path, overwrite = True)
        else:
            _logs.warning(f'No data found for {ticker}')

    @staticmethod
    def get_stock_price_data(ticker, start_date, end_date):
        '''
        Download stock prices for a given ticker and date range.
        '''

        _logs.info(f'Getting stock price data for {ticker} from {start_date} to {end_date}')
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        
        if isinstance(stock_data.columns, pd.core.indexes.multi.MultiIndex):
            stock_data.columns = stock_data.columns.get_level_values('Price')
        return stock_data


    def featurize(self):
        '''
        Create futures and target data.
        '''
        _logs.info(f'Creating features data.')
        self.load_prices()
        self.create_features()
        self.create_target()
        self.save_features()


    def load_prices(self):
        '''
        Give a set of parquet files, load them into a dask dataframe.
        '''
        _logs.info(f'Loading price data from {self.price_dir}')
        parquet_files = glob(os.path.join(self.price_dir, "*/*/*.parquet"))
        self.price_dd = dd.read_parquet(parquet_files).set_index("ticker")

    def create_features(self):
        '''
        Create features from price data.
        '''
        _logs.info(f'Creating features')
        price_dd = self.price_dd
        features = (price_dd
                   .groupby('ticker', group_keys=False)
                   .apply(
                        lambda x: x.assign(Close_lag_1 = x['Close'].shift(1))
                    ).assign(
                        returns = lambda x: x['Close']/x['Close_lag_1'] - 1,
                    ).assign(
                        positive_return = lambda x: (x['returns'] > 0)*1
                    ))
        self.features = features

    def create_target(self, target_name = 'positive_return', target_window = 1):
        '''
        Create target variable.
        '''

        _logs.info(f'Creating target')
        self.features = (self.features.groupby('ticker', group_keys=False).apply(
                        lambda x: x.sort_values('Date').assign(
                            target = lambda x: x[target_name].shift(-target_window)
                        )))

    def save_features(self):
        '''
        Save to parquet.
        '''
        _logs.info(f'Saving features to {self.features_path}')
        feat_out = (self
         .features
         .repartition(npartitions = 500))
        feat_out.to_parquet(
            self.features_path, 
            write_index = True, 
            overwrite = True)
