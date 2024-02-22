import pandas as pd
from sacred import Ingredient
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
from logger import get_logger

load_dotenv()
DB_URL = os.getenv('DB_URL')
_logs = get_logger(__name__)

db_ingredient = Ingredient('db_ingredient')

db_ingredient.logger = _logs

@db_ingredient.config
def cfg():
    db_url = DB_URL

@db_ingredient.capture
def df_to_sql(df, table_name, db_url=DB_URL, schema = None, if_exists = 'append'):
    '''Convenience function to interface with db.'''
    _logs.info(f'Writing {df.shape} to {table_name} with {db_url}')
    engine = create_engine(db_url)
    with engine.connect() as con:
        df.to_sql(table_name, 
                  con = con, 
                  if_exists = if_exists, 
                  schema = schema, 
                  index = False)

