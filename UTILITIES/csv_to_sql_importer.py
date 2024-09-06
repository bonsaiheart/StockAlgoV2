import json
import os
from logging.handlers import RotatingFileHandler, QueueHandler, QueueListener
from queue import Queue

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from datetime import datetime, date
import pytz
import logging
from database_operations import create_schema_and_tables
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from db_schema_models import Symbol, Option, OptionQuote, SymbolQuote
from PrivateData import sql_db
from csv_db_mappings import symbol_quote_mapping
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from collections import defaultdict

DATABASE_URI = sql_db.DATABASE_URI
engine = create_engine(DATABASE_URI, echo=False, pool_size=50, max_overflow=100)
from pathlib import Path

PROGRESS_DIR = Path("progress")
# logging.basicConfig(filename='import_log.txt', level=logging.ERROR,
#                     format='%(asctime)s - %(levelname)s - %(message)s')

from logging.handlers import RotatingFileHandler
import multiprocessing
import os


os
import logging
from logging.handlers import RotatingFileHandler
import multiprocessing


import logging
from logging.handlers import RotatingFileHandler, QueueHandler, QueueListener
from queue import Queue

def setup_logging():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(script_dir, 'import_log.txt')
    max_log_size = 10 * 1024 * 1024  # 10 MB
    backup_count = 5

    # Create the RotatingFileHandler
    os.makedirs(os.path.dirname(log_file), exist_ok=True)  # Ensure log directory exists
    handler = RotatingFileHandler(log_file, 'a', maxBytes=max_log_size, backupCount=backup_count, encoding='utf-8', delay=0)
    formatter = logging.Formatter('%(asctime)s - %(processName)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    # Create the queue and QueueHandler
    log_queue = Queue()
    queue_handler = QueueHandler(log_queue)

    # Create the logger and add the QueueHandler
    logger = logging.getLogger("multiprocessing_logger")
    logger.setLevel(logging.INFO)
    logger.addHandler(queue_handler)

    # Create and start the QueueListener
    listener = QueueListener(log_queue, handler)
    listener.start()

    return logger, listener
# Global variable for logger
# logger = setup_logging()


def save_ticker_progress(ticker, progress, data_type):
    progress_file = PROGRESS_DIR / f"{ticker}_{data_type}_progress.json"
    with open(progress_file, 'w') as f:
        json.dump(progress, f)

SCHEMA_NAME = 'csvimport'

def create_schema_if_not_exists(engine, schema_name):
    with engine.connect() as connection:
        connection.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema_name}"))
        connection.commit()

def get_session():
    Session = sessionmaker(bind=engine)
    return Session(bind=engine.connect().execution_options(
        schema_translate_map={None: SCHEMA_NAME}
    ))
def parse_timestamp_from_filename(filename,logger):
    try:
        parts = filename.split('_')
        date_str, time_str = parts[1], parts[2].split('.')[0]
        year = int(date_str[:2]) + 2000
        month, day = int(date_str[2:4]), int(date_str[4:6])
        hour, minute = int(time_str[:2]), int(time_str[2:4])
        naive_timestamp = datetime(year, month, day, hour, minute)
        return pytz.timezone('US/Eastern').localize(naive_timestamp)
    except (IndexError, ValueError) as e:
        logger.error(f"Error parsing timestamp from filename {filename}: {str(e)}")
        return None

# note 23 5-30 was last spy  roku 230725
def convert_unix_to_datetime(timestamp,logger):
    if timestamp is None or pd.isna(timestamp) or timestamp == 0:
        return None
    try:
        # If it's already a datetime object, return it
        if isinstance(timestamp, datetime):
            return timestamp.replace(tzinfo=pytz.timezone('US/Eastern'))

        # If it's a datetime string, parse it
        if isinstance(timestamp, str):
            try:
                return pytz.timezone('US/Eastern').localize(datetime.fromisoformat(timestamp.replace('+00:00', '')))
            except ValueError:
                pass  # If it's not a valid datetime string, continue to treat it as a unix timestamp

        # If it's a Unix timestamp
        timestamp = int(float(timestamp))
        if len(str(timestamp)) == 13:
            return datetime.fromtimestamp(timestamp / 1000, tz=pytz.timezone('US/Eastern'))
        elif len(str(timestamp)) == 10:
            return datetime.fromtimestamp(timestamp, tz=pytz.timezone('US/Eastern'))
        else:
            logger.warning(f"Unexpected timestamp format: {timestamp}")
            return None
    except ValueError as e:
        logger.warning(f"Invalid timestamp: {timestamp}. Error: {str(e)}")
        return None

def parse_date(date_value):
    if isinstance(date_value, int):
        try:
            year = 2000 + (date_value // 10000)
            month = (date_value // 100) % 100
            day = date_value % 100
            return date(year, month, day)
        except ValueError:
            # logging.error(f"Unable to parse date from integer: {date_value}")
            return None
    elif isinstance(date_value, str):
        for format in ['%Y-%m-%d', '%m/%d/%Y']:
            try:
                return datetime.strptime(date_value, format).date()
            except ValueError:
                pass
        # logging.error(f"Unable to parse date string: {date_value}")
        return None
    else:
        # logging.error(f"Unexpected date format: {date_value}")
        return None

def ensure_symbol_exists(session, ticker,logger):
    symbol = session.query(Symbol).filter_by(symbol_name=ticker).first()
    if not symbol:
        symbol = Symbol(symbol_name=ticker)
        session.add(symbol)
        try:
            session.commit()
            logger.info(f"Added new symbol: {ticker}")
        except IntegrityError:
            session.rollback()
            logger.warning(f"Symbol {ticker} already exists")
    return symbol
def ensure_progress_dir():
    PROGRESS_DIR.mkdir(parents=True, exist_ok=True)

def load_ticker_progress(ticker, data_type,logger):
    ensure_progress_dir()  # Ensure the directory exists
    progress_file = PROGRESS_DIR / f"{ticker}_{data_type}_progress.json"
    if progress_file.exists():
        with open(progress_file, 'r') as f:
            return json.load(f)
    else:
        logger.info(f"{ticker} had no progress file for {data_type}")
        return []
def get_value(row, prefix, field):
    call_prefix = 'Call_' if prefix == 'c_' else 'c_'
    put_prefix = 'Put_' if prefix == 'p_' else 'p_'
    if prefix in ['c_', 'Call_']:
        return row.get(f'{prefix}{field}') or row.get(f'{call_prefix}{field}')
    else:
        return row.get(f'{prefix}{field}') or row.get(f'{put_prefix}{field}')

def parse_greeks(greeks_str,logger):
    if not greeks_str or pd.isna(greeks_str):
        return None
    try:
        return json.loads(greeks_str.replace("'", '"'))
    except json.JSONDecodeError:
        try:
            return {'implied_volatility': float(greeks_str)}
        except ValueError:
            logger.warning(f"Failed to parse Greeks: {greeks_str}")
            return None

def process_option_csv(csv_path, ticker,logger):
    df = pd.read_csv(csv_path)
    filename = os.path.basename(csv_path)
    fetch_timestamp = parse_timestamp_from_filename(filename,logger)
    if fetch_timestamp is None:
        return None, None

    options_to_insert = []
    option_quotes_to_insert = []

    for _, row in df.iterrows():
        common_data = {
            'underlying': ticker,
            'expiration_date': parse_date(row.get('ExpDate')),
            'strike': row.get('Strike'),
        }

        for option_type, prefix, suffix in [('call', 'c_', '_y'), ('put', 'p_', '_x')]:
            contract_id = row.get(f'{prefix}contractSymbol')
            if contract_id:
                option_data = {
                    **common_data,
                    'option_type': option_type,
                    'contract_id': contract_id,
                    'contract_size': row.get(f'contract_size{suffix}'),
                    'expiration_type': row.get(f'expiration_type{suffix}'),
                }
                options_to_insert.append(option_data)

                quote_data = {
                    'contract_id': contract_id,
                    'fetch_timestamp': fetch_timestamp,
                    'last': get_value(row, prefix, 'LastPrice'),
                    'change': get_value(row, prefix, 'change'),
                    'volume': get_value(row, prefix, 'Volume'),
                    'open_interest': get_value(row, prefix, 'OI'),
                    'bid': get_value(row, prefix, 'bid'),
                    'ask': get_value(row, prefix, 'ask'),
                    'change_percentage': get_value(row, prefix, 'PercentChange'),
                    'trade_date': convert_unix_to_datetime(get_value(row, prefix, 'lastTrade'),logger),
                    'greeks': parse_greeks(row.get(f'{prefix}greeks'),logger),
                    'open': row.get(f'open{suffix}'),
                    'high': row.get(f'high{suffix}'),
                    'low': row.get(f'low{suffix}'),
                    # 'close': row.get(f'close{suffix}'),
                    'last_volume': row.get(f'last_volume{suffix}'),
                    'prevclose': row.get(f'prevclose{suffix}'),
                    'bidsize': row.get(f'bidsize{suffix}'),
                    'bid_date': convert_unix_to_datetime(row.get(f'bid_date{suffix}'),logger),
                    'asksize': row.get(f'asksize{suffix}'),
                    'ask_date': convert_unix_to_datetime(row.get(f'ask_date{suffix}'),logger),
                }
                option_quotes_to_insert.append(quote_data)

    return options_to_insert, option_quotes_to_insert


from sqlalchemy.dialects.postgresql import insert
import math
from sqlalchemy.exc import IntegrityError
from collections import defaultdict


def import_option_data(session, csv_path, ticker,logger):
    batch_size = 4000  # Adjust this value based on your system's capabilities
    options_to_insert, option_quotes_to_insert = process_option_csv(csv_path, ticker,logger)

    try:
        # Batch insert options
        for i in range(0, len(options_to_insert), batch_size):
            batch = options_to_insert[i:i + batch_size]
            stmt = insert(Option).values(batch)
            stmt = stmt.on_conflict_do_nothing(index_elements=['contract_id'])
            session.execute(stmt)

        session.commit()
    except Exception as e:
        session.rollback()
        print(f"Error importing option data for {ticker} from {csv_path}: {str(e)}")
    try:
        # Clean option quotes
        cleaned_quotes = []
        for quote in option_quotes_to_insert:
            cleaned_quote = clean_option_quote(quote, ticker, csv_path,logger)
            if cleaned_quote is not None:
                cleaned_quotes.append(cleaned_quote)

        # Batch insert option quotes
        for i in range(0, len(cleaned_quotes), batch_size):
            batch = cleaned_quotes[i:i + batch_size]
            stmt = insert(OptionQuote).values(batch)
            stmt = stmt.on_conflict_do_nothing(index_elements=['contract_id', 'fetch_timestamp'])
            session.execute(stmt)

        session.commit()

    except Exception as e:
        session.rollback()
        print(f"Error importing optionquote data for {ticker} from {csv_path}: {str(e)}")
    finally:
        session.close()
def is_nan(value):
    if isinstance(value, (int, float)):
        return math.isnan(value)
    if isinstance(value, str):
        try:
            return math.isnan(float(value))
        except ValueError:
            return True  # Consider non-numeric strings as NaN
    return True  # Consider None and other types as NaN
def clean_option_quote(quote, ticker, file_path,logger):
    try:
        for key, value in quote.items():
            if isinstance(value, (float, np.float64)) and (math.isnan(value) or np.isnan(value)):
                quote[key] = None
            elif value == 'null':
                quote[key] = None
            elif isinstance(value, str) and value.lower() == 'nan':
                quote[key] = None

        # Ensure volume, open_interest, last_volume, bidsize, and asksize are integers or None
        for key in ['volume', 'open_interest', 'last_volume', 'bidsize', 'asksize']:
            if key in quote:
                if quote[key] is None or (isinstance(quote[key], (float, np.float64)) and (math.isnan(quote[key]) or np.isnan(quote[key]))):
                    quote[key] = None
                else:
                    try:
                        quote[key] = int(float(quote[key]))
                    except ValueError:
                        quote[key] = None

        # Convert timestamp fields to datetime objects if they're not None
        for key in ['trade_date', 'bid_date', 'ask_date']:
            if key in quote and quote[key] is not None:
                quote[key] = convert_unix_to_datetime(quote[key],logger)

        # Ensure greeks is a valid JSON or None
        if 'greeks' in quote:
            if quote['greeks'] == 'null' or quote['greeks'] is None:
                quote['greeks'] = None
            elif isinstance(quote['greeks'], str):
                try:
                    quote['greeks'] = json.loads(quote['greeks'])
                except json.JSONDecodeError:
                    quote['greeks'] = None
            elif not isinstance(quote['greeks'], dict):
                quote['greeks'] = None

        return quote
    except Exception as e:
        logger.error(f"Error cleaning option quote for ticker {ticker} from file {file_path}: {str(e)}", exc_info=True)
        return None

def import_stock_data(session, csv_path, ticker,logger):
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            logger.warning(f"CSV file is empty: {csv_path}")
            return

        filename = os.path.basename(csv_path)
        fetch_timestamp = parse_timestamp_from_filename(filename,logger)
        if fetch_timestamp is None:
            logger.error(f"Failed to parse timestamp from filename: {filename}")
            return

        # Process only the first row
        first_row = df.iloc[0]
        symbol_quote_data = {}
        for db_field, csv_field in symbol_quote_mapping.items():
            if isinstance(csv_field, list):
                for field in csv_field:
                    if field in first_row:
                        value = first_row[field]
                        break
                else:
                    continue
            else:
                if csv_field not in first_row:
                    continue
                value = first_row[csv_field]

            # Convert numpy types to Python types
            if pd.isna(value):
                value = None
            elif isinstance(value, (pd.Timestamp, np.datetime64)):
                value = value.to_pydatetime()
            elif isinstance(value, (np.int64, np.int32, np.int16, np.int8)):
                value = int(value)
            elif isinstance(value, (np.float64, np.float32)):
                value = float(value)

            symbol_quote_data[db_field] = value

        symbol_quote_data['symbol_name'] = ticker
        symbol_quote_data['fetch_timestamp'] = fetch_timestamp

        # Convert timestamp fields
        for field in ['last_trade_timestamp','last_1min_timestamp','last_1min_timesale', 'current_bid_date', 'current_ask_date']:
            if field in symbol_quote_data and symbol_quote_data[field] is not None:
                symbol_quote_data[field] = convert_unix_to_datetime(symbol_quote_data[field],logger)

        symbol_quote_data = {k: v for k, v in symbol_quote_data.items() if v is not None}

        # Use insert...on conflict do update
        stmt = insert(SymbolQuote).values(**symbol_quote_data)
        stmt = stmt.on_conflict_do_update(
            index_elements=['symbol_name', 'fetch_timestamp'],
            set_=symbol_quote_data
        )

        try:
            session.execute(stmt)
            session.commit()
            logger.info(f"Successfully imported/updated stock data for {ticker} from {filename}")
        except Exception as e:
            session.rollback()
            logger.error(f"Error inserting/updating quote for {ticker} at {fetch_timestamp}: {str(e)}")

    except Exception as e:
        session.rollback()
        logger.error(f"Error importing stock data for {ticker} from {csv_path}: {str(e)}", exc_info=True)
    finally:
        session.close()


import os
import logging
from sqlalchemy.orm import sessionmaker
import json





def process_ticker_directory(base_path, ticker, logger):
    ensure_progress_dir()
    options_progress = load_ticker_progress(ticker, 'options', logger)
    stocks_progress = load_ticker_progress(ticker, 'stocks', logger)

    Session = sessionmaker(bind=engine)
    session = Session(bind=engine.connect().execution_options(
        schema_translate_map={None: SCHEMA_NAME}
    ))

    try:
        ensure_symbol_exists(session, ticker, logger)
        option_data_path = os.path.join(base_path, "optionchain", ticker)
        stock_data_path = os.path.join(base_path, "ProcessedData", ticker)

        # Process option data
        if os.path.exists(option_data_path):
            date_dirs = [d for d in os.listdir(option_data_path) if os.path.isdir(os.path.join(option_data_path, d))]
            logger.info(f"Found {len(date_dirs)} option date directories for {ticker}")
            for date_dir in date_dirs:
                if date_dir not in options_progress:
                    logger.info(f"Processing option data for {ticker} on {date_dir}")
                    date_path = os.path.join(option_data_path, date_dir)
                    for file in os.listdir(date_path):
                        if file.endswith('.csv'):
                            try:
                                import_option_data(session, os.path.join(date_path, file), ticker, logger)
                            except Exception as e:
                                logger.error(f"Error processing option file {file} for {ticker}: {str(e)}")
                    options_progress.append(date_dir)
                    save_ticker_progress(ticker, options_progress, 'options')
                    logger.info(f"Processed option data for {ticker} on {date_dir}")
                else:
                    logger.info(f"Skipping already processed option data for {ticker} on {date_dir}")

        # Process stock data
        if os.path.exists(stock_data_path):
            date_dirs = [d for d in os.listdir(stock_data_path) if os.path.isdir(os.path.join(stock_data_path, d))]
            logger.info(f"Found {len(date_dirs)} stock date directories for {ticker}")
            for date_dir in date_dirs:
                if date_dir not in stocks_progress:
                    logger.info(f"Processing stock data for {ticker} on {date_dir}")
                    date_path = os.path.join(stock_data_path, date_dir)
                    for file in os.listdir(date_path):
                        if file.endswith('.csv'):
                            try:
                                import_stock_data(session, os.path.join(date_path, file), ticker, logger)
                            except Exception as e:
                                logger.error(f"Error processing stock file {file} for {ticker}: {str(e)}")
                    stocks_progress.append(date_dir)
                    save_ticker_progress(ticker, stocks_progress, 'stocks')
                    logger.info(f"Processed stock data for {ticker} on {date_dir}")
                else:
                    logger.info(f"Skipping already processed stock data for {ticker} on {date_dir}")
        else:
            logger.warning(f"Stock data path does not exist for {ticker}: {stock_data_path}")

    except Exception as e:
        logger.error(f"Error processing directory for ticker {ticker}: {str(e)}", exc_info=True)
    finally:
        session.close()

import time
from concurrent.futures import ProcessPoolExecutor, as_completed


def main():
    lock = multiprocessing.Lock()
    logger, listener = setup_logging()
    try:
        logger.info("Starting data import process")
        create_schema_if_not_exists(engine, SCHEMA_NAME)

        base_path = r"\\BONSAI-SERVER\stockalgo_data\data"
        option_chain_path = os.path.join(base_path, "optionchain")

        tickers = ['SPY','WMT','CHWY','ROKU','TSLA','GOOGL','AMZN','BA']
        num_processes = max(1, multiprocessing.cpu_count() // 2)

        logger.info(f"Starting to process {len(tickers)} tickers using {num_processes} processes")

        start_time = time.time()
        processed_tickers = 0
        processed_date_dirs = 0

        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [executor.submit(process_ticker_directory, base_path, ticker, logger) for ticker in tickers]
            for future in as_completed(futures):
                processed_tickers += 1
                processed_date_dirs = sum(len(load_ticker_progress(ticker, 'options',logger)) +
                                          len(load_ticker_progress(ticker, 'stocks',logger))

                                          for ticker in tickers)
                elapsed_time = time.time() - start_time
                avg_time_per_ticker = elapsed_time / processed_tickers
                estimated_time_remaining = avg_time_per_ticker * (len(tickers) - processed_tickers)

                logger.info(f"Processed {processed_tickers}/{len(tickers)} tickers, "
                            f"{processed_date_dirs} date directories. "
                            f"Estimated time remaining: {estimated_time_remaining:.2f} seconds")

        total_time = time.time() - start_time
        logger.info(f"{datetime.now()}Finished processing all {len(tickers)} tickers, "
                    f"{processed_date_dirs} date directories in {total_time:.2f} seconds")
        logger.info(f"Average time per ticker: {total_time / len(tickers):.2f} seconds")
        logger.info(f"Average time per date directory: {total_time / processed_date_dirs:.2f} seconds")
        logger.info("Data import process completed")
    except Exception as e:
        logger.error(f"An error occurred in the main process: {str(e)}", exc_info=True)
    finally:
        listener.stop()  # Stop the listener when done
if __name__ == "__main__":
    # multiprocessing.freeze_support()  # This helps with multiprocessing on Windows
    # lock = multiprocessing.Lock()
    # logger = setup_logging(lock)
    create_schema_and_tables(engine)
    main()