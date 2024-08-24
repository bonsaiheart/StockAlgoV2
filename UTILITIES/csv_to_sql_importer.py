import json
import os

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

logging.basicConfig(filename='import_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

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
def parse_timestamp_from_filename(filename):
    try:
        parts = filename.split('_')
        date_str, time_str = parts[1], parts[2].split('.')[0]
        year = int(date_str[:2]) + 2000
        month, day = int(date_str[2:4]), int(date_str[4:6])
        hour, minute = int(time_str[:2]), int(time_str[2:4])
        naive_timestamp = datetime(year, month, day, hour, minute)
        return pytz.timezone('US/Eastern').localize(naive_timestamp)
    except (IndexError, ValueError) as e:
        logging.error(f"Error parsing timestamp from filename {filename}: {str(e)}")
        return None


def convert_unix_to_datetime(timestamp):
    if timestamp is None or pd.isna(timestamp):
        return None
    try:
        # If it's already a datetime string, parse it
        if isinstance(timestamp, str):
            return pytz.timezone('US/Eastern').localize(datetime.fromisoformat(timestamp.replace('+00:00', '')))

        # If it's a Unix timestamp
        timestamp = int(float(timestamp))
        if len(str(timestamp)) == 13:
            return datetime.fromtimestamp(timestamp / 1000, tz=pytz.timezone('US/Eastern'))
        elif len(str(timestamp)) == 10:
            return datetime.fromtimestamp(timestamp, tz=pytz.timezone('US/Eastern'))
        else:
            logging.error(f"Unexpected timestamp format: {timestamp}")
            return None
    except ValueError as e:
        logging.error(f"Invalid timestamp: {timestamp}. Error: {str(e)}")
        return None

def parse_date(date_value):
    if isinstance(date_value, int):
        try:
            year = 2000 + (date_value // 10000)
            month = (date_value // 100) % 100
            day = date_value % 100
            return date(year, month, day)
        except ValueError:
            logging.error(f"Unable to parse date from integer: {date_value}")
            return None
    elif isinstance(date_value, str):
        for format in ['%Y-%m-%d', '%m/%d/%Y']:
            try:
                return datetime.strptime(date_value, format).date()
            except ValueError:
                pass
        logging.error(f"Unable to parse date string: {date_value}")
        return None
    else:
        logging.error(f"Unexpected date format: {date_value}")
        return None

def ensure_symbol_exists(session, ticker):
    symbol = session.query(Symbol).filter_by(symbol_name=ticker).first()
    if not symbol:
        symbol = Symbol(symbol_name=ticker)
        session.add(symbol)
        try:
            session.commit()
            logging.info(f"Added new symbol: {ticker}")
        except IntegrityError:
            session.rollback()
            logging.warning(f"Symbol {ticker} already exists")
    return symbol

def get_value(row, prefix, field):
    call_prefix = 'Call_' if prefix == 'c_' else 'c_'
    put_prefix = 'Put_' if prefix == 'p_' else 'p_'
    if prefix in ['c_', 'Call_']:
        return row.get(f'{prefix}{field}') or row.get(f'{call_prefix}{field}')
    else:
        return row.get(f'{prefix}{field}') or row.get(f'{put_prefix}{field}')

def parse_greeks(greeks_str):
    if not greeks_str or pd.isna(greeks_str):
        return None
    try:
        return json.loads(greeks_str.replace("'", '"'))
    except json.JSONDecodeError:
        try:
            return {'implied_volatility': float(greeks_str)}
        except ValueError:
            logging.warning(f"Failed to parse Greeks: {greeks_str}")
            return None

def process_option_csv(csv_path, ticker):
    df = pd.read_csv(csv_path)
    filename = os.path.basename(csv_path)
    fetch_timestamp = parse_timestamp_from_filename(filename)
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
                    'trade_date': convert_unix_to_datetime(get_value(row, prefix, 'lastTrade')),
                    'greeks': parse_greeks(row.get(f'{prefix}greeks')),
                    'open': row.get(f'open{suffix}'),
                    'high': row.get(f'high{suffix}'),
                    'low': row.get(f'low{suffix}'),
                    # 'close': row.get(f'close{suffix}'),
                    'last_volume': row.get(f'last_volume{suffix}'),
                    'prevclose': row.get(f'prevclose{suffix}'),
                    'bidsize': row.get(f'bidsize{suffix}'),
                    'bid_date': convert_unix_to_datetime(row.get(f'bid_date{suffix}')),
                    'asksize': row.get(f'asksize{suffix}'),
                    'ask_date': convert_unix_to_datetime(row.get(f'ask_date{suffix}')),
                }
                option_quotes_to_insert.append(quote_data)

    return options_to_insert, option_quotes_to_insert


from sqlalchemy.dialects.postgresql import insert
import math
from sqlalchemy.exc import IntegrityError
from collections import defaultdict


def import_option_data(session, csv_path, ticker):
    options_inserted = 0
    options_updated = 0
    options_skipped = 0
    quotes_inserted = 0
    quotes_skipped = 0
    error_summary = defaultdict(int)

    try:
        # Insert all options first
        options_to_insert, option_quotes_to_insert = process_option_csv(csv_path, ticker)
        if options_to_insert:
            for option in options_to_insert:
                try:
                    stmt = insert(Option).values(**option)
                    stmt = stmt.on_conflict_do_nothing(
                        index_elements=['contract_id']
                    )
                    result = session.execute(stmt)
                    if result.rowcount == 1:
                        options_inserted += 1
                    else:
                        options_skipped += 1
                except Exception as e:
                    session.rollback()
                    options_skipped += 1
                    logging.warning(f"Error inserting option {option['contract_id']}: {str(e)}")

            session.commit()

            # Now, insert option quotes
            cleaned_quotes = [clean_option_quote(quote) for quote in option_quotes_to_insert if
                              clean_option_quote(quote)]

            for quote in cleaned_quotes:
                try:
                    stmt = insert(OptionQuote).values(**quote)
                    stmt = stmt.on_conflict_do_nothing(
                        index_elements=['contract_id', 'fetch_timestamp']
                    )
                    result = session.execute(stmt)
                    if result.rowcount == 1:
                        quotes_inserted += 1
                    else:
                        quotes_skipped += 1
                except IntegrityError as e:
                    print(e)
                    session.rollback()
                    if "already exists" in str(e.orig).lower():
                        quotes_skipped += 1
                        error_summary["Duplicate option quote"] += 1
                    elif "violates foreign key constraint" in str(e.orig).lower():
                        quotes_skipped += 1
                        error_summary["Missing option for quote"] += 1
                    else:
                        raise
                except SQLAlchemyError:
                    print(SQLAlchemyError)
                    session.rollback()
                    raise

            session.commit()

    except Exception as e:
        session.rollback()
        logging.error(f"Error importing option data for {ticker} from {csv_path}: {str(e)}", exc_info=True)
    finally:
        session.close()

    logging.info(f"Import summary for {ticker} from {os.path.basename(csv_path)}:")
    logging.info(f"Options inserted: {options_inserted}")
    logging.info(f"Options updated: {options_updated}")
    logging.info(f"Options skipped (already exist): {options_skipped}")
    logging.info(f"Option Quotes inserted: {quotes_inserted}")
    logging.info(f"Option Quotes skipped (already exist or missing option): {quotes_skipped}")

    if error_summary:
        logging.info("Error summary:")
        for error, count in error_summary.items():
            logging.info(f"  {error}: occurred {count} times")


def clean_option_quote(quote):
    try:
        # Handle NaN values
        for key in ['volume', 'open_interest']:
            if key in quote and (quote[key] is None or math.isnan(quote[key])):
                quote[key] = 0

        # Ensure numeric values are within range
        for key in ['last', 'bid', 'ask']:
            if key in quote and quote[key] is not None:
                quote[key] = min(max(float(quote[key]), -9999999999), 9999999999)

        # Ensure volume and open_interest are integers
        for key in ['volume', 'open_interest']:
            if key in quote and quote[key] is not None:
                quote[key] = int(quote[key])


        return quote
    except Exception as e:
        logging.error(f"Error cleaning option quote: {str(e)}", exc_info=True)
        return None


def import_stock_data(session, csv_path, ticker):
    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            logging.warning(f"CSV file is empty: {csv_path}")
            return

        filename = os.path.basename(csv_path)
        fetch_timestamp = parse_timestamp_from_filename(filename)
        if fetch_timestamp is None:
            logging.error(f"Failed to parse timestamp from filename: {filename}")
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
                symbol_quote_data[field] = convert_unix_to_datetime(symbol_quote_data[field])

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
            logging.info(f"Successfully imported/updated stock data for {ticker} from {filename}")
        except Exception as e:
            session.rollback()
            logging.error(f"Error inserting/updating quote for {ticker} at {fetch_timestamp}: {str(e)}")

    except Exception as e:
        session.rollback()
        logging.error(f"Error importing stock data for {ticker} from {csv_path}: {str(e)}", exc_info=True)
    finally:
        session.close()

def process_ticker_directory(base_path, ticker):
    Session = sessionmaker(bind=engine)
    session = Session(bind=engine.connect().execution_options(
        schema_translate_map={None: SCHEMA_NAME}
    ))

    ensure_symbol_exists(session, ticker)
    option_data_path = os.path.join(base_path, "optionchain", ticker)
    stock_data_path = os.path.join(base_path, "ProcessedData", ticker)

    if os.path.exists(stock_data_path):
        for root, _, files in os.walk(stock_data_path):
            for file in files:
                if file.endswith('.csv'):
                    import_stock_data(session, os.path.join(root, file), ticker)
    # if os.path.exists(option_data_path):
    #     for root, _, files in os.walk(option_data_path):
    #         for file in files:
    #             if file.endswith('.csv'):
    #                 import_option_data(session, os.path.join(root, file), ticker)


    session.close()


def process_ticker(args):

    base_path, ticker = args
    # if ticker == 'MNMD':
    return process_ticker_directory(base_path, ticker)


def main():
    create_schema_if_not_exists(engine, SCHEMA_NAME)

    logging.info("Starting main function")
    base_path = r"H:\stockalgo_data\data"
    option_chain_path = os.path.join(base_path, "optionchain")

    logging.info(f"Base path: {base_path}")
    logging.info(f"Option chain path: {option_chain_path}")

    tickers = [ticker for ticker in os.listdir(option_chain_path) if
               os.path.isdir(os.path.join(option_chain_path, ticker))]

    logging.info(f"Found {len(tickers)} tickers to process")
    logging.info(f"Tickers: {tickers}")

    num_processes = multiprocessing.cpu_count()
    logging.info(f"Using {num_processes} processes for parallel processing")

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        args_list = [(base_path, ticker) for ticker in tickers]
        list(executor.map(process_ticker, args_list))

    logging.info("Finished processing all tickers")


if __name__ == "__main__":
    logging.info("Script started")
    create_schema_and_tables(engine)
    main()
    logging.info("Script completed")