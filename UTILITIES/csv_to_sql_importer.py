import json
import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime, date
import pytz
import logging
from database_operations import create_database_tables ,create_schema_and_tables
from sqlalchemy.exc import IntegrityError
from db_schema_models import Symbol, Option, OptionQuote, SymbolQuote, TechnicalAnalysis, ProcessedOptionData
from PrivateData import sql_db
# Import the new mappings
from csv_db_mappings import option_mapping, option_quote_mapping, symbol_quote_mapping, technical_analysis_mapping, processed_option_data_mapping
# import main_devmode
DATABASE_URI = sql_db.DATABASE_URI
engine = create_engine(DATABASE_URI, echo=False, pool_size=50, max_overflow=100)

# Set up logging
logging.basicConfig(filename='import_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
def parse_timestamp_from_filename(filename):
    try:
        parts = filename.split('_')
        date_str, time_str = parts[1], parts[2].split('.')[0]

        year = int(date_str[:2]) + 2000
        month = int(date_str[2:4])
        day = int(date_str[4:6])
        hour = int(time_str[:2])
        minute = int(time_str[2:4])

        naive_timestamp = datetime(year, month, day, hour, minute)
        localized_timestamp = pytz.timezone('US/Eastern').localize(naive_timestamp)

        return localized_timestamp
    except (IndexError, ValueError) as e:
        logging.error(f"Error parsing timestamp from filename {filename}: {str(e)}")
        return None

def convert_unix_to_datetime(timestamp):
    if timestamp is None or pd.isna(timestamp):
        return None
    try:
        # Convert scientific notation to float, then to integer
        timestamp = int(float(timestamp))
        # Check if the timestamp is in milliseconds (13 digits) or seconds (10 digits)
        if len(str(timestamp)) == 13:
            return datetime.fromtimestamp(timestamp / 1000, tz=pytz.timezone('US/Eastern'))
        elif len(str(timestamp)) == 10:
            return datetime.fromtimestamp(timestamp, tz=pytz.timezone('US/Eastern'))
        else:
            logging.error(f"Unexpected timestamp format: {timestamp}")
            return None
    except ValueError:
        logging.error(f"Invalid timestamp: {timestamp}")
        return None


def parse_date(date_value):
    # print(date_value)
    # print(type(date_value))
    if isinstance(date_value, int):
        try:
            # Parse YYMMDD format
            year = 2000 + (date_value // 10000)  # Assuming all years are in the 2000s
            month = (date_value // 100) % 100
            day = date_value % 100
            return date(year, month, day)
        except ValueError:
            logging.error(f"Unable to parse date from integer: {date_value}")
            return None
    elif isinstance(date_value, str):
        try:
            return datetime.strptime(date_value, '%Y-%m-%d').date()
        except ValueError:
            try:
                return datetime.strptime(date_value, '%m/%d/%Y').date()
            except ValueError:
                logging.error(f"Unable to parse date string: {date_value}")
                return None
    else:
        logging.error(f"Unexpected date format: {date_value}")
        return None




def map_csv_to_db(csv_row, mapping):
    db_data = {}
    for db_field, csv_fields in mapping.items():
        if callable(csv_fields):
            db_data[db_field] = csv_fields(csv_row)
        else:
            for csv_field in csv_fields:
                if csv_field in csv_row:
                    value = csv_row[csv_field]
                    if db_field in ['trade_date', 'bid_date', 'ask_date']:
                        value = convert_unix_to_datetime(value)
                    db_data[db_field] = value
                    break
    return db_data


# def split_option_row(row):
#     call_row = {}
#     put_row = {}
#     common_fields = {}
#
#     for key, value in row.items():
#         if key.startswith('c_') or key.endswith('_x') or 'Call' in key:
#             call_row[key] = value
#         elif key.startswith('p_') or key.endswith('_y') or 'Put' in key:
#             put_row[key] = value
#         else:
#             common_fields[key] = value
#
#     call_row.update(common_fields)
#     put_row.update(common_fields)
#
#     return call_row, put_row

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


def get_prevclose(row, prefix):
    # First, try to get the value using the standard method
    prevclose = get_value(row, prefix, 'prevclose')

    # # If that doesn't work, try LastPrice
    # if prevclose is None:
    #     prevclose = get_value(row, prefix, 'LastPrice')

    return prevclose


def parse_greeks(greeks_str):
    if not greeks_str or pd.isna(greeks_str):
        return None

    try:
        # Try parsing as JSON first (new format)
        return json.loads(greeks_str.replace("'", '"'))
    except json.JSONDecodeError:
        # If JSON parsing fails, try the old format
        try:
            return {'implied_volatility': float(greeks_str)}
        except ValueError:
            logging.warning(f"Failed to parse Greeks: {greeks_str}")
            return None


def process_single_option(option_data, row, prefix, fetch_timestamp, session):
    # Define fields that belong to the Option model
    option_fields = ['contract_id', 'underlying', 'expiration_date', 'strike', 'option_type']

    # Extract only the fields that belong to Option
    option_specific_data = {k: v for k, v in option_data.items() if k in option_fields}

    # Handle Option data
    option = session.query(Option).filter_by(contract_id=option_specific_data['contract_id']).first()
    if not option:
        option = Option(**option_specific_data)
        session.add(option)
    else:
        for key, value in option_specific_data.items():
            setattr(option, key, value)

    # Handle OptionQuote data
    greeks_key = f"{prefix}greeks"
    iv_key = f"{prefix}IV"

    greeks = parse_greeks(row.get(greeks_key))
    if greeks is None and iv_key in row:
        greeks = {'implied_volatility': row.get(iv_key)}

    quote_data = {
        'contract_id': option_data['contract_id'],
        'fetch_timestamp': fetch_timestamp,
        'root_symbol': row.get('root_symbol'),
        'last': get_value(row, prefix, 'LastPrice'),
        'change': get_value(row, prefix, 'change'),
        'volume': get_value(row, prefix, 'Volume'),
        'open_interest': get_value(row, prefix, 'OI'),
        'bid': get_value(row, prefix, 'bid'),
        'ask': get_value(row, prefix, 'ask'),
        'prevclose': option_data.get('prevclose'),
        'change_percentage': get_value(row, prefix, 'PercentChange'),
        'last_volume': option_data.get('last_volume'),
        'trade_date': convert_unix_to_datetime(get_value(row, prefix, 'lastTrade')),
        'open': option_data.get('open'),
        'high': option_data.get('high'),
        'low': option_data.get('low'),
        'bidsize': option_data.get('bidsize'),
        'bidexch': option_data.get('bidexch'),
        'bid_date': convert_unix_to_datetime(option_data.get('bid_date')),
        'asksize': option_data.get('asksize'),
        'askexch': option_data.get('askexch'),
        'ask_date': convert_unix_to_datetime(option_data.get('ask_date')),
        'greeks': greeks,
    }

    # Remove None values
    quote_data = {k: v for k, v in quote_data.items() if v is not None}

    option_quote = OptionQuote(**quote_data)
    session.add(option_quote)

    return option

def process_option_row(row, ticker, fetch_timestamp, session):
    calls_processed = 0
    puts_processed = 0

    common_data = {
        'underlying': ticker,
        'expiration_date': parse_date(row.get('ExpDate')),
        'strike': row.get('Strike'),
    }

    # Process call option
    if 'c_contractSymbol' in row or 'Call_LastPrice' in row:
        call_data = {
            **common_data,
            'option_type': 'call',
            'contract_id': row.get('c_contractSymbol'),
            'open': row.get('open_y'),
            'high': row.get('high_y'),
            'low': row.get('low_y'),
            'last_volume': row.get('last_volume_y'),
            'prevclose': row.get('prevclose_y'),
            'bidsize': row.get('bidsize_y'),
            'bidexch': row.get('bidexch_y'),
            'bid_date': row.get('bid_date_y'),
            'asksize': row.get('asksize_y'),
            'askexch': row.get('askexch_y'),
            'ask_date': row.get('ask_date_y'),
            'greeks': parse_greeks(row.get('c_greeks')),
        }
        call_option = process_single_option(call_data, row, 'c_', fetch_timestamp, session)
        if call_option:
            calls_processed += 1

    # Process put option
    if 'p_contractSymbol' in row or 'Put_LastPrice' in row:
        put_data = {
            **common_data,
            'option_type': 'put',
            'contract_id': row.get('p_contractSymbol'),
            'open': row.get('open_x'),
            'high': row.get('high_x'),
            'low': row.get('low_x'),
            'last_volume': row.get('last_volume_x'),
            'prevclose': row.get('prevclose_x'),
            'bidsize': row.get('bidsize_x'),
            'bidexch': row.get('bidexch_x'),
            'bid_date': row.get('bid_date_x'),
            'asksize': row.get('asksize_x'),
            'askexch': row.get('askexch_x'),
            'ask_date': row.get('ask_date_x'),
            'greeks': parse_greeks(row.get('p_greeks')),
        }
        put_option = process_single_option(put_data, row, 'p_', fetch_timestamp, session)
        if put_option:
            puts_processed += 1

    return calls_processed, puts_processed



def import_option_data(session, csv_path, ticker):
    try:
        ensure_symbol_exists(session, ticker)
        df = pd.read_csv(csv_path)
        # logging.info(f"Processing CSV file: {csv_path}")
        # logging.info(f"CSV columns: {df.columns.tolist()}")
        filename = os.path.basename(csv_path)
        fetch_timestamp = parse_timestamp_from_filename(filename)
        if fetch_timestamp is None:
            return

        calls_processed = 0
        puts_processed = 0

        for index, row in df.iterrows():
            calls, puts = process_option_row(row, ticker, fetch_timestamp, session)
            calls_processed += calls
            puts_processed += puts

        session.commit()
        logging.info(f"Successfully imported option data for {ticker} from {filename}")
        logging.info(f"Calls processed: {calls_processed}, Puts processed: {puts_processed}")

        # verify_data_insertion(session, ticker, fetch_timestamp)

    except Exception as e:
        session.rollback()
        logging.error(f"Error importing option data for {ticker} from {csv_path}: {str(e)}", exc_info=True)

def verify_data_insertion(session, ticker, fetch_timestamp):
    call_count = session.query(Option).filter_by(underlying=ticker, option_type='call').count()
    put_count = session.query(Option).filter_by(underlying=ticker, option_type='put').count()
    quote_count = session.query(OptionQuote).join(Option).filter(Option.underlying == ticker, OptionQuote.fetch_timestamp == fetch_timestamp).count()
    logging.info(f"Verification - Options for {ticker}: Calls: {call_count}, Puts: {put_count}, Option Quotes for this fetch: {quote_count}")


def import_stock_data(session, csv_path, ticker):
    try:
        ensure_symbol_exists(session, ticker)
        df = pd.read_csv(csv_path)
        filename = os.path.basename(csv_path)
        fetch_timestamp = parse_timestamp_from_filename(filename)
        if fetch_timestamp is None:
            return

        existing_quote = session.query(SymbolQuote).filter_by(
            symbol_name=ticker,
            fetch_timestamp=fetch_timestamp
        ).first()

        if not existing_quote:
            first_row = df.iloc[0]
            symbol_quote_data = map_csv_to_db(first_row, symbol_quote_mapping)
            symbol_quote_data['symbol_name'] = ticker
            symbol_quote_data['fetch_timestamp'] = fetch_timestamp

            # Convert last_trade_timestamp from Unix timestamp to datetime
            if 'last_trade_timestamp' in symbol_quote_data and symbol_quote_data['last_trade_timestamp'] is not None:
                symbol_quote_data['last_trade_timestamp'] = convert_unix_to_datetime(
                    symbol_quote_data['last_trade_timestamp'])

            # Convert other timestamp fields
            for field in ['current_bid_date', 'current_ask_date']:
                if field in symbol_quote_data and symbol_quote_data[field] is not None:
                    symbol_quote_data[field] = convert_unix_to_datetime(symbol_quote_data[field])

            # Remove any None values
            symbol_quote_data = {k: v for k, v in symbol_quote_data.items() if v is not None}

            symbol_quote = SymbolQuote(**symbol_quote_data)
            session.add(symbol_quote)
            logging.info(f"Added stock quote for {ticker} at {fetch_timestamp}")
        else:
            logging.info(f"Stock quote for {ticker} at {fetch_timestamp} already exists. Skipping.")

        try:
            session.commit()
            logging.info(f"Successfully imported data for {ticker} from {filename}")
        except IntegrityError as e:
            session.rollback()
            logging.error(f"IntegrityError while importing data for {ticker} from {filename}: {str(e)}")
    except Exception as e:
        session.rollback()
        logging.error(f"Error importing data for {ticker} from {csv_path}: {str(e)}", exc_info=True)


def process_ticker_directory(session, base_path, ticker):
    option_data_path = os.path.join(base_path, "optionchain", ticker)
    stock_data_path = os.path.join(base_path, "ProcessedData", ticker)


    # Import option data
    if os.path.exists(option_data_path):
        for root, _, files in os.walk(option_data_path):
            for file in files:
                if file.endswith('.csv'):
                    import_option_data(session, os.path.join(root, file), ticker)
    # Import stock data
    if os.path.exists(stock_data_path):
        for root, _, files in os.walk(stock_data_path):
            for file in files:
                if file.endswith('.csv'):
                    import_stock_data(session, os.path.join(root, file), ticker)



def main():
    engine = create_engine(DATABASE_URI)
    Session = sessionmaker(bind=engine)
    session = Session()

    base_path = r"H:\stockalgo_data\data"

    # Process all ticker directories
    option_chain_path = os.path.join(base_path, "optionchain")
    for ticker in os.listdir(option_chain_path):
        if os.path.isdir(os.path.join(option_chain_path, ticker)):
            logging.info(f"Processing ticker: {ticker}")
            process_ticker_directory(session, base_path, ticker)

    session.close()


if __name__ == "__main__":
    create_schema_and_tables(engine)
    main()