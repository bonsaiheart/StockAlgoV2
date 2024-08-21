import os
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from datetime import datetime, date
import pytz
import logging
from sqlalchemy.exc import IntegrityError
from db_schema_models import Symbol, Option, OptionQuote, SymbolQuote, TechnicalAnalysis, ProcessedOptionData
from PrivateData import sql_db
DATABASE_URI = sql_db.DATABASE_URI

# Set up logging
logging.basicConfig(filename='import_log.txt', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def parse_timestamp_from_filename(filename):
    try:
        parts = filename.split('_')
        date_time_str = parts[1] + parts[2].split('.')[0]
        return datetime.strptime(date_time_str, '%y%m%d%H%M').replace(tzinfo=pytz.UTC)
    except (IndexError, ValueError) as e:
        logging.error(f"Error parsing timestamp from filename {filename}: {str(e)}")
        return None

def parse_date(date_value):
    print(date_value)
    print(type(date_value))
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
def import_option_data(session, csv_path, ticker):
    try:
        df = pd.read_csv(csv_path)
        filename = os.path.basename(csv_path)
        fetch_timestamp = parse_timestamp_from_filename(filename)
        if fetch_timestamp is None:
            return

        for _, row in df.iterrows():
            try:
                expiration_date = parse_date(row['ExpDate'])
                if expiration_date is None:
                    continue

                # Create or update Option
                option = session.query(Option).filter_by(contract_id=row['c_contractSymbol']).first()
                if not option:
                    option = Option(
                        contract_id=row['c_contractSymbol'],
                        underlying=ticker,
                        expiration_date=expiration_date,
                        strike=row['Strike'],
                        option_type='call' if row['c_contractSymbol'].endswith('C') else 'put',
                        contract_size=100,
                        description=row.get('description', ''),
                        expiration_type='american',
                        exch=row.get('exch', '')
                    )
                    session.add(option)

                # Create OptionQuote
                option_quote = OptionQuote(
                    contract_id=row['c_contractSymbol'],
                    fetch_timestamp=fetch_timestamp,
                    last=row['Call_LastPrice'] if row['c_contractSymbol'].endswith('C') else row['Put_LastPrice'],
                    change=row['c_change'] if row['c_contractSymbol'].endswith('C') else row['p_change'],
                    volume=row['Call_Volume'] if row['c_contractSymbol'].endswith('C') else row['Put_Volume'],
                    bid=row['c_bid'] if row['c_contractSymbol'].endswith('C') else row['p_bid'],
                    ask=row['c_ask'] if row['c_contractSymbol'].endswith('C') else row['p_ask'],
                    open_interest=row['Call_OI'] if row['c_contractSymbol'].endswith('C') else row['Put_OI'],
                    greeks=row.get('c_greeks', {}) if row['c_contractSymbol'].endswith('C') else row.get('p_greeks', {})
                )
                session.add(option_quote)

            except KeyError as e:
                logging.warning(f"Missing column in file {filename}: {str(e)}")
            except Exception as e:
                logging.error(f"Error processing row in file {filename}: {str(e)}")

        session.commit()
        logging.info(f"Successfully imported option data for {ticker} from {filename}")
    except Exception as e:
        session.rollback()
        logging.error(f"Error importing option data for {ticker} from {csv_path}: {str(e)}")


def import_stock_data(session, csv_path, ticker):
    try:
        df = pd.read_csv(csv_path)
        filename = os.path.basename(csv_path)
        fetch_timestamp = parse_timestamp_from_filename(filename)
        if fetch_timestamp is None:
            return

        symbol_quote = SymbolQuote(
            symbol_name=ticker,
            fetch_timestamp=fetch_timestamp,
            last_price=df['Current Stock Price'].iloc[0],
            # Add other fields as necessary, with checks for column existence
        )
        session.add(symbol_quote)

        processed_data = ProcessedOptionData(
            symbol_name=ticker,
            fetch_timestamp=fetch_timestamp,
            current_stock_price=df['Current Stock Price'].iloc[0],
            current_sp_change_lac=df['Current SP % Change(LAC)'].iloc[0],
            maximumpain=df['Maximum Pain'].iloc[0],
            bonsai_ratio=df['Bonsai Ratio'].iloc[0],
            bonsai_ratio_2=df['Bonsai Ratio 2'].iloc[0],
            pcr_vol=df['PCR-Vol'].iloc[0],
            pcr_oi=df['PCR-OI'].iloc[0],
            # Add other fields as necessary, with checks for column existence
        )
        session.add(processed_data)

        # Import TechnicalAnalysis data if available
        if 'RSI' in df.columns:
            tech_analysis = TechnicalAnalysis(
                symbol_name=ticker,
                fetch_timestamp=fetch_timestamp,
                RSI_14_1min=df['RSI'].iloc[0],
                # Add other technical indicators as necessary, with checks for column existence
            )
            session.add(tech_analysis)

        session.commit()
        logging.info(f"Successfully imported stock data for {ticker} from {filename}")
    except Exception as e:
        session.rollback()
        logging.error(f"Error importing stock data for {ticker} from {csv_path}: {str(e)}")

def process_ticker_directory(session, base_path, ticker):
    option_data_path = os.path.join(base_path, "optionchain", ticker)
    stock_data_path = os.path.join(base_path, "ProcessedData", ticker)

    # Import option data
    # if os.path.exists(option_data_path):
    #     for root, _, files in os.walk(option_data_path):
    #         for file in files:
    #             if file.endswith('.csv'):
    #                 import_option_data(session, os.path.join(root, file), ticker)

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
    main()