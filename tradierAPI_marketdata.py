import PrivateData.tradier_info
import math
import numpy as np
import ta
from sqlalchemy import inspect, UniqueConstraint, PrimaryKeyConstraint, TIMESTAMP
import PrivateData.tradier_info
import asyncio
from datetime import datetime, timedelta
import aiohttp
from sqlalchemy.exc import OperationalError
import pandas as pd
from sqlalchemy.dialects.postgresql import insert
from pangres import upsert
from datetime import datetime
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
import technical_analysis
from UTILITIES.logger_config import logger
from sqlalchemy import func,Column, Integer, String, Float, DateTime, Date, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base
from main_devmode import engine
import pytz

eastern = pytz.timezone('US/Eastern')


class OptionChainError(Exception):
    pass


paper_auth = PrivateData.tradier_info.paper_auth
real_acc = PrivateData.tradier_info.real_acc
real_auth = PrivateData.tradier_info.real_auth
sem = asyncio.Semaphore(100)


# def calculate_batch_size(data_list, total_elements_limit=32680):
#     """
#     Calculates the maximum batch size based on the number of fields per element and a total limit. 32680 WORKS, 42690 DOESNT WORK WITH THIS
#     """
#     if data_list:
#         fields_per_element = len(data_list[0])
#         max_elements_per_batch = total_elements_limit // fields_per_element
#         return max_elements_per_batch
#     else:
#         return 0  # Handle empty list



def insert_calculated_data(ticker, engine, calculated_data_df):
    """
    Inserts calculated option data using a dedicated database session.

    Args:
        ticker: The ticker symbol of the stock.
        calculated_data_dict: Dictionary containing the calculated data.
        engine: The SQLAlchemy engine to create a new session with.
    """
    try:
        # Convert the calculated data dictionary to a DataFrame
        # calculated_data_df = pd.DataFrame([calculated_data_dict])
        # print(calculated_data_df.columns,"colums caslc")
        # print(calculated_data_df.columns)
        calculated_data_df.set_index(['symbol_name', 'fetch_timestamp'], inplace=True)

        # Use the synchronous context of the engine
        with engine.begin() as connection:
            upsert(
                con=connection,
                df=calculated_data_df,
                table_name='processed_option_data',
                if_row_exists='ignore'  # or 'update' if you want to update existing rows
            )
            # print(f"Inserted processed option data for {ticker}")

    except SQLAlchemyError as e:
        print(f"Error inserting processed option data for {ticker}: {e}")

from db_schema_models import Symbol,SymbolQuote, Option, OptionQuote, ProcessedOptionData, TechnicalAnalysis



#Add this index to ensure uniqueness.
# Index('uq_processed_option_data_index', ProcessedOptionData.symbol_id, ProcessedOptionData.current_time, unique=True)
# async def get_option_data(db_session, symbol_id):
#     stmt = (
#         select(Option.root_symbol, Option.expiration_date, Option.strike, Option.option_type, Option.option_id)
#         .filter(Option.root_symbol == symbol_id)
#         .order_by(Option.expiration_date, Option.strike)
#     )
#     result = db_session.execute(stmt)
#     return {(o.root_symbol, o.expiration_date, o.strike, o.option_type): o.option_id for o in result}







# Ensure tables are created before querying
#TODO DO SYNCHRONOUS BULK INSERT
async def post_market_quotes(session, ticker, real_auth):
    url = "https://api.tradier.com/v1/markets/quotes"
    headers = {"Authorization": f"Bearer {real_auth}", "Accept": "application/json"}
    all_contracts = await lookup_all_option_contracts(session, ticker, real_auth)

    # Optimized batching: Use list comprehension for faster symbol string creation
    BATCH_SIZE = 8000
    results = [
        await fetch_quote_batch(
            session,
            url,
            headers,
            ",".join(all_contracts[i : i + BATCH_SIZE]),
        )
        for i in range(0, len(all_contracts), BATCH_SIZE)
    ]

    # Combine results and handle potential empty results using a single expression
    combined_df = pd.concat(results, ignore_index=True) if results and not all(df.empty for df in results) else None
    return combined_df

async def fetch_quote_batch(session, url, headers, symbols_str):
    payload = {"symbols": symbols_str, "greeks": "true"}
    timeout = aiohttp.ClientTimeout(total=45)

    try:
        async with sem:
            async with session.post(url, data=payload, headers=headers, timeout=timeout) as response:
                response.raise_for_status()
                data = await response.json()
                handle_rate_limit(response)

                if "quotes" in data and "quote" in data["quotes"]:
                    quotes = data["quotes"]["quote"]
                    return pd.DataFrame(quotes)
                else:
                    logger.error(f"Error: No market quote data found for symbols {symbols_str}.")
                    raise OptionChainError(f"No market quote data found for symbols {symbols_str}")

    except (aiohttp.ClientError, OptionChainError) as e:
        logger.exception(f"Error fetching market quotes for symbols {symbols_str}: {e}")
        raise

def handle_rate_limit(response):
    rate_limit_allowed = int(response.headers.get("X-Ratelimit-Allowed", "0"))
    rate_limit_used = int(response.headers.get("X-Ratelimit-Used", "0"))

    if rate_limit_used >= (rate_limit_allowed * 0.99):
        logger.error(
            f"Rate limit exceeded: Used {rate_limit_used} out of {rate_limit_allowed}"
        )

async def fetch(session, url, params, headers):
    rate_limit_allowed, rate_limit_used = None, None
    try:
        timeout = aiohttp.ClientTimeout(total=45)
        async with session.get(url, params=params, headers=headers, timeout=timeout) as response:
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "").lower()
            rate_limit_allowed = int(response.headers.get("X-Ratelimit-Allowed", "0"))
            rate_limit_used = int(response.headers.get("X-Ratelimit-Used", "0"))
            if rate_limit_used >= (rate_limit_allowed * 0.99):
                logger.error(
                    f"{url},{params}----Rate limit exceeded: Used {rate_limit_used} out of {rate_limit_allowed}"
                )
            if "application/json" in content_type:
                return await response.json()
            else:
                raise OptionChainError(
                    f"Fetch error: {content_type} with params {params} {url}"
                )
    except Exception as e:
        raise OptionChainError(f"Fetch error: {e} with params {params} {url}")


async def lookup_all_option_contracts(session, underlying, real_auth):
    async with sem:
        url = "https://api.tradier.com/v1/markets/options/lookup"
        params = {"underlying": underlying}
        headers = {"Authorization": f"Bearer {real_auth}", "Accept": "application/json"}
        try:
            async with session.get(url, params=params, headers=headers) as response:
                response.raise_for_status()
                data = await response.json()
                if "symbols" in data and data["symbols"]:
                    option_contracts = data["symbols"][0]["options"]
                    return sorted(option_contracts)
                else:
                    raise OptionChainError(f"No option lookup data found for {underlying}.")
        except aiohttp.ClientError as e:
            logger.error(f"Error fetching option contracts: {e}")
            return None

def process_option_quotes(all_contract_quotes, current_price, last_close_price):  # Add last_close_price
    df = all_contract_quotes
    # df = all_contract_quotes.groupby("option_type")
    # calls_df = grouped.get_group("call").copy()
    # puts_df = grouped.get_group("put").copy()
    # for df in grouped:
    # print(df.columns)
    df['contract_id'] = df['symbol']
    # df['Moneyness'] = np.where(df['option_type'] == 'call', df['strike'] - current_price,
    #                           current_price - df['strike']) #shape was off by 1?
    df["dollarsFromStrike"] = abs(df["strike"] - last_close_price)  # Use last_close_price here
    df["expiration_date"] = pd.to_datetime(df["expiration_date"]).dt.strftime('%Y-%m-%d')

    df["dollarsFromStrikeXoi"] = df["dollarsFromStrike"] * df["open_interest"]
    # df["MoneynessXoi"] = df["Moneyness"] * df["open_interest"]
    df["lastPriceXoi"] = df["last"] * df["open_interest"]

    # columns_to_drop = ["description", "type", "exch", "underlying", "bidexch", "askexch",
    #                    "expiration_date", "root_symbol"]
    #TODO wait why not use underlying and root symbol etc?
    # calls_df = calls_df.drop(columns_to_drop, axis=1)
    # puts_df = puts_df.drop(columns_to_drop, axis=1)
    return df




def convert_unix_to_datetime(unix_timestamp):
    if unix_timestamp is None or pd.isna(unix_timestamp) or unix_timestamp == 0:
        return None  # Handle None, NaN, and 0 explicitly

    try:
        timestamp = int(float(unix_timestamp))
        if len(str(timestamp)) == 13:
            return datetime.fromtimestamp(timestamp / 1000, tz=eastern)
        elif len(str(timestamp)) == 10:
            return datetime.fromtimestamp(timestamp, tz=eastern)
        else:
            logger.error(f"Unexpected timestamp format: {timestamp}")
            return None
    except ValueError:
        logger.error(f"Invalid timestamp: {timestamp}")
        return None
async def get_timesales(session, ticker, lookback_minutes):
    eastern = pytz.timezone('US/Eastern')
    end = datetime.now(eastern)
    start = end - timedelta(minutes=lookback_minutes)

    start_str = start.strftime("%Y-%m-%d %H:%M")
    end_str = end.strftime("%Y-%m-%d %H:%M")

    headers = {"Authorization": f"Bearer {real_auth}", "Accept": "application/json"}

    try:
        time_sale_response = await fetch(
            session,
            "https://api.tradier.com/v1/markets/timesales",
            params={
                "symbol": ticker,
                "interval": "1min",
                "start": start_str,
                "end": end_str,
                "session_filter": "all",
            },
            headers=headers,
        )

        if time_sale_response and "series" in time_sale_response and "data" in time_sale_response["series"]:
            data = time_sale_response["series"]["data"]

            # Check if the data is already a list
            if isinstance(data, list):
                df = pd.DataFrame(data)
            else:
                df = pd.DataFrame([data])  # Convert scalar values to a DataFrame with a single row

            # df['time'] = pd.to_datetime(df['time']).dt.tz_localize('UTC').dt.tz_convert('US/Eastern')
            # df.set_index('time', inplace=True)

            # Ensure we only return the latest minute of data
            latest_minute = df.index.min()
            latest_data = df.loc[latest_minute:latest_minute]

            if not latest_data.empty:
                return latest_data.iloc[-1].to_dict()
            else:
                logger.warning(f"No timesales data found for {ticker} in the last minute")
                return None
        else:
            logger.error(f"Failed to retrieve timesales data for {ticker}: Invalid response structure")
            return None
    except Exception as e:
        logger.error(f"Error fetching timesales data for {ticker}: {str(e)}")
        return None

async def get_options_data(db_session, session, ticker, loop_start_time):
    headers = {"Authorization": f"Bearer {real_auth}", "Accept": "application/json"}
    ticker_quote = await fetch(
        session,
        "https://api.tradier.com/v1/markets/quotes",
        params={"symbols": ticker, "greeks": "false"},
        headers=headers,
    )



    # Process Stock Quote Data
    quote_df = pd.DataFrame.from_dict(ticker_quote["quotes"]["quote"], orient="index").T

    current_price = quote_df.at[0, "last"]
    prevclose = quote_df.at[0, "prevclose"]
    symbol_name = ticker


    try:
        # Upsert Symbol
        insert_stmt = insert(Symbol).values(
            symbol_name=ticker,
            description=quote_df.at[0, "description"],
            type=quote_df.at[0, "type"]
        )
        update_dict = {c.name: c for c in insert_stmt.excluded}
        upsert_stmt = insert_stmt.on_conflict_do_update(
            index_elements=[Symbol.symbol_name],
            set_=update_dict
        ).returning(Symbol.symbol_name)  # Return the symbol_name

        result = db_session.execute(upsert_stmt)
        symbol_name_result = result.one_or_none()  # Use one_or_none() to fetch 0 or 1 row

        if not symbol_name_result:  # Check if the result is None
            raise Exception(f"Failed to insert or update symbol {ticker}. Check for database constraints or errors.")

        # print("close?",quote_df.at[0, "close"])
    except Exception as e:
        db_session.rollback()
        print(f"Error handling symbol for {ticker}: {e}")
        raise
    # print(symbol_id)
#TODO quotes will take multiple tickers/options as args. maybe be faster ?
    stock_price_data = {
        'symbol_name': symbol_name,
        'fetch_timestamp': loop_start_time.astimezone(eastern),
        'last_trade_timestamp': convert_unix_to_datetime(quote_df.at[0, "trade_date"]),
        'last_trade_price': quote_df.at[0, "last"],
        'current_bid': quote_df.at[0, "bid"],
        'current_ask': quote_df.at[0, "ask"],
        'daily_open': quote_df.at[0, "open"],
        'daily_high': quote_df.at[0, "high"],
        'daily_low': quote_df.at[0, "low"],
        'previous_close': quote_df.at[0, "prevclose"],
        'last_trade_volume': quote_df.at[0, "last_volume"],
        'daily_volume': quote_df.at[0, "volume"],
        'average_daily_volume': quote_df.at[0, "average_volume"],
        'week_52_high': quote_df.at[0, "week_52_high"],
        'week_52_low': quote_df.at[0, "week_52_low"],
        'daily_change': quote_df.at[0, "change"],
        'daily_change_percentage': quote_df.at[0, "change_percentage"],
        'current_bidsize': quote_df.at[0, "bidsize"],
        'bidexch': quote_df.at[0, "bidexch"],
        'current_bid_date': convert_unix_to_datetime(quote_df.at[0, "bid_date"]),
        'current_asksize': quote_df.at[0, "asksize"],
        'askexch': quote_df.at[0, "askexch"],
        'current_ask_date': convert_unix_to_datetime(quote_df.at[0, "ask_date"]),
        'exch': quote_df.at[0, "exch"],
    }
    # Fetch timesales data
    timesales_data = await get_timesales(session, ticker, lookback_minutes=1)
    print(timesales_data)
    if timesales_data:
        stock_price_data.update({
            'last_1min_timesale': timesales_data['time'],
            'last_1min_timestamp': convert_unix_to_datetime(timesales_data["timestamp"]),

            'last_1min_open': timesales_data['open'],
            'last_1min_high': timesales_data['high'],
            'last_1min_low': timesales_data['low'],
            'last_1min_close': timesales_data['close'],
            'last_1min_volume': timesales_data['volume'],
            'last_1min_vwap': timesales_data['vwap']
        })

    db_session.execute(
        insert(SymbolQuote)
        .values(stock_price_data)
        .on_conflict_do_update(
            constraint='symbol_quote_unique_constraint',
            set_=stock_price_data
        )
    )
    db_session.commit()


    all_contract_quotes = await post_market_quotes(session, ticker, real_auth)

    if all_contract_quotes is not None:
        options_df = process_option_quotes(all_contract_quotes, current_price, prevclose)
        options_df['fetch_timestamp'] = loop_start_time

        # Convert datetime columns directly in the DataFrame
        # datetime_cols = ['trade_date', 'ask_date', 'bid_date']
        # for col in datetime_cols:
        #     options_df[col] = pd.to_datetime(options_df[col], unit='ms')

        # options_df['fetch_timestamp'] = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')

        # Select and rename columns for the Option table
        option_columns = ['contract_id','expiration_date', 'strike', 'option_type', 'underlying',
                          'contract_size', 'description', 'expiration_type', 'exch']
        option_data_df = options_df[option_columns].copy()

        # Rename columns to match database schema

        # option_data_df['symbol_name'] = symbol_name
        # print(option_data_df.columns)
        # Reset the index to ensure it does not interfere with the upsert operation
        option_data_df.reset_index(drop=True, inplace=True)
        # print(option_data_df['root_symbol'])
        # if ticker == 'GOEV':
        #     option_data_df.to_csv('goev.csv')
        # if options_df['root_symbol'] is type(list):
        #     logger.warning(f"{option_data_df['root_symbol']}")
        #     option_data_df['root_symbol'] = option_data_df['root_symbol'].iloc[0]
        #     print(option_data_df['root_symbol'])
        # Set the index for the upsert operation
        option_data_df.set_index(['underlying', 'expiration_date', 'strike', 'option_type'], inplace=True)
        # option_data_df.to_csv("options_data.csv")


        # Using pangres for Option table (without index_col)
        upsert(
            con=engine,
            df=option_data_df,
            table_name="options",
            schema="public",
            if_row_exists='update',
            create_table=False,
        )

        # Fetch the inserted options to create a mapping of (symbol_id, expiration_date, strike, option_type) to option_id
        db_session.commit()

        # query = (
        #     select(Option)
        #     .filter(Option.underlying == symbol_name)
        # )
        # options_in_db =  db_session.execute(query)
        # options = {
        #     (row.underlying, row.expiration_date, row.strike, row.option_type): row.contract_id
        #     for row in options_in_db.scalars().all()
        # }
        # print( options)  TODO no
        # ... (rest of the code)

        # Convert 'expiration_date' to datetime with timezone directly in options_df
        options_df['expiration_date'] = pd.to_datetime(options_df['expiration_date']).dt.tz_localize(
            'UTC').dt.tz_convert('US/Eastern')

        # Create a list of dictionaries representing option quotes data
        option_quotes_data = [
            {
                "contract_id": row['contract_id'],
                "root_symbol": row['root_symbol'],
                "fetch_timestamp": loop_start_time.astimezone(eastern),
                "last": row["last"],
                "bid": row["bid"],
                "ask": row["ask"],
                "volume": row["volume"],
                "greeks": row["greeks"],
                "change_percentage": row["change_percentage"],
                "average_volume": row["average_volume"],
                "last_volume": row["last_volume"],
                "trade_date": convert_unix_to_datetime(row["trade_date"]),
                "prevclose": row["prevclose"],
                "week_52_high": row["week_52_high"],
                "week_52_low": row["week_52_low"],
                "bidsize": row["bidsize"],
                "bidexch": row["bidexch"],
                "bid_date": convert_unix_to_datetime(row["bid_date"]),
                "asksize": row["asksize"],
                "askexch": row["askexch"],
                "ask_date": convert_unix_to_datetime(row["ask_date"]),
                "open_interest": row["open_interest"],
                "change": row["change"],
                "open": row["open"],
                "high": row["high"],
                "low": row["low"],
                "close": row["close"],
            }
            for _, row in options_df.iterrows() if row['contract_id']
        ]

        # Use SQLAlchemy's Core API for bulk insert
        with engine.begin() as conn:
            conn.execute(
                insert(OptionQuote),
                option_quotes_data
            )

        # Get technical analysis data
        # ta_df = await technical_analysis.get_ta(session, ticker)
        # if not ta_df.empty:
        #     ta_data_list = ta_df.to_dict(orient='records')
        #     for data in ta_data_list:
        #         data["symbol_name"] = symbol_name
        #
        #     # Create DataFrame for bulk insert
        #     ta_data_df = pd.DataFrame(ta_data_list)
        #
        #     # Use bulk_insert_mappings for faster inserts
        #     with engine.begin() as conn:
        #         conn.execute(
        #             insert(TechnicalAnalysis),
        #             ta_data_df.to_dict(orient="records")
        #         )
        #
        # else:
        #     print("TA DataFrame is empty")

        db_session.commit()

    # options_df.to_csv("options_df_test.csv")
    return prevclose, current_price, options_df, symbol_name
