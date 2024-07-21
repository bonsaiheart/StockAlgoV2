import traceback

import PrivateData.tradier_info
import math

import numpy as np
import ta
from sqlalchemy import  inspect, UniqueConstraint, PrimaryKeyConstraint, desc
import PrivateData.tradier_info
import asyncio
from datetime import datetime, timedelta
import aiohttp
import pandas as pd
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from UTILITIES.logger_config import logger
from sqlalchemy import select, Column, Integer, String, Float, DateTime, ForeignKey, JSON
from sqlalchemy.orm import relationship, joinedload
from sqlalchemy.ext.declarative import declarative_base
from main_devmode import engine
from sqlalchemy.dialects.postgresql import insert

# schemaname = {'schema': 'stockalgov2'}
class OptionChainError(Exception):
    pass


paper_auth = PrivateData.tradier_info.paper_auth
real_acc = PrivateData.tradier_info.real_acc
real_auth = PrivateData.tradier_info.real_auth
sem = asyncio.Semaphore(10000)


Base = declarative_base()
def calculate_batch_size(data_list, total_elements_limit=32680):
    """
    Calculates the maximum batch size based on the number of fields per element and a total limit.
    """
    if data_list:
        fields_per_element = len(data_list[0])
        max_elements_per_batch = total_elements_limit // fields_per_element
        return max_elements_per_batch
    else:
        return 0  # Handle empty list

# async def insert_calculated_data(ticker,db_session,calculated_data_dict):
#     try:
#         await db_session.execute(
#             insert(ProcessedOptionData)
#             .values(calculated_data_dict)
#             .on_conflict_do_nothing(constraint="uq_symbol_current_time_constraint")
#         )
#         await db_session.commit()
#         print(f"Inserted processed option data for {ticker}")
#     except SQLAlchemyError as e:
#         print(f"Error inserting processed option data: {e}")
#         await db_session.rollback()
#     return "Inserted caclulated data"
async def insert_calculated_data(ticker, db_session, calculated_data_dict):
    try:
        await db_session.execute(
            insert(ProcessedOptionData)
            .values(calculated_data_dict)
            .on_conflict_do_nothing(constraint="uq_symbol_current_time_constraint")
        )
        print(f"Inserted processed option data for {ticker}")  # Committing will happen later
    except SQLAlchemyError as e:
        print(f"Error inserting processed option data: {e}")
async def get_ta_data(db_session, symbol):
    result = await db_session.execute(
        select(TechnicalAnalysis).where(TechnicalAnalysis.symbol_id == symbol.symbol_id)
    )
    ta_df = pd.DataFrame(result.all(), columns=TechnicalAnalysis.__table__.columns.keys())

    # Convert the fetch_timestamp column to datetime
    ta_df['fetch_timestamp'] = pd.to_datetime(ta_df['fetch_timestamp'])
    return ta_df


class Symbol(Base):
    __tablename__ = 'symbols'
    symbol_id = Column(Integer, primary_key=True)
    symbol_name = Column(String, unique=True)


class Option(Base):
    __tablename__ = 'options'

    option_id = Column(Integer, primary_key=True)
    symbol_id = Column(Integer, ForeignKey('symbols.symbol_id', ondelete='CASCADE'))

    # Establish the relationship with Symbol
    symbol = relationship("Symbol", backref="options")

    expiration_date = Column(DateTime, index=True)
    strike_price = Column(Float)
    option_type = Column(String)
    root_symbol = Column(String)  # Assuming root_symbol is directly stored here
    contract_size = Column(Integer)
    description = Column(String)
    expiration_type = Column(String)
    exch = Column(String)

    __table_args__ = (
        UniqueConstraint('symbol_id', 'expiration_date', 'strike_price', 'option_type', name='uq_option_constraint'),
    )
class OptionQuote(Base):
    __tablename__ = 'option_quotes'
    __table_args__ = (
        UniqueConstraint('option_id', 'fetch_timestamp', name='uq_option_quote_constraint'),
    )

    quote_id = Column(Integer, primary_key=True, autoincrement=True)  # Auto-increment quote_id
    option_id = Column(Integer, ForeignKey('options.option_id'))
    option = relationship("Option", backref="quotes")
    timestamp = Column(DateTime, default=datetime.utcnow)
    fetch_timestamp = Column(DateTime, default=datetime.utcnow)
    last = Column(Float)
    change = Column(Float)
    volume = Column(Integer)
    open = Column(Float)
    high = Column(Float)
    low = Column(Float)
    close = Column(Float)
    bid = Column(Float)
    ask = Column(Float)
    greeks = Column(JSON)
    change_percentage = Column(Float)
    average_volume = Column(Integer)
    last_volume = Column(Integer)
    trade_date = Column(DateTime)
    prevclose = Column(Float)
    week_52_high = Column(Float)
    week_52_low = Column(Float)
    bidsize = Column(Integer)
    bidexch = Column(String)
    bid_date = Column(DateTime)
    asksize = Column(Integer)
    askexch = Column(String)
    ask_date = Column(DateTime)
    open_interest = Column(Integer)


class SymbolQuote(Base):
    __tablename__ = 'symbol_quotes'

    id = Column(Integer, primary_key=True, autoincrement=True)  # Add a primary key column
    symbol_id = Column(Integer, ForeignKey('symbols.symbol_id'))
    symbol = relationship("Symbol", backref="symbol_quotes")  # Relationship with Symbol
    timestamp = Column(DateTime)
    fetch_timestamp = Column(DateTime, default=datetime.utcnow)
    last_trade = Column(DateTime)
    last_price = Column(Float)
    bid = Column(Float)
    ask = Column(Float)
    open_price = Column(Float)
    high_price = Column(Float)
    low_price = Column(Float)
    close_price = Column(Float)
    volume = Column(Integer)
    average_volume = Column(Integer)
    week_52_high = Column(Float)  # Added field for 52-week high
    week_52_low = Column(Float)   # Added field for 52-week low
    bidsize = Column(Integer)
    bidexch = Column(String)
    bid_date = Column(DateTime)
    asksize = Column(Integer)
    askexch = Column(String)
    ask_date = Column(DateTime)
    description = Column(String(255))  # Added field for description
    exch = Column(String(1))  # Added field for exchange
    type = Column(String(10))  # Added field for type
    trade_date = Column(DateTime)  # Changed to DateTime for consistency
    prevclose = Column(Float)  # Added field for previous close
    change = Column(Float)  # Added field for change
    change_percentage = Column(Float)  # Added field for change percentage
    __table_args__ = (
        UniqueConstraint('symbol_id', 'fetch_timestamp', name='symbol_quote_unique_constraint'),
    )
class TechnicalAnalysis(Base):
    __tablename__ = 'technical_analysis'
    ta_id = Column(Integer, primary_key=True, autoincrement=True)
    symbol_id = Column(Integer, ForeignKey('symbols.symbol_id'), index=True)

    # 1mintimestamp_ = Column(DateTime, index=True, nullable=True)
    # timestamp_5min = Column(DateTime, index=True, nullable=True)
    # timestamp_15min = Column(DateTime, index=True, nullable=True)
    # Define other columns for each indicator and interval
    for interval in ["1min", "5min", "15min"]:
        # globals()[f"timestamp_{interval}"] = Column(DateTime, index=True)

        for indicator, data_type in [("timestamp", DateTime),
            ("price", Float), ("open", Float), ("high", Float), ("low", Float), ("close", Float),("volume", Float),
            ("vwap", Float), ("MACD_12_26", Float), ("Signal_Line_12_26", Float),
            ("MACD_diff_12_26", Float), ("MACD_diff_prev_12_26", Float),
            ("MACD_signal_12_26", String), ("AwesomeOsc", Float), ("SMA_20", Float),
            ("ADX", Float), ("CCI", Float), ("Williams_R", Float), ("PVO", Float),
            ("PPO", Float), ("CMF", Float), ("EoM", Float), ("OBV", Integer),
            ("MFI", Float), ("Keltner_Upper", Float), ("Keltner_Lower", Float), ("BB_high_20", Float),("BB_mid_20", Float),("BB_low_20", Float),("VPT", Float)
        ]:
            column_name = f"{interval}_{indicator}"
            locals()[column_name] = Column(data_type, nullable=True)
        # Explicitly define an interval column after the loops
    fetch_timestamp = Column(DateTime, index=True, nullable=False)
    # interval = Column(String, index=True)
    __table_args__ = (
        PrimaryKeyConstraint('ta_id'),
        UniqueConstraint('symbol_id', 'fetch_timestamp', name='uq_symbol_interval_timestamps'),
    )

from sqlalchemy.types import Float, Integer, String, DateTime, Boolean,Numeric

class ProcessedOptionData(Base):
    __tablename__ = 'processed_option_data'
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol_id = Column(Integer, ForeignKey('symbols.symbol_id'), index=True)

    current_time = Column(DateTime)
    exp_date = Column(DateTime)
    current_stock_price = Column(Float)
    current_sp_change_lac = Column(Float)
    maximumpain = Column(Float)
    bonsai_ratio = Column(Float)
    bonsai_ratio_2 = Column(Float)
    b1_dividedby_b2 = Column(Float)
    b2_dividedby_b1 = Column(Float)
    pcr_vol = Column(Float)
    pcr_oi = Column(Float)
    # pcrv_cp_strike = Column(Float)
    # pcroi_cp_strike = Column(Float)
    pcrv_up1 = Column(Float)
    pcrv_up2 = Column(Float)
    pcrv_up3 = Column(Float)
    pcrv_up4 = Column(Float)
    pcrv_down1 = Column(Float)
    pcrv_down2 = Column(Float)
    pcrv_down3 = Column(Float)
    pcrv_down4 = Column(Float)
    pcroi_up1 = Column(Float)
    pcroi_up2 = Column(Float)
    pcroi_up3 = Column(Float)
    pcroi_up4 = Column(Float)
    pcroi_down1 = Column(Float)
    pcroi_down2 = Column(Float)
    pcroi_down3 = Column(Float)
    pcroi_down4 = Column(Float)
    itm_pcr_vol = Column(Float)
    itm_pcr_oi = Column(Float)
    itm_pcrv_up1 = Column(Float)
    itm_pcrv_up2 = Column(Float)
    itm_pcrv_up3 = Column(Float)
    itm_pcrv_up4 = Column(Float)
    itm_pcrv_down1 = Column(Float)
    itm_pcrv_down2 = Column(Float)
    itm_pcrv_down3 = Column(Float)
    itm_pcrv_down4 = Column(Float)
    itm_pcroi_up1 = Column(Float)
    itm_pcroi_up2 = Column(Float)
    itm_pcroi_up3 = Column(Float)
    itm_pcroi_up4 = Column(Float)
    itm_pcroi_down1 = Column(Float)
    itm_pcroi_down2 = Column(Float)
    itm_pcroi_down3 = Column(Float)
    itm_pcroi_down4 = Column(Float)
    itm_oi = Column(Integer)
    total_oi = Column(Integer)
    itm_contracts_percent = Column(Float)
    net_iv = Column(Float)
    net_itm_iv = Column(Float)
    net_iv_mp = Column(Float)
    # net_iv_lac = Column(Float)#TODO fix this
    niv_current_strike = Column(Float)
    niv_1higher_strike = Column(Float)
    niv_1lower_strike = Column(Float)
    niv_2higher_strike = Column(Float)
    niv_2lower_strike = Column(Float)
    niv_3higher_strike = Column(Float)
    niv_3lower_strike = Column(Float)
    niv_4higher_strike = Column(Float)
    niv_4lower_strike = Column(Float)
    niv_highers_minus_lowers1thru2 = Column(Float)
    niv_highers_minus_lowers1thru4 = Column(Float)
    niv_1thru2_avg_percent_from_mean = Column(Float)
    niv_1thru4_avg_percent_from_mean = Column(Float)
    niv_dividedby_oi = Column(Float)
    itm_avg_niv_dividedby_itm_oi = Column(Float)
    closest_strike_to_cp = Column(Float)


    __table_args__ = (
        UniqueConstraint('symbol_id', 'current_time', name='uq_symbol_current_time_constraint'),
    )

#Add this index to ensure uniqueness.
# Index('uq_processed_option_data_index', ProcessedOptionData.symbol_id, ProcessedOptionData.current_time, unique=True)

async def get_ta(session, ticker):
    days_to_fetch = {
        "1min": 5,
        "5min": 20,
        "15min": 40,
    }

    def safe_calculation(df, column_name, calculation_function, *args, **kwargs):
        try:
           df[column_name] = calculation_function(*args, **kwargs)
        except Exception as e:
            logger.warning(
                f"{ticker} - Problem with: column_name={column_name}, function={calculation_function.__name__}, error={e}. This is usually caused by missing data from yfinance."
            )
            df[column_name] = np.nan

    async def fetch_and_process_data(interval):
        start = (datetime.today() - timedelta(days=days_to_fetch[interval])).strftime("%Y-%m-%d %H:%M")
        end = datetime.today().strftime("%Y-%m-%d %H:%M")

        headers = {"Authorization": f"Bearer {real_auth}", "Accept": "application/json"}
        time_sale_response = await fetch(
            session,
            "https://api.tradier.com/v1/markets/timesales",
            params={
                "symbol": ticker,
                "interval": interval,
                "start": start,
                "end": end,
                "session_filter": "all",
            },
            headers=headers,
        )

        if not time_sale_response or "series" not in time_sale_response or "data" not in time_sale_response["series"]:
            logger.warning(f"Failed to retrieve TA data for ticker {ticker} and interval {interval}")
            return pd.DataFrame()

        df = pd.DataFrame(time_sale_response["series"]["data"]).set_index("time")
        df.index = pd.to_datetime(df.index)
        latest_minute_data = df.tail(1).copy()

        # Define the calculations
        def perform_calculations():
            ema_windows = [5, 14, 20, 50, 200]
            rsi_windows = [2, 7, 14, 21]
            macd_windows = [(12, 26, 9)]
            cci_window = 20
            adx_window = 7
            williams_r_params = 14
            cmf_window = 20
            eom_window = 14
            mfi_window = 14
            keltner_window = 20
            keltner_atr_window = 10
            for window in ema_windows:
                safe_calculation(df, f"EMA_{window}_{interval}", ta.trend.ema_indicator, close=df["close"], window=window)

            for window in rsi_windows:
                safe_calculation(df, f"RSI_{window}_{interval}", ta.momentum.rsi, close=df["close"], window=window)


            for fast_window, slow_window, signal_window in macd_windows:
                macd_object = ta.trend.MACD(
                    close=df["close"],
                    window_slow=slow_window,
                    window_fast=fast_window,
                    window_sign=signal_window,
                    fillna=False
                )
                latest_minute_data[f"MACD_{fast_window}_{slow_window}"] = macd_object.macd().iloc[-1]
                latest_minute_data[f"Signal_Line_{fast_window}_{slow_window}"] = macd_object.macd_signal().iloc[-1]
                latest_minute_data[f"MACD_diff_{fast_window}_{slow_window}"] = macd_object.macd_diff().iloc[-1]
                latest_minute_data[f"MACD_diff_prev_{fast_window}_{slow_window}"] = macd_object.macd_diff().shift(1).iloc[-1]

                bullish_cross = (
                    (latest_minute_data[f"MACD_diff_{fast_window}_{slow_window}"] > 0) &
                    (latest_minute_data[f"MACD_diff_prev_{fast_window}_{slow_window}"] <= 0)
                )

                bearish_cross = (
                    (latest_minute_data[f"MACD_diff_{fast_window}_{slow_window}"] < 0) &
                    (latest_minute_data[f"MACD_diff_prev_{fast_window}_{slow_window}"] >= 0)
                )

                latest_minute_data[f"MACD_signal_{fast_window}_{slow_window}"] = np.where(
                    bullish_cross, "buy", np.where(bearish_cross, "sell", "hold")
                )

            safe_calculation(df, f"AwesomeOsc", ta.momentum.awesome_oscillator, high=df["high"], low=df["low"], window1=5, window2=34)
            latest_minute_data[f"AwesomeOsc"] = df[f"AwesomeOsc"].iloc[-1]
            safe_calculation(df, f"SMA_20", ta.trend.sma_indicator, close=df["close"], window=20)
            latest_minute_data[f"SMA_20"] = df[f"SMA_20"].iloc[-1]

            # ADX Calculation (handle potential insufficient data)
            if len(df) >= adx_window:
                safe_calculation(df, f"ADX", ta.trend.adx, high=df["high"], low=df["low"], close=df["close"], window=adx_window, fillna=False)
                latest_minute_data[f"ADX"] = df[f"ADX"].iloc[-1]
            else:
                print("Insufficient data for ADX calculation.")
                latest_minute_data[f"ADX"] = pd.NA  # or any other default value

            safe_calculation(df, f"CCI", ta.trend.cci, high=df["high"], low=df["low"], close=df["close"], window=cci_window)
            latest_minute_data[f"CCI"] = df[f"CCI"].iloc[-1]

            williams_r_object = ta.momentum.WilliamsRIndicator(high=df["high"], low=df["low"], close=df["close"], lbp=williams_r_params,fillna=False)
            # print(williams_r_object.williams_r()) NEED TO ALLIGN THE TIMESTAMP FOR EACH
            latest_minute_data[f"Williams_R"] = williams_r_object.williams_r().iloc[-1]


            # PVO Calculation (using the Williams %R-like approach)
            pvo_object = ta.momentum.PercentageVolumeOscillator(
                volume=df["volume"], window_slow=26, window_fast=12, window_sign=9, fillna=False
            )
            latest_minute_data[f"PVO"] = pvo_object.pvo().iloc[-1]

            # PPO Calculation (using the Williams %R-like approach)
            ppo_object = ta.momentum.PercentagePriceOscillator(
                close=df["close"], window_slow=26, window_fast=12, window_sign=9, fillna=False
            )
            latest_minute_data[f"PPO"] = ppo_object.ppo().iloc[-1]


            safe_calculation(df, f"CMF", ta.volume.chaikin_money_flow, high=df["high"], low=df["low"], close=df["close"], volume=df["volume"], window=cmf_window)
            latest_minute_data[f"CMF"] = df[f"CMF"].iloc[-1]

            safe_calculation(df, f"EoM", ta.volume.ease_of_movement, high=df["high"], low=df["low"], volume=df["volume"], window=eom_window)
            latest_minute_data[f"EoM"] = df[f"EoM"].mean()  # Use mean to get single value

            safe_calculation(df, f"OBV", ta.volume.on_balance_volume, close=df["close"], volume=df["volume"])
            latest_minute_data[f"OBV"] = df[f"OBV"].iloc[-1]

            safe_calculation(df, f"MFI", ta.volume.money_flow_index, high=df["high"], low=df["low"], close=df["close"], volume=df["volume"], window=mfi_window)
            latest_minute_data[f"MFI"] = df[f"MFI"].iloc[-1]


            keltner_channel = ta.volatility.KeltnerChannel(
                high=latest_minute_data["high"],
                low=latest_minute_data["low"],
                close=latest_minute_data["close"],
                window=keltner_window,
                window_atr=keltner_atr_window
            )
            safe_calculation(latest_minute_data, f"Keltner_Upper", keltner_channel.keltner_channel_hband_indicator)
            safe_calculation(latest_minute_data, f"Keltner_Lower", keltner_channel.keltner_channel_lband_indicator)
            # safe_calculation(latest_minute_data, f"VPT", ta.volume.volume_price_trend, close=latest_minute_data["close"], volume=latest_minute_data["volume"])
            # Bollinger Bands Calculation (with enough historical data)
            bb_windows = [20]  # Standard BB window, add more if needed
            for window in bb_windows:
                bb_object = ta.volatility.BollingerBands(close=df["close"], window=window, window_dev=2)
                latest_minute_data[f"BB_high_{window}"] = bb_object.bollinger_hband().iloc[-1]
                latest_minute_data[f"BB_mid_{window}"] = bb_object.bollinger_mavg().iloc[-1]
                latest_minute_data[f"BB_low_{window}"] = bb_object.bollinger_lband().iloc[-1]
                # VPT Calculation (using the Williams %R-like approach)
            vpt_object = ta.volume.VolumePriceTrendIndicator(close=df["close"], volume=df["volume"])
            latest_minute_data[f"VPT"] = vpt_object.volume_price_trend().iloc[-1]



        perform_calculations()

        latest_minute_data[f"timestamp"] = latest_minute_data.index
        # print("latestdata",latest_minute_data)
        return latest_minute_data.reset_index(drop=True)

    intervals = ["1min", "5min", "15min"]
    results = await asyncio.gather(*[fetch_and_process_data(interval) for interval in intervals])

    processed_results = []
    for i, result in enumerate(results):
        result.columns = [f"{intervals[i]}_{col}" for col in result.columns]
        # result = result.dropna(axis=0, how='all')

        processed_results.append(result)
    # Concatenate the DataFrames horizontally (side by side)

    final_df = pd.concat(processed_results, axis=1,join='outer')
    final_df['fetch_timestamp'] = datetime.utcnow() # Add fetch_timestamp here
    # print(final_df)

    # Drop columns with all NaN values after concatenation
    # final_df.dropna(axis=0, how='all', inplace=True)

    # print("final df = ", final_df)
    return final_df





def convert_unix_to_datetime(unix_timestamp):
    # Convert Unix timestamp (milliseconds) to datetime
    return datetime.fromtimestamp(unix_timestamp / 1000.0)


async def create_database_tables(engine):
    # await create_schema(engine)

    async with engine.begin() as conn:
        def sync_inspect(connection):
            inspector = inspect(connection)
            return inspector.get_table_names()

        existing_tables = await conn.run_sync(sync_inspect)
        tables_to_create = [Symbol, Option, OptionQuote, SymbolQuote, TechnicalAnalysis, ProcessedOptionData]

        # Create Symbol table first
        if Symbol.__table__.name not in existing_tables:
            await conn.run_sync(Symbol.__table__.create)

        # Then create the other tables
        for table in tables_to_create:
            if table.__table__.name != Symbol.__table__.name and table.__table__.name not in existing_tables:
                await conn.run_sync(table.__table__.create)

        logger.info("Database tables created or already exist.")



# Ensure tables are created before querying

async def post_market_quotes(session, ticker, real_auth):
    url = "https://api.tradier.com/v1/markets/quotes"
    headers = {"Authorization": f"Bearer {real_auth}", "Accept": "application/json"}
    all_contracts = await lookup_all_option_contracts(session, ticker, real_auth)
    # Batching logic
    BATCH_SIZE = 8000
    num_batches = math.ceil(len(all_contracts) / BATCH_SIZE)
    results = []
    timeout = aiohttp.ClientTimeout(total=45)
    for batch_num in range(num_batches):
        start_idx = batch_num * BATCH_SIZE
        end_idx = (batch_num + 1) * BATCH_SIZE
        batch_contracts = all_contracts[start_idx:end_idx]
        symbols_str = ",".join(batch_contracts)

        payload = {"symbols": symbols_str, "greeks": "true"}

        try:
            async with sem:
                async with session.post(url, data=payload, headers=headers, timeout=timeout) as response:
                    response.raise_for_status()  # Raise an exception for HTTP errors
                    data = await response.json()
                    rate_limit_allowed = int(response.headers.get("X-Ratelimit-Allowed", "0"))
                    rate_limit_used = int(response.headers.get("X-Ratelimit-Used", "0"))

                    if rate_limit_used >= (rate_limit_allowed * 0.99):
                        logger.error(
                            f"POST_{url},----Rate limit exceeded: Used {rate_limit_used} out of {rate_limit_allowed}"
                        )

                    if "quotes" in data and "quote" in data["quotes"]:
                        quotes = data["quotes"]["quote"]
                        results.append(pd.DataFrame(quotes))
                    else:
                        logger.error(f"Error: No market quote data found for {ticker} (batch {batch_num}).")
                        raise OptionChainError(f"No market quote data found for {ticker} (batch {batch_num})")
            await asyncio.sleep(0.1)  # Short delay to avoid overwhelming API (adjust as needed)

        # Catch and log the exception with more information
        except (aiohttp.ClientError, OptionChainError) as e:  # Add OptionChainError to the except clause
            logger.exception(f"Error fetching market quotes (batch {batch_num}): {e}")
            raise  # Re-raise the exception

    # Combine results and handle potential empty results
    combined_df = pd.concat(results, ignore_index=True) if results else None

    return combined_df


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
    grouped = all_contract_quotes.groupby("option_type")
    calls_df = grouped.get_group("call").copy()
    puts_df = grouped.get_group("put").copy()
    for df in [calls_df, puts_df]:
        # df['Moneyness'] = np.where(df['option_type'] == 'call', df['strike'] - current_price,
        #                           current_price - df['strike']) #shape was off by 1?
        df["dollarsFromStrike"] = abs(df["strike"] - last_close_price)  # Use last_close_price here
        df["ExpDate"] = pd.to_datetime(df["expiration_date"]).dt.strftime('%Y-%m-%d')
        df["Strike"] = df["strike"]
        df["dollarsFromStrikeXoi"] = df["dollarsFromStrike"] * df["open_interest"]
        # df["MoneynessXoi"] = df["Moneyness"] * df["open_interest"]
        df["lastPriceXoi"] = df["last"] * df["open_interest"]

    # columns_to_drop = ["description", "type", "exch", "underlying", "bidexch", "askexch",
    #                    "expiration_date", "root_symbol"]
    #TODO wait why not use underlying and root symbol etc?
    # calls_df = calls_df.drop(columns_to_drop, axis=1)
    # puts_df = puts_df.drop(columns_to_drop, axis=1)
    return calls_df, puts_df




def convert_unix_to_datetime(unix_timestamp):
    return datetime.fromtimestamp(unix_timestamp / 1000.0)





async def get_options_data(db_session, session, ticker, loop_start_time):
    headers = {"Authorization": f"Bearer {real_auth}", "Accept": "application/json"}
    ticker_quote = await fetch(
        session,
        "https://api.tradier.com/v1/markets/quotes",
        params={"symbols": ticker, "greeks": "false"},
        headers=headers,
    )

    all_contract_quotes = await post_market_quotes(session, ticker, real_auth)

    # Process Stock Quote Data (directly use the quote_df)
    quote_df = pd.DataFrame.from_dict(ticker_quote["quotes"]["quote"], orient="index").T
    current_price = quote_df.at[0, "last"]  # Define current_price here
    prevclose = quote_df.at[0, "prevclose"]  # Define prevclose here


    symbol_result = await db_session.execute(
        select(Symbol).filter_by(symbol_name=ticker)
    )
    symbol = symbol_result.scalars().first()

    # Upsert the stock information
    if not symbol:
        symbol = Symbol(symbol_name=ticker)
        db_session.add(symbol)
        await db_session.flush()  # Flush to get the generated ID


    last_trade_time = datetime.fromtimestamp(quote_df.at[0, "trade_date"] / 1000)
    stock_price_data = {
        'symbol_id': symbol.symbol_id,
        'timestamp': last_trade_time,
        'fetch_timestamp': datetime.utcnow(),
        'last_trade': last_trade_time,
        'last_price': quote_df.at[0, "last"],
        'bid': quote_df.at[0, "bid"],  # Replace with your bid value from quote_df or other source
        'bid_date': datetime.fromtimestamp(quote_df.at[0, "bid_date"] / 1000) if not pd.isnull(quote_df.at[0, "bid_date"]) else None,
        'bidsize': quote_df.at[0, "bidsize"],
        'ask': quote_df.at[0, "ask"],
        'ask_date': datetime.fromtimestamp(quote_df.at[0, "ask_date"] / 1000) if not pd.isnull(quote_df.at[0, "ask_date"]) else None,
        'asksize': quote_df.at[0, "asksize"],
        'open_price': quote_df.at[0, "open"],
        'high_price': quote_df.at[0, "high"],
        'low_price': quote_df.at[0, "low"],
        'close_price': quote_df.at[0, "close"],
        'volume': quote_df.at[0, "volume"],
        'average_volume': quote_df.at[0, "average_volume"]
    }

    await db_session.execute(
        insert(SymbolQuote)
        .values(stock_price_data)
        .on_conflict_do_nothing(
            constraint='symbol_quote_unique_constraint'
    ))

    #TODO look at the above.
    if all_contract_quotes is not None:
        calls_df, puts_df = process_option_quotes(all_contract_quotes, current_price, prevclose)
        # Combine calls_df and puts_df
        options_df = pd.concat([calls_df, puts_df], ignore_index=True)


        options_df['trade_date'] = options_df['trade_date'].apply(convert_unix_to_datetime)
        options_df['ask_date'] = options_df['ask_date'].apply(convert_unix_to_datetime)
        options_df['bid_date'] = options_df['bid_date'].apply(convert_unix_to_datetime)
        options_df['fetch_timestamp'] = datetime.utcnow()
        options_df['ExpDate'] = pd.to_datetime(options_df['ExpDate'])


        option_data = []
        for _, row in options_df.iterrows():
            option_data.append({
                'symbol_id': symbol.symbol_id,
                'expiration_date': row['ExpDate'],
                'strike_price': row['Strike'],
                'option_type': row['option_type'],
                'root_symbol': row['root_symbol'],
                'contract_size': row['contract_size'],
                'description': row['description'],
                'expiration_type': row['expiration_type'],
                'exch': row['exch'],
            })
        OPTION_BATCH_SIZE = calculate_batch_size(option_data)

        # Batch insert Option objects
        # OPTION_BATCH_SIZE = 1000  # Adjust this value as needed
        for i in range(0, len(option_data), OPTION_BATCH_SIZE):
            batch = option_data[i : i + OPTION_BATCH_SIZE]
            await db_session.execute(insert(Option).values(batch).on_conflict_do_nothing(
                constraint='uq_option_constraint'

            ))
        await db_session.flush()


        # Fetch all inserted options to get their IDs
        stmt = select(Option).filter(Option.symbol_id == symbol.symbol_id).options(joinedload(Option.quotes))
        result = await db_session.execute(stmt)

        # Apply the unique() method to de-duplicate results
        unique_options = result.unique()
        options = {(o.root_symbol, o.expiration_date, o.strike_price, o.option_type): o for o in unique_options.scalars()}
        # Prepare OptionQuote data
        option_quotes_data = []
        for _, row in options_df.iterrows():
            option = options.get((row['root_symbol'], row['ExpDate'], row['Strike'], row['option_type']))
            if option:
                option_quotes_data.append({
                    "option_id": option.option_id,
                    "timestamp": row["trade_date"],
                    "fetch_timestamp": loop_start_time,
                    "last": row["last"],
                    "bid": row["bid"],
                    "ask": row["ask"],
                    "volume": row["volume"],
                    "greeks": row["greeks"],
                    "change_percentage": row["change_percentage"],
                    "average_volume": row["average_volume"],
                    "last_volume": row["last_volume"],
                    "trade_date": row["trade_date"],
                    "prevclose": row["prevclose"],
                    "week_52_high": row["week_52_high"],
                    "week_52_low": row["week_52_low"],
                    "bidsize": row["bidsize"],
                    "bidexch": row["bidexch"],
                    "bid_date": row["bid_date"],
                    "asksize": row["asksize"],
                    "askexch": row["askexch"],
                    "ask_date": row["ask_date"],
                    "open_interest": row["open_interest"],
                    "change": row["change"],
                    "open": row["open"],
                    "high": row["high"],
                    "low": row["low"],
                    "close": row["close"],
                })

        # Define the total elements limit
        TOTAL_ELEMENTS_LIMIT = 32680 #TODO i think this is the correct upper limit?

        # Ensure there is at least one element in option_quotes_data to determine the number of fields
        if option_quotes_data:
            # Dynamically determine the number of fields per quote
            fields_per_option_quote = len(option_quotes_data[0])
            print(f"Fields per option quote: {fields_per_option_quote}")

            # Calculate the maximum number of quotes per batch to stay within the limit
            max_quotes_per_batch = TOTAL_ELEMENTS_LIMIT // fields_per_option_quote

        QUOTE_BATCH_SIZE = max_quotes_per_batch
        print(QUOTE_BATCH_SIZE*fields_per_option_quote, QUOTE_BATCH_SIZE)#total elments gotta be less than 32600 or somethin?
        for i in range(0, len(option_quotes_data), QUOTE_BATCH_SIZE):
            batch = option_quotes_data[i : i + QUOTE_BATCH_SIZE]
            await db_session.execute(insert(OptionQuote).values(batch).on_conflict_do_nothing(
                constraint='uq_option_quote_constraint'
            ))

        ta_df = await get_ta(session,ticker)
        # Process technical analysis data
        if not ta_df.empty:
            ta_data_list = ta_df.to_dict(orient='records')
            for data in ta_data_list:
                data["symbol_id"] = symbol.symbol_id

            # Batch insertion for TechnicalAnalysis
            TA_BATCH_SIZE = 1000
            for i in range(0, len(ta_data_list), TA_BATCH_SIZE):
                batch = ta_data_list[i:i+TA_BATCH_SIZE]
                try:
                    await db_session.execute(
                        insert(TechnicalAnalysis).values(batch).on_conflict_do_nothing(
                            constraint='uq_symbol_interval_timestamps'
                        )
                    )
                except SQLAlchemyError as e:
                    print(f"SQLAlchemy error during insertion: {e}")
                    print(traceback.format_exc())
                    await db_session.rollback()
                except Exception as e:
                    print(f"Unexpected error during insertion: {e}")
                    print(traceback.format_exc())
                    await db_session.rollback()
        else:
            print("TA DataFrame is empty")

        await db_session.commit()
        options_df.to_csv("options_df_test.csv")
        return prevclose, current_price,  options_df , symbol

# async def get_required_data(db_session, ticker):
#     # Fetch the latest SymbolQuote for last adjusted close (LAC), current price, and last trade time
#     latest_quote_query = (
#         select(SymbolQuote)
#         .filter(SymbolQuote.symbol_name == ticker)
#         .order_by(SymbolQuote.timestamp.desc())
#     )
#     latest_quote_result = await db_session.execute(latest_quote_query)
#     latest_quote = latest_quote_result.scalars().first()
#
#
#     if latest_quote:
#         last_adj_close = latest_quote['prevclose'] # Use the correct column name
#         current_price = latest_quote.last_price
#         last_trade_time = latest_quote.last_trade.strftime("%Y-%m-%d %H:%M:%S")
#         yymmdd = latest_quote.last_trade.strftime("%y%m%d")  # Format as YYMMDD
#     else:
#         # Handle the case where no quote is found (return defaults or raise an exception)
#         last_adj_close = None  # or some default value
#         current_price = None  # or some default value
#         last_trade_time = None  # or some default value
#         yymmdd = datetime.today().strftime("%y%m%d")
#
#     return last_adj_close, current_price, last_trade_time, yymmdd