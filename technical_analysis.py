import asyncio
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import ta
from concurrent.futures import ThreadPoolExecutor
import pytz
from UTILITIES.logger_config import logger
async def insert_ta_data(conn, data, interval, symbol_name):
    # Mapping of DataFrame column names to database column names
    column_mapping = {
        'RSI_2': 'rsi_2',
        'RSI_7': 'rsi_7',
        'RSI_14': 'rsi_14',
        'RSI_21': 'rsi_21',
        'EMA_5': 'ema_5',
        'EMA_14': 'ema_14',
        'EMA_20': 'ema_20',
        'EMA_50': 'ema_50',
        'EMA_200': 'ema_200',
        'AwesomeOsc': 'awesome_oscillator',
        'SMA_20': 'sma_20',
        'ADX': 'adx',
        'CCI': 'cci',
        'CMF': 'cmf',
        'EoM': 'eom',
        'OBV': 'obv',
        'MFI': 'mfi',
        'MACD': 'macd',
        'Signal_Line': 'macd_signal',
        'MACD_diff': 'macd_diff',
        'MACD_diff_prev': 'macd_diff_prev',
        'Williams_R': 'williams_r',
        'PVO': 'pvo',
        'PPO': 'ppo',
        'Keltner_Upper': 'keltner_upper',
        'Keltner_Lower': 'keltner_lower',
        'BB_high': 'bb_high',
        'BB_mid': 'bb_mid',
        'BB_low': 'bb_low',
        'VPT': 'vpt',
        'open': 'open',
        'high': 'high',
        'low': 'low',
        'close': 'close',
        'volume': 'volume',
        'vwap': 'vwap'
    }

    # Prepare the data for insertion
    insert_data = {
        'symbol_name': symbol_name,
        'interval': interval,
        'timestamp': data.index[0],
        'fetch_timestamp': data['fetch_timestamp'].iloc[0] #changed this too from fetchdf
    }

    for df_col, db_col in column_mapping.items():
        col_name = f"{df_col}_{interval}"
        if col_name in data.columns:
            insert_data[db_col] = data[col_name].iloc[0]

    # Construct the SQL query
    columns = ', '.join(insert_data.keys())
    placeholders = ', '.join(f'${i+1}' for i in range(len(insert_data)))
    values = list(insert_data.values())

    query = f"""
    INSERT INTO csvimport.technical_analysis ({columns})
    VALUES ({placeholders})
    ON CONFLICT (symbol_name, interval, timestamp) 
    DO UPDATE SET
    """
    query += ', '.join(f"{col} = EXCLUDED.{col}" for col in insert_data.keys() if col not in ['symbol_name', 'interval', 'timestamp'])

    try:
        await conn.execute(query, *values)
        # logger.info(f"Successfully inserted/updated TA data for {symbol_name} at {interval} interval")
    except Exception as e:
        logger.error(f"Error inserting TA data for {symbol_name} at {interval} interval: {str(e)}")
        logger.error(f"Data: {data}")
        logger.error(f"Insert data: {insert_data}")

def calculate_indicators(df_non_negative, interval):
    df_non_negative.to_csv(f'df_non_negative.csv')

    if df_non_negative.empty:
        logger.warning(f"Empty DataFrame passed to calculate_indicators for interval {interval}")
        return pd.DataFrame()

    latest_minute_data = pd.DataFrame(index=[df_non_negative.index[-1]])

    def calculate_indicator(indicator_name, indicator_function, **kwargs):
        try:
            result = indicator_function(**kwargs)
            if isinstance(result, pd.Series):
                return result.iloc[-1]
            elif isinstance(result, pd.DataFrame):
                return result.iloc[-1, 0]  # Assuming the first column contains the indicator values
            elif callable(result):
                # For indicators that return a callable object
                return result().iloc[-1]
            else:
                return result
        except Exception as e:
            logger.warning(f"Error calculating {indicator_name} for interval {interval}: {str(e)}")
            return np.nan

    # EMA calculations
    for window in [5, 14, 20, 50, 200]:
        latest_minute_data[f"EMA_{window}_{interval}"] = calculate_indicator(
            f"EMA_{window}", ta.trend.ema_indicator,
            close=df_non_negative["close"], window=window
        )

    # RSI calculations
    for window in [2, 7, 14, 21]:
        latest_minute_data[f"RSI_{window}_{interval}"] = calculate_indicator(
            f"RSI_{window}", ta.momentum.rsi,
            close=df_non_negative["close"], window=window
        )

    # MACD
    macd = ta.trend.MACD(close=df_non_negative["close"])
    latest_minute_data[f"MACD_{interval}"] = macd.macd().iloc[-1]
    latest_minute_data[f"MACD_signal_{interval}"] = macd.macd_signal().iloc[-1]
    latest_minute_data[f"MACD_diff_{interval}"] = macd.macd_diff().iloc[-1]

    # PPO
    latest_minute_data[f"PPO_{interval}"] = calculate_indicator(
        "PPO", lambda close: ta.momentum.PercentagePriceOscillator(close=close).ppo(),
        close=df_non_negative["close"]
    )

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(close=df_non_negative["close"], window=20, window_dev=2)
    latest_minute_data[f"BB_high_{interval}"] = bb.bollinger_hband().iloc[-1]
    latest_minute_data[f"BB_mid_{interval}"] = bb.bollinger_mavg().iloc[-1]
    latest_minute_data[f"BB_low_{interval}"] = bb.bollinger_lband().iloc[-1]

    def calculate_vpt(close, volume):
        close_pct_change = close.pct_change(fill_method=None)
        vpt = (close_pct_change * volume).cumsum()
        return vpt.iloc[-1]

    # VPT
    latest_minute_data[f"VPT_{interval}"] = calculate_vpt(
        df_non_negative["close"], df_non_negative["volume"]
    )

    # Williams %R
    latest_minute_data[f"Williams_R_{interval}"] = calculate_indicator(
        "Williams %R", lambda high, low, close: ta.momentum.WilliamsRIndicator(high, low, close).williams_r(),
        high=df_non_negative["high"], low=df_non_negative["low"], close=df_non_negative["close"]
    )

    # MFI
    latest_minute_data[f"MFI_{interval}"] = calculate_indicator(
        "MFI", lambda high, low, close, volume: ta.volume.MFIIndicator(high, low, close, volume).money_flow_index(),
        high=df_non_negative["high"], low=df_non_negative["low"],
        close=df_non_negative["close"], volume=df_non_negative["volume"]
    )

    # SMA 20
    latest_minute_data[f"SMA_20_{interval}"] = calculate_indicator(
        "SMA_20", ta.trend.sma_indicator, close=df_non_negative["close"], window=20
    )

    # Awesome Oscillator
    latest_minute_data[f"AwesomeOsc_{interval}"] = calculate_indicator(
        "AwesomeOsc", lambda high, low: ta.momentum.AwesomeOscillatorIndicator(high, low).awesome_oscillator(),
        high=df_non_negative["high"], low=df_non_negative["low"]
    )
    # latest_minute_data[f"AwesomeOsc_{interval}"] = ta.momentum.AwesomeOscillatorIndicator(df_non_negative['high'], df_non_negative['low']).awesome_oscillator()

    # ADX
    latest_minute_data[f"ADX_{interval}"] = calculate_indicator(
        "ADX", lambda high, low, close: ta.trend.ADXIndicator(high, low, close).adx(),
        high=df_non_negative["high"], low=df_non_negative["low"], close=df_non_negative["close"]
    )

    # CCI
    latest_minute_data[f"CCI_{interval}"] = calculate_indicator(
        "CCI", lambda high, low, close: ta.trend.CCIIndicator(high, low, close).cci(),
        high=df_non_negative["high"], low=df_non_negative["low"], close=df_non_negative["close"]
    )

    # CMF
    latest_minute_data[f"CMF_{interval}"] = calculate_indicator(
        "CMF", lambda high, low, close, volume: ta.volume.ChaikinMoneyFlowIndicator(high, low, close, volume).chaikin_money_flow(),
        high=df_non_negative["high"], low=df_non_negative["low"], close=df_non_negative["close"],
        volume=df_non_negative["volume"]
    )

    # EoM
    latest_minute_data[f"EoM_{interval}"] = calculate_indicator(
        "EoM", lambda high, low, volume: ta.volume.EaseOfMovementIndicator(high, low, volume).ease_of_movement(),
        high=df_non_negative["high"], low=df_non_negative["low"], volume=df_non_negative["volume"]
    )

    # OBV
    latest_minute_data[f"OBV_{interval}"] = calculate_indicator(
        "OBV", lambda close, volume: ta.volume.OnBalanceVolumeIndicator(close, volume).on_balance_volume(),
        close=df_non_negative["close"], volume=df_non_negative["volume"]
    )

    # PVO
    latest_minute_data[f"PVO_{interval}"] = calculate_indicator(
        "PVO", lambda volume: ta.momentum.PercentageVolumeOscillator(volume).pvo(),
        volume=df_non_negative["volume"]
    )

    # Keltner Channel
    keltner = ta.volatility.KeltnerChannel(
        high=df_non_negative["high"], low=df_non_negative["low"],
        close=df_non_negative["close"], window=20, window_atr=10
    )
    latest_minute_data[f"Keltner_Upper_{interval}"] = keltner.keltner_channel_hband().iloc[-1]
    latest_minute_data[f"Keltner_Lower_{interval}"] = keltner.keltner_channel_lband().iloc[-1]

    # Add OHLCV data
    for col in ['open', 'high', 'low', 'close', 'volume', 'vwap']:
        latest_minute_data[f"{col}_{interval}"] = df_non_negative[col].iloc[-1]

    # Remove any columns with all NaN values
    # latest_minute_data = latest_minute_data.dropna(axis=1, how='all')
    # latest_minute_data = latest_minute_data.tail(1).reset_index(drop=False)
    # print(latest_minute_data.columns)
    return latest_minute_data
async def get_ta(conn, ticker):
    days_to_fetch = 40  # Fetch enough data for the longest interval
    eastern = pytz.timezone('US/Eastern')
    current_time = datetime.now(eastern)
    start_date = current_time - timedelta(days=days_to_fetch)

    async def fetch_data_from_db(conn):
        query = """
        SELECT 
            fetch_timestamp,
            last_1min_timestamp,
            last_1min_open as open,
            last_1min_high as high,
            last_1min_low as low,
            last_1min_close as close,
            last_1min_volume as volume,
            last_1min_vwap as vwap
        FROM 
            csvimport.symbol_quotes
        WHERE 
            symbol_name = $1 
            AND fetch_timestamp BETWEEN $2 AND $3
        ORDER BY 
            last_1min_timestamp ASC
        """

        rows = await conn.fetch(query, ticker, start_date, current_time)

        df = pd.DataFrame(rows, columns=['fetch_timestamp', 'last_1min_timestamp', 'open', 'high', 'low', 'close', 'volume', 'vwap'])
        df.set_index('last_1min_timestamp', inplace=True)  #Sets 1st instance, discard recurrences
        df.index = pd.to_datetime(df.index)
        return df

    def process_data(df, interval):
        if df.empty:
            logger.warning(f"Empty DataFrame for ticker at interval {interval}")
            return pd.DataFrame()


        """Example of how i may use rolling time based windows which smartly includes pror day:import pandas as pd
import numpy as np
from pandas.tseries.offsets import CustomBusinessDay
from pandas.tseries.holiday import USFederalHolidayCalendar

def process_data(df, interval):
    if df.empty:
        logger.warning(f"Empty DataFrame for ticker at interval {interval}")
        return pd.DataFrame()

    # Define custom business day
    business_day = CustomBusinessDay(calendar=USFederalHolidayCalendar())

    # Ensure the index is sorted
    df = df.sort_index()

    # Forward fill the last known values for overnight periods
    df = df.resample('1min').ffill()

    # Create a continuous time index including only market hours
    full_index = pd.date_range(start=df.index[0].floor('D'),
                               end=df.index[-1].ceil('D'),
                               freq='1min')
    market_hours = ((full_index.time >= pd.Timestamp('9:30').time()) &
                    (full_index.time < pd.Timestamp('16:00').time()))
    market_index = full_index[market_hours]

    # Reindex the dataframe to include all market minutes
    df = df.reindex(market_index)

    # Convert interval string to minutes
    if interval.endswith('min'):
        window = pd.Timedelta(minutes=int(interval[:-3]))
    elif interval.endswith('H'):
        window = pd.Timedelta(hours=int(interval[:-1]))
    else:
        raise ValueError(f"Unsupported interval format: {interval}")

    # Apply rolling window
    rolled_df = df.rolling(window=window, closed='right').agg({
        'fetch_timestamp': 'last',
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'vwap': lambda x: np.average(x, weights=df.loc[x.index, 'volume'])
    })

    return rolled_df

intervals = ["1min", "5min", "15min", "1H"]
processed_results = []

for interval in intervals:
    result = process_data(df, interval)
    if not result.empty:
        processed_results.append(result)
    else:
        logger.warning(f"Empty processed result for interval {interval}")"""
        try:
            # Resample data to the desired interval
            resampled_df = df.resample(interval, closed='right', label='left').agg({
            #changed to rolling windows:
            # resampled_df =  df.rolling(window=interval, closed='right').agg({

                    # 'last_1min_timestamp': 'last', #changesd from fetch
                'fetch_timestamp': 'last',  # changesd from fetch

                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'vwap': 'mean'
            })

            if resampled_df.empty:
                logger.warning(f"Empty resampled DataFrame for ticker at interval {interval}")
                return pd.DataFrame()

            # Clip numeric columns
            numeric_cols = resampled_df.select_dtypes(include=['number']).columns
            df_non_negative = resampled_df.copy()
            df_non_negative[numeric_cols] = df_non_negative[numeric_cols] #.clip(lower=0) #TODO NOT actually non-negaTIVE
            # Calculate indicators
            latest_minute_data = calculate_indicators(df_non_negative, interval)
            latest_minute_data.to_csv(f'{interval}df_nonnegative.csv')

            # Add OHLCV data
            for col in ['open', 'high', 'low', 'close', 'volume', 'vwap']:
                latest_minute_data[f"{col}_{interval}"] = resampled_df[col].iloc[-1]

            latest_minute_data['fetch_timestamp'] = resampled_df['fetch_timestamp'].iloc[-1]
            # latest_minute_data['timestamp'] = resampled_df.index[-1]

            return latest_minute_data

        except Exception as e:
            # logger.error(f"Error in process_data for interval {interval}: {str(e)}", exc_info=True)
            return pd.DataFrame()


    try:
        # Fetch data once
        df = await fetch_data_from_db(conn)

        if df.empty:
            logger.warning(f"No data fetched from database for {ticker}")
            return pd.DataFrame()

        intervals = ["1min", "5min", "15min"]
        processed_results = []

        for interval in intervals:
            # print(df.columns,df.index)
            result = process_data(df, interval)
            if not result.empty:
                processed_results.append(result)
                # logger.info(f"Processed data for {ticker} at {interval} interval. Shape: {result.shape}")
            else:
                logger.warning(f"Empty processed result for {ticker} at interval {interval}")

        if not processed_results:
            logger.warning(f"No valid processed results for {ticker}")
            return pd.DataFrame()

        # Ensure all processed results have the same index
        common_index = processed_results[0].index
        aligned_results = [result.reindex(common_index) for result in processed_results]

        final_df = pd.concat(aligned_results, axis=1)
        final_df['fetch_timestamp'] = current_time

        if final_df.empty:
            logger.warning(f"Empty final DataFrame for {ticker}")
            return pd.DataFrame()

        # logger.info(f"Final DataFrame shape for {ticker}: {final_df.shape}")
        # logger.info(f"Final DataFrame columns: {final_df.columns}")

        # Save to CSV file
        # try:
        #     final_df.to_csv(f'final_data_{ticker}.csv')
        #     logger.info(f"Successfully saved CSV for {ticker}")
        # except Exception as csv_error:
        #     logger.error(f"Error saving CSV for {ticker}: {str(csv_error)}")

        # Insert TA data into the database

        for interval, data in zip(intervals, processed_results):
            if not data.empty:
                await insert_ta_data(conn, data, interval,ticker)
            else:
                logger.warning(f"Skipping database insertion for {ticker} at interval {interval} due to empty data")

        logger.info(f"Successfully processed and inserted TA data for {ticker}")

        return final_df

    except Exception as e:
        logger.error(f"Error in get_ta for {ticker}: {str(e)}", exc_info=True)
        raise
