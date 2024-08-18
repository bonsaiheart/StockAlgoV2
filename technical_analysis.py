import asyncio
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import ta
from concurrent.futures import ThreadPoolExecutor
import pytz
from TradierAPI import real_auth
from UTILITIES.logger_config import logger
import tradierAPI_marketdata

async def get_ta(session, ticker):
    days_to_fetch = {
        "1min": 5,
        "5min": 20,
        "15min": 40,
    }

    async def fetch_data(interval):
        start = (datetime.today() - timedelta(days=days_to_fetch[interval])).strftime("%Y-%m-%d %H:%M")
        end = datetime.today().strftime("%Y-%m-%d %H:%M")

        headers = {"Authorization": f"Bearer {real_auth}", "Accept": "application/json"}
        time_sale_response = await tradierAPI_marketdata.fetch(
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
        # print(df.columns)
        return df

    def process_data(df, interval):
        if df.empty:
            logger.warning(f"Empty DataFrame for {ticker} at interval {interval}")
            return pd.DataFrame()

        latest_minute_data = df.tail(1).copy()

        # Ensure timestamp is in datetime format
        if 'timestamp' in latest_minute_data.columns:
            latest_minute_data['timestamp'] = pd.to_datetime(latest_minute_data['timestamp'])

        # Rename OHLCV and VWAP columns to include interval
        for col in ['timestamp', 'price', 'open', 'high', 'low', 'close', 'volume', 'vwap']:
            if col in latest_minute_data.columns:
                new_col_name = f"{col}_{interval}"
                latest_minute_data[new_col_name] = latest_minute_data[col]
                if col == 'timestamp':
                    latest_minute_data[new_col_name] = latest_minute_data[new_col_name].dt.tz_localize(None)
                latest_minute_data.drop(columns=[col], inplace=True)

        # Ensure non-negative values for calculations
        df_non_negative = df.clip(lower=0)
        def calculate_indicator(indicator_name, indicator_function, **kwargs):
            try:
                result = indicator_function(**kwargs)
                if isinstance(result, pd.Series):
                    return result.iloc[-1]
                elif hasattr(result, indicator_name.lower()):
                    return getattr(result, indicator_name.lower())().iloc[-1]
                else:
                    return result
            except Exception as e:
                logger.warning(f"Error calculating {indicator_name} for {ticker} at interval {interval}: {str(e)}")
                return np.nan

        with ThreadPoolExecutor() as executor:
            # EMA calculations
            ema_tasks = {f"EMA_{window}_{interval}": executor.submit(
                calculate_indicator, f"EMA_{window}", ta.trend.ema_indicator,
                close=df_non_negative["close"], window=window
            ) for window in [5, 14, 20, 50, 200]}

            # RSI calculations
            rsi_tasks = {f"RSI_{window}_{interval}": executor.submit(
                calculate_indicator, f"RSI_{window}", ta.momentum.rsi,
                close=df_non_negative["close"], window=window
            ) for window in [2, 7, 14, 21]}

            # Other indicator calculations
            other_tasks = {
                f"AwesomeOsc_{interval}": executor.submit(
                    calculate_indicator, "AwesomeOsc", ta.momentum.awesome_oscillator,
                    high=df_non_negative["high"], low=df_non_negative["low"], window1=5, window2=34
                ),
                f"SMA_20_{interval}": executor.submit(
                    calculate_indicator, "SMA_20", ta.trend.sma_indicator,
                    close=df_non_negative["close"], window=20
                ),
                f"ADX_{interval}": executor.submit(
                    calculate_indicator, "ADX", ta.trend.adx,
                    high=df_non_negative["high"], low=df_non_negative["low"], close=df_non_negative["close"], window=7
                ),
                f"CCI_{interval}": executor.submit(
                    calculate_indicator, "CCI", ta.trend.cci,
                    high=df_non_negative["high"], low=df_non_negative["low"], close=df_non_negative["close"], window=20
                ),
                f"CMF_{interval}": executor.submit(
                    calculate_indicator, "CMF", ta.volume.chaikin_money_flow,
                    high=df_non_negative["high"], low=df_non_negative["low"], close=df_non_negative["close"], volume=df_non_negative["volume"], window=20
                ),
                f"EoM_{interval}": executor.submit(
                    calculate_indicator, "EoM", ta.volume.ease_of_movement,
                    high=df_non_negative["high"], low=df_non_negative["low"], volume=df_non_negative["volume"], window=14
                ),
                f"OBV_{interval}": executor.submit(
                    calculate_indicator, "OBV", ta.volume.on_balance_volume,
                    close=df_non_negative["close"], volume=df_non_negative["volume"]
                ),
                f"MFI_{interval}": executor.submit(
                    calculate_indicator, "MFI", ta.volume.money_flow_index,
                    high=df_non_negative["high"], low=df_non_negative["low"], close=df_non_negative["close"], volume=df_non_negative["volume"], window=14
                )
            }

            # Combine all tasks
            all_tasks = {**ema_tasks, **rsi_tasks, **other_tasks}

            # Wait for all tasks to complete and add results to latest_minute_data
            for name, future in all_tasks.items():
                latest_minute_data[name] = future.result()

        # MACD calculation
        macd_object = ta.trend.MACD(close=df_non_negative["close"], window_slow=26, window_fast=12, window_sign=9, fillna=False)
        latest_minute_data[f"MACD_12_26_{interval}"] = macd_object.macd().iloc[-1]
        latest_minute_data[f"Signal_Line_12_26_{interval}"] = macd_object.macd_signal().iloc[-1]
        latest_minute_data[f"MACD_diff_12_26_{interval}"] = macd_object.macd_diff().iloc[-1]
        latest_minute_data[f"MACD_diff_prev_12_26_{interval}"] = macd_object.macd_diff().shift(1).iloc[-1]
        #TODO missing the bullish/bearish macd cross signal
        # Other indicators
        latest_minute_data[f"Williams_R_{interval}"] = calculate_indicator(
            "Williams_R", ta.momentum.WilliamsRIndicator,
            high=df_non_negative["high"], low=df_non_negative["low"], close=df_non_negative["close"], lbp=14
        )
        latest_minute_data[f"PVO_{interval}"] = calculate_indicator(
            "PVO", ta.momentum.PercentageVolumeOscillator,
            volume=df_non_negative["volume"], window_slow=26, window_fast=12, window_sign=9
        )
        latest_minute_data[f"PPO_{interval}"] = calculate_indicator(
            "PPO", ta.momentum.PercentagePriceOscillator,
            close=df_non_negative["close"], window_slow=26, window_fast=12, window_sign=9
        )
        keltner = ta.volatility.KeltnerChannel(
            high=df_non_negative["high"], low=df_non_negative["low"],
            close=df_non_negative["close"], window=20, window_atr=10
        )
        latest_minute_data[f"Keltner_Upper_{interval}"] = keltner.keltner_channel_hband().iloc[-1]
        latest_minute_data[f"Keltner_Lower_{interval}"] = keltner.keltner_channel_lband().iloc[-1]

        bb_object = ta.volatility.BollingerBands(close=df_non_negative["close"], window=20, window_dev=2)
        latest_minute_data[f"BB_high_20_{interval}"] = bb_object.bollinger_hband().iloc[-1]
        latest_minute_data[f"BB_mid_20_{interval}"] = bb_object.bollinger_mavg().iloc[-1]
        latest_minute_data[f"BB_low_20_{interval}"] = bb_object.bollinger_lband().iloc[-1]

        # VPT calculation
        vpt_indicator = ta.volume.VolumePriceTrendIndicator(
            close=df_non_negative["close"], volume=df_non_negative["volume"]
        )
        latest_minute_data[f"VPT_{interval}"] = vpt_indicator.volume_price_trend().iloc[-1]

        # latest_minute_data[f"timestamp_{interval}"] = latest_minute_data.index
        return latest_minute_data.reset_index(drop=True)
    intervals = ["1min", "5min", "15min"]
    df_results = await asyncio.gather(*[fetch_data(interval) for interval in intervals])

    processed_results = [process_data(df, interval) for df, interval in zip(df_results, intervals)]

    final_df = pd.concat(processed_results, axis=1)
    # Get your local timezone
    # Get the Eastern timezone (handles DST automatically)

    eastern = pytz.timezone('US/Eastern')

    # Get the current time in your local timezone
    local_time = datetime.now(eastern)
    final_df['fetch_timestamp'] = local_time

    # for x in final_df.columns:
    #     print(x)

    return final_df