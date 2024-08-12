import asyncio
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import ta

from TradierAPI import real_auth
from UTILITIES.logger_config import logger
import tradierAPI_marketdata


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
        time_sale_response = await (tradierAPI_marketdata.fetch
            (
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
        ))

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
                latest_minute_data[f"EMA_{window}"] = df[f"EMA_{window}_{interval}"].iloc[-1]
            for window in rsi_windows:
                safe_calculation(df, f"RSI_{window}_{interval}", ta.momentum.rsi, close=df["close"], window=window)
                latest_minute_data[f"RSI_{window}"] = df[f"RSI_{window}_{interval}"].iloc[-1]


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
                high=df["high"],
                low=df["low"],
                close=df["close"],
                window=keltner_window,
                window_atr=keltner_atr_window
            )
            safe_calculation(df, f"Keltner_Upper", keltner_channel.keltner_channel_hband_indicator)
            safe_calculation(df, f"Keltner_Lower", keltner_channel.keltner_channel_lband_indicator)
            latest_minute_data['Keltner_Upper'] = df[f"Keltner_Upper"].iloc[-1]
            latest_minute_data['Keltner_Lower'] = df[f"Keltner_Lower"].iloc[-1]

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
#TODO need to change the open hoigh low close from quotes to daily.  the timesales has ohlc for the time period. i guess jst store tail1 since im retieving timesals anyway for now.  eventually i will only need curent?
    # Drop columns with all NaN values after concatenation
    # final_df.dropna(axis=0, how='all', inplace=True)

    # print("final df = ", final_df)
    return final_df


