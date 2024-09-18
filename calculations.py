import numpy as np
import pandas as pd
import pytz

from UTILITIES.logger_config import logger
import json

def calculate_option_metrics(df, current_price, last_adj_close, current_time, symbol_name):
    # Basic calculations
    df['moneyness'] = np.where(df['option_type'] == 'call',
                               current_price / df['strike'],
                               df['strike'] / current_price)
    df['time_to_expiry'] = (df['expiration_date'] - current_time) / np.timedelta64(1, 'Y')

    # IV and Greeks (assuming they're in the 'greeks' column)
    for greek in ['iv', 'delta', 'gamma', 'theta', 'vega']:
        df[greek] = df['greeks'].apply(lambda x: x.get(greek, np.nan) if isinstance(x, dict) else np.nan)

    # ITM filters
    df['itm'] = ((df['option_type'] == 'call') & (df['strike'] <= current_price)) | \
                ((df['option_type'] == 'put') & (df['strike'] >= current_price))

    # Aggregate calculations
    agg = df.groupby('expiration_date').agg({
        'volume': 'sum',
        'open_interest': 'sum',
        'iv': 'mean',
        'delta': 'mean',
        'gamma': 'mean',
        'theta': 'mean',
        'vega': 'mean',
    }).reset_index()
    # Rename 'volume' to 'total_volume' to match the database column name
    agg = agg.rename(columns={'volume': 'total_volume'})
    # Calculate PCR and other metrics
    for exp_date, group in df.groupby('expiration_date'):
        calls = group[group['option_type'] == 'call']
        puts = group[group['option_type'] == 'put']
        itm_calls = calls[calls['itm']]
        itm_puts = puts[puts['itm']]

        total_put_vol = puts['volume'].sum()
        total_call_vol = calls['volume'].sum()
        total_put_oi = puts['open_interest'].sum()
        total_call_oi = calls['open_interest'].sum()
        itm_put_vol = itm_puts['volume'].sum()
        itm_call_vol = itm_calls['volume'].sum()
        itm_put_oi = itm_puts['open_interest'].sum()
        itm_call_oi = itm_calls['open_interest'].sum()

        agg.loc[agg['expiration_date'] == exp_date, 'pcr_vol'] = np.divide(total_put_vol, total_call_vol, where=total_call_vol!=0)
        agg.loc[agg['expiration_date'] == exp_date, 'pcr_oi'] = np.divide(total_put_oi, total_call_oi, where=total_call_oi!=0)
        agg.loc[agg['expiration_date'] == exp_date, 'itm_pcr_vol'] = np.divide(itm_put_vol, itm_call_vol, where=itm_call_vol!=0)
        agg.loc[agg['expiration_date'] == exp_date, 'itm_pcr_oi'] = np.divide(itm_put_oi, itm_call_oi, where=itm_call_oi!=0)
        agg.loc[agg['expiration_date'] == exp_date, 'itm_oi'] = itm_call_oi + itm_put_oi
        agg.loc[agg['expiration_date'] == exp_date, 'total_oi'] = total_call_oi + total_put_oi
        agg.loc[agg['expiration_date'] == exp_date, 'itm_contracts_percent'] = np.divide(itm_call_oi + itm_put_oi, total_call_oi + total_put_oi, where=total_call_oi + total_put_oi!=0)
        agg.loc[agg['expiration_date'] == exp_date, 'avg_call_iv'] = calls['iv'].mean()
        agg.loc[agg['expiration_date'] == exp_date, 'avg_put_iv'] = puts['iv'].mean()
        agg.loc[agg['expiration_date'] == exp_date, 'net_iv'] = calls['iv'].sum() - puts['iv'].sum()
        agg.loc[agg['expiration_date'] == exp_date, 'net_itm_iv'] = itm_calls['iv'].sum() - itm_puts['iv'].sum()

        # Max Pain calculation
        def pain(strike):
            call_pain = (strike - current_price) * calls[calls['strike'] <= strike]['open_interest'].sum()
            put_pain = (current_price - strike) * puts[puts['strike'] >= strike]['open_interest'].sum()
            return call_pain + put_pain

        unique_strikes = group['strike'].unique()
        max_pain = min(unique_strikes, key=pain)
        agg.loc[agg['expiration_date'] == exp_date, 'max_pain'] = max_pain

        # Bonsai Ratio calculation
        bonsai_ratio = np.divide(
            np.multiply(np.divide(itm_put_vol, total_put_vol, where=total_put_vol!=0),
                        np.divide(itm_put_oi, total_put_oi, where=total_put_oi!=0)),
            np.multiply(np.divide(itm_call_vol, total_call_vol, where=total_call_vol!=0),
                        np.divide(itm_call_oi, total_call_oi, where=total_call_oi!=0)),
            where=np.multiply(np.divide(itm_call_vol, total_call_vol, where=total_call_vol!=0),
                              np.divide(itm_call_oi, total_call_oi, where=total_call_oi!=0)) != 0
        )
        agg.loc[agg['expiration_date'] == exp_date, 'bonsai_ratio'] = bonsai_ratio

        # Closest strike to current price
        agg.loc[agg['expiration_date'] == exp_date, 'closest_strike_to_cp'] = group.loc[abs(group['strike'] - current_price).idxmin(), 'strike']

        # At-the-money IV
        atm_option = group.loc[abs(group['strike'] - current_price).idxmin()]
        agg.loc[agg['expiration_date'] == exp_date, 'atm_iv'] = atm_option['iv']

        # Additional metrics (stored in JSON)
        additional_metrics = {
            'pcrv_up1': calculate_pcr(puts, calls, current_price, 1, 'volume'),
            'pcrv_up2': calculate_pcr(puts, calls, current_price, 2, 'volume'),
            'pcrv_down1': calculate_pcr(puts, calls, current_price, -1, 'volume'),
            'pcrv_down2': calculate_pcr(puts, calls, current_price, -2, 'volume'),
            'pcroi_up1': calculate_pcr(puts, calls, current_price, 1, 'open_interest'),
            'pcroi_up2': calculate_pcr(puts, calls, current_price, 2, 'open_interest'),
            'pcroi_down1': calculate_pcr(puts, calls, current_price, -1, 'open_interest'),
            'pcroi_down2': calculate_pcr(puts, calls, current_price, -2, 'open_interest'),
            # Add more metrics as needed
        }
        agg.loc[agg['expiration_date'] == exp_date, 'additional_metrics'] = json.dumps(additional_metrics)

    agg['symbol_name'] = symbol_name
    agg['fetch_timestamp'] = pd.Timestamp(current_time, tz=pytz.UTC)
    agg['current_stock_price'] = current_price
    agg['current_sp_change_lac'] = (current_price - last_adj_close) / last_adj_close * 100

    # # Ensure exp_date is timezone-aware
    agg['exp_date'] = pd.to_datetime(agg['expiration_date']).dt.tz_localize(pytz.UTC)

    return agg

def calculate_pcr(puts, calls, current_price, offset, column):
    put_value = puts[puts['strike'] >= current_price + offset][column].sum()
    call_value = calls[calls['strike'] <= current_price + offset][column].sum()
    return np.divide(put_value, call_value, where=call_value!=0)

def calculate_pcr(puts, calls, current_price, offset, column):
    put_value = puts[puts['strike'] >= current_price + offset]['volume'].sum()
    call_value = calls[calls['strike'] <= current_price + offset]['volume'].sum()
    return put_value / call_value if call_value != 0 else np.inf


async def insert_processed_data(conn, data):
    # Ensure data is a DataFrame
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame([data])

    all_columns = [
        'symbol_name', 'fetch_timestamp', 'exp_date', 'current_stock_price',
        'current_sp_change_lac', 'max_pain', 'bonsai_ratio', 'pcr_vol', 'pcr_oi',
        'itm_pcr_vol', 'itm_pcr_oi', 'itm_oi', 'total_oi', 'itm_contracts_percent',
        'avg_call_iv', 'avg_put_iv', 'net_iv', 'net_itm_iv', 'avg_delta', 'avg_gamma',
        'avg_theta', 'avg_vega', 'total_volume', 'closest_strike_to_cp', 'atm_iv', 'additional_metrics'
    ]


    # Ensure all columns exist in the DataFrame, fill with None if missing
    for col in all_columns:
        if col not in data.columns:
            data[col] = None

    # Filter columns that are actually in the data
    columns = [col for col in all_columns if col in data.columns]
    #
    # # # Convert naive timestamps to timezone-aware timestamps
    # if 'fetch_timestamp' in data.columns:
    #     data['fetch_timestamp'] = data['fetch_timestamp'].dt.tz_localize(pytz.UTC)
    # if 'exp_date' in data.columns:
    #     data['exp_date'] = data['exp_date'].dt.tz_localize(pytz.UTC)


    # Prepare values
    values = [tuple(row) for _, row in data[columns].iterrows()]

    # Prepare the query
    placeholders = ', '.join(f'${i+1}' for i in range(len(columns)))
    query = f"""
    INSERT INTO csvimport.optimized_processed_option_data (
        {', '.join(columns)}
    ) VALUES ({placeholders})
    ON CONFLICT (symbol_name, fetch_timestamp, exp_date) 
    DO UPDATE SET
        {', '.join([f"{col} = EXCLUDED.{col}" for col in columns if col not in ['symbol_name', 'fetch_timestamp', 'exp_date']])}
    """

    try:
        await conn.executemany(query, values)
    except Exception as e:
        logger.error(f"Error inserting processed data: {str(e)}", exc_info=True)
        raise

async def perform_operations(db_pool, ticker, last_adj_close, current_price, current_time, symbol_name):
    async with db_pool.acquire() as conn:
        try:
            # Fetch the latest option data from the database
            option_query = """
            WITH latest_quotes AS (
                SELECT contract_id, MAX(fetch_timestamp) as max_fetch_timestamp
                FROM csvimport.option_quotes
                WHERE fetch_timestamp <= $2
                GROUP BY contract_id
            )
            SELECT o.expiration_date, o.option_type, o.strike, 
                   oq.volume, oq.open_interest, oq.greeks, oq.last,
                   oq.fetch_timestamp
            FROM csvimport.options o
            JOIN latest_quotes lq ON o.contract_id = lq.contract_id
            JOIN csvimport.option_quotes oq ON o.contract_id = oq.contract_id
                AND oq.fetch_timestamp = lq.max_fetch_timestamp
            WHERE o.underlying = $1
            """
            option_data = await conn.fetch(option_query, ticker, current_time)

            if not option_data:
                logger.warning(f"No option data found for {ticker} at or before {current_time}")
                return None, ticker

            df = pd.DataFrame(option_data, columns=['expiration_date', 'option_type', 'strike',
                                                    'volume', 'open_interest', 'greeks', 'last',
                                                    'fetch_timestamp'])

            # Calculate option metrics
            processed_data = calculate_option_metrics(df, current_price, last_adj_close,current_time,symbol_name)

            # Add symbol_name and fetch_timestamp
            processed_data['symbol_name'] = symbol_name
            processed_data['fetch_timestamp'] = current_time

            # Insert processed data into the database
            await insert_processed_data(conn, processed_data)

            return df, processed_data, processed_data.to_dict('records')

        except Exception as e:
            logger.error(f"Error in perform_operations for {ticker}: {str(e)}", exc_info=True)
            return None, ticker