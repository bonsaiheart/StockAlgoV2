from functools import lru_cache
import psycopg2.extras
#TODO make the db populate null instead of NaN
from psycopg2.extras import execute_batch
from sqlalchemy import and_
import PrivateData.tradier_info
import math
import time
import fredapi
import numpy as np
import ta
from sqlalchemy import inspect, UniqueConstraint, PrimaryKeyConstraint, TIMESTAMP
import PrivateData.tradier_info
import asyncio
from datetime import datetime, timedelta, timezone
import aiohttp
from sqlalchemy.exc import OperationalError

import pandas as pd
from sqlalchemy.dialects.postgresql import insert
from pangres import upsert
from datetime import datetime
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from main_devmode import TICKERS_FOR_TRADE_ALGOS
import technical_analysis
from UTILITIES.logger_config import logger
from sqlalchemy import func,Column, Integer, String, Float, DateTime, Date, ForeignKey, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.ext.declarative import declarative_base

import pytz
from scipy.stats import norm
from db_schema_models import Symbol,SymbolQuote, Option, OptionQuote, Dividend
eastern = pytz.timezone('US/Eastern')


class OptionChainError(Exception):
    pass


from datetime import datetime, timedelta
import asyncio
from sqlalchemy.engine import Connection

import json
from database_operations import create_schema_and_tables
from db_schema_models import Base, OptionQuote

#
# def ensure_tables_exist(engine):
#     inspector = inspect(engine)
#     if 'csvimport' not in inspector.get_schema_names():
#         create_schema_and_tables(engine)
#     else:
#         if 'option_quotes' not in inspector.get_table_names(schema='csvimport'):
#             Base.metadata.create_all(engine)
#
# # Call this function before any database operations
# ensure_tables_exist(engine)
import numpy as np
from scipy.stats import norm

import numpy as np
from scipy.stats import norm
import pandas as pd
import warnings

#
# def black_scholes_call(S, K, T, r, sigma):
#     d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
#     d2 = d1 - sigma * np.sqrt(T)
#     return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
#
#
# def black_scholes_put(S, K, T, r, sigma):
#     d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
#     d2 = d1 - sigma * np.sqrt(T)
#     return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
# def calculate_implied_volatility_and_greeks(option_price, S, K, T, r, option_type, q=0, precision=1e-4, max_iterations=120):
#     """
#     Calculates implied volatility and option Greeks in a single function.
#     """
#
#     if option_price <= 0 or S <= 0 or K <= 0 or T <= 0:
#         return None, None
#
#     sigma = 0.4  # Initial guess
#     for i in range(max_iterations):
#         try:
#             if option_type == 'call':
#                 price = black_scholes_call(S, K, T, r, sigma)
#             else:
#                 price = black_scholes_put(S, K, T, r, sigma)
#
#             d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
#             vega = S * np.sqrt(T) * norm.pdf(d1)
#
#             if abs(vega) < 1e-10:
#                 return None, None
#
#             diff = option_price - price
#
#             if abs(diff) < precision:
#                 # Calculate Greeks directly here using the converged sigma
#                 greeks = calculate_option_greeks(S, K, T, r, sigma, option_type, q)
#                 return sigma, greeks
#
#             sigma = sigma + diff / vega
#
#             if sigma <= 0:
#                 return None, None
#
#         except (OverflowError, ZeroDivisionError, ValueError):
#             return None, None
#
#     return None, None
# # def calculate_implied_volatility(option_price, S, K, T, r, option_type, precision=1e-2, max_iterations=100):
# #     if option_price <= 0 or S <= 0 or K <= 0 or T <= 0:
# #         print(f"Invalid input values: option_price={option_price}, S={S}, K={K}, T={T}")
# #         return None
# #
# #     # Initial guess and boundaries
# #     lower_bound = 0.01  # Minimum realistic volatility
# #     upper_bound = 3.0  # Maximum realistic volatility (300%)
# #     sigma = 0.5  # Initial guess within the bounds
# #
# #     previous_sigma = None  # Track the previous sigma for oscillation detection
# #
# #     for i in range(max_iterations):
# #         try:
# #             if option_type == 'call':
# #                 price = black_scholes_call(S, K, T, r, sigma)
# #             else:
# #                 price = black_scholes_put(S, K, T, r, sigma)
# #             # Calculate d1 here
# #             d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
# #
# #             vega = S * np.sqrt(T) * norm.pdf(d1)
# #
# #             if abs(vega) < 1e-10:
# #                 print("Vega is too small, potential convergence issue")
# #                 return None
# #
# #             diff = option_price - price
# #
# #             if abs(diff) < precision:
# #                 return sigma
# #
# #             # Bisection method if Newton-Raphson leads to out-of-bounds sigma
# #             new_sigma = sigma + diff / vega
# #             if new_sigma <= lower_bound or new_sigma >= upper_bound:
# #                 if option_price > price:  # Option is overpriced, increase volatility
# #                     lower_bound = sigma
# #                 else:  # Option is underpriced, decrease volatility
# #                     upper_bound = sigma
# #                 sigma = (lower_bound + upper_bound) / 2
# #             else:
# #                 sigma = new_sigma
# #
# #             # Oscillation detection
# #             if previous_sigma is not None and abs(sigma - previous_sigma) < 1e-4:
# #                 print("Oscillation detected, potential convergence issue")
# #                 return None
# #             previous_sigma = sigma
# #
# #         except (OverflowError, ZeroDivisionError, ValueError) as e:
# #             print(f"Exception during calculation: {e}")
# #             return None
# #
# #     print("Failed to converge within max iterations")
# #     return None
# def calculate_implied_volatility(option_price, S, K, T, r, option_type, precision=1e-4, max_iterations=120):
#     if option_price <= 0 or S <= 0 or K <= 0 or T <= 0:
#         # print(option_price, S, K, T)
#         return None
#
#     sigma = 0.4  # Initial guess
#     for i in range(max_iterations):
#         try:
#             if option_type == 'call':
#                 price = black_scholes_call(S, K, T, r, sigma)
#             else:
#                 price = black_scholes_put(S, K, T, r, sigma)
#
#             d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
#             vega = S * np.sqrt(T) * norm.pdf(d1)
#
#             if abs(vega) < 1e-10:  # Avoid division by zero
#                 return None
#
#             diff = option_price - price
#
#             if abs(diff) < precision:
#                 return sigma
#
#             sigma = sigma + diff / vega
#
#             if sigma <= 0:
#
#                 return None
#         except (OverflowError, ZeroDivisionError, ValueError):
#             return None
#
#     return None  # Failed to converge

def vectorized_black_scholes(S, K, T, r, sigma, option_types):
    valid_mask = (S > 0) & (K > 0) & (T > 0) & (sigma > 0)
    d1 = np.full_like(S, np.nan)
    d2 = np.full_like(S, np.nan)

    np.divide(np.log(S / K) + (r + 0.5 * sigma ** 2) * T, sigma * np.sqrt(T), out=d1, where=valid_mask)
    d2 = d1 - sigma * np.sqrt(T)

    call_prices = np.where(valid_mask,
                           S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2),
                           np.nan)
    put_prices = np.where(valid_mask,
                          K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1),
                          np.nan)

    return np.where(option_types == 'call', call_prices, put_prices)


def adaptive_initial_guess(S, K, T):
    moneyness = np.log(S / K)
    atm_vol = 0.3  # At-the-money volatility estimate
    return np.where(T < 30 / 365, atm_vol * (1 + 0.1 * np.abs(moneyness)),
                    atm_vol * (1 + 0.05 * np.abs(moneyness)))


def vectorized_implied_volatility_improved(option_prices, S, K, T, r, option_types, precision=1e-5, max_iterations=100):
    sigma = adaptive_initial_guess(S, K, T)

    for i in range(max_iterations):
        prices = vectorized_black_scholes(S, K, T, r, sigma, option_types)
        vega = S * np.sqrt(T) * norm.pdf((np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T)))
        diff = option_prices - prices

        update = diff / (vega + 1e-8)
        sigma_new = sigma + update

        # Bisection-like adjustment for stability
        sigma_new = np.where(sigma_new < 0.0001, (sigma + 0.0001) / 2, sigma_new)
        sigma_new = np.where(sigma_new > 10, (sigma + 10) / 2, sigma_new)

        if np.all(np.abs(diff) < precision):
            return sigma_new#, i  # Return iterations taken for convergence analysis

        sigma = sigma_new

    # Flag non-converged options
    non_converged = np.abs(diff) >= precision
    sigma[non_converged] = None

    return sigma #,max_iterations

def vectorized_greeks(S, K, T, r, sigma, option_types, q=0):
    valid_mask = (S > 0) & (K > 0) & (T > 0) & (sigma > 0)
    d1 = np.full_like(S, np.nan)
    d2 = np.full_like(S, np.nan)

    np.divide(np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T, sigma * np.sqrt(T), out=d1, where=valid_mask)
    d2 = d1 - sigma * np.sqrt(T)

    delta = np.where(valid_mask,
                     np.where(option_types == 'call',
                              np.exp(-q * T) * norm.cdf(d1),
                              -np.exp(-q * T) * norm.cdf(-d1)),
                     np.nan)

    gamma = np.where(valid_mask,
                     np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T)),
                     np.nan)

    theta = np.where(valid_mask,
                     np.where(option_types == 'call',
                              -(S * sigma * np.exp(-q * T) * norm.pdf(d1)) / (2 * np.sqrt(T)) -
                              r * K * np.exp(-r * T) * norm.cdf(d2) + q * S * np.exp(-q * T) * norm.cdf(d1),
                              -(S * sigma * np.exp(-q * T) * norm.pdf(d1)) / (2 * np.sqrt(T)) +
                              r * K * np.exp(-r * T) * norm.cdf(-d2) - q * S * np.exp(-q * T) * norm.cdf(-d1)),
                     np.nan)

    vega = np.where(valid_mask,
                    S * np.exp(-q * T) * np.sqrt(T) * norm.pdf(d1) / 100,
                    np.nan)

    rho = np.where(valid_mask,
                   np.where(option_types == 'call',
                            K * T * np.exp(-r * T) * norm.cdf(d2) / 100,
                            -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100),
                   np.nan)

    # Convert theta to daily
    theta = np.where(valid_mask, theta / 365, np.nan)  # or use 252 for trading days

    return {'delta': delta, 'gamma': gamma, 'theta': theta, 'vega': vega, 'rho': rho}

#
# async def bulk_insert_option_quotes(conn: Connection, option_quotes_data):
#     insert_query = """
#     INSERT INTO csvimport.option_quotes (
#         contract_id, fetch_timestamp, root_symbol, last, change, volume,
#         open, high, low, bid, ask, greeks, change_percentage, last_volume,
#         trade_date, prevclose, bidsize, bidexch, bid_date, asksize, askexch,
#         ask_date, open_interest, implied_volatility, realtime_calculated_greeks,
#         risk_free_rate
#     ) VALUES (
#         %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s,
#         %s, %s, %s, %s, %s, %s, %s
#     )
#     ON CONFLICT (contract_id, fetch_timestamp) DO UPDATE SET
#         root_symbol = EXCLUDED.root_symbol,
#         last = EXCLUDED.last,
#         change = EXCLUDED.change,
#         volume = EXCLUDED.volume,
#         open = EXCLUDED.open,
#         high = EXCLUDED.high,
#         low = EXCLUDED.low,
#         bid = EXCLUDED.bid,
#         ask = EXCLUDED.ask,
#         greeks = EXCLUDED.greeks,
#         change_percentage = EXCLUDED.change_percentage,
#         last_volume = EXCLUDED.last_volume,
#         trade_date = EXCLUDED.trade_date,
#         prevclose = EXCLUDED.prevclose,
#         bidsize = EXCLUDED.bidsize,
#         bidexch = EXCLUDED.bidexch,
#         bid_date = EXCLUDED.bid_date,
#         asksize = EXCLUDED.asksize,
#         askexch = EXCLUDED.askexch,
#         ask_date = EXCLUDED.ask_date,
#         open_interest = EXCLUDED.open_interest,
#         implied_volatility = EXCLUDED.implied_volatility,
#         realtime_calculated_greeks = EXCLUDED.realtime_calculated_greeks,
#         risk_free_rate = EXCLUDED.risk_free_rate
#     """
#
#     # Convert dict data to tuple format
#     option_quotes_tuples = [
#         (
#             d['contract_id'], d['fetch_timestamp'], d['root_symbol'], d['last'],
#             d['change'], d['volume'], d['open'], d['high'], d['low'], d['bid'],
#             d['ask'], json.dumps(d['greeks']), d['change_percentage'],
#             d['last_volume'], d['trade_date'], d['prevclose'], d['bidsize'],
#             d['bidexch'], d['bid_date'], d['asksize'], d['askexch'], d['ask_date'],
#             d['open_interest'], d['implied_volatility'],
#             json.dumps(d['realtime_calculated_greeks']), d['risk_free_rate']
#         )
#         for d in option_quotes_data
#     ]
#
#     total_rows = len(option_quotes_tuples)
#     print(f"Total rows to insert: {total_rows}")
#
#     with conn.connection.cursor() as cur:
#         start_time = time.time()
#         try:
#             execute_batch(cur, insert_query, option_quotes_tuples, page_size=1000)
#             conn.commit()
#             end_time = time.time()
#             print(f"Bulk insert completed. Time taken: {end_time - start_time:.2f} seconds")
#             print(f"Insertion rate: {total_rows / (end_time - start_time):.2f} rows/second")
#         except Exception as e:
#             conn.rollback()
#             print(f"Error during execute_batch: {str(e)}")
#             raise


class DividendYieldCache:
    def __init__(self):
        self.cache = {}
        self.lock = asyncio.Lock()
        self.last_update_date = {}

    async def get_dividend_yield(self, conn, session, ticker, real_auth, current_price):
        async with self.lock:
            current_date = datetime.now().date()

            # Check if we have cached data and if it's still valid (updated today)
            if ticker in self.cache and self.last_update_date.get(ticker) == current_date:
                return self.cache[ticker]

            # If no valid cached data, calculate new yield
            dividend_yield = await self._calculate_dividend_yield(conn, session, ticker, real_auth, current_price)

            # Update cache and last update date
            self.cache[ticker] = dividend_yield
            self.last_update_date[ticker] = current_date

            return dividend_yield

    async def _calculate_dividend_yield(self, conn, session, ticker, real_auth, current_price):
        # Check if we have recent dividend data in the database
        one_year_ago = datetime.now() - timedelta(days=365)
        db_dividends = await conn.fetch('''
            SELECT * FROM csvimport.dividends
            WHERE symbol_name = $1 AND ex_date >= $2
            ORDER BY ex_date DESC
        ''', ticker, one_year_ago)

        if not db_dividends:
            # If no recent data in database, fetch from API
            api_dividends = await fetch_dividend_data(session, ticker, real_auth)
            if api_dividends:
                # Convert API data to Dividend objects and store in DB
                await conn.executemany('''
                    INSERT INTO csvimport.dividends (
                        symbol_name, dividend_type, ex_date, cash_amount, currency_id,
                        declaration_date, frequency, pay_date, record_date
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ''', [(
                    ticker,
                    div['dividend_type'],
                    datetime.strptime(div['ex_date'], '%Y-%m-%d').date(),
                    div['cash_amount'],
                    div['currency_i_d'],
                    datetime.strptime(div['declaration_date'], '%Y-%m-%d').date(),
                    div['frequency'],
                    datetime.strptime(div['pay_date'], '%Y-%m-%d').date(),
                    datetime.strptime(div['record_date'], '%Y-%m-%d').date()
                ) for div in api_dividends])

                db_dividends = await conn.fetch('''
                    SELECT * FROM csvimport.dividends
                    WHERE symbol_name = $1 AND ex_date >= $2
                    ORDER BY ex_date DESC
                ''', ticker, one_year_ago)

        # Calculate dividend yield
        if db_dividends:
            # Sum all dividends paid in the last year if certain types
            annual_dividend = sum(div['cash_amount'] for div in db_dividends
                                  if div['dividend_type'] in ['CD', 'SC', 'CG', 'RC', 'LQ'] or div[
                                      'dividend_type'] is None)
            dividend_yield = annual_dividend / current_price if current_price > 0 else 0
        else:
            dividend_yield = 0

        return dividend_yield


# Initialize the cache
dividend_yield_cache = DividendYieldCache()

# Global cache for Treasury yields
treasury_yield_cache = {}
CACHE_EXPIRY = 3600  # 1 hour in seconds

paper_auth = PrivateData.tradier_info.paper_auth
real_acc = PrivateData.tradier_info.real_acc
real_auth = PrivateData.tradier_info.real_auth
sem = asyncio.Semaphore(50)


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
    Inserts calculated option data using a deffffffffffffdicated database session.

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




# Initialize FRED API
fred = fredapi.Fred(api_key='0fd2a19c651aa21bbab822b3b20a7352 ')  # Replace with your FRED API key
#TODO add mid/bid/ask iv/greeks?

# Define series IDs and their corresponding max days
series_config = [
    ('DGS1MO', 30),
    ('DGS3MO', 90),
    ('DGS6MO', 180),
    ('DGS1', 365),
    ('DGS2', 365 * 2),
    ('DGS5', 365 * 5),
    ('DGS10', 365 * 10),
    ('DGS30', float('inf'))
]


last_used_series = None

def get_treasury_yield(days_to_expiration):
    global last_used_series
    current_time = time.time()

    if not treasury_yield_cache or current_time - treasury_yield_cache.get('last_update', 0) > CACHE_EXPIRY:
        logger.info("Refreshing Treasury yield cache")
        # Fetch each series
        for series_id, _ in series_config:
            try:
                data = fred.get_series(series_id, observation_start=datetime.now() - timedelta(days=30),
                                       observation_end=datetime.now())
                treasury_yield_cache[series_id] = data.dropna().iloc[-1] / 100 if not data.empty else None
            except Exception as e:
                logger.error(f"Error fetching Treasury yield for {series_id}: {e}")
                treasury_yield_cache[series_id] = None

        treasury_yield_cache['last_update'] = current_time

    # Determine which series to use based on days_to_expiration
    for series_id, max_days in series_config:
        if days_to_expiration <= max_days:
            yield_value = treasury_yield_cache.get(series_id, 0.02)
            if series_id != last_used_series:
                # logger.info(
                #     f"Switching to yield for {series_id}: {yield_value} (days to expiration: {days_to_expiration})")
                last_used_series = series_id
            return yield_value

    yield_value = treasury_yield_cache.get('DGS30', 0.02)
    if last_used_series != 'DGS30':
        logger.info(f"Using 30-year yield: {yield_value} (days to expiration: {days_to_expiration})")
        last_used_series = 'DGS30'
    return yield_value
def parse_timestamp(timestamp_str):
    if timestamp_str is None:
        return None
    try:
        return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00')).replace(tzinfo=timezone.utc)
    except ValueError:
        logger.error(f"Failed to parse timestamp: {timestamp_str}")
        return None


def calculate_option_greeks(S, K, T, r, sigma, option_type, q=0):
    """
    Calculate option Greeks using the Black-Scholes-Merton model (with dividends).
     :param S: Current stock price
    :param K: Option strike price
    :param T: Time to expiration (in years)
    :param r: Risk-free interest rate
    :param sigma: Volatility of the underlying stock
    :param option_type: 'call' or 'put'
    :param q: Dividend yield (annualized)
    :return: Dictionary of Greeks
    """

    try:
        N = norm.cdf
        N_prime = norm.pdf

        # Avoid division by zero
        if sigma <= 0 or T <= 0:
            return None

        d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        if option_type == 'call':
            delta = np.exp(-q * T) * N(d1)
            gamma = np.exp(-q * T) * N_prime(d1) / (S * sigma * np.sqrt(T))
            theta = -((S * sigma * np.exp(-q * T) * N_prime(d1)) / (2 * np.sqrt(T))) - r * K * np.exp(-r * T) * N(
                d2) + q * S * np.exp(-q * T) * N(d1)
            vega = S * np.exp(-q * T) * np.sqrt(T) * N_prime(d1) / 100
            rho = K * T * np.exp(-r * T) * N(d2) / 100
        else:  # put option
            delta = -np.exp(-q * T) * N(-d1)
            gamma = np.exp(-q * T) * N_prime(d1) / (S * sigma * np.sqrt(T))
            theta = -((S * sigma * np.exp(-q * T) * N_prime(d1)) / (2 * np.sqrt(T))) + r * K * np.exp(-r * T) * N(
                -d2) - q * S * np.exp(-q * T) * N(-d1)
            vega = S * np.exp(-q * T) * np.sqrt(T) * N_prime(d1) / 100
            rho = -K * T * np.exp(-r * T) * N(-d2) / 100

        # Convert theta to daily
        theta = theta / 365  # or use 252 for trading days

        result = {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
#TODO should i use a differne representation of inf?
        # Replace inf and -inf with None
        for key, value in result.items():
            if np.isinf(value) or np.isnan(value):
                result[key] = None

        return result

    except Exception as e:
        logger.error(f"Error in calculate_option_greeks: {e}")
        return None

async def fetch_dividend_data(session, ticker, real_auth):
    url = "https://api.tradier.com/beta/markets/fundamentals/dividends"
    headers = {"Authorization": f"Bearer {real_auth}", "Accept": "application/json"}
    params = {"symbols": ticker}

    try:
        async with session.get(url, params=params, headers=headers) as response:
            response.raise_for_status()
            data = await response.json()

        if data and isinstance(data, list) and data[0].get("results"):
            dividends = data[0]["results"][0]["tables"]["cash_dividends"]
            return dividends
        else:
            logger.warning(f"No dividend data found for {ticker}")
            return None
    except Exception as e:
        logger.error(f"Error fetching dividend data for {ticker}: {e}")
        return None

# Ensure tables are created before querying
#TODO DO SYNCHRONOUS BULK INSERT
async def post_market_quotes(session, ticker, real_auth):
    url = "https://api.tradier.com/v1/markets/quotes"
    headers = {"Authorization": f"Bearer {real_auth}", "Accept": "application/json"}
    all_contracts = await lookup_all_option_contracts(session, ticker, real_auth)

    # Optimized batching: Use list comprehension for faster symbol string creation
    BATCH_SIZE = 2000 #was 8000
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
                # response.raise_for_status()
                data = await response.json()
                # handle_rate_limit(response)

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


def calculate_time_to_expiration(df):
    # Use New York time as it's the primary US stock market timezone
    ny_tz = pytz.timezone('America/New_York')
    now = pd.Timestamp.now(tz=ny_tz)

    # Function to get the end of day for expiration
    def get_expiration_datetime(date):
        # Convert to datetime if it's not already
        if not isinstance(date, pd.Timestamp):
            date = pd.Timestamp(date)

        # Set to 4:00 PM NY time on the expiration date
        expiry = date.tz_localize(ny_tz).replace(hour=16, minute=0, second=0, microsecond=0)

        # If it's a weekend, move to the previous business day
        while expiry.dayofweek > 4:  # 5 = Saturday, 6 = Sunday
            expiry -= pd.Timedelta(days=1)

        return expiry

    # Apply the function to get proper expiration datetimes
    df['expiration_datetime'] = df['expiration_date'].apply(get_expiration_datetime)

    # Calculate time to expiration
    df['time_to_expiration'] = (df['expiration_datetime'] - now).dt.total_seconds() / (365.25 * 24 * 60 * 60)
    df['days_to_expiration'] = (df['expiration_datetime'] - now).dt.total_seconds() / (24 * 60 * 60)

    # Handle cases where time to expiration is negative (option has expired)
    # df.loc[df['time_to_expiration'] < 0, 'time_to_expiration'] = 0
    # df.loc[df['days_to_expiration'] < 0, 'days_to_expiration'] = 0

    return df
def process_option_quotes(all_contract_quotes, current_price, last_close_price, dividend_yield,ticker):
    df = all_contract_quotes.copy()
    df['contract_id'] = df['symbol']
    df["dollarsFromStrike"] = abs(df["strike"] - last_close_price)
    df["expiration_date"] = pd.to_datetime(df["expiration_date"]).dt.strftime('%Y-%m-%d')

    df["dollarsFromStrikeXoi"] = df["dollarsFromStrike"] * df["open_interest"]
    df["lastPriceXoi"] = df["last"] * df["open_interest"]

    # Calculate time to expiration in years and days
    # now = pd.Timestamp.now(tz='UTC')
    #now fasctors in hours left of exp date
    df['expiration_date'] = pd.to_datetime(df['expiration_date']).dt.date #already done?

    df = calculate_time_to_expiration(df)

    # Calculate implied volatility (this is a simplification, you might want to use a more sophisticated method)
    # df['implied_volatility'] = df['greeks'].apply(lambda x: x.get('mid_iv', 0.3) if isinstance(x, dict) else 0.3)
    # Calculate implied volatility
    # Get appropriate Treasury yield for each option
    df['risk_free_rate'] = df['days_to_expiration'].apply(get_treasury_yield)
    # if ticker in TICKERS_FOR_TRADE_ALGOS:
    #     df['implied_volatility'] = df.apply(
    #         lambda row: calculate_implied_volatility(
    #             row['last'],  # Use the last price as the option price
    #             current_price,
    #             row['strike'],
    #             row['time_to_expiration'],
    #             row['risk_free_rate'],
    #             row['option_type']
    #         ),
    #         axis=1)
    #         # Calculate realtime Greeks
    #     df['realtime_calculated_greeks'] = df.apply(
    #         lambda row: calculate_option_greeks(
    #             current_price,
    #
    #             row['strike'],
    #             row['time_to_expiration'],
    #             row['risk_free_rate'],
    #             row['implied_volatility'],
    #             row['option_type'],
    #             q=dividend_yield
    #         ),
    #         axis=1
    #     )
    # else:
    #     df['implied_volatility'] = None
    #     df['realtime_calculated_greeks'] = None
    #Using new IV + Greeks function:
    # if ticker in TICKERS_FOR_TRADE_ALGOS:
    #     df[['implied_volatility', 'realtime_calculated_greeks']] = df.apply(
    #         lambda row: calculate_implied_volatility_and_greeks(
    #             row['last'],
    #             current_price,
    #             row['strike'],
    #             row['time_to_expiration'],
    #             row['risk_free_rate'],
    #             row['option_type'],
    #             q=dividend_yield
    #         ),
    #         axis=1,
    #         result_type='expand'  # This is important to unpack the tuple into two columns
    #     )
    # else:
    #     df['implied_volatility'] = None
    #     df['realtime_calculated_greeks'] = None
    #
    # # print("Implied Volatility Sample:")
    # # print(df['implied_volatility'].head())
    # return df
    #TODO use mid?
    """def calculate_mid_price(df):
    
    return (df['bid'] + df['ask']) / 2

def check_price_quality(df):
    df['bid_ask_spread'] = df['ask'] - df['bid']
    df['spread_percentage'] = df['bid_ask_spread'] / df['mid_price']
    
    # Flag wide spreads (e.g., more than 10%)
    wide_spreads = df[df['spread_percentage'] > 0.1]
    if not wide_spreads.empty:
        logger.warning(f"Wide bid-ask spreads detected for {len(wide_spreads)} options")

def process_option_quotes(df, current_price, last_close_price, dividend_yield, ticker):
    # ... (other code)

    if ticker in TICKERS_FOR_TRADE_ALGOS:
        df['mid_price'] = calculate_mid_price(df)
        check_price_quality(df)

        # Use mid-price for calculations
        option_prices = df['mid_price'].values

        # Calculate implied volatility and Greeks using mid-price
        implied_volatilities = vectorized_implied_volatility(option_prices, S, K, T, r, option_types)
        # ... (rest of the calculations)"""
    if ticker in TICKERS_FOR_TRADE_ALGOS:
        # Prepare arrays for vectorized calculations
        S = np.full(len(df), current_price)
        K = df['strike'].values
        T = df['time_to_expiration'].values
        r = df['risk_free_rate'].values
        option_prices = df['last'].values
        option_types = df['option_type'].values
        # Calculate implied volatility
        implied_volatilities = vectorized_implied_volatility_improved(option_prices, S, K, T, r, option_types)
        df['implied_volatility'] = implied_volatilities

        # Calculate Greeks
        greeks = vectorized_greeks(S, K, T, r, implied_volatilities, option_types, q=dividend_yield)

        # Assign Greeks to DataFrame
        for greek, values in greeks.items():
            df[f'calculated_{greek}'] = values

        # Combine Greeks into a single column
        df['realtime_calculated_greeks'] = df.apply(lambda row: {
            greek: row[f'calculated_{greek}'] for greek in ['delta', 'gamma', 'theta', 'vega', 'rho']
            if not np.isnan(row[f'calculated_{greek}'])
        }, axis=1)

        # Handle cases where all Greeks are NaN
        df['realtime_calculated_greeks'] = df['realtime_calculated_greeks'].apply(
            lambda x: x if x else None
        )

        # Replace NaN with None for individual Greeks and implied volatility
        for col in ['implied_volatility'] + [f'calculated_{greek}' for greek in
                                             ['delta', 'gamma', 'theta', 'vega', 'rho']]:
            df[col] = df[col].where(df[col].notna(), None)

    else:
        df['implied_volatility'] = None
        df['realtime_calculated_greeks'] = None

    # Calculate ITM flag for each contract
    df['itm'] = ((df['option_type'] == 'call') & (df['strike'] <= current_price)) | \
                ((df['option_type'] == 'put') & (df['strike'] >= current_price))

    return df


def calculate_pcr(puts, calls, current_price, offset, column):
    put_value = puts[(puts['strike'] >= current_price + offset)][column].sum()
    call_value = calls[(calls['strike'] <= current_price - offset)][column].sum()
    return put_value / call_value if call_value != 0 else np.inf
def calculate_aggregated_metrics(df, current_price, last_close_price):
    # Group by expiration date
    grouped_df = df.groupby('expiration_date')

    # Initialize a list to store aggregated metrics for each expiration date
    all_agg_metrics = []

    # Calculate metrics for each group
    for exp_date, group in grouped_df:
        agg_metrics = {}  # Dictionary to store metrics for this expiration date
        # Calculate PCR for volume and open interest (within the group)
        total_put_vol = group[group['option_type'] == 'put']['volume'].sum()
        total_call_vol = group[group['option_type'] == 'call']['volume'].sum()
        total_put_oi = group[group['option_type'] == 'put']['open_interest'].sum()
        total_call_oi = group[group['option_type'] == 'call']['open_interest'].sum()
        agg_metrics['closest_strike_to_cp'] = group.loc[abs(group['strike'] - current_price).idxmin(), 'strike']

        # Check for zero or NaN in denominator before division
        agg_metrics['pcr_vol'] = total_put_vol / total_call_vol if total_call_vol != 0 and not np.isnan(
            total_call_vol) else np.nan
        agg_metrics['pcr_oi'] = total_put_oi / total_call_oi if total_call_oi != 0 and not np.isnan(
            total_call_oi) else np.nan

        # Calculate ITM PCR for volume and open interest (within the group)
        itm_puts = group[(group['option_type'] == 'put') & group['itm']]
        itm_calls = group[(group['option_type'] == 'call') & group['itm']]
        itm_put_vol = itm_puts['volume'].sum()
        itm_call_vol = itm_calls['volume'].sum()
        itm_put_oi = itm_puts['open_interest'].sum()
        itm_call_oi = itm_calls['open_interest'].sum()

        # Check for zero or NaN in denominator before division
        agg_metrics['itm_pcr_vol'] = itm_put_vol / itm_call_vol if itm_call_vol != 0 and not np.isnan(
            itm_call_vol) else np.nan
        agg_metrics['itm_pcr_oi'] = itm_put_oi / itm_call_oi if itm_call_oi != 0 and not np.isnan(
            itm_call_oi) else np.nan

        # Calculate OI metrics (within the group)
        agg_metrics['itm_oi'] = itm_calls['open_interest'].sum() + itm_puts['open_interest'].sum()
        agg_metrics['total_oi'] = group['open_interest'].sum()  # Use 'group' here
        # Check for zero or NaN in denominator before division
        if agg_metrics['total_oi'] != 0 and not np.isnan(agg_metrics['total_oi']):
            agg_metrics['itm_contracts_percent'] = agg_metrics['itm_oi'] / agg_metrics['total_oi']
        else:
            agg_metrics['itm_contracts_percent'] = np.nan  # Or handle it differently based on your needs

        # Calculate net IV (within the group)
        agg_metrics['net_iv'] = group[group['option_type'] == 'call']['implied_volatility'].sum() - \
                            group[group['option_type'] == 'put']['implied_volatility'].sum()
        agg_metrics['net_itm_iv'] = itm_calls['implied_volatility'].sum() - itm_puts['implied_volatility'].sum()

        # Calculate Bonsai Ratio
        itm_call_vol, itm_put_vol = itm_calls['volume'].sum(), itm_puts['volume'].sum()
        itm_call_oi, itm_put_oi = itm_calls['open_interest'].sum(), itm_puts['open_interest'].sum()

        calls_in_group = group[group['option_type'] == 'call']
        puts_in_group = group[group['option_type'] == 'put']

        total_call_vol, total_put_vol = calls_in_group['volume'].sum(), puts_in_group['volume'].sum()
        total_call_oi, total_put_oi = calls_in_group['open_interest'].sum(), puts_in_group['open_interest'].sum()

        # Check for zero or NaN in denominators before division
        ratio_put_vol = np.divide(itm_put_vol, total_put_vol, where=total_put_vol != 0 and not np.isnan(total_put_vol))
        ratio_put_oi = np.divide(itm_put_oi, total_put_oi, where=total_put_oi != 0 and not np.isnan(total_put_oi))
        ratio_call_vol = np.divide(itm_call_vol, total_call_vol,
                                   where=total_call_vol != 0 and not np.isnan(total_call_vol))
        ratio_call_oi = np.divide(itm_call_oi, total_call_oi, where=total_call_oi != 0 and not np.isnan(total_call_oi))

        numerator = np.multiply(ratio_put_vol, ratio_put_oi)
        denominator = np.multiply(ratio_call_vol, ratio_call_oi)

        # Check for zero or NaN in denominator before final division
        agg_metrics['bonsai_ratio'] = np.divide(numerator, denominator,
                                                where=denominator != 0 and not np.isnan(denominator))

        # Calculate at-the-money IV (within the group)
        atm_option = group.loc[abs(group['strike'] - current_price).idxmin()]
        agg_metrics['atm_iv'] = atm_option['implied_volatility']

        # Calculate max pain
        # Calculate max pain
        def pain(strike):
            # Use 'group' instead of 'df' here
            call_pain = (strike - current_price) * \
                        group[(group['option_type'] == 'call') & (group['strike'] <= strike)]['open_interest'].sum()
            put_pain = (current_price - strike) * group[(group['option_type'] == 'put') & (group['strike'] >= strike)][
                'open_interest'].sum()
            return call_pain + put_pain

        unique_strikes = group['strike'].unique()
        agg_metrics['max_pain'] = min(unique_strikes, key=pain)
        # Add calculations from `calculations.py`
        calls = group[group['option_type'] == 'call']
        puts = group[group['option_type'] == 'put']

        agg_metrics['avg_call_iv'] = calls['implied_volatility'].mean()
        agg_metrics['avg_put_iv'] = puts['implied_volatility'].mean()
        # Extract Greeks from JSON and calculate averages
        greeks_df = group['realtime_calculated_greeks'].apply(pd.Series)  # Convert JSON to DataFrame
        # print(greeks_df)
        for greek in ['delta', 'gamma', 'theta', 'vega']:
            agg_metrics[f'avg_{greek}'] = greeks_df[greek].mean()  # Calculate mean for each Greek
            if agg_metrics[f'avg_{greek}'] == np.inf:
                agg_metrics[f'avg_{greek}'] = None
            # print(agg_metrics[f'avg_{greek}'])

        agg_metrics['total_volume'] = group['volume'].sum()

        # print(f"Group DataFrame for expiration date {exp_date}:")
        # print(group)
        #
        # print(f"Aggregated metrics for expiration date {exp_date}:")
        # print(agg_metrics)

        # Additional metrics (similar to your existing code)
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

        # Replace np.inf with None before converting to JSON
        for key, value in additional_metrics.items():
            if value == np.inf:
                additional_metrics[key] = None

        agg_metrics['additional_metrics'] = json.dumps(additional_metrics)

        # Add current price and price change
        agg_metrics['current_stock_price'] = current_price
        agg_metrics['current_sp_change_lac'] = (current_price - last_close_price) / last_close_price * 100

        # Add expiration date to the metrics
        agg_metrics['expiration_date'] = exp_date

        # Append metrics for this expiration date to the list
        all_agg_metrics.append(agg_metrics)

    return all_agg_metrics  # Return


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
            latest_minute = df.index.max() #LMAO this was set to .min, so it would have all been wrong! glad we checked.
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




async def insert_option_data(conn, options_df):
    # Convert expiration_date to datetime.date object
    options_df['expiration_date'] = pd.to_datetime(options_df['expiration_date']).dt.date

    # Insert option data (similar to your existing code)
    await conn.executemany('''
        INSERT INTO csvimport.options (
            contract_id, underlying, expiration_date, strike, option_type,
            contract_size, description, expiration_type
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        ON CONFLICT (contract_id) DO UPDATE SET
            underlying = EXCLUDED.underlying,
            expiration_date = EXCLUDED.expiration_date,
            strike = EXCLUDED.strike,
            option_type = EXCLUDED.option_type,
            contract_size = EXCLUDED.contract_size,
            description = EXCLUDED.description,
            expiration_type = EXCLUDED.expiration_type
    ''', [(row['contract_id'], row['underlying'], row['expiration_date'],
           row['strike'], row['option_type'], row['contract_size'],
           row['description'], row['expiration_type'])
          for _, row in options_df.iterrows()])

    # Insert option quotes data (similar to your existing code)
    await conn.executemany('''
        INSERT INTO csvimport.option_quotes (
            contract_id, fetch_timestamp, root_symbol, last, change, volume,
            open, high, low, bid, ask, greeks, change_percentage, last_volume,
            trade_date, prevclose, bidsize, bidexch, bid_date, asksize, askexch,
            ask_date, open_interest, implied_volatility, realtime_calculated_greeks,
            risk_free_rate
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
                  $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26)
        ON CONFLICT (contract_id, fetch_timestamp) DO UPDATE SET
            root_symbol = EXCLUDED.root_symbol,
            last = EXCLUDED.last,
            change = EXCLUDED.change,
            volume = EXCLUDED.volume,
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            bid = EXCLUDED.bid,
            ask = EXCLUDED.ask,
            greeks = EXCLUDED.greeks,
            change_percentage = EXCLUDED.change_percentage,
            last_volume = EXCLUDED.last_volume,
            trade_date = EXCLUDED.trade_date,
            prevclose = EXCLUDED.prevclose,
            bidsize = EXCLUDED.bidsize,
            bidexch = EXCLUDED.bidexch,
            bid_date = EXCLUDED.bid_date,
            asksize = EXCLUDED.asksize,
            askexch = EXCLUDED.askexch,
            ask_date = EXCLUDED.ask_date,
            open_interest = EXCLUDED.open_interest,
            implied_volatility = EXCLUDED.implied_volatility,
            realtime_calculated_greeks = EXCLUDED.realtime_calculated_greeks,
            risk_free_rate = EXCLUDED.risk_free_rate
    ''', [(row['contract_id'], row['fetch_timestamp'],
           row['root_symbol'], row['last'], row['change'], row['volume'],
           row['open'], row['high'], row['low'], row['bid'], row['ask'],
           json.dumps(row['greeks']), row['change_percentage'], row['last_volume'],
           convert_unix_to_datetime(row['trade_date']), row['prevclose'], row['bidsize'], row['bidexch'],
           convert_unix_to_datetime(row['bid_date']), row['asksize'], row['askexch'], convert_unix_to_datetime(row['ask_date']),
           row['open_interest'], row['implied_volatility'],
           json.dumps(row['realtime_calculated_greeks']) if row['realtime_calculated_greeks'] else None, row['risk_free_rate'])
          for _, row in options_df.iterrows()])

async def insert_aggregated_metrics(conn, agg_metrics, ticker, fetch_timestamp, exp_date):
    await conn.execute('''
        INSERT INTO csvimport.optimized_processed_option_data (
            symbol_name, fetch_timestamp, exp_date, current_stock_price, current_sp_change_lac,
            max_pain, bonsai_ratio, pcr_vol, pcr_oi, itm_pcr_vol, itm_pcr_oi,
            itm_oi, total_oi, itm_contracts_percent, net_iv, net_itm_iv,
            closest_strike_to_cp, atm_iv,
            avg_call_iv, avg_put_iv, avg_delta, avg_gamma, avg_theta, avg_vega, total_volume, additional_metrics
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, 
                  $18, $19, $20, $21, $22, $23, $24, $25, $26)
        ON CONFLICT (symbol_name, fetch_timestamp, exp_date) DO UPDATE SET
            current_stock_price = EXCLUDED.current_stock_price,
            current_sp_change_lac = EXCLUDED.current_sp_change_lac,
            max_pain = EXCLUDED.max_pain,
            bonsai_ratio = EXCLUDED.bonsai_ratio,
            pcr_vol = EXCLUDED.pcr_vol,
            pcr_oi = EXCLUDED.pcr_oi,
            itm_pcr_vol = EXCLUDED.itm_pcr_vol,
            itm_pcr_oi = EXCLUDED.itm_pcr_oi,
            itm_oi = EXCLUDED.itm_oi,
            total_oi = EXCLUDED.total_oi,
            itm_contracts_percent = EXCLUDED.itm_contracts_percent,
            net_iv = EXCLUDED.net_iv,
            net_itm_iv = EXCLUDED.net_itm_iv,
            closest_strike_to_cp = EXCLUDED.closest_strike_to_cp,
            atm_iv = EXCLUDED.atm_iv,
            avg_call_iv = EXCLUDED.avg_call_iv,
            avg_put_iv = EXCLUDED.avg_put_iv,
            avg_delta = EXCLUDED.avg_delta,
            avg_gamma = EXCLUDED.avg_gamma,
            avg_theta = EXCLUDED.avg_theta,
            avg_vega = EXCLUDED.avg_vega,
            total_volume = EXCLUDED.total_volume,
            additional_metrics = EXCLUDED.additional_metrics
    ''', ticker, fetch_timestamp, exp_date, agg_metrics['current_stock_price'], agg_metrics['current_sp_change_lac'],
        agg_metrics['max_pain'], agg_metrics['bonsai_ratio'], agg_metrics['pcr_vol'], agg_metrics['pcr_oi'],
        agg_metrics['itm_pcr_vol'], agg_metrics['itm_pcr_oi'], agg_metrics['itm_oi'], agg_metrics['total_oi'],
        agg_metrics['itm_contracts_percent'], agg_metrics['net_iv'], agg_metrics['net_itm_iv'],
        agg_metrics['closest_strike_to_cp'], agg_metrics['atm_iv'],
        agg_metrics['avg_call_iv'], agg_metrics['avg_put_iv'], agg_metrics['avg_delta'],
        agg_metrics['avg_gamma'], agg_metrics['avg_theta'], agg_metrics['avg_vega'],
        agg_metrics['total_volume'], agg_metrics['additional_metrics'])
async def get_options_data(conn, session, ticker, loop_start_time):
    headers = {"Authorization": f"Bearer {real_auth}", "Accept": "application/json"}
    ticker_quote = await fetch(
        session,
        "https://api.tradier.com/v1/markets/quotes",
        params={"symbols": ticker, "greeks": "false"},
        headers=headers,
    )



    # Process Stock Quote Data
    quote_df = pd.DataFrame.from_dict(ticker_quote["quotes"]["quote"], orient="index").T
    quote_df.to_csv('quotedftest.csv')
    current_price = quote_df.at[0, "last"]
    prevclose = quote_df.at[0, "prevclose"]
    symbol_name = ticker

 # Insert symbol data
    await conn.execute('''
        INSERT INTO csvimport.symbols (symbol_name, description, type)
        VALUES ($1, $2, $3)
        ON CONFLICT (symbol_name) DO UPDATE SET
        description = EXCLUDED.description,
        type = EXCLUDED.type
    ''', ticker, quote_df.at[0, "description"], quote_df.at[0, "type"])



    # Fetch timesales data
    timesales_data = await get_timesales(session, ticker, lookback_minutes=1200)

    # Prepare the query
    query = '''
          INSERT INTO csvimport.symbol_quotes (
              symbol_name, fetch_timestamp, last_trade_price, current_bid, current_ask,
              daily_open, daily_high, daily_low, previous_close, last_trade_volume,
              daily_volume, average_daily_volume, last_trade_timestamp, week_52_high,
              week_52_low, daily_change, daily_change_percentage, current_bidsize,
              bidexch, current_bid_date, current_asksize, askexch, current_ask_date,
              exch, last_1min_timesale, last_1min_timestamp, last_1min_open,
              last_1min_high, last_1min_low, last_1min_close, last_1min_volume,
              last_1min_vwap
          ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
                    $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26, $27, $28,
                    $29, $30, $31, $32)
          ON CONFLICT (symbol_name, fetch_timestamp) DO UPDATE SET
              last_trade_price = EXCLUDED.last_trade_price,
              current_bid = EXCLUDED.current_bid,
              current_ask = EXCLUDED.current_ask,
              daily_open = EXCLUDED.daily_open,
              daily_high = EXCLUDED.daily_high,
              daily_low = EXCLUDED.daily_low,
              previous_close = EXCLUDED.previous_close,
              last_trade_volume = EXCLUDED.last_trade_volume,
              daily_volume = EXCLUDED.daily_volume,
              average_daily_volume = EXCLUDED.average_daily_volume,
              last_trade_timestamp = EXCLUDED.last_trade_timestamp,
              week_52_high = EXCLUDED.week_52_high,
              week_52_low = EXCLUDED.week_52_low,
              daily_change = EXCLUDED.daily_change,
              daily_change_percentage = EXCLUDED.daily_change_percentage,
              current_bidsize = EXCLUDED.current_bidsize,
              bidexch = EXCLUDED.bidexch,
              current_bid_date = EXCLUDED.current_bid_date,
              current_asksize = EXCLUDED.current_asksize,
              askexch = EXCLUDED.askexch,
              current_ask_date = EXCLUDED.current_ask_date,
              exch = EXCLUDED.exch,
              last_1min_timesale = EXCLUDED.last_1min_timesale,
              last_1min_timestamp = EXCLUDED.last_1min_timestamp,
              last_1min_open = EXCLUDED.last_1min_open,
              last_1min_high = EXCLUDED.last_1min_high,
              last_1min_low = EXCLUDED.last_1min_low,
              last_1min_close = EXCLUDED.last_1min_close,
              last_1min_volume = EXCLUDED.last_1min_volume,
              last_1min_vwap = EXCLUDED.last_1min_vwap
      '''

    # Prepare the values
    values = [
        ticker, loop_start_time.astimezone(eastern), quote_df.at[0, "last"],
        quote_df.at[0, "bid"], quote_df.at[0, "ask"], quote_df.at[0, "open"],
        quote_df.at[0, "high"], quote_df.at[0, "low"], quote_df.at[0, "prevclose"],
        quote_df.at[0, "last_volume"], quote_df.at[0, "volume"],
        quote_df.at[0, "average_volume"], convert_unix_to_datetime(quote_df.at[0, "trade_date"]),
        quote_df.at[0, "week_52_high"], quote_df.at[0, "week_52_low"],
        quote_df.at[0, "change"], quote_df.at[0, "change_percentage"],
        quote_df.at[0, "bidsize"], quote_df.at[0, "bidexch"],
        convert_unix_to_datetime(quote_df.at[0, "bid_date"]), quote_df.at[0, "asksize"],
        quote_df.at[0, "askexch"], convert_unix_to_datetime(quote_df.at[0, "ask_date"]),
        quote_df.at[0, "exch"]
    ]

    # Add timesales data if available
    if timesales_data:
        values.extend([
            parse_timestamp(timesales_data['time']),
            convert_unix_to_datetime(timesales_data["timestamp"]),
            timesales_data['open'],
            timesales_data['high'],
            timesales_data['low'],
            timesales_data['close'],
            timesales_data['volume'],
            timesales_data['vwap']
        ])
    else:
        values.extend([None] * 8)  # Add None for all timesales fields if data is not available

    # Execute the query
    await conn.execute(query, *values)

    all_contract_quotes = await post_market_quotes(session, ticker, real_auth)

    if all_contract_quotes is not None:
        dividend_yield = await dividend_yield_cache.get_dividend_yield(conn, session, ticker, real_auth,
                                                                       current_price)
        options_df = process_option_quotes(all_contract_quotes, current_price, prevclose,dividend_yield,ticker)
        loop_start_time_eastern = loop_start_time.astimezone(eastern)

        options_df['fetch_timestamp'] = loop_start_time_eastern
        # print(options_df.columns)

        # Insert individual option data
        await insert_option_data(conn, options_df)
        # Calculate aggregated metrics
        if ticker in TICKERS_FOR_TRADE_ALGOS:
            all_agg_metrics = calculate_aggregated_metrics(options_df, current_price, prevclose)

        # Insert aggregated metrics
        # Iterate over aggregated metrics and insert each one
        # Insert aggregated metrics
            for agg_metrics in all_agg_metrics:
                exp_date = agg_metrics['expiration_date']  # Get the expiration date from the dictionary
                await insert_aggregated_metrics(conn, agg_metrics, ticker, loop_start_time, exp_date)  # Pass exp_date

    return prevclose, current_price, options_df, symbol_name

    # async def insert_option_data(conn, options_df):
    #     # Insert option data (similar to your existing code)
    #     await conn.executemany('''
    #         INSERT INTO csvimport.options (
    #             contract_id, underlying, expiration_date, strike, option_type,
    #             contract_size, description, expiration_type
    #         ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
    #         ON CONFLICT (contract_id) DO UPDATE SET
    #             underlying = EXCLUDED.underlying,
    #             expiration_date = EXCLUDED.expiration_date,
    #             strike = EXCLUDED.strike,
    #             option_type = EXCLUDED.option_type,
    #             contract_size = EXCLUDED.contract_size,
    #             description = EXCLUDED.description,
    #             expiration_type = EXCLUDED.expiration_type
    #     ''', [(row['contract_id'], row['underlying'], row['expiration_date'],
    #            row['strike'], row['option_type'], row['contract_size'],
    #            row['description'], row['expiration_type'])
    #           for _, row in options_df.iterrows()])
    #
    #     # Insert option quotes data (similar to your existing code)
    #     await conn.executemany('''
    #         INSERT INTO csvimport.option_quotes (
    #             contract_id, fetch_timestamp, root_symbol, last, change, volume,
    #             open, high, low, bid, ask, greeks, change_percentage, last_volume,
    #             trade_date, prevclose, bidsize, bidexch, bid_date, asksize, askexch,
    #             ask_date, open_interest, implied_volatility, realtime_calculated_greeks,
    #             risk_free_rate
    #         ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
    #                   $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26)
    #         ON CONFLICT (contract_id, fetch_timestamp) DO UPDATE SET
    #             root_symbol = EXCLUDED.root_symbol,
    #             last = EXCLUDED.last,
    #             change = EXCLUDED.change,
    #             volume = EXCLUDED.volume,
    #             open = EXCLUDED.open,
    #             high = EXCLUDED.high,
    #             low = EXCLUDED.low,
    #             bid = EXCLUDED.bid,
    #             ask = EXCLUDED.ask,
    #             greeks = EXCLUDED.greeks,
    #             change_percentage = EXCLUDED.change_percentage,
    #             last_volume = EXCLUDED.last_volume,
    #             trade_date = EXCLUDED.trade_date,
    #             prevclose = EXCLUDED.prevclose,
    #             bidsize = EXCLUDED.bidsize,
    #             bidexch = EXCLUDED.bidexch,
    #             bid_date = EXCLUDED.bid_date,
    #             asksize = EXCLUDED.asksize,
    #             askexch = EXCLUDED.askexch,
    #             ask_date = EXCLUDED.ask_date,
    #             open_interest = EXCLUDED.open_interest,
    #             implied_volatility = EXCLUDED.implied_volatility,
    #             realtime_calculated_greeks = EXCLUDED.realtime_calculated_greeks,
    #             risk_free_rate = EXCLUDED.risk_free_rate
    #     ''', [(row['contract_id'], row['fetch_timestamp'],
    #            row['root_symbol'], row['last'], row['change'], row['volume'],
    #            row['open'], row['high'], row['low'], row['bid'], row['ask'],
    #            json.dumps(row['greeks']), row['change_percentage'], row['last_volume'],
    #            convert_unix_to_datetime(row['trade_date']), row['prevclose'], row['bidsize'], row['bidexch'],
    #            convert_unix_to_datetime(row['bid_date']), row['asksize'], row['askexch'],
    #            convert_unix_to_datetime(row['ask_date']),
    #            row['open_interest'], row['implied_volatility'],
    #            json.dumps(row['realtime_calculated_greeks']) if row['realtime_calculated_greeks'] else None,
    #            row['risk_free_rate'])
    #           for _, row in options_df.iterrows()])
    #
    # async def insert_aggregated_metrics(conn, agg_metrics, ticker, fetch_timestamp):
    #     await conn.execute('''
    #         INSERT INTO csvimport.optimized_processed_option_data (
    #             symbol_name, fetch_timestamp, current_stock_price, current_sp_change_lac,
    #             max_pain, bonsai_ratio, pcr_vol, pcr_oi, itm_pcr_vol, itm_pcr_oi,
    #             itm_oi, total_oi, itm_contracts_percent, net_iv, net_itm_iv,
    #             closest_strike_to_cp, atm_iv
    #         ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
    #         ON CONFLICT (symbol_name, fetch_timestamp) DO UPDATE SET
    #             current_stock_price = EXCLUDED.current_stock_price,
    #             current_sp_change_lac = EXCLUDED.current_sp_change_lac,
    #             max_pain = EXCLUDED.max_pain,
    #             bonsai_ratio = EXCLUDED.bonsai_ratio,
    #             pcr_vol = EXCLUDED.pcr_vol,
    #             pcr_oi = EXCLUDED.pcr_oi,
    #             itm_pcr_vol = EXCLUDED.itm_pcr_vol,
    #             itm_pcr_oi = EXCLUDED.itm_pcr_oi,
    #             itm_oi = EXCLUDED.itm_oi,
    #             total_oi = EXCLUDED.total_oi,
    #             itm_contracts_percent = EXCLUDED.itm_contracts_percent,
    #             net_iv = EXCLUDED.net_iv,
    #             net_itm_iv = EXCLUDED.net_itm_iv,
    #             closest_strike_to_cp = EXCLUDED.closest_strike_to_cp,
    #             atm_iv = EXCLUDED.atm_iv
    #     ''', ticker, fetch_timestamp, agg_metrics['current_stock_price'], agg_metrics['current_sp_change_lac'],
    #                        agg_metrics['max_pain'], agg_metrics['bonsai_ratio'], agg_metrics['pcr_vol'],
    #                        agg_metrics['pcr_oi'],
    #                        agg_metrics['itm_pcr_vol'], agg_metrics['itm_pcr_oi'], agg_metrics['itm_oi'],
    #                        agg_metrics['total_oi'],
    #                        agg_metrics['itm_contracts_percent'], agg_metrics['net_iv'], agg_metrics['net_itm_iv'],
    #                        agg_metrics['closest_strike_to_cp'], agg_metrics['atm_iv'])
    # # Insert option data
    #     await conn.executemany('''
    #         INSERT INTO csvimport.options (
    #             contract_id, underlying, expiration_date, strike, option_type,
    #             contract_size, description, expiration_type
    #         ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
    #         ON CONFLICT (contract_id) DO UPDATE SET
    #             underlying = EXCLUDED.underlying,
    #             expiration_date = EXCLUDED.expiration_date,
    #             strike = EXCLUDED.strike,
    #             option_type = EXCLUDED.option_type,
    #             contract_size = EXCLUDED.contract_size,
    #             description = EXCLUDED.description,
    #             expiration_type = EXCLUDED.expiration_type
    #     ''', [(row['contract_id'], row['underlying'], row['expiration_date'],
    #            row['strike'], row['option_type'], row['contract_size'],
    #            row['description'], row['expiration_type'])
    #           for _, row in options_df.iterrows()])
    #
    #     # Insert option quotes data
    #     await conn.executemany('''
    #         INSERT INTO csvimport.option_quotes (
    #             contract_id, fetch_timestamp, root_symbol, last, change, volume,
    #             open, high, low, bid, ask, greeks, change_percentage, last_volume,
    #             trade_date, prevclose, bidsize, bidexch, bid_date, asksize, askexch,
    #             ask_date, open_interest, implied_volatility, realtime_calculated_greeks,
    #             risk_free_rate
    #         ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15,
    #                   $16, $17, $18, $19, $20, $21, $22, $23, $24, $25, $26)
    #         ON CONFLICT (contract_id, fetch_timestamp) DO UPDATE SET
    #             root_symbol = EXCLUDED.root_symbol,
    #             last = EXCLUDED.last,
    #             change = EXCLUDED.change,
    #             volume = EXCLUDED.volume,
    #             open = EXCLUDED.open,
    #             high = EXCLUDED.high,
    #             low = EXCLUDED.low,
    #             bid = EXCLUDED.bid,
    #             ask = EXCLUDED.ask,
    #             greeks = EXCLUDED.greeks,
    #             change_percentage = EXCLUDED.change_percentage,
    #             last_volume = EXCLUDED.last_volume,
    #             trade_date = EXCLUDED.trade_date,
    #             prevclose = EXCLUDED.prevclose,
    #             bidsize = EXCLUDED.bidsize,
    #             bidexch = EXCLUDED.bidexch,
    #             bid_date = EXCLUDED.bid_date,
    #             asksize = EXCLUDED.asksize,
    #             askexch = EXCLUDED.askexch,
    #             ask_date = EXCLUDED.ask_date,
    #             open_interest = EXCLUDED.open_interest,
    #             implied_volatility = EXCLUDED.implied_volatility,
    #             realtime_calculated_greeks = EXCLUDED.realtime_calculated_greeks,
    #             risk_free_rate = EXCLUDED.risk_free_rate
    #     ''', [(row['contract_id'], loop_start_time.astimezone(eastern),
    #            row['root_symbol'], row['last'], row['change'], row['volume'],
    #            row['open'], row['high'], row['low'], row['bid'], row['ask'],
    #            json.dumps(row['greeks']), row['change_percentage'], row['last_volume'],
    #            convert_unix_to_datetime(row['trade_date']), row['prevclose'], row['bidsize'], row['bidexch'],
    #            convert_unix_to_datetime(row['bid_date']), row['asksize'], row['askexch'], convert_unix_to_datetime(row['ask_date']),
    #            row['open_interest'], row['implied_volatility'],
    #            json.dumps(row['realtime_calculated_greeks']) if row['realtime_calculated_greeks'] else None, row['risk_free_rate'])
    #           for _, row in options_df.iterrows()])
    #
    # return prevclose, current_price, options_df, symbol_name
#     try:
#         # Upsert Symbol
#         insert_stmt = insert(Symbol).values(
#             symbol_name=ticker,
#             description=quote_df.at[0, "description"],
#             type=quote_df.at[0, "type"]
#         )
#         update_dict = {c.name: c for c in insert_stmt.excluded}
#         upsert_stmt = insert_stmt.on_conflict_do_update(
#             index_elements=[Symbol.symbol_name],
#             set_=update_dict
#         ).returning(Symbol.symbol_name)  # Return the symbol_name
#
#         result = db_session.execute(upsert_stmt)
#         symbol_name_result = result.one_or_none()  # Use one_or_none() to fetch 0 or 1 row
#
#         if not symbol_name_result:  # Check if the result is None
#             raise Exception(f"Failed to insert or update symbol {ticker}. Check for database constraints or errors.")
#
#         # print("close?",quote_df.at[0, "close"])
#     except Exception as e:
#         db_session.rollback()
#         print(f"Error handling symbol for {ticker}: {e}")
#         raise
#     # print(symbol_id)
# #TODO quotes will take multiple tickers/options as args. maybe be faster ?
#     stock_price_data = {
#         'symbol_name': symbol_name,
#         'fetch_timestamp': loop_start_time.astimezone(eastern),
#         'last_trade_timestamp': convert_unix_to_datetime(quote_df.at[0, "trade_date"]),
#         'last_trade_price': quote_df.at[0, "last"],
#         'current_bid': quote_df.at[0, "bid"],
#         'current_ask': quote_df.at[0, "ask"],
#         'daily_open': quote_df.at[0, "open"],
#         'daily_high': quote_df.at[0, "high"],
#         'daily_low': quote_df.at[0, "low"],
#         'previous_close': quote_df.at[0, "prevclose"],
#         'last_trade_volume': quote_df.at[0, "last_volume"],
#         'daily_volume': quote_df.at[0, "volume"],
#         'average_daily_volume': quote_df.at[0, "average_volume"],
#         'week_52_high': quote_df.at[0, "week_52_high"],
#         'week_52_low': quote_df.at[0, "week_52_low"],
#         'daily_change': quote_df.at[0, "change"],
#         'daily_change_percentage': quote_df.at[0, "change_percentage"],
#         'current_bidsize': quote_df.at[0, "bidsize"],
#         'bidexch': quote_df.at[0, "bidexch"],
#         'current_bid_date': convert_unix_to_datetime(quote_df.at[0, "bid_date"]),
#         'current_asksize': quote_df.at[0, "asksize"],
#         'askexch': quote_df.at[0, "askexch"],
#         'current_ask_date': convert_unix_to_datetime(quote_df.at[0, "ask_date"]),
#         'exch': quote_df.at[0, "exch"],
#     }
#     # Fetch timesales data
#
#     timesales_data = await get_timesales(session, ticker, lookback_minutes=1)
#     # print(timesales_data)
#     if timesales_data:
#         stock_price_data.update({
#             'last_1min_timesale': timesales_data['time'],
#             'last_1min_timestamp': convert_unix_to_datetime(timesales_data["timestamp"]),
#
#             'last_1min_open': timesales_data['open'],
#             'last_1min_high': timesales_data['high'],
#             'last_1min_low': timesales_data['low'],
#             'last_1min_close': timesales_data['close'],
#             'last_1min_volume': timesales_data['volume'],
#             'last_1min_vwap': timesales_data['vwap']
#         })
#
#     db_session.execute(
#         insert(SymbolQuote)
#         .values(stock_price_data)
#         .on_conflict_do_update(
#             constraint='symbol_quote_unique_constraint',
#             set_=stock_price_data
#         )
#     )
#     # db_session.commit()
#
#     #TODO ADD IV to the table for optionquotes?
#     all_contract_quotes = await post_market_quotes(session, ticker, real_auth)
#
#     if all_contract_quotes is not None:
#         # # Fetch risk-free rate (you need to implement this function)
#         # risk_free_rate = await get_risk_free_rate(session)
#         # Fetch dividend yield using the cache
#
#         dividend_yield = await dividend_yield_cache.get_dividend_yield(db_session, session, ticker, real_auth,
#                                                                        current_price)
#
#         # if dividend_yield > 0:
#         #     logger.info(f"Dividend yield for {ticker}: {dividend_yield:.4f} ({dividend_yield * 100:.2f}%)")
#         # else:
#         #     logger.info(f"No dividend yield found for {ticker}")
#         options_df = process_option_quotes(all_contract_quotes, current_price, prevclose, dividend_yield)
#         options_df['fetch_timestamp'] = loop_start_time
#
#         # Select and rename columns for the Option table
#         option_columns = ['contract_id','expiration_date', 'strike', 'option_type', 'underlying',
#                           'contract_size', 'description', 'expiration_type']#Got rid of exch
#         option_data_df = options_df[option_columns].copy()
#
#
#         # Reset the index to ensure it does not interfere with the upsert operation
#         option_data_df.reset_index(drop=True, inplace=True)
#
#         option_data_df.set_index(['contract_id'], inplace=True)
#         # option_data_df.to_csv("options_data.csv")
#
#
#         # Using pangres for Option table (without index_col)
#         upsert(
#             con=engine,
#             df=option_data_df,
#             table_name="options",
#             schema="csvimport",
#             if_row_exists='update',
#             create_table=False,
#         )
#
#         # Fetch the inserted options to create a mapping of (symbol_id, expiration_date, strike, option_type) to option_id
#         db_session.commit()
#
#         # First, ensure the column is datetime type
#         options_df['expiration_date'] = pd.to_datetime(options_df['expiration_date'])
#
#         # If it's not tz-aware, localize it to UTC
#         if options_df['expiration_date'].dt.tz is None:
#             options_df['expiration_date'] = options_df['expiration_date'].dt.tz_localize('UTC')
#
#         # Now convert to US/Eastern
#         options_df['expiration_date'] = options_df['expiration_date'].dt.tz_convert('US/Eastern')
#
#         # Create a list of dictionaries representing option quotes data
#         option_quotes_data = [
#             {
#                 "contract_id": row['contract_id'],
#                 "root_symbol": row['root_symbol'],
#                 "fetch_timestamp": loop_start_time.astimezone(eastern),
#                 "last": row["last"],
#                 "bid": row["bid"],
#                 "ask": row["ask"],
#                 "volume": row["volume"],
#                 "greeks": row["greeks"],
#                 "change_percentage": row["change_percentage"],
#                 "last_volume": row["last_volume"],
#                 "trade_date": convert_unix_to_datetime(row["trade_date"]),
#                 "prevclose": row["prevclose"],
#                 "bidsize": row["bidsize"],
#                 "bidexch": row["bidexch"],
#                 "bid_date": convert_unix_to_datetime(row["bid_date"]),
#                 "asksize": row["asksize"],
#                 "askexch": row["askexch"],
#                 "ask_date": convert_unix_to_datetime(row["ask_date"]),
#                 "open_interest": row["open_interest"],
#                 "change": row["change"],
#                 "open": row["open"],
#                 "high": row["high"],
#                 "low": row["low"],
#                 "implied_volatility": row["implied_volatility"],
#                 "realtime_calculated_greeks": row["realtime_calculated_greeks"],
#                 "risk_free_rate": row["risk_free_rate"]
#
#             }
#             for _, row in options_df.iterrows() if row['contract_id']
#         ]
#         # for item in option_quotes_data:
#         #     print(item.get('realtime_calculated_greeks', 'Not found'))
#
#         # # Use SQLAlchemy's Core API for bulk insert
#         # with engine.begin() as conn:
#         #     conn.execute(
#         #         insert(OptionQuote),
#         #         option_quotes_data
#         #     )
#         # Use execute_batch for bulk insert
#         with engine.begin() as conn:
#             await bulk_insert_option_quotes(conn, option_quotes_data)
#             conn.commit()
#         # Get technical analysis data
#         # ta_df = await technical_analysis.get_ta(session, ticker)
#         # if not ta_df.empty:
#         #     ta_data_list = ta_df.to_dict(orient='records')
#         #     for data in ta_data_list:
#         #         data["symbol_name"] = symbol_name
#         #
#         #     # Create DataFrame for bulk insert
#         #     ta_data_df = pd.DataFrame(ta_data_list)
#         #
#         #     # Use bulk_insert_mappings for faster inserts
#         #     with engine.begin() as conn:
#         #         conn.execute(
#         #             insert(TechnicalAnalysis),
#         #             ta_data_df.to_dict(orient="records")
#         #         )
#         #
#         # else:
#         #     print("TA DataFrame is empty")
#
#         db_session.commit()
#
#     # options_df.to_csv("options_df_test.csv")
#     return prevclose, current_price, options_df, symbol_name
