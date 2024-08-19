import pandas as pd
from matplotlib import pyplot as plt

from db_schema_models import Symbol,SymbolQuote, Option, OptionQuote, ProcessedOptionData, TechnicalAnalysis
import numpy as np
from sqlalchemy import func, select
from db_schema_models import Symbol, SymbolQuote, Option, OptionQuote, ProcessedOptionData, TechnicalAnalysis
from collections import defaultdict

async def calculate_in_the_money_pcr(session, ticker, current_time):
    # Fetch the latest stock quote using SymbolQuote
    stock_quote = session.query(SymbolQuote).filter_by(symbol_name=ticker).order_by(SymbolQuote.fetch_timestamp.desc()).first()

    if not stock_quote:
        return 0  # Or handle the case where there's no stock quote

    # Fetch option contracts and their quotes for the given ticker and timestamp
    option_contracts_with_quotes = session.query(Option, OptionQuote).join(OptionQuote).filter(
        Option.underlying == ticker,
        OptionQuote.fetch_timestamp == current_time
    ).all()

    # Now you can use stock_quote.last_price for comparison
    in_the_money_puts = [
        option_quote for option, option_quote in option_contracts_with_quotes
        if option.option_type == 'put' and option.strike > stock_quote.last_price
    ]
    in_the_money_calls = [
        option_quote for option, option_quote in option_contracts_with_quotes
        if option.option_type == 'call' and option.strike < stock_quote.last_price
    ]

    # Calculate the ratio
    if not in_the_money_calls:
        return float('inf')  # Or handle the case where there are no in-the-money calls

    put_call_ratio = len(in_the_money_puts) / len(in_the_money_calls)
    return put_call_ratio

async def calculate_bonsai_ratios(session, ticker, current_time):
    # Fetch option contracts and their quotes for the given ticker and timestamp
    option_contracts_with_quotes = session.query(Option, OptionQuote).join(OptionQuote).filter(
        Option.underlying == ticker,
        OptionQuote.fetch_timestamp == current_time
    ).all()

    # Fetch the latest stock quote
    stock_quote = session.query(SymbolQuote).filter_by(symbol_name=ticker).order_by(SymbolQuote.fetch_timestamp.desc()).first()

    if not stock_quote:
        return None, None  # Or handle the case where there's no stock quote

    # Calculate ITM and all volumes and open interests
    ITM_PutsVol = sum(oq.volume for o, oq in option_contracts_with_quotes if o.option_type == 'put' and o.strike > stock_quote.last_price)
    all_PutsVol = sum(oq.volume for o, oq in option_contracts_with_quotes if o.option_type == 'put')
    ITM_PutsOI = sum(oq.open_interest for o, oq in option_contracts_with_quotes if o.option_type == 'put' and o.strike > stock_quote.last_price)
    all_PutsOI = sum(oq.open_interest for o, oq in option_contracts_with_quotes if o.option_type == 'put')
    ITM_CallsVol = sum(oq.volume for o, oq in option_contracts_with_quotes if o.option_type == 'call' and o.strike < stock_quote.last_price)
    all_CallsVol = sum(oq.volume for o, oq in option_contracts_with_quotes if o.option_type == 'call')
    ITM_CallsOI = sum(oq.open_interest for o, oq in option_contracts_with_quotes if o.option_type == 'call' and o.strike < stock_quote.last_price)
    all_CallsOI = sum(oq.open_interest for o, oq in option_contracts_with_quotes if o.option_type == 'call')

    # Calculate Bonsai_Ratio
    if (ITM_PutsVol == 0) or (all_PutsVol == 0) or (ITM_PutsOI == 0) or (all_PutsOI == 0) or (ITM_CallsVol == 0) or (all_CallsVol == 0) or (ITM_CallsOI == 0) or (all_CallsOI == 0):
        Bonsai_Ratio = None
    else:
        Bonsai_Ratio = ((ITM_PutsVol / all_PutsVol) * (ITM_PutsOI / all_PutsOI)) / ((ITM_CallsVol / all_CallsVol) * (ITM_CallsOI / all_CallsOI))

    # Calculate Bonsai2_Ratio
    if (all_PutsVol == 0) or (ITM_PutsVol == 0) or (all_PutsOI == 0) or (ITM_PutsOI == 0) or (all_CallsVol == 0) or (ITM_CallsVol == 0) or (all_CallsOI == 0) or (ITM_CallsOI == 0):
        Bonsai2_Ratio = None
    else:
        Bonsai2_Ratio = ((all_PutsVol / ITM_PutsVol) / (all_PutsOI / ITM_PutsOI)) * ((all_CallsVol / ITM_CallsVol) / (all_CallsOI / ITM_CallsOI))

    return Bonsai_Ratio, Bonsai2_Ratio

async def calculate_pcr_ratios(session, ticker, current_time):
    # Fetch option contracts and their quotes for the given ticker and timestamp
    option_contracts_with_quotes = session.query(Option, OptionQuote).join(OptionQuote).filter(
        Option.underlying == ticker,
        OptionQuote.fetch_timestamp == current_time
    ).all()

    # Calculate total put and call volumes and open interests
    put_volume = sum(oq.volume for o, oq in option_contracts_with_quotes if o.option_type == 'put')
    call_volume = sum(oq.volume for o, oq in option_contracts_with_quotes if o.option_type == 'call')
    put_oi = sum(oq.open_interest for o, oq in option_contracts_with_quotes if o.option_type == 'put')
    call_oi = sum(oq.open_interest for o, oq in option_contracts_with_quotes if o.option_type == 'call')

    # Calculate PCR ratios
    pcr_vol = put_volume / call_volume if call_volume != 0 else None
    pcr_oi = put_oi / call_oi if call_oi != 0 else None

    return pcr_vol, pcr_oi

async def calculate_net_iv(session, ticker, current_time):
    # Fetch option contracts and their quotes for the given ticker and timestamp
    option_contracts_with_quotes = session.query(Option, OptionQuote).join(OptionQuote).filter(
        Option.underlying == ticker,
        OptionQuote.fetch_timestamp == current_time
    ).all()

    # Calculate total call and put IVs
    call_iv = sum(oq.greeks['mid_iv'] for o, oq in option_contracts_with_quotes if o.option_type == 'call' and oq.greeks and 'mid_iv' in oq.greeks)
    put_iv = sum(oq.greeks['mid_iv'] for o, oq in option_contracts_with_quotes if o.option_type == 'put' and oq.greeks and 'mid_iv' in oq.greeks)

    # Calculate Net IV
    net_iv = call_iv - put_iv

    return net_iv
async def calculate_pcr_ratios(session, ticker, current_time):
    # Fetch option contracts and their quotes for the given ticker and timestamp
    option_contracts_with_quotes = session.query(Option, OptionQuote).join(OptionQuote).filter(
        Option.underlying == ticker,
        OptionQuote.fetch_timestamp == current_time
    ).all()

    # Calculate total put and call volumes and open interests
    put_volume = sum(oq.volume for o, oq in option_contracts_with_quotes if o.option_type == 'put')
    call_volume = sum(oq.volume for o, oq in option_contracts_with_quotes if o.option_type == 'call')
    put_oi = sum(oq.open_interest for o, oq in option_contracts_with_quotes if o.option_type == 'put')
    call_oi = sum(oq.open_interest for o, oq in option_contracts_with_quotes if o.option_type == 'call')

    # Calculate PCR ratios
    pcr_vol = put_volume / call_volume if call_volume != 0 else None
    pcr_oi = put_oi / call_oi if call_oi != 0 else None

    return pcr_vol, pcr_oi

async def calculate_net_iv(session, ticker, current_time):
    # Fetch option contracts and their quotes for the given ticker and timestamp
    option_contracts_with_quotes = session.query(Option, OptionQuote).join(OptionQuote).filter(
        Option.underlying == ticker,
        OptionQuote.fetch_timestamp == current_time
    ).all()

    # Calculate total call and put IVs
    call_iv = sum(oq.greeks['mid_iv'] for o, oq in option_contracts_with_quotes if o.option_type == 'call' and oq.greeks and 'mid_iv' in oq.greeks)
    put_iv = sum(oq.greeks['mid_iv'] for o, oq in option_contracts_with_quotes if o.option_type == 'put' and oq.greeks and 'mid_iv' in oq.greeks)

    # Calculate Net IV
    net_iv = call_iv - put_iv

    return net_iv


async def calculate_maximum_pain(session, ticker, current_time):
    # Fetch option contracts and their quotes for the given ticker and timestamp
    option_contracts_with_quotes = session.query(Option, OptionQuote).join(OptionQuote).filter(
        Option.underlying == ticker,
        OptionQuote.fetch_timestamp == current_time
    ).all()

    # Group options by expiration date and strike price
    expirations = defaultdict(lambda: defaultdict(lambda: {'call_oi': 0, 'put_oi': 0}))
    for option, quote in option_contracts_with_quotes:
        expirations[option.expiration_date][option.strike][f'{option.option_type}_oi'] += quote.open_interest

    max_pain_by_expiration = {}

    for exp_date, strikes in expirations.items():
        max_pain = 0
        min_pain = float('inf')

        for potential_price in strikes.keys():
            pain = sum(
                max(potential_price - strike, 0) * oi['call_oi'] +
                max(strike - potential_price, 0) * oi['put_oi']
                for strike, oi in strikes.items()
            )

            if pain < min_pain:
                min_pain = pain
                max_pain = potential_price

        max_pain_by_expiration[exp_date] = max_pain

    return max_pain_by_expiration

async def calculate_historical_max_pain(session, ticker, start_date, end_date):
    # Fetch all relevant option data within the date range
    option_data = session.query(Option, OptionQuote).join(OptionQuote).filter(
        Option.underlying == ticker,
        OptionQuote.fetch_timestamp.between(start_date, end_date)
    ).all()

    # Group data by fetch_timestamp and expiration_date
    grouped_data = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {'call_oi': 0, 'put_oi': 0})))
    for option, quote in option_data:
        grouped_data[quote.fetch_timestamp][option.expiration_date][option.strike][f'{option.option_type}_oi'] += quote.open_interest

    # Calculate max pain for each fetch_timestamp and expiration_date
    results = []
    for fetch_date, expirations in grouped_data.items():
        for exp_date, strikes in expirations.items():
            max_pain = 0
            min_pain = float('inf')
            for potential_price in strikes.keys():
                pain = sum(
                    max(potential_price - strike, 0) * oi['call_oi'] +
                    max(strike - potential_price, 0) * oi['put_oi']
                    for strike, oi in strikes.items()
                )
                if pain < min_pain:
                    min_pain = pain
                    max_pain = potential_price
            results.append({
                'fetch_date': fetch_date,
                'expiration_date': exp_date,
                'max_pain': max_pain
            })

    # Convert results to DataFrame
    df = pd.DataFrame(results)
    return df

def plot_historical_max_pain(df):
    # Pivot the dataframe to have expiration dates as columns
    pivot_df = df.pivot(index='fetch_date', columns='expiration_date', values='max_pain')

    # Plot
    plt.figure(figsize=(15, 10))
    for column in pivot_df.columns:
        plt.plot(pivot_df.index, pivot_df[column], label=column.strftime('%Y-%m-%d'))

    plt.title('Historical Max Pain by Expiration Date')
    plt.xlabel('Fetch Date')
    plt.ylabel('Max Pain Strike Price')
    plt.legend(title='Expiration Date', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.grid(True)
    return plt

# Example usage
async def analyze_historical_max_pain(session, ticker, start_date, end_date):
    df = await calculate_historical_max_pain(session, ticker, start_date, end_date)
    plt = plot_historical_max_pain(df)
    plt.savefig(f'{ticker}_historical_max_pain.png')
    plt.close()
    return df
async def calculate_pcr_oi(session, ticker, current_time):
    # Fetch option contracts and their quotes for the given ticker and timestamp
    option_contracts_with_quotes = session.query(Option, OptionQuote).join(OptionQuote).filter(
        Option.underlying == ticker,
        OptionQuote.fetch_timestamp == current_time
    ).all()

    # Calculate total put and call open interests
    put_oi = sum(oq.open_interest for o, oq in option_contracts_with_quotes if o.option_type == 'put')
    call_oi = sum(oq.open_interest for o, oq in option_contracts_with_quotes if o.option_type == 'call')

    # Calculate PCR for open interest
    pcr_oi = put_oi / call_oi if call_oi != 0 else None

    return pcr_oi

async def calculate_itm_pcr_oi(session, ticker, current_time):
    # Fetch the latest stock quote
    stock_quote = session.query(SymbolQuote).filter_by(symbol_name=ticker).order_by(SymbolQuote.fetch_timestamp.desc()).first()

    if not stock_quote:
        return None  # Or handle the case where there's no stock quote

    # Fetch option contracts and their quotes for the given ticker and timestamp
    option_contracts_with_quotes = session.query(Option, OptionQuote).join(OptionQuote).filter(
        Option.underlying == ticker,
        OptionQuote.fetch_timestamp == current_time
    ).all()

    # Calculate ITM put and call open interests
    itm_put_oi = sum(oq.open_interest for o, oq in option_contracts_with_quotes if o.option_type == 'put' and o.strike > stock_quote.last_price)
    itm_call_oi = sum(oq.open_interest for o, oq in option_contracts_with_quotes if o.option_type == 'call' and o.strike < stock_quote.last_price)

    # Calculate ITM PCR for open interest
    itm_pcr_oi = itm_put_oi / itm_call_oi if itm_call_oi != 0 else None

    return itm_pcr_oi