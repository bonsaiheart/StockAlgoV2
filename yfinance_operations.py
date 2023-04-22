
from pathlib import Path

import numpy as np
import yfinance as yf
import pandas as pd
import datetime as dt

YYMMDD = dt.datetime.today().strftime('%y%m%d')

def get_options_data(ticker):
# get last adjusted close

    tickerinfo = yf.Ticker(ticker).history_metadata

    LAC = tickerinfo['previousClose']
    print(f"LAC: ${LAC}")
    CurrentPrice = tickerinfo['regularMarketPrice']
    print(f'Current Price: ${CurrentPrice}')
    price_change_percent= (CurrentPrice - LAC) / LAC *100
    StockLastTradeTime = tickerinfo['regularMarketTime']
    StockLastTradeTime = tickerinfo['regularMarketTime'].time().strftime('%H%M')
    print(f"Last Trade Time: {StockLastTradeTime}")
    # convert the string to a datetime object
    # StockLastTradeTime = dt.datetime.strptime(StockLastTradeTime, '%I:%M%p')

    # print the datetime object

    exp_dates = yf.Ticker(ticker).options

    callsChain = []
    putsChain = []

    for exp_date in exp_dates:

        chain = yf.Ticker(ticker).option_chain(exp_date)
        newcall = chain.calls
        newput = chain.puts
        callsChain.append(newcall)
        putsChain.append(newput)



###TODO Note that Yfinance doesn't update previous day close until.. after 845, ill keep chekcing.
    ###CALLS OPS
    calls_df = pd.concat(callsChain,ignore_index=True)
    calls_df['lastTrade'] = pd.to_datetime(calls_df['lastTradeDate'])
    calls_df['lastTrade'] = calls_df['lastTrade'].dt.strftime('%y%m%d_%H%M')
    calls_df['expDate'] = calls_df['contractSymbol'].str[-15:-9]
    ###TODO for puts, use current price - strike i think.
    calls_df['dollarsFromStrike'] = abs(calls_df['strike'] - LAC)
    calls_df['dollarsFromStrikexOI'] = calls_df['dollarsFromStrike'] * calls_df['openInterest']
    calls_df['lastContractPricexOI'] = calls_df['lastPrice'] * calls_df['openInterest']
    calls_df.rename(columns={'contractSymbol':'c_contractSymbol','lastTradeDate':'c_lastTrade', 'lastPrice':'c_lastPrice', 'bid':'c_bid', 'ask':'c_ask',
           'change':'c_change', 'percentChange':'c_percentChange', 'volume':'c_volume', 'openInterest':'c_openInterest',
           'impliedVolatility':'c_impliedVolatility', 'inTheMoney':'c_inTheMoney',
           'lastTrade':'c_lastTrade',  'dollarsFromStrike':'c_dollarsFromStrike', 'dollarsFromStrikexOI':'c_dollarsFromStrikexOI',
           'lastContractPricexOI':'c_lastContractPricexOI'}, inplace=True)
    ###PUTS OPS


    puts_df = pd.concat(putsChain, ignore_index=True)
    puts_df['lastTrade'] = pd.to_datetime(puts_df['lastTradeDate'])
    puts_df['lastTrade'] = puts_df['lastTrade'].dt.strftime('%y%m%d_%H%M')
    puts_df['expDate'] = puts_df['contractSymbol'].str[-15:-9]
    ###TODO for puts, use current price - strike i think.
    puts_df['dollarsFromStrike'] =   abs(puts_df['strike'] - LAC)
    puts_df['dollarsFromStrikexOI'] = puts_df['dollarsFromStrike'] * puts_df['openInterest']
    puts_df['lastContractPricexOI'] = puts_df['lastPrice'] * puts_df['openInterest']

    puts_df.rename(columns={'contractSymbol':'p_contractSymbol','lastTradeDate': 'p_lastTrade', 'lastPrice': 'p_lastPrice', 'bid': 'p_bid', 'ask': 'p_ask',
           'change':'p_change', 'percentChange':'p_percentChange', 'volume':'p_volume', 'openInterest':'p_openInterest',
           'impliedVolatility':'p_impliedVolatility', 'inTheMoney':'p_inTheMoney',
           'lastTrade':'p_lastTrade',  'dollarsFromStrike':'p_dollarsFromStrike', 'dollarsFromStrikexOI':'p_dollarsFromStrikexOI',
           'lastContractPricexOI':'p_lastContractPricexOI'}, inplace=True)

    combined = pd.merge(puts_df, calls_df, on=['expDate', 'strike'])

    combined = combined[['expDate','strike','c_contractSymbol', 'c_lastTrade', 'c_lastPrice', 'c_bid', 'c_ask',
           'c_change', 'c_percentChange', 'c_volume', 'c_openInterest',
           'c_impliedVolatility', 'c_inTheMoney', 'c_lastTrade',
           'c_dollarsFromStrike', 'c_dollarsFromStrikexOI',
           'c_lastContractPricexOI', 'p_contractSymbol', 'p_lastTrade',
           'p_lastPrice', 'p_bid', 'p_ask', 'p_change', 'p_percentChange',
           'p_volume', 'p_openInterest', 'p_impliedVolatility',
           'p_inTheMoney', 'p_lastTrade', 'p_dollarsFromStrike',
           'p_dollarsFromStrikexOI', 'p_lastContractPricexOI'
           ]]
    combined.rename(columns={'expDate':'ExpDate','strike':'Strike', 'c_lastPrice':'Call_LastPrice', 'c_percentChange':'Call_PercentChange', 'c_volume':'Call_Volume', 'c_openInterest':'Call_OI',
           'c_impliedVolatility':'Call_IV',
           'c_dollarsFromStrike':'Calls_dollarsFromStrike', 'c_dollarsFromStrikexOI':'Calls_dollarsFromStrikeXoi',
           'c_lastContractPricexOI':'Calls_lastPriceXoi',
           'p_lastPrice':'Put_LastPrice',
           'p_volume':'Put_Volume', 'p_openInterest':'Put_OI', 'p_impliedVolatility':'Put_IV',
            'p_dollarsFromStrike':'Puts_dollarsFromStrike',
           'p_dollarsFromStrikexOI':'Puts_dollarsFromStrikeXoi', 'p_lastContractPricexOI':'Puts_lastPriceXoi'}, inplace=True)


    output_dir = Path(f'data/optionchain/{ticker}/{YYMMDD}')
    output_dir.mkdir(mode=0o755, parents=True, exist_ok=True)

    combined.to_csv(f'data/optionchain/{ticker}/{YYMMDD}/{ticker}_{YYMMDD}_{StockLastTradeTime}.csv')
    ###strike, exp, call last price, call oi, iv,vol, $ from strike, dollars from strike x OI, last price x OI
    return (LAC, CurrentPrice,price_change_percent, StockLastTradeTime)



def perform_operations(ticker,last_adj_close, current_price, price_change_percent,StockLastTradeTime):
    results = []

    data = pd.read_csv(f'data/optionchain/{ticker}/{YYMMDD}/{ticker}_{YYMMDD}_{StockLastTradeTime}.csv')

    groups = data.groupby("ExpDate")
    # divide into groups by exp date, call info from group.
    for exp_date, group in groups:
        pain_list = []
        strike_LASTPRICExOI_list = []
        call_LASTPRICExOI_list = []
        put_LASTPRICExOI_list = []
        # itmDFSxOI_list = []
        strike_DFSxOI_list = []
        call_DFSxOI_list = []
        put_DFSxOI_list = []
        strike = group['Strike']
        # pain is ITM puts/calls
        # for each strike,  all the dollar values of the puts beneath.
        # calls_OI_dict = group.loc[group['Call_OI'] >= 0, ["Strike", 'Call_OI']].set_index('Strike').to_dict()
        # puts_OI_dict = group.loc[group['Put_OI'] >= 0, ["Strike", 'Put_OI']].set_index('Strike').to_dict()
        calls_LASTPRICExOI_dict = group.loc[
            group['Calls_lastPriceXoi'] >= 0, ["Strike", 'Calls_lastPriceXoi']].set_index('Strike').to_dict()
        puts_LASTPRICExOI_dict = group.loc[group['Puts_lastPriceXoi'] >= 0, ["Strike", 'Puts_lastPriceXoi']].set_index(
            'Strike').to_dict()
        calls_DFSxOI_dict = group.loc[
            group['Calls_dollarsFromStrikeXoi'] >= 0, ["Strike", 'Calls_dollarsFromStrikeXoi']].set_index(
            'Strike').to_dict()
        puts_DFSxOI_dict = group.loc[
            group['Puts_dollarsFromStrikeXoi'] >= 0, ["Strike", 'Puts_dollarsFromStrikeXoi']].set_index(
            'Strike').to_dict()
        # itm_calls_OI_dict = group.loc[
        #     (group['Strike'] < last_adj_close) & (~group['Call_OI'].isnull()), ["Strike", 'Call_OI']].set_index(
        #     'Strike').to_dict()
        # itm_puts_OI_dict = group.loc[
        #     (group['Strike'] > last_adj_close) & (~group['Put_OI'].isnull()), ["Strike", 'Put_OI']].set_index(
        #     'Strike').to_dict()
        ITM_CallsVol = group.loc[(group["Strike"] <= current_price), 'Call_Volume'].sum()
        ITM_PutsVol = group.loc[(group["Strike"] >= current_price), 'Put_Volume'].sum()
        ITM_CallsOI = group.loc[(group["Strike"] <= current_price), 'Call_OI'].sum()
        ITM_PutsOI = group.loc[(group["Strike"] >= current_price), 'Put_OI'].sum()
        ITM_OI = ITM_CallsOI + ITM_PutsOI
        all_CallsVol = group.Call_Volume.sum()
        all_PutsVol = group.Put_Volume.sum()
        # ITM_Call_Vol = group.loc[(group["Strike"] < last_adj_close), 'Call_Volume'].sum()
        # ITM_Put_Vol = group.loc[(group["Strike"] < last_adj_close), 'Put_Volume'].sum()
        all_CallsOI = group.Call_OI.sum()
        all_PutsOI = group.Put_OI.sum()
        all_OI = all_PutsOI + all_CallsOI

        ITM_Call_IV = group.loc[(group["Strike"] <= current_price), 'Call_IV'].sum()
        ITM_Put_IV = group.loc[(group["Strike"] >= current_price), 'Put_IV'].sum()
        Call_IV = group['Call_IV'].sum()
        Put_IV = group['Put_IV'].sum()
        ITM_Avg_Net_IV = ITM_Call_IV - ITM_Put_IV
        Net_IV = Call_IV - Put_IV

        if all_CallsVol != 0 and not np.isnan(all_CallsVol):
            PC_Ratio_Vol = all_PutsVol / all_CallsVol
        else:
            PC_Ratio_Vol = np.nan
            continue
        if ITM_CallsVol != 0 and not np.isnan(ITM_CallsVol):
            ITM_PC_Ratio_Vol = ITM_PutsVol / ITM_CallsVol
        else:
            ITM_PC_Ratio_Vol = np.nan
            continue


        if all_CallsOI != 0 and not np.isnan(all_CallsOI):
            PC_Ratio_OI = all_PutsOI / all_CallsOI
        else:
            PC_Ratio_OI = np.nan
            continue
        if ITM_CallsOI != 0 and not np.isnan(ITM_CallsOI):
            ITM_PC_Ratio_OI = ITM_PutsOI / ITM_CallsOI
        else:
            ITM_PC_Ratio_OI = np.nan
            continue
        DFSxOI_dict = group.loc[group['Puts_dollarsFromStrikeXoi'] >= 0, ["Strike", 'Puts_dollarsFromStrikeXoi']].set_index('Strike').to_dict()

        # All_PC_Ratio =
        # Money_weighted_PC_Ratio =
        ###TODO figure out WHEN this needs to run... probalby after 6pm eastern and before mrkt open.  remove otm
###TODO add highest premium puts/calls, greeks corelation?

        ###TODO correlate volume and IV, high volume high iv = contracts being bought, high volume, low vol. = contracts being sold.

        for strikeprice in strike:
            itmCalls_dollarsFromStrikeXoiSum = group.loc[
                (group["Strike"] < strikeprice), 'Calls_dollarsFromStrikeXoi'].sum()
            itmPuts_dollarsFromStrikeXoiSum = group.loc[
                (group["Strike"] > strikeprice), 'Puts_dollarsFromStrikeXoi'].sum()

            call_LASTPRICExOI = calls_LASTPRICExOI_dict.get("Calls_lastPriceXoi", {}).get(strikeprice, 0)
            put_LASTPRICExOI = puts_LASTPRICExOI_dict.get("Puts_lastPriceXoi", {}).get(strikeprice, 0)
            # call_DFSxOI = calls_DFSxOI_dict.get("Calls_dollarsFromStrikeXoi", {}).get(strikeprice, 0)
            # put_DFSxOI = puts_DFSxOI_dict.get("Puts_dollarsFromStrikeXoi", {}).get(strikeprice, 0)
            pain_value = itmPuts_dollarsFromStrikeXoiSum + itmCalls_dollarsFromStrikeXoiSum
            pain_list.append((strikeprice, pain_value))
            strike_LASTPRICExOI = call_LASTPRICExOI + put_LASTPRICExOI
            strike_LASTPRICExOI_list.append((strikeprice, strike_LASTPRICExOI))
            call_LASTPRICExOI_list.append((strikeprice, call_LASTPRICExOI))
            put_LASTPRICExOI_list.append((strikeprice, put_LASTPRICExOI))
            # strike_DFSxOI = call_DFSxOI + put_DFSxOI

            # strike_DFSxOI_list.append((strikeprice, strike_DFSxOI))
            # call_DFSxOI_list.append((strikeprice, call_DFSxOI))
            # put_DFSxOI_list.append((strikeprice, put_DFSxOI))
        # print(f'ITM put/call ratio : {ticker} - {ITM_PC_Ratio_OI}')
        # ITM_PC_Ratio_OI = group["Put"].sum() / group["Call"].sum()

        highest_premium_strike = max(strike_LASTPRICExOI_list, key=lambda x: x[1])[0]
        highest_premium_call = max(call_LASTPRICExOI_list, key=lambda x: x[1])[0]
        highest_premium_put = max(put_LASTPRICExOI_list, key=lambda x: x[1])[0]
        max_pain = min(pain_list, key=lambda x: x[1])[0]

        # max_DFSxOI = max(strike_DFSxOI_list, key=lambda x: x[1])[0]
        # max_call_DFSxOI = max(call_DFSxOI_list, key=lambda x: x[1])[0]
        # max_put_DFSxOI = max(put_DFSxOI_list, key=lambda x: x[1])[0]
        # implied_percentage_move= ((max_pain - last_adj_close) / last_adj_close) * 100
        top_five_calls = group.loc[group['Call_OI'] > 0].sort_values(by='Call_OI', ascending=False).head(5)
        top_five_calls_dict = top_five_calls[['Strike', 'Call_OI']].set_index('Strike').to_dict()['Call_OI']
        # highestTotalOI = group.loc[group['TotalOI'] > 0].sort_values(by='TotalOI', ascending=False).head(2)
        # highestTotalOI_dict = highestTotalOI[['Strike', 'TotalOI']].set_index('Strike').to_dict()['TotalOI']
        top_five_puts = group.loc[group['Put_OI'] > 0].sort_values(by='Put_OI', ascending=False).head(5)
        top_five_puts_dict = top_five_puts[['Strike', 'Put_OI']].set_index('Strike').to_dict()['Put_OI']

        Net_Call_IV_at_MP = group.loc[(group["Strike"] == max_pain), 'Call_IV'].sum()
        Net_Put_IV_at_MP = group.loc[(group["Strike"] == max_pain), 'Put_IV'].sum()
        Net_IV_at_MP = Net_Call_IV_at_MP - Net_Put_IV_at_MP
        ### FINDING CLOSEST STRIKE TO LAC
        # target number from column A
        # calculate difference between target and each value in column B
        data['strike_lac_diff'] = group['Strike'].apply(lambda x: abs(x - last_adj_close))
        # find index of row with smallest difference
        smallest_change_from_lac = data['strike_lac_diff'].idxmin()
        closest_strike = group.loc[smallest_change_from_lac, 'Strike']


        # extract value from column B using index

        mp_put_vol = group.loc[group["Strike"] == max_pain, "Put_Volume"].values[0]
        mp_call_vol = group.loc[group["Strike"] == max_pain, "Call_Volume"].values[0]

        if np.isnan(mp_put_vol) or np.isnan(mp_call_vol) or mp_call_vol == 0:
            PC_Ratio_Vol_atMP = np.nan
        else:
            PC_Ratio_Vol_atMP = mp_put_vol / mp_call_vol

        mp_put_OI = group.loc[group["Strike"] == max_pain, "Put_OI"].values[0]
        mp_call_OI = group.loc[group["Strike"] == max_pain, "Call_OI"].values[0]
        if np.isnan(mp_put_OI) or np.isnan(mp_call_OI) or mp_call_OI == 0:
            PC_Ratio_OI_atMP = np.nan
        else:
            PC_Ratio_OI_atMP = mp_put_OI / mp_call_OI


     ###Same but for lac
        # lac_put_vol = group.loc[group["Strike"] == closest_strike, "Put_Volume"].values[0]
        # lac_call_vol = group.loc[group["Strike"] == closest_strike, "Call_Volume"].values[0]
        # if np.isnan(lac_put_vol) or np.isnan(lac_call_vol) or lac_call_vol == 0:
        #     PC_Ratio_Vol_atLAC = np.nan
        # else:
        #     PC_Ratio_Vol_atLAC = lac_put_vol / lac_call_vol
        #
        # lac_put_OI = group.loc[group["Strike"] == closest_strike, "Put_OI"].values[0]
        # lac_call_OI = group.loc[group["Strike"] == closest_strike, "Call_OI"].values[0]
        # if np.isnan(lac_put_OI) or np.isnan(lac_call_OI) or lac_call_OI == 0:
        #     PC_Ratio_OI_atLAC = np.nan
        # else:
        #     PC_Ratio_OI_atLAC = lac_put_OI / lac_call_OI

###TODO use above/below lac strikes instead of just closest.
        lac_put_vol = group.loc[group["Strike"] == closest_strike, "Put_Volume"].values[0]
        lac_call_vol = group.loc[group["Strike"] == closest_strike, "Call_Volume"].values[0]
        if np.isnan(lac_put_vol) or np.isnan(lac_call_vol) or lac_call_vol == 0:
            PC_Ratio_Vol_Closest_Strike_LAC = np.nan
        else:
            PC_Ratio_Vol_Closest_Strike_LAC = lac_put_vol / lac_call_vol

        lac_put_OI = group.loc[group["Strike"] == closest_strike, "Put_OI"].values[0]
        lac_call_OI = group.loc[group["Strike"] == closest_strike, "Call_OI"].values[0]
        if np.isnan(lac_put_OI) or np.isnan(lac_call_OI) or lac_call_OI == 0:
            PC_Ratio_OI_Closest_Strike_LAC = np.nan
        else:
            PC_Ratio_OI_Closest_Strike_LAC = lac_put_OI / lac_call_OI

        if np.isnan(PC_Ratio_Vol_atMP) or np.isnan(PC_Ratio_OI_atMP) or PC_Ratio_OI_atMP == 0:
            PCR_vol_OI_at_MP = np.nan
        else:
            PCR_vol_OI_at_MP = round((PC_Ratio_Vol_atMP / PC_Ratio_OI_atMP), 3)

        if np.isnan(PC_Ratio_Vol_Closest_Strike_LAC) or np.isnan(
                PC_Ratio_OI_Closest_Strike_LAC) or PC_Ratio_OI_Closest_Strike_LAC == 0:
            PCR_vol_OI_at_LAC = np.nan
        else:
            PCR_vol_OI_at_LAC = round((PC_Ratio_Vol_Closest_Strike_LAC / PC_Ratio_OI_Closest_Strike_LAC), 3)

        Call_IV_Closest_Strike_LAC = group.loc[(group["Strike"] == closest_strike), 'Call_IV'].sum()
        Put_IV_Closest_Strike_LAC = group.loc[(group["Strike"] == closest_strike), 'Put_IV'].sum()
        Net_IV_Closest_Strike_LAC = Call_IV_Closest_Strike_LAC - Put_IV_Closest_Strike_LAC

        results.append({
###TODO change all price data to percentage change?
            ###TODO RUNTIME WARNING C:\Users\natha\PycharmProjects\IntradayOptionTrading\dataHarvester.py:296      : RuntimeWarning: invalid value encountered in scalar divide
            ###TODO change closest strike to average of closest above/closest below
            'ExpDate': exp_date,
            'Current Stock Price': float(current_price),
            'Current SP % Change(LAC)': round(float(price_change_percent), 2),
            # 'IV 30': iv30,
            # 'IV 30 % change': iv30_change_percent,
            'Maximum Pain': max_pain,
            'Bonsai Ratio': round(((ITM_PutsVol / all_PutsVol) * (ITM_PutsOI / all_PutsOI)) / (
                        (ITM_CallsVol / all_CallsVol) * (ITM_CallsOI / all_CallsOI)), 3),
            'Bonsai Ratio 2': round(((all_PutsVol / ITM_PutsVol) / (all_PutsOI / ITM_PutsOI)) * (
                        (all_CallsVol / ITM_CallsVol) / (all_CallsOI / ITM_CallsOI)), 3),
            # 'Maximum Pain % -LAC': round((last_adj_close-max_pain)/max_pain,2),
            # 'Implied % Move -previousMP': 0,
            # 'Implied % Move-LAC': round(implied_percentage_move, 2),
            'ITM OI' : ITM_OI,
            'Total OI' : all_OI,
            'Net_IV': round(Net_IV,3),
            'Net ITM IV': round(ITM_Avg_Net_IV, 3),
            'Net IV MP': round(Net_IV_at_MP, 3),
            'Net IV LAC':round(Net_IV_Closest_Strike_LAC,3),
            'Net_IV/OI' : Net_IV/all_OI,
            'Net ITM_IV/ITM_OI' : ITM_Avg_Net_IV/ITM_OI,
            #TODO ITM contract $ %
            'ITM Contracts %' : ITM_OI/all_OI,

            #PCR
            'PCR-Vol': round(PC_Ratio_Vol, 3),

            'PCR-OI': round(PC_Ratio_OI, 3),
            'PCR Vol/OI': round(PC_Ratio_Vol / PC_Ratio_OI, 3),
            'ITM PCR-Vol': round(ITM_PC_Ratio_Vol, 2),

            'ITM PCR-OI': round(ITM_PC_Ratio_OI, 3),
            'ITM PCR Vol/OI': round(ITM_PC_Ratio_Vol / ITM_PC_Ratio_OI, 3),
            'PCR @MP-Vol': round(PC_Ratio_Vol_atMP, 3),
            'PCR @MP-OI': round(PC_Ratio_OI_atMP, 3),
            'PCR @MP Vol/OI ': round((PC_Ratio_Vol_atMP / PC_Ratio_OI_atMP), 3),
            'PCR @LAC-Vol': round(PC_Ratio_Vol_Closest_Strike_LAC, 3),
            'PCR @LAC-OI': round(PC_Ratio_OI_Closest_Strike_LAC, 3),
            'PCR @LAC Vol/OI ': round(PCR_vol_OI_at_LAC, 3),


            # 'Novel ITM_CallsOI/CallsOI': round(
            #     ITM_CallsOI / all_CallsOI, 3),
            # 'Novel ITM_CallsVol/CallsVol': round(
            #     ITM_CallsVol / all_CallsVol, 3),
            # 'Novel ITM_PutsOI/PutsOI': round(
            #     ITM_PutsOI / all_PutsOI, 3),

            # 'Novel ITM_PutsVol/PutsVol': round(
            #     ITM_PutsVol / all_PutsVol, 3),
            # 'Novel (ITM_CallsVol/CallsVol)/(ITM_CallsOI/CallsOI)': round(
            #     (ITM_CallsVol / all_CallsVol) / (ITM_CallsOI / all_CallsOI), 3),
            # 'Novel (ITM_PutsVol/PutsVol)/(ITM_PutsOI/PutsOI)': round(
            #     (ITM_PutsVol / all_PutsVol) / (ITM_PutsOI / all_PutsOI), 3),
            # 'Novel (ITM_PutsOI/PutsOI)/(ITM_CallsOI/CallsOI)': round((ITM_PutsOI/all_PutsOI)/(ITM_CallsOI/all_CallsOI),3),
            # 'Novel (ITM_PutsVol/PutsVol)/(ITM_CallsVol/CallsVol)': round((ITM_PutsVol/all_PutsVol)/(ITM_CallsVol/all_CallsVol),3),


            ###SEE BELOW FOR  implied move -last mp
            # 'Highest DFSxOI Call % from SP': round((current_price - max_call_DFSxOI)/max_call_DFSxOI,3),
            # 'Highest DFSxOI Put% from SP': round((current_price - max_put_DFSxOI)/max_put_DFSxOI,3),
            # 'Highest DFSxOI Strike% from SP': round((current_price - max_DFSxOI) / max_DFSxOI, 3),            ###SEE BELOW FOR  implied move -last mp
            # 'Highest Premium Call% from SP': round((current_price - highest_premium_call) / highest_premium_call, 3),
            # 'Highest Premium Put% from SP': round((current_price - highest_premium_put) / highest_premium_put, 3),
            # 'Highest Premium Strike% from SP': round((current_price - highest_premium_strike) / highest_premium_strike, 3),

            # 'Highest DFSxOI Call % from MP': round((max_pain - max_call_DFSxOI)/max_call_DFSxOI,3),
            # 'Highest DFSxOI Put% from MP': round((max_pain - max_put_DFSxOI)/max_put_DFSxOI,3),
            # 'Highest DFSxOI Strike% from MP': round((max_pain - max_DFSxOI) / max_DFSxOI, 3),

            #
            # 'Highest Premium Call% from MP': round((max_pain - highest_premium_call)/highest_premium_call,3),
            # 'Highest Premium Put% from MP': round((max_pain -  highest_premium_put)/highest_premium_put,3),
            # 'Highest Premium Strike% from MP': round((max_pain - highest_premium_strike) / highest_premium_strike, 3),

            # 'Highest DFSxOI Call': max_call_DFSxOI,
            # 'Highest DFSxOI Put': max_put_DFSxOI,
            #
            # 'Highest Premium Strike': highest_premium_strike,
            # 'Highest Premium Call': highest_premium_call,
            # 'Highest Premium Put': highest_premium_put,
            # 'Highest DFSxOI Strike': max_DFSxOI,
            # 'Top 2 OI Strikes': highestTotalOI_dict,
            # 'Top 5 ITM OI Calls': sorted(itm_calls_OI_dict["Call_OI"].items(), key=lambda item: item[1], reverse=True)[:5],
            # 'Top 5 ITM OI Puts': sorted(itm_puts_OI_dict["Put_OI"].items(), key=lambda item: item[1], reverse=True)[:5],
            # 'Top 5 OI Calls': top_five_calls_dict,
            # 'Top 5 OI Puts': top_five_puts_dict,

        })


        # if len(results) >= 2:
        #     # print(results[-2]['Maximum Pain'])
        #     implied_move_from_last_MP = ((max_pain - results[-2]['Maximum Pain']) / results[-2]['Maximum Pain']) * 100
        #     # print(implied_move_fom_last_MP)
        #     results[-1]['Implied % Move -previousMP'] = round(implied_move_from_last_MP,2)

        ##TODO Signal strength 1-3, bull/bear.
        #
        # if results('Bonsai Ratio') < 1:
        #     if results('Maximum Pain') < results('Current Stock Price'):
        #


    df = pd.DataFrame(results)

    output_dir = Path(f'data/ProcessedData/{ticker}/{YYMMDD}/')
    output_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    df.to_csv(f'data/ProcessedData/{ticker}/{YYMMDD}/{ticker}_{YYMMDD}_{StockLastTradeTime}.csv', index=False)

    # print("process finished")
#

# ###Check for prior file based on name date.
# directory_path = "/path/to/directory" # Replace with the path to your directory
# current_file = "210326.txt" # Replace with the name of your current file
#
# try:
#     current_date = datetime.strptime(current_file[:6], '%y%m%d').date()
#     prior_file = None
#     prior_date = None
#     for file in os.listdir(directory_path):
#         if file.endswith('.txt'):
#             file_date = datetime.strptime(file[:6], '%y%m%d').date()
#             if file_date < current_date:
#                 if prior_date is None or file_date > prior_date:
#                     prior_file = file
#                     prior_date = file_date
#     if prior_file:
#         print(f"The prior file is: {prior_file}")
#     else:
#         print("There is no prior file.")
# except ValueError:
#     print("Invalid date format.")
