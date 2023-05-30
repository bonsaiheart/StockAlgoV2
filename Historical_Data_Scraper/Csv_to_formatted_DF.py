from datetime import datetime, timedelta

import pandas as pd
import ta
import numpy as np
from pathlib import Path
import yfinance as yf

# Set the batch size
batch_size = 100000

data = pd.read_csv(f"spy_historical_chains.csv", chunksize=batch_size)
combined_chunks = []
yf_ticker_obj = yf.Ticker("SPY")

# Iterate over the data in chunks
for chunk in data:
    print(chunk)
    close = chunk["Close"]
    ticker = "spy"
    current_price = chunk['Close']
    print(chunk.columns)
    grouped = chunk.groupby('Put_Call')
    try:
        call_group = grouped.get_group('C').copy()
    except KeyError:
        pass  # Handle the case when 'C' group does not exist

    try:
        put_group = grouped.get_group('P').copy()
    except KeyError:
        pass


    callsChain = [call_group]
    putsChain = [put_group]

    calls_df = pd.concat(callsChain, ignore_index=True).sort_values("Date")
    calls_df["expDate"] = calls_df["ExpDate"]

    ##CHANGED LAC TO CLOSE
    calls_df["dollarsFromStrike"] = abs(calls_df["Strike"] - close)
    calls_df["dollarsFromStrikexOI"] = calls_df["dollarsFromStrike"] * calls_df["Open Interest"]
    calls_df.rename(
        columns={
            "symbol": "c_contractSymbol",
            "last": "c_lastPrice",
            "dollarsFromStrike": "c_dollarsFromStrike",
            "dollarsFromStrikexOI": "c_dollarsFromStrikexOI",
            "volume": "c_volume",
            "open_interest": "c_openInterest",
        },
        inplace=True,
    )

    puts_df = pd.concat(putsChain, ignore_index=True).sort_values("Date")

    puts_df["expDate"] = puts_df["ExpDate"]

    ###CHANGED LAC TO CLOSE
    puts_df["dollarsFromStrike"] = abs(puts_df["Strike"] - close)
    puts_df["dollarsFromStrikexOI"] = puts_df["dollarsFromStrike"] * puts_df["Open Interest"]
    puts_df.rename(
        columns={
            "symbol": "p_contractSymbol",
            "last": "p_lastPrice",
            "volume": "p_volume",
            "open_interest": "p_openInterest",

            "dollarsFromStrike": "p_dollarsFromStrike",
            "dollarsFromStrikexOI": "p_dollarsFromStrikexOI",
        },
        inplace=True,
    )

    combined_chunk = pd.merge(puts_df, calls_df, on=["Date", "ExpDate", "Strike"])
    combined_chunks.append(combined_chunk)

# Concatenate the processed chunks into a single DataFrame
combined = pd.concat(combined_chunks, ignore_index=True)
combined['Strike'] = combined['Strike'] / 1000





# ...

# Save the combined DataFrame to a CSV file
combined.to_csv("combined_data.csv", index=False)

##TODO abot to add current price(close) for each date in df uphere.
for x in combined.iterrows():

    date_str = str(combined['Date'][x[0]])
    date_obj = datetime.strptime(date_str, "%Y%m%d")
    formatted_date = date_obj.strftime("%Y-%m-%d")
    next_day_obj = date_obj + timedelta(days=1)
    formatted_date_plusoneday = next_day_obj.strftime("%Y-%m-%d")


    price_data = yf_ticker_obj.history(start=formatted_date, end=formatted_date_plusoneday
                                       )
    if not price_data.empty:
        current_price = price_data["Close"][0]
        combined.at[x[0], "Current Price"] = current_price
print(combined.columns)
combined = combined[
    ["Date","ExpDate","Strike","Current Price","Contract_x","Ticker_x","Open_x","High_x","Low_x","Close_x","Volume_x","Open Interest_x","Contract_y","Ticker_y","Open_y","High_y","Low_y","Close_y","Volume_y","Open Interest_y",'p_dollarsFromStrike','p_dollarsFromStrikexOI','c_dollarsFromStrike','c_dollarsFromStrikexOI'



    ]
]
combined.rename(
    columns={
        # "expDate": "ExpDate",


        "Volume_x": "Put_Volume",
        "Open Interest_x": "Put_OI",

        "Volume_y": "Call_Volume",
        "Open Interest_y": "Call_OI",
"Contract_x":"Put_Contract",
        "Open_x":"Put_Open",
        "High_x":"Put_High",
    'Low_x':"Put_Low",
        "Close_x":"Put_Close_x",
    "Contract_y":"Call_Contract",
    'Open_y':'Call_Open',
    'High_y':'Call_High',
    'Low_y':'Call_Low',
    'Close_y':'Call_Close'
    },
    inplace=True
)




output_dir = Path(f"data/historical_optionchain")
output_dir.mkdir(mode=0o755, parents=True, exist_ok=True)

try:
    combined.to_csv(f"data/historical_optionchain/{ticker}.csv", mode="x")
except Exception as e:

        print(f"An error occurred while writing the CSV file,: {e}")
        combined.to_csv(f"data/historical_optionchain/{ticker}(ERRORR).csv")
# combined.to_csv(f"combined_tradier.csv")




results = []


groups = combined.groupby("Date")
# divide into groups by exp date, call info from group.
for Date,  group,in groups:

    date_str = str(Date)
    date_obj = datetime.strptime(date_str, "%Y%m%d")
    formatted_date = date_obj.strftime("%Y-%m-%d")

    print(Date,"DATEEE")
    pain_list = []
    strike_LASTPRICExOI_list = []
    call_LASTPRICExOI_list = []
    put_LASTPRICExOI_list = []


    strike = group["Strike"]

    ITM_CallsVol = group.loc[(group["Strike"] <= group["Current_Price"]), "Call_Volume"].sum()
    ITM_PutsVol = group.loc[(group["Strike"] >= group["Current_Price"]), "Put_Volume"].sum()
    ITM_CallsOI = group.loc[(group["Strike"] <= group["Current_Price"]), "Call_OI"].sum()
    ITM_PutsOI = group.loc[(group["Strike"] >= group["Current_Price"]), "Put_OI"].sum()
    ITM_OI = ITM_CallsOI + ITM_PutsOI
    all_CallsVol = group.Call_Volume.sum()
    all_PutsVol = group.Put_Volume.sum()

    all_CallsOI = group.Call_OI.sum()
    all_PutsOI = group.Put_OI.sum()
    all_OI = all_PutsOI + all_CallsOI


    if all_CallsVol != 0 and not np.isnan(all_CallsVol):
        PC_Ratio_Vol = all_PutsVol / all_CallsVol
    else:
        PC_Ratio_Vol = np.nan

    if ITM_CallsVol != 0 and not np.isnan(ITM_CallsVol):
        ITM_PC_Ratio_Vol = ITM_PutsVol / ITM_CallsVol
    else:
        ITM_PC_Ratio_Vol = np.nan


    if all_CallsOI != 0 and not np.isnan(all_CallsOI):
        PC_Ratio_OI = all_PutsOI / all_CallsOI
    else:
        PC_Ratio_OI = np.nan

    if ITM_CallsOI != 0 and not np.isnan(ITM_CallsOI):
        ITM_PC_Ratio_OI = ITM_PutsOI / ITM_CallsOI
    else:
        ITM_PC_Ratio_OI = np.nan


    for strikeprice in strike:
        itmCalls_dollarsFromStrikeXoiSum = group.loc[
            (group["Strike"] < strikeprice), "c_dollarsFromStrikexOI"
        ].sum()
        itmPuts_dollarsFromStrikeXoiSum = group.loc[
            (group["Strike"] > strikeprice), "p_dollarsFromStrikexOI"
        ].sum()

        pain_value = itmPuts_dollarsFromStrikeXoiSum + itmCalls_dollarsFromStrikeXoiSum
        pain_list.append((strikeprice, pain_value))



    max_pain = min(pain_list, key=lambda x: x[1])[0]




    strike_PCRv_dict = {}
    strike_PCRoi_dict = {}
    strike_ITMPCRv_dict = {}
    strike_ITMPCRoi_dict = {}


    Bonsai_Ratio = ((ITM_PutsVol / all_PutsVol) * (ITM_PutsOI / all_PutsOI)) / (
        (ITM_CallsVol / all_CallsVol) * (ITM_CallsOI / all_CallsOI)
    )
    Bonsai2_Ratio = (
        (all_PutsOI == 0 or ITM_PutsOI == 0 or all_CallsOI == 0 or ITM_CallsVol == 0 or ITM_CallsOI == 0)
        and float("inf")
        or ((all_PutsVol / ITM_PutsVol) / (all_PutsOI / ITM_PutsOI))
        * ((all_CallsVol / ITM_CallsVol) / (all_CallsOI / ITM_CallsOI))
    )



    results.append(
        {
            ###TODO change all price data to percentage change?
            ###TODO change closest strike to average of closest above/closest below
            "Date":Date,
            # "ExpDate": ,
            "Current Stock Price": float(current_price),
            # "Current SP % Change(LAC)": round(float(price_change_percent), 2),
            "Maximum Pain": max_pain,
            "Bonsai Ratio": round(Bonsai_Ratio, 5),
            "Bonsai Ratio 2": round(Bonsai2_Ratio, 5),
            "B1/B2":round((Bonsai_Ratio/Bonsai2_Ratio),4),
            "B2/B1":round((Bonsai2_Ratio/Bonsai_Ratio),4),
            # 'Bonsai_2 %change': bonsai2_percent_change,
            "PCR-Vol": round(PC_Ratio_Vol, 3),
            "PCR-OI": round(PC_Ratio_OI, 3),
            # 'PCR Vol/OI': round(PC_Ratio_Vol / PC_Ratio_OI, 3),
            # 'ITM PCR Vol/OI': float('inf') if ITM_PC_Ratio_OI == 0 else 0 if ITM_PC_Ratio_Vol == 0 else round(ITM_PC_Ratio_Vol / ITM_PC_Ratio_OI, 3),
            # 'PCR @MP Vol/OI ': round((PC_Ratio_Vol_atMP / PC_Ratio_OI_atMP), 3),
            # 'PCR @LAC Vol/OI ': round(PCR_vol_OI_at_LAC, 3),
            # "PCRv @CP Strike": round(PCRv_cp_strike, 3),
            # "PCRoi @CP Strike": round(PCRoi_cp_strike, 3),
            # "PCRv Up1": round(strike_PCRv_dict[closest_higher_strike1], 3),
            # "PCRv Up2": round(strike_PCRv_dict[closest_higher_strike2], 3),
            # "PCRv Up3": round(strike_PCRv_dict[closest_higher_strike3], 3),
            # "PCRv Up4": round(strike_PCRv_dict[closest_higher_strike4], 3),
            # "PCRv Down1": round(strike_PCRv_dict[closest_lower_strike1], 3),
            # "PCRv Down2": round(strike_PCRv_dict[closest_lower_strike2], 3),
            # "PCRv Down3": round(strike_PCRv_dict[closest_lower_strike3], 3),
            # "PCRv Down4": round(strike_PCRv_dict[closest_lower_strike4], 3),
            # "PCRoi Up1": round(strike_PCRoi_dict[closest_higher_strike1], 3),
            # "PCRoi Up2": round(strike_PCRoi_dict[closest_higher_strike2], 3),
            # "PCRoi Up3": round(strike_PCRoi_dict[closest_higher_strike3], 3),
            # "PCRoi Up4": round(strike_PCRoi_dict[closest_higher_strike4], 3),
            # "PCRoi Down1": round(strike_PCRoi_dict[closest_lower_strike1], 3),
            # "PCRoi Down2": round(strike_PCRoi_dict[closest_lower_strike2], 3),
            # "PCRoi Down3": round(strike_PCRoi_dict[closest_lower_strike3], 3),
            # "PCRoi Down4": round(strike_PCRoi_dict[closest_lower_strike4], 3),
            # "ITM PCR-Vol": round(ITM_PC_Ratio_Vol, 2),
            # "ITM PCR-OI": round(ITM_PC_Ratio_OI, 3),
            # "ITM PCRv Up1": strike_ITMPCRv_dict[closest_higher_strike1],
            # "ITM PCRv Up2": strike_ITMPCRv_dict[closest_higher_strike2],
            # "ITM PCRv Up3": strike_ITMPCRv_dict[closest_higher_strike3],
            # "ITM PCRv Up4": strike_ITMPCRv_dict[closest_higher_strike4],
            # "ITM PCRv Down1": strike_ITMPCRv_dict[closest_lower_strike1],
            # "ITM PCRv Down2": strike_ITMPCRv_dict[closest_lower_strike2],
            # "ITM PCRv Down3": strike_ITMPCRv_dict[closest_lower_strike3],
            # "ITM PCRv Down4": strike_ITMPCRv_dict[closest_lower_strike4],
            # "ITM PCRoi Up1": strike_ITMPCRoi_dict[closest_higher_strike1],
            # "ITM PCRoi Up2": strike_ITMPCRoi_dict[closest_higher_strike2],
            # "ITM PCRoi Up3": strike_ITMPCRoi_dict[closest_higher_strike3],
            # "ITM PCRoi Up4": strike_ITMPCRoi_dict[closest_higher_strike4],
            # "ITM PCRoi Down1": strike_ITMPCRoi_dict[closest_lower_strike1],
            # "ITM PCRoi Down2": strike_ITMPCRoi_dict[closest_lower_strike2],
            # "ITM PCRoi Down3": strike_ITMPCRoi_dict[closest_lower_strike3],
            # "ITM PCRoi Down4": strike_ITMPCRoi_dict[closest_lower_strike4],
            # "ITM OI": ITM_OI,
            # "Total OI": all_OI,
            # "ITM Contracts %": ITM_OI / all_OI,
#             "Net_IV": round(Net_IV, 3),
#             "Net ITM IV": round(ITM_Avg_Net_IV, 3),
#             "Net IV MP": round(Net_IV_at_MP, 3),
#             "Net IV LAC": round(Net_IV_Closest_Strike_LAC, 3),
#             "NIV Current Strike": round(NIV_CurrentStrike, 3),
#             "NIV 1Higher Strike": round(NIV_1HigherStrike, 3),
#             "NIV 1Lower Strike": round(NIV_1LowerStrike, 3),
#             "NIV 2Higher Strike": round(NIV_2HigherStrike, 3),
#             "NIV 2Lower Strike": round(NIV_2LowerStrike, 3),
#             "NIV 3Higher Strike": round(NIV_3HigherStrike, 3),
#             "NIV 3Lower Strike": round(NIV_3LowerStrike, 3),
#             "NIV 4Higher Strike": round(NIV_4HigherStrike, 3),
#             "NIV 4Lower Strike": round(NIV_4LowerStrike, 3),
#             ###Positive number means NIV highers are higher, and price will drop.
#             #TODO should do as percentage change from total niv numbers to see if its big diff.
#             "NIV highers(-)lowers1-2": (
#                                                    NIV_1HigherStrike + NIV_2HigherStrike ) - (
#                                                    NIV_1LowerStrike + NIV_2LowerStrike ),
#
#             "NIV highers(-)lowers1-4": (NIV_1HigherStrike+NIV_2HigherStrike+NIV_3HigherStrike+NIV_4HigherStrike)-(NIV_1LowerStrike+NIV_2LowerStrike+NIV_3LowerStrike+NIV_4LowerStrike),
#             "NIV 1-2 % from mean": (
#                     ((NIV_1HigherStrike + NIV_2HigherStrike) - (
#                                                NIV_1LowerStrike + NIV_2LowerStrike))/((NIV_1HigherStrike+NIV_2HigherStrike+NIV_1LowerStrike+NIV_2LowerStrike)/4))*100,
#
#             "NIV 1-4 % from mean": (
#                 (NIV_1HigherStrike + NIV_2HigherStrike + NIV_3HigherStrike + NIV_4HigherStrike) - (
#                                                    NIV_1LowerStrike + NIV_2LowerStrike + NIV_3LowerStrike + NIV_4LowerStrike)/((NIV_1HigherStrike+NIV_2HigherStrike+ NIV_3HigherStrike + NIV_4HigherStrike+NIV_1LowerStrike+NIV_2LowerStrike+NIV_3LowerStrike + NIV_4LowerStrike)/8))*100,
# ##TODO swap (/) with result = np.divide(x, y)
#             "Net_IV/OI": Net_IV / all_OI,
#             "Net ITM_IV/ITM_OI": ITM_Avg_Net_IV / ITM_OI,
#             "Closest Strike to CP": closest_strike_currentprice,

        }
    )


df = pd.DataFrame(results)
data["AwesomeOsc"] = ta.momentum.awesome_oscillator(
    high=data["High"], low=data["Low"], window1=1, window2=5, fillna=False
)
data["RSI"] = ta.momentum.rsi(close=close, window=5, fillna=False)


# output_dir = Path(f"data/ProcessedData/{ticker}/{YYMMDD}/")
#
# output_dir.mkdir(mode=0o755, parents=True, exist_ok=True)

df.to_csv(f"data/Historical_Processed_ChainData/{ticker}.csv", mode="x", index=False)

