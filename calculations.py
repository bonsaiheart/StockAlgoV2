from pathlib import Path
import numpy as np
import pandas as pd


def perform_operations(
    ticker,
    last_adj_close,
    current_price,
    StockLastTradeTime,
    YYMMDD,
    CurrentTime,optionchaindf
):
    results = []
    price_change_percent = ((current_price - last_adj_close) / last_adj_close) * 100
    # TODO could pass in optionchain.
    if optionchaindf is None or optionchaindf.empty:
        optionchain_df = pd.read_csv(
        f"data/optionchain/{ticker}/{YYMMDD}/{ticker}_{CurrentTime}.csv"
    )
    else:
        optionchain_df = optionchaindf
    optionchain_df["Put_IV"] = optionchain_df["p_greeks"].str.get("mid_iv")
    optionchain_df["Call_IV"] = optionchain_df["c_greeks"].str.get("mid_iv")

    # Create an empty column to hold the results
    optionchain_df["strike_lac_diff"] = 0

    # Pre-calculate all differences
    all_strike_lac_diffs = optionchain_df.apply(
        lambda row: abs(row["Strike"] - last_adj_close), axis=1
    )
    optionchain_df["strike_lac_diff"] = all_strike_lac_diffs

    # No need to calculate within the loop anymore


    groups = optionchain_df.groupby("ExpDate")
    # # Calculate strike_lac_diff ONCE before the loop edit, nvm, not all expdates have same strikes.
    # optionchain_df["strike_lac_diff"] = optionchain_df["Strike"].apply(lambda x: abs(x - last_adj_close))

    # divide into groups by exp date, call info from group.
    for exp_date, group in groups:
        pain_list = []
        strike_LASTPRICExOI_list = []
        call_LASTPRICExOI_list = []
        put_LASTPRICExOI_list = []
        call_price_dict = (
            group.loc[group["Call_LastPrice"] >= 0, ["Strike", "Call_LastPrice"]]
            .set_index("Strike")
            .to_dict()
        )

        strike = group["Strike"]
        # print("strike column for group",strike)
        # pain is ITM puts/calls
        calls_LASTPRICExOI_dict = (
            group.loc[
                group["Calls_lastPriceXoi"] >= 0, ["Strike", "Calls_lastPriceXoi"]
            ]
            .set_index("Strike")
            .to_dict()
        )
        puts_LASTPRICExOI_dict = (
            group.loc[group["Puts_lastPriceXoi"] >= 0, ["Strike", "Puts_lastPriceXoi"]]
            .set_index("Strike")
            .to_dict()
        )

        ITM_CallsVol = group.loc[
            (group["Strike"] <= current_price), "Call_Volume"
        ].sum()
        ITM_PutsVol = group.loc[(group["Strike"] >= current_price), "Put_Volume"].sum()
        ITM_CallsOI = group.loc[(group["Strike"] <= current_price), "Call_OI"].sum()
        ITM_PutsOI = group.loc[(group["Strike"] >= current_price), "Put_OI"].sum()

        all_CallsVol = group.Call_Volume.sum()
        all_PutsVol = group.Put_Volume.sum()

        all_CallsOI = group.Call_OI.sum()
        all_PutsOI = group.Put_OI.sum()

        all_OI = all_PutsOI + all_CallsOI
        ITM_OI = ITM_CallsOI + ITM_PutsOI

        ITM_Call_IV = group.loc[(group["Strike"] <= current_price), "Call_IV"].sum()
        ITM_Put_IV = group.loc[(group["Strike"] >= current_price), "Put_IV"].sum()

        Call_IV = group["Call_IV"].sum()
        Put_IV = group["Put_IV"].sum()

        # Now that we have calculated sums, we use them for further calculations

        ITM_Avg_Net_IV = ITM_Call_IV - ITM_Put_IV
        Net_IV = Call_IV - Put_IV

        PC_Ratio_Vol = (
            all_PutsVol / all_CallsVol
            if all_CallsVol != 0 and not np.isnan(all_CallsVol)
            else np.nan
        )
        ITM_PC_Ratio_Vol = (
            ITM_PutsVol / ITM_CallsVol
            if ITM_CallsVol != 0 and not np.isnan(ITM_CallsVol)
            else np.nan
        )
        PC_Ratio_OI = (
            all_PutsOI / all_CallsOI
            if all_CallsOI != 0 and not np.isnan(all_CallsOI)
            else np.nan
        )
        ITM_PC_Ratio_OI = (
            ITM_PutsOI / ITM_CallsOI
            if ITM_CallsOI != 0 and not np.isnan(ITM_CallsOI)
            else np.nan
        )

        # DFSxOI_dict = (
        #     group.loc[group["Puts_dollarsFromStrikeXoi"] >= 0, ["Strike", "Puts_dollarsFromStrikeXoi"]]
        #     .set_index("Strike")
        #     .to_dict()
        # )
        # Money_weighted_PC_Ratio =
        ###TODO add highest premium puts/calls, greeks corelation?
        ###TODO correlate volume and IV, high volume high iv = contracts being bought, high volume, low vol. = contracts being sold.
        # print("wassup123",strike)
        for strikeprice in strike:
            itmCalls_dollarsFromStrikeXoiSum = group.loc[
                (group["Strike"] < strikeprice), "Calls_dollarsFromStrikeXoi"
            ].sum()
            itmPuts_dollarsFromStrikeXoiSum = group.loc[
                (group["Strike"] > strikeprice), "Puts_dollarsFromStrikeXoi"
            ].sum()
            call_LASTPRICExOI = calls_LASTPRICExOI_dict.get(
                "Calls_lastPriceXoi", {}
            ).get(strikeprice, 0)
            put_LASTPRICExOI = puts_LASTPRICExOI_dict.get("Puts_lastPriceXoi", {}).get(
                strikeprice, 0
            )
            # call_DFSxOI = calls_DFSxOI_dict.get("Calls_dollarsFromStrikeXoi", {}).get(strikeprice, 0)
            # put_DFSxOI = puts_DFSxOI_dict.get("Puts_dollarsFromStrikeXoi", {}).get(strikeprice, 0)
            pain_value = (
                itmPuts_dollarsFromStrikeXoiSum + itmCalls_dollarsFromStrikeXoiSum
            )
            pain_list.append((strikeprice, pain_value))
            strike_LASTPRICExOI = call_LASTPRICExOI + put_LASTPRICExOI
            strike_LASTPRICExOI_list.append((strikeprice, strike_LASTPRICExOI))
            call_LASTPRICExOI_list.append((strikeprice, call_LASTPRICExOI))
            put_LASTPRICExOI_list.append((strikeprice, put_LASTPRICExOI))

        highest_premium_strike = max(strike_LASTPRICExOI_list, key=lambda x: x[1])[0]
        highest_premium_call = max(call_LASTPRICExOI_list, key=lambda x: x[1])[0]
        highest_premium_put = max(put_LASTPRICExOI_list, key=lambda x: x[1])[0]
        max_pain = min(pain_list, key=lambda x: x[1])[0]
        top_five_calls = (
            group.loc[group["Call_OI"] > 0]
            .sort_values(by="Call_OI", ascending=False)
            .head(5)
        )
        top_five_calls_dict = (
            top_five_calls[["Strike", "Call_OI"]]
            .set_index("Strike")
            .to_dict()["Call_OI"]
        )
        top_five_puts = (
            group.loc[group["Put_OI"] > 0]
            .sort_values(by="Put_OI", ascending=False)
            .head(5)
        )
        top_five_puts_dict = (
            top_five_puts[["Strike", "Put_OI"]].set_index("Strike").to_dict()["Put_OI"]
        )

        ### FINDING CLOSEST STRIKE TO LAc


        ###############################
        if not group.empty:
            smallest_change_from_lac = group["strike_lac_diff"].abs().idxmin()
            closest_strike_lac = group.loc[smallest_change_from_lac, "Strike"]

            # Find index of row with the closest strike to the current price
            current_price_index = group["Strike"].sub(current_price).abs().idxmin()

            # Create a list of higher and lower strike indexes
            higher_strike_indexes = [
                i
                for i in range(current_price_index + 1, current_price_index + 5)
                if i in group.index
            ]
            lower_strike_indexes = [
                i
                for i in range(current_price_index - 1, current_price_index - 5, -1)
                if i in group.index
            ]

            # Initialize the lists for the closest strikes
            closest_higher_strikes = group.loc[higher_strike_indexes, "Strike"].tolist()
            closest_lower_strikes = group.loc[lower_strike_indexes, "Strike"].tolist()

            # Append None values to the lists to ensure they have a length of 4
            closest_higher_strikes += [None] * (4 - len(closest_higher_strikes))
            closest_lower_strikes += [None] * (4 - len(closest_lower_strikes))

            closest_strike_currentprice = group.loc[current_price_index, "Strike"]
        else:
            closest_strike_lac = None
            closest_strike_currentprice = None
            closest_higher_strikes = [None] * 4
            closest_lower_strikes = [None] * 4

        # Create the strikeindex_abovebelow list
        strikeindex_abovebelow = (
            closest_lower_strikes[::-1]
            + [closest_strike_currentprice]
            + closest_higher_strikes
        )
        closest_lower_strike4 = strikeindex_abovebelow[0]
        closest_lower_strike3 = strikeindex_abovebelow[1]
        closest_lower_strike2 = strikeindex_abovebelow[2]
        closest_lower_strike1 = strikeindex_abovebelow[3]
        closest_higher_strike1 = strikeindex_abovebelow[5]
        closest_higher_strike2 = strikeindex_abovebelow[6]
        closest_higher_strike3 = strikeindex_abovebelow[7]
        closest_higher_strike4 = strikeindex_abovebelow[8]

        #################

        ##Gettting pcr-vol for individual strikes above/below CP(closest to current price strike)
        def calculate_pcr_ratio(put_data, call_data):
            if np.isnan(put_data) or np.isnan(call_data) or call_data == 0:
                return (
                    np.inf
                    if call_data == 0 and put_data != 0 and not np.isnan(put_data)
                    else np.nan
                )
            else:
                return put_data / call_data

        group_strike = group.groupby("Strike")

        # Initialize dictionaries for storing PCR values
        strike_PCRv_dict = {}
        strike_PCRoi_dict = {}
        strike_ITMPCRv_dict = {}
        strike_ITMPCRoi_dict = {}

        # Calculate PCR values for all strikes in strikeindex_abovebelow
        for strikeabovebelow in strikeindex_abovebelow:
            strike_data = (
                group_strike.get_group(strikeabovebelow)
                if strikeabovebelow is not None
                else None
            )

            if strike_data is None:
                strike_PCRv_dict[strikeabovebelow] = np.nan
                strike_PCRoi_dict[strikeabovebelow] = np.nan
                strike_ITMPCRv_dict[strikeabovebelow] = np.nan
                strike_ITMPCRoi_dict[strikeabovebelow] = np.nan
                continue
            strike_PCRv_dict[strikeabovebelow] = calculate_pcr_ratio(
                strike_data["Put_Volume"].values[0],
                strike_data["Call_Volume"].values[0],
            )
            strike_PCRoi_dict[strikeabovebelow] = calculate_pcr_ratio(
                strike_data["Put_OI"].values[0], strike_data["Call_OI"].values[0]
            )

            # Calculate ITM PCR values for strikes above and below the current strike
            # For puts, the strike is higher

            itm_put_strike_data = group.loc[group["Strike"] >= strikeabovebelow]
            itm_call_strike_data = group.loc[group["Strike"] <= strikeabovebelow]

            itm_put_volume = itm_put_strike_data["Put_Volume"].sum()
            itm_call_volume = itm_call_strike_data["Call_Volume"].sum()
            if itm_call_volume == 0:
                strike_ITMPCRv_dict[strikeabovebelow] = np.nan
            else:
                strike_ITMPCRv_dict[strikeabovebelow] = itm_put_volume / itm_call_volume

            itm_put_oi = itm_put_strike_data["Put_OI"].sum()
            itm_call_oi = itm_call_strike_data["Call_OI"].sum()
            if itm_call_oi == 0:
                strike_ITMPCRoi_dict[strikeabovebelow] = np.nan
            else:
                strike_ITMPCRoi_dict[strikeabovebelow] = itm_put_oi / itm_call_oi

        def get_ratio_and_iv(strike):
            if strike is None:
                # handle the case where strike is None
                return None
            else:
                strike_data = group_strike.get_group(strike)
                ratio_v = calculate_pcr_ratio(
                    strike_data["Put_Volume"].values[0],
                    strike_data["Call_Volume"].values[0],
                )
                ratio_oi = calculate_pcr_ratio(
                    strike_data["Put_OI"].values[0], strike_data["Call_OI"].values[0]
                )
                call_iv = strike_data["Call_IV"].sum()
                put_iv = strike_data["Put_IV"].sum()
                net_iv = call_iv - put_iv
                return ratio_v, ratio_oi, call_iv, put_iv, net_iv

        # Calculate PCR values for the closest strike to LAC
        (
            PC_Ratio_Vol_Closest_Strike_LAC,
            PC_Ratio_OI_Closest_Strike_LAC,
            Call_IV_Closest_Strike_LAC,
            Put_IV_Closest_Strike_LAC,
            Net_IV_Closest_Strike_LAC,
        ) = get_ratio_and_iv(closest_strike_lac)
        # Calculate PCR values for the closest strike to CP
        PCRv_cp_strike, PCRoi_cp_strike, _, _, _ = get_ratio_and_iv(
            closest_strike_currentprice
        )

        # Calculate PCR values for Max Pain strike
        (
            PC_Ratio_Vol_atMP,
            PC_Ratio_OI_atMP,
            Net_Call_IV_at_MP,
            Net_Put_IV_at_MP,
            Net_IV_at_MP,
        ) = get_ratio_and_iv(max_pain)

        NIV_CurrentStrike = (
            get_ratio_and_iv(closest_strike_currentprice)[4]
            if closest_strike_currentprice is not None
            else np.nan
        )
        NIV_1HigherStrike = (
            get_ratio_and_iv(closest_higher_strike1)[4]
            if closest_higher_strike1 is not None
            else np.nan
        )
        NIV_2HigherStrike = (
            get_ratio_and_iv(closest_higher_strike2)[4]
            if closest_higher_strike2 is not None
            else np.nan
        )
        NIV_3HigherStrike = (
            get_ratio_and_iv(closest_higher_strike3)[4]
            if closest_higher_strike3 is not None
            else np.nan
        )
        NIV_4HigherStrike = (
            get_ratio_and_iv(closest_higher_strike4)[4]
            if closest_higher_strike4 is not None
            else np.nan
        )
        NIV_1LowerStrike = (
            get_ratio_and_iv(closest_lower_strike1)[4]
            if closest_lower_strike1 is not None
            else np.nan
        )
        NIV_2LowerStrike = (
            get_ratio_and_iv(closest_lower_strike2)[4]
            if closest_lower_strike2 is not None
            else np.nan
        )
        NIV_3LowerStrike = (
            get_ratio_and_iv(closest_lower_strike3)[4]
            if closest_lower_strike3 is not None
            else np.nan
        )
        NIV_4LowerStrike = (
            get_ratio_and_iv(closest_lower_strike4)[4]
            if closest_lower_strike4 is not None
            else np.nan
        )

        ###TODO error handling for scalar divide of zero denominator

        Bonsai_Ratio = ((ITM_PutsVol / all_PutsVol) * (ITM_PutsOI / all_PutsOI)) / (
            (ITM_CallsVol / all_CallsVol) * (ITM_CallsOI / all_CallsOI)
        )
        Bonsai2_Ratio = ((all_PutsVol / ITM_PutsVol) / (all_PutsOI / ITM_PutsOI)) * (
            (all_CallsVol / ITM_CallsVol) / (all_CallsOI / ITM_CallsOI)
        )
        round(strike_PCRv_dict[closest_higher_strike1], 3),

        results.append(
            {
                ###TODO change all price data to percentage change?
                ###TODO change closest strike to average of closest above/closest below
                "CurrentTime:": CurrentTime,
                "ExpDate": exp_date,
                "LastTradeTime": StockLastTradeTime,
                "Current Stock Price": float(current_price),
                "Current SP % Change(LAC)": round(float(price_change_percent), 2),
                # 'IV 30': iv30,
                # 'IV 30 % change': iv30_change_percent,
                "Maximum Pain": max_pain,
                "Bonsai Ratio": round(Bonsai_Ratio, 5),
                # 'Bonsai %change': bonsai_percent_change,
                "Bonsai Ratio 2": round(Bonsai2_Ratio, 5),
                "B1/B2": round((Bonsai_Ratio / Bonsai2_Ratio), 4),
                "B2/B1": round((Bonsai2_Ratio / Bonsai_Ratio), 4),
                # TODO ITM contract $ %
                "PCR-Vol": round(PC_Ratio_Vol, 3),
                "PCR-OI": round(PC_Ratio_OI, 3),
                "PCRv @CP Strike": round(PCRv_cp_strike, 3),
                "PCRoi @CP Strike": round(PCRoi_cp_strike, 3),
                "PCRv Up1": round(strike_PCRv_dict[closest_higher_strike1], 3),
                "PCRv Up2": round(strike_PCRv_dict[closest_higher_strike2], 3),
                "PCRv Up3": round(strike_PCRv_dict[closest_higher_strike3], 3),
                "PCRv Up4": round(strike_PCRv_dict[closest_higher_strike4], 3),
                "PCRv Down1": round(strike_PCRv_dict[closest_lower_strike1], 3),
                "PCRv Down2": round(strike_PCRv_dict[closest_lower_strike2], 3),
                "PCRv Down3": round(strike_PCRv_dict[closest_lower_strike3], 3),
                "PCRv Down4": round(strike_PCRv_dict[closest_lower_strike4], 3),
                "PCRoi Up1": round(strike_PCRoi_dict[closest_higher_strike1], 3),
                "PCRoi Up2": round(strike_PCRoi_dict[closest_higher_strike2], 3),
                "PCRoi Up3": round(strike_PCRoi_dict[closest_higher_strike3], 3),
                "PCRoi Up4": round(strike_PCRoi_dict[closest_higher_strike4], 3),
                "PCRoi Down1": round(strike_PCRoi_dict[closest_lower_strike1], 3),
                "PCRoi Down2": round(strike_PCRoi_dict[closest_lower_strike2], 3),
                "PCRoi Down3": round(strike_PCRoi_dict[closest_lower_strike3], 3),
                "PCRoi Down4": round(strike_PCRoi_dict[closest_lower_strike4], 3),
                "ITM PCR-Vol": round(ITM_PC_Ratio_Vol, 2),
                "ITM PCR-OI": round(ITM_PC_Ratio_OI, 3),
                "ITM PCRv Up1": strike_ITMPCRv_dict[closest_higher_strike1],
                "ITM PCRv Up2": strike_ITMPCRv_dict[closest_higher_strike2],
                "ITM PCRv Up3": strike_ITMPCRv_dict[closest_higher_strike3],
                "ITM PCRv Up4": strike_ITMPCRv_dict[closest_higher_strike4],
                "ITM PCRv Down1": strike_ITMPCRv_dict[closest_lower_strike1],
                "ITM PCRv Down2": strike_ITMPCRv_dict[closest_lower_strike2],
                "ITM PCRv Down3": strike_ITMPCRv_dict[closest_lower_strike3],
                "ITM PCRv Down4": strike_ITMPCRv_dict[closest_lower_strike4],
                "ITM PCRoi Up1": strike_ITMPCRoi_dict[closest_higher_strike1],
                "ITM PCRoi Up2": strike_ITMPCRoi_dict[closest_higher_strike2],
                "ITM PCRoi Up3": strike_ITMPCRoi_dict[closest_higher_strike3],
                "ITM PCRoi Up4": strike_ITMPCRoi_dict[closest_higher_strike4],
                "ITM PCRoi Down1": strike_ITMPCRoi_dict[closest_lower_strike1],
                "ITM PCRoi Down2": strike_ITMPCRoi_dict[closest_lower_strike2],
                "ITM PCRoi Down3": strike_ITMPCRoi_dict[closest_lower_strike3],
                "ITM PCRoi Down4": strike_ITMPCRoi_dict[closest_lower_strike4],
                "ITM OI": ITM_OI,
                "Total OI": all_OI,
                "ITM Contracts %": ITM_OI / all_OI,
                "Net_IV": round(Net_IV, 3),
                "Net ITM IV": round(ITM_Avg_Net_IV, 3),
                "Net IV MP": round(Net_IV_at_MP, 3),
                "Net IV LAC": round(Net_IV_Closest_Strike_LAC, 3),
                "NIV Current Strike": round(NIV_CurrentStrike, 3),
                "NIV 1Higher Strike": round(NIV_1HigherStrike, 3),
                "NIV 1Lower Strike": round(NIV_1LowerStrike, 3),
                "NIV 2Higher Strike": round(NIV_2HigherStrike, 3),
                "NIV 2Lower Strike": round(NIV_2LowerStrike, 3),
                "NIV 3Higher Strike": round(NIV_3HigherStrike, 3),
                "NIV 3Lower Strike": round(NIV_3LowerStrike, 3),
                "NIV 4Higher Strike": round(NIV_4HigherStrike, 3),
                "NIV 4Lower Strike": round(NIV_4LowerStrike, 3),
                ###Positive number means NIV highers are higher, and price will drop.
                # TODO should do as percentage change from total niv numbers to see if its big diff.
                "NIV highers(-)lowers1-2": (NIV_1HigherStrike + NIV_2HigherStrike)
                - (NIV_1LowerStrike + NIV_2LowerStrike),
                "NIV highers(-)lowers1-4": (
                    NIV_1HigherStrike
                    + NIV_2HigherStrike
                    + NIV_3HigherStrike
                    + NIV_4HigherStrike
                )
                - (
                    NIV_1LowerStrike
                    + NIV_2LowerStrike
                    + NIV_3LowerStrike
                    + NIV_4LowerStrike
                ),
                "NIV 1-2 % from mean": (
                    (
                        (NIV_1HigherStrike + NIV_2HigherStrike)
                        - (NIV_1LowerStrike + NIV_2LowerStrike)
                    )
                    / (
                        (
                            NIV_1HigherStrike
                            + NIV_2HigherStrike
                            + NIV_1LowerStrike
                            + NIV_2LowerStrike
                        )
                        / 4
                    )
                )
                * 100,
                "NIV 1-4 % from mean": (
                    (
                        NIV_1HigherStrike
                        + NIV_2HigherStrike
                        + NIV_3HigherStrike
                        + NIV_4HigherStrike
                    )
                    - (
                        NIV_1LowerStrike
                        + NIV_2LowerStrike
                        + NIV_3LowerStrike
                        + NIV_4LowerStrike
                    )
                    / (
                        (
                            NIV_1HigherStrike
                            + NIV_2HigherStrike
                            + NIV_3HigherStrike
                            + NIV_4HigherStrike
                            + NIV_1LowerStrike
                            + NIV_2LowerStrike
                            + NIV_3LowerStrike
                            + NIV_4LowerStrike
                        )
                        / 8
                    )
                )
                * 100,
                ##TODO swap (/) with result = np.divide(x, y)
                "Net_IV/OI": Net_IV / all_OI,
                "Net ITM_IV/ITM_OI": ITM_Avg_Net_IV / ITM_OI,
                "Closest Strike to CP": closest_strike_currentprice,
            }
        )
    processed_data_df = pd.DataFrame(results)

    output_dir = Path(f"data/ProcessedData/{ticker}/{YYMMDD}/")

    output_dir.mkdir(mode=0o755, parents=True, exist_ok=True)
    output_dir_dailyminutes = Path(f"data/DailyMinutes/{ticker}")
    output_file_dailyminutes = Path(f"data/DailyMinutes/{ticker}/{ticker}_{YYMMDD}.csv")
    output_dir_dailyminutes_w_algo_results = Path(
        f"data/DailyMinutes_w_algo_results/{ticker}"
    )
    output_dir_dailyminutes_w_algo_results.mkdir(
        mode=0o755, parents=True, exist_ok=True
    )
    output_dir_dailyminutes.mkdir(mode=0o755, parents=True, exist_ok=True)
    processed_data_df = pd.merge(processed_data_df, optionchain_df, on='ExpDate', how='outer')

    # # Use the function
    if output_file_dailyminutes.exists():
        dailyminutes_df = pd.read_csv(output_file_dailyminutes)
        # dailyminutes_df = dailyminutes_df.drop_duplicates(subset="CurrentTime")
        dailyminutes_df = pd.concat(
            [dailyminutes_df, processed_data_df.head(1)], ignore_index=True
        )
        # replace_inf(
        #     dailyminutes_df
        # )  # It will only run if inf or -inf values are present
    else:
        # pass
        dailyminutes_df = pd.concat([processed_data_df.head(1)], ignore_index=True)
        # replace_inf(    #     dailyminutes_df = pd.concat([processed_data_df.head(1)], ignore_index=True)
        #     dailyminutes_df
        # )  # It will only run if inf or -inf values are present

    dailyminutes_df.to_csv(output_file_dailyminutes, index=False)

    try:
        processed_data_df.to_csv(
            f"data/ProcessedData/{ticker}/{YYMMDD}/{ticker}_{CurrentTime}.csv",
            mode="w",
            index=False,
        )
        # print("processed data saved for",ticker)
    except FileExistsError as e:
        print(f"data/ProcessedData/{ticker}/{YYMMDD}/{ticker}_{StockLastTradeTime}.csv", "File Already Exists.")
        raise
    # print(type(optionchain_df),type(dailyminutes_df),type(processed_data_df),type(ticker))
    return optionchain_df, dailyminutes_df, processed_data_df, ticker
#TODO Creates different/more modular processing paths; so that depending on what inputs the model needs, that is the only processing done