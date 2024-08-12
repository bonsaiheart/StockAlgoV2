
import numpy as np
import pandas as pd


# add this line

def calculate_bonsai_ratios(
    ITM_PutsVol, all_PutsVol, ITM_PutsOI, all_PutsOI, ITM_CallsVol, all_CallsVol, ITM_CallsOI, all_CallsOI
):
    # Handle potential zero divisions for Bonsai_Ratio
    if (ITM_PutsVol == 0) or (all_PutsVol == 0) or (ITM_PutsOI == 0) or (all_PutsOI == 0) or (ITM_CallsVol == 0) or (all_CallsVol == 0) or (ITM_CallsOI == 0) or (all_CallsOI == 0):
        Bonsai_Ratio = np.nan  # Or set to a default value like 0 or 1
    else:
        # Calculate Bonsai_Ratio safely
        Bonsai_Ratio = (
            (ITM_PutsVol / all_PutsVol) * (ITM_PutsOI / all_PutsOI)
        ) / (
            (ITM_CallsVol / all_CallsVol) * (ITM_CallsOI / all_CallsOI)
        )

    # Handle potential zero divisions for Bonsai2_Ratio
    if (all_PutsVol == 0) or (ITM_PutsVol == 0) or (all_PutsOI == 0) or (ITM_PutsOI == 0) or (all_CallsVol == 0) or (ITM_CallsVol == 0) or (all_CallsOI == 0) or (ITM_CallsOI == 0):
        Bonsai2_Ratio = np.nan  # Or set to a default value
    else:
        # Calculate Bonsai2_Ratio safely
        Bonsai2_Ratio = (
            (all_PutsVol / ITM_PutsVol) / (all_PutsOI / ITM_PutsOI)
        ) * (
            (all_CallsVol / ITM_CallsVol) / (all_CallsOI / ITM_CallsOI)
        )

    return Bonsai_Ratio, Bonsai2_Ratio

def perform_operations(
        ticker,
        last_adj_close,
        current_price,
        CurrentTime,
        optionchain_df, symbol_name
):
    results = []
    price_change_percent = ((current_price - last_adj_close) / last_adj_close) * 100

    # Use optionchain_df directly as it already contains the technical analysis data
    if not optionchain_df.empty:
        # print(type(optionchain_df['fetch_timestamp'][0]))  # No need to print type anymore
        # Ensure the fetch_timestamp is in datetime format
        optionchain_df['fetch_timestamp'] = pd.to_datetime(optionchain_df['fetch_timestamp'])
        # Group by expiry date and proceed with calculations
        groups = optionchain_df.groupby("expiration_date")

        for exp_date, group in groups:
            pain_list = []
            strike_LASTPRICExOI_list = []
            call_LASTPRICExOI_list = []
            put_LASTPRICExOI_list = []

            strike = group["strike"]

            # Filter by option_type
            call_options = group[group["option_type"] == "call"]
            put_options = group[group["option_type"] == "put"]

            # Filter and get dictionaries (using the filtered DataFrames)
            calls_LASTPRICExOI_dict = (
                call_options.loc[call_options["lastPriceXoi"] >= 0, ["strike", "lastPriceXoi"]]  # Fixed column name
                .set_index("strike")
                .to_dict()
            )
            puts_LASTPRICExOI_dict = (
                put_options.loc[put_options["lastPriceXoi"] >= 0, ["strike", "lastPriceXoi"]]  # Fixed column name
                .set_index("strike")
                .to_dict()
            )

            # Calculate sums using the filtered DataFrames
            ITM_CallsVol = call_options.loc[(call_options["strike"] <= current_price), "volume"].sum()
            ITM_PutsVol = put_options.loc[(put_options["strike"] >= current_price), "volume"].sum()
            ITM_CallsOI = call_options.loc[(call_options["strike"] <= current_price), "open_interest"].sum()
            ITM_PutsOI = put_options.loc[(put_options["strike"] >= current_price), "open_interest"].sum()

            all_CallsVol = call_options["volume"].sum()
            all_PutsVol = put_options["volume"].sum()
            all_CallsOI = call_options["open_interest"].sum()
            all_PutsOI = put_options["open_interest"].sum()

            all_OI = all_PutsOI + all_CallsOI
            ITM_OI = ITM_CallsOI + ITM_PutsOI
            # print(call_options.columns)  #TODO from here down, convert to use callopiotn/put option df
            # optionchain_df['Put_IV'] = optionchain_df['greeks'].apply(lambda x: x.get('mid_iv') if isinstance(x, dict) else None)
            # optionchain_df['Call_IV'] = optionchain_df['greeks'].apply(lambda x: x.get('mid_iv') if isinstance(x, dict) else None)
            ITM_Call_IV = call_options.loc[(call_options["strike"] <= current_price), 'greeks'].apply(lambda x: x.get('mid_iv', 0) if isinstance(x, dict) else 0).sum()
            ITM_Put_IV = put_options.loc[(group["strike"] >= current_price), 'greeks'].apply(lambda x: x.get('mid_iv', 0) if isinstance(x, dict) else 0).sum()

            # Extract 'mid_iv' values, handle None values, convert to numeric for safe calculations
            call_iv_values = call_options['greeks'].apply(lambda x: x.get('mid_iv') if x is not None else np.nan).astype(float)
            put_iv_values = put_options['greeks'].apply(lambda x: x.get('mid_iv') if x is not None else np.nan).astype(float)

            # Calculate sums (using np.nansum to ignore NaN values)
            Call_IV = np.nansum(call_iv_values)
            Put_IV = np.nansum(put_iv_values)

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
                # ... (Calculations inside this loop, using call_options and put_options)
                itmCalls_dollarsFromStrikeXoiSum = call_options.loc[
                    (call_options["strike"] < strikeprice), "dollarsFromStrikeXoi"
                ].sum()  # Use call_options here
                itmPuts_dollarsFromStrikeXoiSum = put_options.loc[
                    (put_options["strike"] > strikeprice), "dollarsFromStrikeXoi"
                ].sum()  # Use put_options here
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
            # top_five_calls = (
            #     group.loc[group["Call_OI"] > 0]
            #     .sort_values(by="Call_OI", ascending=False)
            #     .head(5)
            # )
            # top_five_calls_dict = (
            #     top_five_calls[["Strike", "Call_OI"]]
            #     .set_index("Strike")
            #     .to_dict()["Call_OI"]
            # )
            # top_five_puts = (
            #     group.loc[group["Put_OI"] > 0]
            #     .sort_values(by="Put_OI", ascending=False)
            #     .head(5)
            # )
            # top_five_puts_dict = (
            #     top_five_puts[["Strike", "Put_OI"]].set_index("Strike").to_dict()["Put_OI"]
            # )

            ### FINDING CLOSEST STRIKE TO LAc


            ###############################
            if not group.empty:
                # smallest_change_from_lac = group["strike_lac_diff"].abs().idxmin()
                # closest_strike_lac = group.loc[smallest_change_from_lac, "Strike"]

                # Find index of row with the closest strike to the current price
                current_price_index = group["strike"].sub(current_price).abs().idxmin()

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
                closest_higher_strikes = group.loc[higher_strike_indexes, "strike"].tolist()
                closest_lower_strikes = group.loc[lower_strike_indexes, "strike"].tolist()

                # Append None values to the lists to ensure they have a length of 4
                closest_higher_strikes += [None] * (4 - len(closest_higher_strikes))
                closest_lower_strikes += [None] * (4 - len(closest_lower_strikes))

                closest_strike_currentprice = group.loc[current_price_index, "strike"]
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
            # print(closest_higher_strike4,closest_higher_strike3,closest_higher_strike2)
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

            group_strike = group.groupby("strike")  #TODO has strike and Strike.
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
                    strike_data.loc[strike_data["option_type"] == "put", "volume"].values[0],  # Filter puts
                    strike_data.loc[strike_data["option_type"] == "call", "volume"].values[0]) # Filter calls

                strike_PCRoi_dict[strikeabovebelow] = calculate_pcr_ratio(
                    strike_data.loc[strike_data["option_type"] == 'put' , "open_interest"].values[0], strike_data.loc[strike_data['option_type'] == "call" , "open_interest"].values[0]
                )

                # Calculate ITM PCR values for strikes above and below the current strike
                # For puts, the strike is higher

                itm_put_strike_data = put_options.loc[put_options["strike"] >= strikeabovebelow]  # Use put_options
                itm_call_strike_data = call_options.loc[call_options["strike"] <= strikeabovebelow]  # Use call_options

                itm_put_volume = itm_put_strike_data["volume"].sum()  # Corrected column name
                itm_call_volume = itm_call_strike_data["volume"].sum()  # Corrected column name
                if itm_call_volume == 0:
                    strike_ITMPCRv_dict[strikeabovebelow] = np.nan
                else:
                    strike_ITMPCRv_dict[strikeabovebelow] = itm_put_volume / itm_call_volume

                itm_put_oi = itm_put_strike_data["open_interest"].sum()  # Corrected column name
                itm_call_oi = itm_call_strike_data["open_interest"].sum()  # Corrected column name

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
                    put_strike_data = strike_data[strike_data["strike"] == strike]
                    call_strike_data = strike_data[strike_data["strike"] == strike]
                    ratio_v = calculate_pcr_ratio(
                        put_strike_data["volume"].values[0],
                        call_strike_data["volume"].values[0],
                    )
                    ratio_oi = calculate_pcr_ratio(
                        put_strike_data["open_interest"].values[0], call_strike_data["open_interest"].values[0]
                    )

                    call_iv = call_strike_data['greeks'].apply(lambda x: x.get('mid_iv') if x else 0).sum()
                    put_iv = put_strike_data['greeks'].apply(lambda x: x.get('mid_iv') if x else 0).sum()
                    net_iv = call_iv - put_iv
                    return ratio_v, ratio_oi, call_iv, put_iv, net_iv

            # Calculate PCR values for the closest strike to LAC
            # (
            #     PC_Ratio_Vol_Closest_Strike_LAC,
            #     PC_Ratio_OI_Closest_Strike_LAC,
            #     Call_IV_Closest_Strike_LAC,
            #     Put_IV_Closest_Strike_LAC,
            #     Net_IV_Closest_Strike_LAC,
            # ) = get_ratio_and_iv(closest_strike_lac)
            # Calculate PCR values for the closest strike to CP
            # PCRv_cp_strike, PCRoi_cp_strike, _, _, _ = get_ratio_and_iv(
            #     closest_strike_currentprice
            # )

            # Calculate PCR values for Max Pain strike
            (
                PC_Ratio_Vol_atMP,
                PC_Ratio_OI_atMP,
                Net_Call_IV_at_MP,
                Net_Put_IV_at_MP,
                Net_IV_at_MP,
            ) = get_ratio_and_iv(max_pain) #TODO add this?

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


            # Calculate Bonsai Ratios
            Bonsai_Ratio, Bonsai2_Ratio = calculate_bonsai_ratios(
                ITM_PutsVol, all_PutsVol, ITM_PutsOI, all_PutsOI, ITM_CallsVol, all_CallsVol, ITM_CallsOI, all_CallsOI
            )

            # Handle the result
            if Bonsai_Ratio is None:
                # Handle the case where Bonsai_Ratio couldn't be calculated
                print("Bonsai_Ratio calculation resulted in a division by zero.")
            # else:
            #     # Use the calculated Bonsai_Ratio
            #     # print(f"Bonsai_Ratio: {Bonsai_Ratio}")

            round(strike_PCRv_dict[closest_higher_strike1], 3),

            results.append(
                {
                    ###TODO change all price data to percentage change?
                    ###TODO change closest strike to average of closest above/closest below
                    "exp_date": exp_date,
                    # "LastTradeTime": StockLastTradeTime,  #TODO wont need this be cuase im adding calculations to the TA df still figuring out waht i need to return after swapping to sql.?
                    "current_stock_price": float(current_price),
                    "current_sp_change_lac": round(float(price_change_percent), 2),
                    # 'IV 30': iv30,
                    # 'IV 30 % change': iv30_change_percent,
                    "maximumpain": max_pain,
                    "bonsai_ratio": round(Bonsai_Ratio, 5) if Bonsai_Ratio is not None else None,
                    # 'Bonsai %change': bonsai_percent_change,
                    "bonsai_ratio_2": round(Bonsai2_Ratio, 5) if Bonsai2_Ratio is not None else None,  # Check Bonsai2_Ratio
                    "b1_dividedby_b2": round((Bonsai_Ratio / Bonsai2_Ratio), 4) if Bonsai_Ratio is not None and Bonsai2_Ratio is not None and Bonsai2_Ratio != 0 else None,  # Check both and avoid division by zero
                    "b2_dividedby_b1": round((Bonsai2_Ratio / Bonsai_Ratio), 4) if Bonsai2_Ratio is not None and Bonsai_Ratio != 0 else None,  # Check both and avoid division by zero
                    # TODO ITM contract $ %
                    "pcr_vol": round(PC_Ratio_Vol, 3) if PC_Ratio_Vol is not None else None,  # Check PC_Ratio_Vol

                    "pcr_oi": round(PC_Ratio_OI, 3),
                    # "PCRv @CP Strike": round(PCRv_cp_strike, 3),
                    # "PCRoi @CP Strike": round(PCRoi_cp_strike, 3),
                    "pcrv_up1": round(strike_PCRv_dict[closest_higher_strike1], 3),
                    "pcrv_up2": round(strike_PCRv_dict[closest_higher_strike2], 3),
                    "pcrv_up3": round(strike_PCRv_dict[closest_higher_strike3], 3),
                    "pcrv_up4": round(strike_PCRv_dict[closest_higher_strike4], 3),
                    "pcrv_down1": round(strike_PCRv_dict[closest_lower_strike1], 3),
                    "pcrv_down2": round(strike_PCRv_dict[closest_lower_strike2], 3),
                    "pcrv_down3": round(strike_PCRv_dict[closest_lower_strike3], 3),
                    "pcrv_down4": round(strike_PCRv_dict[closest_lower_strike4], 3),
                    "pcroi_up1": round(strike_PCRoi_dict[closest_higher_strike1], 3),
                    "pcroi_up2": round(strike_PCRoi_dict[closest_higher_strike2], 3),
                    "pcroi_up3": round(strike_PCRoi_dict[closest_higher_strike3], 3),
                    "pcroi_up4": round(strike_PCRoi_dict[closest_higher_strike4], 3),
                    "pcroi_down1": round(strike_PCRoi_dict[closest_lower_strike1], 3),
                    "pcroi_down2": round(strike_PCRoi_dict[closest_lower_strike2], 3),
                    "pcroi_down3": round(strike_PCRoi_dict[closest_lower_strike3], 3),
                    "pcroi_down4": round(strike_PCRoi_dict[closest_lower_strike4], 3),
                    "itm_pcr_vol": round(ITM_PC_Ratio_Vol, 2),
                    "itm_pcr_oi": round(ITM_PC_Ratio_OI, 3),
                    "itm_pcrv_up1": strike_ITMPCRv_dict[closest_higher_strike1],
                    "itm_pcrv_up2": strike_ITMPCRv_dict[closest_higher_strike2],
                    "itm_pcrv_up3": strike_ITMPCRv_dict[closest_higher_strike3],
                    "itm_pcrv_up4": strike_ITMPCRv_dict[closest_higher_strike4],
                    "itm_pcrv_down1": strike_ITMPCRv_dict[closest_lower_strike1],
                    "itm_pcrv_down2": strike_ITMPCRv_dict[closest_lower_strike2],
                    "itm_pcrv_down3": strike_ITMPCRv_dict[closest_lower_strike3],
                    "itm_pcrv_down4": strike_ITMPCRv_dict[closest_lower_strike4],
                    "itm_pcroi_up1": strike_ITMPCRoi_dict[closest_higher_strike1],
                    "itm_pcroi_up2": strike_ITMPCRoi_dict[closest_higher_strike2],
                    "itm_pcroi_up3": strike_ITMPCRoi_dict[closest_higher_strike3],
                    "itm_pcroi_up4": strike_ITMPCRoi_dict[closest_higher_strike4],
                    "itm_pcroi_down1": strike_ITMPCRoi_dict[closest_lower_strike1],
                    "itm_pcroi_down2": strike_ITMPCRoi_dict[closest_lower_strike2],
                    "itm_pcroi_down3": strike_ITMPCRoi_dict[closest_lower_strike3],
                    "itm_pcroi_down4": strike_ITMPCRoi_dict[closest_lower_strike4],
                    "itm_oi": ITM_OI,
                    "total_oi": all_OI,
                    "itm_contracts_percent": ITM_OI / all_OI,
                    "net_iv": round(Net_IV, 3),
                    "net_itm_iv": round(ITM_Avg_Net_IV, 3),
                    "net_iv_mp": round(Net_IV_at_MP, 3),
                    # "Net IV LAC": round(Net_IV_Closest_Strike_LAC, 3),
                    "niv_current_strike": round(NIV_CurrentStrike, 3),
                    "niv_1higher_strike": round(NIV_1HigherStrike, 3),
                    "niv_1lower_strike": round(NIV_1LowerStrike, 3),
                    "niv_2higher_strike": round(NIV_2HigherStrike, 3),
                    "niv_2lower_strike": round(NIV_2LowerStrike, 3),
                    "niv_3higher_strike": round(NIV_3HigherStrike, 3),
                    "niv_3lower_strike": round(NIV_3LowerStrike, 3),
                    "niv_4higher_strike": round(NIV_4HigherStrike, 3),
                    "niv_4lower_strike": round(NIV_4LowerStrike, 3),
                    ###Positive number means NIV highers are higher, and price will drop.
                    # TODO should do as percentage change from total niv numbers to see if its big diff.
                    "niv_highers_minus_lowers1thru2": (NIV_1HigherStrike + NIV_2HigherStrike)
                                               - (NIV_1LowerStrike + NIV_2LowerStrike),
                    "niv_highers_minus_lowers1thru4": (
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
                    "niv_1thru2_avg_percent_from_mean": (
                                                   (
                                                           (NIV_1HigherStrike + NIV_2HigherStrike)
                                                           - (NIV_1LowerStrike + NIV_2LowerStrike)
                                                   )
                                                   / (
                                                           (
                                                                   NIV_1HigherStrike
                                                                   + NIV_2HigherStrike
                                                                   + NIV_1LowerStrike
                                                                   + NIV_2LowerStrike + 2e-308 #to remove possible div. by 0
                                                           )
                                                           / 4
                                                   )
                                           )
                                           * 100,
                    "niv_1thru4_avg_percent_from_mean": (
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
                                                                   + NIV_4LowerStrike + 2e-308 #to remove possible div. by 0
                                                           )
                                                           / 8
                                                   )
                                           )
                                           * 100,
                    ##TODO swap (/) with result = np.divide(x, y)
                    "niv_dividedby_oi": Net_IV / all_OI,
                    "itm_avg_niv_dividedby_itm_oi": ITM_Avg_Net_IV / ITM_OI,
                    "closest_strike_to_cp": closest_strike_currentprice,
                }
            )

            processed_data_df = pd.DataFrame(results)
            # Convert processed_data_df to list of dictionaries, with appropriate type conversions
            data_to_insert = processed_data_df
            data_to_insert["symbol_name"] = symbol_name #TODO  bshould be loopstarttime from teh pased in df.
            data_to_insert["fetch_timestamp"] = CurrentTime


            #  (No need to save to CSV anymore)
            return optionchain_df,  processed_data_df, ticker,data_to_insert # Return statements