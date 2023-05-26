def perform_operations(

    StockLastTradeTime,
    this_minute_ta_frame,
    closest_exp_date,
):

    results = []



    data = pd.read_csv(f"data/optionchain/{ticker}/{YYMMDD}/{ticker}_{StockLastTradeTime}.csv")

    groups = data.groupby("ExpDate")
    # divide into groups by exp date, call info from group.
    for exp_date, group in groups:

        pain_list = []
        strike_LASTPRICExOI_list = []
        call_LASTPRICExOI_list = []
        put_LASTPRICExOI_list = []
        call_price_dict = (
            group.loc[group["Call_LastPrice"] >= 0, ["Strike", "Call_LastPrice"]].set_index("Strike").to_dict()
        )

        strike = group["Strike"]
        # pain is ITM puts/calls
        calls_LASTPRICExOI_dict = (
            group.loc[group["Calls_lastPriceXoi"] >= 0, ["Strike", "Calls_lastPriceXoi"]].set_index("Strike").to_dict()
        )
        puts_LASTPRICExOI_dict = (
            group.loc[group["Puts_lastPriceXoi"] >= 0, ["Strike", "Puts_lastPriceXoi"]].set_index("Strike").to_dict()
        )
        calls_DFSxOI_dict = (
            group.loc[group["Calls_dollarsFromStrikeXoi"] >= 0, ["Strike", "Calls_dollarsFromStrikeXoi"]]
            .set_index("Strike")
            .to_dict()
        )
        puts_DFSxOI_dict = (
            group.loc[group["Puts_dollarsFromStrikeXoi"] >= 0, ["Strike", "Puts_dollarsFromStrikeXoi"]]
            .set_index("Strike")
            .to_dict()
        )

        ITM_CallsVol = group.loc[(group["Strike"] <= current_price), "Call_Volume"].sum()
        ITM_PutsVol = group.loc[(group["Strike"] >= current_price), "Put_Volume"].sum()
        ITM_CallsOI = group.loc[(group["Strike"] <= current_price), "Call_OI"].sum()
        ITM_PutsOI = group.loc[(group["Strike"] >= current_price), "Put_OI"].sum()
        ITM_OI = ITM_CallsOI + ITM_PutsOI
        all_CallsVol = group.Call_Volume.sum()
        all_PutsVol = group.Put_Volume.sum()

        all_CallsOI = group.Call_OI.sum()
        all_PutsOI = group.Put_OI.sum()
        all_OI = all_PutsOI + all_CallsOI

        ITM_Call_IV = group.loc[(group["Strike"] <= current_price), "Call_IV"].sum()
        ITM_Put_IV = group.loc[(group["Strike"] >= current_price), "Put_IV"].sum()
        Call_IV = group["Call_IV"].sum()
        Put_IV = group["Put_IV"].sum()
        ITM_Avg_Net_IV = ITM_Call_IV - ITM_Put_IV
        Net_IV = Call_IV - Put_IV


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

        DFSxOI_dict = (
            group.loc[group["Puts_dollarsFromStrikeXoi"] >= 0, ["Strike", "Puts_dollarsFromStrikeXoi"]]
            .set_index("Strike")
            .to_dict()
        )

        # All_PC_Ratio =
        # Money_weighted_PC_Ratio =
        ###TODO figure out WHEN this needs to run... probalby after 6pm eastern and before mrkt open.  remove otm
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



        highest_premium_strike = max(strike_LASTPRICExOI_list, key=lambda x: x[1])[0]
        highest_premium_call = max(call_LASTPRICExOI_list, key=lambda x: x[1])[0]
        highest_premium_put = max(put_LASTPRICExOI_list, key=lambda x: x[1])[0]
        max_pain = min(pain_list, key=lambda x: x[1])[0]

        top_five_calls = group.loc[group["Call_OI"] > 0].sort_values(by="Call_OI", ascending=False).head(5)
        top_five_calls_dict = top_five_calls[["Strike", "Call_OI"]].set_index("Strike").to_dict()["Call_OI"]

        top_five_puts = group.loc[group["Put_OI"] > 0].sort_values(by="Put_OI", ascending=False).head(5)
        top_five_puts_dict = top_five_puts[["Strike", "Put_OI"]].set_index("Strike").to_dict()["Put_OI"]


        ### FINDING CLOSEST STRIKE TO LAC
        # target number from column A
        # calculate difference between target and each value in column B
        data["strike_lac_diff"] = group["Strike"].apply(lambda x: abs(x - last_adj_close))
        # find index of row with smallest difference

        if not group.empty:
            smallest_change_from_lac = data["strike_lac_diff"].abs().idxmin()
            closest_strike_lac = group.loc[smallest_change_from_lac, "Strike"]

            current_price_index = group["Strike"].sub(current_price).abs().idxmin()
            closest_strike_currentprice = group.loc[current_price_index, "Strike"]
            ###RETURNS index of strike closest to CP

            higher_strike_index1 = current_price_index + 1

            higher_strike_index2 = current_price_index + 2
            higher_strike_index3 = current_price_index + 3
            higher_strike_index4 = current_price_index + 4
            # get the index of the row with the closest lower strike
            lower_strike_index1 = current_price_index - 1
            lower_strike_index2 = current_price_index - 2
            lower_strike_index3 = current_price_index - 3

            lower_strike_index4 = current_price_index - 4
            # get the strikes for the closest higher and lower strikes
            try:

                closest_strike_currentprice = group.loc[current_price_index, "Strike"]

            except KeyError as e:
                closest_strike_currentprice = None
                print("KeyError:", e)
            try:
                closest_higher_strike1 = group.loc[higher_strike_index1, "Strike"]
            except KeyError:
                closest_higher_strike1 = None

            try:
                closest_higher_strike2 = group.loc[higher_strike_index2, "Strike"]
            except KeyError:
                closest_higher_strike2 = None

            try:
                closest_higher_strike3 = group.loc[higher_strike_index3, "Strike"]
            except KeyError:
                closest_higher_strike3 = None

            try:
                closest_higher_strike4 = group.loc[higher_strike_index4, "Strike"]
            except KeyError:
                closest_higher_strike4 = None

            try:
                closest_lower_strike1 = group.loc[lower_strike_index1, "Strike"]
            except KeyError:
                closest_lower_strike1 = None

            try:
                closest_lower_strike2 = group.loc[lower_strike_index2, "Strike"]
            except KeyError:
                closest_lower_strike2 = None

            try:
                closest_lower_strike3 = group.loc[lower_strike_index3, "Strike"]
            except KeyError:
                closest_lower_strike3 = None

            try:
                closest_lower_strike4 = group.loc[lower_strike_index4, "Strike"]
            except KeyError:
                closest_lower_strike4 = None

        else:
            # handle empty dataframe here
            closest_strike_lac = None
            closest_strike_currentprice = None
            closest_higher_strike1 = None
            closest_higher_strike2 = None
            closest_higher_strike3 = None
            closest_higher_strike4 = None
            closest_lower_strike1 = None
            closest_lower_strike2 = None
            closest_lower_strike3 = None
            closest_lower_strike4 = None

        # closest_strike_currentprice_dict[exp_date] = closest_strike_currentprice

        strikeindex_abovebelow = [
            closest_lower_strike4,
            closest_lower_strike3,
            closest_lower_strike2,
            closest_lower_strike1,
            closest_higher_strike1,
            closest_higher_strike2,
            closest_higher_strike3,
            closest_higher_strike4,
        ]
        strike_PCRv_dict = {}
        strike_PCRoi_dict = {}
        strike_ITMPCRv_dict = {}
        strike_ITMPCRoi_dict = {}
        for strike in strikeindex_abovebelow:
            if strike == None:
                strike_PCRv_dict[strike] = np.nan
            else:
                strikeindex_abovebelowput_volume = group.loc[group["Strike"] == strike, "Put_Volume"].values[0]
                strikeindex_abovebelowcall_volume = group.loc[group["Strike"] == strike, "Call_Volume"].values[0]
                if strikeindex_abovebelowcall_volume == 0:
                    strike_PCRv_dict[strike] = np.nan
                else:
                    strike_PCRv_dict[strike] = strikeindex_abovebelowput_volume / strikeindex_abovebelowcall_volume

        for strike in strikeindex_abovebelow:
            if strike == None:
                strike_PCRoi_dict[strike] = np.nan
            else:
                strikeindex_abovebelowput_oi = group.loc[group["Strike"] == strike, "Put_OI"].values[0]
                strikeindex_abovebelowcall_oi = group.loc[group["Strike"] == strike, "Call_OI"].values[0]
                if strikeindex_abovebelowcall_oi == 0:
                    strike_PCRoi_dict[strike] = np.nan
                else:
                    strike_PCRoi_dict[strike] = strikeindex_abovebelowput_oi / strikeindex_abovebelowcall_oi

        ##CP PCR V/oi
        cp_put_vol = group.loc[group["Strike"] == closest_strike_currentprice, "Put_Volume"].values[0]
        cp_call_vol = group.loc[group["Strike"] == closest_strike_currentprice, "Call_Volume"].values[0]
        if np.isnan(cp_put_vol) or np.isnan(cp_call_vol) or cp_call_vol == 0:
            PCRv_cp_strike = np.nan
        else:
            PCRv_cp_strike = cp_put_vol / cp_call_vol

        cp_put_OI = group.loc[group["Strike"] == closest_strike_currentprice, "Put_OI"].values[0]
        cp_call_OI = group.loc[group["Strike"] == closest_strike_currentprice, "Call_OI"].values[0]
        if np.isnan(cp_put_OI) or np.isnan(cp_call_OI) or cp_call_OI == 0:
            PCRoi_cp_strike = np.nan
        else:
            PCRoi_cp_strike = cp_put_OI / cp_call_OI

        ###MP V PCR
        mp_put_vol = group.loc[group["Strike"] == max_pain, "Put_Volume"].values[0]
        mp_call_vol = group.loc[group["Strike"] == max_pain, "Call_Volume"].values[0]

        if np.isnan(mp_put_vol) or np.isnan(mp_call_vol) or mp_call_vol == 0:
            PC_Ratio_Vol_atMP = np.nan
        else:
            PC_Ratio_Vol_atMP = mp_put_vol / mp_call_vol
        ##MP OI PCR
        mp_put_OI = group.loc[group["Strike"] == max_pain, "Put_OI"].values[0]
        mp_call_OI = group.loc[group["Strike"] == max_pain, "Call_OI"].values[0]
        if np.isnan(mp_put_OI) or np.isnan(mp_call_OI) or mp_call_OI == 0:
            PC_Ratio_OI_atMP = np.nan
        else:
            PC_Ratio_OI_atMP = mp_put_OI / mp_call_OI

        ####ITM for four up and four down.

        for strikeabovebelow in strikeindex_abovebelow:
            if strikeabovebelow == None:
                strike_ITMPCRv_dict[strikeabovebelow] = np.nan
            else:
                strike_ITMPCRvput_volume = group.loc[(group["Strike"] >= strikeabovebelow), "Put_Volume"].sum()
                strike_ITMPCRvcall_volume = group.loc[(group["Strike"] <= strikeabovebelow), "Call_Volume"].sum()
                if strike_ITMPCRvcall_volume == 0:
                    strike_ITMPCRv_dict[strikeabovebelow] = np.nan
                else:
                    strike_ITMPCRv_dict[strikeabovebelow] = strike_ITMPCRvput_volume / strike_ITMPCRvcall_volume

        for strikeabovebelow in strikeindex_abovebelow:
            if strikeabovebelow == None:
                strike_ITMPCRoi_dict[strikeabovebelow] = np.nan
            else:
                strike_ITMPCRoiput_volume = group.loc[(group["Strike"] >= strikeabovebelow), "Put_OI"].sum()
                strike_ITMPCRoicall_volume = group.loc[(group["Strike"] <= strikeabovebelow), "Call_OI"].sum()
                if strike_ITMPCRoicall_volume == 0:
                    strike_ITMPCRoi_dict[strikeabovebelow] = np.nan
                else:
                    strike_ITMPCRoi_dict[strikeabovebelow] = strike_ITMPCRoiput_volume / strike_ITMPCRoicall_volume

        ###TODO use above/below lac strikes instead of just closest.
        ##LAC V/oi PCR
        lac_put_vol = group.loc[group["Strike"] == closest_strike_lac, "Put_Volume"].values[0]
        lac_call_vol = group.loc[group["Strike"] == closest_strike_lac, "Call_Volume"].values[0]
        if np.isnan(lac_put_vol) or np.isnan(lac_call_vol) or lac_call_vol == 0:
            PC_Ratio_Vol_Closest_Strike_LAC = np.nan
        else:
            PC_Ratio_Vol_Closest_Strike_LAC = lac_put_vol / lac_call_vol

        lac_put_OI = group.loc[group["Strike"] == closest_strike_lac, "Put_OI"].values[0]
        lac_call_OI = group.loc[group["Strike"] == closest_strike_lac, "Call_OI"].values[0]
        if np.isnan(lac_put_OI) or np.isnan(lac_call_OI) or lac_call_OI == 0:
            PC_Ratio_OI_Closest_Strike_LAC = np.nan
        else:
            PC_Ratio_OI_Closest_Strike_LAC = lac_put_OI / lac_call_OI

        if np.isnan(PC_Ratio_Vol_atMP) or np.isnan(PC_Ratio_OI_atMP) or PC_Ratio_OI_atMP == 0:
            PCR_vol_OI_at_MP = np.nan
        else:
            PCR_vol_OI_at_MP = round((PC_Ratio_Vol_atMP / PC_Ratio_OI_atMP), 3)

        if (
            np.isnan(PC_Ratio_Vol_Closest_Strike_LAC)
            or np.isnan(PC_Ratio_OI_Closest_Strike_LAC)
            or PC_Ratio_OI_Closest_Strike_LAC == 0
        ):
            PCR_vol_OI_at_LAC = np.nan
        else:
            PCR_vol_OI_at_LAC = round((PC_Ratio_Vol_Closest_Strike_LAC / PC_Ratio_OI_Closest_Strike_LAC), 3)

        Net_Call_IV_at_MP = group.loc[(group["Strike"] == max_pain), "Call_IV"].sum()
        Net_Put_IV_at_MP = group.loc[(group["Strike"] == max_pain), "Put_IV"].sum()
        Net_IV_at_MP = Net_Call_IV_at_MP - Net_Put_IV_at_MP
        NIV_CurrentStrike = (group.loc[(group["Strike"] == closest_strike_currentprice), "Call_IV"].sum()) - (
            group.loc[(group["Strike"] == closest_strike_currentprice), "Put_IV"].sum()
        )
        NIV_1HigherStrike = (group.loc[(group["Strike"] == closest_higher_strike1), "Call_IV"].sum()) - (
            group.loc[(group["Strike"] == closest_higher_strike1), "Put_IV"].sum()
        )
        NIV_2HigherStrike = (group.loc[(group["Strike"] == closest_higher_strike2), "Call_IV"].sum()) - (
            group.loc[(group["Strike"] == closest_higher_strike2), "Put_IV"].sum()
        )
        NIV_3HigherStrike = (group.loc[(group["Strike"] == closest_higher_strike3), "Call_IV"].sum()) - (
            group.loc[(group["Strike"] == closest_higher_strike3), "Put_IV"].sum()
        )
        NIV_4HigherStrike = (group.loc[(group["Strike"] == closest_higher_strike4), "Call_IV"].sum()) - (
            group.loc[(group["Strike"] == closest_higher_strike4), "Put_IV"].sum()
        )
        NIV_1LowerStrike = (group.loc[(group["Strike"] == closest_lower_strike1), "Call_IV"].sum()) - (
            group.loc[(group["Strike"] == closest_lower_strike1), "Put_IV"].sum()
        )
        NIV_2LowerStrike = (group.loc[(group["Strike"] == closest_lower_strike2), "Call_IV"].sum()) - (
            group.loc[(group["Strike"] == closest_lower_strike2), "Put_IV"].sum()
        )
        NIV_3LowerStrike = (group.loc[(group["Strike"] == closest_lower_strike3), "Call_IV"].sum()) - (
            group.loc[(group["Strike"] == closest_lower_strike3), "Put_IV"].sum()
        )
        NIV_4LowerStrike = (group.loc[(group["Strike"] == closest_lower_strike4), "Call_IV"].sum()) - (
            group.loc[(group["Strike"] == closest_lower_strike4), "Put_IV"].sum()
        )

        Call_IV_Closest_Strike_LAC = group.loc[(group["Strike"] == closest_strike_lac), "Call_IV"].sum()
        Put_IV_Closest_Strike_LAC = group.loc[(group["Strike"] == closest_strike_lac), "Put_IV"].sum()
        Net_IV_Closest_Strike_LAC = Call_IV_Closest_Strike_LAC - Put_IV_Closest_Strike_LAC
        ###TODO error handling for scalar divide of zero denominator

        Bonsai_Ratio = ((ITM_PutsVol / all_PutsVol) * (ITM_PutsOI / all_PutsOI)) / (
            (ITM_CallsVol / all_CallsVol) * (ITM_CallsOI / all_CallsOI)
        )
        Bonsai2_Ratio = (
            (all_PutsOI == 0 or ITM_PutsOI == 0 or all_CallsOI == 0 or ITM_CallsVol == 0 or ITM_CallsOI == 0)
            and float("inf")
            or ((all_PutsVol / ITM_PutsVol) / (all_PutsOI / ITM_PutsOI))
            * ((all_CallsVol / ITM_CallsVol) / (all_CallsOI / ITM_CallsOI))
        )

        # Calculate the percentage change###TODO figure out how to look at bonsai %change, will need to transform to timesheet.
        # if last_Bonsai_Ratio is not None:
        #     bonsai_percent_change = ((Bonsai_Ratio - last_Bonsai_Ratio) / last_Bonsai_Ratio) * 100
        # else:
        #     bonsai_percent_change = 0.0
        # if last_Bonsai_Ratio_2 is not None:
        #     bonsai2_percent_change = ((Bonsai2_Ratio - last_Bonsai_Ratio_2) / last_Bonsai_Ratio_2) * 100
        # else:
        #     bonsai2_percent_change = 0.0

        round(strike_PCRv_dict[closest_higher_strike1], 3),

        results.append(
            {
                ###TODO change all price data to percentage change?
                ###TODO change closest strike to average of closest above/closest below
                "ExpDate": exp_date,
                "Current Stock Price": float(current_price),
                "Current SP % Change(LAC)": round(float(price_change_percent), 2),
                # 'IV 30': iv30,
                # 'IV 30 % change': iv30_change_percent,
                "Maximum Pain": max_pain,
                "Bonsai Ratio": round(Bonsai_Ratio, 5),
                # 'Bonsai %change': bonsai_percent_change,
                "Bonsai Ratio 2": round(Bonsai2_Ratio, 5),
                "B1/B2":round((Bonsai_Ratio/Bonsai2_Ratio),4),
                "B2/B1":round((Bonsai2_Ratio/Bonsai_Ratio),4),
                # 'Bonsai_2 %change': bonsai2_percent_change,
                # 'Maximum Pain % -LAC': round((last_adj_close-max_pain)/max_pain,2),
                # 'Implied % Move -previousMP': 0,
                # 'Implied % Move-LAC': round(implied_percentage_move, 2),
                # TODO ITM contract $ %
                # PCR
                "PCR-Vol": round(PC_Ratio_Vol, 3),
                "PCR-OI": round(PC_Ratio_OI, 3),
                # 'PCR Vol/OI': round(PC_Ratio_Vol / PC_Ratio_OI, 3),
                # 'ITM PCR Vol/OI': float('inf') if ITM_PC_Ratio_OI == 0 else 0 if ITM_PC_Ratio_Vol == 0 else round(ITM_PC_Ratio_Vol / ITM_PC_Ratio_OI, 3),
                # 'PCR @MP Vol/OI ': round((PC_Ratio_Vol_atMP / PC_Ratio_OI_atMP), 3),
                # 'PCR @LAC Vol/OI ': round(PCR_vol_OI_at_LAC, 3),
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
                #TODO should do as percentage change from total niv numbers to see if its big diff.
                "NIV highers(-)lowers1-2": (
                                                       NIV_1HigherStrike + NIV_2HigherStrike ) - (
                                                       NIV_1LowerStrike + NIV_2LowerStrike ),

                "NIV highers(-)lowers1-4": (NIV_1HigherStrike+NIV_2HigherStrike+NIV_3HigherStrike+NIV_4HigherStrike)-(NIV_1LowerStrike+NIV_2LowerStrike+NIV_3LowerStrike+NIV_4LowerStrike),
                "NIV 1-2 % from mean": (
                        ((NIV_1HigherStrike + NIV_2HigherStrike) - (
                                                   NIV_1LowerStrike + NIV_2LowerStrike))/((NIV_1HigherStrike+NIV_2HigherStrike+NIV_1LowerStrike+NIV_2LowerStrike)/4))*100,

                "NIV 1-4 % from mean": (
                    (NIV_1HigherStrike + NIV_2HigherStrike + NIV_3HigherStrike + NIV_4HigherStrike) - (
                                                       NIV_1LowerStrike + NIV_2LowerStrike + NIV_3LowerStrike + NIV_4LowerStrike)/((NIV_1HigherStrike+NIV_2HigherStrike+ NIV_3HigherStrike + NIV_4HigherStrike+NIV_1LowerStrike+NIV_2LowerStrike+NIV_3LowerStrike + NIV_4LowerStrike)/8))*100,
##TODO swap (/) with result = np.divide(x, y)
                "Net_IV/OI": Net_IV / all_OI,
                "Net ITM_IV/ITM_OI": ITM_Avg_Net_IV / ITM_OI,
                "Closest Strike to CP": closest_strike_currentprice,

            }
        )

        # if len(results) >= 2:
        #     # print(results[-2]['Maximum Pain'])
        #     implied_move_from_last_MP = ((max_pain - results[-2]['Maximum Pain']) / results[-2]['Maximum Pain']) * 100
        #     # print(implied_move_fom_last_MP)
        #     results[-1]['Implied % Move -previousMP'] = round(implied_move_from_last_MP,2)

        ##TODO Signal strength 1-3, bull/bear.
        #
        # if results('Bonsai Ratio') < 1:
        #     if results('Maximum Pain') < results('Current Stock Price'):

        # if results('Bonsai Ratio') > 1:

    df = pd.DataFrame(results)
    df["RSI"] = this_minute_ta_frame["RSI"]
    df["AwesomeOsc"] = this_minute_ta_frame["AwesomeOsc"]
    # this_minute_ta_frame['exp_date'] = '230427.0'
    # df = pd.concat([this_minute_ta_frame,df])
    # df['']
    output_dir = Path(f"data/ProcessedData/{ticker}/{YYMMDD}/")

    output_dir.mkdir(mode=0o755, parents=True, exist_ok=True)

    try:
            df.to_csv(f"data/ProcessedData/{ticker}/{YYMMDD}/{ticker}_{StockLastTradeTime}.csv", mode="x", index=False)
    except FileExistsError:
            df.to_csv(f"data/ProcessedData/{ticker}/{YYMMDD}/{ticker}_{StockLastTradeTime}.csv", index=False)
    return (
            f"data/optionchain/{ticker}/{YYMMDD}/{ticker}_{StockLastTradeTime}.csv",
            f"data/ProcessedData/{ticker}/{YYMMDD}/{ticker}_{StockLastTradeTime}.csv",
            closest_strike_currentprice,
            closest_exp_date,
            ticker,
        )
