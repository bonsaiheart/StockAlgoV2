import os

import pandas as pd
import Strategy_Testing.Trained_Models.trained_minute_models as tm

dir = "../data/historical_multiday_minute_DF"
for filename in os.listdir(dir):
    filepath = os.path.join(dir, filename)

    if filename.endswith(".csv"):
        dailyminutes_df = pd.read_csv(filepath)

    ##TODO cant locate files... somethign to do with base_dir mabe

    dailyminutes_df.dropna(axis=1, how="all", inplace=True)
    ###TODO adding new
    # print(new_data_df.shape)

    # predictions = loaded_model.predict(prep_df[features])
    #
    # predictions_df = pd.DataFrame(predictions, index=prep_df.index)
    # print(predictions_df)
    Buy_1hr_A1 = tm.Buy_1hr_A1(dailyminutes_df)
    print(Buy_1hr_A1)
    dailyminutes_df["Buy_1hr_A1"] = Buy_1hr_A1

    Sell_1hr_A1 = tm.Sell_1hr_A1(dailyminutes_df)
    dailyminutes_df["Sell_1hr_A1"] = Sell_1hr_A1

    Buy_20min_A1 = tm.Buy_20min_A1(dailyminutes_df)

    dailyminutes_df["Buy_20min_A1"] = tm.Buy_20min_A1

    Sell_20min_A1 = tm.Sell_20min_A1(dailyminutes_df)
    dailyminutes_df["Sell_20min_A1"] = Sell_20min_A1
    Buy_15min_A2 = tm.Buy_15min_A2(dailyminutes_df)
    dailyminutes_df["Buy_15min_A2"] = Buy_15min_A2

    Sell_15min_A2 = tm.Sell_15min_A2(dailyminutes_df)
    dailyminutes_df["Sell_15min_A2"] = Sell_15min_A2

    Buy_15min_A1 = tm.Buy_15min_A1(dailyminutes_df)
    dailyminutes_df["Buy_15min_A1"] = Buy_15min_A1

    Sell_15min_A1 = tm.Sell_15min_A1(dailyminutes_df)
    dailyminutes_df["Sell_15min_A1"] = Sell_15min_A1
    # dailyminutes_df_w_ALGO_results=dailyminutes_df
    Buy_5D = tm.Buy_5D(dailyminutes_df[["ITM PCRv Up4", "ITM PCRv Down4", "ITM PCRoi Down4", "RSI14"]])
    dailyminutes_df["Buy_5D"] = Buy_5D

    Sell_5D = tm.Sell_5D(dailyminutes_df[["Bonsai Ratio", "PCRv Up4", "ITM PCRv Up4", "ITM PCRoi Up4"]])
    dailyminutes_df["Sell_5D"] = Sell_5D

    Buy_5A = tm.Buy_5A(
        dailyminutes_df[
            [
                "Bonsai Ratio",
                "Bonsai Ratio 2",
                "B1/B2",
                "PCRv Up4",
                "PCRv Down4",
                "ITM PCRv Up4",
                "ITM PCRv Down4",
                "ITM PCRoi Up4",
                "ITM PCRoi Down4",
            ]
        ]
    )
    dailyminutes_df["Buy_5A"] = Buy_5A
    Sell_5A = tm.Sell_5A(
        dailyminutes_df[
            ["Bonsai Ratio", "B1/B2", "PCRv Up4", "ITM PCRv Up4", "ITM PCRv Down4", "ITM PCRoi Up4", "ITM PCRoi Down4"]
        ]
    )
    dailyminutes_df["Sell_5A"] = Sell_5A

    Buy_5B = tm.Buy_5B(
        dailyminutes_df[
            [
                "Bonsai Ratio",
                "Bonsai Ratio 2",
                "B1/B2",
                "PCRv Up4",
                "PCRv Down4",
                "ITM PCRv Up4",
                "ITM PCRv Down4",
                "ITM PCRoi Up4",
                "ITM PCRoi Down4",
            ]
        ]
    )
    dailyminutes_df["Buy_5B"] = Buy_5B

    Sell_5B = tm.Sell_5B(dailyminutes_df[["Bonsai Ratio", "B1/B2", "PCRv Up4", "ITM PCRoi Up4", "ITM PCRoi Down4"]])
    dailyminutes_df["Sell_5B"] = Sell_5B

    Buy_5C = tm.Buy_5C(
        dailyminutes_df[
            [
                "Bonsai Ratio",
                "B1/B2",
                "PCRv Up4",
                "PCRv Down4",
                "ITM PCRv Up4",
                "ITM PCRv Down4",
                "ITM PCRoi Up4",
                "ITM PCRoi Down4",
            ]
        ]
    )
    dailyminutes_df["Buy_5C"] = Buy_5C

    Sell_5C = tm.Sell_5C(dailyminutes_df[["Bonsai Ratio", "PCRv Up4", "ITM PCRv Up4", "ITM PCRoi Up4"]])
    dailyminutes_df["Sell_5C"] = Sell_5C

    Buy_A5 = tm.Buy_A5(
        dailyminutes_df[
            ["Bonsai Ratio", "Bonsai Ratio 2", "B1/B2", "PCRv Up4", "ITM PCRv Up4", "ITM PCRoi Up4", "ITM PCRoi Down4"]
        ]
    )
    dailyminutes_df["Buy_A5"] = Buy_A5
    Sell_A5 = tm.Sell_A5(
        dailyminutes_df[["Bonsai Ratio", "B1/B2", "PCRv Up4", "ITM PCRv Up4", "ITM PCRoi Up4", "ITM PCRoi Down4"]]
    )

    dailyminutes_df["Sell_A5"] = Sell_A5
    Buy_A4 = tm.Buy_A4(
        dailyminutes_df[
            ["Bonsai Ratio", "Bonsai Ratio 2", "B1/B2", "ITM PCRv Down4", "ITM PCRoi Up4", "ITM PCRoi Down4"]
        ]
    )
    dailyminutes_df["Buy_A4"] = Buy_A4

    Sell_A4 = tm.Sell_A4(
        dailyminutes_df[
            [
                "Bonsai Ratio",
                "Bonsai Ratio 2",
                "B1/B2",
                "PCRv Up4",
                "PCRv Down4",
                "ITM PCRv Up4",
                "ITM PCRv Down4",
                "ITM PCRoi Up4",
                "ITM PCRoi Down4",
            ]
        ]
    )

    dailyminutes_df["Sell_A4"] = Sell_A4

    Buy_A3 = tm.Buy_A3(
        dailyminutes_df[
            ["Bonsai Ratio", "Bonsai Ratio 2", "B1/B2", "ITM PCRv Down4", "ITM PCRoi Up4", "ITM PCRoi Down4"]
        ]
    )
    dailyminutes_df["Buy_A3"] = Buy_A3

    Sell_A3 = tm.Sell_A3(
        dailyminutes_df[
            ["Bonsai Ratio", "B1/B2", "PCRv Up4", "ITM PCRv Up4", "ITM PCRv Down4", "ITM PCRoi Up4", "ITM PCRoi Down4"]
        ]
    )

    dailyminutes_df["Sell_A3"] = Sell_A3

    # Buy_30min_9sallaround = Buy_30min_9sallaround(dailyminutes_df[['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'PCRv Up4', 'PCRv Down4',
    # 'ITM PCRv Up4', 'ITM PCRv Down4', 'ITM PCRoi Up4', 'ITM PCRoi Down4',
    # 'RSI', 'AwesomeOsc', 'RSI14', 'RSI2', 'AwesomeOsc5_34']]   )
    # dailyminutes_df['Buy_30min_9sallaround'] = Buy_30min_9sallaround

    Sell_30min_9sallaround = tm.Sell_30min_9sallaround(
        dailyminutes_df[["Bonsai Ratio", "B1/B2", "PCRv Up4", "ITM PCRv Up4", "ITM PCRoi Up4", "ITM PCRoi Down4"]]
    ).astype(int)

    dailyminutes_df["Sell_30min_9sallaround"] = Sell_30min_9sallaround

    # Trythisone2_4Buy = Trythisone2_4Buy(dailyminutes_df[['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'PCR-Vol', 'PCR-OI', 'PCRv Up2', 'PCRv Up3', 'PCRv Up4', 'PCRv Down2', 'PCRv Down3',
    # 'PCRv Down4', 'PCRoi Up2', 'PCRoi Up3', 'PCRoi Up4', 'PCRoi Down2',
    # 'PCRoi Down3', 'PCRoi Down4', 'ITM PCR-Vol', 'ITM PCR-OI',
    # 'ITM PCRv Up1', 'ITM PCRv Up2', 'ITM PCRv Up3', 'ITM PCRv Up4',
    # 'ITM PCRv Down1', 'ITM PCRv Down2', 'ITM PCRv Down3', 'ITM PCRv Down4',
    # 'ITM PCRoi Up2', 'ITM PCRoi Up3', 'ITM PCRoi Up4', 'ITM PCRoi Down2',
    # 'ITM PCRoi Down3', 'ITM PCRoi Down4', 'Net_IV', 'Net ITM IV',
    # 'NIV 2Higher Strike', 'NIV 2Lower Strike', 'NIV 3Higher Strike',
    # 'NIV 3Lower Strike', 'NIV 4Higher Strike', 'NIV 4Lower Strike',
    # 'NIV highers(-)lowers1-4', 'NIV 1-2 % from mean', 'NIV 1-4 % from mean',
    # 'RSI', 'AwesomeOsc', 'RSI14', 'RSI2', 'AwesomeOsc5_34']]   )
    # dailyminutes_df['Trythisone2_4Buy'] = Trythisone2_4Buy

    # Trythisone2_4Sell = Trythisone2_4Sell(
    # dailyminutes_df[['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'PCR-Vol', 'PCRv Up2',
    # 'PCRv Down3', 'PCRv Down4', 'PCRoi Up3', 'PCRoi Up4', 'PCRoi Down3',
    # 'ITM PCR-Vol', 'ITM PCR-OI', 'ITM PCRv Up1', 'ITM PCRv Up2',
    # 'ITM PCRv Up3', 'ITM PCRv Up4', 'ITM PCRv Down2', 'ITM PCRv Down3',
    # 'ITM PCRoi Up2', 'ITM PCRoi Up3', 'ITM PCRoi Up4', 'ITM PCRoi Down3',
    # 'ITM PCRoi Down4', 'RSI', 'AwesomeOsc', 'RSI14', 'RSI2']])
    #
    # dailyminutes_df['Trythisone2_4Sell'] = Trythisone2_4Sell

    A1_Buy = tm.A1_Buy(dailyminutes_df[["Bonsai Ratio", "Bonsai Ratio 2", "PCRoi Up1", "ITM PCRoi Up1"]])
    dailyminutes_df["A1_Buy"] = A1_Buy

    # A1_Sell = A1_Sell(
    #     dailyminutes_df[['Bonsai Ratio', 'Bonsai Ratio 2', 'PCRoi Up1', 'ITM PCRoi Up1', 'Net IV LAC']])
    #
    # dailyminutes_df['A1_Sell'] = A1_Sell

    # try:
    #     A2_Buy = trained_models.A2_Buy(dailyminutes_df[['Bonsai Ratio', 'Bonsai Ratio 2', 'PCRoi Up1', 'ITM PCRoi Up1']]   )
    #     dailyminutes_df['A2_Buy'] = A1_Buy
    #
    #     if A2_Buy[-1]:
    #         print("A1_Buy Signal")
    #         send_notifications.email_me_string("A1_Buy:", "Call",
    #                                            ticker)
    #         try:
    #             IB.ibAPI.placeCallBracketOrder(ticker, IB_option_date, ib_one_strike_below, DownOne_Call_Price, 1)
    #             IB.ibAPI.placeBuyBracketOrder(ticker, current_price)
    #
    #
    #             if ticker == "SPY":
    #                 print("sending tweet")
    #                 send_notifications.send_tweet_w_5hour_followup(ticker, current_price, 'up',
    #                                                            f"${ticker} has hit a temporal LOW at ${current_price}.This is a signal that the price has a high chance of rising significantly in a 3-5 hour window.")
    #
    #         except Exception as e:
    #             print(e)
    #         finally:
    #             pass
    #
    #
    #     else:
    #         print('No A1_Buy Signal')
    #
    # except KeyError as e1:
    #     print(Exception)
    # try:
    #     A2_Sell = trained_models.A1_Sell(
    #         dailyminutes_df[['Bonsai Ratio', 'Bonsai Ratio 2', 'PCRoi Up1', 'ITM PCRoi Up1', 'Net IV LAC']])
    #
    #     dailyminutes_df['A1_Sell'] = A1_Sell
    #     if A1_Sell[-1]:
    #         print('A1_Sell signal')
    #         send_notifications.email_me_string("A1_Sell", "Put",
    #                                            ticker)
    #
    #         try:
    #             IB.ibAPI.placePutBracketOrder(ticker, IB_option_date, ib_one_strike_above, UpOne_Put_Price, 1)
    #             IB.ibAPI.placeSellBracketOrder(ticker, current_price)
    #             print("sending tweet")
    #             send_notifications.send_tweet_w_5hour_followup(ticker, current_price, 'down',
    #                                                        f"${ticker} has hit a temporal HIGH at ${current_price}.This is a signal that the price has a high likelihood of falling significantly in a 3-5 hour window.")
    #
    #         except Exception as e:
    #             print(e)
    #         finally:
    #             pass
    #
    #
    #
    #     else:
    #         print('No A1_Sell Signal.')
    #
    # except Exception as e1:
    # try:
    #
    buy_signal1 = tm.get_buy_B1B2_Bonsai_Ratio_RSI_ITM_PCRVol_threshUp5_threshDown5_30_min_later_change_SPY(
        dailyminutes_df[["B1/B2", "Bonsai Ratio", "RSI", "ITM PCR-Vol"]]
    )
    dailyminutes_df["buy_signal1"] = buy_signal1
    #
    # sell_signal1 = get_sell_B1B2_Bonsai_Ratio_RSI_ITM_PCRVol_threshUp5_threshDown5_30_min_later_change_SPY(
    #     dailyminutes_df[["B1/B2", "Bonsai Ratio", "RSI", 'ITM PCR-Vol']])
    # dailyminutes_df['sell_signal1']= sell_signal1

    #
    # buy_signal2 = get_buy_signal_NEWONE_PRECISE(
    #     dailyminutes_df[['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2',
    #                   'PCR-Vol', 'PCRv Up1', 'PCRv Up2',
    #                   'PCRv Up3', 'PCRv Up4', 'PCRv Down1',
    #                   'PCRv Down2', 'PCRv Down3', 'PCRv Down4',
    #                   'ITM PCR-Vol', 'ITM PCRv Up1',
    #                   'ITM PCRv Up2', 'ITM PCRv Up3',
    #                   'ITM PCRv Up4', 'ITM PCRv Down1',
    #                   'ITM PCRv Down2', 'ITM PCRv Down3',
    #                   'ITM PCRv Down4', 'ITM PCRoi Up2',
    #                   'ITM OI', 'ITM Contracts %', 'Net_IV',
    #                   'Net ITM IV', 'NIV 1Lower Strike',
    #                   'NIV 2Higher Strike', 'NIV 2Lower Strike',
    #                   'NIV 3Higher Strike', 'NIV 3Lower Strike',
    #                   'NIV 4Higher Strike', 'NIV 4Lower Strike',
    #                   'NIV highers(-)lowers1-4',
    #                   'NIV 1-4 % from mean', 'RSI',
    #                   'AwesomeOsc']])
    #
    # dailyminutes_df['buy_signal2'] = buy_signal2

    # buy_signal3 = get_buy_signal_NEWONE_TESTED_WELL_MOSTLY_UP(dailyminutes_df[
    #                                                                              ['Bonsai Ratio', 'Bonsai Ratio 2',
    #                                                                               'PCR-Vol', 'PCRv Down1',
    #                                                                               'PCRv Down2',
    #                                                                               'PCRv Down3', 'ITM PCRv Up3',
    #                                                                               'ITM PCRv Up4', 'ITM PCRv Down2',
    #                                                                               'ITM PCRv Down3', 'Net_IV',
    #                                                                               'NIV 2Lower Strike',
    #                                                                               'NIV 4Higher Strike',
    #                                                                               'NIV highers(-)lowers1-4']
    #                                                                          ])
    # dailyminutes_df['buy_signal3'] = buy_signal3

    #
    # new_buy_signal1 = get_buy_signal_1to4hourNewGreatPrecNumbersBonsai1NETitmIV(dailyminutes_df[
    #                                                                                ['Bonsai Ratio','Net ITM IV','RSI']   ])
    # dailyminutes_df['new_buy_signal1'] = new_buy_signal1
    #
    #

    # new_sell_signal1 = get_sell_signal_1to4hourNewGreatPrecNumbersBonsai1NETitmIV(
    #     dailyminutes_df[["Bonsai Ratio", "Net ITM IV"]])
    #
    # dailyminutes_df['new_sell_signal1'] = new_sell_signal1
    #
    # new_buy_signal2 = get_buy_signal_NewPerhapsExcellentTargetDown5to15minSPY(dailyminutes_df[
    #                                                                                ['Bonsai Ratio','Net ITM IV']   ])
    # dailyminutes_df['new_buy_signal2'] = new_buy_signal2

    # new_sell_signal2 = get_sell_signal_NewPerhapsExcellentTargetDown5to15minSPY(
    #     dailyminutes_df[["Bonsai Ratio", "Net ITM IV",'RSI']])
    # dailyminutes_df['new_sell_signal2'] = new_sell_signal2

    #     #     sell_signal2 = trained_models.get_sell_signal_NEWONE_PRECISE(dailyminutes_df[
    #     #                                                                      ['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2',
    #     #                                                                       'PCR-Vol', 'PCRv Up1', 'PCRv Up2', 'PCRv Up3',
    #     #                                                                       'PCRv Up4', 'PCRv Down1', 'PCRv Down2',
    #     #                                                                       'PCRv Down3', 'PCRv Down4', 'PCRoi Up1',
    #     #                                                                       'PCRoi Down1', 'PCRoi Down2', 'PCRoi Down3',
    #     #                                                                       'PCRoi Down4', 'ITM PCR-Vol', 'ITM PCRv Up1',
    #     #                                                                       'ITM PCRv Up2', 'ITM PCRv Up3',
    #     #                                                                       'ITM PCRv Up4',
    #     #                                                                       'ITM PCRv Down1', 'ITM PCRv Down2',
    #     #                                                                       'ITM PCRv Down3', 'ITM PCRv Down4',
    #     #                                                                       'ITM PCRoi Up1', 'ITM PCRoi Up3',
    #     #                                                                       'ITM PCRoi Down4', 'ITM OI',
    #     #                                                                       'ITM Contracts %',
    #     #                                                                       'Net_IV', 'Net ITM IV', 'NIV 1Higher Strike',
    #     #                                                                       'NIV 1Lower Strike', 'NIV 2Higher Strike',
    #     #                                                                       'NIV 2Lower Strike', 'NIV 3Higher Strike',
    #     #                                                                       'NIV 3Lower Strike', 'NIV 4Higher Strike',
    #     #                                                                       'NIV 4Lower Strike',
    #     #                                                                       'NIV highers(-)lowers1-2',
    #     #                                                                       'NIV highers(-)lowers1-4',
    #     #                                                                       'NIV 1-2 % from mean',
    #     #                                                                       'NIV 1-4 % from mean', 'RSI', 'AwesomeOsc']
    #     #                                                                  ])
    #     #     if sell_signal2[-1]:
    #     #         send_notifications.email_me_string("sell_signal2[-1]:", "Put",
    #     #                                            ticker)
    #     #         try:
    #     #             IB.ibAPI.placePutBracketOrder(ticker, IB_option_date, ib_one_strike_above, UpOne_Put_Price, 1)
    #     #         except Exception as e:
    #     #             print(e)
    #     #         finally:
    #     #             pass
    #     #         print('Sell signal!')
    #     #     else:
    #     #         print('No sell signal.')
    #     #
    #     #
    #     # except Exception as e1:
    #     #     print(Exception)
    #
    #     # try:
    #     #     sell_signal3 = trained_models.get_sell_signal_NEWONE_TESTED_WELL_MOSTLY_UP(dailyminutes_df[
    #     #                                                                                    ['Bonsai Ratio',
    #     #                                                                                     'Bonsai Ratio 2',
    #     #                                                                                     'B1/B2', 'PCR-Vol', 'PCRv Up1',
    #     #                                                                                     'PCRv Up2', 'PCRv Up3',
    #     #                                                                                     'PCRv Up4',
    #     #                                                                                     'PCRv Down1', 'PCRv Down2',
    #     #                                                                                     'PCRv Down3', 'PCRv Down4',
    #     #                                                                                     'PCRoi Up1', 'PCRoi Up2',
    #     #                                                                                     'PCRoi Up3', 'PCRoi Up4',
    #     #                                                                                     'PCRoi Down3', 'PCRoi Down4',
    #     #                                                                                     'ITM PCR-Vol', 'ITM PCR-OI',
    #     #                                                                                     'ITM PCRv Up1', 'ITM PCRv Up2',
    #     #                                                                                     'ITM PCRv Up3', 'ITM PCRv Up4',
    #     #                                                                                     'ITM PCRv Down1',
    #     #                                                                                     'ITM PCRv Down2',
    #     #                                                                                     'ITM PCRv Down3',
    #     #                                                                                     'ITM PCRv Down4',
    #     #                                                                                     'ITM PCRoi Up1',
    #     #                                                                                     'ITM PCRoi Up2',
    #     #                                                                                     'ITM PCRoi Up3',
    #     #                                                                                     'ITM PCRoi Up4',
    #     #                                                                                     'ITM PCRoi Down1',
    #     #                                                                                     'ITM PCRoi Down2',
    #     #                                                                                     'ITM PCRoi Down4', 'ITM OI',
    #     #                                                                                     'Total OI', 'ITM Contracts %',
    #     #                                                                                     'Net_IV', 'Net ITM IV',
    #     #                                                                                     'NIV 1Higher Strike',
    #     #                                                                                     'NIV 1Lower Strike',
    #     #                                                                                     'NIV 2Higher Strike',
    #     #                                                                                     'NIV 2Lower Strike',
    #     #                                                                                     'NIV 3Higher Strike',
    #     #                                                                                     'NIV 3Lower Strike',
    #     #                                                                                     'NIV 4Higher Strike',
    #     #                                                                                     'NIV 4Lower Strike',
    #     #                                                                                     'NIV highers(-)lowers1-2',
    #     #                                                                                     'NIV highers(-)lowers1-4',
    #     #                                                                                     'NIV 1-2 % from mean',
    #     #                                                                                     'NIV 1-4 % from mean', 'RSI']
    #     #                                                                                ])
    #     #     print(sell_signal3)
    #     #     if sell_signal3[-1]:
    #     #         print("sell signal 333333333333333333333333333333")
    #     #         send_notifications.email_me_string("sell_signal3[-1]:", "Put",
    #     #                                            ticker)
    #     #         try:
    #     #             IB.ibAPI.placePutBracketOrder(ticker, IB_option_date, ib_one_strike_above, UpOne_Put_Price,1)
    #     #         except Exception as e:
    #     #             print(e)
    #     #         finally:
    #     #             pass
    #     #
    #     #         print('Sell signal!')
    #     #     else:
    #     #         print('No sell signal 3.')
    #     #
    #     # except KeyError as e1:
    #     #     print(Exception)
    #
    #
    #     if dailyminutes_df['B2/B1'].iloc[-1] >500 and dailyminutes_df['Bonsai Ratio'].iloc[-1]<.0001 and dailyminutes_df['ITM PCRv Up2'].iloc[-1]<.01 and dailyminutes_df['ITM PCRv Down2'].iloc[-1]<5 and dailyminutes_df['NIV 1-2 % from mean'].iloc[-1]>dailyminutes_df['NIV 1-4 % from mean'].iloc[-1]>0:
    #         send_notifications.email_me_string("['B2/B1'][-1] >500 and dailyminutes_df['Bonsai Ratio'][0]<.0001 and dailyminutes_df['ITM PCRv Up2'][0]<.01 and dailyminutes_df['ITM PCRv Down2'][0]<5 and dailyminutes_df['NIV 1-2 % from mean'][0]>dailyminutes_df['NIV 1-4 % from mean'][0]>0:", "Put",
    #                                            ticker)
    #         # try:
    #         #     # IB.ibAPI.placePutBracketOrder(ticker,IB_option_date,ib_one_strike_above, UpOne_Put_Price,10,custom_takeprofit=1.2)
    #         # except Exception as e:
    #         #     print(e)
    #         # finally:
    #         #     pass
    #         if ticker == "SPY":
    #             send_notifications.send_tweet_w_countdown_followup(ticker, current_price, 'down', f"${ticker} looks ripe for a short term drop at ${current_price}. [STdownf1]")
    # # 1.15-(hold until) 0 and <0.0, hold call until .3   (hold them until the b1/b2 doubles/halves?) with conditions to make sure its profitable.
    #
    #     if dailyminutes_df['B1/B2'].iloc[-1] > 1.15 and dailyminutes_df['RSI'].iloc[-1]<30:
    #         send_notifications.email_me_string("dailyminutes_df['B1/B2'][0] > 1.15 and dailyminutes_df['RSI'][0]<30:", "Call",
    #                                            ticker)
    #         # try:
    #         #     # IB.ibAPI.placeCallBracketOrder(ticker,IB_option_date,ib_one_strike_below, DownOne_Call_Price,1)
    #         # except Exception as e:
    #         #     print(e)
    #         # finally:
    #         #     pass
    #         # TradierAPI.buy(x)
    #
    #         #
    #         #         IB.ibAPI.placeCallBracketOrder(ticker,IB_option_date,ib_one_strike_below, DownOne_Call_Price,1), response.status_code, json_response)
    #
    #         if ticker=="SPY":
    #             send_notifications.send_tweet_w_countdown_followup(ticker, current_price, 'up', f"${ticker} has hit a trough at ${current_price}. Short term upwards movement expected.")
    # ####THis one is good for a very short term peak before drop.  Maybe tighter profit/loss
    #     if dailyminutes_df['B1/B2'].iloc[-1] < 0.25 and dailyminutes_df["RSI"].iloc[-1]>70:
    #         send_notifications.email_me_string("dailyminutes_df['B1/B2'][-1] < 0.25 and dailyminutes_df['RSI'][-1]>77:", "Put",
    #                                            ticker)
    #         # try:
    #         #     # IB.ibAPI.placePutBracketOrder(ticker,IB_option_date,ib_one_strike_above, UpOne_Put_Price,1)
    #         #
    #         # except Exception as e:
    #         #     print(e)
    #         # finally:
    #         #     pass
    #         if ticker == "SPY":
    #             send_notifications.send_tweet_w_countdown_followup(ticker, current_price, 'down', f"${ticker} has hit a peak at ${current_price}. Short term downwards movement expected.")
    #
    # #### TESTING JUST RSI
    #     if dailyminutes_df['RSI'].iloc[-1] < 22:
    #         send_notifications.email_me_string("['RSI'][-1] < 22:", "Call",
    #                                            ticker)
    #         # try:
    #         #     # IB.ibAPI.placeCallBracketOrder(ticker,IB_option_date,ib_one_strike_below, DownOne_Call_Price,1,custom_takeprofit=1.01)
    #         # except Exception as e:
    #         #     print(e)
    #         # finally:
    #         #     pass
    #         if ticker == "SPY":
    #             send_notifications.send_tweet_w_countdown_followup(ticker, current_price, 'up',
    #                                           f"${ticker} has hit a trough at ${current_price}. Short term upwards movement expected.")
    #         ####THis one is good for a very short term peak before drop.  Maybe tighter profit/loss
    #     if dailyminutes_df['RSI'].iloc[-1] > 80 and dailyminutes_df['RSI14'].iloc[-1]>75:
    #         send_notifications.email_me_string("['RSI'].iloc[-1] > 80 and dailyminutes_df['RSI14']>75:", "Put",
    #                                            ticker)
    # try:
    #     # IB.ibAPI.placePutBracketOrder(ticker,IB_option_date,ib_one_strike_above, UpOne_Put_Price,1,custom_takeprofit=1.02)
    # except Exception as e:
    #     print(e)
    # finally:
    #     pass

    ###JUST b1/b2
    # if dailyminutes_df['B1/B2'].iloc[-1] > 1.15 :
    #     send_notifications.email_me_string("['B1/B2''][-1] > 1.15:", "Call",
    #                                        ticker)
    #     try:
    #         IB.ibAPI.placeCallBracketOrder(ticker,IB_option_date,ib_one_strike_below, DownOne_Call_Price,1,custom_takeprofit=1.3,custom_trailamount=.7)
    #     except Exception as e:
    #         print(e)
    #     finally:
    #         pass

    # ####THis one is good for a very short term peak before drop.  Maybe tighter profit/loss
    # if dailyminutes_df['B1/B2'].iloc[-1] < 0.25 :
    #     send_notifications.email_me_string("['B1/B2'][-1] < 0.25:", "Put",
    #                                            ticker)
    #     try:
    #         IB.ibAPI.placePutBracketOrder(ticker,IB_option_date,ib_one_strike_above, UpOne_Put_Price,1,custom_takeprofit=1.005)
    #     except Exception as e:
    #         print(e)
    #     finally:
    #         pass

    # predictor_values = {'Bonsai Ratio': .0007, 'ITM PCR-Vol': 20}
    # predictor_df = pd.DataFrame(predictor_values, index=[0])

    #
    # except (ValueError, ConnectionError) as e:
    #     logging.error('Error: %s', e)
    #
    #     pass

    # send_notifications.email_me_string(order, response.status_code, json_response)
    # send_notifications.email_me_string(order,response.status_code,json_response)

    # buy_signal2 = trained_models.get_buy_signal_B1B2_RSI_1hr_threshUp7(dailyminutes_df[["B1/B2", "RSI"]].head(1))
    # if buy_signal2:
    #     x = (f'get_buy_signal_b1b2_RSI_1hr_thresh7  {ticker}',
    #          f"{optionchain.loc[optionchain['c_contractSymbol'] == call_contract]['Call_LastPrice'].values[0]}",
    #          "1",
    #          call_contract, .8, 1.25
    #          )
    #
    #     TradierAPI.buy(x)
    #     print('Buy signal!')
    # else:
    #     print('No buy signal.')
    # sell_signal2 = trained_models.get_sell_signal_B1B2_RSI_1hr_threshDown7(dailyminutes_df[["B1/B2", "RSI"]].head(1))
    # if sell_signal2:
    #     x = (f'get_sell_signal_b1b2_RSI_1hr_thresh7 {ticker}',
    #          f"{optionchain.loc[optionchain['p_contractSymbol'] == put_contract]['Put_LastPrice'].values[0]}",
    #          "1",
    #          put_contract, .8, 1.25
    #          )
    #
    #     TradierAPI.buy(x)
    #     print('Sell signal!')
    # else:
    #     print('No sell signal.')
    # buy_signal3 = trained_models.get_buy_B1B2_Bonsai_Ratio_RSI_ITM_PCRVol_threshUp7_threshDown7_30_min_later_change_TSLA(dailyminutes_df[["B1/B2","Bonsai Ratio","RSI",'ITM PCR-Vol']].head(1))
    #
    # if buy_signal3:
    #     x = (f'get_buy_B1B2_Bonsai_Ratio_RSI_ITM_PCRVol_threshUp7_threshDown7_30_min_later_change_TSLA  {ticker}',
    #          f"{optionchain.loc[optionchain['c_contractSymbol'] == call_contract]['Call_LastPrice'].values[0]}",
    #          "1",
    #          call_contract, .8, 1.2
    #          )
    #
    #     TradierAPI.buy(x)
    #     print('Buy signal!')
    # else:
    #     print('No buy signal.')
    # sell_signal3 = trained_models.get_sell_B1B2_Bonsai_Ratio_RSI_ITM_PCRVol_threshUp7_threshDown7_30_min_later_change_TSLA(dailyminutes_df[["B1/B2","Bonsai Ratio","RSI",'ITM PCR-Vol']].head(1))
    # if sell_signal3:
    #     x = (f'get_sell_B1B2_Bonsai_Ratio_RSI_ITM_PCRVol_threshUp7_threshDown7_30_min_later_change_TSLA {ticker}',
    #          f"{optionchain.loc[optionchain['p_contractSymbol'] == put_contract]['Put_LastPrice'].values[0]}",
    #          "1",
    #          put_contract, .8, 1.2
    #          )
    #
    #     TradierAPI.buy(x)
    #     print('Sell signal!')
    # else:
    #     print('No sell signal.')
    # if dailyminutes_df["B1/B2"][0] > 1.15 :
    #     x = (f'b1/b2>1.15  {ticker}',
    #         f"{optionchain.loc[optionchain['c_contractSymbol'] == call_contract]['Call_LastPrice'].values[0]}",
    #         "10",
    #         call_contract, .6, 1.6
    #     )
    #
    #     TradierAPI.buy(x)
    #
    # if dailyminutes_df["B1/B2"][0] < 0.01:
    #     x = (f'b1/b2<.01  {ticker}',
    #         f"{optionchain.loc[optionchain['p_contractSymbol'] == put_contract]['Put_LastPrice'].values[0]}",
    #         "10",
    #         put_contract, .6, 1.6
    #     )
    #
    #     TradierAPI.buy(x)

    # if dailyminutes_df["NIV 1-2 % from mean"][0] < -100 and dailyminutes_df["NIV 1-4 % from mean"][0] <-200:
    #     x = ("NIV 1-2 % from mean< -100 & NIV 1-4 % from mean<-200",
    #         f"{optionchain.loc[optionchain['p_contractSymbol'] == put_contract]['Put_LastPrice'].values[0]  }",
    #         "8",
    #         put_contract, .6,1.6
    #     )
    #
    #     TradierAPI.buy(x)
    # if dailyminutes_df["NIV 1-2 % from mean"][0] > 100 and dailyminutes_df["NIV 1-4 % from mean"][0] > 200:
    #     x = ('NIV 1-2 % from mean> 100 & NIV 1-4 % from mean > 200',
    #         f"{optionchain.loc[optionchain['c_contractSymbol'] == call_contract]['Call_LastPrice'].values[0]}",
    #         "7",
    #         call_contract, .6, 1.6
    #     )
    #
    #     TradierAPI.buy(x)
    # if dailyminutes_df["NIV highers(-)lowers1-4"][0] < -20:
    #     x = ("NIV highers(-)lowers1-4 < -20",
    #         f"{optionchain.loc[optionchain['c_contractSymbol'] == call_contract]['Call_LastPrice'].values[0]  }",
    #         "6",
    #         call_contract, .6,1.6
    #     )
    #
    #     TradierAPI.buy(x)
    # if dailyminutes_df["NIV highers(-)lowers1-4"][0] > 20:
    #     x = ('NIV highers(-)lowers1-4> 20'
    #         f"{optionchain.loc[optionchain['p_contractSymbol'] == put_contract]['Put_LastPrice'].values[0]}",
    #         "5",
    #         put_contract, .6, 1.6
    #     )

    # TradierAPI.buy(x)

    # if dailyminutes_df["ITM PCR-Vol"][0] > 1.3 and dailyminutes_df["RSI"][0] > 70:
    #     x = (
    #         f"{optionchain.loc[optionchain['p_contractSymbol'] == put_contract]['Put_LastPrice'].values[0] }",
    #         "99",
    #         put_contract,.9,1.05
    #     )
    #
    #     TradierAPI.buy(x)

    #     ###MADE 100% on this near close 5/12
    # if dailyminutes_df["Bonsai Ratio"][0] < .8 and dailyminutes_df["ITM PCR-Vol"][0] < 0.8 and dailyminutes_df["RSI"][0] < 30:
    #     x = ('Bonsai Ratio < .8 and ITM PCR-Vol < 0.8',
    #         f"{optionchain.loc[optionchain['c_contractSymbol'] == call_contract]['Call_LastPrice'].values[0]}",
    #         "3",
    #         call_contract,.8,1.2
    #     )
    #     print(x)
    #     TradierAPI.buy(x)
    #
    #
    # if dailyminutes_df["Bonsai Ratio"][0] > 1.5  and dailyminutes_df["ITM PCR-Vol"][0] > 1.2 and dailyminutes_df["RSI"][0] > 70:
    #     x = ('Bonsai Ratio > 1.5  andITM PCR-Vol > 1.2',
    #         f"{optionchain.loc[optionchain['p_contractSymbol'] == put_contract]['Put_LastPrice'].values[0] }",
    #         "2",
    #         put_contract,.9,1.05
    #     )
    #
    #     TradierAPI.buy(x)
    # if (
    #     dailyminutes_df["Bonsai Ratio"][0] < 0.7
    #     and dailyminutes_df["Net_IV"][0] < -50
    #     and dailyminutes_df["Net ITM IV"][0] > -41
    # ):
    #     x = ('Bonsai Ratio < 0.7 and Net_IV < -50 and Net ITM IV> -41',
    #         f"{optionchain.loc[optionchain['c_contractSymbol'] == call_contract]['Call_LastPrice'].values[0]}",
    #         "1",
    #         call_contract,.9,1.05
    #     )

    # TradierAPI.buy(x)
    dailyminutes_df.to_csv(filename)
