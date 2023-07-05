import logging
import traceback

# Configure the logging settings
logging.basicConfig(filename='error.log', level=logging.ERROR)
from datetime import datetime,timedelta
from Strategy_Testing.trained_models import *
import numpy as np
from Strategy_Testing import trained_models
import pandas as pd
import IB.ibAPI
import send_notifications as send_notifications
logging.basicConfig(filename='trade_algos_error.log', level=logging.ERROR,    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
# dailyminutes_df= pd.read_csv('data/DailyMinutes/SPY/SPY_230616.csv')
# print(trained_models.get_sell_signal_NewPerhapsExcellentTargetDown5to15minSPY(
#             dailyminutes_df[["Bonsai Ratio", "Net ITM IV",'RSI']]))
now = datetime.now()
formatted_time = now.strftime("%y%m%d %H:%M EST")

def actions(optionchain, dailyminutes,dailyminuteswithALGOresults, processeddata, ticker, current_price):
    ###strikeindex_abovebelow is a list [lowest,3 lower,2 lower, 1 lower, 1 higher,2 higher,3 higher, 4 higher]
    import pandas as pd
    expdates_strikes_dict = {}
    for index, row in processeddata.iterrows():
        key = row['ExpDate']
        value = row['Closest Strike Above/Below(below to above,4 each) list']
        expdates_strikes_dict[key] = value
    # expiration_dates = processeddata['ExpDate'].astype(str)
    # strikeindex_abovebelow = processeddata['Closest Strike Above/Below(below to above,4 each) list']


    closest_exp_date =  list(expdates_strikes_dict.keys())[0]
    closest_strike_expdate_currentprice = expdates_strikes_dict[closest_exp_date][4]
    strikeindex_closest_expdate = expdates_strikes_dict[closest_exp_date]
    optionchain = pd.read_csv(optionchain)
    dailyminutes_df = pd.read_csv(dailyminutes)
    dailyminuteswithALGOresults_df = pd.read_csv(dailyminuteswithALGOresults)
    print(ticker, current_price)
    date_string = str(closest_exp_date)
    date_object = datetime.strptime(date_string, "%y%m%d")
    new_date_string = date_object.strftime("%y%m%d")
    IB_option_date = date_object.strftime("%Y%m%d")

####Different strikes converted to contract form.
    ##This one is the strike one above Closest to Current price strike


    if strikeindex_closest_expdate[0] != np.nan:
        ib_four_strike_below = strikeindex_closest_expdate[0]
        four_strike_below_closest_cp_strike_int_num = int(strikeindex_closest_expdate[0] * 1000)

    if strikeindex_closest_expdate[1] != np.nan:
        ib_three_strike_below = strikeindex_closest_expdate[1]
        three_strike_below_closest_cp_strike_int_num = int(strikeindex_closest_expdate[1] * 1000)
    if strikeindex_closest_expdate[2] != np.nan:
        ib_two_strike_below = strikeindex_closest_expdate[2]
        two_strike_below_closest_cp_strike_int_num = int(strikeindex_closest_expdate[2] * 1000)
    if strikeindex_closest_expdate[3] != np.nan:
        ib_one_strike_below = strikeindex_closest_expdate[3]
        one_strike_below_closest_cp_strike_int_num = int(strikeindex_closest_expdate[3] * 1000)
    if strikeindex_closest_expdate[4] != np.nan:
        ib_one_strike_above = strikeindex_closest_expdate[5]
        one_strike_above_closest_cp_strike_int_num = int(strikeindex_closest_expdate[4] * 1000)
    if strikeindex_closest_expdate[5] != np.nan:
        ib_two_strike_above = strikeindex_closest_expdate[6]
        two_strike_above_closest_cp_strike_int_num = int(strikeindex_closest_expdate[5] * 1000)
    if strikeindex_closest_expdate[6] != np.nan:
        ib_three_strike_above = strikeindex_closest_expdate[7]
        three_strike_above_closest_cp_strike_int_num = int(strikeindex_closest_expdate[6] * 1000)
    if strikeindex_closest_expdate[7] != np.nan:
        ib_four_strike_above = strikeindex_closest_expdate[8]
        four_strike_above_closest_cp_strike_int_num = int(strikeindex_closest_expdate[7] * 1000)
    # print(dailyminutes_df["Closest Strike to CP"])
    closest_strike_exp_int_num = int(strikeindex_closest_expdate[5] * 1000)  # Convert decimal to integer
    ###TODO add different exp date options in addition to diff strike optoins.

    one_strike_above_CCPS ="{:08d}".format(one_strike_above_closest_cp_strike_int_num)
    one_strike_below_CCPS ="{:08d}".format(one_strike_below_closest_cp_strike_int_num)
    two_strike_above_CCPS ="{:08d}".format(two_strike_above_closest_cp_strike_int_num)
    two_strike_below_CCPS ="{:08d}".format(two_strike_below_closest_cp_strike_int_num)
    three_strike_above_CCPS ="{:08d}".format(three_strike_above_closest_cp_strike_int_num)
    three_strike_below_CCPS ="{:08d}".format(three_strike_below_closest_cp_strike_int_num)
    four_strike_above_CCPS ="{:08d}".format(four_strike_above_closest_cp_strike_int_num)
    four_strike_below_CCPS ="{:08d}".format(four_strike_below_closest_cp_strike_int_num)
    closest_contract_strike = "{:08d}".format(closest_strike_exp_int_num)

    CCP_upone_call_contract = f"{ticker}{new_date_string}C{one_strike_above_CCPS}"
    CCP_upone_put_contract = f"{ticker}{new_date_string}P{one_strike_above_CCPS}"
    CCP_downone_call_contract = f"{ticker}{new_date_string}C{one_strike_below_CCPS}"
    CCP_downone_put_contract = f"{ticker}{new_date_string}P{one_strike_below_CCPS}"
    CCP_uptwo_call_contract = f"{ticker}{new_date_string}C{two_strike_above_CCPS}"
    CCP_uptwo_put_contract = f"{ticker}{new_date_string}P{two_strike_above_CCPS}"
    CCP_downtwo_call_contract = f"{ticker}{new_date_string}C{two_strike_below_CCPS}"
    CCP_downtwo_put_contract = f"{ticker}{new_date_string}P{two_strike_below_CCPS}"
    CCP_upthree_call_contract = f"{ticker}{new_date_string}C{three_strike_above_CCPS}"
    CCP_upthree_put_contract = f"{ticker}{new_date_string}P{three_strike_above_CCPS}"
    CCP_downthree_call_contract = f"{ticker}{new_date_string}C{three_strike_below_CCPS}"
    CCP_downthree_put_contract = f"{ticker}{new_date_string}P{three_strike_below_CCPS}"
    CCP_upfour_call_contract = f"{ticker}{new_date_string}C{four_strike_above_CCPS}"
    CCP_upfour_put_contract = f"{ticker}{new_date_string}P{four_strike_above_CCPS}"
    CCP_downfour_call_contract = f"{ticker}{new_date_string}C{four_strike_below_CCPS}"
    CCP_downfour_put_contract = f"{ticker}{new_date_string}P{four_strike_below_CCPS}"
    CCP_call_contract = f"{ticker}{new_date_string}C{closest_contract_strike}"
    CCP_put_contract = f"{ticker}{new_date_string}P{closest_contract_strike}"


    CCP_Put_Price = optionchain.loc[optionchain['p_contractSymbol'] == CCP_put_contract]['Put_LastPrice'].values[0]
    CCP_Call_Price = optionchain.loc[optionchain['c_contractSymbol'] == CCP_call_contract]['Call_LastPrice'].values[0]
    try:
        DownOne_Call_Price =optionchain.loc[optionchain['c_contractSymbol'] == CCP_downone_call_contract]['Call_LastPrice'].values[0]
        UpOne_Put_Price=optionchain.loc[optionchain['p_contractSymbol'] == CCP_upone_put_contract]['Put_LastPrice'].values[0]
        DownOne_Put_Price =optionchain.loc[optionchain['p_contractSymbol'] == CCP_downone_put_contract]['Put_LastPrice'].values[0]
        UpOne_Call_Price = optionchain.loc[optionchain['c_contractSymbol'] == CCP_upone_call_contract]['Call_LastPrice'].values[0]
        UpTwo_Call_Price = optionchain.loc[optionchain['c_contractSymbol'] == CCP_uptwo_call_contract]['Call_LastPrice'].values[0]
        DownTwo_Put_Price = optionchain.loc[optionchain['p_contractSymbol'] == CCP_downtwo_put_contract]['Put_LastPrice'].values[0]
        DownTwo_Call_Price =optionchain.loc[optionchain['c_contractSymbol'] == CCP_downtwo_call_contract]['Call_LastPrice'].values[0]
        UpTwo_Put_Price = optionchain.loc[optionchain['p_contractSymbol'] == CCP_uptwo_put_contract]['Put_LastPrice'].values[0]
        DownThree_Put_Price = optionchain.loc[optionchain['p_contractSymbol'] == CCP_downthree_put_contract]['Put_LastPrice'].values[0]
        DownThree_Call_Price = optionchain.loc[optionchain['c_contractSymbol'] == CCP_downthree_call_contract]['Call_LastPrice'].values[0]
        DownFour_Call_Price = optionchain.loc[optionchain['c_contractSymbol'] == CCP_downfour_call_contract]['Call_LastPrice'].values[0]
        UpThree_Call_Price = optionchain.loc[optionchain['c_contractSymbol'] == CCP_upthree_call_contract]['Call_LastPrice'].values[0]
        Upfour_Call_Price = optionchain.loc[optionchain['c_contractSymbol'] == CCP_upfour_call_contract]['Call_LastPrice'].values[0]

        DownFour_Put_Price = optionchain.loc[optionchain['p_contractSymbol'] == CCP_downfour_put_contract]['Put_LastPrice'].values[0]
        UpThree_Put_Price = optionchain.loc[optionchain['p_contractSymbol'] == CCP_upthree_put_contract]['Put_LastPrice'].values[0]
        Upfour_Put_Price = optionchain.loc[optionchain['p_contractSymbol'] == CCP_upfour_put_contract]['Put_LastPrice'].values[0]
    except Exception as e:
        print(e)
        traceback.print_exc()
        pass
    #             # IB.ibAPI.placeSellBracketOrder(ticker, current_price, "SELL")
    #         IB.ibAPI.placeBuyBracketOrder(ticker,current_price,"BUY")


    """
    Algo Rules go here:
    """
    # dailyminutes_df_w_ALGO_results=dailyminutes_df

    try:
        Buy_5C = trained_models.Buy_5C(
            dailyminutes_df[['Bonsai Ratio', 'B1/B2', 'PCRv Up4', 'PCRv Down4', 'ITM PCRv Up4',
       'ITM PCRv Down4', 'ITM PCRoi Up4', 'ITM PCRoi Down4']])
        dailyminuteswithALGOresults_df['Buy_5C'] = Buy_5C
        print(Buy_5C)
        if Buy_5C[-1]:
            print("Buy_5C Signal")
            send_notifications.email_me_string("Buy_5C:", "Call",
                                               ticker)
            try:
                IB.ibAPI.placeCallBracketOrder(ticker, IB_option_date, ib_one_strike_below, DownOne_Call_Price, 5,'buy_5c')
                # IB.ibAPI.placeBuyBracketOrder(ticker, current_price)


                print("sending tweet")
                send_notifications.send_tweet_w_countdown_followup(ticker, current_price, 'up',
                                                               f"${ticker} ${current_price}. 5 minutes to profit on a Call. #5C {formatted_time}",300)

            except Exception as e:
                print(e)
            finally:
                pass


        else:
            print('No Buy_5C Signal')

    except KeyError as e1:
        print(Exception)
    try:
        Sell_5C = trained_models.Sell_5C(
            dailyminutes_df[['Bonsai Ratio', 'PCRv Up4', 'ITM PCRv Up4', 'ITM PCRoi Up4']])

        dailyminuteswithALGOresults_df['Sell_5C'] = Sell_5C
        if Sell_5C[-1]:
            print('Sell_5C  Signal.')
            send_notifications.email_me_string("Sell_5C", "Put",
                                               ticker)

            try:
                IB.ibAPI.placePutBracketOrder(ticker, IB_option_date, ib_one_strike_above, UpOne_Put_Price, 5)

                print("sending tweet")
                send_notifications.send_tweet_w_countdown_followup(ticker, current_price, 'down',
                                                               f"${ticker}  ${current_price}. 5 min till profit on a PUT. #5C {formatted_time}", 300)

            except Exception as e:
                print(e)
            finally:
                pass



        else:
            print('No Sell_5C Signal.')

    except Exception as e1:
        print(Exception)
    try:
        Buy_5B = trained_models.Buy_5B(
            dailyminutes_df[['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'PCRv Up4', 'PCRv Down4',
       'ITM PCRv Up4', 'ITM PCRv Down4', 'ITM PCRoi Up4', 'ITM PCRoi Down4']])
        dailyminuteswithALGOresults_df['Buy_5B'] = Buy_5B
        print(Buy_5B)
        if Buy_5B[-1]:
            print("Buy_5B Signal")
            send_notifications.email_me_string("Buy_5B:", "Call",
                                               ticker)
            try:
                IB.ibAPI.placeCallBracketOrder(ticker, IB_option_date, ib_one_strike_below, DownOne_Call_Price, 5)
                # IB.ibAPI.placeBuyBracketOrder(ticker, current_price)

                if ticker == "SPY":
                    print("sending tweet")
                    send_notifications.send_tweet_w_countdown_followup(ticker, current_price, 'up',
                                                                   f"${ticker} ${current_price}. 5 minutes to profit on a Call. #5B {formatted_time}", 300)

            except Exception as e:
                print(e)
            finally:
                pass


        else:
            print('No Buy_5B Signal')

    except KeyError as e1:
        print(Exception)
    try:
        Sell_5B = trained_models.Sell_5B(
            dailyminutes_df[['Bonsai Ratio', 'B1/B2', 'PCRv Up4', 'ITM PCRoi Up4',
       'ITM PCRoi Down4']])

        dailyminuteswithALGOresults_df['Sell_5B'] = Sell_5B
        if Sell_5B[-1]:
            print('Sell_5B  Signal.')
            send_notifications.email_me_string("Sell_5B", "Put",
                                               ticker)
            send_notifications.send_tweet_w_countdown_followup(ticker, current_price, 'down',
                                                              f"${ticker}  ${current_price}. 5 min till profit on a PUT. #5B {formatted_time}",
                                                              300)
            try:
                IB.ibAPI.placePutBracketOrder(ticker, IB_option_date, ib_one_strike_above, UpOne_Put_Price, 5)

                print("sending tweet")


            except Exception as e:
                print(e)
            finally:
                pass



        else:
            print('No Sell_5B Signal.')

    except Exception as e1:
        print(Exception)
    try:
        Buy_5A = trained_models.Buy_5A(
            dailyminutes_df[['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'PCRv Up4', 'PCRv Down4',
       'ITM PCRv Up4', 'ITM PCRv Down4', 'ITM PCRoi Up4', 'ITM PCRoi Down4']])
        dailyminuteswithALGOresults_df['Buy_5A'] = Buy_5A
        print(Buy_5A)
        if Buy_5A[-1]:
            print("5A_Buy Signal")
            send_notifications.email_me_string("Buy_5A:", "Call",
                                               ticker)
            try:
                IB.ibAPI.placeCallBracketOrder(ticker, IB_option_date, ib_one_strike_below, DownOne_Call_Price, 1)
                IB.ibAPI.placeBuyBracketOrder(ticker, current_price)

                if ticker == "SPY":
                    print("sending tweet")
                    send_notifications.send_tweet_w_countdown_followup(ticker, current_price, 'up',
                                                                   f"${ticker} ${current_price}. 5 minutes to profit on a Call. #5A {formatted_time}", 300)

            except Exception as e:
                print(e)
            finally:
                pass


        else:
            print('No Buy_5A Signal')

    except KeyError as e1:
        print(Exception)
    try:
        Sell_5A = trained_models.Sell_5A(
            dailyminutes_df[['Bonsai Ratio', 'B1/B2', 'PCRv Up4', 'ITM PCRv Up4', 'ITM PCRv Down4',
       'ITM PCRoi Up4', 'ITM PCRoi Down4']])

        dailyminuteswithALGOresults_df['Sell_5A'] = Sell_5A
        if Sell_5A[-1]:
            print('Sell_5A  Signal.')
            send_notifications.email_me_string("Sell_5A", "Put",
                                               ticker)

            try:
                IB.ibAPI.placePutBracketOrder(ticker, IB_option_date, ib_one_strike_above, UpOne_Put_Price, 1)

                print("sending tweet")
                send_notifications.send_tweet_w_countdown_followup(ticker, current_price, 'down',
                                                               f"${ticker}  ${current_price}. 5 min till profit on a PUT. #5A {formatted_time}", 300)

            except Exception as e:
                print(e)
            finally:
                pass



        else:
            print('No 5A_Sell Signal.')

    except Exception as e1:
        print(Exception)
    try:
        Buy_A5 = trained_models.Buy_A5(
            dailyminutes_df[['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'PCRv Up4', 'ITM PCRv Up4',
       'ITM PCRoi Up4', 'ITM PCRoi Down4']])
        dailyminuteswithALGOresults_df['Buy_A5'] = Buy_A5
        print(Buy_A5)
        if Buy_A5[-1]:
            print("A5_Buy Signal")
            send_notifications.email_me_string("Buy_A5:", "Call",
                                               ticker)
            try:
                IB.ibAPI.placeCallBracketOrder(ticker, IB_option_date, ib_one_strike_below, DownOne_Call_Price, 1)
                IB.ibAPI.placeBuyBracketOrder(ticker, current_price)

                if ticker == "SPY":
                    print("sending tweet")
                    send_notifications.send_tweet_w_countdown_followup(ticker, current_price, 'up',
                                                                   f"${ticker} ${current_price}. 30 minutes to profit on a Call. #A5 {formatted_time}", 1800)

            except Exception as e:
                print(e)
            finally:
                pass


        else:
            print('No Buy_A5 Signal')

    except KeyError as e1:
        print(Exception)
    try:
        Sell_A5 = trained_models.Sell_A5(
            dailyminutes_df[['Bonsai Ratio', 'B1/B2', 'PCRv Up4', 'ITM PCRv Up4', 'ITM PCRoi Up4',
       'ITM PCRoi Down4']])

        dailyminuteswithALGOresults_df['Sell_A5'] = Sell_A5
        if Sell_A5[-1]:
            print('Sell_A5  Signal.')
            send_notifications.email_me_string("Sell_A5", "Put",
                                               ticker)

            try:
                IB.ibAPI.placeBuyBracketOrder(ticker, current_price)
                IB.ibAPI.placePutBracketOrder(ticker, IB_option_date, ib_one_strike_above, UpOne_Put_Price, 1)

                print("sending tweet")
                send_notifications.send_tweet_w_countdown_followup(ticker, current_price, 'down',
                                                               f"${ticker}  ${current_price}. 30 min till profit on a PUT. #A5 {formatted_time}", 1800)

            except Exception as e:
                print(e)
            finally:
                pass



        else:
            print('No A5_Sell Signal.')

    except Exception as e1:
        print(Exception)
    try:
        Buy_A4 = trained_models.Buy_A4(
            dailyminutes_df[['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'ITM PCRv Down4',
       'ITM PCRoi Up4', 'ITM PCRoi Down4']])
        dailyminuteswithALGOresults_df['Buy_A4'] = Buy_A4
        print(Buy_A4)
        if Buy_A4[-1]:
            print("A4_Buy Signal")
            send_notifications.email_me_string("Buy_A4:", "Call",
                                               ticker)
            try:
                # IB.ibAPI.placeCallBracketOrder(ticker, IB_option_date, ib_one_strike_below, DownOne_Call_Price, 1)
                # IB.ibAPI.placeBuyBracketOrder(ticker, current_price)

                if ticker == "SPY":
                    print("sending tweet")
                    send_notifications.send_tweet_w_countdown_followup(ticker, current_price, 'up',
                                                                   f"${ticker} has hit a temporal LOW at ${current_price}. 20 minutes to profit on a Call. #A4 {formatted_time}", 1200)

            except Exception as e:
                print(e)
            finally:
                pass


        else:
            print('No Buy_A4 Signal')

    except KeyError as e1:
        print(Exception)
    try:
        Sell_A4 = trained_models.Sell_A4(
            dailyminutes_df[['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'PCRv Up4', 'PCRv Down4',
       'ITM PCRv Up4', 'ITM PCRv Down4', 'ITM PCRoi Up4', 'ITM PCRoi Down4']])

        dailyminuteswithALGOresults_df['Sell_A4'] = Sell_A4
        if Sell_A4[-1]:
            print('A4 Sell Signal.')
            send_notifications.email_me_string("Sell_A4", "Put",
                                               ticker)

            try:
                IB.ibAPI.placeBuyBracketOrder(ticker, current_price)
                IB.ibAPI.placePutBracketOrder(ticker, IB_option_date, ib_one_strike_above, UpOne_Put_Price, 1)

                print("sending tweet")
                send_notifications.send_tweet_w_countdown_followup(ticker, current_price, 'down',
                                                               f"${ticker} has hit ${current_price}. 20 min and you'll make profit on a PUT. #A4 {formatted_time}", 1200)

            except Exception as e:
                print(e)
            finally:
                pass



        else:
            print('No A4_Sell Signal.')

    except Exception as e1:
        print(Exception)
    try:
        Buy_A3 = trained_models.Buy_A3(
            dailyminutes_df[['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'ITM PCRv Down4',
       'ITM PCRoi Up4', 'ITM PCRoi Down4']])
        dailyminuteswithALGOresults_df['Buy_A3'] = Buy_A3
        print(Buy_A3)
        # if Buy_A3[-1]:
        #     print("A3_Buy Signal")
        #     send_notifications.email_me_string("Buy_A3:", "Call",
        #                                        ticker)
        #     try:
        #         IB.ibAPI.placeCallBracketOrder(ticker, IB_option_date, ib_one_strike_below, DownOne_Call_Price, 1)
        #         IB.ibAPI.placeBuyBracketOrder(ticker, current_price)
        #
        #         if ticker == "SPY":
        #             print("sending tweet")
        #             send_notifications.send_tweet_w_countdown_followup(ticker, current_price, 'up',
        #                                                            f"${ticker} has hit a temporal LOW at ${current_price}. 45 minutes to profit on a Call. #A3")
        #
        #     except Exception as e:
        #         print(e)
        #     finally:
        #         pass
        #
        #
        # else:
        #     print('No Buy_A3 Signal')

    except KeyError as e1:
        print(Exception)
    try:
        Sell_A3 = trained_models.Sell_A3(
            dailyminutes_df[['Bonsai Ratio', 'B1/B2', 'PCRv Up4', 'ITM PCRv Up4', 'ITM PCRv Down4',
       'ITM PCRoi Up4', 'ITM PCRoi Down4']])

        dailyminuteswithALGOresults_df['Sell_A3'] = Sell_A3
        # if Sell_A3[-1]:
        #     print('A3 Sell Signal.')
        #     send_notifications.email_me_string("Sell_A3", "Put",
        #                                        ticker)
        #
        #     try:
        #         IB.ibAPI.placeBuyBracketOrder(ticker, current_price)
        #         IB.ibAPI.placePutBracketOrder(ticker, IB_option_date, ib_one_strike_above, UpOne_Put_Price, 1)
        #
        #         print("sending tweet")
        #         send_notifications.send_tweet_w_countdown_followup(ticker, current_price, 'down',
        #                                                        f"${ticker} has hit ${current_price}. 45 min and you'll make profit on a PUT. #A3")
        #
        #     except Exception as e:
        #         print(e)
        #     finally:
        #         pass



        # else:
        #     print('No A3_Sell Signal.')

    except Exception as e1:
        print(Exception)

    try:
        Buy_30min_9sallaround = trained_models.Buy_30min_9sallaround(dailyminutes_df[['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'PCRv Up4', 'PCRv Down4',
       'ITM PCRv Up4', 'ITM PCRv Down4', 'ITM PCRoi Up4', 'ITM PCRoi Down4',
       'RSI', 'AwesomeOsc', 'RSI14', 'RSI2', 'AwesomeOsc5_34']]   )
        dailyminuteswithALGOresults_df['Buy_30min_9sallaround'] = Buy_30min_9sallaround
        print(Buy_30min_9sallaround)
        # if Buy_30min_9sallaround[-1]:
        #     print("A1_Buy Signal")
        #     send_notifications.email_me_string("Buy_30min_9sallaround:", "Call",
        #                                        ticker)
        #     try:
        #         IB.ibAPI.placeCallBracketOrder(ticker, IB_option_date, ib_one_strike_below, DownOne_Call_Price, 1)
        #         IB.ibAPI.placeBuyBracketOrder(ticker, current_price)
        #
        #
        #         if ticker == "SPY":
        #             print("sending tweet")
        #             send_notifications.send_tweet_w_countdown_followup(ticker, current_price, 'up',
        #                                                        f"${ticker} has hit a temporal LOW at ${current_price}.You got 30 minutes or so to get some profit on a Call.")
        #
        #     except Exception as e:
        #         print(e)
        #     finally:
        #         pass
        #
        #
        # else:
        #     print('No Buy_30min_9sallaround Signal')

    except KeyError as e1:
        print(Exception)
    try:
        Sell_30min_9sallaround = trained_models.Sell_30min_9sallaround(
            dailyminutes_df[['Bonsai Ratio', 'B1/B2', 'PCRv Up4', 'ITM PCRv Up4', 'ITM PCRoi Up4',
       'ITM PCRoi Down4']])

        dailyminuteswithALGOresults_df['Sell_30min_9sallaround'] = Sell_30min_9sallaround
        # if Sell_30min_9sallaround[-1]:
        #     print('Sell_30min_9sallaround signal')
        #     send_notifications.email_me_string("Sell_30min_9sallaround", "Put",
        #                                        ticker)
        #
        #     try:
        #         IB.ibAPI.placeBuyBracketOrder(ticker, current_price)
        #         IB.ibAPI.placePutBracketOrder(ticker, IB_option_date, ib_one_strike_above, UpOne_Put_Price, 1)
        #
        #         print("sending tweet")
        #         send_notifications.send_tweet_w_countdown_followup(ticker, current_price, 'down',
        #                                                    f"${ticker} has hit a temporal HIGH at ${current_price}.You PROBS have like 30 min and you'll make profit on a PUT.")
        #
        #     except Exception as e:
        #         print(e)
        #     finally:
        #         pass
        #
        #
        #
        # else:
        #     print('No A1_Sell Signal.')

    except Exception as e1:
        print(Exception)
####TODO When A1_sell and trythisone2_4 buy line up, its a short term peak?!?!?!
    try:
        Trythisone2_4Buy = trained_models.Trythisone2_4Buy(dailyminutes_df[['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'PCR-Vol', 'PCR-OI', 'PCRv Up2', 'PCRv Up3', 'PCRv Up4', 'PCRv Down2', 'PCRv Down3',
       'PCRv Down4', 'PCRoi Up2', 'PCRoi Up3', 'PCRoi Up4', 'PCRoi Down2',
       'PCRoi Down3', 'PCRoi Down4', 'ITM PCR-Vol', 'ITM PCR-OI',
       'ITM PCRv Up1', 'ITM PCRv Up2', 'ITM PCRv Up3', 'ITM PCRv Up4',
       'ITM PCRv Down1', 'ITM PCRv Down2', 'ITM PCRv Down3', 'ITM PCRv Down4',
       'ITM PCRoi Up2', 'ITM PCRoi Up3', 'ITM PCRoi Up4', 'ITM PCRoi Down2',
       'ITM PCRoi Down3', 'ITM PCRoi Down4', 'Net_IV', 'Net ITM IV',
       'NIV 2Higher Strike', 'NIV 2Lower Strike', 'NIV 3Higher Strike',
       'NIV 3Lower Strike', 'NIV 4Higher Strike', 'NIV 4Lower Strike',
       'NIV highers(-)lowers1-4', 'NIV 1-2 % from mean', 'NIV 1-4 % from mean',
       'RSI', 'AwesomeOsc', 'RSI14', 'RSI2', 'AwesomeOsc5_34']]   )
        dailyminuteswithALGOresults_df['Trythisone2_4Buy'] = Trythisone2_4Buy

        print(Trythisone2_4Buy)
        # if Trythisone2_4Buy[-1]:
        #     print("Trythisone2_4Buy Signal")
        #     send_notifications.email_me_string("Trythisone2_4Buy:", "Call",
        #                                        ticker)
        #     try:
        #         IB.ibAPI.placeCallBracketOrder(ticker, IB_option_date, ib_one_strike_below, DownOne_Call_Price, 1)
        #         IB.ibAPI.placeBuyBracketOrder(ticker, current_price)
        #
        #
        #         # if ticker == "SPY":
        #         #     print("sending tweet")
        #         #     send_notifications.send_tweet_w_5hour_followup(ticker, current_price, 'up',
        #         #                                                f"${ticker} has hit a temporal LOW at ${current_price}.This is a signal that the price has a high chance of rising significantly in a 3-5 hour window.")
        #
        #     except Exception as e:
        #         print(e)
        #     finally:
        #         pass
        #
        #
        # else:
        #     print('No Trythisone2_4Buy Signal')

    except KeyError as e1:
        print(Exception)
        pass

    try:
        Trythisone2_4Sell = trained_models.Trythisone2_4Sell(
        dailyminutes_df[['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'PCR-Vol', 'PCRv Up2',
       'PCRv Down3', 'PCRv Down4', 'PCRoi Up3', 'PCRoi Up4', 'PCRoi Down3',
       'ITM PCR-Vol', 'ITM PCR-OI', 'ITM PCRv Up1', 'ITM PCRv Up2',
       'ITM PCRv Up3', 'ITM PCRv Up4', 'ITM PCRv Down2', 'ITM PCRv Down3',
       'ITM PCRoi Up2', 'ITM PCRoi Up3', 'ITM PCRoi Up4', 'ITM PCRoi Down3',
       'ITM PCRoi Down4', 'RSI', 'AwesomeOsc', 'RSI14', 'RSI2']])

        dailyminuteswithALGOresults_df['Trythisone2_4Sell'] = Trythisone2_4Sell
        # if Trythisone2_4Sell[-1]:
        #     print('Trythisone2_Sell signal')
        #     send_notifications.email_me_string("Trythisone2_4Sell", "Put",
        #                                        ticker)
        #
        #     try:
        #         IB.ibAPI.placeBuyBracketOrder(ticker, current_price)
        #         IB.ibAPI.placePutBracketOrder(ticker, IB_option_date, ib_one_strike_above, UpOne_Put_Price, 1)
        #
        #         # print("sending tweet")
        #         # send_notifications.send_tweet_w_5hour_followup(ticker, current_price, 'down',
        #         #                                            f"${ticker} has hit a temporal HIGH at ${current_price}.This is a signal that the price has a high likelihood of falling significantly in a 3-5 hour window.")
        #
        #     except Exception as e:
        #         print(e)
        #     finally:
        #         pass
        #
        #
        #
        # else:
        #     print('No Trythisone2_4Sell Signal.')
    finally:
        pass


    try:
        A1_Buy = trained_models.A1_Buy(dailyminutes_df[['Bonsai Ratio', 'Bonsai Ratio 2', 'PCRoi Up1', 'ITM PCRoi Up1']]   )
        dailyminutes_df['A1_Buy'] = A1_Buy
        dailyminuteswithALGOresults_df['A1_Buy'] = A1_Buy
        print(A1_Buy)
        # if A1_Buy[-1]:
        #     print("A1_Buy Signal")
        #     send_notifications.email_me_string("A1_Buy:", "Call",
        #                                        ticker)
        #     try:
        #         IB.ibAPI.placeCallBracketOrder(ticker, IB_option_date, ib_one_strike_below, DownOne_Call_Price, 1)
        #         IB.ibAPI.placeBuyBracketOrder(ticker, current_price)
        #
        #
        #         if ticker == "SPY":
        #             print("sending tweet")
        #             send_notifications.send_tweet_w_5hour_followup(ticker, current_price, 'up',
        #                                                        f"${ticker} has hit a temporal LOW at ${current_price}.This is a signal that the price has a high chance of rising significantly in a 3-5 hour window.")
        #
        #     except Exception as e:
        #         print(e)
        #     finally:
        #         pass
        #
        #
        # else:
        #     print('No A1_Buy Signal')

    except KeyError as e1:
        print(Exception)
    try:
        A1_Sell = trained_models.A1_Sell(
            dailyminutes_df[['Bonsai Ratio', 'Bonsai Ratio 2', 'PCRoi Up1', 'ITM PCRoi Up1', 'Net IV LAC']])

        dailyminutes_df['A1_Sell'] = A1_Sell
        dailyminuteswithALGOresults_df['A1_Sell'] = A1_Sell
        # if A1_Sell[-1]:
        #     print('A1_Sell signal')
        #     send_notifications.email_me_string("A1_Sell", "Put",
        #                                        ticker)
        #
        #     try:
        #         IB.ibAPI.placeBuyBracketOrder(ticker, current_price)
        #         IB.ibAPI.placePutBracketOrder(ticker, IB_option_date, ib_one_strike_above, UpOne_Put_Price, 1)
        #
        #         print("sending tweet")
        #         send_notifications.send_tweet_w_5hour_followup(ticker, current_price, 'down',
        #                                                    f"${ticker} has hit a temporal HIGH at ${current_price}.This is a signal that the price has a high likelihood of falling significantly in a 3-5 hour window.")
        #
        #     except Exception as e:
        #         print(e)
        #     finally:
        #         pass
        #
        #
        #
        # else:
        #     print('No A1_Sell Signal.')

    except Exception as e1:
        print(Exception)
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
        buy_signal1 = trained_models.get_buy_B1B2_Bonsai_Ratio_RSI_ITM_PCRVol_threshUp5_threshDown5_30_min_later_change_SPY(
            dailyminutes_df[["B1/B2", "Bonsai Ratio", "RSI", 'ITM PCR-Vol']])

        if buy_signal1[-1]:
            # if ticker=="SPY":
            # # send_notifications.send_tweet(ticker,current_price,'up',f"${ticker} has hit a temporal low at ${current_price}. 80% chance of going higher within 1 hr..")
            #
            #
            # else:
            #     pass
            send_notifications.email_me_string("buy_signal1[-1]:", "Call",
                                               ticker)
            # try:
            #     # IB.ibAPI.placeCallBracketOrder(ticker, IB_option_date, ib_one_strike_below, DownOne_Call_Price, 1)
            # except Exception as e:
            #     print(e)
            # finally:
            #     pass
            print('Buy signal!')

        else:
            print('No buy signal.')
    except Exception as e1:
        print(Exception)
    try:
        buy_signal2 = trained_models.get_buy_signal_NEWONE_PRECISE(
            dailyminutes_df[['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2',
                          'PCR-Vol', 'PCRv Up1', 'PCRv Up2',
                          'PCRv Up3', 'PCRv Up4', 'PCRv Down1',
                          'PCRv Down2', 'PCRv Down3', 'PCRv Down4',
                          'ITM PCR-Vol', 'ITM PCRv Up1',
                          'ITM PCRv Up2', 'ITM PCRv Up3',
                          'ITM PCRv Up4', 'ITM PCRv Down1',
                          'ITM PCRv Down2', 'ITM PCRv Down3',
                          'ITM PCRv Down4', 'ITM PCRoi Up2',
                          'ITM OI', 'ITM Contracts %', 'Net_IV',
                          'Net ITM IV', 'NIV 1Lower Strike',
                          'NIV 2Higher Strike', 'NIV 2Lower Strike',
                          'NIV 3Higher Strike', 'NIV 3Lower Strike',
                          'NIV 4Higher Strike', 'NIV 4Lower Strike',
                          'NIV highers(-)lowers1-4',
                          'NIV 1-4 % from mean', 'RSI',
                          'AwesomeOsc']])

        dailyminuteswithALGOresults_df['buy_signal2'] = buy_signal2

        if buy_signal2[-1]:
            send_notifications.email_me_string("buy_signal2[-1]:", "Call",
                                               ticker)
            # try:
            #     IB.ibAPI.placeCallBracketOrder(ticker, IB_option_date, ib_one_strike_below, DownOne_Call_Price,1)
            # except Exception as e:
            #     print(e)
            # finally:
            #     pass
            print('Buy signal!')
        else:
            print('No buy signal.')

    except Exception as e2:
        print(Exception)


    try:
        buy_signal3 = trained_models.get_buy_signal_NEWONE_TESTED_WELL_MOSTLY_UP(dailyminutes_df[
                                                                                     ['Bonsai Ratio', 'Bonsai Ratio 2',
                                                                                      'PCR-Vol', 'PCRv Down1',
                                                                                      'PCRv Down2',
                                                                                      'PCRv Down3', 'ITM PCRv Up3',
                                                                                      'ITM PCRv Up4', 'ITM PCRv Down2',
                                                                                      'ITM PCRv Down3', 'Net_IV',
                                                                                      'NIV 2Lower Strike',
                                                                                      'NIV 4Higher Strike',
                                                                                      'NIV highers(-)lowers1-4']
                                                                                 ])
        dailyminuteswithALGOresults_df['buy_signal3'] = buy_signal3
        if buy_signal3[-1]:
            send_notifications.email_me_string("buy_signal3[-1]:", "Call",
                                               ticker)
            # try:
            #     IB.ibAPI.placeCallBracketOrder(ticker, IB_option_date, ib_one_strike_below, DownOne_Call_Price, 1)
            # except Exception as e:
            #     print(e)
            # finally:
            #     pass
            print('Buy signal 3!')
        else:

            print('No buy signal 3.')


    except Exception as e3:
        print(Exception)
    try:
        new_buy_signal1 = trained_models.get_buy_signal_1to4hourNewGreatPrecNumbersBonsai1NETitmIV(dailyminutes_df[
                                                                                       ['Bonsai Ratio','Net ITM IV','RSI']   ])
        dailyminuteswithALGOresults_df['new_buy_signal1'] = new_buy_signal1




        print(new_buy_signal1)
        if new_buy_signal1[-1]:
            print("New Buy Signal 1")
            send_notifications.email_me_string("New Buy Signal 1[-1]:", "Call",
                                               ticker)
            # try:
            #     IB.ibAPI.placeBuyBracketOrder(ticker, current_price)
            #
            #     # IB.ibAPI.placeCallBracketOrder(ticker, IB_option_date, ib_one_strike_below, DownOne_Call_Price,1)
            # except Exception as e:
            #     print(e)
            # finally:
            #     pass


        else:
            print('No New Buy Signal 1.')

    except KeyError as e1:
        print(Exception)
    try:
        new_sell_signal1 = trained_models.get_sell_signal_1to4hourNewGreatPrecNumbersBonsai1NETitmIV(
            dailyminutes_df[["Bonsai Ratio", "Net ITM IV"]])
        print(new_sell_signal1)
        dailyminuteswithALGOresults_df['new_sell_signal1'] = new_sell_signal1
        if new_sell_signal1[-1]:
            print('New Sell signal 1!')
            # send_notifications.email_me_string("new_sell_signal 1[-1]:", "Put",
            #                                    ticker)
            #
            # try:
            #     IB.ibAPI.placeSellBracketOrder(ticker, current_price)
            #
            #     # IB.ibAPI.placePutBracketOrder(ticker, IB_option_date, ib_one_strike_above, UpOne_Put_Price,1)
            # except Exception as e:
            #     print(e)
            # finally:
            #     pass



        else:
            print('No New Sell Signal 1.')
    finally:
        pass

#     except Exception as e1:
#         print(Exception)
#     try:
#         new_buy_signal2 = trained_models.get_buy_signal_NewPerhapsExcellentTargetDown5to15minSPY(dailyminutes_df[
#                                                                                        ['Bonsai Ratio','Net ITM IV']   ])
#         dailyminutes_df['new_buy_signal2'] = new_buy_signal2
#
#         print(new_buy_signal2)
#         if new_buy_signal2[-1]:
#             print("New Buy Signal 2")
#             send_notifications.email_me_string("New Buy Signal 2[-1]:", "Call",
#                                                ticker)
#             try:
#                 IB.ibAPI.placeBuyBracketOrder(ticker, current_price)
#
#                 # IB.ibAPI.placeCallBracketOrder(ticker, IB_option_date, ib_one_strike_below, DownOne_Call_Price,1)
#             except Exception as e:
#                 print(e)
#             finally:
#                 pass
#
#
#         else:
#             print('No New Buy Signal 2.')
#
#     except KeyError as e1:
#         print(Exception)
    try:
        new_sell_signal2 = trained_models.get_sell_signal_NewPerhapsExcellentTargetDown5to15minSPY(
            dailyminutes_df[["Bonsai Ratio", "Net ITM IV",'RSI']])
        print(new_sell_signal2)
        dailyminuteswithALGOresults_df['new_sell_signal2'] = new_sell_signal2
        if new_sell_signal2[-1]:
            print('New Sell signal 2!')
            send_notifications.email_me_string("new_sell_signal 2[-1]:", "Put",
                                               ticker)

            # try:
            #     # IB.ibAPI.placeSellBracketOrder(ticker, current_price)
            #
            #     # IB.ibAPI.placePutBracketOrder(ticker, IB_option_date, ib_one_strike_above, UpOne_Put_Price,1)
            # except Exception as e:
            #     print(e)
            # finally:
            #     pass



        else:
            print('No New Sell Signal 2.')

    except Exception as e1:
        print(Exception)
#     # try:
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
    if dailyminutes_df['B1/B2'].iloc[-1] > 1.15 and dailyminutes_df['RSI'].iloc[-1]<30:
        send_notifications.email_me_string("dailyminutes_df['B1/B2'][0] > 1.15 and dailyminutes_df['RSI'][0]<30:", "Call",
                                           ticker)


        try:
            IB.ibAPI.placeBuyBracketOrder(ticker, current_price)
            IB.ibAPI.placeCallBracketOrder(ticker,IB_option_date,ib_one_strike_below, DownOne_Call_Price,19,'b1/b2 + rsi')

            print("sending tweet")
            # send_notifications.send_tweet_w_countdown_followup(ticker, current_price, 'down',
            #                                                    f"${ticker} has hit ${current_price}. 20 min and you'll make profit on a PUT. #A4 {formatted_time}",
            #                                                    1200)

        except Exception as e:
            print(e)
        finally:
            pass



####THis one is good for a very short term peak before drop.  Maybe tighter profit/loss
    if dailyminutes_df['B1/B2'].iloc[-1] < 0.25 and dailyminutes_df["RSI"].iloc[-1]>70:
        send_notifications.email_me_string("dailyminutes_df['B1/B2'][-1] < 0.25 and dailyminutes_df['RSI'][-1]>77:", "Put",
                                           ticker)
        try:
            IB.ibAPI.placePutBracketOrder(ticker,IB_option_date,ib_one_strike_above, UpOne_Put_Price,19,"b1/b2+rsi")

        except Exception as e:
            print(e)
        finally:
            pass
        # if ticker == "SPY":
        #     send_notifications.send_tweet_w_countdown_followup(ticker, current_price, 'down', f"${ticker} has hit a peak at ${current_price}. Short term downwards movement expected.")

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
    dailyminutes_df.to_csv(dailyminutes,index=False)
    dailyminuteswithALGOresults_df.to_csv(dailyminuteswithALGOresults,index=False)
# Define functions for each prediction model
# def make_sell_5C_predictions(data):
#     return trained_models.Sell_5C(data)
#
# def make_buy_5B_predictions(data):
#     return trained_models.Buy_5B(data)
#
# def make_sell_5B_predictions(data):
#     return trained_models.Sell_5B(data)
#
# # Define a function for sending email notification
# def send_email_notification(signal_type, option_type, ticker):
#     send_notifications.email_me_string(signal_type, option_type, ticker)
#
# # Define a function for placing bracket orders
# def place_bracket_order(ticker, option_date, strike_price, price, quantity):
#     try:
#         if option_type == 'put':
#             IB.ibAPI.placePutBracketOrder(ticker, option_date, strike_price, price, quantity)
#         else:
#             IB.ibAPI.placeCallBracketOrder(ticker, option_date, strike_price, price, quantity)
#     except Exception as e:
#         print(e)
#
# # Define a function for sending tweets
# def send_tweet_with_countdown(ticker, current_price, direction, message, countdown):
#     try:
#         send_notifications.send_tweet_w_countdown_followup(ticker, current_price, direction, message, countdown)
#     except Exception as e:
#         print(e)
#
# # Process each prediction model
# def process_prediction_model(model_name, column_names, option_date, strike_price_above, strike_price_below, option_type):
#     try:
#         predictions = model_name(dailyminutes_df[column_names])
#         dailyminuteswithALGOresults_df[model_name.__name__] = predictions
#         if predictions[-1]:
#             print(f'{model_name.__name__} Signal')
#             send_email_notification(model_name.__name__, option_type, ticker)
#             place_bracket_order(ticker, option_date, strike_price_above, UpOne_Put_Price, 5)
#             if ticker == "SPY":
#                 formatted_time = datetime.now().strftime("%y%m%d %H:%M")
#                 print("sending tweet")
#                 send_tweet_with_countdown(ticker, current_price, 'up',
#                                           f"${ticker} ${current_price}. 5 minutes to profit on a Call. #{model_name.__name__} {formatted_time}", 300)
#         else:
#             print(f'No {model_name.__name__} Signal')
#     except Exception as e:
#         print(e)
#
# # Main code
# try:
#     process_prediction_model(make_sell_5C_predictions, ['Bonsai Ratio', 'PCRv Up4', 'ITM PCRv Up4', 'ITM PCRoi Up4'],
#                              IB_option_date, ib_one_strike_above, None, 'put')
#
#     process_prediction_model(make_buy_5B_predictions,
#                              ['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'PCRv Up4', 'PCRv Down4', 'ITM PCRv Up4',
#                               'ITM PCRv Down4', 'ITM PCRoi Up4', 'ITM PCRoi Down4'],
#                              IB_option_date, None, ib_one_strike_below, 'call')
#
#     process_prediction_model(make_sell_5B_predictions, ['Bonsai Ratio', 'B1/B2', 'PCRv Up4', 'ITM PCRoi Up4',
#                                                        'ITM PCRoi Down4'],
#                              IB_option_date, ib_one_strike_above, None, 'put')
#
# except KeyError as e:
#     print(e)
# except Exception as e:
#     print(e)
