import logging

# Configure the logging settings
logging.basicConfig(filename='error.log', level=logging.ERROR)
from datetime import datetime,timedelta

import numpy as np
from Strategy_Testing import trained_models
import pandas as pd
import IB.ibAPI
import send_notifications as send_notifications
logging.basicConfig(filename='trade_algos_error.log', level=logging.ERROR,    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')



def actions(optionchain, dailyminutes, closest_strike_currentprice, strikeindex_abovebelow, closest_exp_date, ticker, current_price):
    print(dailyminutes)
    ###strikeindex_abovebelow is a list [lowest,3 lower,2 lower, 1 lower, 1 higher,2 higher,3 higher, 4 higher]
    import pandas as pd
    optionchain = pd.read_csv(optionchain)
    dailyminutes = pd.read_csv(dailyminutes).dropna()

    try:
        dailyminutes["B1% Change"] = ((dailyminutes["Bonsai Ratio"].astype(float) - dailyminutes["Bonsai Ratio"].astype(float).shift(1)) / dailyminutes["Bonsai Ratio"].astype(float).shift(1)) * 100
        dailyminutes["B2% Change"] = ((dailyminutes["Bonsai Ratio 2"].astype(float) - dailyminutes["Bonsai Ratio 2"].astype(float).shift(1)) / dailyminutes["Bonsai Ratio 2"].astype(float).shift(1)) * 100
    finally:
        pass
    print(ticker, current_price)
    date_string = closest_exp_date
    date_object = datetime.strptime(date_string, "%Y-%m-%d")
    new_date_string = date_object.strftime("%y%m%d")
    IB_option_date = date_object.strftime("%Y%m%d")

####Different strikes converted to contract form.
    ##This one is the strike one above Closest to Current price strike

    print(strikeindex_abovebelow)
    if strikeindex_abovebelow[4] != np.nan:
        ib_one_strike_above = strikeindex_abovebelow[4]
        one_strike_above_closest_cp_strike_int_num = int(strikeindex_abovebelow[4] * 1000)

    if strikeindex_abovebelow[3] != np.nan:
        ib_one_strike_below = strikeindex_abovebelow[3]
        one_strike_below_closest_cp_strike_int_num = int(strikeindex_abovebelow[3] * 1000)
    closest_strike_int_num = int(dailyminutes["Closest Strike to CP"].iloc[-1] * 1000)  # Convert decimal to integer
    ###TODO add different exp date options in addition to diff strike optoins.

    one_strike_above_CCPS ="{:08d}".format(one_strike_above_closest_cp_strike_int_num)
    one_strike_below_CCPS ="{:08d}".format(one_strike_below_closest_cp_strike_int_num)
    closest_contract_strike = "{:08d}".format(closest_strike_int_num)

    CCP_upone_call_contract = f"{ticker}{new_date_string}C{one_strike_above_CCPS}"
    CCP_upone_put_contract = f"{ticker}{new_date_string}P{one_strike_above_CCPS}"
    CCP_downone_call_contract = f"{ticker}{new_date_string}C{one_strike_below_CCPS}"
    CCP_downone_put_contract = f"{ticker}{new_date_string}P{one_strike_below_CCPS}"
    CCP_call_contract = f"{ticker}{new_date_string}C{closest_contract_strike}"
    CCP_put_contract = f"{ticker}{new_date_string}P{closest_contract_strike}"
    ###TODO for puts, use a higher strike, for calls use a lower strike.  ATM strikes get down to pennies EOD.

    CCP_Put_Price = optionchain.loc[optionchain['p_contractSymbol'] == CCP_put_contract]['Put_LastPrice'].values[0]
    CCP_Call_Price = optionchain.loc[optionchain['c_contractSymbol'] == CCP_call_contract]['Call_LastPrice'].values[0]

    DownOne_Call_Price =optionchain.loc[optionchain['c_contractSymbol'] == CCP_downone_call_contract]['Call_LastPrice'].values[0]
    UpOne_Put_Price=optionchain.loc[optionchain['p_contractSymbol'] == CCP_upone_put_contract]['Put_LastPrice'].values[0]
    #             # IB.ibAPI.placeSellBracketOrder(ticker, current_price, "SELL")
    #         IB.ibAPI.placeBuyBracketOrder(ticker,current_price,"BUY")


    """
    Algo Rules go here:
    """


    try:

        buy_signal1 = trained_models.get_buy_B1B2_Bonsai_Ratio_RSI_ITM_PCRVol_threshUp5_threshDown5_30_min_later_change_SPY(
            dailyminutes[["B1/B2", "Bonsai Ratio", "RSI", 'ITM PCR-Vol']])

        if buy_signal1[-1]:

            send_notifications.email_me_string("buy_signal1[-1]:", "Call",
                                               ticker)
            try:
                IB.ibAPI.placeCallBracketOrder(ticker, IB_option_date, ib_one_strike_below, DownOne_Call_Price, 1)
            except Exception as e:
                print(e)
            finally:
                pass
            print('Buy signal!')

        else:
            print('No buy signal.')
    except Exception as e1:
        print(Exception)
    try:
        buy_signal2 = trained_models.get_buy_signal_NEWONE_PRECISE(
            dailyminutes[['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2',
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


        if buy_signal2[-1]:
            send_notifications.email_me_string("buy_signal2[-1]:", "Call",
                                               ticker)
            try:
                IB.ibAPI.placeCallBracketOrder(ticker, IB_option_date, ib_one_strike_below, DownOne_Call_Price,1)
            except Exception as e:
                print(e)
            finally:
                pass
            print('Buy signal!')
        else:
            print('No buy signal.')

    except Exception as e2:
        print(Exception)


    try:
        buy_signal3 = trained_models.get_buy_signal_NEWONE_TESTED_WELL_MOSTLY_UP(dailyminutes[
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
        if buy_signal3[-1]:
            send_notifications.email_me_string("buy_signal3[-1]:", "Call",
                                               ticker)
            try:
                IB.ibAPI.placeCallBracketOrder(ticker, IB_option_date, ib_one_strike_below, DownOne_Call_Price, 1)
            except Exception as e:
                print(e)
            finally:
                pass
            print('Buy signal!')
        else:

            print('No buy signal.')


    except Exception as e3:
        print(Exception)
    try:
        sell_signal1 = trained_models.get_sell_B1B2_Bonsai_Ratio_RSI_ITM_PCRVol_threshUp5_threshDown5_30_min_later_change_SPY(
            dailyminutes[["B1/B2", "Bonsai Ratio", "RSI", 'ITM PCR-Vol']])

        if sell_signal1[-1]:
            send_notifications.email_me_string("sell_signal1[-1]:", "Put",
                                               ticker)
            try:

                IB.ibAPI.placePutBracketOrder(ticker, IB_option_date, ib_one_strike_above, UpOne_Put_Price,1)
            except Exception as e:
                print(e)
            finally:
                pass

            print('Sell signal!')

        else:
            print('No sell signal.')

    except Exception as e1:
        print(Exception)

    try:
        sell_signal2 = trained_models.get_sell_signal_NEWONE_PRECISE(dailyminutes[
                                                                         ['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2',
                                                                          'PCR-Vol', 'PCRv Up1', 'PCRv Up2', 'PCRv Up3',
                                                                          'PCRv Up4', 'PCRv Down1', 'PCRv Down2',
                                                                          'PCRv Down3', 'PCRv Down4', 'PCRoi Up1',
                                                                          'PCRoi Down1', 'PCRoi Down2', 'PCRoi Down3',
                                                                          'PCRoi Down4', 'ITM PCR-Vol', 'ITM PCRv Up1',
                                                                          'ITM PCRv Up2', 'ITM PCRv Up3',
                                                                          'ITM PCRv Up4',
                                                                          'ITM PCRv Down1', 'ITM PCRv Down2',
                                                                          'ITM PCRv Down3', 'ITM PCRv Down4',
                                                                          'ITM PCRoi Up1', 'ITM PCRoi Up3',
                                                                          'ITM PCRoi Down4', 'ITM OI',
                                                                          'ITM Contracts %',
                                                                          'Net_IV', 'Net ITM IV', 'NIV 1Higher Strike',
                                                                          'NIV 1Lower Strike', 'NIV 2Higher Strike',
                                                                          'NIV 2Lower Strike', 'NIV 3Higher Strike',
                                                                          'NIV 3Lower Strike', 'NIV 4Higher Strike',
                                                                          'NIV 4Lower Strike',
                                                                          'NIV highers(-)lowers1-2',
                                                                          'NIV highers(-)lowers1-4',
                                                                          'NIV 1-2 % from mean',
                                                                          'NIV 1-4 % from mean', 'RSI', 'AwesomeOsc']
                                                                     ])
        if sell_signal2[-1]:
            send_notifications.email_me_string("sell_signal2[-1]:", "Put",
                                               ticker)
            try:
                IB.ibAPI.placePutBracketOrder(ticker, IB_option_date, ib_one_strike_above, UpOne_Put_Price, 1)
            except Exception as e:
                print(e)
            finally:
                pass
            print('Sell signal!')
        else:
            print('No sell signal.')


    except Exception as e1:
        print(Exception)

    try:
        sell_signal3 = trained_models.get_sell_signal_NEWONE_TESTED_WELL_MOSTLY_UP(dailyminutes[
                                                                                       ['Bonsai Ratio',
                                                                                        'Bonsai Ratio 2',
                                                                                        'B1/B2', 'PCR-Vol', 'PCRv Up1',
                                                                                        'PCRv Up2', 'PCRv Up3',
                                                                                        'PCRv Up4',
                                                                                        'PCRv Down1', 'PCRv Down2',
                                                                                        'PCRv Down3', 'PCRv Down4',
                                                                                        'PCRoi Up1', 'PCRoi Up2',
                                                                                        'PCRoi Up3', 'PCRoi Up4',
                                                                                        'PCRoi Down3', 'PCRoi Down4',
                                                                                        'ITM PCR-Vol', 'ITM PCR-OI',
                                                                                        'ITM PCRv Up1', 'ITM PCRv Up2',
                                                                                        'ITM PCRv Up3', 'ITM PCRv Up4',
                                                                                        'ITM PCRv Down1',
                                                                                        'ITM PCRv Down2',
                                                                                        'ITM PCRv Down3',
                                                                                        'ITM PCRv Down4',
                                                                                        'ITM PCRoi Up1',
                                                                                        'ITM PCRoi Up2',
                                                                                        'ITM PCRoi Up3',
                                                                                        'ITM PCRoi Up4',
                                                                                        'ITM PCRoi Down1',
                                                                                        'ITM PCRoi Down2',
                                                                                        'ITM PCRoi Down4', 'ITM OI',
                                                                                        'Total OI', 'ITM Contracts %',
                                                                                        'Net_IV', 'Net ITM IV',
                                                                                        'NIV 1Higher Strike',
                                                                                        'NIV 1Lower Strike',
                                                                                        'NIV 2Higher Strike',
                                                                                        'NIV 2Lower Strike',
                                                                                        'NIV 3Higher Strike',
                                                                                        'NIV 3Lower Strike',
                                                                                        'NIV 4Higher Strike',
                                                                                        'NIV 4Lower Strike',
                                                                                        'NIV highers(-)lowers1-2',
                                                                                        'NIV highers(-)lowers1-4',
                                                                                        'NIV 1-2 % from mean',
                                                                                        'NIV 1-4 % from mean', 'RSI']
                                                                                   ])
        print(sell_signal3)
        if sell_signal3[-1]:
            print("sell signal 333333333333333333333333333333")
            send_notifications.email_me_string("sell_signal3[-1]:", "Put",
                                               ticker)
            try:
                IB.ibAPI.placePutBracketOrder(ticker, IB_option_date, ib_one_strike_above, UpOne_Put_Price,1)
            except Exception as e:
                print(e)
            finally:
                pass

            print('Sell signal!')
        else:
            print('No sell signal 3.')

    except KeyError as e1:
        print(Exception)


    if dailyminutes['B2/B1'].iloc[-1] >500 and dailyminutes['Bonsai Ratio'].iloc[-1]<.0001 and dailyminutes['ITM PCRv Up2'].iloc[-1]<.01 and dailyminutes['ITM PCRv Down2'].iloc[-1]<5 and dailyminutes['NIV 1-2 % from mean'].iloc[-1]>dailyminutes['NIV 1-4 % from mean'].iloc[-1]>0:
        send_notifications.email_me_string("['B2/B1'][-1] >500 and dailyminutes['Bonsai Ratio'][0]<.0001 and dailyminutes['ITM PCRv Up2'][0]<.01 and dailyminutes['ITM PCRv Down2'][0]<5 and dailyminutes['NIV 1-2 % from mean'][0]>dailyminutes['NIV 1-4 % from mean'][0]>0:", "Put",
                                           ticker)
        try:
            IB.ibAPI.placePutBracketOrder(ticker,IB_option_date,ib_one_strike_above, UpOne_Put_Price,10)
        except Exception as e:
            print(e)
        finally:
            pass
        if ticker == "SPY":
            send_notifications.send_tweet(ticker,current_price,'down',f"${ticker} looks ripe for a short term drop at ${current_price}. [STdownf1]")
# 1.15-(hold until) 0 and <0.0, hold call until .3   (hold them until the b1/b2 doubles/halves?) with conditions to make sure its profitable.

    if dailyminutes['B1/B2'].iloc[-1] > 1.15 and dailyminutes['RSI'].iloc[-1]<30:
        send_notifications.email_me_string("dailyminutes['B1/B2'][0] > 1.15 and dailyminutes['RSI'][0]<30:", "Call",
                                           ticker)
        try:
            IB.ibAPI.placeCallBracketOrder(ticker,IB_option_date,ib_one_strike_below, DownOne_Call_Price,1)
        except Exception as e:
            print(e)
        finally:
            pass
        # TradierAPI.buy(x)

        #
        #         IB.ibAPI.placeCallBracketOrder(ticker,IB_option_date,ib_one_strike_below, DownOne_Call_Price,1), response.status_code, json_response)

        if ticker=="SPY":
            send_notifications.send_tweet(ticker,current_price,'up',f"${ticker} has hit a trough at ${current_price}. Short term upwards movement expected.")
####THis one is good for a very short term peak before drop.  Maybe tighter profit/loss
    if dailyminutes['B1/B2'].iloc[-1] < 0.25 and dailyminutes["RSI"].iloc[-1]>70:
        send_notifications.email_me_string("dailyminutes['B1/B2'][-1] < 0.25 and dailyminutes['RSI'][-1]>70:", "Put",
                                           ticker)
        try:
            IB.ibAPI.placePutBracketOrder(ticker,IB_option_date,ib_one_strike_above, UpOne_Put_Price,1)

        except Exception as e:
            print(e)
        finally:
            pass
        if ticker == "SPY":
            send_notifications.send_tweet(ticker,current_price,'down',f"${ticker} has hit a peak at ${current_price}. Short term downwards movement expected.")

#### TESTING JUST RSI
    if dailyminutes['RSI'].iloc[-1] < 30:
        send_notifications.email_me_string("['RSI'][-1] < 30:", "Call",
                                           ticker)
        try:
            IB.ibAPI.placeCallBracketOrder(ticker,IB_option_date,ib_one_strike_below, DownOne_Call_Price,1)
        except Exception as e:
            print(e)
        finally:
            pass
        if ticker == "SPY":
            send_notifications.send_tweet(ticker, current_price, 'up',
                                          f"${ticker} has hit a trough at ${current_price}. Short term upwards movement expected.")
        ####THis one is good for a very short term peak before drop.  Maybe tighter profit/loss
    if dailyminutes['RSI'].iloc[-1] > 70:
        send_notifications.email_me_string("['RSI'][-1] > 70:", "Put",
                                           ticker)
        try:
            IB.ibAPI.placePutBracketOrder(ticker,IB_option_date,ib_one_strike_above, UpOne_Put_Price,1)
        except Exception as e:
            print(e)
        finally:
            pass

###JUST b1/b2
    if dailyminutes['B1/B2'].iloc[-1] > 1.15 :
        send_notifications.email_me_string("['B1/B2''][-1] > 1.15:", "Call",
                                           ticker)
        try:
            IB.ibAPI.placeCallBracketOrder(ticker,IB_option_date,ib_one_strike_below, DownOne_Call_Price,1)
        except Exception as e:
            print(e)
        finally:
            pass

    ####THis one is good for a very short term peak before drop.  Maybe tighter profit/loss
    if dailyminutes['B1/B2'].iloc[-1] < 0.25 :
        send_notifications.email_me_string("['B1/B2'][-1] < 0.25:", "Put",
                                               ticker)
        try:
            IB.ibAPI.placePutBracketOrder(ticker,IB_option_date,ib_one_strike_above, UpOne_Put_Price,1)
        except Exception as e:
            print(e)
        finally:
            pass


# predictor_values = {'Bonsai Ratio': .0007, 'ITM PCR-Vol': 20}
# predictor_df = pd.DataFrame(predictor_values, index=[0])











    #
    # except (ValueError, ConnectionError) as e:
    #     logging.error('Error: %s', e)
    #
    #     pass

    # send_notifications.email_me_string(order, response.status_code, json_response)
    # send_notifications.email_me_string(order,response.status_code,json_response)













    # buy_signal2 = trained_models.get_buy_signal_B1B2_RSI_1hr_threshUp7(dailyminutes[["B1/B2", "RSI"]].head(1))
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
    # sell_signal2 = trained_models.get_sell_signal_B1B2_RSI_1hr_threshDown7(dailyminutes[["B1/B2", "RSI"]].head(1))
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
    # buy_signal3 = trained_models.get_buy_B1B2_Bonsai_Ratio_RSI_ITM_PCRVol_threshUp7_threshDown7_30_min_later_change_TSLA(dailyminutes[["B1/B2","Bonsai Ratio","RSI",'ITM PCR-Vol']].head(1))
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
    # sell_signal3 = trained_models.get_sell_B1B2_Bonsai_Ratio_RSI_ITM_PCRVol_threshUp7_threshDown7_30_min_later_change_TSLA(dailyminutes[["B1/B2","Bonsai Ratio","RSI",'ITM PCR-Vol']].head(1))
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
    # if dailyminutes["B1/B2"][0] > 1.15 :
    #     x = (f'b1/b2>1.15  {ticker}',
    #         f"{optionchain.loc[optionchain['c_contractSymbol'] == call_contract]['Call_LastPrice'].values[0]}",
    #         "10",
    #         call_contract, .6, 1.6
    #     )
    #
    #     TradierAPI.buy(x)
    #
    # if dailyminutes["B1/B2"][0] < 0.01:
    #     x = (f'b1/b2<.01  {ticker}',
    #         f"{optionchain.loc[optionchain['p_contractSymbol'] == put_contract]['Put_LastPrice'].values[0]}",
    #         "10",
    #         put_contract, .6, 1.6
    #     )
    #
    #     TradierAPI.buy(x)

    # if dailyminutes["NIV 1-2 % from mean"][0] < -100 and dailyminutes["NIV 1-4 % from mean"][0] <-200:
    #     x = ("NIV 1-2 % from mean< -100 & NIV 1-4 % from mean<-200",
    #         f"{optionchain.loc[optionchain['p_contractSymbol'] == put_contract]['Put_LastPrice'].values[0]  }",
    #         "8",
    #         put_contract, .6,1.6
    #     )
    #
    #     TradierAPI.buy(x)
    # if dailyminutes["NIV 1-2 % from mean"][0] > 100 and dailyminutes["NIV 1-4 % from mean"][0] > 200:
    #     x = ('NIV 1-2 % from mean> 100 & NIV 1-4 % from mean > 200',
    #         f"{optionchain.loc[optionchain['c_contractSymbol'] == call_contract]['Call_LastPrice'].values[0]}",
    #         "7",
    #         call_contract, .6, 1.6
    #     )
    #
    #     TradierAPI.buy(x)
    # if dailyminutes["NIV highers(-)lowers1-4"][0] < -20:
    #     x = ("NIV highers(-)lowers1-4 < -20",
    #         f"{optionchain.loc[optionchain['c_contractSymbol'] == call_contract]['Call_LastPrice'].values[0]  }",
    #         "6",
    #         call_contract, .6,1.6
    #     )
    #
    #     TradierAPI.buy(x)
    # if dailyminutes["NIV highers(-)lowers1-4"][0] > 20:
    #     x = ('NIV highers(-)lowers1-4> 20'
    #         f"{optionchain.loc[optionchain['p_contractSymbol'] == put_contract]['Put_LastPrice'].values[0]}",
    #         "5",
    #         put_contract, .6, 1.6
    #     )

        # TradierAPI.buy(x)


    # if dailyminutes["ITM PCR-Vol"][0] > 1.3 and dailyminutes["RSI"][0] > 70:
    #     x = (
    #         f"{optionchain.loc[optionchain['p_contractSymbol'] == put_contract]['Put_LastPrice'].values[0] }",
    #         "99",
    #         put_contract,.9,1.05
    #     )
    #
    #     TradierAPI.buy(x)

    #     ###MADE 100% on this near close 5/12
    # if dailyminutes["Bonsai Ratio"][0] < .8 and dailyminutes["ITM PCR-Vol"][0] < 0.8 and dailyminutes["RSI"][0] < 30:
    #     x = ('Bonsai Ratio < .8 and ITM PCR-Vol < 0.8',
    #         f"{optionchain.loc[optionchain['c_contractSymbol'] == call_contract]['Call_LastPrice'].values[0]}",
    #         "3",
    #         call_contract,.8,1.2
    #     )
    #     print(x)
    #     TradierAPI.buy(x)
    #
    #
    # if dailyminutes["Bonsai Ratio"][0] > 1.5  and dailyminutes["ITM PCR-Vol"][0] > 1.2 and dailyminutes["RSI"][0] > 70:
    #     x = ('Bonsai Ratio > 1.5  andITM PCR-Vol > 1.2',
    #         f"{optionchain.loc[optionchain['p_contractSymbol'] == put_contract]['Put_LastPrice'].values[0] }",
    #         "2",
    #         put_contract,.9,1.05
    #     )
    #
    #     TradierAPI.buy(x)
    # if (
    #     dailyminutes["Bonsai Ratio"][0] < 0.7
    #     and dailyminutes["Net_IV"][0] < -50
    #     and dailyminutes["Net ITM IV"][0] > -41
    # ):
    #     x = ('Bonsai Ratio < 0.7 and Net_IV < -50 and Net ITM IV> -41',
    #         f"{optionchain.loc[optionchain['c_contractSymbol'] == call_contract]['Call_LastPrice'].values[0]}",
    #         "1",
    #         call_contract,.9,1.05
    #     )

        # TradierAPI.buy(x)


