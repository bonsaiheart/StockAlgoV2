import os

import joblib
from pathlib import Path
base_dir = os.path.dirname(__file__)
# base_dir = Path(__file__)

# percent_up=.1

# percent_down=-.1
###TODO could make features = modle.info "features"
###supposed to be for 30 min .3 spy tsla
def Buy_15min_A2(new_data_df):
    features = ['Bonsai Ratio', 'B1/B2', 'PCRv Up4', 'PCRv Down4', 'ITM PCRv Up4', 'ITM PCRv Down4', 'RSI14', 'AwesomeOsc5_34', 'RSI', 'AwesomeOsc']
    model_filename = f'{base_dir}/Trained_Models/_15min_A2/target_up.joblib'
    loaded_model = joblib.load(model_filename)
    predictions = loaded_model.predict(new_data_df[features])
    return predictions

def Sell_15min_A2(new_data_df):
    features = ['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'PCRv Up4', 'PCRv Down4', 'ITM PCRv Up4', 'ITM PCRv Down4', 'RSI14', 'AwesomeOsc5_34', 'RSI']
    model_filename = 'Strategy_Testing/Trained_Models/_15min_A2/target_down.joblib'
    loaded_model = joblib.load(model_filename)
    predictions = loaded_model.predict(new_data_df[features])
    return predictions

def Buy_15min_A1(new_data_df):
    features = ['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'PCRv Up4', 'PCRv Down4', 'ITM PCRv Up4', 'ITM PCRv Down4', 'AwesomeOsc5_34', 'RSI', 'RSI2']
    model_filename = f'{base_dir}/Trained_Models/_15min_A1/target_up.joblib'
    loaded_model = joblib.load(model_filename)
    predictions = loaded_model.predict(new_data_df[features])
    return predictions

def Sell_15min_A1(new_data_df):
    features = ['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'PCRv Up4', 'PCRv Down4', 'ITM PCRv Up4', 'ITM PCRv Down4', 'RSI14', 'AwesomeOsc5_34', 'RSI2']
    model_filename = 'Strategy_Testing/Trained_Models/_15min_A1/target_down.joblib'
    loaded_model = joblib.load(model_filename)
    predictions = loaded_model.predict(new_data_df[features])
    return predictions

def Buy_5D(new_data_df):
    features = ['ITM PCRv Up4', 'ITM PCRv Down4', 'ITM PCRoi Down4', 'RSI14']
    model_filename = f'{base_dir}/Trained_Models/5D/target_up.joblib'
    loaded_model = joblib.load(model_filename)
    predictions = loaded_model.predict(new_data_df[features])
    return predictions

def Sell_5D(new_data_df):
    features = ['Bonsai Ratio', 'PCRv Up4', 'ITM PCRv Up4', 'ITM PCRoi Up4']
    model_filename = 'Strategy_Testing/Trained_Models/5D/target_down.joblib'
    loaded_model = joblib.load(model_filename)
    predictions = loaded_model.predict(new_data_df[features])
    return predictions

def Buy_5C(new_data_df):
    features = ['Bonsai Ratio', 'B1/B2', 'PCRv Up4', 'PCRv Down4', 'ITM PCRv Up4',
       'ITM PCRv Down4', 'ITM PCRoi Up4', 'ITM PCRoi Down4']
    model_filename = f'{base_dir}/Trained_Models/5C_5min_spy/target_up.joblib'
    loaded_model = joblib.load(model_filename)
    predictions = loaded_model.predict(new_data_df[features])
    return predictions

def Sell_5C(new_data_df):
    features =['Bonsai Ratio', 'PCRv Up4', 'ITM PCRv Up4', 'ITM PCRoi Up4']
    model_filename = 'Strategy_Testing/Trained_Models/5C_5min_spy/target_down.joblib'
    loaded_model = joblib.load(model_filename)

    predictions = loaded_model.predict(new_data_df[features])

    return predictions
def Buy_5B(new_data_df):
    features =['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'PCRv Up4', 'PCRv Down4',
       'ITM PCRv Up4', 'ITM PCRv Down4', 'ITM PCRoi Up4', 'ITM PCRoi Down4']

    model_filename = f'{base_dir}/Trained_Models/5B_5min_spy/target_up.joblib'
    loaded_model = joblib.load(model_filename)

    predictions = loaded_model.predict(new_data_df[features])

    return predictions

def Sell_5B(new_data_df):
    features =['Bonsai Ratio', 'B1/B2', 'PCRv Up4', 'ITM PCRoi Up4',
       'ITM PCRoi Down4']

    model_filename = 'Strategy_Testing/Trained_Models/5B_5min_spy/target_down.joblib'
    loaded_model = joblib.load(model_filename)

    predictions = loaded_model.predict(new_data_df[features])

    return predictions
def Buy_5A(new_data_df):
    features =['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'PCRv Up4', 'PCRv Down4',
       'ITM PCRv Up4', 'ITM PCRv Down4', 'ITM PCRoi Up4', 'ITM PCRoi Down4']

    model_filename = f'{base_dir}/Trained_Models/5A_5min_spy/target_up.joblib'
    loaded_model = joblib.load(model_filename)

    predictions = loaded_model.predict(new_data_df[features])

    return predictions

def Sell_5A(new_data_df):
    features =['Bonsai Ratio', 'B1/B2', 'PCRv Up4', 'ITM PCRv Up4', 'ITM PCRv Down4',
       'ITM PCRoi Up4', 'ITM PCRoi Down4']

    model_filename = 'Strategy_Testing/Trained_Models/5A_5min_spy/target_down.joblib'
    loaded_model = joblib.load(model_filename)

    predictions = loaded_model.predict(new_data_df[features])

    return predictions
def Buy_A5(new_data_df):
    features =['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'PCRv Up4', 'ITM PCRv Up4',
       'ITM PCRoi Up4', 'ITM PCRoi Down4']

    model_filename = f'{base_dir}/Trained_Models/A5_30_min_spy_tsla/target_up.joblib'
    loaded_model = joblib.load(model_filename)

    predictions = loaded_model.predict(new_data_df[features])

    return predictions

def Sell_A5(new_data_df):
    features =['Bonsai Ratio', 'B1/B2', 'PCRv Up4', 'ITM PCRv Up4', 'ITM PCRoi Up4',
       'ITM PCRoi Down4']

    model_filename = 'Strategy_Testing/Trained_Models/A5_30_min_spy_tsla/target_down.joblib'
    loaded_model = joblib.load(model_filename)

    predictions = loaded_model.predict(new_data_df[features])

    return predictions
def Buy_A4(new_data_df):
    features =['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'ITM PCRv Down4',
       'ITM PCRoi Up4', 'ITM PCRoi Down4']

    model_filename = f'{base_dir}/Trained_Models/A4_20min_02percent/target_up.joblib'
    loaded_model = joblib.load(model_filename)

    predictions = loaded_model.predict(new_data_df[features])

    return predictions

def Sell_A4(new_data_df):
    features =['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'PCRv Up4', 'PCRv Down4',
       'ITM PCRv Up4', 'ITM PCRv Down4', 'ITM PCRoi Up4', 'ITM PCRoi Down4']

    model_filename = 'Strategy_Testing/Trained_Models/A4_20min_02percent/target_down.joblib'
    loaded_model = joblib.load(model_filename)

    predictions = loaded_model.predict(new_data_df[features])

    return predictions
def Buy_A3(new_data_df):

    features =['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'ITM PCRv Down4',
       'ITM PCRoi Up4', 'ITM PCRoi Down4']
    model_filename = 'Strategy_Testing/Trained_Models/A3_Looks_Best_45min/target_up.joblib'
    loaded_model = joblib.load(model_filename)

    predictions = loaded_model.predict(new_data_df[features])

    return predictions

def Sell_A3(new_data_df):
    features =['Bonsai Ratio', 'B1/B2', 'PCRv Up4', 'ITM PCRv Up4', 'ITM PCRv Down4',
       'ITM PCRoi Up4', 'ITM PCRoi Down4']
    model_filename = 'Strategy_Testing/Trained_Models/A3_Looks_Best_45min/target_down.joblib'
    loaded_model = joblib.load(model_filename)

    predictions = loaded_model.predict(new_data_df[features])

    return predictions
def Buy_30min_9sallaround(new_data_df):
    features =['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'PCRv Up4', 'PCRv Down4',
       'ITM PCRv Up4', 'ITM PCRv Down4', 'ITM PCRoi Up4', 'ITM PCRoi Down4',
       'RSI', 'AwesomeOsc', 'RSI14', 'RSI2', 'AwesomeOsc5_34']
    model_filename = 'Strategy_Testing/Trained_Models/30MIN9sallAround/target_up.joblib'
    loaded_model = joblib.load(model_filename)

    predictions = loaded_model.predict(new_data_df[features])

    return predictions

def Sell_30min_9sallaround(new_data_df):
    features =['Bonsai Ratio', 'B1/B2', 'PCRv Up4', 'ITM PCRv Up4', 'ITM PCRoi Up4',
       'ITM PCRoi Down4']
    model_filename = 'Strategy_Testing/Trained_Models/30MIN9sallAround/target_down.joblib'
    loaded_model = joblib.load(model_filename)

    predictions = loaded_model.predict(new_data_df[features])

    return predictions
def Trythisone2_4Buy(new_data_df):
    features =['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'PCR-Vol', 'PCR-OI', 'PCRv Up2', 'PCRv Up3', 'PCRv Up4', 'PCRv Down2', 'PCRv Down3',
       'PCRv Down4', 'PCRoi Up2', 'PCRoi Up3', 'PCRoi Up4', 'PCRoi Down2',
       'PCRoi Down3', 'PCRoi Down4', 'ITM PCR-Vol', 'ITM PCR-OI',
       'ITM PCRv Up1', 'ITM PCRv Up2', 'ITM PCRv Up3', 'ITM PCRv Up4',
       'ITM PCRv Down1', 'ITM PCRv Down2', 'ITM PCRv Down3', 'ITM PCRv Down4',
       'ITM PCRoi Up2', 'ITM PCRoi Up3', 'ITM PCRoi Up4', 'ITM PCRoi Down2',
       'ITM PCRoi Down3', 'ITM PCRoi Down4', 'Net_IV', 'Net ITM IV',
       'NIV 2Higher Strike', 'NIV 2Lower Strike', 'NIV 3Higher Strike',
       'NIV 3Lower Strike', 'NIV 4Higher Strike', 'NIV 4Lower Strike',
       'NIV highers(-)lowers1-4', 'NIV 1-2 % from mean', 'NIV 1-4 % from mean',
       'RSI', 'AwesomeOsc', 'RSI14', 'RSI2', 'AwesomeOsc5_34']
    model_filename = 'Strategy_Testing/Trained_Models/trythismodelfor2_4hours/target_up.joblib'
    loaded_model = joblib.load(model_filename)

    predictions = loaded_model.predict(new_data_df[features])

    return predictions
def Trythisone2_4Sell(new_data_df):
    features =['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'PCR-Vol', 'PCRv Up2',
       'PCRv Down3', 'PCRv Down4', 'PCRoi Up3', 'PCRoi Up4', 'PCRoi Down3',
       'ITM PCR-Vol', 'ITM PCR-OI', 'ITM PCRv Up1', 'ITM PCRv Up2',
       'ITM PCRv Up3', 'ITM PCRv Up4', 'ITM PCRv Down2', 'ITM PCRv Down3',
       'ITM PCRoi Up2', 'ITM PCRoi Up3', 'ITM PCRoi Up4', 'ITM PCRoi Down3',
       'ITM PCRoi Down4', 'RSI', 'AwesomeOsc', 'RSI14', 'RSI2']
    model_filename = 'Strategy_Testing/Trained_Models/trythismodelfor2_4hours/target_down.joblib'
    loaded_model = joblib.load(model_filename)

    predictions = loaded_model.predict(new_data_df[features])

    return predictions
def A1_Buy(new_data_df):
    features =['Bonsai Ratio', 'Bonsai Ratio 2', 'PCRoi Up1', 'ITM PCRoi Up1']
    model_filename = 'Strategy_Testing/Trained_Models/A1_3_5hour_b1_b2_pcroiup1_itmpcroiup1_nivlac/target_up.joblib'
    loaded_model = joblib.load(model_filename)

    predictions = loaded_model.predict(new_data_df[features])

    return predictions


def A1_Sell(new_data_df):
    features =['Bonsai Ratio', 'Bonsai Ratio 2', 'PCRoi Up1', 'ITM PCRoi Up1', 'Net IV LAC']
    model_filename = 'Strategy_Testing/Trained_Models/A1_3_5hour_b1_b2_pcroiup1_itmpcroiup1_nivlac/target_down.joblib'
    loaded_model = joblib.load(model_filename)

    predictions = loaded_model.predict(new_data_df[features])

    return predictions

###3-5 hours, was good on spy
def A2_Buy(new_data_df):
    features =['Bonsai Ratio', 'PCRoi Up1', 'ITM PCRoi Up1']
    model_filename = 'Strategy_Testing/Trained_Models/SPY3_5_RSI14_Awesome5_34etc/target_up.joblib'
    loaded_model = joblib.load(model_filename)

    predictions = loaded_model.predict(new_data_df[features])

    return predictions

###3-5 hours, was good on spy
def A2_Sell(new_data_df):
    features =['Bonsai Ratio', 'Bonsai Ratio 2', 'PCRoi Up1', 'ITM PCRoi Up1', 'Net IV LAC']
    model_filename = 'Strategy_Testing/Trained_Models/SPY3_5_RSI14_Awesome5_34etc/target_down.joblib'
    loaded_model = joblib.load(model_filename)

    predictions = loaded_model.predict(new_data_df[features])

    return predictions
def get_buy_signal_NewPerhapsExcellentTargetDown5to15minSPY(new_data_df):
    features =['Bonsai Ratio','Net ITM IV']
    model_filename = 'Strategy_Testing/Trained_Models/NewGreatPrecNumbersBonsai1NETitmIV/target_up.joblib'
    loaded_model = joblib.load(model_filename)

    predictions = loaded_model.predict(new_data_df[features])

    return predictions


def get_sell_signal_NewPerhapsExcellentTargetDown5to15minSPY(new_data_df):
    features =["Bonsai Ratio", "Net ITM IV",'RSI']
    model_filename = 'Strategy_Testing/Trained_Models/NewPerhapsExcellentTargetDown5to15minSPY/target_down.joblib'
    loaded_model = joblib.load(model_filename)

    predictions = loaded_model.predict(new_data_df[features])

    return predictions
def get_buy_signal_1to4hourNewGreatPrecNumbersBonsai1NETitmIV(new_data_df):
    features = ['Bonsai Ratio','Net ITM IV','RSI']
    model_filename = 'Strategy_Testing/Trained_Models/NewPerhapsExcellentTargetDown5to15minSPY/target_up.joblib'
    loaded_model = joblib.load(model_filename)

    predictions = loaded_model.predict(new_data_df[features])

    return predictions

def get_sell_signal_1to4hourNewGreatPrecNumbersBonsai1NETitmIV(new_data_df):
    features =["Bonsai Ratio", "Net ITM IV"]
    model_filename = 'Strategy_Testing/Trained_Models/NewGreatPrecNumbersBonsai1NETitmIV/target_down.joblib'
    loaded_model = joblib.load(model_filename)

    predictions = loaded_model.predict(new_data_df[features])

    return predictions

def get_buy_B1B2_Bonsai_Ratio_RSI_ITM_PCRVol_threshUp7_threshDown7_30_min_later_change_TSLA(new_data_df):
    features =["B1/B2","Bonsai Ratio","RSI",'ITM PCR-Vol']
    model_filename = 'Strategy_Testing/Trained_Models/B1B2_Bonsai_Ratio_RSI_ITM_PCRVol_threshUp7_threshDown7_30_min_later_change_TSLA/target_up.joblib'
    loaded_model = joblib.load(model_filename)

    predictions = loaded_model.predict(new_data_df[features])

    return predictions

def get_sell_B1B2_Bonsai_Ratio_RSI_ITM_PCRVol_threshUp7_threshDown7_30_min_later_change_TSLA(new_data_df):
    features =["B1/B2","Bonsai Ratio","RSI",'ITM PCR-Vol']
    model_filename = 'Strategy_Testing/Trained_Models/B1B2_Bonsai_Ratio_RSI_ITM_PCRVol_threshUp7_threshDown7_30_min_later_change_TSLA/target_down.joblib'
    loaded_model = joblib.load(model_filename)

    predictions = loaded_model.predict(new_data_df[features])

    return predictions
def A1_Sell_historical_prediction(new_data_df):
    features =['B1/B2', 'ITM PCRoi Down2']
    model_filename = f'{base_dir}/Trained_Models/DAILYHISTORICALOVERNIGHTPREDICTION/target_down.joblib'
    loaded_model = joblib.load(model_filename)

    predictions = loaded_model.predict(new_data_df[features])

    return predictions
def A1_Buy_historical_prediction(new_data_df):
    features =['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'ITM PCR-Vol',
       'ITM PCRv Up2', 'ITM PCRv Down2', 'ITM PCRoi Up2']
    model_filename = f'{base_dir}/Trained_Models/DAILYHISTORICALOVERNIGHTPREDICTION/target_up.joblib'
    loaded_model = joblib.load(model_filename)

    predictions = loaded_model.predict(new_data_df[features])

    return predictions
# percent_up=.05
# percent_down=-.05
def get_buy_B1B2_Bonsai_Ratio_RSI_ITM_PCRVol_threshUp5_threshDown5_30_min_later_change_SPY(new_data_df):
    features =["B1/B2", "Bonsai Ratio", "RSI", 'ITM PCR-Vol']
    model_filename = 'Strategy_Testing/Trained_Models/B1B2_Bonsai_Ratio_RSI_ITM_PCRVol_threshUp5_threshDown5_30_min_later_change_SPY/target_up.joblib'
    loaded_model = joblib.load(model_filename)

    predictions = loaded_model.predict(new_data_df[features])

    return predictions

# def get_sell_B1B2_Bonsai_Ratio_RSI_ITM_PCRVol_threshUp5_threshDown5_30_min_later_change_SPY(new_data_df):
#     features =
#     model_filename = 'Strategy_Testing/Trained_Models/B1B2_Bonsai_Ratio_RSI_ITM_PCRVol_threshUp5_threshDown5_30_min_later_change_SPY/target_down.joblib'
#     loaded_model = joblib.load(model_filename)
#
#     predictions = loaded_model.predict(new_data_df[features])
#
#     return predictions


def get_buy_signal_B1B2_RSI_1hr_threshUp7(new_data_df):
    features =["B1/B2", "RSI"]
    model_filename = 'Strategy_Testing/Trained_Models/1_hour_later_change_B1-B2_RSI_threshUp0.7_threshDown0.7/target_up.joblib'
    loaded_model = joblib.load(model_filename)

    predictions = loaded_model.predict(new_data_df[features])

    return predictions

def get_sell_signal_B1B2_RSI_1hr_threshDown7(new_data_df):
    features =["B1/B2", "RSI"]
    model_filename = 'Strategy_Testing/Trained_Models/1_hour_later_change_B1-B2_RSI_threshUp0.7_threshDown0.7/target_down.joblib'
    loaded_model = joblib.load(model_filename)

    predictions = loaded_model.predict(new_data_df[features])

    return predictions


# def get_buy_signal_BonsaiRatio_ITMPCRV_30min(new_data_df):
#     features =
#
#     model_filename = 'Strategy_Testing/Trained_Models/BonsaiRatio_ITMPCRV_30min/trained_model_target_up.joblib'
#     loaded_model = joblib.load(model_filename)
#
#
#     predictions = loaded_model.predict(new_data_df[features])
#
#     return predictions




def get_buy_signal_NEWONE_PRECISE(new_data_df):
    features =['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2',
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
                          'AwesomeOsc']
    model_filename = 'Strategy_Testing/Trained_Models/looks like itmight work well for everything`PRECISE/target_up.joblib'
    loaded_model = joblib.load(model_filename)

    predictions = loaded_model.predict(new_data_df[features])

    return predictions

def get_sell_signal_NEWONE_PRECISE(new_data_df):
    features =['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2',
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
    model_filename = 'Strategy_Testing/Trained_Models/looks like itmight work well for everything`PRECISE/target_down.joblib'
    loaded_model = joblib.load(model_filename)


    predictions = loaded_model.predict(new_data_df[features])

    return predictions

def get_buy_signal_NEWONE_TESTED_WELL_MOSTLY_UP(new_data_df):
    features = ['Bonsai Ratio', 'Bonsai Ratio 2','PCR-Vol', 'PCRv Down1','PCRv Down2','PCRv Down3', 'ITM PCRv Up3',
    'ITM PCRv Up4', 'ITM PCRv Down2',
    'ITM PCRv Down3', 'Net_IV',
    'NIV 2Lower Strike',
    'NIV 4Higher Strike',
    'NIV highers(-)lowers1-4']

    model_filename = 'Strategy_Testing/Trained_Models/TESTED_WELL_SPYandTSLA_MOSTLY_UP_tho/target_up.joblib'
    loaded_model = joblib.load(model_filename)

    predictions = loaded_model.predict(new_data_df[features])

    return predictions

def get_sell_signal_NEWONE_TESTED_WELL_MOSTLY_UP(new_data_df):
    features =['Bonsai Ratio',
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
    model_filename = 'Strategy_Testing/Trained_Models/TESTED_WELL_SPYandTSLA_MOSTLY_UP_tho/target_down.joblib'
    loaded_model = joblib.load(model_filename)


    predictions = loaded_model.predict(new_data_df[features])

    return predictions
# In the get_buy_signal() function, it loads the model (trained_model_target_up.joblib) specifically trained for the "buy" signal (target_up). It accepts predictor inputs as a list (predictors) and assumes you will provide the corresponding values for the predictors. It then creates a DataFrame (new_data_df) with the new data and makes predictions using the loaded model. The predictions are returned as the buy signal.
#
# Similarly, the get_sell_signal() function loads the model (trained_model_target_down.joblib) specifically trained for the "sell" signal (target_down). It follows the same process as the get_buy_signal() function to make predictions based on the provided predictor inputs and returns the sell signal.
#
# Note: Make sure you have trained and saved the models separately for each target before using these functions, and replace <value> with the actual values for the predictors you want to use.
#





