
import joblib

#
# percent_up=.1
# percent_down=-.1

def get_buy_B1B2_Bonsai_Ratio_RSI_ITM_PCRVol_threshUp7_threshDown7_30_min_later_change_TSLA(new_data_df):
    model_filename = 'Strategy_Testing/Trained_Models/B1B2_Bonsai_Ratio_RSI_ITM_PCRVol_threshUp7_threshDown7_30_min_later_change_TSLA/target_up.joblib'
    loaded_model = joblib.load(model_filename)

    predictions = loaded_model.predict(new_data_df)

    return predictions

def get_sell_B1B2_Bonsai_Ratio_RSI_ITM_PCRVol_threshUp7_threshDown7_30_min_later_change_TSLA(new_data_df):
    model_filename = 'Strategy_Testing/Trained_Models/B1B2_Bonsai_Ratio_RSI_ITM_PCRVol_threshUp7_threshDown7_30_min_later_change_TSLA/target_down.joblib'
    loaded_model = joblib.load(model_filename)

    predictions = loaded_model.predict(new_data_df)

    return predictions



# percent_up=.05
# percent_down=-.05
def get_buy_B1B2_Bonsai_Ratio_RSI_ITM_PCRVol_threshUp5_threshDown5_30_min_later_change_SPY(new_data_df):
    model_filename = 'Strategy_Testing/Trained_Models/B1B2_Bonsai_Ratio_RSI_ITM_PCRVol_threshUp5_threshDown5_30_min_later_change_SPY/target_up.joblib'
    loaded_model = joblib.load(model_filename)

    predictions = loaded_model.predict(new_data_df)

    return predictions

def get_sell_B1B2_Bonsai_Ratio_RSI_ITM_PCRVol_threshUp5_threshDown5_30_min_later_change_SPY(new_data_df):
    model_filename = 'Strategy_Testing/Trained_Models/B1B2_Bonsai_Ratio_RSI_ITM_PCRVol_threshUp5_threshDown5_30_min_later_change_SPY/target_down.joblib'
    loaded_model = joblib.load(model_filename)

    predictions = loaded_model.predict(new_data_df)

    return predictions


def get_buy_signal_B1B2_RSI_1hr_threshUp7(new_data_df):
    model_filename = 'Strategy_Testing/Trained_Models/1_hour_later_change_B1-B2_RSI_threshUp0.7_threshDown0.7/target_up.joblib'
    loaded_model = joblib.load(model_filename)

    predictions = loaded_model.predict(new_data_df)

    return predictions

def get_sell_signal_B1B2_RSI_1hr_threshDown7(new_data_df):
    model_filename = 'Strategy_Testing/Trained_Models/1_hour_later_change_B1-B2_RSI_threshUp0.7_threshDown0.7/target_down.joblib'
    loaded_model = joblib.load(model_filename)

    predictions = loaded_model.predict(new_data_df)

    return predictions


def get_buy_signal_BonsaiRatio_ITMPCRV_30min(new_data_df):

    model_filename = 'Strategy_Testing/Trained_Models/BonsaiRatio_ITMPCRV_30min/trained_model_target_up.joblib'
    loaded_model = joblib.load(model_filename)


    predictions = loaded_model.predict(new_data_df)

    return predictions



def get_sell_signal_BonsaiRatio_ITMPCRV_30min(new_data_df):
    model_filename = 'Strategy_Testing/Trained_Models/BonsaiRatio_ITMPCRV_30min/trained_model_target_down.joblib'
    loaded_model = joblib.load(model_filename)


    predictions = loaded_model.predict(new_data_df)

    return predictions

def get_buy_signal_NEWONE_PRECISE(new_data_df):
    model_filename = 'Strategy_Testing/Trained_Models/looks like itmight work well for everything`PRECISE/target_up.joblib'
    loaded_model = joblib.load(model_filename)

    predictions = loaded_model.predict(new_data_df)

    return predictions

def get_sell_signal_NEWONE_PRECISE(new_data_df):
    model_filename = 'Strategy_Testing/Trained_Models/looks like itmight work well for everything`PRECISE/target_down.joblib'
    loaded_model = joblib.load(model_filename)


    predictions = loaded_model.predict(new_data_df)

    return predictions

def get_buy_signal_NEWONE_TESTED_WELL_MOSTLY_UP(new_data_df):
    model_filename = 'Strategy_Testing/Trained_Models/TESTED_WELL_SPYandTSLA_MOSTLY_UP_tho/target_up.joblib'
    loaded_model = joblib.load(model_filename)

    predictions = loaded_model.predict(new_data_df)

    return predictions

def get_sell_signal_NEWONE_TESTED_WELL_MOSTLY_UP(new_data_df):
    model_filename = 'Strategy_Testing/Trained_Models/TESTED_WELL_SPYandTSLA_MOSTLY_UP_tho/target_down.joblib'
    loaded_model = joblib.load(model_filename)


    predictions = loaded_model.predict(new_data_df)

    return predictions
# In the get_buy_signal() function, it loads the model (trained_model_target_up.joblib) specifically trained for the "buy" signal (target_up). It accepts predictor inputs as a list (predictors) and assumes you will provide the corresponding values for the predictors. It then creates a DataFrame (new_data_df) with the new data and makes predictions using the loaded model. The predictions are returned as the buy signal.
#
# Similarly, the get_sell_signal() function loads the model (trained_model_target_down.joblib) specifically trained for the "sell" signal (target_down). It follows the same process as the get_buy_signal() function to make predictions based on the provided predictor inputs and returns the sell signal.
#
# Note: Make sure you have trained and saved the models separately for each target before using these functions, and replace <value> with the actual values for the predictors you want to use.
#





