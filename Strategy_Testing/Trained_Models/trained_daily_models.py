import os
import joblib
import numpy as np
import pandas as pd

base_dir = os.path.dirname(__file__)


def A1_Sell_historical_prediction(new_data_df):
    new_data_df = new_data_df[~new_data_df.duplicated()]

    features = ["B1/B2", "ITM PCRoi Down2"]
    model_filename = f"{base_dir}/DAILYHISTORICALOVERNIGHTPREDICTION/target_down.joblib"
    loaded_model = joblib.load(model_filename)
    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(
        subset=features, inplace=True
    )  # Drop rows with missing values in specified features
    threshold = 1e10
    tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
    predictions = loaded_model.predict(tempdf[features])
    print("sellpreds:", predictions)
    # Create a new Series with the predictions and align it with the original DataFrame
    prediction_series = pd.Series(predictions, index=tempdf.index)
    result = new_data_df.copy()  # Create a copy of the original DataFrame
    result[
        "Predictions"
    ] = np.nan  # Initialize the 'Predictions' column with NaN values
    result.loc[
        prediction_series.index, "Predictions"
    ] = prediction_series.values  # Assign predictions to corresponding rows
    return result["Predictions"]


def A1_Buy_historical_prediction(new_data_df):
    features = [
        "Bonsai Ratio",
        "Bonsai Ratio 2",
        "B1/B2",
        "ITM PCR-Vol",
        "ITM PCRv Up2",
        "ITM PCRv Down2",
        "ITM PCRoi Up2",
    ]
    new_data_df = new_data_df[~new_data_df.duplicated()]

    model_filename = f"{base_dir}/DAILYHISTORICALOVERNIGHTPREDICTION/target_up.joblib"
    loaded_model = joblib.load(model_filename)
    print(new_data_df.index)
    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
    print(tempdf.index)
    print(tempdf)
    tempdf.dropna(
        subset=features, inplace=True
    )  # Drop rows with missing values in specified features
    threshold = 1e10
    tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
    predictions = loaded_model.predict(tempdf[features])
    print("buypreds:", predictions)

    # Create a new Series with the predictions and align it with the original DataFrame
    prediction_series = pd.Series(predictions, index=tempdf.index)
    result = new_data_df  # Create a copy of the original DataFrame
    result[
        "Predictions"
    ] = np.nan  # Initialize the 'Predictions' column with NaN values
    print(prediction_series)

    result.loc[
        prediction_series.index, "Predictions"
    ] = prediction_series.values  # Assign predictions to corresponding rows
    return result["Predictions"]
