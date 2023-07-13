import os

import joblib
from pathlib import Path

import numpy as np
import pandas as pd

base_dir = os.path.dirname(__file__)
# base_dir = Path(__file__)

# percent_up=.1


# percent_down=-.1
###TODO could make features = modle.info "features"
###supposed to be for 30 min .3 spy tsla
def Buy_2hr_A1(new_data_df):
    features = ['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'PCRv Up4', 'PCRv Down4',
       'ITM PCRv Up4', 'ITM PCRv Down4', 'RSI14', 'AwesomeOsc5_34',
       'AwesomeOsc']
    model_filename = f"{base_dir}/_2hr_A1/target_up.joblib"
    loaded_model = joblib.load(model_filename)
    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    threshold = 1e10
    tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
    predictions = loaded_model.predict(tempdf[features])
    # Create a new Series with the predictions and align it with the original DataFrame
    prediction_series = pd.Series(predictions, index=tempdf.index)
    result = new_data_df.copy()  # Create a copy of the original DataFrame
    result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
    result.loc[
        prediction_series.index, "Predictions"
    ] = prediction_series.values  # Assign predictions to corresponding rows
    return result["Predictions"]


def Sell_2hr_A1(new_data_df):
    features = ['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'PCRv Up4', 'PCRv Down4',
       'ITM PCRv Up4', 'ITM PCRv Down4', 'AwesomeOsc5_34', 'RSI2',
       'AwesomeOsc']
    model_filename = f"{base_dir}/_2hr_A1/target_down.joblib"
    loaded_model = joblib.load(model_filename)
    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    threshold = 1e10
    tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
    predictions = loaded_model.predict(tempdf[features])
    # Create a new Series with the predictions and align it with the original DataFrame
    prediction_series = pd.Series(predictions, index=tempdf.index)
    result = new_data_df.copy()  # Create a copy of the original DataFrame
    result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
    result.loc[
        prediction_series.index, "Predictions"
    ] = prediction_series.values  # Assign predictions to corresponding rows
    return result["Predictions"]
def Buy_1hr_A3(new_data_df):
    features = ['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'PCRv Up4', 'PCRv Down4', 'ITM PCRv Up4', 'ITM PCRv Down4', 'RSI14', 'AwesomeOsc5_34', 'RSI']
    model_filename = f"{base_dir}/_1hr_A3/target_up.joblib"
    loaded_model = joblib.load(model_filename)
    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    threshold = 1e10
    tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
    predictions = loaded_model.predict(tempdf[features])
    # Create a new Series with the predictions and align it with the original DataFrame
    prediction_series = pd.Series(predictions, index=tempdf.index)
    result = new_data_df.copy()  # Create a copy of the original DataFrame
    result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
    result.loc[
        prediction_series.index, "Predictions"
    ] = prediction_series.values  # Assign predictions to corresponding rows
    return result["Predictions"]


def Sell_1hr_A3(new_data_df):
    features = ['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'PCRv Up4', 'PCRv Down4', 'ITM PCRv Up4', 'ITM PCRv Down4', 'RSI14', 'AwesomeOsc5_34', 'RSI2']
    model_filename = f"{base_dir}/_1hr_A3/target_down.joblib"
    loaded_model = joblib.load(model_filename)
    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    threshold = 1e10
    tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
    predictions = loaded_model.predict(tempdf[features])
    # Create a new Series with the predictions and align it with the original DataFrame
    prediction_series = pd.Series(predictions, index=tempdf.index)
    result = new_data_df.copy()  # Create a copy of the original DataFrame
    result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
    result.loc[
        prediction_series.index, "Predictions"
    ] = prediction_series.values  # Assign predictions to corresponding rows
    return result["Predictions"]
def Buy_1hr_A2(new_data_df):
    features = ['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'PCRv Up4', 'PCRv Down4', 'ITM PCRv Up4', 'ITM PCRv Down4', 'RSI14', 'AwesomeOsc5_34', 'RSI']
    model_filename = f"{base_dir}/_1hr_A2/target_up.joblib"
    loaded_model = joblib.load(model_filename)
    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    threshold = 1e10
    tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
    predictions = loaded_model.predict(tempdf[features])
    # Create a new Series with the predictions and align it with the original DataFrame
    prediction_series = pd.Series(predictions, index=tempdf.index)
    result = new_data_df.copy()  # Create a copy of the original DataFrame
    result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
    result.loc[
        prediction_series.index, "Predictions"
    ] = prediction_series.values  # Assign predictions to corresponding rows
    return result["Predictions"]


def Sell_1hr_A2(new_data_df):
    features = ['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'PCRv Up4', 'PCRv Down4', 'ITM PCRv Up4', 'ITM PCRv Down4', 'RSI14', 'AwesomeOsc5_34', 'RSI2']
    model_filename = f"{base_dir}/_1hr_A2/target_down.joblib"
    loaded_model = joblib.load(model_filename)
    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    threshold = 1e10
    tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
    predictions = loaded_model.predict(tempdf[features])
    # Create a new Series with the predictions and align it with the original DataFrame
    prediction_series = pd.Series(predictions, index=tempdf.index)
    result = new_data_df.copy()  # Create a copy of the original DataFrame
    result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
    result.loc[
        prediction_series.index, "Predictions"
    ] = prediction_series.values  # Assign predictions to corresponding rows
    return result["Predictions"]

def Buy_1hr_A1(new_data_df):
    features = [
        "Bonsai Ratio",
        "Bonsai Ratio 2",
        "B1/B2",
        "PCRv Up4",
        "PCRv Down4",
        "ITM PCRv Up4",
        "ITM PCRv Down4",
        "AwesomeOsc5_34",
        "RSI",
        "RSI2",
    ]
    model_filename = f"{base_dir}/_1hr_A1/target_up.joblib"
    loaded_model = joblib.load(model_filename)
    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    threshold = 1e10
    tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
    predictions = loaded_model.predict(tempdf[features])
    # Create a new Series with the predictions and align it with the original DataFrame
    prediction_series = pd.Series(predictions, index=tempdf.index)
    result = new_data_df.copy()  # Create a copy of the original DataFrame
    result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
    result.loc[
        prediction_series.index, "Predictions"
    ] = prediction_series.values  # Assign predictions to corresponding rows
    return result["Predictions"]


def Sell_1hr_A1(new_data_df):
    features = [
        "Bonsai Ratio",
        "Bonsai Ratio 2",
        "B1/B2",
        "PCRv Up4",
        "PCRv Down4",
        "ITM PCRv Up4",
        "ITM PCRv Down4",
        "RSI14",
        "AwesomeOsc5_34",
        "RSI2",
    ]
    model_filename = f"{base_dir}/_1hr_A1/target_down.joblib"
    loaded_model = joblib.load(model_filename)
    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    threshold = 1e10
    tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
    predictions = loaded_model.predict(tempdf[features])
    # Create a new Series with the predictions and align it with the original DataFrame
    prediction_series = pd.Series(predictions, index=tempdf.index)
    result = new_data_df.copy()  # Create a copy of the original DataFrame
    result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
    result.loc[
        prediction_series.index, "Predictions"
    ] = prediction_series.values  # Assign predictions to corresponding rows
    return result["Predictions"]
def Buy_45min_A1(new_data_df):
    features = ['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'PCRv Up4', 'PCRv Down4', 'ITM PCRv Up4', 'ITM PCRv Down4', 'RSI14', 'AwesomeOsc5_34', 'RSI']
    model_filename = f"{base_dir}/_45min_A1/target_up.joblib"
    loaded_model = joblib.load(model_filename)
    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    threshold = 1e10
    tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
    predictions = loaded_model.predict(tempdf[features])
    # Create a new Series with the predictions and align it with the original DataFrame
    prediction_series = pd.Series(predictions, index=tempdf.index)
    result = new_data_df.copy()  # Create a copy of the original DataFrame
    result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
    result.loc[
        prediction_series.index, "Predictions"
    ] = prediction_series.values  # Assign predictions to corresponding rows
    return result["Predictions"]


def Sell_45min_A1(new_data_df):
    features = ['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'PCRv Up4', 'PCRv Down4', 'ITM PCRv Up4', 'ITM PCRv Down4', 'RSI14', 'AwesomeOsc5_34', 'RSI2']
    model_filename = f"{base_dir}/_45min_A1/target_down.joblib"
    loaded_model = joblib.load(model_filename)
    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    threshold = 1e10
    tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
    predictions = loaded_model.predict(tempdf[features])
    # Create a new Series with the predictions and align it with the original DataFrame
    prediction_series = pd.Series(predictions, index=tempdf.index)
    result = new_data_df.copy()  # Create a copy of the original DataFrame
    result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
    result.loc[
        prediction_series.index, "Predictions"
    ] = prediction_series.values  # Assign predictions to corresponding rows
    return result["Predictions"]
def Buy_30min_A1(new_data_df):
    features = ['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'PCRv Up4', 'PCRv Down4', 'ITM PCRv Up4', 'ITM PCRv Down4', 'RSI14', 'AwesomeOsc5_34', 'RSI']
    model_filename = f"{base_dir}/_30min_A1/target_up.joblib"
    loaded_model = joblib.load(model_filename)
    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    threshold = 1e10
    tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
    predictions = loaded_model.predict(tempdf[features])
    # Create a new Series with the predictions and align it with the original DataFrame
    prediction_series = pd.Series(predictions, index=tempdf.index)
    result = new_data_df.copy()  # Create a copy of the original DataFrame
    result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
    result.loc[
        prediction_series.index, "Predictions"
    ] = prediction_series.values  # Assign predictions to corresponding rows
    return result["Predictions"]


def Sell_30min_A1(new_data_df):
    features = ['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'PCRv Up4', 'PCRv Down4', 'ITM PCRv Up4', 'ITM PCRv Down4', 'AwesomeOsc5_34', 'RSI2', 'AwesomeOsc']
    model_filename = f"{base_dir}/_30min_A1/target_down.joblib"
    loaded_model = joblib.load(model_filename)
    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    threshold = 1e10
    tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
    predictions = loaded_model.predict(tempdf[features])
    # Create a new Series with the predictions and align it with the original DataFrame
    prediction_series = pd.Series(predictions, index=tempdf.index)
    result = new_data_df.copy()  # Create a copy of the original DataFrame
    result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
    result.loc[
        prediction_series.index, "Predictions"
    ] = prediction_series.values  # Assign predictions to corresponding rows
    return result["Predictions"]
def Buy_20min_A1(new_data_df):
    features = [
        "Bonsai Ratio",
        "Bonsai Ratio 2",
        "B1/B2",
        "PCRv Up4",
        "PCRv Down4",
        "ITM PCRv Up4",
        "ITM PCRv Down4",
        "RSI14",
        "RSI",
        "AwesomeOsc",
    ]
    model_filename = f"{base_dir}/_20min_A1/target_up.joblib"
    loaded_model = joblib.load(model_filename)
    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    threshold = 1e10
    tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
    predictions = loaded_model.predict(tempdf[features])
    # Create a new Series with the predictions and align it with the original DataFrame
    prediction_series = pd.Series(predictions, index=tempdf.index)
    result = new_data_df.copy()  # Create a copy of the original DataFrame
    result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
    result.loc[
        prediction_series.index, "Predictions"
    ] = prediction_series.values  # Assign predictions to corresponding rows
    return result["Predictions"]


def Sell_20min_A1(new_data_df):
    features = [
        "Bonsai Ratio",
        "Bonsai Ratio 2",
        "B1/B2",
        "PCRv Up4",
        "PCRv Down4",
        "ITM PCRv Up4",
        "ITM PCRv Down4",
        "RSI14",
        "AwesomeOsc5_34",
        "RSI",
    ]
    model_filename = f"{base_dir}/_20min_A1/target_down.joblib"
    loaded_model = joblib.load(model_filename)
    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    threshold = 1e10
    tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
    predictions = loaded_model.predict(tempdf[features])
    # Create a new Series with the predictions and align it with the original DataFrame
    prediction_series = pd.Series(predictions, index=tempdf.index)
    result = new_data_df.copy()  # Create a copy of the original DataFrame
    result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
    result.loc[
        prediction_series.index, "Predictions"
    ] = prediction_series.values  # Assign predictions to corresponding rows
    return result["Predictions"]
def Buy_15min_A2(new_data_df):
    features = [
        "Bonsai Ratio",
        "B1/B2",
        "PCRv Up4",
        "PCRv Down4",
        "ITM PCRv Up4",
        "ITM PCRv Down4",
        "RSI14",
        "AwesomeOsc5_34",
        "RSI",
        "AwesomeOsc",
    ]
    model_filename = f"{base_dir}/_15min_A2/target_up.joblib"
    loaded_model = joblib.load(model_filename)
    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    threshold = 1e10
    tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
    predictions = loaded_model.predict(tempdf[features])
    # Create a new Series with the predictions and align it with the original DataFrame
    prediction_series = pd.Series(predictions, index=tempdf.index)
    result = new_data_df.copy()  # Create a copy of the original DataFrame
    result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
    result.loc[
        prediction_series.index, "Predictions"
    ] = prediction_series.values  # Assign predictions to corresponding rows
    return result["Predictions"]


def Sell_15min_A2(new_data_df):
    features = [
        "Bonsai Ratio",
        "Bonsai Ratio 2",
        "B1/B2",
        "PCRv Up4",
        "PCRv Down4",
        "ITM PCRv Up4",
        "ITM PCRv Down4",
        "RSI14",
        "AwesomeOsc5_34",
        "RSI",
    ]
    model_filename = f"{base_dir}/_15min_A2/target_down.joblib"
    loaded_model = joblib.load(model_filename)
    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    threshold = 1e10
    tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
    predictions = loaded_model.predict(tempdf[features])
    # Create a new Series with the predictions and align it with the original DataFrame
    prediction_series = pd.Series(predictions, index=tempdf.index)
    result = new_data_df.copy()  # Create a copy of the original DataFrame
    result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
    result.loc[
        prediction_series.index, "Predictions"
    ] = prediction_series.values  # Assign predictions to corresponding rows
    return result["Predictions"]



def Buy_15min_A2(new_data_df):
    features = [
        "Bonsai Ratio",
        "B1/B2",
        "PCRv Up4",
        "PCRv Down4",
        "ITM PCRv Up4",
        "ITM PCRv Down4",
        "RSI14",
        "AwesomeOsc5_34",
        "RSI",
        "AwesomeOsc",
    ]
    model_filename = f"{base_dir}/_15min_A2/target_up.joblib"
    loaded_model = joblib.load(model_filename)
    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    threshold = 1e10
    tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
    predictions = loaded_model.predict(tempdf[features])
    # Create a new Series with the predictions and align it with the original DataFrame
    prediction_series = pd.Series(predictions, index=tempdf.index)
    result = new_data_df.copy()  # Create a copy of the original DataFrame
    result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
    result.loc[
        prediction_series.index, "Predictions"
    ] = prediction_series.values  # Assign predictions to corresponding rows
    return result["Predictions"]


def Sell_15min_A2(new_data_df):
    features = [
        "Bonsai Ratio",
        "Bonsai Ratio 2",
        "B1/B2",
        "PCRv Up4",
        "PCRv Down4",
        "ITM PCRv Up4",
        "ITM PCRv Down4",
        "RSI14",
        "AwesomeOsc5_34",
        "RSI",
    ]
    model_filename = f"{base_dir}/_15min_A2/target_down.joblib"
    loaded_model = joblib.load(model_filename)
    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    threshold = 1e10
    tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
    predictions = loaded_model.predict(tempdf[features])
    # Create a new Series with the predictions and align it with the original DataFrame
    prediction_series = pd.Series(predictions, index=tempdf.index)
    result = new_data_df.copy()  # Create a copy of the original DataFrame
    result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
    result.loc[
        prediction_series.index, "Predictions"
    ] = prediction_series.values  # Assign predictions to corresponding rows
    return result["Predictions"]


def Buy_15min_A1(new_data_df):
    features = [
        "Bonsai Ratio",
        "Bonsai Ratio 2",
        "B1/B2",
        "PCRv Up4",
        "PCRv Down4",
        "ITM PCRv Up4",
        "ITM PCRv Down4",
        "AwesomeOsc5_34",
        "RSI",
        "RSI2",
    ]
    model_filename = f"{base_dir}/_15min_A1/target_up.joblib"
    loaded_model = joblib.load(model_filename)
    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    threshold = 1e10
    tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
    predictions = loaded_model.predict(tempdf[features])
    # Create a new Series with the predictions and align it with the original DataFrame
    prediction_series = pd.Series(predictions, index=tempdf.index)
    result = new_data_df.copy()  # Create a copy of the original DataFrame
    result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
    result.loc[
        prediction_series.index, "Predictions"
    ] = prediction_series.values  # Assign predictions to corresponding rows
    return result["Predictions"]


def Sell_15min_A1(new_data_df):
    features = [
        "Bonsai Ratio",
        "Bonsai Ratio 2",
        "B1/B2",
        "PCRv Up4",
        "PCRv Down4",
        "ITM PCRv Up4",
        "ITM PCRv Down4",
        "RSI14",
        "AwesomeOsc5_34",
        "RSI2",
    ]
    model_filename = f"{base_dir}/_15min_A1/target_down.joblib"
    loaded_model = joblib.load(model_filename)
    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    threshold = 1e10
    tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
    predictions = loaded_model.predict(tempdf[features])
    # Create a new Series with the predictions and align it with the original DataFrame
    prediction_series = pd.Series(predictions, index=tempdf.index)
    result = new_data_df.copy()  # Create a copy of the original DataFrame
    result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
    result.loc[
        prediction_series.index, "Predictions"
    ] = prediction_series.values  # Assign predictions to corresponding rows
    return result["Predictions"]


def Buy_5D(new_data_df):
    features = ["ITM PCRv Up4", "ITM PCRv Down4", "ITM PCRoi Down4", "RSI14"]
    model_filename = f"{base_dir}/5D/target_up.joblib"
    loaded_model = joblib.load(model_filename)
    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    threshold = 1e10
    tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
    predictions = loaded_model.predict(tempdf[features])
    # Create a new Series with the predictions and align it with the original DataFrame
    prediction_series = pd.Series(predictions, index=tempdf.index)
    result = new_data_df.copy()  # Create a copy of the original DataFrame
    result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
    result.loc[
        prediction_series.index, "Predictions"
    ] = prediction_series.values  # Assign predictions to corresponding rows
    return result["Predictions"]


def Buy_5C(new_data_df):
    features = [
        "Bonsai Ratio",
        "B1/B2",
        "PCRv Up4",
        "PCRv Down4",
        "ITM PCRv Up4",
        "ITM PCRv Down4",
        "ITM PCRoi Up4",
        "ITM PCRoi Down4",
    ]
    model_filename = f"{base_dir}/5C_5min_spy/target_up.joblib"
    loaded_model = joblib.load(model_filename)
    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    threshold = 1e10
    tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
    predictions = loaded_model.predict(tempdf[features])
    # Create a new Series with the predictions and align it with the original DataFrame
    prediction_series = pd.Series(predictions, index=tempdf.index)
    result = new_data_df.copy()  # Create a copy of the original DataFrame
    result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
    result.loc[
        prediction_series.index, "Predictions"
    ] = prediction_series.values  # Assign predictions to corresponding rows
    return result["Predictions"]


###TODO make a new layer of model that uses these modles as features.


def Buy_5A(new_data_df):
    features = [
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

    model_filename = f"{base_dir}/5A_5min_spy/target_up.joblib"
    loaded_model = joblib.load(model_filename)
    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    threshold = 1e10
    tempdf[features] = np.clip(tempdf[features], -threshold, threshold)

    predictions = loaded_model.predict(tempdf[features])
    # Create a new Series with the predictions and align it with the original DataFrame
    prediction_series = pd.Series(predictions, index=tempdf.index)
    result = new_data_df.copy()  # Create a copy of the original DataFrame
    result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
    result.loc[
        prediction_series.index, "Predictions"
    ] = prediction_series.values  # Assign predictions to corresponding rows
    return result["Predictions"]


def Sell_5A(new_data_df):
    features = [
        "Bonsai Ratio",
        "B1/B2",
        "PCRv Up4",
        "ITM PCRv Up4",
        "ITM PCRv Down4",
        "ITM PCRoi Up4",
        "ITM PCRoi Down4",
    ]

    model_filename = f"{base_dir}/5A_5min_spy/target_down.joblib"
    loaded_model = joblib.load(model_filename)
    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    threshold = 1e10
    tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
    predictions = loaded_model.predict(tempdf[features])
    # Create a new Series with the predictions and align it with the original DataFrame
    prediction_series = pd.Series(predictions, index=tempdf.index)
    result = new_data_df.copy()  # Create a copy of the original DataFrame
    result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
    result.loc[
        prediction_series.index, "Predictions"
    ] = prediction_series.values  # Assign predictions to corresponding rows
    return result["Predictions"]


def Buy_A5(new_data_df):
    features = [
        "Bonsai Ratio",
        "Bonsai Ratio 2",
        "B1/B2",
        "PCRv Up4",
        "ITM PCRv Up4",
        "ITM PCRoi Up4",
        "ITM PCRoi Down4",
    ]

    model_filename = f"{base_dir}/A5_30_min_spy_tsla/target_up.joblib"
    loaded_model = joblib.load(model_filename)
    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    threshold = 1e10
    tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
    predictions = loaded_model.predict(tempdf[features])
    # Create a new Series with the predictions and align it with the original DataFrame
    prediction_series = pd.Series(predictions, index=tempdf.index)
    result = new_data_df.copy()  # Create a copy of the original DataFrame
    result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
    result.loc[
        prediction_series.index, "Predictions"
    ] = prediction_series.values  # Assign predictions to corresponding rows
    return result["Predictions"]


def Sell_A5(new_data_df):
    features = ["Bonsai Ratio", "B1/B2", "PCRv Up4", "ITM PCRv Up4", "ITM PCRoi Up4", "ITM PCRoi Down4"]

    model_filename = f"{base_dir}/A5_30_min_spy_tsla/target_down.joblib"
    loaded_model = joblib.load(model_filename)
    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    threshold = 1e10
    tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
    predictions = loaded_model.predict(tempdf[features])
    # Create a new Series with the predictions and align it with the original DataFrame
    prediction_series = pd.Series(predictions, index=tempdf.index)
    result = new_data_df.copy()  # Create a copy of the original DataFrame
    result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
    result.loc[
        prediction_series.index, "Predictions"
    ] = prediction_series.values  # Assign predictions to corresponding rows
    return result["Predictions"]


def Buy_A4(new_data_df):
    features = ["Bonsai Ratio", "Bonsai Ratio 2", "B1/B2", "ITM PCRv Down4", "ITM PCRoi Up4", "ITM PCRoi Down4"]

    model_filename = f"{base_dir}/A4_20min_02percent/target_up.joblib"
    loaded_model = joblib.load(model_filename)
    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    threshold = 1e10
    tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
    predictions = loaded_model.predict(tempdf[features])
    # Create a new Series with the predictions and align it with the original DataFrame
    prediction_series = pd.Series(predictions, index=tempdf.index)
    result = new_data_df.copy()  # Create a copy of the original DataFrame
    result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
    result.loc[
        prediction_series.index, "Predictions"
    ] = prediction_series.values  # Assign predictions to corresponding rows
    return result["Predictions"]


def Sell_A4(new_data_df):
    features = [
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

    model_filename = f"{base_dir}/A4_20min_02percent/target_down.joblib"
    loaded_model = joblib.load(model_filename)
    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    threshold = 1e10
    tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
    predictions = loaded_model.predict(tempdf[features])
    # Create a new Series with the predictions and align it with the original DataFrame
    prediction_series = pd.Series(predictions, index=tempdf.index)
    result = new_data_df.copy()  # Create a copy of the original DataFrame
    result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
    result.loc[
        prediction_series.index, "Predictions"
    ] = prediction_series.values  # Assign predictions to corresponding rows
    return result["Predictions"]


def Buy_A3(new_data_df):
    features = ["Bonsai Ratio", "Bonsai Ratio 2", "B1/B2", "ITM PCRv Down4", "ITM PCRoi Up4", "ITM PCRoi Down4"]
    model_filename = f"{base_dir}/A3_Looks_Best_45min/target_up.joblib"
    loaded_model = joblib.load(model_filename)
    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    threshold = 1e10
    tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
    predictions = loaded_model.predict(tempdf[features])
    # Create a new Series with the predictions and align it with the original DataFrame
    prediction_series = pd.Series(predictions, index=tempdf.index)
    result = new_data_df.copy()  # Create a copy of the original DataFrame
    result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
    result.loc[
        prediction_series.index, "Predictions"
    ] = prediction_series.values  # Assign predictions to corresponding rows
    return result["Predictions"]



def A1_Buy(new_data_df):
    features = ["Bonsai Ratio", "Bonsai Ratio 2", "PCRoi Up1", "ITM PCRoi Up1"]
    model_filename = f"{base_dir}/A1_3_5hour_b1_b2_pcroiup1_itmpcroiup1_nivlac/target_up.joblib"
    loaded_model = joblib.load(model_filename)
    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame

    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    threshold = 1e10
    tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
    predictions = loaded_model.predict(tempdf[features])
    # Create a new Series with the predictions and align it with the original DataFrame
    prediction_series = pd.Series(predictions, index=tempdf.index)
    result = new_data_df.copy()  # Create a copy of the original DataFrame
    result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
    result.loc[
        prediction_series.index, "Predictions"
    ] = prediction_series.values  # Assign predictions to corresponding rows
    return result["Predictions"]


def A1_Sell(new_data_df):
    features = ["Bonsai Ratio", "Bonsai Ratio 2", "PCRoi Up1", "ITM PCRoi Up1", "Net IV LAC"]
    model_filename = f"{base_dir}/A1_3_5hour_b1_b2_pcroiup1_itmpcroiup1_nivlac/target_down.joblib"
    loaded_model = joblib.load(model_filename)
    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    threshold = 1e10
    tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
    predictions = loaded_model.predict(tempdf[features])
    # Create a new Series with the predictions and align it with the original DataFrame
    prediction_series = pd.Series(predictions, index=tempdf.index)
    result = new_data_df.copy()  # Create a copy of the original DataFrame
    result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
    result.loc[
        prediction_series.index, "Predictions"
    ] = prediction_series.values  # Assign predictions to corresponding rows
    return result["Predictions"]


# In the get_buy_signal() function, it loads the model (trained_model_target_up.joblib) specifically trained for the "buy" signal (target_up). It accepts predictor inputs as a list (predictors) and assumes you will provide the corresponding values for the predictors. It then creates a DataFrame (new_data_df) with the new data and makes predictions using the loaded model. The predictions are returned as the buy signal.
#
# Similarly, the get_sell_signal() function loads the model (trained_model_target_down.joblib) specifically trained for the "sell" signal (target_down). It follows the same process as the get_buy_signal() function to make predictions based on the provided predictor inputs and returns the sell signal.
#
# Note: Make sure you have trained and saved the models separately for each target before using these functions, and replace <value> with the actual values for the predictors you want to use.
#
