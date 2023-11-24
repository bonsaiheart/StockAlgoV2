import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
base_dir = os.path.dirname(__file__)



#spy

def Buy_1hr_15minA2base_spya1(new_data_df):
    with open(f"{base_dir}/_1hr_15minA2base_spya1/min_max_values.json", 'r') as f:
        min_max_dict = json.load(f)
    with open(f"{base_dir}/_1hr_15minA2base_spya1/features_up.json", 'r') as f2:
        features = json.load(f2)
    features = features
    print(features)
    df = new_data_df.copy()
    for col in features:
        min_val = min_max_dict[col]['min_val']
        max_val = min_max_dict[col]['max_val']
        df[col].replace([np.inf, -np.inf], [max_val, min_val], inplace=True)

    model_filename = f"{base_dir}/_1hr_15minA2base_spya1/target_up.joblib"
    loaded_model = joblib.load(model_filename)
    tempdf = df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    tempdf['LastTradeTime'] = tempdf['LastTradeTime'].apply(
        lambda x: datetime.strptime(str(x), '%y%m%d_%H%M') if not pd.isna(x) else np.nan)
    tempdf['LastTradeTime'] = tempdf['LastTradeTime'].apply(lambda x: x.timestamp())
    tempdf['LastTradeTime'] = tempdf['LastTradeTime'] / (60 * 60 * 24 * 7)
    predictions = loaded_model.predict(tempdf[features])
    # Create a new Series with the predictions and align it with the original DataFrame
    prediction_series = pd.Series(predictions, index=tempdf.index)
    result = df.copy()  # Create a copy of the original DataFrame
    result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
    result.loc[
        prediction_series.index, "Predictions"
    ] = prediction_series.values  # Assign predictions to corresponding rows
    return result["Predictions"]
#spy
def Sell_1hr_15minA2base_spya1(new_data_df):
    with open(f"{base_dir}/_1hr_15minA2base_spya1/min_max_values.json", 'r') as f:
        min_max_dict = json.load(f)
    with open(f"{base_dir}/_1hr_15minA2base_spya1/features_down.json", 'r') as f2:
        features = json.load(f2)
    # features = ['LastTradeTime', 'Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'PCRv Up4', 'PCRv Down4', 'ITM PCRv Up2', 'ITM PCRv Up4']
    df = new_data_df.copy()
    print(features)

    for col in features:
        min_val = min_max_dict[col]['min_val']
        max_val = min_max_dict[col]['max_val']
        df[col].replace([np.inf, -np.inf], [max_val, min_val], inplace=True)

    model_filename = f"{base_dir}/_1hr_15minA2base_spya1/target_down.joblib"
    loaded_model = joblib.load(model_filename)
    tempdf = df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    tempdf['LastTradeTime'] = tempdf['LastTradeTime'].apply(
        lambda x: datetime.strptime(str(x), '%y%m%d_%H%M') if not pd.isna(x) else np.nan)
    tempdf['LastTradeTime'] = tempdf['LastTradeTime'].apply(lambda x: x.timestamp())
    tempdf['LastTradeTime'] = tempdf['LastTradeTime'] / (60 * 60 * 24 * 7)
    predictions = loaded_model.predict(tempdf[features])
    # Create a new Series with the predictions and align it with the original DataFrame
    prediction_series = pd.Series(predictions, index=tempdf.index)
    result = df.copy()  # Create a copy of the original DataFrame
    result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
    result.loc[
        prediction_series.index, "Predictions"
    ] = prediction_series.values  # Assign predictions to corresponding rows
    return result["Predictions"]
def Buy_4hr_15minA2base_spyA1(new_data_df):
    with open(f"{base_dir}/_4hr_15minA2base_spyA1/min_max_values.json", 'r') as f:
        min_max_dict = json.load(f)
    features =  ['LastTradeTime', 'Bonsai Ratio', 'B1/B2', 'PCRv Up4', 'PCRv Down4', 'ITM PCRv Up2', 'ITM PCRv Up4', 'ITM PCRv Down4']
    df = new_data_df.copy()
    print(df['LastTradeTime'])

    for col in features:
        min_val = min_max_dict[col]['min_val']
        max_val = min_max_dict[col]['max_val']
        df[col].replace([np.inf, -np.inf], [max_val, min_val], inplace=True)

    model_filename = f"{base_dir}/_4hr_15minA2base_spyA1/target_up.joblib"
    loaded_model = joblib.load(model_filename)
    tempdf = df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    tempdf['LastTradeTime'] = tempdf['LastTradeTime'].apply(
        lambda x: datetime.strptime(str(x), '%y%m%d_%H%M') if not pd.isna(x) else np.nan)
    tempdf['LastTradeTime'] = tempdf['LastTradeTime'].apply(lambda x: x.timestamp())
    tempdf['LastTradeTime'] = tempdf['LastTradeTime'] / (60 * 60 * 24 * 7)
    predictions = loaded_model.predict(tempdf[features])
    # Create a new Series with the predictions and align it with the original DataFrame
    prediction_series = pd.Series(predictions, index=tempdf.index)
    result = df.copy()  # Create a copy of the original DataFrame
    result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
    result.loc[
        prediction_series.index, "Predictions"
    ] = prediction_series.values  # Assign predictions to corresponding rows
    return result["Predictions"]
#spy
def Sell_4hr_15minA2base_spyA1(new_data_df):
    with open(f"{base_dir}/_4hr_15minA2base_spyA1/min_max_values.json", 'r') as f:
        min_max_dict = json.load(f)
    features =['LastTradeTime', 'Bonsai Ratio', 'B1/B2', 'PCRv Down4', 'ITM PCRv Up2', 'ITM PCRv Down2', 'ITM PCRv Up4', 'ITM PCRv Down4']
    df = new_data_df.copy()
    print(df['LastTradeTime'])

    for col in features:
        min_val = min_max_dict[col]['min_val']
        max_val = min_max_dict[col]['max_val']
        df[col].replace([np.inf, -np.inf], [max_val, min_val], inplace=True)

    model_filename = f"{base_dir}/_4hr_15minA2base_spyA1/target_down.joblib"
    loaded_model = joblib.load(model_filename)
    tempdf = df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    tempdf['LastTradeTime'] = tempdf['LastTradeTime'].apply(
        lambda x: datetime.strptime(str(x), '%y%m%d_%H%M') if not pd.isna(x) else np.nan)
    tempdf['LastTradeTime'] = tempdf['LastTradeTime'].apply(lambda x: x.timestamp())
    tempdf['LastTradeTime'] = tempdf['LastTradeTime'] / (60 * 60 * 24 * 7)
    predictions = loaded_model.predict(tempdf[features])
    # Create a new Series with the predictions and align it with the original DataFrame
    print(tempdf['LastTradeTime'])
    prediction_series = pd.Series(predictions, index=tempdf.index)
    result = df.copy()  # Create a copy of the original DataFrame
    result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
    result.loc[
        prediction_series.index, "Predictions"
    ] = prediction_series.values  # Assign predictions to corresponding rows
    return result["Predictions"]
def Buy_4hr_15minA2base_ROKUa1(new_data_df):
    with open(f"{base_dir}/_4hr_15minA2base_ROKUa1/min_max_values.json", 'r') as f:
        min_max_dict = json.load(f)
    features =  ['LastTradeTime', 'Bonsai Ratio', 'B1/B2', 'PCRv Down4', 'ITM PCRv Up2', 'ITM PCRv Down2', 'ITM PCRv Up4', 'ITM PCRv Down4']
    df = new_data_df.copy()

    for col in features:
        min_val = min_max_dict[col]['min_val']
        max_val = min_max_dict[col]['max_val']
        df[col].replace([np.inf, -np.inf], [max_val, min_val], inplace=True)

    model_filename = f"{base_dir}/_4hr_15minA2base_ROKUa1/target_up.joblib"
    loaded_model = joblib.load(model_filename)
    tempdf = df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    tempdf['LastTradeTime'] = tempdf['LastTradeTime'].apply(
        lambda x: datetime.strptime(str(x), '%y%m%d_%H%M') if not pd.isna(x) else np.nan)
    tempdf['LastTradeTime'] = tempdf['LastTradeTime'].apply(lambda x: x.timestamp())
    tempdf['LastTradeTime'] = tempdf['LastTradeTime'] / (60 * 60 * 24 * 7)
    predictions = loaded_model.predict(tempdf[features])
    # Create a new Series with the predictions and align it with the original DataFrame
    prediction_series = pd.Series(predictions, index=tempdf.index)
    result = df.copy()  # Create a copy of the original DataFrame
    result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
    result.loc[
        prediction_series.index, "Predictions"
    ] = prediction_series.values  # Assign predictions to corresponding rows
    return result["Predictions"]
#spy
def Sell_4hr_15minA2base_ROKUa1(new_data_df):
    with open(f"{base_dir}/_4hr_15minA2base_ROKUa1/min_max_values.json", 'r') as f:
        min_max_dict = json.load(f)
    features =['LastTradeTime', 'Bonsai Ratio', 'B1/B2', 'PCRv Down4', 'ITM PCRv Up2', 'ITM PCRv Down2', 'ITM PCRv Up4', 'ITM PCRv Down4']
    df = new_data_df.copy()
    print(df['LastTradeTime'])

    for col in features:
        min_val = min_max_dict[col]['min_val']
        max_val = min_max_dict[col]['max_val']
        df[col].replace([np.inf, -np.inf], [max_val, min_val], inplace=True)

    model_filename = f"{base_dir}/_4hr_15minA2base_ROKUa1/target_down.joblib"
    loaded_model = joblib.load(model_filename)
    tempdf = df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    tempdf['LastTradeTime'] = tempdf['LastTradeTime'].apply(
        lambda x: datetime.strptime(str(x), '%y%m%d_%H%M') if not pd.isna(x) else np.nan)
    tempdf['LastTradeTime'] = tempdf['LastTradeTime'].apply(lambda x: x.timestamp())
    tempdf['LastTradeTime'] = tempdf['LastTradeTime'] / (60 * 60 * 24 * 7)
    predictions = loaded_model.predict(tempdf[features])
    # Create a new Series with the predictions and align it with the original DataFrame
    prediction_series = pd.Series(predictions, index=tempdf.index)
    result = df.copy()  # Create a copy of the original DataFrame
    result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
    result.loc[
        prediction_series.index, "Predictions"
    ] = prediction_series.values  # Assign predictions to corresponding rows
    return result["Predictions"]
#spy
def Buy_4hr_15mina2base_a1(new_data_df):
    with open(f"{base_dir}/_4hr_15mina2base_a1/min_max_values.json", 'r') as f:
        min_max_dict = json.load(f)
    features =   ['LastTradeTime', 'Bonsai Ratio', 'B1/B2', 'PCRv Up4', 'PCRv Down4', 'ITM PCRv Up2', 'ITM PCRv Up4', 'ITM PCRv Down4']
    df = new_data_df.copy()
    print(df['LastTradeTime'])

    for col in features:
        min_val = min_max_dict[col]['min_val']
        max_val = min_max_dict[col]['max_val']
        df[col].replace([np.inf, -np.inf], [max_val, min_val], inplace=True)

    model_filename = f"{base_dir}/_4hr_15mina2base_a1/target_up.joblib"
    loaded_model = joblib.load(model_filename)
    tempdf = df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    tempdf['LastTradeTime'] = tempdf['LastTradeTime'].apply(
        lambda x: datetime.strptime(str(x), '%y%m%d_%H%M') if not pd.isna(x) else np.nan)
    tempdf['LastTradeTime'] = tempdf['LastTradeTime'].apply(lambda x: x.timestamp())
    tempdf['LastTradeTime'] = tempdf['LastTradeTime'] / (60 * 60 * 24 * 7)
    predictions = loaded_model.predict(tempdf[features])
    # Create a new Series with the predictions and align it with the original DataFrame
    prediction_series = pd.Series(predictions, index=tempdf.index)
    result = df.copy()  # Create a copy of the original DataFrame
    result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
    result.loc[
        prediction_series.index, "Predictions"
    ] = prediction_series.values  # Assign predictions to corresponding rows
    return result["Predictions"]
#spy
def Sell_4hr_15mina2base_a1(new_data_df):
    with open(f"{base_dir}/_4hr_15mina2base_a1/min_max_values.json", 'r') as f:
        min_max_dict = json.load(f)
    features =['LastTradeTime', 'Bonsai Ratio', 'B1/B2', 'PCRv Down4', 'ITM PCRv Up2', 'ITM PCRv Down2', 'ITM PCRv Up4', 'ITM PCRv Down4']
    df = new_data_df.copy()
    print(df['LastTradeTime'])

    for col in features:
        min_val = min_max_dict[col]['min_val']
        max_val = min_max_dict[col]['max_val']
        df[col].replace([np.inf, -np.inf], [max_val, min_val], inplace=True)

    model_filename = f"{base_dir}/_4hr_15mina2base_a1/target_down.joblib"
    loaded_model = joblib.load(model_filename)
    tempdf = df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    tempdf['LastTradeTime'] = tempdf['LastTradeTime'].apply(
        lambda x: datetime.strptime(str(x), '%y%m%d_%H%M') if not pd.isna(x) else np.nan)
    tempdf['LastTradeTime'] = tempdf['LastTradeTime'].apply(lambda x: x.timestamp())
    tempdf['LastTradeTime'] = tempdf['LastTradeTime'] / (60 * 60 * 24 * 7)
    predictions = loaded_model.predict(tempdf[features])
    # Create a new Series with the predictions and align it with the original DataFrame
    prediction_series = pd.Series(predictions, index=tempdf.index)
    result = df.copy()  # Create a copy of the original DataFrame
    result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
    result.loc[
        prediction_series.index, "Predictions"
    ] = prediction_series.values  # Assign predictions to corresponding rows
    return result["Predictions"]
def Buy_3hr_15minA2baseSPYA1(new_data_df):
    with open(f"{base_dir}/_3hr_15minA2baseSPYA1/min_max_values.json", 'r') as f:
        min_max_dict = json.load(f)
    features = ['LastTradeTime', 'Bonsai Ratio', 'B1/B2', 'PCRv Up4', 'PCRv Down4', 'ITM PCRv Up2', 'ITM PCRv Up4', 'ITM PCRv Down4']
    for col in features:
        min_val = min_max_dict[col]['min_val']
        max_val = min_max_dict[col]['max_val']
        new_data_df[col].replace([np.inf, -np.inf], [max_val, min_val], inplace=True)

    model_filename = f"{base_dir}/_3hr_15minA2baseSPYA1/target_up.joblib"
    loaded_model = joblib.load(model_filename)
    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    predictions = loaded_model.predict(tempdf[features])
    # Create a new Series with the predictions and align it with the original DataFrame
    prediction_series = pd.Series(predictions, index=tempdf.index)
    result = new_data_df.copy()  # Create a copy of the original DataFrame
    result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
    result.loc[
        prediction_series.index, "Predictions"
    ] = prediction_series.values  # Assign predictions to corresponding rows
    return result["Predictions"]
def Sell_3hr_15minA2baseSPYA1(new_data_df):
    with open(f"{base_dir}/_3hr_15minA2baseSPYA1/min_max_values.json", 'r') as f:
        min_max_dict = json.load(f)
    features =['LastTradeTime', 'Bonsai Ratio', 'B1/B2', 'PCRv Down4', 'ITM PCRv Up2',
       'ITM PCRv Down2', 'ITM PCRv Up4', 'ITM PCRv Down4']
    for col in features:
        min_val = min_max_dict[col]['min_val']
        max_val = min_max_dict[col]['max_val']
        new_data_df[col].replace([np.inf, -np.inf], [max_val, min_val], inplace=True)

    model_filename = f"{base_dir}/_3hr_15minA2baseSPYA1/target_down.joblib"
    loaded_model = joblib.load(model_filename)
    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    predictions = loaded_model.predict(tempdf[features])
    # Create a new Series with the predictions and align it with the original DataFrame
    prediction_series = pd.Series(predictions, index=tempdf.index)
    result = new_data_df.copy()  # Create a copy of the original DataFrame
    result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
    result.loc[
        prediction_series.index, "Predictions"
    ] = prediction_series.values  # Assign predictions to corresponding rows
    return result["Predictions"]
def Buy_30min_15minA2SPY_A1_test(new_data_df):
    with open(f"{base_dir}/_30min_15minA2SPY_A1_test/min_max_values.json", 'r') as f:
        min_max_dict = json.load(f)
    features =['LastTradeTime', 'Bonsai Ratio', 'B1/B2', 'PCRv Up4', 'PCRv Down4',
       'ITM PCRv Up2', 'ITM PCRv Up4', 'ITM PCRv Down4']
    for col in features:
        min_val = min_max_dict[col]['min_val']
        max_val = min_max_dict[col]['max_val']
        new_data_df[col].replace([np.inf, -np.inf], [max_val, min_val], inplace=True)

    model_filename = f"{base_dir}/_30min_15minA2SPY_A1_test/target_up.joblib"
    loaded_model = joblib.load(model_filename)
    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    predictions = loaded_model.predict(tempdf[features])
    # Create a new Series with the predictions and align it with the original DataFrame
    prediction_series = pd.Series(predictions, index=tempdf.index)
    result = new_data_df.copy()  # Create a copy of the original DataFrame
    result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
    result.loc[
        prediction_series.index, "Predictions"
    ] = prediction_series.values  # Assign predictions to corresponding rows
    return result["Predictions"]
def Sell_30min_15minA2SPY_A1_test(new_data_df):
    with open(f"{base_dir}/_30min_15minA2SPY_A1_test/min_max_values.json", 'r') as f:
        min_max_dict = json.load(f)
    features =['LastTradeTime', 'Bonsai Ratio', 'B1/B2', 'PCRv Up4', 'PCRv Down4',
       'ITM PCRv Up2', 'ITM PCRv Down2', 'ITM PCRv Up4']
    for col in features:
        min_val = min_max_dict[col]['min_val']
        max_val = min_max_dict[col]['max_val']
        new_data_df[col].replace([np.inf, -np.inf], [max_val, min_val], inplace=True)

    model_filename = f"{base_dir}/_30min_15minA2SPY_A1_test/target_down.joblib"
    loaded_model = joblib.load(model_filename)
    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    predictions = loaded_model.predict(tempdf[features])
    # Create a new Series with the predictions and align it with the original DataFrame
    prediction_series = pd.Series(predictions, index=tempdf.index)
    result = new_data_df.copy()  # Create a copy of the original DataFrame
    result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
    result.loc[
        prediction_series.index, "Predictions"
    ] = prediction_series.values  # Assign predictions to corresponding rows
    return result["Predictions"]

def Buy_2hr_RFSPYA2(new_data_df):
    features =['Bonsai Ratio', 'B1/B2', 'PCRv Down3', 'PCRv Down2', 'ITM PCRv Up3', 'ITM PCRv Down3', 'ITM PCRv Up4']

    model_filename = f"{base_dir}/_2hr_RFSPYA2/target_up.joblib"
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

def Sell_2hr_RFSPYA2(new_data_df):
    features = ['Bonsai Ratio', 'PCRv Down3', 'PCRv Down4', 'ITM PCRv Up3', 'ITM PCRv Down3', 'ITM PCRv Down2', 'ITM PCRv Down4']

    model_filename = f"{base_dir}/_2hr_RFSPYA2/target_down.joblib"
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
def Buy_2hr_RFSPYA1(new_data_df):
    features =['Bonsai Ratio', 'B1/B2', 'PCRv Down2', 'ITM PCRv Up3', 'ITM PCRv Down3', 'ITM PCRv Up4', 'ITM PCRv Up2']
    model_filename = f"{base_dir}/_2hr_RFSPYA1/target_up.joblib"
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

def Sell_2hr_RFSPYA1(new_data_df):
    features = ['Bonsai Ratio', 'PCRv Down3', 'PCRv Down4', 'ITM PCRv Up3', 'ITM PCRv Down3', 'ITM PCRv Down2', 'ITM PCRv Down4']

    model_filename = f"{base_dir}/_2hr_RFSPYA1/target_down.joblib"
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
def Buy_90min_A5(new_data_df):
    features =['Bonsai Ratio', 'PCRv Up3', 'PCRv Down3', 'PCRv Up4', 'ITM PCRv Down3',
       'ITM PCRv Down4']
    model_filename = f"{base_dir}/_90min_A5/target_up.joblib"
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

def Sell_90min_A5(new_data_df):
    features = ['Bonsai Ratio', 'PCRv Down3', 'PCRv Down4', 'ITM PCRv Up3',
       'ITM PCRv Down3', 'ITM PCRv Down4']
    model_filename = f"{base_dir}/_90min_A5/target_down.joblib"
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

# def Sell_2hr_nnA1(new_data_df):
#     features =     ["Bonsai Ratio",
#     "Bonsai Ratio 2",
#     "B1/B2",
#     "PCRv Up3", "PCRv Up2",
#     "PCRv Down3", "PCRv Down2",
#     "PCRv Up4",
#     "PCRv Down4",
#     "ITM PCRv Up3",
#     "ITM PCRv Down3", "ITM PCRv Up4", "ITM PCRv Down2", "ITM PCRv Up2",
#     "ITM PCRv Down4",
#     "RSI14",
#     "AwesomeOsc5_34",
#     "RSI",
#     "RSI2",
#     "AwesomeOsc",
# ]
#     model_filename = f"{base_dir}/_2hr_nnA1/target_down/"
#     print(model_filename)
#     loaded_model = load_model(model_filename)
#     tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
#     tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
#     threshold = 1e10
#     print(tempdf[features])
#     tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
#     predictions = loaded_model.predict(tempdf[features])
#     print(predictions)
#     # Create a new Series with the predictions and align it with the original DataFrame
#     prediction_series = pd.Series(predictions.flatten(), index=tempdf.index)
#     result = new_data_df.copy()  # Create a copy of the original DataFrame
#     result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
#     result.loc[  prediction_series.index, "Predictions"
#     ] = prediction_series.values  # Assign predictions to corresponding rows
#     return result["Predictions"]
# def Buy_2hr_nnA2(new_data_df):
#     features =[
#     "Bonsai Ratio",
#     "Bonsai Ratio 2",
#     "B1/B2",
#     "PCRv Up3", "PCRv Up2",
#     "PCRv Down3", "PCRv Down2",
# 'ITM PCRoi Up1','ITM PCRoi Down1',
#     "ITM PCRv Up3", 'Net_IV', 'Net ITM IV',
#     "ITM PCRv Down3", "ITM PCRv Up4", "ITM PCRv Down2", "ITM PCRv Up2",
#     "ITM PCRv Down4",
#     "RSI14",
#     "AwesomeOsc5_34",
#     "RSI",
#     "RSI2",
#     "AwesomeOsc",
# ]
#     model_filename = f"{base_dir}/_2hr_nnA2/target_up"
#     loaded_model  = load_model(model_filename)
#     tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
#     tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
#     threshold = 1e10
#     tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
#     print(tempdf[features])
#     predictions = loaded_model.predict(tempdf[features])
#     print(predictions)
#     # Create a new Series with the predictions and align it with the original DataFrame
#     prediction_series = pd.Series(predictions.flatten(), index=tempdf.index)
#     result = new_data_df.copy()  # Create a copy of the original DataFrame
#     result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
#     result.loc[
#         prediction_series.index, "Predictions"
#     ] = prediction_series.values  # Assign predictions to corresponding rows
#     return result["Predictions"]
#
# def Sell_2hr_nnA2(new_data_df):
#     features = [
#     "Bonsai Ratio",
#     "Bonsai Ratio 2",
#     "B1/B2",
#     "PCRv Up3", "PCRv Up2",
#     "PCRv Down3", "PCRv Down2",
# 'ITM PCRoi Up1','ITM PCRoi Down1',
#     "ITM PCRv Up3", 'Net_IV', 'Net ITM IV',
#     "ITM PCRv Down3", "ITM PCRv Up4", "ITM PCRv Down2", "ITM PCRv Up2",
#     "ITM PCRv Down4",
#     "RSI14",
#     "AwesomeOsc5_34",
#     "RSI",
#     "RSI2",
#     "AwesomeOsc",
# ]
#     model_filename = f"{base_dir}/_2hr_nnA2/target_down/"
#     print(model_filename)
#     loaded_model = load_model(model_filename)
#     tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
#     tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
#     threshold = 1e10
#     print(tempdf[features])
#     tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
#     predictions = loaded_model.predict(tempdf[features])
#     print(predictions)
#     # Create a new Series with the predictions and align it with the original DataFrame
#     prediction_series = pd.Series(predictions.flatten(), index=tempdf.index)
#     result = new_data_df.copy()  # Create a copy of the original DataFrame
#     result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
#     result.loc[  prediction_series.index, "Predictions"
#     ] = prediction_series.values  # Assign predictions to corresponding rows
#     return result["Predictions"]

def Buy_90min_A4(new_data_df):
    features =['Bonsai Ratio', 'B1/B2', 'PCRv Down4', 'ITM PCRv Up3']
    model_filename = f"{base_dir}/_90min_A4/target_up.joblib"
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

def Sell_90min_A4(new_data_df):
    features = ['PCRv Down3', 'PCRv Down4', 'ITM PCRv Down3', 'ITM PCRv Down4']
    model_filename = f"{base_dir}/_90min_A4/target_down.joblib"
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

def Buy_90min_A3(new_data_df):
    features =['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'PCRv Up3', 'PCRv Down3', 'PCRv Down4', 'ITM PCRv Up3', 'ITM PCRv Down3', 'ITM PCRv Up4', 'ITM PCRv Down4']
    model_filename = f"{base_dir}/_90min_A3/target_up.joblib"
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

def Sell_90min_A3(new_data_df):
    features = ['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'PCRv Down3', 'PCRv Up4', 'PCRv Down4', 'ITM PCRv Up3', 'ITM PCRv Down3', 'ITM PCRv Up4', 'ITM PCRv Down4']

    model_filename = f"{base_dir}/_90min_A3/target_down.joblib"
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

def Buy_90min_A2(new_data_df):
    features =['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'PCRv Down3', 'PCRv Up4', 'PCRv Down4', 'ITM PCRv Up3', 'ITM PCRv Down3', 'ITM PCRv Up4', 'ITM PCRv Down4']
    model_filename = f"{base_dir}/_90min_A2/target_up.joblib"
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
#
def Sell_90min_A2(new_data_df):
    features = ['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'PCRv Down3', 'PCRv Up4',
       'PCRv Down4', 'ITM PCRv Up3', 'ITM PCRv Down3', 'ITM PCRv Up4',
       'ITM PCRv Down4']
    model_filename = f"{base_dir}/_90min_A2/target_down.joblib"
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

def Buy_90min_A1(new_data_df):
    features =['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'PCRv Down3', 'PCRv Up4', 'PCRv Down4', 'ITM PCRv Up3', 'ITM PCRv Down3', 'ITM PCRv Up4', 'ITM PCRv Down4']

    model_filename = f"{base_dir}/_90min_A1/target_up.joblib"
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

def Sell_90min_A1(new_data_df):
    features = ['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'PCRv Down3', 'PCRv Up4', 'PCRv Down4', 'ITM PCRv Up3', 'ITM PCRv Down3', 'ITM PCRv Up4', 'ITM PCRv Down4']

    model_filename = f"{base_dir}/_90min_A1/target_down.joblib"
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
###TODO could make features = modle.info "features"
###supposed to be for 30 min .3 spy tsla
def Buy_2hr_A2(new_data_df):
    features = ['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'PCRv Up3', 'PCRv Up2',
       'PCRv Down3', 'PCRv Down2', 'PCRv Up4', 'PCRv Down4', 'ITM PCRv Up3',
       'ITM PCRv Down3', 'ITM PCRv Up4', 'ITM PCRv Down2', 'ITM PCRv Up2',
       'ITM PCRv Down4', 'RSI14', 'AwesomeOsc5_34', 'RSI', 'RSI2',
       'AwesomeOsc']
    model_filename = f"{base_dir}/_2hr_A2/target_up.joblib"
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
def Sell_2hr_A2(new_data_df):
    features = ['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'PCRv Up3', 'PCRv Up2', 'PCRv Down3', 'PCRv Down2', 'PCRv Up4', 'PCRv Down4', 'ITM PCRv Up3', 'ITM PCRv Down3', 'ITM PCRv Up4', 'ITM PCRv Down2', 'ITM PCRv Up2', 'ITM PCRv Down4', 'RSI14', 'AwesomeOsc5_34', 'RSI', 'RSI2', 'AwesomeOsc']

    model_filename = f"{base_dir}/_2hr_A2/target_down.joblib"
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

def Buy_2hr_A1(new_data_df):
    features = ['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'PCRv Up4', 'PCRv Down4', 'ITM PCRv Up4', 'ITM PCRv Down4', 'AwesomeOsc5_34', 'RSI', 'AwesomeOsc']
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
def Buy_1hr_A9(new_data_df):
    features =['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'PCRv Up3', 'PCRv Up2', 'PCRv Down3', 'PCRv Down2', 'PCRv Up4', 'PCRv Down4', 'ITM PCRv Up3', 'ITM PCRv Down3', 'ITM PCRv Up4', 'ITM PCRv Down2', 'ITM PCRv Up2', 'ITM PCRv Down4', 'RSI14', 'AwesomeOsc5_34', 'RSI', 'RSI2', 'AwesomeOsc']

    model_filename = f"{base_dir}/_1hr_A9/target_up.joblib"
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

def Sell_1hr_A9(new_data_df):
    features = ['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'PCRv Up3', 'PCRv Up2',
       'PCRv Down3', 'PCRv Down2', 'PCRv Up4', 'PCRv Down4', 'ITM PCRv Up3',
       'ITM PCRv Down3', 'ITM PCRv Up4', 'ITM PCRv Down2', 'ITM PCRv Up2',
       'ITM PCRv Down4', 'RSI14', 'AwesomeOsc5_34', 'RSI', 'RSI2',
       'AwesomeOsc']
    model_filename = f"{base_dir}/_1hr_A9/target_down.joblib"
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
def Buy_1hr_A8(new_data_df):
    features =['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'PCRv Down2', 'ITM PCRv Down2']

    model_filename = f"{base_dir}/_1hr_A8/target_up.joblib"
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

def Sell_1hr_A8(new_data_df):
    features = ['PCRv Down3', 'PCRv Down2', 'PCRv Down4', 'ITM PCRv Down3',
       'ITM PCRv Down4']
    model_filename = f"{base_dir}/_1hr_A8/target_down.joblib"
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
def Buy_1hr_A7(new_data_df):
    features =['Bonsai Ratio', 'B1/B2', 'PCRv Down4', 'ITM PCRv Up3', 'ITM PCRv Down4']

    model_filename = f"{base_dir}/_1hr_A7/target_up.joblib"
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

def Sell_1hr_A7(new_data_df):
    features = ['Bonsai Ratio', 'PCRv Down3', 'PCRv Down4', 'ITM PCRv Down3',
       'ITM PCRv Down4']
    model_filename = f"{base_dir}/_1hr_A7/target_down.joblib"
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

def Buy_1hr_A6(new_data_df):
    features =['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'PCRv Down3', 'PCRv Up4',
       'PCRv Down4', 'ITM PCRv Up3', 'ITM PCRv Down3', 'ITM PCRv Up4',
       'ITM PCRv Down4']

    model_filename = f"{base_dir}/_1hr_A6/target_up.joblib"
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

def Sell_1hr_A6(new_data_df):
    features = ['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'PCRv Up4', 'PCRv Down4',
       'ITM PCRv Up4', 'ITM PCRv Down4', 'RSI14', 'AwesomeOsc5_34', 'RSI']
    model_filename = f"{base_dir}/_1hr_A6/target_down.joblib"
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
def Buy_1hr_A5(new_data_df):
    features = ['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'PCRv Up4', 'PCRv Down4',
       'ITM PCRv Up4', 'ITM PCRv Down4', 'RSI14', 'AwesomeOsc5_34', 'RSI']

    model_filename = f"{base_dir}/_1hr_A5/target_up.joblib"
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
def Sell_1hr_A5(new_data_df):
    features = ['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'PCRv Down3', 'PCRv Up4',
       'PCRv Down4', 'ITM PCRv Up3', 'ITM PCRv Down3', 'ITM PCRv Up4',
       'ITM PCRv Down4']
    model_filename = f"{base_dir}/_1hr_A5/target_down.joblib"
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
def Buy_1hr_A4(new_data_df):
    features = ['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'PCRv Up4', 'PCRv Down4', 'ITM PCRv Up4', 'ITM PCRv Down4', 'RSI14', 'AwesomeOsc5_34', 'AwesomeOsc']

    model_filename = f"{base_dir}/_1hr_A4/target_up.joblib"
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
def Sell_1hr_A4(new_data_df):
    features = ['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'PCRv Up4', 'PCRv Down4',
       'ITM PCRv Up4', 'ITM PCRv Down4', 'RSI14', 'AwesomeOsc5_34',
       'AwesomeOsc']
    model_filename = f"{base_dir}/_1hr_A4/target_down.joblib"
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
#
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
#
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
#
#
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
#
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
#
#
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


# In the get_buy_signal() function, it loads the model (trained_model_target_up.joblib) specifically trained for the "buy" signal (target_up). It accepts predictor inputs as a list (predictors) and assumes you will provide the corresponding values for the predictors. It then creates a DataFrame (new_data_df) with the new data and makes predictions using the loaded model. The predictions are returned as the buy signal.
#
# Similarly, the get_sell_signal() function loads the model (trained_model_target_down.joblib) specifically trained for the "sell" signal (target_down). It follows the same process as the get_buy_signal() function to make predictions based on the provided predictor inputs and returns the sell signal.
#
# Note: Make sure you have trained and saved the models separately for each target before using these functions, and replace <value> with the actual values for the predictors you want to use.
#
