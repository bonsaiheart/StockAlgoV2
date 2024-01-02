import inspect
import os

import pandas as pd

import Trained_Models  # Import your modules here
import Trained_Models.trained_minute_models  # Import your module
from Trained_Models import trained_minute_models
from Trained_Models import pytorch_trained_minute_models


def get_model_names(module):
    model_names = []
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj) and name != "load_model":  # Exclude 'load_model'
            model_names.append(name)
    return model_names


# List of module names
module_names = [
    # trained_minute_models,
    pytorch_trained_minute_models,
]


def forward_rolling_min(series, window):
    return series.shift(-window + 1).rolling(window=window, min_periods=1).min()


def forward_rolling_max(series, window):
    return series.shift(-window + 1).rolling(window=window, min_periods=1).max()


# Function to apply predictions to DataFrame
def apply_predictions_to_df(module_name, df, filename):
    columns_to_keep = [
        "LastTradeTime",
        "Current Stock Price",
        "1HR_later_Price",
        "2HR_later_Price",
        "3HR_later_Price",
        "4HR_later_Price",
        "5HR_later_Price",
    ]
    df["4hourslater% change"] = (
        (df["Current Stock Price"] - df["Current Stock Price"].shift(-240))
        / df["Current Stock Price"].shift(-240)
    ) * 100
    df["1HR_later_Price"] = df["Current Stock Price"].shift(-60)
    df["2HR_later_Price"] = df["Current Stock Price"].shift(-120)
    df["3HR_later_Price"] = df["Current Stock Price"].shift(-180)
    df["4HR_later_Price"] = df["Current Stock Price"].shift(-240)
    df["5HR_later_Price"] = df["Current Stock Price"].shift(-300)
    # Calculate future high and low prices
    timeframes = [20, 60, 120, 180, 240]  # 20min, 1hr, 2hr, 3hr, 4hr in minutes
    for t in timeframes:
        df[f"High_{t}min"] = forward_rolling_max(df["Current Stock Price"], t)
        df[f"Low_{t}min"] = forward_rolling_min(df["Current Stock Price"], t)

    # Keep only the specified columns

    for module in module_name:
        model_names = get_model_names(module)
        for model_name in model_names:
            # print(model_namesme)
            # if model_name == "Buy_20min_05pct_ptclass_A1" or model_name=='Sell_20min_05pct_ptclass_S1':
            print(f"Applying model: {model_name}")
            model_func = getattr(module, model_name)
            result = model_func(df)

            if isinstance(result, tuple):
                (
                    df[model_name],
                    stock_takeprofit,
                    stock_stoploss,
                    option_takeprofit,
                    option_stoploss,
                ) = result
            else:
                df[model_name] = result

            columns_to_keep.append(model_name)
    df = df[
        columns_to_keep
        + [f"High_{t}min" for t in timeframes]
        + [f"Low_{t}min" for t in timeframes]
    ]

    # df = df[columns_to_keep]
    df.to_csv(f"newnewnew_algooutput_{filename}")


# Directory containing CSV files
dir = "../data/historical_multiday_minute_DF"
prefixes_to_match = ["SPY", "GOOG", "TSLA"]  # Add your prefixes here

for filename in os.listdir(dir):
    filepath = os.path.join(dir, filename)

    # Check if the filename ends with ".csv" and starts with any of the specified prefixes
    if filename.endswith(".csv") and any(
        filename.startswith(prefix) for prefix in prefixes_to_match
    ):
        df = pd.read_csv(filepath)
        apply_predictions_to_df(module_names, df, filename)
# threshold = 1e10
# Define a threshold value to limit the range

# for feature in features:
#     feature_values = prep_df[feature].values.astype(float)
#     feature_values = np.clip(feature_values, -threshold, threshold)
#     prep_df.loc[:, feature] = feature_values
#    predictions_df = pd.DataFrame(predictions, index=prep_df.index)
#     print(predictions_df)
#

# threshold = 1e10
# modified_df = prep_df.copy()
#
# for feature in features:
#     feature_values = modified_df[feature].values
#     feature_values = feature_values.astype(float)
#     feature_values = np.clip(feature_values, -threshold, threshold)
#     modified_df[feature] = feature_values

# TODO added these before i lose them forever.
# dailyminutes_df["B1/B2"] = (dailyminutes_df["B1/B2"] > 1.15).astype(int)
#
# dailyminutes_df["B1/B2"] = (dailyminutes_df["B1/B2"] < 0.01).astype(int)
#
# dailyminutes_df["NIV 1-2 % from mean & NIV 1-4 % from mean"] = (
#     (dailyminutes_df["NIV 1-2 % from mean"] < -100) & (dailyminutes_df["NIV 1-4 % from mean"] < -200)
# ).astype(int)
#
# dailyminutes_df["NIV 1-2 % from mean & NIV 1-4 % from mean"] = (
#     (dailyminutes_df["NIV 1-2 % from mean"] > 100) & (dailyminutes_df["NIV 1-4 % from mean"] > 200)
# ).astype(int)
#
# dailyminutes_df["NIV highers(-)lowers1-4"] = (dailyminutes_df["NIV highers(-)lowers1-4"] < -20).astype(int)
#
# dailyminutes_df["NIV highers(-)lowers1-4"] = (dailyminutes_df["NIV highers(-)lowers1-4"] > 20).astype(int)
#
# dailyminutes_df["ITM PCR-Vol & RSI"] = (
#     (dailyminutes_df["ITM PCR-Vol"] > 1.3) & (dailyminutes_df["RSI"] > 70)
# ).astype(int)
#
# dailyminutes_df["Bonsai Ratio & ITM PCR-Vol & RSI"] = (
#     (dailyminutes_df["Bonsai Ratio"] < 0.8) & (dailyminutes_df["ITM PCR-Vol"] < 0.8) & (dailyminutes_df["RSI"] < 30)
# ).astype(int)
#
# dailyminutes_df["Bonsai Ratio & ITM PCR-Vol & RSI"] = (
#     (dailyminutes_df["Bonsai Ratio"] > 1.5) & (dailyminutes_df["ITM PCR-Vol"] > 1.2) & (dailyminutes_df["RSI"] > 70)
# ).astype(int)
#
# dailyminutes_df["Bonsai Ratio < 0.7 & Net_IV < -50 & Net ITM IV > -41"] = (
#     (dailyminutes_df["Bonsai Ratio"] < 0.7)
#     & (dailyminutes_df["Net_IV"] < -50)
#     & (dailyminutes_df["Net ITM IV"] > -41)
# ).astype(int)
#
# dailyminutes_df[
#     "B2/B1>500 Bonsai Ratio<.0001 ITM PCRv Up2<.01 ITM PCRv Down2<5 NIV 1-2 % from mean>NIV 1-4 % from mean>0"
# ] = int(
#     (dailyminutes_df["B2/B1"].iloc[-1] > 500)
#     and (dailyminutes_df["Bonsai Ratio"].iloc[-1] < 0.0001)
#     and (dailyminutes_df["ITM PCRv Up2"].iloc[-1] < 0.01)
#     and (dailyminutes_df["ITM PCRv Down2"].iloc[-1] < 5)
#     and (dailyminutes_df["NIV 1-2 % from mean"].iloc[-1] > dailyminutes_df["NIV 1-4 % from mean"].iloc[-1] > 0)
# )
#
# # 1.15-(hold until) 0 and <0.0, hold call until .3   (hold them until the b1/b2 doubles/halves?) with conditions to make sure its profitable.
# dailyminutes_df["b1/b2 and rsi"] = int(
#     (dailyminutes_df["B1/B2"].iloc[-1] > 1.15) and (dailyminutes_df["RSI"].iloc[-1] < 30)
# )

# if dailyminutes_df["B1/B2"].iloc[-1] < 0.25 and dailyminutes_df["RSI"].iloc[-1] > 70:
#     send_notifications.email_me_string(
#         "dailyminutes_df['B1/B2'][-1] < 0.25 and dailyminutes_df['RSI'][-1]>77:", "Put", ticker
#     )
