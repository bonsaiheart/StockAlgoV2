import inspect
import os
import pandas as pd
import Trained_Models.trained_minute_models  # Import your module


def get_model_names(module):
    model_names = []
    for name, obj in inspect.getmembers(module):
        if inspect.isfunction(obj):
            model_names.append(name)
    return model_names


module_name = Trained_Models.trained_minute_models  # Provide the correct module name
model_names = get_model_names(module_name)
print(model_names)


def apply_predictions_to_df(model_names, df, filename):
    df.dropna(axis=1, how="all", inplace=True)

    # Columns to keep
    columns_to_keep = ["LastTradeTime", "Current SP % Change(LAC)"] + model_names

    # Filter the DataFrame to keep only the desired columns

    for model_name in model_names:
        model_func = getattr(Trained_Models.trained_minute_models, model_name)
        prediction = model_func(df)
        df[model_name] = prediction

    df_filtered = df[columns_to_keep]
    df_filtered.to_csv(f"algooutput_{filename}")


dir = "../data/historical_multiday_minute_DF"
for filename in sorted(os.listdir(dir)):
    filepath = os.path.join(dir, filename)

    if filename.endswith(".csv"):
        df = pd.read_csv(filepath)
        apply_predictions_to_df(model_names, df, filename)

# threshold = 1e10  # Define a threshold value to limit the range

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