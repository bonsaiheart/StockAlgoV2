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
for filename in os.listdir(dir):
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
