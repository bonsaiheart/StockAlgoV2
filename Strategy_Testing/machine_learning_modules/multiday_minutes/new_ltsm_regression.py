import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

import numpy as np
import joblib
import os

DF_filename = r"../../../data/historical_multiday_minute_DF/Copy of SPY_historical_multiday_min.csv"

ml_dataframe = pd.read_csv(DF_filename)
print(ml_dataframe.columns)

Chosen_Predictor = [
    "Bonsai Ratio",

    "Bonsai Ratio 2",
    "B1/B2", 'ITM PCR-Vol',
    "PCRv Up3", "PCRv Up2",
    "PCRv Down3", "PCRv Down2",
    "ITM PCRv Up3", 'Net_IV', 'Net ITM IV',
    "ITM PCRv Down3",
    "ITM PCRv Up4", "ITM PCRv Down2", "ITM PCRv Up2",
    "ITM PCRv Down4",
    "RSI14",
    "AwesomeOsc5_34",
    "RSI",
]

ml_dataframe.dropna(subset=Chosen_Predictor, inplace=True)
length = ml_dataframe.shape[0]
print("Length of ml_dataframe:", length)

ml_dataframe["Target_Down"] = ml_dataframe["Current Stock Price"].shift(-30)
ml_dataframe["Target_Up"] = ml_dataframe["Current Stock Price"].shift(-30)


ml_dataframe.dropna(subset=["Target_Up", "Target_Down"], inplace=True)
y_up = ml_dataframe["Target_Up"]
y_down = ml_dataframe["Target_Down"]
X = ml_dataframe[Chosen_Predictor]
X.reset_index(drop=True, inplace=True)

X_train, X_test, y_up_train, y_up_test, y_down_train, y_down_test = train_test_split(
    X, y_up, y_down, test_size=0.2, random_state=None
)

# Define your neural network model using TensorFlow
model_up_nn = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])
model_down_nn = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1)
])

# Compile the models with Mean Squared Error loss for regression task
model_up_nn.compile(optimizer=Adam(), loss='mean_squared_error')
model_down_nn.compile(optimizer=Adam(), loss='mean_squared_error')

# Train the models on your data
model_up_nn.fit(X_train, y_up_train, epochs=50, batch_size=32)
model_down_nn.fit(X_train, y_down_train, epochs=50, batch_size=32)

# Use the trained models for prediction
predicted_values_up = model_up_nn.predict(X_test)
predicted_values_down = model_down_nn.predict(X_test)

# Calculate MSE for both models

mse_up = mean_squared_error(y_up_test, predicted_values_up)
mae_up = mean_absolute_error(y_up_test, predicted_values_up)
r2_up = r2_score(y_up_test, predicted_values_up)

mse_down = mean_squared_error(y_down_test, predicted_values_down)
mae_down = mean_absolute_error(y_down_test, predicted_values_down)
r2_down = r2_score(y_down_test, predicted_values_down)

print("MSE for Up Model:", mse_up)
print("MAE for Up Model:", mae_up)
print("R^2 for Up Model:", r2_up)

print("MSE for Down Model:", mse_down)
print("MAE for Down Model:", mae_down)
print("R^2 for Down Model:", r2_down)
# Save the models using joblib
input_val = input("Would you like to save these models? y/n: ").upper()
if input_val == "Y":
    model_summary = input("Save this set of models as: ")
    model_directory = os.path.join("../../Trained_Models", f"{model_summary}")
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    model_filename_up = os.path.join(model_directory, "target_up")
    model_filename_down = os.path.join(model_directory, "target_down")
    model_up_nn.save(model_filename_up)
    model_down_nn.save(model_filename_down)

with open(f"../../Trained_Models/{model_summary}/info.txt", "w") as info_txt:
    info_txt.write("This file contains information about the model.\n\n")
    info_txt.write(
        f"File analyzed: {DF_filename}\nMSE for Up Model: {mse_up}\nMSE for Down Model: {mse_down}\n\MAE for Up Model: {mae_up}\nMAE for Down Model: {mae_down}\n\nR^2 for Up Model: {r2_up}\nR^2 for Down Model: {r2_down}"
    )
    info_txt.write(
        f"Predictors: {Chosen_Predictor}\n"
    )
