import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# Load the data
DF_filename = "../../../data/historical_multiday_minute_DF/SPY_historical_multiday_min.csv"
ml_dataframe = pd.read_csv(DF_filename)

# Define the chosen predictors
Chosen_Predictor = ['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'B2/B1', 'ITM PCR-Vol', 'ITM PCR-OI', 'ITM PCRv Up2', 'ITM PCRv Down2', 'ITM PCRoi Up2', 'ITM PCRoi Down2', 'RSI', 'AwesomeOsc']

# Preprocess the data
ml_dataframe.dropna(subset=Chosen_Predictor + ["Current SP % Change(LAC)"], inplace=True)
data = ml_dataframe[Chosen_Predictor].values
target_column = ml_dataframe["Current SP % Change(LAC)"].values

# Scale the data and target column
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)
target_scaler = MinMaxScaler(feature_range=(0, 1))
target_scaled = target_scaler.fit_transform(target_column.reshape(-1, 1))

# Split the data into training and testing sets
train_size = int(len(data_scaled) * 0.8)
train_data = data_scaled[:train_size]
test_data = data_scaled[train_size:]
train_target = target_scaled[:train_size].flatten()
test_target = target_scaled[train_size:].flatten()

# Define the sequence length
sequence_length = 10

# Create the input sequences and corresponding labels
X_train, y_train = [], []
for i in range(sequence_length, len(train_data)):
    X_train.append(train_data[i-sequence_length:i])
    y_train.append(train_target[i])

X_test, y_test = [], []
for i in range(sequence_length, len(test_data)):
    X_test.append(test_data[i-sequence_length:i])
    y_test.append(test_target[i])

# Convert the data to numpy arrays
X_train, y_train = np.array(X_train), np.array(y_train)
X_test, y_test = np.array(X_test), np.array(y_test)

# Design the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=50))
model.add(Dense(units=1))  # Output layer with 1 neuron for regression
# Compile and train the model
model.compile(optimizer='adam', loss='mean_squared_error')

early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])

# Make predictions
predicted_train = model.predict(X_train)
predicted_test = model.predict(X_test)

# Inverse transform the predictions
predicted_train = target_scaler.inverse_transform(predicted_train)
predicted_test = target_scaler.inverse_transform(predicted_test)

# Inverse transform the actual values
y_train = target_scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
y_test = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

# Evaluate the model
mse_train = mean_squared_error(y_train, predicted_train)
mae_train = mean_absolute_error(y_train, predicted_train)
mse_test = mean_squared_error(y_test, predicted_test)
mae_test = mean_absolute_error(y_test, predicted_test)
r2_train = r2_score(y_train, predicted_train)
r2_test = r2_score(y_test, predicted_test)

# Print the evaluation metrics
print("Metrics for training set:")
print("Mean Squared Error:", mse_train)
print("Mean Absolute Error:", mae_train)
print("R-squared:", r2_train)
print()
print("Metrics for testing set:")
print("Mean Squared Error:", mse_test)
print("Mean Absolute Error:", mae_test)
print("R-squared:", r2_test)

input_val = input("Would you like to save this model? (y/n): ").upper()
if input_val == "Y":
    model_summary = input("Save this model as: ")
    model_directory = os.path.join("../../Trained_Models", model_summary)
    os.makedirs(model_directory, exist_ok=True)
    model_filename = os.path.join(model_directory, "model.keras")
    model.save(model_filename)

    with open(os.path.join(model_directory, "info.txt"), "w") as info_file:
        info_file.write("This file contains information about the model.\n\n")
        info_file.write(f"Dataset: {DF_filename}\n")
        info_file.write(f"Chosen Predictors: {Chosen_Predictor}\n")
        info_file.write(f"Target Variable: Current SP % Change(LAC)\n")
        info_file.write(f"Sequence Length: {sequence_length}\n\n")
        info_file.write("Evaluation Metrics:\n")
        info_file.write(f"Training Set:\nMean Squared Error: {mse_train}\nMean Absolute Error: {mae_train}\nR-squared: {r2_train}\n")
        info_file.write(f"Testing Set:\nMean Squared Error: {mse_test}\nMean Absolute Error: {mae_test}\nR-squared: {r2_test}\n")
else:
    exit()
