import os

import joblib
import numpy as np
import torch.nn as nn
import torch
import pandas as pd
from joblib import load
class BinaryClassificationNNwithDropout(nn.Module):
    def __init__(self, input_dim, num_hidden_units, dropout_rate):
        super(BinaryClassificationNNwithDropout, self).__init__()
        self.layer1 = nn.Linear(input_dim, num_hidden_units)
        self.layer2 = nn.Linear(num_hidden_units, int(num_hidden_units/2))
        self.layer3 = nn.Linear(int(num_hidden_units/2), int(num_hidden_units/4))
        self.output_layer = nn.Linear(int(num_hidden_units/4), 1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate) # Dropout layer
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.dropout(x) # Apply dropout after the activation
        x = self.activation(self.layer2(x))
        x = self.dropout(x) # Apply dropout after the activation
        x = self.activation(self.layer3(x))
        x = self.dropout(x) # Apply dropout after the activation
        x = self.sigmoid(self.output_layer(x))
        return x

class RegressionModel(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate):
        super(RegressionModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.dropout3 = nn.Dropout(dropout_rate)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.dropout4 = nn.Dropout(dropout_rate)
        self.fc5 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)
        x = torch.relu(self.fc4(x))
        x = self.dropout4(x)
        x = self.fc5(x)
        return x  # Reshape predictions to 1D tensor




def load_model_and_predict(new_data_df, model_folder_name):
    base_dir = "..\Trained_Models"

    model_folder = os.path.join(base_dir, model_folder_name)
    model_filename_up = os.path.join(model_folder, "target_up.pt")
    info_filename = os.path.join(model_folder, "info.pkl")
    info = joblib.load(info_filename)
    feature_columns=['Current SP % Change(LAC)', 'Maximum Pain', 'Bonsai Ratio',
       'Bonsai Ratio 2', 'B1/B2', 'B2/B1', 'PCR-Vol', 'PCR-OI',
       'PCRv @CP Strike', 'PCRoi @CP Strike', 'PCRv Up1', 'PCRv Up2',
       'PCRv Up3', 'PCRv Up4', 'PCRv Down1', 'PCRv Down2', 'PCRv Down3',
       'PCRv Down4', 'PCRoi Up1', 'PCRoi Up2', 'PCRoi Up3', 'PCRoi Up4',
       'PCRoi Down1', 'PCRoi Down2', 'PCRoi Down3', 'PCRoi Down4',
       'ITM PCR-Vol', 'ITM PCR-OI', 'ITM PCRv Up1', 'ITM PCRv Up2',
       'ITM PCRv Up3', 'ITM PCRv Up4', 'ITM PCRv Down1', 'ITM PCRv Down2',
       'ITM PCRv Down3', 'ITM PCRv Down4', 'ITM PCRoi Up1', 'ITM PCRoi Up2',
       'ITM PCRoi Up3', 'ITM PCRoi Up4', 'ITM PCRoi Down1', 'ITM PCRoi Down2',
       'ITM PCRoi Down3', 'ITM PCRoi Down4', 'ITM OI', 'Total OI',
       'ITM Contracts %', 'Net_IV', 'Net ITM IV', 'Net IV MP', 'Net IV LAC',
       'NIV Current Strike', 'NIV 1Higher Strike', 'NIV 1Lower Strike',
       'NIV 2Higher Strike', 'NIV 2Lower Strike', 'NIV 3Higher Strike',
       'NIV 3Lower Strike', 'NIV 4Higher Strike', 'NIV 4Lower Strike',
       'NIV highers(-)lowers1-2', 'NIV highers(-)lowers1-4',
       'NIV 1-2 % from mean', 'NIV 1-4 % from mean', 'Net_IV/OI',
       'Net ITM_IV/ITM_OI', 'Closest Strike to CP', 'RSI', 'AwesomeOsc',
       'RSI14', 'RSI2', 'AwesomeOsc5_34']
    X_scaler = info['X_scaler']
    y_up_scaler = info['y_up_scaler']
    columns_to_keep = ["LastTradeTime", "Current Stock Price"]

    # Select the same feature columns as used during training
    df = new_data_df.copy()[feature_columns]

    # Drop rows with missing values in specified features
    df.dropna(subset=feature_columns, inplace=True)

    # Clip the values in the DataFrame to avoid issues with scaling
    threshold = 1e10
    df[feature_columns] = np.clip(df[feature_columns], -threshold, threshold)

    # Transform the new data using the same X_scaler
    new_data_scaled = X_scaler.transform(df)

    # Convert the scaled data to a PyTorch tensor
    new_data_tensor = torch.tensor(new_data_scaled, dtype=torch.float32)

    # Load the model
    model = RegressionModel(input_size=len(feature_columns), hidden_size=32, dropout_rate=0.3)

    # Load the model weights
    model.load_state_dict(torch.load(model_filename_up))

    # Make predictions
    model.eval()

    with torch.no_grad():
        predicted_values_scaled = model(new_data_tensor).detach().numpy()

    # Inverse transform the predicted values to get the final predictions
    # predicted_values = y_up_scaler.inverse_transform(predicted_values_scaled)"""this would give the predicted priceo fthe stock i trained on. rather than  a 0-1 value"""
        # Apply Min-Max scaling to the predictions
        min_value = np.min(predicted_values_scaled)
        max_value = np.max(predicted_values_scaled)
        scaled_predictions = (predicted_values_scaled - min_value) / (max_value - min_value)

    # Create a new DataFrame with the predictions and align it with the original DataFrame
    prediction_df = pd.DataFrame(scaled_predictions, index=df.index, columns=["Predictions"])
    result_df = new_data_df.join(prediction_df)

    # Return the predictions along with other parameters
    return result_df["Predictions"], 0.6, 0.3
print("Current working directory:", os.getcwd())

# Get the absolute path to the current Python script
script_dir = os.path.dirname(os.path.abspath(__file__))
# Define the relative path to the data file from the script directory
relative_data_path = "..\\algooutput_TSLA_historical_multiday_min.csv"
# relative_data_path = "..\..\data\historical_multiday_minute_DF\ROKU_historical_multiday_min.csv"
# Construct the absolute file path
data_file_path = os.path.join(script_dir, relative_data_path)
# Now you can use the `data_file_path` variable to load the data
ml_dataframe = pd.read_csv(data_file_path)
new_data = pd.read_csv(data_file_path)  # Replace ... with the new data in the format of your DataFrame
print(new_data.columns[-50:] )

columnstokeep = new_data[["LastTradeTime", "Current Stock Price"]]

result = load_model_and_predict(new_data,'_1hr_ptnnSPYA1')
print(new_data)
if isinstance(result, tuple):
    new_data['_1hr_ptnnSPYA1'], takeprofit, stoploss = result
else:
    new_data['_1hr_ptnnSPYA1'], takeprofit, stoploss = result, None, None
    print(new_data['_1hr_ptnnSPYA1'])
new_data[["LastTradeTime", "Current Stock Price"]] = columnstokeep[["LastTradeTime", "Current Stock Price"]]
# df_filtered.to_csv(f"algooutput_{filename}")
new_data.to_csv(f"algooutput_ROKU.csv")