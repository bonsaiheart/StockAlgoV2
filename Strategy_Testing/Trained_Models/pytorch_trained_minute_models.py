import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
base_dir = os.path.dirname(__file__)
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

class BinaryClassificationNN(nn.Module):
    def __init__(self, input_dim,num_hidden_units):
        super(BinaryClassificationNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, num_hidden_units)
        self.layer2 = nn.Linear(num_hidden_units, num_hidden_units)
        self.layer3 = nn.Linear(num_hidden_units, num_hidden_units)
        self.output_layer = nn.Linear(num_hidden_units, 1)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.activation(self.layer3(x))
        x = self.sigmoid(self.output_layer(x))
        return x

def Buy_3hr_PTminClassSPYA1(new_data_df):
    checkpoint = torch.load(f'{base_dir}/_3hr_ptclassA1/target_up.pth')
    features = checkpoint['features']
    input_dim = checkpoint['input_dim']
    num_hidden_units = checkpoint['num_hidden_units']
    loaded_model = BinaryClassificationNNwithDropout(input_dim, num_hidden_units)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()  # Set the model to evaluation mode
    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    threshold = 1e10
    tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
    # Convert DataFrame to a PyTorch tensor
    input_tensor = torch.tensor(tempdf[features].values, dtype=torch.float32)
    # Pass the tensor through the model to get predictions
    predictions = loaded_model(input_tensor)
    # Convrt predictions to a NumPy array
    predictions_numpy = predictions.detach().numpy()
    # Create a new Series with the predictions and align it with the original DataFrame
    prediction_series = pd.Series(predictions_numpy.flatten(), index=tempdf.index)
    result = new_data_df.copy()  # Create a copy of the original DataFrame
    result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
    result.loc[
        prediction_series.index, "Predictions"] = prediction_series.values  # Assign predictions to corresponding rows
    return result["Predictions"]
def Buy_2hr_ptminclassSPYA2(new_data_df):
    checkpoint = torch.load(f'{base_dir}/_2hr_ptminclassSPYA2/target_up.pth')
    features =  ['Bonsai Ratio','Bonsai Ratio 2','PCRoi Up1','ITM PCRoi Up1', 'RSI14','AwesomeOsc5_34', 'Net IV LAC']
    input_dim = checkpoint['input_dim']
    num_hidden_units = checkpoint['num_hidden_units']
    loaded_model = BinaryClassificationNN(input_dim, num_hidden_units)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()  # Set the model to evaluation mode
    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    threshold = 1e10
    tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
    # Convert DataFrame to a PyTorch tensor
    input_tensor = torch.tensor(tempdf[features].values, dtype=torch.float32)
    # Pass the tensor through the model to get predictions
    predictions = loaded_model(input_tensor)
    # Convrt predictions to a NumPy array
    predictions_numpy = predictions.detach().numpy()
    # Create a new Series with the predictions and align it with the original DataFrame
    prediction_series = pd.Series(predictions_numpy.flatten(), index=tempdf.index)
    result = new_data_df.copy()  # Create a copy of the original DataFrame
    result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
    result.loc[
        prediction_series.index, "Predictions"] = prediction_series.values  # Assign predictions to corresponding rows
    return result["Predictions"]
def Buy_2hr_ptminclassSPYA1(new_data_df):
    features =  [
    "Bonsai Ratio",
    "Bonsai Ratio 2",
    "B1/B2"]
    checkpoint = torch.load(f'{base_dir}/_2hr_ptminclassSPYA1/target_up.pth', map_location=torch.device('cpu'))
    input_dim = checkpoint['input_dim']
    num_hidden_units = checkpoint['num_hidden_units']
    loaded_model = BinaryClassificationNN(input_dim, num_hidden_units)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()  # Set the model to evaluation mode
    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    threshold = 1e10
    tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
    # Convert DataFrame to a PyTorch tensor
    input_tensor = torch.tensor(tempdf[features].values, dtype=torch.float32)
    # Pass the tensor through the model to get predictions
    predictions = loaded_model(input_tensor)
    # Convrt predictions to a NumPy array
    predictions_numpy = predictions.detach().numpy()
    # Create a new Series with the predictions and align it with the original DataFrame
    prediction_series = pd.Series(predictions_numpy.flatten(), index=tempdf.index)
    result = new_data_df.copy()  # Create a copy of the original DataFrame
    result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
    result.loc[
        prediction_series.index, "Predictions"] = prediction_series.values  # Assign predictions to corresponding rows
    return result["Predictions"]
def Buy_1hr_ptmin1A1(new_data_df):
    print("Making PT Preditction")
    features =  [
    "Bonsai Ratio",
    "Bonsai Ratio 2",
    "B1/B2"]

    checkpoint = torch.load(f'{base_dir}/_1hr_ptmin1A1/target_up.pth', map_location=torch.device('cpu'))
    input_dim = checkpoint['input_dim']
    num_hidden_units = checkpoint['num_hidden_units']
    loaded_model = BinaryClassificationNN(input_dim, num_hidden_units)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])

    loaded_model.eval()  # Set the model to evaluation mode

    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    threshold = 1e10
    tempdf[features] = np.clip(tempdf[features], -threshold, threshold)

    # Convert DataFrame to a PyTorch tensor
    input_tensor = torch.tensor(tempdf[features].values, dtype=torch.float32)

    # Pass the tensor through the model to get predictions
    predictions = loaded_model(input_tensor)

    # Convert predictions to a NumPy array
    predictions_numpy = predictions.detach().numpy()

    # Create a new Series with the predictions and align it with the original DataFrame
    prediction_series = pd.Series(predictions_numpy.flatten(), index=tempdf.index)
    result = new_data_df.copy()  # Create a copy of the original DataFrame
    result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
    result.loc[
        prediction_series.index, "Predictions"] = prediction_series.values  # Assign predictions to corresponding rows
    return (result["Predictions"],.6,.3)
# current_directory = os.getcwd()
# print("Current Directory:", current_directory)
# df = pd.read_csv("../../data/DailyMinutes/SPY/SPY_230721.csv")
# predictions, _, _ = Buy_1hr_ptmin1A1(df)  # Unpack the tuple returned by Buy_1hr_ptmin1A1
# df['Predictions'] = predictions
# df.to_csv("testing123.csv")

