import os
from datetime import datetime
import torch.nn.functional as F

from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler

base_dir = os.path.dirname(__file__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BinaryClassificationNNwithDropout(nn.Module):
    def __init__(self, input_dim, num_hidden_units, dropout_rate):
        super(BinaryClassificationNNwithDropout, self).__init__()
        self.layer1 = nn.Linear(input_dim, num_hidden_units)
        self.layer2 = nn.Linear(num_hidden_units, int(num_hidden_units / 2))
        self.layer3 = nn.Linear(int(num_hidden_units / 2), int(num_hidden_units / 4))
        self.output_layer = nn.Linear(int(num_hidden_units / 4), 1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)  # Dropout layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.dropout(x)  # Apply dropout after the activation
        x = self.activation(self.layer2(x))
        x = self.dropout(x)  # Apply dropout after the activation
        x = self.activation(self.layer3(x))
        x = self.dropout(x)  # Apply dropout after the activation
        x = self.sigmoid(self.output_layer(x))
        return x


class BinaryClassificationNN(nn.Module):
    def __init__(self, input_dim, num_hidden_units):
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


class RegressionNN(nn.Module):
    def __init__(self, input_dim, num_hidden_units, dropout_rate, num_layers=1):
        super(RegressionNN, self).__init__()

        self.layers = nn.ModuleList()  # Create a ModuleList to hold the layers

        # Add the first linear layer
        self.layers.append(nn.Linear(input_dim, num_hidden_units))
        self.layers.append(nn.BatchNorm1d(num_hidden_units))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout_rate))

        # Add intermediate hidden layers if num_layers > 1
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(num_hidden_units, num_hidden_units))
            self.layers.append(nn.BatchNorm1d(num_hidden_units))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_rate))

        # Add the final linear layer
        self.layers.append(nn.Linear(num_hidden_units, 1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DynamicNNwithDropout(nn.Module):
    def __init__(self, input_dim, layers, dropout_rate):
        super(DynamicNNwithDropout, self).__init__()
        self.layers = nn.ModuleList()

        # Create hidden layers
        prev_units = input_dim
        for units in layers:
            self.layers.append(nn.Linear(prev_units, units))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(p=dropout_rate))
            self.layers.append(nn.BatchNorm1d(units))  # Batch normalization
            prev_units = units

        # Output layer
        self.layers.append(nn.Linear(prev_units, 1))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def SPY_2hr_50pct_Down_PTNNclass(new_data_df):
    model_dir = "SPY_2hr_50pct_Down_PTNNclass"
    checkpoint = torch.load(
        f"{base_dir}/{model_dir}/target_up.pth", map_location=torch.device("cpu")
    )
    features = checkpoint["features"]
    # print(features)
    dropout_rate = checkpoint["dropout_rate"]
    input_dim = checkpoint["input_dim"]
    layers = checkpoint["layers"]
    scaler_X = checkpoint["scaler_X"]
    # Initialize the new model architecture
    loaded_model = DynamicNNwithDropout(input_dim, layers, dropout_rate)

    # Load the saved state_dict into the model
    loaded_model.load_state_dict(checkpoint["model_state_dict"])

    loaded_model.eval()  # Set the model to evaluation mode

    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
    tempdf = tempdf[features]
    # tempdf.dropna(inplace=True)  # Drop rows with missing values in specified features
    for col in tempdf.columns:
        max_val = tempdf[col].replace([np.inf, -np.inf], np.nan).max()
        min_val = tempdf[col].replace([np.inf, -np.inf], np.nan).min()
        # Adjust max_val based on its sign
        max_val = max_val * 1.5 if max_val >= 0 else max_val / 1.5
        # Adjust min_val based on its sign
        min_val = min_val * 1.5 if min_val < 0 else min_val / 1.5
        # Apply the same max_val and min_val to training, validation, and test sets
        tempdf[col] = tempdf[col].replace([np.inf, -np.inf], [max_val, min_val])

    tempdf = pd.DataFrame(tempdf.values, columns=features, index=tempdf.index)

    # scale the new data features and generate predictions

    scaled_features = scaler_X.transform(tempdf)
    input_tensor = torch.tensor(scaled_features, dtype=torch.float32)
    predictions = loaded_model(input_tensor)
    predictions_prob = torch.sigmoid(predictions)
    predictions_numpy = predictions_prob.detach().numpy()
    # Create a Series from the predictions with the index aligned to tempdf
    prediction_series = pd.Series(predictions_numpy.flatten(), index=tempdf.index)

    # Create a new DataFrame to hold the complete predictions
    result = new_data_df.copy()
    result["Predictions"] = np.nan  # Initialize with NaNs
    result.loc[
        prediction_series.index, "Predictions"
    ] = prediction_series  # Align predictions with original DataFrame

    return result["Predictions"], 0.5, 0.5, 5, 20


# def Buy_20min_1pctup_ptclass_B1(new_data_df):
#     model_dir = "_20min_1pctup_ptclass_B1"
#
#     checkpoint = torch.load(
#         f"{base_dir}/{model_dir}/target_up.pth", map_location=torch.device("cpu")
#     )
#     features = checkpoint["features"]
#     # print(features)
#     dropout_rate = checkpoint["dropout_rate"]
#     input_dim = checkpoint["input_dim"]
#     layers = checkpoint["layers"]
#     scaler_X = checkpoint["scaler_X"]
#
#     # Initialize the new model architecture
#     loaded_model = DynamicNNwithDropout(input_dim, layers, dropout_rate)
#
#     # Load the saved state_dict into the model
#     loaded_model.load_state_dict(checkpoint["model_state_dict"])
#
#     loaded_model.eval()  # Set the model to evaluation mode
#
#     tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
#     tempdf.dropna(
#         subset=features, inplace=True
#     )  # Drop rows with missing values in specified features
#     tempdf = tempdf[features]
#     for col in tempdf.columns:
#         max_val = tempdf[col].replace([np.inf, -np.inf], np.nan).max()
#         min_val = tempdf[col].replace([np.inf, -np.inf], np.nan).min()
#         # Adjust max_val based on its sign
#         max_val = max_val * 1.5 if max_val >= 0 else max_val / 1.5
#         # Adjust min_val based on its sign
#         min_val = min_val * 1.5 if min_val < 0 else min_val / 1.5
#         # Apply the same max_val and min_val to training, validation, and test sets
#         tempdf[col].replace([np.inf, -np.inf], [max_val, min_val], inplace=True)
#
#     tempdf = pd.DataFrame(tempdf.values, columns=features)
#
#     # scale the new data features
#     scaled_features = scaler_X.transform(tempdf)
#     # Convert DataFrame to a PyTorch tensor
#     input_tensor = torch.tensor(scaled_features, dtype=torch.float32)
#
#     # Pass the tensor through the model to get predictions
#     predictions = loaded_model(input_tensor)
#     predictions_prob = torch.sigmoid(predictions)
#
#     # Convert predictions to a NumPy array
#     predictions_numpy = predictions_prob.detach().numpy()
#
#     # Create a new Series with the predictions and align it with the original DataFrame
#     prediction_series = pd.Series(predictions_numpy.flatten(), index=tempdf.index)
#
#     result = new_data_df.copy()  # Create a copy of the original DataFrame
#     result[
#         "Predictions"
#     ] = np.nan  # Initialize the 'Predictions' column with NaN values
#     result.loc[
#         prediction_series.index, "Predictions"
#     ] = prediction_series.values  # Assign predictions to corresponding rows
#
#     return result["Predictions"], 0.1, 0.1, None, None
#
#
# def Sell_20min_05pctdown_ptclass_S1(new_data_df):
#     model_dir = "_20min_05pctdown_ptclass_S1"
#
#     checkpoint = torch.load(
#         f"{base_dir}/{model_dir}/target_up.pth", map_location=torch.device("cpu")
#     )
#     features = checkpoint["features"]
#     print(features)
#     dropout_rate = checkpoint["dropout_rate"]
#     input_dim = checkpoint["input_dim"]
#     layers = checkpoint["layers"]
#     scaler_X = checkpoint["scaler_X"]
#
#     # Initialize the new model architecture
#     loaded_model = DynamicNNwithDropout(input_dim, layers, dropout_rate)
#
#     # Load the saved state_dict into the model
#     loaded_model.load_state_dict(checkpoint["model_state_dict"])
#
#     loaded_model.eval()  # Set the model to evaluation mode
#
#     tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
#     tempdf.dropna(
#         subset=features, inplace=True
#     )  # Drop rows with missing values in specified features
#     tempdf = tempdf[features]
#     for col in tempdf.columns:
#         max_val = tempdf[col].replace([np.inf, -np.inf], np.nan).max()
#         min_val = tempdf[col].replace([np.inf, -np.inf], np.nan).min()
#         # Adjust max_val based on its sign
#         max_val = max_val * 1.5 if max_val >= 0 else max_val / 1.5
#         # Adjust min_val based on its sign
#         min_val = min_val * 1.5 if min_val < 0 else min_val / 1.5
#         # Apply the same max_val and min_val to training, validation, and test sets
#         tempdf[col].replace([np.inf, -np.inf], [max_val, min_val], inplace=True)
#
#     tempdf = pd.DataFrame(tempdf.values, columns=features)
#
#     # scale the new data features
#     scaled_features = scaler_X.transform(tempdf)
#     # Convert DataFrame to a PyTorch tensor
#     input_tensor = torch.tensor(scaled_features, dtype=torch.float32)
#
#     # Pass the tensor through the model to get predictions
#     predictions = loaded_model(input_tensor)
#     predictions_prob = torch.sigmoid(predictions)
#
#     # Convert predictions to a NumPy array
#     predictions_numpy = predictions_prob.detach().numpy()
#
#     # Create a new Series with the predictions and align it with the original DataFrame
#     prediction_series = pd.Series(predictions_numpy.flatten(), index=tempdf.index)
#
#     result = new_data_df.copy()  # Create a copy of the original DataFrame
#     result[
#         "Predictions"
#     ] = np.nan  # Initialize the 'Predictions' column with NaN values
#     result.loc[
#         prediction_series.index, "Predictions"
#     ] = prediction_series.values  # Assign predictions to corresponding rows
#
#     return result["Predictions"]


def Buy_20min_05pctup_ptclass_B1(new_data_df):
    model_dir = "_20min_05pctup_ptclass_B1"

    checkpoint = torch.load(
        f"{base_dir}/{model_dir}/target_up.pth", map_location=torch.device("cpu")
    )
    features = checkpoint["features"]
    # print(features)
    dropout_rate = checkpoint["dropout_rate"]
    input_dim = checkpoint["input_dim"]
    layers = checkpoint["layers"]
    scaler_X = checkpoint["scaler_X"]

    # Initialize the new model architecture
    loaded_model = DynamicNNwithDropout(input_dim, layers, dropout_rate)

    # Load the saved state_dict into the model
    loaded_model.load_state_dict(checkpoint["model_state_dict"])

    loaded_model.eval()  # Set the model to evaluation mode

    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(
        subset=features, inplace=True
    )  # Drop rows with missing values in specified features
    tempdf = tempdf[features]
    for col in tempdf.columns:
        max_val = tempdf[col].replace([np.inf, -np.inf], np.nan).max()
        min_val = tempdf[col].replace([np.inf, -np.inf], np.nan).min()
        # Adjust max_val based on its sign
        max_val = max_val * 1.5 if max_val >= 0 else max_val / 1.5
        # Adjust min_val based on its sign
        min_val = min_val * 1.5 if min_val < 0 else min_val / 1.5
        # Apply the same max_val and min_val to training, validation, and test sets
        tempdf[col].replace([np.inf, -np.inf], [max_val, min_val], inplace=True)
    tempdf = pd.DataFrame(tempdf.values, columns=features)

    # scale the new data features
    scaled_features = scaler_X.transform(tempdf)
    # Convert DataFrame to a PyTorch tensor
    input_tensor = torch.tensor(scaled_features, dtype=torch.float32)

    # Pass the tensor through the model to get predictions
    predictions = loaded_model(input_tensor)
    predictions_prob = torch.sigmoid(predictions)

    # Convert predictions to a NumPy array
    predictions_numpy = predictions_prob.detach().numpy()

    # Create a new Series with the predictions and align it with the original DataFrame
    prediction_series = pd.Series(predictions_numpy.flatten(), index=tempdf.index)

    result = new_data_df.copy()  # Create a copy of the original DataFrame
    result[
        "Predictions"
    ] = np.nan  # Initialize the 'Predictions' column with NaN values
    result.loc[
        prediction_series.index, "Predictions"
    ] = prediction_series.values  # Assign predictions to corresponding rows

    return result["Predictions"], 0.1, 0.1, None, None


# def Buy_2hr_ptclassV3_A1(new_data_df):
#     model_dir = "_2hr_ptclassV3_A1"
#
#     checkpoint = torch.load(f'{base_dir}/{model_dir}/target_up.pth', map_location=torch.device('cpu'))
#     features = checkpoint['features']
#     dropout_rate = checkpoint['dropout_rate']
#     input_dim = checkpoint['input_dim']
#     layers = checkpoint['layers']
#     scaler_X = checkpoint['scaler_X']
#
#     # Initialize the new model architecture
#     loaded_model = DynamicNNwithDropout(input_dim, layers, dropout_rate)
#
#     # Load the saved state_dict into the model
#     loaded_model.load_state_dict(checkpoint['model_state_dict'])
#
#     loaded_model.eval()  # Set the model to evaluation mode
#
#     tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
#     tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
#     tempdf = tempdf[features]
#     for col in tempdf.columns:
#         max_val = tempdf[col].replace([np.inf, -np.inf], np.nan).max()
#         min_val = tempdf[col].replace([np.inf, -np.inf], np.nan).min()
#         # Adjust max_val based on its sign
#         max_val = max_val * 1.5 if max_val >= 0 else max_val / 1.5
#         # Adjust min_val based on its sign
#         min_val = min_val * 1.5 if min_val < 0 else min_val / 1.5
#         # Apply the same max_val and min_val to training, validation, and test sets
#         tempdf[col].replace([np.inf, -np.inf], [max_val, min_val], inplace=True)
#     tempdf = pd.DataFrame(tempdf.values, columns=features)
#
#     #scale the new data features
#     scaled_features = scaler_X.transform(tempdf)  # Using the transform method
#
#     # Convert DataFrame to a PyTorch tensor
#     input_tensor = torch.tensor(scaled_features, dtype=torch.float32)
#
#     # Pass the tensor through the model to get predictions
#     predictions = loaded_model(input_tensor)
#     predictions_prob = torch.sigmoid(predictions)
#
#     # Convert predictions to a NumPy array
#     predictions_numpy = predictions_prob.detach().numpy()
#
#     # Create a new Series with the predictions and align it with the original DataFrame
#     prediction_series = pd.Series(predictions_numpy.flatten(), index=tempdf.index)
#
#     result = new_data_df.copy()  # Create a copy of the original DataFrame
#     result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
#     result.loc[
#         prediction_series.index, "Predictions"] = prediction_series.values  # Assign predictions to corresponding rows
#
#     return result["Predictions"]
# def Buy_4hr_ptclassA100(new_data_df):
#     checkpoint = torch.load(f'{base_dir}/_4hr_ptclassA100/target_up.pth', map_location=torch.device('cpu'))
#     features = checkpoint['features']
#     dropout_rate = checkpoint['dropout_rate']
#     input_dim = checkpoint['input_dim']
#     layers = checkpoint['layers']
#     scaler_X = checkpoint['scaler_X']
#
#     # Initialize the new model architecture
#     loaded_model = DynamicNNwithDropout(input_dim, layers, dropout_rate)
#
#     # Load the saved state_dict into the model
#     loaded_model.load_state_dict(checkpoint['model_state_dict'])
#
#     loaded_model.eval()  # Set the model to evaluation mode
#
#     tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
#     tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
#     tempdf = tempdf[features]
#     for col in tempdf.columns:
#         max_val = tempdf[col].replace([np.inf, -np.inf], np.nan).max()
#         min_val = tempdf[col].replace([np.inf, -np.inf], np.nan).min()
#         # Adjust max_val based on its sign
#         max_val = max_val * 1.5 if max_val >= 0 else max_val / 1.5
#         # Adjust min_val based on its sign
#         min_val = min_val * 1.5 if min_val < 0 else min_val / 1.5
#         # Apply the same max_val and min_val to training, validation, and test sets
#         tempdf[col].replace([np.inf, -np.inf], [max_val, min_val], inplace=True)
#
#     #scale the new data features
#     scaled_features = scaler_X.transform(tempdf.values)  # Using the transform method
#
#     # Convert DataFrame to a PyTorch tensor
#     input_tensor = torch.tensor(scaled_features, dtype=torch.float32)
#
#     # Pass the tensor through the model to get predictions
#     predictions = loaded_model(input_tensor)
#     predictions_prob = torch.sigmoid(predictions)
#
#     # Convert predictions to a NumPy array
#     predictions_numpy = predictions_prob.detach().numpy()
#
#     # Create a new Series with the predictions and align it with the original DataFrame
#     prediction_series = pd.Series(predictions_numpy.flatten(), index=tempdf.index)
#
#     result = new_data_df.copy()  # Create a copy of the original DataFrame
#     result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
#     result.loc[
#         prediction_series.index, "Predictions"] = prediction_series.values  # Assign predictions to corresponding rows
#
#     return result["Predictions"]
# def Buy_4hr_FFNNCVREGA1_SPY_230825(new_data_df):
#     checkpoint = torch.load(f'{base_dir}/_4hr_FFNNCVREGA1_SPY_230825/target_up.pth', map_location=torch.device('cpu'))
#     features = checkpoint['features']
#     scaler_y = checkpoint['scaler_y']    # scaler_X = MinMaxScaler(feature_range=(-1, 1))
#     scaler_X = checkpoint['scaler_X']
#
#     class_name_str = checkpoint['model_class']
#     dropout_rate = checkpoint['dropout_rate']
#     input_dim = checkpoint['input_dim']
#     num_hdden_units = checkpoint['num_hidden_units']
#     num_layers = checkpoint['num_layers']
#     # Get the class object using the class name string
#     model_class = globals().get(class_name_str)
#
#     if model_class is None:
#         raise ValueError(f'Unknown class name: {class_name_str}')
#
#     # Make sure its using right model.
#     loaded_model = model_class(input_dim, num_hidden_units, dropout_rate, num_layers)
#     loaded_model.load_state_dict(checkpoint['model_state_dict'])
#     loaded_model.eval()  # Set the model to evaluation mode
#     tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
#     tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
#     tempdf['LastTradeTime'] = tempdf['LastTradeTime'].apply(
#         lambda x: datetime.strptime(x, '%y%m%d_%H%M'))
#     tempdf['LastTradeTime'] = tempdf['LastTradeTime'].apply(lambda x: x.timestamp())
#     # for col in tempdf.columns:
#     #     finite_max = tempdf.loc[tempdf[col] != np.inf, col].max()
#     #     tempdf.loc[tempdf[col] == np.inf, col] = finite_max
#     tempdf['ExpDate'] = tempdf['ExpDate'].astype(float)
#     threshold = 1e10
#     tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
#     scaled_features = scaler_X.transform(tempdf[features].values)  # Using the transform method
#
#     # Convert DataFrame to a PyTorch tensor
#     input_tensor = torch.tensor(scaled_features, dtype=torch.float32)
#     # Pass the tensor through the model to get predictions
#     predictions = loaded_model(input_tensor)
#
#     # Convert predictions to a NumPy array
#     predictions_numpy = predictions.detach().numpy()
#
#     # Unscale predictions to obtain final predictions in the original scale
#     unscaled_predictions = (predictions_numpy * scaler_y.scale_) + scaler_y.min_
#
#     # Create a new Series with the unscaled predictions and align it with the original DataFrame
#     prediction_series = pd.Series(unscaled_predictions.flatten(), index=tempdf.index)
#     result = new_data_df.copy()
#     result["Predictions"] = np.nan
#     result.loc[prediction_series.index, "Predictions"] = unscaled_predictions
#
#     return result["Predictions"]
# def Buy_4hr_ffTSLA230817F(new_data_df):
#     checkpoint = torch.load(f'{base_dir}/_4hr_ffTSLA230817F/target_up.pth', map_location=torch.device('cpu'))
#     features = checkpoint['features']
#     scaler_X = MinMaxScaler()
#     scaler_X.min_ = checkpoint['scaler_X_min']
#     scaler_X.scale_ = checkpoint['scaler_X_scale']
#
#     scaler_y = MinMaxScaler()
#     scaler_y.min_ = checkpoint['scaler_y_min']
#     scaler_y.scale_ = checkpoint['scaler_y_scale']
#     class_name_str = checkpoint['model_class']
#     dropout_rate = checkpoint['dropout_rate']
#     input_dim = checkpoint['input_dim']
#     num_hidden_units = checkpoint['num_hidden_units']
#     num_layers = checkpoint['num_layers']
#     # Get the class object using the class name string
#     model_class = globals().get(class_name_str)
#
#     if model_class is None:
#         raise ValueError(f'Unknown class name: {class_name_str}')
#
#     # Make sure its using right model.
#     loaded_model = model_class(input_dim, num_hidden_units, dropout_rate, num_layers)
#     loaded_model.load_state_dict(checkpoint['model_state_dict'])
#     loaded_model.eval()  # Set the model to evaluation mode
#     tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
#     tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
#     tempdf['LastTradeTime'] = tempdf['LastTradeTime'].apply(
#         lambda x: datetime.strptime(x, '%y%m%d_%H%M'))
#     tempdf['LastTradeTime'] = tempdf['LastTradeTime'].apply(lambda x: x.timestamp())
#     tempdf['ExpDate'] = tempdf['ExpDate'].astype(float)
#
#
#
#     tempdf[features] = (tempdf[features] - scaler_X.min_) * scaler_X.scale_
#
#     tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
#     threshold = 1e10
#     tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
#     scaler_X = MinMaxScaler(feature_range=(-1, 1))
#     # Fit and transform the scaler to the features
#     scaled_features = scaler_X.fit_transform(tempdf[features].values)
#     # Replace the old values with the scaled values
#     tempdf[features] = scaled_features
#     # Convert DataFrame to a PyTorch tensor
#     input_tensor = torch.tensor(tempdf[features].values, dtype=torch.float32)
#     # Pass the tensor through the model to get predictions
#     predictions = loaded_model(input_tensor)
#     # Convrt predictions to a NumPy array
#     predictions_numpy = predictions.detach().numpy()
#     # Create a new Series with the predictions and align it with the original DataFrame
#     prediction_series = pd.Series(predictions_numpy.flatten(), index=tempdf.index)
#     result = new_data_df.copy()  # Create a copy of the original DataFrame
#     result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
#     result.loc[
#         prediction_series.index, "Predictions"] = prediction_series.values  # Assign predictions to corresponding rows
#     return result["Predictions"]
# def Buy_4hr_ffTSLA230817D(new_data_df):
#     checkpoint = torch.load(f'{base_dir}/_4hr_ffTSLA230817D/target_up.pth', map_location=torch.device('cpu'))
#     features = checkpoint['features']
#     scaler_X = MinMaxScaler()
#     scaler_X.min_ = checkpoint['scaler_X_min']
#     scaler_X.scale_ = checkpoint['scaler_X_scale']
#
#     scaler_y = MinMaxScaler()
#     scaler_y.min_ = checkpoint['scaler_y_min']
#     scaler_y.scale_ = checkpoint['scaler_y_scale']
#     class_name_str = checkpoint['model_class']
#     dropout_rate = checkpoint['dropout_rate']
#     input_dim = checkpoint['input_dim']
#     num_hidden_units = checkpoint['num_hidden_units']
#     num_layers = checkpoint['num_layers']
#     # Get the class object using the class name string
#     model_class = globals().get(class_name_str)
#
#     if model_class is None:
#         raise ValueError(f'Unknown class name: {class_name_str}')
#
#     # Make sure its using right model.
#     loaded_model = model_class(input_dim, num_hidden_units, dropout_rate, num_layers)
#     loaded_model.load_state_dict(checkpoint['model_state_dict'])
#     loaded_model.eval()  # Set the model to evaluation mode
#     tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
#     tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
#     tempdf['LastTradeTime'] = tempdf['LastTradeTime'].apply(
#         lambda x: datetime.strptime(x, '%y%m%d_%H%M'))
#     tempdf['LastTradeTime'] = tempdf['LastTradeTime'].apply(lambda x: x.timestamp())
#     tempdf['ExpDate'] = tempdf['ExpDate'].astype(float)
#
#
#
#     tempdf[features] = (tempdf[features] - scaler_X.min_) * scaler_X.scale_
#
#     tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
#     threshold = 1e10
#     tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
#     scaler_X = MinMaxScaler(feature_range=(-1, 1))
#     # Fit and transform the scaler to the features
#     scaled_features = scaler_X.fit_transform(tempdf[features].values)
#     # Replace the old values with the scaled values
#     tempdf[features] = scaled_features
#     # Convert DataFrame to a PyTorch tensor
#     input_tensor = torch.tensor(tempdf[features].values, dtype=torch.float32)
#     # Pass the tensor through the model to get predictions
#     predictions = loaded_model(input_tensor)
#     # Convrt predictions to a NumPy array
#     predictions_numpy = predictions.detach().numpy()
#     # Create a new Series with the predictions and align it with the original DataFrame
#     prediction_series = pd.Series(predictions_numpy.flatten(), index=tempdf.index)
#     result = new_data_df.copy()  # Create a copy of the original DataFrame
#     result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
#     result.loc[
#         prediction_series.index, "Predictions"] = prediction_series.values  # Assign predictions to corresponding rows
#     return result["Predictions"]
# def Buy_4hr_ffTSLA230817B(new_data_df):
#     checkpoint = torch.load(f'{base_dir}/_4hr_ffTSLA230817B/target_up.pth', map_location=torch.device('cpu'))
#     features = checkpoint['features']
#     scaler_X = MinMaxScaler()
#     scaler_X.min_ = checkpoint['scaler_X_min']
#     scaler_X.scale_ = checkpoint['scaler_X_scale']
#
#     scaler_y = MinMaxScaler()
#     scaler_y.min_ = checkpoint['scaler_y_min']
#     scaler_y.scale_ = checkpoint['scaler_y_scale']
#     class_name_str = checkpoint['model_class']
#     dropout_rate = checkpoint['dropout_rate']
#     input_dim = checkpoint['input_dim']
#     num_hidden_units = checkpoint['num_hidden_units']
#     num_layers = checkpoint['num_layers']
#     # Get the class object using the class name string
#     model_class = globals().get(class_name_str)
#
#     if model_class is None:
#         raise ValueError(f'Unknown class name: {class_name_str}')
#
#     # Make sure its using right model.
#     loaded_model = model_class(input_dim, num_hidden_units, dropout_rate, num_layers)
#     loaded_model.load_state_dict(checkpoint['model_state_dict'])
#     loaded_model.eval()  # Set the model to evaluation mode
#     tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
#     tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
#     tempdf['LastTradeTime'] = tempdf['LastTradeTime'].apply(
#         lambda x: datetime.strptime(x, '%y%m%d_%H%M'))
#     tempdf['LastTradeTime'] = tempdf['LastTradeTime'].apply(lambda x: x.timestamp())
#     tempdf['ExpDate'] = tempdf['ExpDate'].astype(float)
#
#     tempdf[features] = (tempdf[features] - scaler_X.min_) * scaler_X.scale_
#
#     tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
#     threshold = 1e10
#     tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
#     scaler_X = MinMaxScaler(feature_range=(-1, 1))
#     # Fit and transform the scaler to the features
#     scaled_features = scaler_X.fit_transform(tempdf[features].values)
#     # Replace the old values with the scaled values
#     tempdf[features] = scaled_features
#     # Convert DataFrame to a PyTorch tensor
#     input_tensor = torch.tensor(tempdf[features].values, dtype=torch.float32)
#     # Pass the tensor through the model to get predictions
#     predictions = loaded_model(input_tensor)
#     # Convrt predictions to a NumPy array
#     predictions_numpy = predictions.detach().numpy()
#     # Create a new Series with the predictions and align it with the original DataFrame
#     prediction_series = pd.Series(predictions_numpy.flatten(), index=tempdf.index)
#     result = new_data_df.copy()  # Create a copy of the original DataFrame
#     result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
#     result.loc[
#         prediction_series.index, "Predictions"] = prediction_series.values  # Assign predictions to corresponding rows
#     return result["Predictions"]
# def Buy_4hr_ffTSLA230817(new_data_df):
#     checkpoint = torch.load(f'{base_dir}/_4hr_ffTSLA230817/target_up.pth', map_location=torch.device('cpu'))
#     features = checkpoint['features']
#     scaler_X = MinMaxScaler()
#     scaler_X.min_ = checkpoint['scaler_X_min']
#     scaler_X.scale_ = checkpoint['scaler_X_scale']
#
#     scaler_y = MinMaxScaler()
#     scaler_y.min_ = checkpoint['scaler_y_min']
#     scaler_y.scale_ = checkpoint['scaler_y_scale']
#     class_name_str = checkpoint['model_class']
#     dropout_rate = checkpoint['dropout_rate']
#     input_dim = checkpoint['input_dim']
#     num_hidden_units = checkpoint['num_hidden_units']
#     num_layers = checkpoint['num_layers']
#     # Get the class object using the class name string
#     model_class = globals().get(class_name_str)
#
#     if model_class is None:
#         raise ValueError(f'Unknown class name: {class_name_str}')
#
#     # Make sure its using right model.
#     loaded_model = model_class(input_dim, num_hidden_units, dropout_rate, num_layers)
#     loaded_model.load_state_dict(checkpoint['model_state_dict'])
#     loaded_model.eval()  # Set the model to evaluation mode
#     tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
#     tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
#     tempdf['LastTradeTime'] = tempdf['LastTradeTime'].apply(
#         lambda x: datetime.strptime(x, '%y%m%d_%H%M'))
#     tempdf['LastTradeTime'] = tempdf['LastTradeTime'].apply(lambda x: x.timestamp())
#     tempdf['ExpDate'] = tempdf['ExpDate'].astype(float)
#
#
#
#     tempdf[features] = (tempdf[features] - scaler_X.min_) * scaler_X.scale_
#
#     tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
#     threshold = 1e10
#     tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
#     scaler_X = MinMaxScaler(feature_range=(-1, 1))
#     # Fit and transform the scaler to the features
#     scaled_features = scaler_X.fit_transform(tempdf[features].values)
#     # Replace the old values with the scaled values
#     tempdf[features] = scaled_features
#     # Convert DataFrame to a PyTorch tensor
#     input_tensor = torch.tensor(tempdf[features].values, dtype=torch.float32)
#     # Pass the tensor through the model to get predictions
#     predictions = loaded_model(input_tensor)
#     # Convrt predictions to a NumPy array
#     predictions_numpy = predictions.detach().numpy()
#     # Create a new Series with the predictions and align it with the original DataFrame
#     prediction_series = pd.Series(predictions_numpy.flatten(), index=tempdf.index)
#     result = new_data_df.copy()  # Create a copy of the original DataFrame
#     result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
#     result.loc[
#         prediction_series.index, "Predictions"] = prediction_series.values  # Assign predictions to corresponding rows
#     return result["Predictions"]
# def Buy_4hr_ffSPY230805A(new_data_df):
#     checkpoint = torch.load(f'{base_dir}/_4hr_ffSPY230805A/target_up.pth', map_location=torch.device('cpu'))
#     features = checkpoint['features']
#     scaler_X = MinMaxScaler()
#     scaler_X.min_ = checkpoint['scaler_X_min']
#     scaler_X.scale_ = checkpoint['scaler_X_scale']
#
#     scaler_y = MinMaxScaler()
#     scaler_y.min_ = checkpoint['scaler_y_min']
#     scaler_y.scale_ = checkpoint['scaler_y_scale']
#     class_name_str = checkpoint['model_class']
#     dropout_rate = checkpoint['dropout_rate']
#     input_dim = checkpoint['input_dim']
#     num_hidden_units = checkpoint['num_hidden_units']
#     num_layers = checkpoint['num_layers']
#     # Get the class object using the class name string
#     model_class = globals().get(class_name_str)
#
#     if model_class is None:
#         raise ValueError(f'Unknown class name: {class_name_str}')
#
#     # Make sure its using right model.
#     loaded_model = model_class(input_dim, num_hidden_units, dropout_rate, num_layers)
#     loaded_model.load_state_dict(checkpoint['model_state_dict'])
#     loaded_model.eval()  # Set the model to evaluation mode
#     tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
#     tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
#     tempdf['LastTradeTime'] = tempdf['LastTradeTime'].apply(
#         lambda x: datetime.strptime(x, '%y%m%d_%H%M'))
#     tempdf['LastTradeTime'] = tempdf['LastTradeTime'].apply(lambda x: x.timestamp())
#     tempdf['ExpDate'] = tempdf['ExpDate'].astype(float)
#
#
#
#     tempdf[features] = (tempdf[features] - scaler_X.min_) * scaler_X.scale_
#
#     tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
#     threshold = 1e10
#     tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
#     scaler_X = MinMaxScaler(feature_range=(-1, 1))
#     # Fit and transform the scaler to the features
#     scaled_features = scaler_X.fit_transform(tempdf[features].values)
#     # Replace the old values with the scaled values
#     tempdf[features] = scaled_features
#     # Convert DataFrame to a PyTorch tensor
#     input_tensor = torch.tensor(tempdf[features].values, dtype=torch.float32)
#     # Pass the tensor through the model to get predictions
#     predictions = loaded_model(input_tensor)
#     # Convrt predictions to a NumPy array
#     predictions_numpy = predictions.detach().numpy()
#     # Create a new Series with the predictions and align it with the original DataFrame
#     prediction_series = pd.Series(predictions_numpy.flatten(), index=tempdf.index)
#     result = new_data_df.copy()  # Create a copy of the original DataFrame
#     result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
#     result.loc[
#         prediction_series.index, "Predictions"] = prediction_series.values  # Assign predictions to corresponding rows
#     return result["Predictions"]
# def Buy_4hr_ffSPY230805(new_data_df):
#     checkpoint = torch.load(f'{base_dir}/_4hr_ffSPY230805/target_up.pth', map_location=torch.device('cpu'))
#     features = checkpoint['features']
#     # Recreate the scalers
#     # scaler_X = MinMaxScaler()
#     # scaler_X.min_ = checkpoint['scaler_X_min']
#     # scaler_X.scale_ = checkpoint['scaler_X_scale']
#
#     # scaler_y = MinMaxScaler()
#     # scaler_y.min_ = checkpoint['scaler_y_min']
#     # scaler_y.scale_ = checkpoint['scaler_y_scale']
#     class_name_str = checkpoint['model_class']
#     dropout_rate = checkpoint['dropout_rate']
#     input_dim = checkpoint['input_dim']
#     num_hidden_units = checkpoint['num_hidden_units']
#     num_layers = checkpoint['num_layers']
#     # Get the class object using the class name string
#     model_class = globals().get(class_name_str)
#
#     if model_class is None:
#         raise ValueError(f'Unknown class name: {class_name_str}')
#
#     # Make sure its using right model.
#     loaded_model = model_class(input_dim, num_hidden_units, dropout_rate, num_layers)
#     loaded_model.load_state_dict(checkpoint['model_state_dict'])
#     loaded_model.eval()  # Set the model to evaluation mode
#     tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
#     tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
#     tempdf['LastTradeTime'] = tempdf['LastTradeTime'].apply(
#         lambda x: datetime.strptime(x, '%y%m%d_%H%M'))
#     tempdf['LastTradeTime'] = tempdf['LastTradeTime'].apply(lambda x: x.timestamp())
#     tempdf['ExpDate'] = tempdf['ExpDate'].astype(float)
#
#
#     # Assuming new_y_values contains the y values for the new stock
#     new_y_values = tempdf['Current Stock Price'].values.reshape(-1, 1)
#     # Create a new MinMaxScaler instance
#     scaler_y_new = MinMaxScaler(feature_range=(-1, 1))
#     # Fit the scaler to the new y values
#     scaler_y_new.fit_transform(new_y_values)
#
#
#
#     tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
#     threshold = 1e10
#     tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
#     scaler_X = MinMaxScaler(feature_range=(-1, 1))
#     # Fit and transform the scaler to the features
#     scaled_features = scaler_X.fit_transform(tempdf[features].values)
#     # Replace the old values with the scaled values
#     tempdf[features] = scaled_features
#     # Convert DataFrame to a PyTorch tensor
#     input_tensor = torch.tensor(tempdf[features].values, dtype=torch.float32)
#     # Pass the tensor through the model to get predictions
#     predictions = loaded_model(input_tensor)
#     # Convrt predictions to a NumPy array
#     predictions_numpy = predictions.detach().numpy()
#     # Create a new Series with the predictions and align it with the original DataFrame
#     prediction_series = pd.Series(predictions_numpy.flatten(), index=tempdf.index)
#     result = new_data_df.copy()  # Create a copy of the original DataFrame
#     result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
#     result.loc[
#         prediction_series.index, "Predictions"] = prediction_series.values  # Assign predictions to corresponding rows
#     return result["Predictions"]


# df = pd.read_csv('../../data/historical_multiday_minute_DF/Copy of SPY_historical_multiday_min.csv')
# Buy_4hr_ffSPY230805(df)


# def Buy_1hr_FFNNRegSPYA1(new_data_df):
#     checkpoint = torch.load(f'{base_dir}/_1hr_FFNNRegSPYA1/target_up.pth', map_location=torch.device('cpu'))
#     features = checkpoint['features']
#     # class_name_str = checkpoint['class_name']
#     class_name_str = 'RegressionNN'
#     dropout_rate = checkpoint['dropout_rate']
#     input_dim = checkpoint['input_dim']
#     num_hidden_units = checkpoint['num_hidden_units']
#     # num_layers= checkpoint['num_layers']
#     num_layers = 3
#     loaded_model = RegressionNN(input_dim, num_hidden_units, dropout_rate, num_layers)
#     loaded_model.load_state_dict(checkpoint['model_state_dict'])
#     loaded_model.eval()  # Set the model to evaluation mode
#     tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
#     tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
#     threshold = 1e10
#     X = np.clip(tempdf[features], -threshold, threshold)
#     scaler = StandardScaler()
#     X = scaler.fit_transform(X)
#     # Convert DataFrame to a PyTorch tensor
#     input_tensor = torch.tensor(X, dtype=torch.float32)
#
#     # Pass the tensor through the model to get predictions
#     predictions = loaded_model(input_tensor)
#     # Convrt predictions to a NumPy array
#     predictions_numpy = predictions.detach().numpy()
#     # Create a new Series with the predictions and align it with the original DataFrame
#     prediction_series = pd.Series(predictions_numpy.flatten(), index=tempdf.index)
#     result = new_data_df.copy()  # Create a copy of the original DataFrame
#     result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
#     result.loc[
#         prediction_series.index, "Predictions"] = prediction_series.values  # Assign predictions to corresponding rows
#     return result["Predictions"]


# def Buy_1hr_ptminclassSPYA1(new_data_df):
#     checkpoint = torch.load(f'{base_dir}/_1hr_ptminclassSPYA1/target_up.pth', map_location=torch.device('cpu'))
#     features = checkpoint['features']
#     # class_name_str = checkpoint['class_name']
#     class_name_str = 'BinaryClassificationNNwithDropout'
#     dropout_rate = checkpoint['dropout_rate']
#     input_dim = checkpoint['input_dim']
#     num_hidden_units = checkpoint['num_hidden_units']
#     class_mapping = {
#         'BinaryClassificationNNwithDropout': BinaryClassificationNNwithDropout
#         # Add other class mappings if needed
#     }
#
#     # Get the class object using the class name string
#     model_class = class_mapping.get(class_name_str)
#     if model_class is None:
#         raise ValueError(f'Unknown class name: {class_name_str}')
#     # if model_class is None:
#     # if class_name == 'BinaryClassificationNNwithDropout':
#     #     model_class = BinaryClassificationNNwithDropout
#     # # Handle other classes if needed
#     # else:
#     #     raise ValueError(f'Unknown class name: {class_name}')
#     # Make sure its using right model.
#     loaded_model = model_class(input_dim, num_hidden_units, dropout_rate)
#     loaded_model.load_state_dict(checkpoint['model_state_dict'])
#     loaded_model.eval()  # Set the model to evaluation mode
#     tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
#     tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
#     threshold = 1e10
#     tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
#     # Convert DataFrame to a PyTorch tensor
#     input_tensor = torch.tensor(tempdf[features].values, dtype=torch.float32)
#     # Pass the tensor through the model to get predictions
#     predictions = loaded_model(input_tensor)
#     # Convrt predictions to a NumPy array
#     predictions_numpy = predictions.detach().numpy()
#     # Create a new Series with the predictions and align it with the original DataFrame
#     prediction_series = pd.Series(predictions_numpy.flatten(), index=tempdf.index)
#     result = new_data_df.copy()  # Create a copy of the original DataFrame
#     result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
#     result.loc[
#         prediction_series.index, "Predictions"] = prediction_series.values  # Assign predictions to corresponding rows
#     return result["Predictions"]


def Buy_3hr_olderPTminClassSPYA1(new_data_df):
    checkpoint = torch.load(
        f"{base_dir}/_3hr_ptclassA1/target_up.pth", map_location=torch.device("cpu")
    )
    features = checkpoint["features"]
    dropout_rate = 0.05
    input_dim = checkpoint["input_dim"]
    num_hidden_units = checkpoint["num_hidden_units"]
    loaded_model = BinaryClassificationNNwithDropout(
        input_dim, num_hidden_units, dropout_rate
    )
    loaded_model.load_state_dict(checkpoint["model_state_dict"])
    loaded_model.eval()  # Set the model to evaluation mode
    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(
        subset=features, inplace=True
    )  # Drop rows with missing values in specified features
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
    result[
        "Predictions"
    ] = np.nan  # Initialize the 'Predictions' column with NaN values
    result.loc[
        prediction_series.index, "Predictions"
    ] = prediction_series.values  # Assign predictions to corresponding rows
    return result["Predictions"], 0.25, 0.3, 0.5, 0.35


def Buy_3hr_PTminClassSPYA1(new_data_df):
    checkpoint = torch.load(
        f"{base_dir}/_3hr_ptclassA1/target_up.pth", map_location=torch.device("cpu")
    )
    features = checkpoint["features"]
    dropout_rate = 0.05
    input_dim = checkpoint["input_dim"]
    num_hidden_units = checkpoint["num_hidden_units"]
    loaded_model = BinaryClassificationNNwithDropout(
        input_dim, num_hidden_units, dropout_rate
    )
    loaded_model.load_state_dict(checkpoint["model_state_dict"])
    loaded_model.eval()  # Set the model to evaluation mode
    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(
        subset=features, inplace=True
    )  # Drop rows with missing values in specified features
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
    result[
        "Predictions"
    ] = np.nan  # Initialize the 'Predictions' column with NaN values
    result.loc[
        prediction_series.index, "Predictions"
    ] = prediction_series.values  # Assign predictions to corresponding rows
    return result["Predictions"], 0.3, 0.5, 5, 10


# def Buy_2hr_ptminclassSPYA2(new_data_df):
#     checkpoint = torch.load(f'{base_dir}/_2hr_ptminclassSPYA2/target_up.pth', map_location=torch.device('cpu'))
#     features = ['Bonsai Ratio', 'Bonsai Ratio 2', 'PCRoi Up1', 'ITM PCRoi Up1', 'RSI14', 'AwesomeOsc5_34', 'Net IV LAC']
#     input_dim = checkpoint['input_dim']
#     num_hidden_units = checkpoint['num_hidden_units']
#     loaded_model = BinaryClassificationNN(input_dim, num_hidden_units)
#     loaded_model.load_state_dict(checkpoint['model_state_dict'])
#     loaded_model.eval()  # Set the model to evaluation mode
#     tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
#     tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
#     threshold = 1e10
#     tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
#     # Convert DataFrame to a PyTorch tensor
#     input_tensor = torch.tensor(tempdf[features].values, dtype=torch.float32)
#     # Pass the tensor through the model to get predictions
#     predictions = loaded_model(input_tensor)
#     # Convrt predictions to a NumPy array
#     predictions_numpy = predictions.detach().numpy()
#     # Create a new Series with the predictions and align it with the original DataFrame
#     prediction_series = pd.Series(predictions_numpy.flatten(), index=tempdf.index)
#     result = new_data_df.copy()  # Create a copy of the original DataFrame
#     result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
#     result.loc[
#         prediction_series.index, "Predictions"] = prediction_series.values  # Assign predictions to corresponding rows
#     return result["Predictions"]
#
#
# def Buy_2hr_ptminclassSPYA1(new_data_df):
#     features = [
#         "Bonsai Ratio",
#         "Bonsai Ratio 2",
#         "B1/B2"]
#     checkpoint = torch.load(f'{base_dir}/_2hr_ptminclassSPYA1/target_up.pth', map_location=torch.device('cpu'))
#     input_dim = checkpoint['input_dim']
#     num_hidden_units = checkpoint['num_hidden_units']
#     loaded_model = BinaryClassificationNN(input_dim, num_hidden_units)
#     loaded_model.load_state_dict(checkpoint['model_state_dict'])
#     loaded_model.eval()  # Set the model to evaluation mode
#     tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
#     tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
#     threshold = 1e10
#     tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
#     # Convert DataFrame to a PyTorch tensor
#     input_tensor = torch.tensor(tempdf[features].values, dtype=torch.float32)
#     # Pass the tensor through the model to get predictions
#     predictions = loaded_model(input_tensor)
#     # Convrt predictions to a NumPy array
#     predictions_numpy = predictions.detach().numpy()
#     # Create a new Series with the predictions and align it with the original DataFrame
#     prediction_series = pd.Series(predictions_numpy.flatten(), index=tempdf.index)
#     result = new_data_df.copy()  # Create a copy of the original DataFrame
#     result["Predictions"] = np.nan  # Initialize the 'Predictions' column with NaN values
#     result.loc[
#         prediction_series.index, "Predictions"] = prediction_series.values  # Assign predictions to corresponding rows
#     print(result["Predictions"])
#     return result["Predictions"], .5, .5, 10, 10
# # def SPY_2hr_50pct_Down_PTNNclass2(new_data_df):
#     checkpoint = torch.load(f'{base_dir}/fakeand_gay_Up_2hr/target_up.pth', map_location=torch.device('cpu'))
#     features = checkpoint['features']
#     dropout_rate = checkpoint['dropout_rate']
#     input_dim = checkpoint['input_dim']
#     layers = checkpoint['layers']
#     scaler_X = checkpoint['scaler_X']
#
#     loaded_model = DynamicNNwithDropout(input_dim, layers, dropout_rate)
#     loaded_model.load_state_dict(checkpoint['model_state_dict'])
#     loaded_model.eval()
#
#     tempdf = new_data_df.copy()
#     tempdf.dropna(subset=features, inplace=True)
#     tempdf = tempdf[features]
#
#     for col in tempdf.columns:
#         max_val = tempdf[col].replace([np.inf, -np.inf], np.nan).max()
#         min_val = tempdf[col].replace([np.inf, -np.inf], np.nan).min()
#         max_val = max_val * 1.5 if max_val >= 0 else max_val / 1.5
#         min_val = min_val * 1.5 if min_val < 0 else min_val / 1.5
#         tempdf[col].replace([np.inf, -np.inf], [max_val, min_val], inplace=True)
#
#     tempdf = pd.DataFrame(scaler_X.transform(tempdf), columns=features)
#     input_tensor = torch.tensor(tempdf.values, dtype=torch.float32)
#     predictions = loaded_model(input_tensor)
#     predictions_prob = torch.sigmoid(predictions)
#     predictions_numpy = predictions_prob.detach().numpy()
#     prediction_series = pd.Series(predictions_numpy.flatten(), index=tempdf.index)
#
#     result = new_data_df.copy()
#     result["Predictions"] = np.nan
#     result.loc[prediction_series.index, "Predictions"] = prediction_series.values
#     return result["Predictions"], .5, .5, 10, 10


def _3hr_40pt_down_FeatSet2_shuf_exc_test_onlyvalloss(new_data_df):
    checkpoint = torch.load(
        f"{base_dir}/_3hr_40pt_down_FeatSet2_shuf_exc_test_onlyvalloss/target_up.pth",
        map_location=torch.device(device),
    )
    features = checkpoint["features"]
    dropout_rate = checkpoint["dropout_rate"]
    input_dim = checkpoint["input_dim"]
    layers = checkpoint["layers"]
    scaler_X = checkpoint["scaler_X"]

    loaded_model = DynamicNNwithDropout(input_dim, layers, dropout_rate)
    loaded_model.load_state_dict(checkpoint["model_state_dict"])
    loaded_model.eval()

    tempdf = new_data_df.copy()
    tempdf.dropna(subset=features, inplace=True)
    tempdf = tempdf[features]

    very_large_number = 1e15  # Placeholder for positive infinity
    very_small_number = -1e15  # Placeholder for negative infinity

    for col in tempdf.columns:
        # Replace positive and negative infinity with the defined large and small numbers
        tempdf[col] = tempdf[col].replace([np.inf, -np.inf], [very_large_number, very_small_number])

    tempdf = pd.DataFrame(
        scaler_X.transform(tempdf), columns=features, index=tempdf.index
    )
    input_tensor = torch.tensor(tempdf.values, dtype=torch.float32)
    predictions = loaded_model(input_tensor)
    predictions_prob = torch.sigmoid(predictions)
    predictions_numpy = predictions_prob.detach().numpy()
    prediction_series = pd.Series(predictions_numpy.flatten(), index=tempdf.index)

    result = new_data_df.copy()
    result["Predictions"] = np.nan
    result.loc[prediction_series.index, "Predictions"] = prediction_series.values
    return result["Predictions"], 0.3, 0.5, 10, 30

def MSFT_2hr_50pct_Down_PTNNclass(new_data_df):
    checkpoint = torch.load(f'{base_dir}/MSFT_2hr_50pct_Down_PTNNclass/target_up.pth', map_location=torch.device('cpu'))
    features = checkpoint['features']
    dropout_rate = checkpoint['dropout_rate']
    input_dim = checkpoint['input_dim']
    layers = checkpoint['layers']
    scaler_X = checkpoint['scaler_X']

    loaded_model = DynamicNNwithDropout(input_dim, layers, dropout_rate)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()

    tempdf = new_data_df.copy()
    tempdf.dropna(subset=features, inplace=True)
    tempdf = tempdf[features]

    for col in tempdf.columns:
        max_val = tempdf[col].replace([np.inf, -np.inf], np.nan).max()
        min_val = tempdf[col].replace([np.inf, -np.inf], np.nan).min()
        max_val = max_val * 1.5 if max_val >= 0 else max_val / 1.5
        min_val = min_val * 1.5 if min_val < 0 else min_val / 1.5
        tempdf[col].replace([np.inf, -np.inf], [max_val, min_val], inplace=True)

    tempdf = pd.DataFrame(tempdf.values, columns=features, index=tempdf.index)

    # scale the new data features and generate predictions

    scaled_features = scaler_X.transform(tempdf)
    input_tensor = torch.tensor(scaled_features, dtype=torch.float32)
    predictions = loaded_model(input_tensor)
    predictions_prob = torch.sigmoid(predictions)
    predictions_numpy = predictions_prob.detach().numpy()
    prediction_series = pd.Series(predictions_numpy.flatten(), index=tempdf.index)

    result = new_data_df.copy()
    result["Predictions"] = np.nan
    result.loc[prediction_series.index, "Predictions"] = prediction_series
    return result["Predictions"], 0.5, 0.5, 5, 20


    result.loc[prediction_series.index, "Predictions"] = prediction_series
    return result["Predictions"], 0.5, 0.5, 5, 20
def SPY_2hr_50pct_Down_PTNNclass_240124(new_data_df):
    checkpoint = torch.load(f'{base_dir}/SPY_2hr_50pct_Down_PTNNclass_240124/target_up.pth', map_location=torch.device('cpu'))
    features = checkpoint['features']
    dropout_rate = checkpoint['dropout_rate']
    input_dim = checkpoint['input_dim']
    layers = checkpoint['layers']
    scaler_X = checkpoint['scaler_X']

    loaded_model = DynamicNNwithDropout(input_dim, layers, dropout_rate)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()

    tempdf = new_data_df.copy()
    # tempdf.dropna(subset=features, inplace=True)
    tempdf = tempdf[features]

    for col in tempdf.columns:
        max_val = tempdf[col].replace([np.inf, -np.inf], np.nan).max()
        min_val = tempdf[col].replace([np.inf, -np.inf], np.nan).min()
        max_val = max_val * 1.5 if max_val >= 0 else max_val / 1.5
        min_val = min_val * 1.5 if min_val < 0 else min_val / 1.5
        tempdf[col] = tempdf[col].replace([np.inf, -np.inf], [max_val, min_val])

    tempdf = pd.DataFrame(tempdf.values, columns=features, index=tempdf.index)

    # scale the new data features and generate predictions

    scaled_features = scaler_X.transform(tempdf)
    input_tensor = torch.tensor(scaled_features, dtype=torch.float32)
    predictions = loaded_model(input_tensor)
    predictions_prob = torch.sigmoid(predictions)
    predictions_numpy = predictions_prob.detach().numpy()
    prediction_series = pd.Series(predictions_numpy.flatten(), index=tempdf.index)

    result = new_data_df.copy()
    result["Predictions"] = np.nan
    result.loc[prediction_series.index, "Predictions"] = prediction_series
    return result["Predictions"], 0.5, 0.5, 5, 20

def TSLA_ptminclassA1Base_2hr50ptdown_2401290106(new_data_df):
    checkpoint = torch.load(f'{base_dir}/TSLA_ptminclassA1Base_2hr50ptdown_2401290106/target_up.pth', map_location=torch.device('cpu'))
    features = checkpoint['features']
    dropout_rate = checkpoint['dropout_rate']
    input_dim = checkpoint['input_dim']
    layers = checkpoint['layers']
    scaler_X = checkpoint['scaler_X']

    loaded_model = DynamicNNwithDropout(input_dim, layers, dropout_rate)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()

    tempdf = new_data_df.copy()
    # tempdf.dropna(subset=features, inplace=True)
    tempdf = tempdf[features]

    very_large_number = 1e15  # Placeholder for positive infinity
    very_small_number = -1e15  # Placeholder for negative infinity

    for col in tempdf.columns:
        # Replace positive and negative infinity with the defined large and small numbers. this reaplce the 1.5x multiplier logic.
        tempdf[col] = tempdf[col].replace([np.inf, -np.inf], [very_large_number, very_small_number])


    tempdf = pd.DataFrame(tempdf.values, columns=features, index=tempdf.index)

    # scale the new data features and generate predictions

    scaled_features = scaler_X.transform(tempdf)
    input_tensor = torch.tensor(scaled_features, dtype=torch.float32)
    predictions = loaded_model(input_tensor)
    predictions_prob = torch.sigmoid(predictions)
    predictions_numpy = predictions_prob.detach().numpy()
    prediction_series = pd.Series(predictions_numpy.flatten(), index=tempdf.index)

    result = new_data_df.copy()
    result["Predictions"] = np.nan
    result.loc[prediction_series.index, "Predictions"] = prediction_series
    return result["Predictions"], 0.5, 0.5, 5, 25

def MSFT_ptminclassA1Base_2hr50ptdown_2401290107(new_data_df):
    checkpoint = torch.load(f'{base_dir}/MSFT_ptminclassA1Base_2hr50ptdown_2401290107/target_up.pth', map_location=torch.device('cpu'))
    features = checkpoint['features']
    dropout_rate = checkpoint['dropout_rate']
    input_dim = checkpoint['input_dim']
    layers = checkpoint['layers']
    scaler_X = checkpoint['scaler_X']

    loaded_model = DynamicNNwithDropout(input_dim, layers, dropout_rate)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()

    tempdf = new_data_df.copy()
    # tempdf.dropna(subset=features, inplace=True)
    tempdf = tempdf[features]

    very_large_number = 1e15  # Placeholder for positive infinity
    very_small_number = -1e15  # Placeholder for negative infinity

    for col in tempdf.columns:
        # Replace positive and negative infinity with the defined large and small numbers. this reaplce the 1.5x multiplier logic.
        tempdf[col] = tempdf[col].replace([np.inf, -np.inf], [very_large_number, very_small_number])


    tempdf = pd.DataFrame(tempdf.values, columns=features, index=tempdf.index)

    # scale the new data features and generate predictions

    scaled_features = scaler_X.transform(tempdf)
    input_tensor = torch.tensor(scaled_features, dtype=torch.float32)
    predictions = loaded_model(input_tensor)
    predictions_prob = torch.sigmoid(predictions)
    predictions_numpy = predictions_prob.detach().numpy()
    prediction_series = pd.Series(predictions_numpy.flatten(), index=tempdf.index)

    result = new_data_df.copy()
    result["Predictions"] = np.nan
    result.loc[prediction_series.index, "Predictions"] = prediction_series
    return result["Predictions"], 0.5, 0.5, 5, 25

def SPY_ptminclassA1Base_2hr50ptdown_2401290107(new_data_df):
    checkpoint = torch.load(f'{base_dir}/SPY_ptminclassA1Base_2hr50ptdown_2401290107/target_up.pth', map_location=torch.device('cpu'))
    features = checkpoint['features']
    dropout_rate = checkpoint['dropout_rate']
    input_dim = checkpoint['input_dim']
    layers = checkpoint['layers']
    scaler_X = checkpoint['scaler_X']

    loaded_model = DynamicNNwithDropout(input_dim, layers, dropout_rate)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()

    tempdf = new_data_df.copy()
    # tempdf.dropna(subset=features, inplace=True)
    tempdf = tempdf[features]

    very_large_number = 1e15  # Placeholder for positive infinity
    very_small_number = -1e15  # Placeholder for negative infinity

    for col in tempdf.columns:
        # Replace positive and negative infinity with the defined large and small numbers. this reaplce the 1.5x multiplier logic.
        tempdf[col] = tempdf[col].replace([np.inf, -np.inf], [very_large_number, very_small_number])


    tempdf = pd.DataFrame(tempdf.values, columns=features, index=tempdf.index)

    # scale the new data features and generate predictions

    scaled_features = scaler_X.transform(tempdf)
    input_tensor = torch.tensor(scaled_features, dtype=torch.float32)
    predictions = loaded_model(input_tensor)
    predictions_prob = torch.sigmoid(predictions)
    predictions_numpy = predictions_prob.detach().numpy()
    prediction_series = pd.Series(predictions_numpy.flatten(), index=tempdf.index)

    result = new_data_df.copy()
    result["Predictions"] = np.nan
    result.loc[prediction_series.index, "Predictions"] = prediction_series
    return result["Predictions"], 0.5, 0.5, 5, 25
    
def TSLA_ptminclassA1Base_2hr50ptdown_2401292134(new_data_df):
    checkpoint = torch.load(f'{base_dir}/TSLA_ptminclassA1Base_2hr50ptdown_2401292134/target_up.pth', map_location=torch.device('cpu'))
    features = checkpoint['features']
    dropout_rate = checkpoint['dropout_rate']
    input_dim = checkpoint['input_dim']
    layers = checkpoint['layers']
    scaler_X = checkpoint['scaler_X']

    loaded_model = DynamicNNwithDropout(input_dim, layers, dropout_rate)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()

    tempdf = new_data_df.copy()
    # tempdf.dropna(subset=features, inplace=True)
    tempdf = tempdf[features]

    very_large_number = 1e15  # Placeholder for positive infinity
    very_small_number = -1e15  # Placeholder for negative infinity

    for col in tempdf.columns:
        # Replace positive and negative infinity with the defined large and small numbers. this reaplce the 1.5x multiplier logic.
        tempdf[col] = tempdf.replace([np.inf, -np.inf], [very_large_number, very_small_number])


    tempdf = pd.DataFrame(tempdf.values, columns=features, index=tempdf.index)

    # scale the new data features and generate predictions

    scaled_features = scaler_X.transform(tempdf)
    input_tensor = torch.tensor(scaled_features, dtype=torch.float32)
    predictions = loaded_model(input_tensor)
    predictions_prob = torch.sigmoid(predictions)
    predictions_numpy = predictions_prob.detach().numpy()
    prediction_series = pd.Series(predictions_numpy.flatten(), index=tempdf.index)

    result = new_data_df.copy()
    result["Predictions"] = np.nan
    result.loc[prediction_series.index, "Predictions"] = prediction_series
    return result["Predictions"], 0.5, 0.5, 5, 25

def MSFT_ptminclassA1Base_2hr50ptdown_2401292135(new_data_df):
    checkpoint = torch.load(f'{base_dir}/MSFT_ptminclassA1Base_2hr50ptdown_2401292135/target_up.pth', map_location=torch.device('cpu'))
    features = checkpoint['features']
    dropout_rate = checkpoint['dropout_rate']
    input_dim = checkpoint['input_dim']
    layers = checkpoint['layers']
    scaler_X = checkpoint['scaler_X']

    loaded_model = DynamicNNwithDropout(input_dim, layers, dropout_rate)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()

    tempdf = new_data_df.copy()
    # tempdf.dropna(subset=features, inplace=True)
    tempdf = tempdf[features]

    very_large_number = 1e15  # Placeholder for positive infinity
    very_small_number = -1e15  # Placeholder for negative infinity

    for col in tempdf.columns:
        # Replace positive and negative infinity with the defined large and small numbers. this reaplce the 1.5x multiplier logic.
        tempdf[col] = tempdf[col].replace([np.inf, -np.inf], [very_large_number, very_small_number])


    tempdf = pd.DataFrame(tempdf.values, columns=features, index=tempdf.index)

    # scale the new data features and generate predictions

    scaled_features = scaler_X.transform(tempdf)
    input_tensor = torch.tensor(scaled_features, dtype=torch.float32)
    predictions = loaded_model(input_tensor)
    predictions_prob = torch.sigmoid(predictions)
    predictions_numpy = predictions_prob.detach().numpy()
    prediction_series = pd.Series(predictions_numpy.flatten(), index=tempdf.index)

    result = new_data_df.copy()
    result["Predictions"] = np.nan
    result.loc[prediction_series.index, "Predictions"] = prediction_series
    return result["Predictions"], 0.5, 0.5, 5, 20

def SPY_ptminclassA1Base_2hr50ptdown_2401292135(new_data_df):
    checkpoint = torch.load(f'{base_dir}/SPY_ptminclassA1Base_2hr50ptdown_2401292135/target_up.pth', map_location=torch.device('cpu'))
    features = checkpoint['features']
    dropout_rate = checkpoint['dropout_rate']
    input_dim = checkpoint['input_dim']
    layers = checkpoint['layers']
    scaler_X = checkpoint['scaler_X']

    loaded_model = DynamicNNwithDropout(input_dim, layers, dropout_rate)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()

    tempdf = new_data_df.copy()
    # tempdf.dropna(subset=features, inplace=True)
    tempdf = tempdf[features]

    very_large_number = 1e15  # Placeholder for positive infinity
    very_small_number = -1e15  # Placeholder for negative infinity

    for col in tempdf.columns:
        # Replace positive and negative infinity with the defined large and small numbers. this reaplce the 1.5x multiplier logic.
        tempdf[col] = tempdf[col].replace([np.inf, -np.inf], [very_large_number, very_small_number])


    tempdf = pd.DataFrame(tempdf.values, columns=features, index=tempdf.index)

    # scale the new data features and generate predictions

    scaled_features = scaler_X.transform(tempdf)
    input_tensor = torch.tensor(scaled_features, dtype=torch.float32)
    predictions = loaded_model(input_tensor)
    predictions_prob = torch.sigmoid(predictions)
    predictions_numpy = predictions_prob.detach().numpy()
    prediction_series = pd.Series(predictions_numpy.flatten(), index=tempdf.index)

    result = new_data_df.copy()
    result["Predictions"] = np.nan
    result.loc[prediction_series.index, "Predictions"] = prediction_series
    return result["Predictions"], 0.5, 0.5, 5, 20

def TSLA_ptminclassA1Base_2hr50ptdown_2401312239(new_data_df):
    checkpoint = torch.load(f'{base_dir}/TSLA_ptminclassA1Base_2hr50ptdown_2401312239/target_up.pth', map_location=torch.device('cpu'))
    features = checkpoint['features']
    dropout_rate = checkpoint['dropout_rate']
    input_dim = checkpoint['input_dim']
    layers = checkpoint['layers']
    scaler_X = checkpoint['scaler_X']

    loaded_model = DynamicNNwithDropout(input_dim, layers, dropout_rate)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()

    tempdf = new_data_df.copy()
    # tempdf.dropna(subset=features, inplace=True)
    tempdf = tempdf[features]

    very_large_number = 1e15  # Placeholder for positive infinity
    very_small_number = -1e15  # Placeholder for negative infinity

    for col in tempdf.columns:
        # Replace positive and negative infinity with the defined large and small numbers. this reaplce the 1.5x multiplier logic.
        tempdf[col] = tempdf[col].replace([np.inf, -np.inf], [very_large_number, very_small_number])
        

    tempdf = pd.DataFrame(tempdf.values, columns=features, index=tempdf.index)

    # scale the new data features and generate predictions

    scaled_features = scaler_X.transform(tempdf)
    input_tensor = torch.tensor(scaled_features, dtype=torch.float32)
    predictions = loaded_model(input_tensor)
    predictions_prob = torch.sigmoid(predictions)
    predictions_numpy = predictions_prob.detach().numpy()
    prediction_series = pd.Series(predictions_numpy.flatten(), index=tempdf.index)

    result = new_data_df.copy()
    result["Predictions"] = np.nan
    result.loc[prediction_series.index, "Predictions"] = prediction_series
    return result["Predictions"], 0.5, 0.5, 5, 20
    
def MSFT_ptminclassA1Base_2hr50ptdown_2401312239(new_data_df):
    checkpoint = torch.load(f'{base_dir}/MSFT_ptminclassA1Base_2hr50ptdown_2401312239/target_up.pth', map_location=torch.device('cpu'))
    features = checkpoint['features']
    dropout_rate = checkpoint['dropout_rate']
    input_dim = checkpoint['input_dim']
    layers = checkpoint['layers']
    scaler_X = checkpoint['scaler_X']

    loaded_model = DynamicNNwithDropout(input_dim, layers, dropout_rate)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()

    tempdf = new_data_df.copy()
    # tempdf.dropna(subset=features, inplace=True)
    tempdf = tempdf[features]

    very_large_number = 1e15  # Placeholder for positive infinity
    very_small_number = -1e15  # Placeholder for negative infinity

    for col in tempdf.columns:
        # Replace positive and negative infinity with the defined large and small numbers. this reaplce the 1.5x multiplier logic.
        tempdf[col] = tempdf[col].replace([np.inf, -np.inf], [very_large_number, very_small_number])
        

    tempdf = pd.DataFrame(tempdf.values, columns=features, index=tempdf.index)

    # scale the new data features and generate predictions

    scaled_features = scaler_X.transform(tempdf)
    input_tensor = torch.tensor(scaled_features, dtype=torch.float32)
    predictions = loaded_model(input_tensor)
    predictions_prob = torch.sigmoid(predictions)
    predictions_numpy = predictions_prob.detach().numpy()
    prediction_series = pd.Series(predictions_numpy.flatten(), index=tempdf.index)

    result = new_data_df.copy()
    result["Predictions"] = np.nan
    result.loc[prediction_series.index, "Predictions"] = prediction_series
    return result["Predictions"], 0.5, 0.5, 5, 20
    
def SPY_ptminclassA1Base_2hr50ptdown_2402010049(new_data_df):
    checkpoint = torch.load(f'{base_dir}/SPY_ptminclassA1Base_2hr50ptdown_2402010049/target_up.pth', map_location=torch.device('cpu'))
    features = checkpoint['features']
    dropout_rate = checkpoint['dropout_rate']
    input_dim = checkpoint['input_dim']
    layers = checkpoint['layers']
    scaler_X = checkpoint['scaler_X']

    loaded_model = DynamicNNwithDropout(input_dim, layers, dropout_rate)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()

    tempdf = new_data_df.copy()
    # tempdf.dropna(subset=features, inplace=True)
    tempdf = tempdf[features]

    very_large_number = 1e15  # Placeholder for positive infinity
    very_small_number = -1e15  # Placeholder for negative infinity

    for col in tempdf.columns:
        # Replace positive and negative infinity with the defined large and small numbers. this reaplce the 1.5x multiplier logic.
        tempdf[col] = tempdf[col].replace([np.inf, -np.inf], [very_large_number, very_small_number])
        

    tempdf = pd.DataFrame(tempdf.values, columns=features, index=tempdf.index)

    # scale the new data features and generate predictions

    scaled_features = scaler_X.transform(tempdf)
    input_tensor = torch.tensor(scaled_features, dtype=torch.float32)
    predictions = loaded_model(input_tensor)
    predictions_prob = torch.sigmoid(predictions)
    predictions_numpy = predictions_prob.detach().numpy()
    prediction_series = pd.Series(predictions_numpy.flatten(), index=tempdf.index)

    result = new_data_df.copy()
    result["Predictions"] = np.nan
    result.loc[prediction_series.index, "Predictions"] = prediction_series
    return result["Predictions"], 0.5, 0.5, 5, 20
    
def MSFT_ptminclassA1Base_2hr50ptdown_2402010051(new_data_df):
    checkpoint = torch.load(f'{base_dir}/MSFT_ptminclassA1Base_2hr50ptdown_2402010051/target_up.pth', map_location=torch.device('cpu'))
    features = checkpoint['features']
    dropout_rate = checkpoint['dropout_rate']
    input_dim = checkpoint['input_dim']
    layers = checkpoint['layers']
    scaler_X = checkpoint['scaler_X']

    loaded_model = DynamicNNwithDropout(input_dim, layers, dropout_rate)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()

    tempdf = new_data_df.copy()
    # tempdf.dropna(subset=features, inplace=True)
    tempdf = tempdf[features]

    very_large_number = 1e15  # Placeholder for positive infinity
    very_small_number = -1e15  # Placeholder for negative infinity

    for col in tempdf.columns:
        # Replace positive and negative infinity with the defined large and small numbers. this reaplce the 1.5x multiplier logic.
        tempdf[col] = tempdf[col].replace([np.inf, -np.inf], [very_large_number, very_small_number])
        

    tempdf = pd.DataFrame(tempdf.values, columns=features, index=tempdf.index)

    # scale the new data features and generate predictions

    scaled_features = scaler_X.transform(tempdf)
    input_tensor = torch.tensor(scaled_features, dtype=torch.float32)
    predictions = loaded_model(input_tensor)
    predictions_prob = torch.sigmoid(predictions)
    predictions_numpy = predictions_prob.detach().numpy()
    prediction_series = pd.Series(predictions_numpy.flatten(), index=tempdf.index)

    result = new_data_df.copy()
    result["Predictions"] = np.nan
    result.loc[prediction_series.index, "Predictions"] = prediction_series
    return result["Predictions"], 0.5, 0.5, 5, 20
    
def TSLA_ptminclassA1Base_2hr50ptdown_2402010052(new_data_df):
    checkpoint = torch.load(f'{base_dir}/TSLA_ptminclassA1Base_2hr50ptdown_2402010052/target_up.pth', map_location=torch.device('cpu'))
    features = checkpoint['features']
    dropout_rate = checkpoint['dropout_rate']
    input_dim = checkpoint['input_dim']
    layers = checkpoint['layers']
    scaler_X = checkpoint['scaler_X']

    loaded_model = DynamicNNwithDropout(input_dim, layers, dropout_rate)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()

    tempdf = new_data_df.copy()
    # tempdf.dropna(subset=features, inplace=True)
    tempdf = tempdf[features]

    very_large_number = 1e15  # Placeholder for positive infinity
    very_small_number = -1e15  # Placeholder for negative infinity

    for col in tempdf.columns:
        # Replace positive and negative infinity with the defined large and small numbers. this reaplce the 1.5x multiplier logic.
        tempdf[col] = tempdf[col].replace([np.inf, -np.inf], [very_large_number, very_small_number])
        

    tempdf = pd.DataFrame(tempdf.values, columns=features, index=tempdf.index)

    # scale the new data features and generate predictions

    scaled_features = scaler_X.transform(tempdf)
    input_tensor = torch.tensor(scaled_features, dtype=torch.float32)
    predictions = loaded_model(input_tensor)
    predictions_prob = torch.sigmoid(predictions)
    predictions_numpy = predictions_prob.detach().numpy()
    prediction_series = pd.Series(predictions_numpy.flatten(), index=tempdf.index)

    result = new_data_df.copy()
    result["Predictions"] = np.nan
    result.loc[prediction_series.index, "Predictions"] = prediction_series
    return result["Predictions"], 0.5, 0.5, 5, 20
    