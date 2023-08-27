import os
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler

base_dir = os.path.dirname(__file__)


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

def Buy_4hr_ffTSLA230817F(new_data_df):
    checkpoint = torch.load(f'{base_dir}/_4hr_ffTSLA230817F/target_up.pth', map_location=torch.device('cpu'))
    features = checkpoint['features']
    scaler_X = MinMaxScaler()
    scaler_X.min_ = checkpoint['scaler_X_min']
    scaler_X.scale_ = checkpoint['scaler_X_scale']

    scaler_y = MinMaxScaler()
    scaler_y.min_ = checkpoint['scaler_y_min']
    scaler_y.scale_ = checkpoint['scaler_y_scale']
    class_name_str = checkpoint['model_class']
    dropout_rate = checkpoint['dropout_rate']
    input_dim = checkpoint['input_dim']
    num_hidden_units = checkpoint['num_hidden_units']
    num_layers = checkpoint['num_layers']
    # Get the class object using the class name string
    model_class = globals().get(class_name_str)

    if model_class is None:
        raise ValueError(f'Unknown class name: {class_name_str}')

    # Make sure its using right model.
    loaded_model = model_class(input_dim, num_hidden_units, dropout_rate, num_layers)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()  # Set the model to evaluation mode
    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    tempdf['LastTradeTime'] = tempdf['LastTradeTime'].apply(
        lambda x: datetime.strptime(x, '%y%m%d_%H%M'))
    tempdf['LastTradeTime'] = tempdf['LastTradeTime'].apply(lambda x: x.timestamp())
    tempdf['ExpDate'] = tempdf['ExpDate'].astype(float)



    tempdf[features] = (tempdf[features] - scaler_X.min_) * scaler_X.scale_

    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    threshold = 1e10
    tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
    scaler_X = MinMaxScaler(feature_range=(-1, 1))
    # Fit and transform the scaler to the features
    scaled_features = scaler_X.fit_transform(tempdf[features].values)
    # Replace the old values with the scaled values
    tempdf[features] = scaled_features
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
def Buy_4hr_ffTSLA230817D(new_data_df):
    checkpoint = torch.load(f'{base_dir}/_4hr_ffTSLA230817D/target_up.pth', map_location=torch.device('cpu'))
    features = checkpoint['features']
    scaler_X = MinMaxScaler()
    scaler_X.min_ = checkpoint['scaler_X_min']
    scaler_X.scale_ = checkpoint['scaler_X_scale']

    scaler_y = MinMaxScaler()
    scaler_y.min_ = checkpoint['scaler_y_min']
    scaler_y.scale_ = checkpoint['scaler_y_scale']
    class_name_str = checkpoint['model_class']
    dropout_rate = checkpoint['dropout_rate']
    input_dim = checkpoint['input_dim']
    num_hidden_units = checkpoint['num_hidden_units']
    num_layers = checkpoint['num_layers']
    # Get the class object using the class name string
    model_class = globals().get(class_name_str)

    if model_class is None:
        raise ValueError(f'Unknown class name: {class_name_str}')

    # Make sure its using right model.
    loaded_model = model_class(input_dim, num_hidden_units, dropout_rate, num_layers)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()  # Set the model to evaluation mode
    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    tempdf['LastTradeTime'] = tempdf['LastTradeTime'].apply(
        lambda x: datetime.strptime(x, '%y%m%d_%H%M'))
    tempdf['LastTradeTime'] = tempdf['LastTradeTime'].apply(lambda x: x.timestamp())
    tempdf['ExpDate'] = tempdf['ExpDate'].astype(float)



    tempdf[features] = (tempdf[features] - scaler_X.min_) * scaler_X.scale_

    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    threshold = 1e10
    tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
    scaler_X = MinMaxScaler(feature_range=(-1, 1))
    # Fit and transform the scaler to the features
    scaled_features = scaler_X.fit_transform(tempdf[features].values)
    # Replace the old values with the scaled values
    tempdf[features] = scaled_features
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
def Buy_4hr_ffTSLA230817B(new_data_df):
    checkpoint = torch.load(f'{base_dir}/_4hr_ffTSLA230817B/target_up.pth', map_location=torch.device('cpu'))
    features = checkpoint['features']
    scaler_X = MinMaxScaler()
    scaler_X.min_ = checkpoint['scaler_X_min']
    scaler_X.scale_ = checkpoint['scaler_X_scale']

    scaler_y = MinMaxScaler()
    scaler_y.min_ = checkpoint['scaler_y_min']
    scaler_y.scale_ = checkpoint['scaler_y_scale']
    class_name_str = checkpoint['model_class']
    dropout_rate = checkpoint['dropout_rate']
    input_dim = checkpoint['input_dim']
    num_hidden_units = checkpoint['num_hidden_units']
    num_layers = checkpoint['num_layers']
    # Get the class object using the class name string
    model_class = globals().get(class_name_str)

    if model_class is None:
        raise ValueError(f'Unknown class name: {class_name_str}')

    # Make sure its using right model.
    loaded_model = model_class(input_dim, num_hidden_units, dropout_rate, num_layers)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()  # Set the model to evaluation mode
    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    tempdf['LastTradeTime'] = tempdf['LastTradeTime'].apply(
        lambda x: datetime.strptime(x, '%y%m%d_%H%M'))
    tempdf['LastTradeTime'] = tempdf['LastTradeTime'].apply(lambda x: x.timestamp())
    tempdf['ExpDate'] = tempdf['ExpDate'].astype(float)

    tempdf[features] = (tempdf[features] - scaler_X.min_) * scaler_X.scale_

    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    threshold = 1e10
    tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
    scaler_X = MinMaxScaler(feature_range=(-1, 1))
    # Fit and transform the scaler to the features
    scaled_features = scaler_X.fit_transform(tempdf[features].values)
    # Replace the old values with the scaled values
    tempdf[features] = scaled_features
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
def Buy_4hr_ffTSLA230817(new_data_df):
    checkpoint = torch.load(f'{base_dir}/_4hr_ffTSLA230817/target_up.pth', map_location=torch.device('cpu'))
    features = checkpoint['features']
    scaler_X = MinMaxScaler()
    scaler_X.min_ = checkpoint['scaler_X_min']
    scaler_X.scale_ = checkpoint['scaler_X_scale']

    scaler_y = MinMaxScaler()
    scaler_y.min_ = checkpoint['scaler_y_min']
    scaler_y.scale_ = checkpoint['scaler_y_scale']
    class_name_str = checkpoint['model_class']
    dropout_rate = checkpoint['dropout_rate']
    input_dim = checkpoint['input_dim']
    num_hidden_units = checkpoint['num_hidden_units']
    num_layers = checkpoint['num_layers']
    # Get the class object using the class name string
    model_class = globals().get(class_name_str)

    if model_class is None:
        raise ValueError(f'Unknown class name: {class_name_str}')

    # Make sure its using right model.
    loaded_model = model_class(input_dim, num_hidden_units, dropout_rate, num_layers)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()  # Set the model to evaluation mode
    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    tempdf['LastTradeTime'] = tempdf['LastTradeTime'].apply(
        lambda x: datetime.strptime(x, '%y%m%d_%H%M'))
    tempdf['LastTradeTime'] = tempdf['LastTradeTime'].apply(lambda x: x.timestamp())
    tempdf['ExpDate'] = tempdf['ExpDate'].astype(float)



    tempdf[features] = (tempdf[features] - scaler_X.min_) * scaler_X.scale_

    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    threshold = 1e10
    tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
    scaler_X = MinMaxScaler(feature_range=(-1, 1))
    # Fit and transform the scaler to the features
    scaled_features = scaler_X.fit_transform(tempdf[features].values)
    # Replace the old values with the scaled values
    tempdf[features] = scaled_features
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
def Buy_4hr_ffSPY230805A(new_data_df):
    checkpoint = torch.load(f'{base_dir}/_4hr_ffSPY230805A/target_up.pth', map_location=torch.device('cpu'))
    features = checkpoint['features']
    scaler_X = MinMaxScaler()
    scaler_X.min_ = checkpoint['scaler_X_min']
    scaler_X.scale_ = checkpoint['scaler_X_scale']

    scaler_y = MinMaxScaler()
    scaler_y.min_ = checkpoint['scaler_y_min']
    scaler_y.scale_ = checkpoint['scaler_y_scale']
    class_name_str = checkpoint['model_class']
    dropout_rate = checkpoint['dropout_rate']
    input_dim = checkpoint['input_dim']
    num_hidden_units = checkpoint['num_hidden_units']
    num_layers = checkpoint['num_layers']
    # Get the class object using the class name string
    model_class = globals().get(class_name_str)

    if model_class is None:
        raise ValueError(f'Unknown class name: {class_name_str}')

    # Make sure its using right model.
    loaded_model = model_class(input_dim, num_hidden_units, dropout_rate, num_layers)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()  # Set the model to evaluation mode
    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    tempdf['LastTradeTime'] = tempdf['LastTradeTime'].apply(
        lambda x: datetime.strptime(x, '%y%m%d_%H%M'))
    tempdf['LastTradeTime'] = tempdf['LastTradeTime'].apply(lambda x: x.timestamp())
    tempdf['ExpDate'] = tempdf['ExpDate'].astype(float)



    tempdf[features] = (tempdf[features] - scaler_X.min_) * scaler_X.scale_

    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    threshold = 1e10
    tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
    scaler_X = MinMaxScaler(feature_range=(-1, 1))
    # Fit and transform the scaler to the features
    scaled_features = scaler_X.fit_transform(tempdf[features].values)
    # Replace the old values with the scaled values
    tempdf[features] = scaled_features
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
def Buy_4hr_ffSPY230805(new_data_df):
    checkpoint = torch.load(f'{base_dir}/_4hr_ffSPY230805/target_up.pth', map_location=torch.device('cpu'))
    features = checkpoint['features']
    # Recreate the scalers
    # scaler_X = MinMaxScaler()
    # scaler_X.min_ = checkpoint['scaler_X_min']
    # scaler_X.scale_ = checkpoint['scaler_X_scale']

    # scaler_y = MinMaxScaler()
    # scaler_y.min_ = checkpoint['scaler_y_min']
    # scaler_y.scale_ = checkpoint['scaler_y_scale']
    class_name_str = checkpoint['model_class']
    dropout_rate = checkpoint['dropout_rate']
    input_dim = checkpoint['input_dim']
    num_hidden_units = checkpoint['num_hidden_units']
    num_layers = checkpoint['num_layers']
    # Get the class object using the class name string
    model_class = globals().get(class_name_str)

    if model_class is None:
        raise ValueError(f'Unknown class name: {class_name_str}')

    # Make sure its using right model.
    loaded_model = model_class(input_dim, num_hidden_units, dropout_rate, num_layers)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()  # Set the model to evaluation mode
    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    tempdf['LastTradeTime'] = tempdf['LastTradeTime'].apply(
        lambda x: datetime.strptime(x, '%y%m%d_%H%M'))
    tempdf['LastTradeTime'] = tempdf['LastTradeTime'].apply(lambda x: x.timestamp())
    tempdf['ExpDate'] = tempdf['ExpDate'].astype(float)


    # Assuming new_y_values contains the y values for the new stock
    new_y_values = tempdf['Current Stock Price'].values.reshape(-1, 1)
    # Create a new MinMaxScaler instance
    scaler_y_new = MinMaxScaler(feature_range=(-1, 1))
    # Fit the scaler to the new y values
    scaler_y_new.fit_transform(new_y_values)



    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    threshold = 1e10
    tempdf[features] = np.clip(tempdf[features], -threshold, threshold)
    scaler_X = MinMaxScaler(feature_range=(-1, 1))
    # Fit and transform the scaler to the features
    scaled_features = scaler_X.fit_transform(tempdf[features].values)
    # Replace the old values with the scaled values
    tempdf[features] = scaled_features
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


# df = pd.read_csv('../../data/historical_multiday_minute_DF/Copy of SPY_historical_multiday_min.csv')
# Buy_4hr_ffSPY230805(df)


def Buy_1hr_FFNNRegSPYA1(new_data_df):
    checkpoint = torch.load(f'{base_dir}/_1hr_FFNNRegSPYA1/target_up.pth', map_location=torch.device('cpu'))
    features = checkpoint['features']
    # class_name_str = checkpoint['class_name']
    class_name_str = 'RegressionNN'
    dropout_rate = checkpoint['dropout_rate']
    input_dim = checkpoint['input_dim']
    num_hidden_units = checkpoint['num_hidden_units']
    # num_layers= checkpoint['num_layers']
    num_layers = 3
    loaded_model = RegressionNN(input_dim, num_hidden_units, dropout_rate, num_layers)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()  # Set the model to evaluation mode
    tempdf = new_data_df.copy()  # Create a copy of the original DataFrame
    tempdf.dropna(subset=features, inplace=True)  # Drop rows with missing values in specified features
    threshold = 1e10
    X = np.clip(tempdf[features], -threshold, threshold)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    # Convert DataFrame to a PyTorch tensor
    input_tensor = torch.tensor(X, dtype=torch.float32)

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


def Buy_1hr_ptminclassSPYA1(new_data_df):
    checkpoint = torch.load(f'{base_dir}/_1hr_ptminclassSPYA1/target_up.pth', map_location=torch.device('cpu'))
    features = checkpoint['features']
    # class_name_str = checkpoint['class_name']
    class_name_str = 'BinaryClassificationNNwithDropout'
    dropout_rate = checkpoint['dropout_rate']
    input_dim = checkpoint['input_dim']
    num_hidden_units = checkpoint['num_hidden_units']
    class_mapping = {
        'BinaryClassificationNNwithDropout': BinaryClassificationNNwithDropout
        # Add other class mappings if needed
    }

    # Get the class object using the class name string
    model_class = class_mapping.get(class_name_str)
    if model_class is None:
        raise ValueError(f'Unknown class name: {class_name_str}')
    # if model_class is None:
    # if class_name == 'BinaryClassificationNNwithDropout':
    #     model_class = BinaryClassificationNNwithDropout
    # # Handle other classes if needed
    # else:
    #     raise ValueError(f'Unknown class name: {class_name}')
    # Make sure its using right model.
    loaded_model = model_class(input_dim, num_hidden_units, dropout_rate)
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


def Buy_3hr_PTminClassSPYA1(new_data_df):
    checkpoint = torch.load(f'{base_dir}/_3hr_ptclassA1/target_up.pth', map_location=torch.device('cpu'))
    features = checkpoint['features']
    dropout_rate = checkpoint['dropout_rate']
    input_dim = checkpoint['input_dim']
    num_hidden_units = checkpoint['num_hidden_units']
    loaded_model = BinaryClassificationNNwithDropout(input_dim, num_hidden_units, dropout_rate)
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


def Buy_3hr_PTminClassSPYA1(new_data_df):
    checkpoint = torch.load(f'{base_dir}/_3hr_ptclassA1/target_up.pth', map_location=torch.device('cpu'))
    features = checkpoint['features']
    dropout_rate = .05
    input_dim = checkpoint['input_dim']
    num_hidden_units = checkpoint['num_hidden_units']
    loaded_model = BinaryClassificationNNwithDropout(input_dim, num_hidden_units, dropout_rate)
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
    checkpoint = torch.load(f'{base_dir}/_2hr_ptminclassSPYA2/target_up.pth', map_location=torch.device('cpu'))
    features = ['Bonsai Ratio', 'Bonsai Ratio 2', 'PCRoi Up1', 'ITM PCRoi Up1', 'RSI14', 'AwesomeOsc5_34', 'Net IV LAC']
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
    features = [
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
    features = [
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
    return (result["Predictions"], .6, .5)
# current_directory = os.getcwd()
# print("Current Directory:", current_directory)
# df = pd.read_csv("../../data/DailyMinutes/SPY/SPY_230721.csv")
# predictions, _, _ = Buy_1hr_ptmin1A1(df)  # Unpack the tuple returned by Buy_1hr_ptmin1A1
# df['Predictions'] = predictions
# df.to_csv("testing123.csv")

# def load_model(path):
#     model_info = torch.load(path)
#     model_class_name = model_info['model_class']
#
#     # Decide which class to instantiate based on the saved class name
#     if model_class_name == 'BinaryClassificationNNwithDropout':
#         model = BinaryClassificationNNwithDropout(input_dim=model_info['input_dim'],
#                                                   dropout_rate=model_info['dropout_rate'],
#                                                   num_hidden_units=model_info['num_hidden_units'])
#     elif model_class_name == 'AnotherModelClass':
#         # Instantiate a different class if needed
#         model = AnotherModelClass(...)
#     else:
#         raise ValueError(f"Unknown model class: {model_class_name}")
#
#     # Load the model state
#     model.load_state_dict(model_info['model_state_dict'])
#     model.eval()
#
#     return model, model_info['features']
