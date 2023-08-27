
import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
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
    #Make sure its using right model.
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