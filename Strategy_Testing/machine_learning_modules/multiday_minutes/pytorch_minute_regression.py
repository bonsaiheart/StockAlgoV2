import joblib
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import ParameterGrid
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
"""Remember to apply the same transformations to any new data you feed into the model, using the X_scaler and y_scaler objects.
After you have trained your model and made predictions, you can use the y_scaler to transform the predictions back into the original scale, like so:

predicted_values_up_scaled = model_up_nn(X_test_tensor).detach().numpy()
predicted_values_up = y_up_scaler.inverse_transform(predicted_values_up_scaled)
Then use the predicted_values_up for calculating your metrics. This way, your error metrics will be in the same units as your original data.

Yes, when using a model that was trained with scaled data, you'll need to scale new input data before making predictions. This is because the model's learned parameters (weights and biases) are optimized for the scale of the training data. If you don't scale the new input data, the model's predictions may be inaccurate.

In the context of your code, if you wanted to make a prediction with some new data new_data (in the form of a pandas DataFrame), you would scale it first:

new_data_scaled = X_scaler.transform(new_data)
new_data_tensor = torch.tensor(new_data_scaled, dtype=torch.float32)
predicted_values_new_scaled = model_up_nn(new_data_tensor).detach().numpy()
predicted_values_new = y_up_scaler.inverse_transform(predicted_values_new_scaled)
In this code:

X_scaler.transform(new_data) scales the new data using the same scaler that was fitted on the training data.
torch.tensor(new_data_scaled, dtype=torch.float32) converts the scaled data to a PyTorch tensor, which is the input format expected by the model.
model_up_nn(new_data_tensor).detach().numpy() makes predictions with the model.
y_up_scaler.inverse_transform(predicted_values_new_scaled) transforms the predictions back to the original scale.
Remember that X_scaler and y_up_scaler should be saved along with the trained model, because they contain information about the mean and standard deviation of the training data, which are needed for scaling ne"""
DF_filename = r"../../../data/historical_multiday_minute_DF/Copy of SPY_historical_multiday_min.csv"

ml_dataframe = pd.read_csv(DF_filename)
print(ml_dataframe.columns)

Chosen_Predictor = [
    "Bonsai Ratio",
    "Bonsai Ratio 2",
    "B1/B2",
    'ITM PCR-Vol',
    "PCRv Up3",
    "PCRv Up2",
    "PCRv Down3",
    "PCRv Down2",
    'Net_IV',
    'Net ITM IV',
    "ITM PCRv Up3",
    "ITM PCRv Down3",
    "ITM PCRv Up2",
    "ITM PCRv Down2",
    "RSI14",
    "AwesomeOsc5_34",
    "RSI",
]

ml_dataframe.dropna(subset=Chosen_Predictor, inplace=True)
length = ml_dataframe.shape[0]
print("Length of ml_dataframe:", length)

ml_dataframe["Target_Down"] = ml_dataframe["Current Stock Price"].shift(-5)
ml_dataframe["Target_Up"] = ml_dataframe["Current Stock Price"].shift(-5)

ml_dataframe.dropna(subset=["Target_Up", "Target_Down"], inplace=True)
ml_dataframe.reset_index(drop=True,inplace=True)
X = ml_dataframe[Chosen_Predictor]
# X.reset_index(drop=True, inplace=True)
print()
y_up = ml_dataframe["Target_Up"].values.reshape(-1,1)
y_down = ml_dataframe["Target_Down"]
# Sequentially split the data into train, validation, and test sets
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15
print(X.shape,y_up.shape)
idx_train = int(train_ratio * X.shape[0])
idx_val = int((train_ratio + val_ratio) * X.shape[0])
print(y_up.shape)
#split data sequentially
X_train, X_val, X_test = X[:idx_train], X[idx_train:idx_val], X[idx_val:]
y_up_train, y_up_val, y_up_test = y_up[:idx_train], y_up[idx_train:idx_val], y_up[idx_val:]
y_down_train, y_down_val, y_down_test = y_down[:idx_train], y_down[idx_train:idx_val], y_down[idx_val:]
X_scaler = StandardScaler()
y_up_scaler = StandardScaler()
y_down_scaler = StandardScaler()

# Fit the scalers
X_scaler.fit(X_train)
y_up_scaler.fit(y_up_train)
# y_down_scaler.fit(y_down_train.values)

# Transform the training data
X_train_scaled = X_scaler.transform(X_train)
y_up_train_scaled = y_up_scaler.transform(y_up_train)
# y_down_train_scaled = y_down_scaler.transform(y_down_train.values)

# Transform the validation data
X_val_scaled = X_scaler.transform(X_val)
y_up_val_scaled = y_up_scaler.transform(y_up_val)
# y_down_val_scaled = y_down_scaler.transform(y_down_val.values)

# Transform the test data
X_test_scaled = X_scaler.transform(X_test)
y_up_test_scaled = y_up_scaler.transform(y_up_test)
# y_down_test_scaled = y_down_scaler.transform(y_down_test.values)

# Replace the old data with the scaled data
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_up_train_tensor = torch.tensor(y_up_train_scaled, dtype=torch.float32)
# y_down_train_tensor = torch.tensor(y_down_train_scaled, dtype=torch.float32)

X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
y_up_val_tensor = torch.tensor(y_up_val_scaled, dtype=torch.float32)
# y_down_val_tensor = torch.tensor(y_down_val_scaled, dtype=torch.float32)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_up_test_tensor = torch.tensor(y_up_test_scaled, dtype=torch.float32)
# y_down_test_tensor = torch.tensor(y_down_test_scaled, dtype=torch.float32)

y_up_val_unscaled = y_up_scaler.inverse_transform(y_up_val_scaled)
y_up_test_unscaled = y_up_scaler.inverse_transform(y_up_test_scaled)
y_up_train_unscaled = y_up_scaler.inverse_transform(y_up_train_scaled)


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


def create_model(input_size, learning_rate, hidden_size, dropout_rate):
    model = RegressionModel(input_size, hidden_size, dropout_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    return model, optimizer, criterion


def evaluate_model(model, X_test_tensor, y_test_unscaled, y_scaler):
    model.eval()
    with torch.no_grad():
        predicted_values_scaled = model(X_test_tensor).detach().numpy()
        predicted_values = y_scaler.inverse_transform(predicted_values_scaled)
    # Unsqueeze the y_test_unscaled tensor to match the shape of predicted_values

    # y_test_unscaled = torch.tensor(y_test_unscaled, dtype=torch.float32)

    mse = mean_squared_error(y_test_unscaled, predicted_values)
    mae = mean_absolute_error(y_test_unscaled, predicted_values)
    r2 = r2_score(y_test_unscaled, predicted_values)

    return mse, mae, r2
# Add scaling for y_val in the training loop and X_test for prediction
def train_model(model, optimizer, criterion, X_train_tensor, y_up_train_tensor, X_val_tensor, y_up_val_tensor,
                batch_size, num_epochs):
    train_dataset = TensorDataset(X_train_tensor, y_up_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    best_loss = float('inf')
    patience = 10
    counter = 0
    last_average_loss = float('inf')  # New variable to track the previous average loss
    best_model_state = None  # New variable to save the best model state

    val_dataset = TensorDataset(X_val_tensor, y_up_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model.train()
    for epoch in range(num_epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                val_outputs = model(X_val_batch)
                print("Val outputs shape:", val_outputs.shape)
                print("y_val_batch shape:", y_val_batch.shape)
                val_loss = criterion(val_outputs, y_val_batch)
                val_losses.append(val_loss.item())

        average_loss = sum(val_losses) / len(val_losses)  # Calculate the average validation loss for the current epoch
        if average_loss < best_loss:  # If the current average loss is better than the best seen so far
            best_loss = average_loss
            best_model_state = model.state_dict()  # Save the current state of the model
            counter = 0
        else:
            if average_loss > last_average_loss:  # If the current average loss is worse than the last one
                counter += 1
            if counter >= patience:  # If we have seen no improvement for a number of epochs specified by the patience parameter
                print("Early stopping triggered")
                model.load_state_dict(best_model_state)  # Load the best model state when early stopping
                return
        last_average_loss = average_loss  # Update the last average loss
        model.train()
def train_and_evaluate_models(X_train_tensor, y_up_train_tensor, X_val_tensor, y_up_val_tensor):
    # Define the validation data loader
    val_batch_size = 32
    y_up_val_tensor = y_up_val_tensor.view(-1, 1)
    val_dataset = TensorDataset(X_val_tensor, y_up_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False)

    # Grid Search for Hyperparameter Optimization
    input_size = X_train.shape[1]
    print(input_size)
    learning_rates = [0.01, 0.001]
    hidden_sizes = [128, 256]
    dropout_rates = [0.2, 0.3]

    param_grid = {
        'learning_rate': learning_rates,
        'hidden_size': hidden_sizes,
        'dropout_rate': dropout_rates,
        'num_epochs': [50],  # Change this to a specific value for the number of epochs
        'batch_size': [16]  # Change this to a specific integer value for batch size
    }

    grid_search_results = []

    for params in ParameterGrid(param_grid):
        print("Training model with parameters:", params)

        model, optimizer, criterion = create_model(input_size, learning_rate=params['learning_rate'],
                                                   hidden_size=params['hidden_size'],
                                                   dropout_rate=params['dropout_rate'])

        train_model(model, optimizer, criterion, X_train_tensor, y_up_train_tensor, X_val_tensor, y_up_val_tensor,
                    batch_size=params['batch_size'], num_epochs=params['num_epochs'])

        # Transform the validation data back to the original scale
        y_up_val_unscaled = y_up_scaler.inverse_transform(y_up_val_tensor.detach().numpy())
        mse, mae, r2 = evaluate_model(model, X_val_tensor, y_up_val_unscaled, y_up_scaler)

        grid_search_results.append({
            'params': params,
            'mse': mse,
            'mae': mae,
            'r2': r2
        })

    # Find the best model based on R-squared
    best_model_result = max(grid_search_results, key=lambda x: x['r2'])
    best_params = best_model_result['params']

    # Train the best model with the full training set
    best_model, _, _ = create_model(input_size, learning_rate=best_params['learning_rate'],
                                    hidden_size=best_params['hidden_size'],
                                    dropout_rate=best_params['dropout_rate'])
    train_model(best_model, optimizer, criterion, X_train_tensor, y_up_train_tensor, X_val_tensor, y_up_val_tensor,
                batch_size=32, num_epochs=50)

    # Evaluate the best model on the test set
    y_up_test_unscaled = y_up_scaler.inverse_transform(y_up_test_tensor.detach().numpy())
    mse, mae, r2 = evaluate_model(best_model, X_test_tensor, y_up_test_unscaled, y_up_scaler)

    print("Best Model - Hyperparameters:", best_params)
    print("MSE for Best Model:", mse)
    print("MAE for Best Model:", mae)
    print("R^2 for Best Model:", r2)

    # Save the best model and scalers
    input_val = input("Would you like to save the best model and scalers? y/n: ").upper()
    if input_val == "Y":
        model_summary = input("Save this best model as: ")
        model_directory = os.path.join("../../Trained_Models", f"{model_summary}")
        scaler_filename = os.path.join(model_directory, "scalers.pkl")
        joblib.dump({'X_scaler': X_scaler, 'y_up_scaler': y_up_scaler, 'y_down_scaler': y_down_scaler}, scaler_filename)

        if not os.path.exists(model_directory):
            os.makedirs(model_directory)
        model_filename_up = os.path.join(model_directory, "target_up.pt")
        torch.save(best_model.state_dict(), model_filename_up)

# Call the function for grid search and model training
train_and_evaluate_models(X_train_tensor, y_up_train_tensor, X_val_tensor, y_up_val_tensor)

""""""