import joblib
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

DF_filename = r"../../../data/historical_multiday_minute_DF/SPY_historical_multiday_min.csv"

frames_to_lookahead = -45  #must be negative to look forwards.

ml_dataframe = pd.read_csv(DF_filename)
print(ml_dataframe.columns)
# Chosen_Predictor = ['Bonsai Ratio','Bonsai Ratio 2','PCRoi Up1', 'B1/B2', 'PCRv Up4']

Chosen_Predictor = [ 'Bonsai Ratio',
       'Bonsai Ratio 2', 'B1/B2', 'PCR-Vol', 'PCR-OI',
      'PCRv Up1', 'PCRv Up2',
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

ml_dataframe.dropna(subset=Chosen_Predictor, inplace=True)
length = ml_dataframe.shape[0]
print("Length of ml_dataframe:", length)

ml_dataframe["Target_Down"] = ml_dataframe["Current Stock Price"].shift(frames_to_lookahead)
ml_dataframe["Target_Up"] = ml_dataframe["Current Stock Price"].shift(frames_to_lookahead)
ml_dataframe.dropna(subset=["Target_Up", "Target_Down"], inplace=True)
ml_dataframe.reset_index(drop=True,inplace=True)
X = ml_dataframe[Chosen_Predictor]
# X.reset_index(drop=True, inplace=True)
y_up = ml_dataframe["Target_Up"].values.reshape(-1,1)
y_down = ml_dataframe["Target_Down"]
# Sequentially split the data into train, validation, and test sets
train_ratio = 0.7
val_ratio = 0.1
test_ratio = 0.2
idx_train = int(train_ratio * X.shape[0])
idx_val = int((train_ratio + val_ratio)* X.shape[0])
print('y_up_shape: ',y_up.shape,idx_val,idx_train)
#split data sequentially
tscv = TimeSeriesSplit(n_splits=5)  # You can adjust the number of splits (folds) as needed

for train_val_index, test_index in tscv.split(X):
    X_train_val, X_test = X.iloc[train_val_index], X.iloc[test_index]
    y_up_train_val, y_up_test = y_up[train_val_index], y_up[test_index]
    y_down_train_val, y_down_test = y_down[train_val_index], y_down[test_index]

    # Further split the training and validation sets
    num_train_val_splits = 4  # You can adjust the number of splits for training and validation

    tscv_train_val = TimeSeriesSplit(n_splits=num_train_val_splits)

    for train_index, val_index in tscv_train_val.split(X_train_val):
        X_train, X_val = X_train_val.iloc[train_index], X_train_val.iloc[val_index]
        y_up_train, y_up_val = y_up_train_val[train_index], y_up_train_val[val_index]
        y_down_train, y_down_val = y_down_train_val[train_index], y_down_train_val[val_index]

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
###TODO note that I unscaled the below X or Feature data.  |Was "X_train_scaled"  now it must be "X_train.values"
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

def save_training_info(model_summary, best_params, mse, mae, r2):
    with open(model_summary + '_info.txt', 'w') as f:
        f.write(f"Best Model - Hyperparameters: {best_params}\n")
        f.write(f"MSE for Best Model: {mse}\n")
        f.write(f"MAE for Best Model: {mae}\n")
        f.write(f"R^2 for Best Model: {r2}\n")
        f.write(f"Chosen Predictors: {Chosen_Predictor}\n")

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
        self.fc5 = nn.Linear(hidden_size, hidden_size//2)
        self.dropout5 = nn.Dropout(dropout_rate)
        self.fc6 = nn.Linear(hidden_size//2, hidden_size//4)
        self.dropout6 = nn.Dropout(dropout_rate)
        self.fc7 = nn.Linear(hidden_size//4, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = torch.relu(self.fc3(x))
        x = self.dropout3(x)
        x = torch.relu(self.fc4(x))
        x = self.dropout4(x)
        x = torch.relu(self.fc5(x))
        x = self.dropout5(x)
        x = torch.relu(self.fc6(x))
        x = self.dropout6(x)
        x = self.fc7(x)
        return x  # Reshape predictions to 1D tensor


def create_model(input_size, learning_rate, hidden_size, dropout_rate):
    model = RegressionModel(input_size, hidden_size, dropout_rate)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    return model, optimizer, criterion


def evaluate_model(model, X_test_tensor, y_up_test_unscaled, y_up_scaler):
    model.eval()
    with torch.no_grad():
        X_test_tensor = X_test_tensor.to(device)  # Move the test data to the device
        predicted_values_scaled = model(X_test_tensor).detach().cpu().numpy()
        # Move the predictions back to CPU (if needed for further processing)
    # Unsqueeze the y_test_unscaled tensor to match the shape of predicted_values
##TODO just uncomented below y_up line.
    predicted_values = y_up_scaler.inverse_transform(predicted_values_scaled)

    # Calculate error metrics
    mse = mean_squared_error(y_up_test_unscaled, predicted_values)
    mae = mean_absolute_error(y_up_test_unscaled, predicted_values)
    #TODO changed r2 to scaled
    r2 = r2_score(y_up_test_unscaled, predicted_values)

    return mse, mae, r2
# Add scaling for y_val in the training loop and X_test for prediction
def train_model(model, optimizer, criterion, X_train_tensor, y_up_train_tensor, X_val_tensor, y_up_val_tensor,
                batch_size, num_epochs):
    # Inside the training loop
    model.to(device)
    X_train_tensor = X_train_tensor.to(device)
    y_up_train_tensor = y_up_train_tensor.to(device)
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
        print(epoch)
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            X_batch = X_batch.to(device)  # Move the batch data to the device
            y_batch = y_batch.to(device)  # Move the batch data to the device
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_val_batch, y_val_batch in val_loader:
                val_outputs = model(X_val_batch)
                val_loss = criterion(val_outputs, y_val_batch)
                val_losses.append(val_loss.item())
        #threshhold for early stop
        min_delta=1e-4
        average_loss = sum(val_losses) / len(val_losses)  # Calculate the average validation loss for the current epoch
        if average_loss < best_loss:  # If the current average loss is better than the best seen so far
            best_loss = average_loss
            best_model_state = model.state_dict()  # Save the current state of the model
            counter = 0
        else:
            if (
                    last_average_loss - average_loss) < min_delta:  # If the decrease in average loss is smaller than a threshold
                counter += 1
            if counter >= patience:  # If we have seen no improvement for a number of epochs specified by the patience parameter
                print("Early stopping triggered")
        last_average_loss = average_loss  # Update the last average loss
        model.train()

    # After training loop
    model.load_state_dict(best_model_state)  # Always restore the best model


def train_and_evaluate_models(X_train_tensor, y_up_train_tensor, X_val_tensor, y_up_val_tensor):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    y_up_val_tensor = y_up_val_tensor

    # Grid Search for Hyperparameter Optimization
    input_size = X_train.shape[1]
# """
#     Best Model - Hyperparameters: {'batch_size': 38400, 'dropout_rate': 0.3, 'hidden_size': 1500, 'learning_rate': 0.0005, 'num_epochs': 50}
# """

    param_grid = {
        'learning_rate': [.001,.0001,.01],
        'hidden_size': [1500,2500,5000],#chose 1500 out of 1250/1500/2000
        'dropout_rate': [0,.1,.3,.5],  #chosen most times, safe number. .2 was also close.
        'num_epochs': [75,150],  # Change this to a specific value for the number of epochs #chose 50 out of 50/100/250
        'batch_size': [1000,2048,10000,38000]  # 38400 over 56400
    }

    grid_search_results = []


    for params in ParameterGrid(param_grid):
        print("Training model with parameters:", params)

        model, optimizer, criterion = create_model(input_size, learning_rate=params['learning_rate'],
                                                   hidden_size=params['hidden_size'],
                                                   dropout_rate=params['dropout_rate'])

        # Move the model, optimizer, and criterion to the appropriate device
        model.to(device)
        X_train_tensor = X_train_tensor.to(device)
        y_up_train_tensor = y_up_train_tensor.to(device)
        X_val_tensor = X_val_tensor.to(device)
        y_up_val_tensor = y_up_val_tensor.to(device)

        train_model(model, optimizer, criterion, X_train_tensor, y_up_train_tensor, X_val_tensor, y_up_val_tensor,
                    batch_size=params['batch_size'], num_epochs=params['num_epochs'])

        # Transform the validation data back to the original scale
        # Transform the validation data back to the original scale on the CPU
        y_up_val_unscaled = y_up_scaler.inverse_transform(y_up_val_tensor.detach().cpu().numpy())
        mse, mae, r2 = evaluate_model(model, X_val_tensor, y_up_val_unscaled, y_up_scaler)

        grid_search_results.append({
            'params': params,
            'mse': mse,
            'mae': mae,
            'r2': r2
        })
    print(grid_search_results)
    # Find the best model based on R-squared
    best_model_result = max(grid_search_results, key=lambda x: x['r2'])
    print(best_model_result)
    best_params = best_model_result['params']

    # Train the best model with the full training set
    best_model, _, _ = create_model(input_size, learning_rate=best_params['learning_rate'],
                                    hidden_size=best_params['hidden_size'],
                                    dropout_rate=best_params['dropout_rate'])
    train_model(best_model, optimizer, criterion, X_train_tensor, y_up_train_tensor, X_val_tensor, y_up_val_tensor,
                batch_size=best_params['batch_size'], num_epochs=best_params['num_epochs'])

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
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)
        info = {
            'feature_columns': Chosen_Predictor,
            'X_scaler': X_scaler,
            'y_up_scaler': y_up_scaler,
            'y_down_scaler': y_down_scaler
        }
        info_filename = os.path.join(model_directory, "info.pkl")
        joblib.dump(info, info_filename)

        if not os.path.exists(model_directory):
            os.makedirs(model_directory)
        model_filename_up = os.path.join(model_directory, "target_up.pt")
        torch.save(best_model.state_dict(), model_filename_up)
        save_training_info(model_summary, best_params, mse, mae, r2)

# Call the function for grid search and model training
train_and_evaluate_models(X_train_tensor, y_up_train_tensor, X_val_tensor, y_up_val_tensor)


""""""