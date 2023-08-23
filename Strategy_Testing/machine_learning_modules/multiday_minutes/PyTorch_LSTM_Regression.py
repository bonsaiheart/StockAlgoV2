import datetime
import os

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torchmetrics
from sklearn.preprocessing import StandardScaler

from UTILITIES.logger_config import logger

DF_filename = r"../../../data/historical_multiday_minute_DF/SPY_historical_multiday_min.csv"
Chosen_Predictor = ['Bonsai Ratio',
                    'Bonsai Ratio 2']

# ##had highest corr for 3-5 hours with these:
# Chosen_Predictor = ['Bonsai Ratio','Bonsai Ratio 2','ITM PCR-Vol','ITM PCRoi Up1', 'RSI14','AwesomeOsc5_34', 'Net_IV']
# Read Data
ml_dataframe = pd.read_csv(DF_filename)
print('Columns in Data:', ml_dataframe.columns)

# Setup Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)

# Parameters
# set_best_params_manually = {
#     'learning_rate': 0.001621715398308046, 'num_epochs': 617,
#     'batch_size': 2250, 'optimizer': 'Adadelta',
#     'dropout_rate': 0.13908048750415472, 'num_hidden_units': 2037
# }
cells_forward_to_check = 30  # rows to check (minutes in this case)
threshold_up = 0.5

# Target Calculation
ml_dataframe['Target_Change'] = ml_dataframe['Current Stock Price'].pct_change(
    periods=cells_forward_to_check * -1) * 100
ml_dataframe.dropna(subset=Chosen_Predictor + ["Target_Change"], inplace=True)
ml_dataframe.reset_index(drop=True, inplace=True)

# Separate Features and Target
y_change = ml_dataframe["Target_Change"]
X = ml_dataframe[Chosen_Predictor]
large_number = 1e100
X = X.replace([np.inf], large_number)

# Convert to Torch Tensors
X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
y_change_tensor = torch.tensor(y_change.values, dtype=torch.float32).to(device)

# Split the Data
trainsizepercent = .6
valsizepercent = .2  # test is leftovers
train_size = int(len(X_tensor) * trainsizepercent)
val_size = train_size + int(len(X_tensor) * valsizepercent)
X_train_tensor = X_tensor[:train_size]
X_val_tensor = X_tensor[train_size:val_size]
X_test_tensor = X_tensor[val_size:]
print("X_train_tensor shape:", X_train_tensor.shape)
print("X_val_tensor shape:", X_val_tensor.shape)
print("X_test_tensor shape:", X_test_tensor.shape)

# Check for NaN and Inf
nan_indices = np.argwhere(np.isnan(X))
inf_indices = np.argwhere(np.isinf(X))
print("NaN values found at indices:" if len(nan_indices) > 0 else "No NaN values found.")
print("Infinite values found at indices:" if len(inf_indices) > 0 else "No infinite values found.")

# Scale features (optional - currently disabled)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_tensor.cpu())
X_train_scaled = torch.tensor(X_train_scaled).to('cuda:0')
X_val_scaled = scaler.transform(X_val_tensor.cpu())
X_val_scaled = torch.tensor(X_val_scaled).to('cuda:0')
X_test_scaled = scaler.transform(X_test_tensor.cpu())
X_test_scaled = torch.tensor(X_test_scaled).to('cuda:0')
# No scaling applied here
# X_train_scaled = X_train_tensor
# X_val_scaled = X_val_tensor
# X_test_scaled = X_test_tensor

# Convert the scaled data back to tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
print("train length:", len(X_train_tensor), "val length:", len(X_val_tensor), "test length:", len(X_test_tensor))

# Prepare target variable, reshaping for consistency
y_change_scaled = y_change.values.reshape(-1, 1)

# Optionally, you could apply scaling to the target as well, using y_scaler
y_scaler = StandardScaler()
# y_change_scaled = y_change.values.reshape(-1, 1)
y_change_scaled = y_scaler.fit_transform(y_change.values.reshape(-1, 1))

# Split target variable
y_change_train_scaled = y_change_scaled[:train_size]
y_change_val_scaled = y_change_scaled[train_size:val_size]
y_change_test_scaled = y_change_scaled[val_size:]

# Convert target variable to torch tensors
y_change_train_tensor = torch.tensor(y_change_train_scaled, dtype=torch.float32).to(device)
y_change_val_tensor = torch.tensor(y_change_val_scaled, dtype=torch.float32).to(device)
y_change_test_tensor = torch.tensor(y_change_test_scaled, dtype=torch.float32).to(device)
print(y_change_train_tensor[0], X_train_tensor[0])

sequence_length = 240  # Define your sequence length here


def to_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequence = data[i:i + seq_length]
        sequences.append(sequence.unsqueeze(0))
    return torch.cat(sequences, dim=0)


X_train_sequences = to_sequences(X_train_tensor, sequence_length)
y_train_sequences = y_change_train_tensor[sequence_length:]  # Ensure the targets align with the sequences
X_val_sequences = to_sequences(X_val_tensor, sequence_length)
y_val_sequences = y_change_val_tensor[sequence_length:]  # Ensure the targets align with the sequences
X_test_sequences = to_sequences(X_test_tensor, sequence_length)
y_test_sequences = y_change_test_tensor[sequence_length:]  # Ensure the targets align with the sequences


class LSTMRegressionNN(nn.Module):
    def __init__(self, input_dim, num_hidden_units, dropout_rate, num_layers):
        super(LSTMRegressionNN, self).__init__()
        print(input_dim)

        self.lstm = nn.LSTM(input_dim, num_hidden_units, num_layers=num_layers, dropout=dropout_rate, batch_first=True)
        self.output_layer = nn.Linear(num_hidden_units, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.output_layer(x[:, -1, :])  # Use the output from the last time step
        return x


class NewLSTMRegressionNN(nn.Module):
    # has reinitialization across batches.  lessens memory constraints b/c model is losing long term dependacies across batches.
    def __init__(self, input_dim, num_hidden_units, dropout_rate, num_layers):
        super(NewLSTMRegressionNN, self).__init__()
        print(input_dim)

        self.lstm = nn.LSTM(input_dim, num_hidden_units, num_layers=num_layers, dropout=dropout_rate, batch_first=True)
        self.output_layer = nn.Linear(num_hidden_units, 1)
        self.num_hidden_units = num_hidden_units
        self.num_layers = num_layers

    def forward(self, x):
        # Initialize hidden state and cell state
        h_0, c_0 = torch.zeros(self.num_layers, x.size(0), self.num_hidden_units).to(x.device), torch.zeros(
            self.num_layers, x.size(0), self.num_hidden_units).to(x.device)
        x, _ = self.lstm(x, (h_0, c_0))
        x = self.output_layer(x[:, -1, :])  # Use the output from the last time step
        return x


class RegressionNN(nn.Module):
    def __init__(self, input_dim, num_hidden_units, dropout_rate, num_layers=None):
        super(RegressionNN, self).__init__()
        print("input_dim", input_dim)
        self.layers = nn.Sequential(
            nn.Linear(input_dim, num_hidden_units),
            nn.BatchNorm1d(num_hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            # Add more hidden layers if needed
            nn.Linear(num_hidden_units, 1)
        )

    def forward(self, x):
        return self.layers(x)


def train_model(hparams, X_train, y_train, X_val, y_val):
    patience = 5  # Number of epochs with no improvement to wait before stopping

    patience_counter = 0
    num_layers = hparams["num_layers"]
    model = RegressionNN(X_train.shape[2], hparams["num_hidden_units"], hparams['dropout_rate'], num_layers).to(device)
    model.train()
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()

    optimizer_name = hparams["optimizer"]
    learning_rate = hparams["learning_rate"]
    momentum = hparams.get("momentum", 0)
    weight_decay = hparams.get("weight_decay")

    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "Adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "Adamax":
        optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "Adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "LBFGS":
        optimizer = torch.optim.LBFGS(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    num_epochs = hparams["num_epochs"]
    batch_size = hparams["batch_size"]
    smallest_val_loss = float('inf')  # Initialize the variable for the smallest MSE loss
    best_mae_score = float('inf')  # Initialize the variable for the best MAE score
    best_model_state_dict = None
    l1_lambda = hparams.get("l1_lambda", 0)  # L1 regularization coefficient

    for epoch in range(num_epochs):
        model.train()
        # Training step
        loss = None  # Define loss outside the inner loop
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i: i + batch_size]
            print(X_batch.shape)

            y_batch = y_train[i: i + batch_size]
            y_batch = y_batch
            if optimizer_name == "LBFGS":
                def closure():
                    nonlocal loss  # Refer to the outer scope's loss variable
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch)
                    # Add L1 regularization to loss
                    l1_reg = torch.tensor(0., requires_grad=True).to(device)
                    for param in model.parameters():
                        l1_reg += torch.norm(param, 1)
                    loss += l1_lambda * l1_reg

                    loss.backward()
                    return loss

                optimizer.step(closure)
            else:
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                # Add L1 regularization to loss
                l1_reg = torch.tensor(0., requires_grad=True).to(device)
                for param in model.parameters():
                    l1_reg += torch.norm(param, 1)
                loss += l1_lambda * l1_reg

                loss.backward()
                optimizer.step()

        model.eval()
        # Validation step
        val_loss_accum = 0
        mae_accum = 0
        r2_accum = 0  # Initialize the variable for accumulating R² values
        num_batches_processed = 0
        r2_metric = torchmetrics.R2Score().to(device)  # Move the metric to the device
        mae_metric = torchmetrics.MeanAbsoluteError().to(device)  # Move the metric to the device
        for i in range(0, len(X_val), batch_size):
            X_val_batch = X_val[i: i + batch_size]
            y_val_batch = y_val[i: i + batch_size]
            y_val_batch = y_val_batch
            with torch.no_grad():
                val_outputs = model(X_val_batch)
                val_loss = criterion(val_outputs, y_val_batch)

                # Add L1 regularization to validation loss
                l1_val_reg = torch.tensor(0.).to(device)
                for param in model.parameters():
                    l1_val_reg += torch.norm(param, 1)
                val_loss += l1_lambda * l1_val_reg

                val_loss_accum += val_loss.item()
                mae_score = mae_metric(val_outputs, y_val_batch)  # Calculate MAE

                mae_accum += mae_score.item()
                if len(y_val_batch) < 2:
                    logger.warning('Fewer than two samples. Skipping R² calculation.')
                    r2_score = None  # or some default value
                else:
                    r2_score = r2_metric(val_outputs, y_val_batch)
                if r2_score is not None:
                    r2_accum += r2_score.item()  # Accumulate R² values

                num_batches_processed += 1
        # Calculate the average MSE loss for the epoch
        val_loss_avg = val_loss_accum / num_batches_processed
        mae_avg = mae_accum / num_batches_processed
        try:
            r2_avg = r2_accum / num_batches_processed  # Calculate the average R² for the epoch
        except Exception as e:
            r2_avg = f'r2avg {e}'
        # Update the smallest MSE loss if the current average loss is smaller
        if val_loss_avg < smallest_val_loss:
            smallest_val_loss = val_loss_avg

            best_model_state_dict = model.state_dict()
            patience_counter = 0  # Reset patience counter if validation loss improves
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break
            best_mae_score = mae_avg
        print(f"VALIDATION Epoch: {epoch + 1}, Training Loss: {loss}, Validation Loss: {val_loss_avg} ")
    return best_mae_score, best_model_state_dict, smallest_val_loss, r2_avg


def objective(trial):
    print(datetime.datetime.now())

    # print(len(X_val_tensor))
    learning_rate = trial.suggest_float("learning_rate", .00001, 0.001, log=True)  # 0003034075497582067
    num_epochs = trial.suggest_int("num_epochs", 5, 30)  # 3800 #230  291  400-700
    # batch_size = trial.suggest_int("batch_size", 1400, 2000)  # 10240  3437
    batch_size = 100
    # batch_size = trial.suggest_int('batch_size', 1400, len(X_val_tensor) - 1)
    # dropout_rate = trial.suggest_float("dropout_rate", 0, .5)  # 30311980533100547  16372372692286732
    dropout_rate = .25
    # num_hidden_units = trial.suggest_int("num_hidden_units", 100, 2500)  # 2560 #83 125 63  #7000
    num_hidden_units = 1000
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1)  # Adding L2 regularization parameter

    l1_lambda = trial.suggest_float("l1_lambda", 1e-5, 1e-1)  # Example range

    optimizer_name = trial.suggest_categorical("optimizer", ["SGD", "Adam", "RMSprop"])
    num_layers = trial.suggest_int("num_layers", 2, 3)  # example

    if optimizer_name == "SGD":
        momentum = trial.suggest_float('momentum', 0, 0.9)  # Only applicable if using SGD with momentum
    else:
        momentum = 0  # Default value if not using SGD

    # TODO add learning rate scheduler
    """from torch.optim.lr_scheduler import StepLR
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    # Training loop
    for epoch in range(num_epochs):
        # Training code here
        # ...
        # Validation code here (if you have it)
   
        # Step the scheduler
        scheduler.step()"""
    # Call the train_model function with the current hyperparameters
    best_mae_score, best_model_state_dict, smallest_mse_loss, r2_avg = train_model(

        {
            "l1_lambda": l1_lambda,
            "learning_rate": learning_rate,
            "optimizer": optimizer_name,
            "num_layers": num_layers,
            "momentum": momentum,  # Include optimizer name here
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "dropout_rate": dropout_rate,
            "num_hidden_units": num_hidden_units,
            "weight_decay": weight_decay
            # Add more hyperparameters as needed
        },
        X_train_sequences, y_train_sequences, X_val_sequences, y_val_sequences
    )
    print("best mae score: ", best_mae_score, "smallest mse loss: ", smallest_mse_loss, "best r2 avg:", r2_avg)

    return smallest_mse_loss  # Note this is actually criterion, which is currently mae.
    #    # return prec_score  # Optuna will try to maximize this value


######################################################################################################################
# TODO Comment out to skip the hyperparameter selection.  Swap "best_params".
study = optuna.create_study(direction="minimize")  # We want to maximize the custom loss score.
study.optimize(objective, n_trials=100)  # You can change the number of trials as needed
best_params = study.best_params

# TODO
# best_params = set_best_params_manually
######################################################################################################################

## Train the model with the best hyperparameters
print("~~~~training model using best params.~~~~")
(best_f1_score, best_prec_score, best_model_state_dict, smallest_custom_loss) = train_model(
    best_params, X_train_tensor, y_change_train_tensor, X_val_tensor, y_change_val_tensor)
model_up_nn = NewLSTMRegressionNN(X_train_tensor.shape[2], best_params["num_hidden_units"],
                                  best_params["dropout_rate"], best_params["num_layers"]).to(device)
# Load the saved state_dict into the model
model_up_nn.load_state_dict(best_model_state_dict)
model_up_nn.eval()

input_val = input("Would you like to save these models? y/n: ").upper()
if input_val == "Y":
    model_summary = input("Save this set of models as: ")
    model_directory = os.path.join("../../Trained_Models", f"{model_summary}")
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    model_filename_up = os.path.join(model_directory, "target_up.pth")

    torch.save({'model_class': model_up_nn.__class__.__name__,  # Save the class name
                'features': Chosen_Predictor,
                'input_dim': X_train_tensor.shape[1],
                'dropout_rate': best_params["dropout_rate"],
                'num_hidden_units': best_params[
                    "num_hidden_units"],
                'model_state_dict': model_up_nn.state_dict(),
                }, model_filename_up)

with open(f"../../Trained_Models/{model_summary}/info.txt", "w") as info_txt:
    info_txt.write("This file contains information about the model.\n\n")
    info_txt.write(
        f"File analyzed: {DF_filename}\nCells_Foward_to_check: {cells_forward_to_check}\n\n"
    )
    info_txt.write(
        f"Metrics for Target_Up:\n\n"
    )
    info_txt.write(
        f"Predictors: {Chosen_Predictor}\n\n\n"
        f"Best Params: {best_params}\n\n\n"
        f"Threshold Up (sensitivity): {threshold_up}\n")
