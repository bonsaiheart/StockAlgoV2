import datetime
import os

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torchmetrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau
from torch.utils.data import Dataset, TensorDataset, DataLoader

from UTILITIES.logger_config import logger

DF_filename = (
    r"../../../data/historical_multiday_minute_DF/SPY_historical_multiday_min.csv"
)
Chosen_Predictor = ["Bonsai Ratio", "Bonsai Ratio 2"]


class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def play_sound():
    # Play the sound file using the default audio player
    # os.system("aplay alert.wav")  # For Linux
    # os.system("afplay alert.wav")  # For macOS
    os.system("start alert.wav")  # For Windows


# ##had highest corr for 3-5 hours with these:
# Chosen_Predictor = ['Bonsai Ratio','Bonsai Ratio 2','ITM PCR-Vol','ITM PCRoi Up1', 'RSI14','AwesomeOsc5_34', 'Net_IV']
ml_dataframe = pd.read_csv(DF_filename)
print("Columns in Data:", ml_dataframe.columns)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
r2_metric = torchmetrics.R2Score().to(device)  # Move the metric to the device
mae_metric = torchmetrics.MeanAbsoluteError().to(device)

cells_forward_to_check = 30  # rows to check (minutes in this case)
threshold_up = 0.5

# Target Calculation
ml_dataframe["Target_Change"] = (
    ml_dataframe["Current Stock Price"].pct_change(periods=cells_forward_to_check * -1)
    * 100
)
ml_dataframe.dropna(subset=Chosen_Predictor + ["Target_Change"], inplace=True)
ml_dataframe.reset_index(drop=True, inplace=True)

# Separate Features and Target
y_change = ml_dataframe["Target_Change"]
X = ml_dataframe[Chosen_Predictor]
large_number = 1e100
X = X.replace([np.inf], large_number)

# Split Data into Training, Validation, and Test Sets
trainsizepercent = 0.6
valsizepercent = 0.2

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y_change, test_size=1 - trainsizepercent, random_state=42, shuffle=False
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=1 - valsizepercent / (1 - trainsizepercent),
    random_state=42,
    shuffle=False,
)

# Scale the Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train.values, dtype=torch.float32).to(device)
X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val = torch.tensor(y_val.values, dtype=torch.float32).to(device)

# Handle NaN and Inf
large_number = 1e100
X = X.replace([np.inf], large_number)
nan_indices = np.argwhere(np.isnan(X))
inf_indices = np.argwhere(np.isinf(X))
print(
    "NaN values found at indices:" if len(nan_indices) > 0 else "No NaN values found."
)
print(
    "Infinite values found at indices:"
    if len(inf_indices) > 0
    else "No infinite values found."
)


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


def train_model(hparams, X_train, y_train, X_val, y_val):
    # rest of the code remains the same, remove the optimizer creation part inside this function
    batch_size = hparams["batch_size"]
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    patience = 5  # Number of epochs with no improvement to wait before stopping
    lr_scheduler = hparams.get(
        "lr_scheduler"
    )  # Get the lr_scheduler from the hyperparameters

    patience_counter = 0
    num_layers = hparams["num_layers"]
    model = RegressionNN(
        X_train.shape[1],
        hparams["num_hidden_units"],
        hparams["dropout_rate"],
        num_layers,
    ).to(device)
    optimizer = create_optimizer(
        hparams["optimizer"],
        hparams["learning_rate"],
        hparams.get("momentum", 0),
        hparams.get("weight_decay"),
        model.parameters(),
    )

    model.train()
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()  # mae
    optimizer_name = hparams["optimizer"]
    num_epochs = hparams["num_epochs"]
    smallest_val_loss = float(
        "inf"
    )  # Initialize the variable for the smallest MSE loss
    best_mae_score = float("inf")  # Initialize the variable for the best MAE score
    best_mse_score = float("inf")  # Initialize the variable for the best MAE score

    best_model_state_dict = None
    l1_lambda = hparams.get("l1_lambda", 0)  # L1 regularization coefficient

    for epoch in range(num_epochs):
        model.train()
        # Training step
        loss = None  # Define loss outside the inner loop
        for X_batch, y_batch in train_loader:
            if optimizer_name == "LBFGS":

                def closure():
                    nonlocal loss  # Refer to the outer scope's loss variable
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    outputs = outputs.squeeze(1)
                    loss = criterion(outputs, y_batch)
                    # Add L1 regularization to loss
                    l1_reg = torch.tensor(0.0, requires_grad=True).to(device)
                    for param in model.parameters():
                        l1_reg += torch.norm(param, 1)
                    loss += l1_lambda * l1_reg
                    loss.backward()
                    return loss

                optimizer.step(closure)
            else:
                optimizer.zero_grad()
                outputs = model(X_batch)
                outputs = outputs.squeeze(1)

                loss = criterion(outputs, y_batch)
                # Add L1 regularization to loss
                l1_reg = torch.tensor(0.0, requires_grad=True).to(device)
                for param in model.parameters():
                    l1_reg += torch.norm(param, 1)
                loss += l1_lambda * l1_reg
                loss.backward()
                optimizer.step()
        if lr_scheduler is not None:
            if isinstance(lr_scheduler, StepLR) or isinstance(
                lr_scheduler, ExponentialLR
            ):
                lr_scheduler.step()
        model.eval()
        # Validation step
        val_loss_accum = 0
        mae_accum = 0
        mse_accum = 0

        r2_accum = 0  # Initialize the variable for accumulating R² values
        num_batches_processed = 0
        r2_metric = torchmetrics.R2Score().to(device)  # Move the metric to the device
        mae_metric = torchmetrics.MeanAbsoluteError().to(
            device
        )  # Move the metric to the device
        mse_metric = torchmetrics.MeanSquaredError().to(
            device
        )  # Move the metric to the device

        for X_val_batch, y_val_batch in val_loader:
            with torch.no_grad():
                val_outputs = model(X_val_batch)
                val_outputs = val_outputs.squeeze(1)
                val_loss = criterion(val_outputs, y_val_batch)

                # Add L1 regularization to validation loss
                l1_val_reg = torch.tensor(0.0).to(device)
                for param in model.parameters():
                    l1_val_reg += torch.norm(param, 1)
                val_loss += l1_lambda * l1_val_reg

                val_loss_accum += val_loss.item()
                mae_score = mae_metric(val_outputs, y_val_batch)  # Calculate MAE
                mse_score = mse_metric(val_outputs, y_val_batch)  # Calculate MAE

                mae_accum += mae_score.item()
                mse_accum += mse_score.item()

                if len(y_val_batch) < 2:
                    logger.warning("Fewer than two samples. Skipping R² calculation.")
                    r2_score = None  # or some default value
                else:
                    r2_score = r2_metric(val_outputs, y_val_batch)
                if r2_score is not None:
                    r2_accum += r2_score.item()  # Accumulate R² values

                num_batches_processed += 1
        # Calculate the average MSE loss for the epoch
        val_loss_avg = val_loss_accum / num_batches_processed
        mae_avg = mae_accum / num_batches_processed
        mse_avg = mse_accum / num_batches_processed

        if isinstance(lr_scheduler, ReduceLROnPlateau):
            # Step the learning rate scheduler
            lr_scheduler.step(val_loss_avg)
        try:
            r2_avg = (
                r2_accum / num_batches_processed
            )  # Calculate the average R² for the epoch
        except Exception as e:
            r2_avg = f"r2avg {e}"
        # Update the smallest MSE loss if the current average loss is smaller
        if val_loss_avg < smallest_val_loss:
            smallest_val_loss = val_loss_avg
            # print(model.state_dict())
            best_model_state_dict = model.state_dict()
            best_mae_score = mae_avg
            best_mse_score = mse_avg

            patience_counter = 0  # Reset patience counter if validation loss improves
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break

        print(
            f"VALIDATION Epoch: {epoch + 1}, Training Loss: {loss}, Validation Loss: {val_loss_avg} "
        )
    if r2_avg > 0:
        play_sound()

    return (
        best_mae_score,
        best_mse_score,
        best_model_state_dict,
        smallest_val_loss,
        r2_avg,
    )


def create_optimizer(
    optimizer_name, learning_rate, momentum, weight_decay, model_parameters
):
    if optimizer_name == "SGD":
        return torch.optim.SGD(
            model_parameters,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    elif optimizer_name == "Adam":
        return torch.optim.Adam(
            model_parameters, lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name == "AdamW":
        return torch.optim.AdamW(
            model_parameters, lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name == "RMSprop":
        return torch.optim.RMSprop(
            model_parameters, lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name == "Adagrad":
        return torch.optim.Adagrad(
            model_parameters, lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name == "Adamax":
        return torch.optim.Adamax(
            model_parameters, lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name == "Adadelta":
        return torch.optim.Adadelta(
            model_parameters, lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name == "LBFGS":
        return torch.optim.LBFGS(
            model_parameters, lr=learning_rate, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def objective(trial):
    print(datetime.datetime.now())

    # print(len(X_val_tensor))
    learning_rate = trial.suggest_float(
        "learning_rate", 0.00001, 0.01, log=True
    )  # 0003034075497582067
    num_epochs = trial.suggest_int("num_epochs", 5, 1000)  # 3800 #230  291  400-700
    # batch_size = trial.suggest_int("batch_size", 20, 3000)  # 10240  3437
    batch_size = trial.suggest_int("batch_size", 20, len(X_val) - 1)
    dropout_rate = trial.suggest_float(
        "dropout_rate", 0, 0.5
    )  # 30311980533100547  16372372692286732
    num_hidden_units = trial.suggest_int(
        "num_hidden_units", 100, 4000
    )  # 2560 #83 125 63  #7000
    weight_decay = trial.suggest_float(
        "weight_decay", 1e-5, 1e-1
    )  # Adding L2 regularization parameter
    l1_lambda = trial.suggest_float("l1_lambda", 1e-5, 1e-1)  # l1 regssss
    lr_scheduler_name = trial.suggest_categorical(
        "lr_scheduler", ["StepLR", "ExponentialLR", "ReduceLROnPlateau"]
    )

    optimizer_name = trial.suggest_categorical(
        "optimizer", ["SGD", "Adam", "AdamW", "RMSprop"]
    )
    num_layers = trial.suggest_int("num_layers", 1, 5)  # example

    if optimizer_name == "SGD":
        momentum = trial.suggest_float(
            "momentum", 0, 0.9
        )  # Only applicable if using SGD with momentum
    else:
        momentum = 0  # Default value if not using SGD
    patience = None  # Define a default value outside the conditional block
    step_size = None  # Define a default value for step_size as well
    gamma = None
    # Inside the objective function, create the model first
    model = RegressionNN(
        X_train.shape[1], num_hidden_units, dropout_rate, num_layers
    ).to(device)

    # Then pass the model's parameters to create_optimizer
    optimizer = create_optimizer(
        optimizer_name, learning_rate, momentum, weight_decay, model.parameters()
    )

    # Now proceed with the rest of the code
    ...

    if lr_scheduler_name == "StepLR":
        step_size = trial.suggest_int("step_size", 5, 50)
        gamma = trial.suggest_float("gamma", 0.1, 1)

        scheduler = StepLR(
            optimizer, step_size, gamma
        )  # Use optimizer, not optimizer_name
    elif lr_scheduler_name == "ExponentialLR":
        gamma = trial.suggest_float("gamma", 0.1, 1)
        scheduler = ExponentialLR(optimizer, gamma)  # Use optimizer
    elif lr_scheduler_name == "ReduceLROnPlateau":
        patience = trial.suggest_int("patience", 5, 20)
        scheduler = ReduceLROnPlateau(
            optimizer, patience=patience, mode="min"
        )  # Use optimizer

    # Call the train_model function with the current hyperparameters
    (
        best_mae_score,
        best_mse_score,
        best_model_state_dict,
        smallest_val_loss,
        r2_avg,
    ) = train_model(
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
            "weight_decay": weight_decay,
            "lr_scheduler": scheduler,  # Pass the scheduler instance here
            "patience": patience,  # Pass the value directly
            "gamma": gamma,  # Pass the value directly
            "step_size": step_size,  # Add step_size parameter for StepLR
            # Add more hyperparameters as needed
        },
        X_train,
        y_train,
        X_val,
        y_val,
    )
    print(
        "best mae score: ",
        best_mae_score,
        "best mse score: ",
        best_mse_score,
        "smallest val loss: ",
        smallest_val_loss,
        "best r2 avg:",
        r2_avg,
    )

    return smallest_val_loss  # Note this is actually criterion, which is currently mae.
    #    # return prec_score  # Optuna will try to maximize this value


######################################################################################################################
# TODO Comment out to skip the hyperparameter selection.  Swap "best_params".
study = optuna.create_study(
    direction="minimize"
)  # We want to maximize the custom loss score.
study.optimize(objective, n_trials=100)  # You can change the number of trials as needed
best_params = study.best_params
print(best_params)
# TODO
# best_params = set_best_params_manually
######################################################################################################################

## Train the model with the best hyperparameters
print("~~~~training model using best params.~~~~")
(
    best_mae_score,
    best_mse_score,
    best_model_state_dict,
    smallest_val_loss,
    r2_avg,
) = train_model(best_params, X_train, y_train, X_val, y_val)
model_up_nn = RegressionNN(
    X_train.shape[1],
    best_params["num_hidden_units"],
    best_params["dropout_rate"],
    best_params["num_layers"],
).to(device)
# Load the saved state_dict into the model
model_up_nn.load_state_dict(best_model_state_dict)
model_up_nn.eval()
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float).to(
    device
)  # Convert to tensor and move to the same device
predicted_values = model_up_nn(X_test_tensor)
predicted_values = predicted_values.squeeze()

# Calculate R2 score
r2 = r2_metric(predicted_values, y_test_tensor)
mae = mae_metric(predicted_values, y_test_tensor)
print(f"R2 Score: {r2}")
# print('selected features: ',selected_features)

input_val = input("Would you like to save these models? y/n: ").upper()
if input_val == "Y":
    model_summary = input("Save this set of models as: ")
    model_directory = os.path.join("../../Trained_Models", f"{model_summary}")
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    model_filename_up = os.path.join(model_directory, "target_up.pth")

    torch.save(
        {
            "model_class": model_up_nn.__class__.__name__,  # Save the class name
            "features": Chosen_Predictor,
            "input_dim": X_train.shape[1],
            "dropout_rate": best_params["dropout_rate"],
            "num_hidden_units": best_params["num_hidden_units"],
            "model_state_dict": model_up_nn.state_dict(),
        },
        model_filename_up,
    )

with open(f"../../Trained_Models/{model_summary}/info.txt", "w") as info_txt:
    info_txt.write("This file contains information about the model.\n\n")
    info_txt.write(
        f"File analyzed: {DF_filename}\nCells_Foward_to_check: {cells_forward_to_check}\n\n"
    )
    info_txt.write(f"Metrics for Target_Up:\n\n")
    info_txt.write(
        f"Predictors: {Chosen_Predictor}\n\n\n"
        f"Best Params: {best_params}\n\n\n"
        f"Threshold Up (sensitivity): {threshold_up}\n"
    )
"""best mae score:  0.13563300474059015 smallest mse loss:  0.19855372572229021 best r2 avg: -0.015685481684548513
2023-08-23 15:03:15.236945
[I 2023-08-23 15:03:15,236] Trial 10 finished with value: 0.19855372572229021 and parameters: {'learning_rate': 0.00040349799579019586, 'num_epochs': 921, 'batch_size': 63, 'dropout_rate': 0.49718892373760204, 'num_hidden_units': 124, 'weight_decay': 0.04210627255213978, 'l1_lambda': 0.029823743908320655, 'lr_scheduler': 'StepLR', 'optimizer': 'AdamW', 'num_layers': 4, 'step_size': 49, 'gamma': 0.15004179438869003}. Best is trial 10 with value: 0.19855372572229021.
VALIDATION Epoch: 1, """
