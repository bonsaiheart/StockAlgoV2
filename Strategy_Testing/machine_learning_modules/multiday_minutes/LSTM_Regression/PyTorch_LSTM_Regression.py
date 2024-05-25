import copy
import os
from datetime import datetime

import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torchmetrics
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
import os

print(os.getcwd())
study_name = "4hrminimize_valloss3"
DF_filename = r"../../../../data/historical_multiday_minute_DF/SPY_historical_multiday_min.csv"
Chosen_Predictor = [
    "Bonsai Ratio",
    "PCRv Up4",
    "PCRv Down4",
    "ITM PCRv Up2",
    "ITM PCRv Down2",
    "ITM PCRv Up4",
    "ITM PCRv Down4",
    "RSI14",
    "AwesomeOsc5_34",
    "RSI",
    "RSI2",
    "AwesomeOsc",
]
# ##had highest corr for 3-5 hours with these:
# Chosen_Predictor = ['Bonsai Ratio','Bonsai Ratio 2','ITM PCR-Vol','ITM PCRoi Up1', 'RSI14','AwesomeOsc5_34', 'Net_IV']
ml_dataframe = pd.read_csv(DF_filename)
print("Columns in Data:", ml_dataframe.columns)
# ml_dataframe["LastTradeTime"] = ml_dataframe["LastTradeTime"].apply(
#     lambda x: datetime.strptime(str(x), "%y%m%d_%H%M") if not pd.isna(x) else np.nan
# )
# ml_dataframe["LastTradeTime"] = ml_dataframe["LastTradeTime"].apply(
#     lambda x: x.timestamp()
# )
# ml_dataframe["LastTradeTime"] = ml_dataframe["LastTradeTime"] / (60 * 60 * 24 * 7)
ml_dataframe["ExpDate"] = ml_dataframe["ExpDate"].astype(float)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
r2_metric = torchmetrics.R2Score().to(device)
mae_metric = torchmetrics.MeanAbsoluteError().to(device)
mse_metric = torchmetrics.MeanSquaredError().to(device)
cells_forward_to_check = 240  # rows to check (minutes in this case)
# Target Calculation
ml_dataframe["Target_Change"] = (
    ml_dataframe["Current Stock Price"].pct_change(
        periods=cells_forward_to_check - 10 * -1
    )
    * 100
)
ml_dataframe["Target_Change"] = (
    ml_dataframe["Target_Change"].rolling(window=10, min_periods=1).mean()
)
ml_dataframe.dropna(subset=Chosen_Predictor + ["Target_Change"], inplace=True)
ml_dataframe.reset_index(drop=True, inplace=True)

y_change = ml_dataframe["Target_Change"].values.reshape(-1, 1)


X = ml_dataframe[Chosen_Predictor].copy()


test_set_percentage = 0.2  # Specify the percentage of the data to use as a test set
split_index = int(len(X) * (1 - test_set_percentage))

X_test = X[split_index:]
y_test = y_change[split_index:]
X = X[:split_index]
y_change = y_change[:split_index]
for column in X.columns:
    # Handle positive infinite values
    finite_max = X.loc[X[column] != np.inf, column].max()

    # Multiply by 1.5, considering the sign of the finite_max
    finite_max_adjusted = finite_max * 1.5 if finite_max > 0 else finite_max / 1.5

    # Apply adjustment to both X and X_test
    X.loc[X[column] == np.inf, column] = finite_max_adjusted
    X_test.loc[X_test[column] == np.inf, column] = finite_max_adjusted

    # Handle negative infinite values
    finite_min = X.loc[X[column] != -np.inf, column].min()

    # Multiply by 1.5, considering the sign of the finite_min
    finite_min_adjusted = finite_min * 1.5 if finite_min < 0 else finite_min / 1.5

    # Apply adjustment to both X and X_test
    X.loc[X[column] == -np.inf, column] = finite_min_adjusted
    X_test.loc[X_test[column] == -np.inf, column] = finite_min_adjusted

for column in Chosen_Predictor:
    (f"The data type of column {column} is {ml_dataframe[column].dtype}")


class LSTMRegressionNNSequence(nn.Module):
    def __init__(
        self, input_dim, num_hidden_units, dropout_rate, num_layers, sequence_length
    ):
        super(LSTMRegressionNNSequence, self).__init__()
        self.lstm = nn.LSTM(
            input_dim,
            num_hidden_units,
            num_layers=num_layers,
            dropout=dropout_rate,
            batch_first=True,
        )
        self.output_layer = nn.Linear(num_hidden_units, 1)
        self.num_hidden_units = num_hidden_units
        self.num_layers = num_layers
        self.sequence_length = (
            sequence_length  # Add the sequence length as an attribute
        )

    def forward(self, x, hidden_state=None):
        if hidden_state is None:
            h_0, c_0 = torch.zeros(
                self.num_layers, x.size(0), self.num_hidden_units
            ).to(x.device), torch.zeros(
                self.num_layers, x.size(0), self.num_hidden_units
            ).to(
                x.device
            )
        else:
            h_0, c_0 = hidden_state

        x, _ = self.lstm(x, (h_0, c_0))
        x = self.output_layer(x[:, -1, :])
        return x


def play_sound():
    # Play the sound file using the default audio player
    # os.system("aplay alert.wav")  # For Linux
    # os.system("afplay alert.wav")  # For macOS
    os.system("start C:\\Windows\\Media\\tada.wav")  # For Windows


def to_sequences(data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequence = data[i : i + seq_length]
        sequences.append(sequence.unsqueeze(0))
    return torch.cat(sequences, dim=0)


def train_model(hparams, X, y_change, trial=None):
    X_np = X.to_numpy()

    tscv = TimeSeriesSplit(n_splits=2)  # was 5
    total_mae = 0
    total_mse = 0
    best_model_state_dict = None  # Initialize the variable before the loop
    sequence_length = hparams["sequence_length"]
    total_r2 = 0
    num_folds = 0
    best_total_avg_val_loss = 10000000
    total_avg_val_loss = 0
    for fold, (train_index, val_index) in enumerate(tscv.split(X)):
        # print(f"Processing Fold {fold + 1}")  # After Each Fold

        X_train, X_val = X_np[train_index], X_np[val_index]
        y_train, y_val = y_change[train_index], y_change[val_index]
        # Scale the predictors
        global scaler_X
        scaler_X = MinMaxScaler(feature_range=(-1, 1))
        X_train_scaled = scaler_X.fit_transform(X_train)
        global scaler_y  # Scale the target
        # scaler_y = MinMaxScaler(feature_range=(-1, 1))
        scaler_y = RobustScaler()
        y_train_scaled = scaler_y.fit_transform(y_train)
        # TODO scaled or unscaled y?
        y_train_scaled = y_train_scaled
        y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
        # Assuming y_train_tensor originally has shape [2244, 1]
        y_train_tensor_trimmed = y_train_tensor[sequence_length:]

        y_val_scaled = scaler_y.transform(y_val)
        y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32).to(device)
        y_val_tensor_trimmed = y_val_tensor[sequence_length:]

        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
        X_train_sequences = to_sequences(
            X_train_tensor, sequence_length
        )  # Convert to sequences
        X_val_scaled = scaler_X.transform(X_val)
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
        X_val_sequences = to_sequences(
            X_val_tensor, sequence_length
        )  # Convert to sequences

        batch_size = hparams["batch_size"]

        train_dataset = TensorDataset(X_train_sequences, y_train_tensor_trimmed)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False, drop_last=False
        )
        val_dataset = TensorDataset(X_val_sequences, y_val_tensor_trimmed)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        patience = 5  # Number of epochs with no improvement to wait before stopping
        patience_counter = 0
        num_layers = hparams["num_layers"]
        hidden_size = hparams["num_hidden_units"]
        dropout_rate = hparams.get("dropout_rate", 0)

        model = LSTMRegressionNNSequence(
            X.shape[1],
            hidden_size,
            dropout_rate,
            num_layers,
            hparams["sequence_length"],
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
        fold_best_model_state_dict = None
        fold_bestmodel_avg_mae_score = float(
            "inf"
        )  # Initialize the variable for the best MAE score
        fold_bestmodel_avg_mse_score = float(
            "inf"
        )  # Initialize the variable for the best MSE score
        fold_bestmodel_r2_score = float(
            "-inf"
        )  # Initialize the variable for the best R² score
        if hparams.get("lr_scheduler") == "StepLR":
            lr_scheduler = StepLR(optimizer, hparams["step_size"], hparams["gamma"])
        elif hparams.get("lr_scheduler") == "ExponentialLR":
            lr_scheduler = ExponentialLR(optimizer, hparams["gamma"])
        elif hparams.get("lr_scheduler") == "ReduceLROnPlateau":
            lr_scheduler = ReduceLROnPlateau(
                optimizer, patience=hparams["patience"], mode="min"
            )
        best_val_loss = np.inf  # Best validation loss
        sum_val_loss = 0.0
        num_epochs_processed = 0
        l1_lambda = hparams.get("l1_lambda", 0)  # L1 regularization coefficient
        for epoch in range(num_epochs):
            # print(f"Epoch {epoch + 1}")  # After Each Epoch
            print(f"{epoch} of {num_epochs}")
            model.train()
            loss = None  # Define loss outside the inner loop
            for X_batch, y_batch in train_loader:
                actual_batch_size = X_batch.size(0)
                h_0 = torch.zeros(num_layers, actual_batch_size, hidden_size).to(device)
                c_0 = torch.zeros(num_layers, actual_batch_size, hidden_size).to(device)

                if optimizer_name == "LBFGS":

                    def closure():
                        nonlocal loss  # Refer to the outer scope's loss variable
                        optimizer.zero_grad()
                        outputs = model(X_batch, (h_0, c_0))

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
                    outputs = model(X_batch, (h_0, c_0))

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

            epoch_sum_val_loss = 0.0
            epoch_total_samples = 0
            mae_accum = 0
            mse_accum = 0
            r2_accum = 0  # Initialize the variable for accumulating R² values
            for i, (X_val_batch, y_val_batch) in enumerate(val_loader):
                if X_val_batch.size(0) != batch_size:
                    # Manually initialize hidden state for the last batch
                    h_0 = torch.zeros(num_layers, X_val_batch.size(0), hidden_size).to(
                        device
                    )
                    c_0 = torch.zeros(num_layers, X_val_batch.size(0), hidden_size).to(
                        device
                    )
                else:
                    # Initialize hidden state as usual
                    h_0 = torch.zeros(num_layers, batch_size, hidden_size).to(device)
                    c_0 = torch.zeros(num_layers, batch_size, hidden_size).to(device)

                with torch.no_grad():
                    val_outputs = model(
                        X_val_batch, (h_0, c_0)
                    )  # Pass hidden state if your model requires it
                    val_loss = criterion(val_outputs, y_val_batch)

                    mae_score = mae_metric(
                        val_outputs, y_val_batch
                    )  # Assuming you have mae_metric defined
                    mse_score = mse_metric(
                        val_outputs, y_val_batch
                    )  # Assuming you have mse_metric defined
                    try:
                        r2_score = r2_metric(
                            val_outputs, y_val_batch
                        )  # Assuming you have r2_metric defined
                    except ValueError as e:
                        print(e)
                        r2_score = float(
                            "-inf"
                        )  # or any other default value that makes sense in your context

                    r2_accum += r2_score.item() * len(y_val_batch)

                    mae_accum += mae_score.item() * len(y_val_batch)
                    mse_accum += mse_score.item() * len(y_val_batch)
                    # Add L1 regularization to validation loss
                    l1_val_reg = torch.tensor(0.0).to(device)
                    for param in model.parameters():
                        l1_val_reg += torch.norm(param, 1)
                    val_loss += l1_lambda * l1_val_reg

                    epoch_sum_val_loss += val_loss.item() * len(
                        y_val_batch
                    )  # Multiply by batch size
                    epoch_total_samples += len(y_val_batch)
            epoch_avg_r2 = (
                r2_accum / epoch_total_samples
            )  # Calculate average R² for the epoch
            # Calculate average validation loss for this epoch
            epoch_avg_val_loss = epoch_sum_val_loss / epoch_total_samples
            epoch_avg_mae = mae_accum / epoch_total_samples

            epoch_avg_mse = mse_accum / epoch_total_samples
            sum_val_loss += epoch_avg_val_loss
            num_epochs_processed += 1
            # After an epoch or batch, report intermediate result to Optuna
            # unique_step = fold * num_epochs + epoch
            # trial.report(epoch_avg_val_loss, step=unique_step)
            # # Prune trials that are unlikely to result in a good solution
            # if trial.should_prune():
            #     raise optuna.exceptions.TrialPruned()
            if epoch_avg_val_loss < best_val_loss:
                fold_bestmodel_avg_mae_score = epoch_avg_mae
                fold_bestmodel_avg_mse_score = epoch_avg_mse
                fold_bestmodel_r2_score = epoch_avg_r2  # Update the best R² score
                # epoch_best_val_loss = epoch_avg_val_loss

                patience_counter = 0  # Reset the early stopping counter
            else:
                patience_counter += 1
            # After your epoch loop, return both the best_model_state_dict and overall_avg_val_loss
            fold_avg_val_loss = sum_val_loss / num_epochs_processed
            if isinstance(lr_scheduler, ReduceLROnPlateau):
                # Step the learning rate scheduler
                lr_scheduler.step(epoch_avg_val_loss)
                patience_counter = (
                    0  # Reset patience counter if validation loss improves
                )
            else:
                patience_counter += 1

            if isinstance(lr_scheduler, ReduceLROnPlateau):
                lr_scheduler.step(epoch_avg_val_loss)
                patience_counter = (
                    0  # Reset patience counter if validation loss improves
                )

            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break
        total_avg_val_loss += fold_avg_val_loss
        total_mae += fold_bestmodel_avg_mae_score
        total_mse += fold_bestmodel_avg_mse_score
        total_r2 += fold_bestmodel_r2_score
        num_folds += 1
    bestmodel_avg_mae = total_mae / num_folds
    bestmodel_avg_mse = total_mse / num_folds
    bestmodel_avg_r2 = total_r2 / num_folds

    play_sound()

    bestmodel_avg_val_loss = total_avg_val_loss / num_folds
    # print(f"VALIDATION Epoch: {epoch + 1}, Training Loss: {loss}, Validation Loss: {val_loss_avg} ")
    # if r2_avg > 0:
    if total_avg_val_loss < best_total_avg_val_loss:
        best_total_avg_val_loss = total_avg_val_loss
        best_model_state_dict = copy.deepcopy(model.state_dict())
    #     play_sound()
    # print(bestmodel_avg_mae, bestmodel_avg_mse, best_model_state_dict, bestmodel_avg_val_loss, bestmodel_avg_r2)

    return (
        bestmodel_avg_mae,
        bestmodel_avg_mse,
        best_model_state_dict,
        bestmodel_avg_val_loss,
        bestmodel_avg_r2,
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
    print(datetime.now())

    # print(len(X_val_tensor))
    learning_rate = trial.suggest_float(
        "learning_rate", 0.001, 0.01, log=True
    )  # ,00001
    num_epochs = trial.suggest_int("num_epochs", 5, 50)
    # batch_size = trial.suggest_int("batch_size", 20, 3000)  # 10240  3437
    num_layers = trial.suggest_int("num_layers", 1, 2)
    batch_size = trial.suggest_int("batch_size", 20, 80)  # 500
    if num_layers > 1:
        dropout_rate = trial.suggest_float("dropout_rate", 0, 0.2)
    else:
        dropout_rate = 0
    num_hidden_units = trial.suggest_int("num_hidden_units", 100, 1000)  # 2000
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1)
    l1_lambda = trial.suggest_float("l1_lambda", 1e-5, 1e-1)
    sequence_length = trial.suggest_int("sequence_length", 20, 30)

    optimizer_name = trial.suggest_categorical(
        "optimizer",
        [
            "Adam",
        ],
    )  # "SGD", "RMSprop"

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
    # model = RegressionNN(X.shape[1], num_hidden_units, dropout_rate, num_layers).to(device)
    #
    # # Then pass the model's parameters to create_optimizer
    # optimizer = create_optimizer(optimizer_name, learning_rate, momentum, weight_decay, model.parameters())

    # Now proceed with the rest of the code
    ...

    lr_scheduler_name = trial.suggest_categorical(
        "lr_scheduler", ["StepLR", "ReduceLROnPlateau"]
    )  #'ExponentialLR',

    if lr_scheduler_name == "StepLR":
        step_size = trial.suggest_int("step_size", 5, 50)
        gamma = trial.suggest_float("gamma", 0.1, 1)
    elif lr_scheduler_name == "ExponentialLR":
        gamma = trial.suggest_float("gamma", 0.1, 1)
    elif lr_scheduler_name == "ReduceLROnPlateau":
        patience = trial.suggest_int("patience", 5, 20)

    # Call the train_model function with the current hyperparameters
    (
        bestmodel_avg_mae_score,
        bestmodel_avg_mse_score,
        best_model_state_dict,
        overall_avg_val_loss,
        bestmodel_r2_score,
    ) = train_model(
        {
            "l1_lambda": l1_lambda,
            "learning_rate": learning_rate,
            "optimizer": optimizer_name,
            "num_layers": num_layers,
            "sequence_length": sequence_length,
            "momentum": momentum,  # Include optimizer name here
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "dropout_rate": dropout_rate,
            "num_hidden_units": num_hidden_units,
            "weight_decay": weight_decay,
            "lr_scheduler": lr_scheduler_name,  # Pass the scheduler instance here
            "patience": patience,  # Pass the value directly
            "gamma": gamma,  # Pass the value directly
            "step_size": step_size,  # Add step_size parameter for StepLR
            # Add more hyperparameters as needed
        },
        X,
        y_change,
        trial=trial,
    )

    # Reshape your scalar into a 2D array with 1 row and 1 column
    reshaped_mae = np.array([bestmodel_avg_mae_score]).reshape(-1, 1)

    # Perform the inverse transform
    inverse_scaled_mae = scaler_y.inverse_transform(reshaped_mae)

    print("Inverse Scaled MAE:", inverse_scaled_mae)

    print(
        "best mae score: ",
        bestmodel_avg_mae_score,
        "Inverse Scaled MAE: ",
        inverse_scaled_mae,
        " best mse score: ",
        bestmodel_avg_mse_score,
        "smallest val loss: ",
        overall_avg_val_loss,
        "best r2 avg:",
        bestmodel_r2_score,
    )

    return (
        overall_avg_val_loss  # Note this is actually criterion, which is currently mae.
    )
    #    # return prec_score  # Optuna will try to maximize this value


# #TODO Comment out to skip the hyperparameter selection.  Swap "best_params".

# study = optuna.create_study(direction="minimize",pruner = Success
# iveHalvingPruner(min_resource=1, reduction_factor=4)
# )

try:
    study = optuna.load_study(
        study_name=f"{study_name}", storage=f"sqlite:///{study_name}.db"
    )
    print("Study Loaded.")
except KeyError:
    study = optuna.create_study(
        direction="minimize",
        study_name=f"{study_name}",
        storage=f"sqlite:///{study_name}.db",
    )
    "Keyerror, new optuna study created."

# Continue with optimization
study.optimize(objective, n_trials=250)


best_params = study.best_params
print("best_params: ", best_params)
##TODO
# best_params = {'learning_rate': 1.1292399886886521e-05, 'num_epochs': 969, 'batch_size': 2905,
#                'dropout_rate': 0.4993665871002279, 'num_hidden_units': 2946, 'weight_decay': 0.0003679260243350177,
#                'l1_lambda': 0.0017721772314088363, 'lr_scheduler': 'ReduceLROnPlateau', 'optimizer': 'Adam',
#                'num_layers': 5, 'patience': 20}
## Train the model with the best hyperparameters
print("~~~~training model using best params.~~~~")
(
    bestmodel_avg_mae_score,
    bestmodel_avg_mse_score,
    best_model_state_dict,
    overall_avg_val_loss,
    bestmodel_r2_score,
) = train_model(best_params, X, y_change)
model_up_nn = LSTMRegressionNNSequence(
    X.shape[1],
    best_params["num_hidden_units"],
    best_params.get("dropout_rate", 0),
    best_params["num_layers"],
    best_params["sequence_length"],
).to(device)
# Load the saved state_dict into the model

model_up_nn.load_state_dict(best_model_state_dict)
model_up_nn.eval()
X_test_scaled = scaler_X.transform(X_test)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
X_test_sequences = to_sequences(
    X_test_tensor, best_params["sequence_length"]
)  # Convert to sequences
predicted_values = model_up_nn(X_test_sequences)
test_r2_accum = 0.0
test_mae_accum = 0.0
test_mse_accum = 0.0
test_total_samples = 0
batch_size = best_params.get("batch_size")
hidden_size = best_params.get("num_hidden_units")
with torch.no_grad():
    for i in range(0, len(X_test_sequences), batch_size):
        batch_X = X_test_sequences[i : i + batch_size]

        # Initialize hidden state based on batch size
        batch_size = batch_X.size(0)
        h_0 = torch.zeros(best_params.get("num_layers"), batch_size, hidden_size).to(
            device
        )
        c_0 = torch.zeros(best_params.get("num_layers"), batch_size, hidden_size).to(
            device
        )

        # Forward pass
        batch_outputs = model_up_nn(batch_X, (h_0, c_0))

        # Calculate metrics for the batch
        batch_r2 = r2_metric(batch_outputs, y_test[i : i + batch_size])
        batch_mae = mae_metric(batch_outputs, y_test[i : i + batch_size])
        batch_mse = mse_metric(batch_outputs, y_test[i : i + batch_size])

        test_r2_accum += batch_r2.item() * batch_size
        test_mae_accum += batch_mae.item() * batch_size
        test_mse_accum += batch_mse.item() * batch_size
        test_total_samples += batch_size

# Calculate average metrics
test_r2 = test_r2_accum / test_total_samples
test_mae = test_mae_accum / test_total_samples
test_mse = test_mse_accum / test_total_samples
print(
    f"Test R2 Score: {test_r2}",
    f"Test mae Score: {test_mae}",
    f"Test mse Score: {test_mse}",
)

input_val = input("Would you like to save these models? y/n: ").upper()
if input_val == "Y":
    model_summary = input("Save this set of models as: ")
    model_directory = os.path.join("../../Trained_Models", f"{model_summary}")
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    model_filename_up = os.path.join(model_directory, "target_up.pth")
    save_dict = {
        "model_class": model_up_nn.__class__.__name__,
        "features": Chosen_Predictor,
        "input_dim": X.shape[1],
        "l1_lambda": best_params.get("l1_lambda"),
        "learning_rate": best_params.get("learning_rate"),
        "optimizer": best_params.get("optimizer"),
        "num_layers": best_params.get("num_layers"),
        "sequence_length": best_params.get("sequence_length"),
        "momentum": best_params.get("momentum"),
        "num_epochs": best_params.get("num_epochs"),
        "batch_size": best_params.get("batch_size"),
        "dropout_rate": best_params.get("dropout_rate"),
        "num_hidden_units": best_params.get("num_hidden_units"),
        "weight_decay": best_params.get("weight_decay"),
        "lr_scheduler": best_params.get("scheduler"),
        "patience": best_params.get("patience"),
        "gamma": best_params.get("gamma"),
        "step_size": best_params.get("step_size"),
        "scaler_X_min": scaler_X.min_,
        "scaler_X_scale": scaler_X.scale_,
        "scaler_y_min": scaler_y.min_,
        "scaler_y_scale": scaler_y.scale_,
        "model_state_dict": model_up_nn.state_dict(),
    }

    # Remove keys with None values
    save_dict = {key: value for key, value in save_dict.items() if value is not None}

    # Save the dictionary if it's not empty
    if save_dict:
        torch.save(save_dict, model_filename_up)

with open(f"../../../Trained_Models/{model_summary}/info.txt", "w") as info_txt:
    info_txt.write("This file contains information about the model.\n\n")
    info_txt.write(
        f"File analyzed: {DF_filename}\nCells_Foward_to_check: {cells_forward_to_check}\n\n"
    )
    info_txt.write(
        f"Metrics for Target_Up:bestmodel mae score:  {bestmodel_avg_mae_score} bestmodel mse score:  {bestmodel_avg_mse_score} overall val loss best modle: {overall_avg_val_loss} bestmodel r2: {bestmodel_r2_score}"
    )
    info_txt.write(
        f"Predictors: {Chosen_Predictor}\n\n\n" f"Best Params: {best_params}\n\n\n"
    )
"""to get preds from trained model # Preprocess input data
X_input = ...  # Your input data
X_input_scaled = scaler_X.transform(X_input)
X_input_tensor = torch.tensor(X_input_scaled, dtype=torch.float32).to(device)
X_input_sequences = to_sequences(X_input_tensor, sequence_length)  # Convert to sequences

# Load the trained model's state dictionary
trained_model_state_dict = ...  # Load the state dict saved after training
model.load_state_dict(trained_model_state_dict)
model.to(device)
model.eval()

# Forward-pass to make predictions
with torch.no_grad():
    h_0, c_0 = torch.zeros(num_layers, X_input_sequences.size(0), hidden_size).to(device), \
               torch.zeros(num_layers, X_input_sequences.size(0), hidden_size).to(device)
    predictions = model(X_input_sequences, (h_0, c_0))

# Assuming predictions is a tensor, you can convert it back to the original scale
predictions_scaled = predictions.squeeze().cpu().numpy()
predictions_original_scale = scaler_y.inverse_transform(predictions_scaled)

# Now predictions_original_scale contains your final predictions
"""
