import copy
import os
from datetime import datetime
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torchmetrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau
from torch.utils.data import Dataset, TensorDataset, DataLoader
from UTILITIES.logger_config import logger

DF_filename = r"../../../data/historical_multiday_minute_DF/TSLA_historical_multiday_min230817.csv"
Chosen_Predictor = ['ExpDate', 'LastTradeTime', 'Current Stock Price',
                    'Current SP % Change(LAC)', 'Maximum Pain', 'Bonsai Ratio',
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
# ##had highest corr for 3-5 hours with these:
# Chosen_Predictor = ['Bonsai Ratio','Bonsai Ratio 2','ITM PCR-Vol','ITM PCRoi Up1', 'RSI14','AwesomeOsc5_34', 'Net_IV']
ml_dataframe = pd.read_csv(DF_filename)
print('Columns in Data:', ml_dataframe.columns)

ml_dataframe['ExpDate'] = ml_dataframe['ExpDate'].astype(float)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Device:', device)
r2_metric = torchmetrics.R2Score().to(device)
mae_metric = torchmetrics.MeanAbsoluteError().to(device)
mse_metric = torchmetrics.MeanSquaredError().to(device)
cells_forward_to_check = 240  # rows to check (minutes in this case)
# Target Calculation
ml_dataframe['Target_Change'] = ml_dataframe['Current Stock Price'].pct_change(
    periods=cells_forward_to_check * -1) * 100
ml_dataframe.dropna(subset=Chosen_Predictor + ["Target_Change"], inplace=True)
ml_dataframe.reset_index(drop=True, inplace=True)
ml_dataframe['LastTradeTime'] = ml_dataframe['LastTradeTime'].apply(
    lambda x: datetime.strptime(str(x), '%y%m%d_%H%M') if not pd.isna(x) else np.nan)
ml_dataframe['LastTradeTime'] = ml_dataframe['LastTradeTime'].apply(lambda x: x.timestamp())

y_change = ml_dataframe["Target_Change"].values.reshape(-1, 1)
X = ml_dataframe[Chosen_Predictor]
large_number = 1e100
small_number = -1e100
nan_indices = np.argwhere(np.isnan(X))
inf_indices = np.argwhere(np.isinf(X))
print("NaN values found at indices:" if len(nan_indices) > 0 else "No NaN values found.")
print("Infinite values found at indices:" if len(inf_indices) > 0 else "No infinite values found.")
X = X.replace([np.inf], large_number)
X = X.replace([-np.inf], small_number)

trainsizepercent = .7
valsizepercent = .2

X_train, X_temp, y_train, y_temp = train_test_split(X, y_change, test_size=1 - trainsizepercent, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1 - valsizepercent / (1 - trainsizepercent),
                                                random_state=42)
print(min(y_train))
print(min(y_val))
print(min(y_test))
# Scale the predictors
scaler_X = MinMaxScaler(feature_range=(-1, 1))
X_train_scaled = scaler_X.fit_transform(X_train)
X_val_scaled = scaler_X.transform(X_val)
X_test_scaled = scaler_X.transform(X_test)

# Scale the target
# scaler_y = MinMaxScaler(feature_range=(-1, 1))
scaler_y = RobustScaler()
# y_train_scaled = scaler_y.fit_transform(y_train)
# y_val_scaled = scaler_y.transform(y_val)
# y_test_scaled = scaler_y.transform(y_test)
#TODO scaled or unscaled y?
y_train_scaled = y_train
y_val_scaled = y_val
y_test_scaled = y_test
print(min(y_train))
print(min(y_val))
print(min(y_test))
print("X_train_scaled shape:", X_train_scaled.shape)
print("X_val_scaled shape:", X_val_scaled.shape)
print("X_test_scaled shape:", X_test_scaled.shape)

print("y_train_scaled shape:", y_train_scaled.shape)
print("y_val_scaled shape:", y_val_scaled.shape)
print("y_test_scaled shape:", y_test_scaled.shape)
# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float).to(
    device)  # Convert to tensor and move to the same device
for column in Chosen_Predictor:
    print(f"The data type of column {column} is {ml_dataframe[column].dtype}")

y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float).to(device)




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


def train_model(hparams, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor):
    # rest of the code remains the same, remove the optimizer creation part inside this function
    batch_size = hparams["batch_size"]
    # print(X_train_tensor.shape)
    # print(y_train_tensor.shape)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    patience = 5  # Number of epochs with no improvement to wait before stopping
    lr_scheduler = hparams.get("lr_scheduler")  # Get the lr_scheduler from the hyperparameters

    patience_counter = 0
    num_layers = hparams["num_layers"]
    model = RegressionNN(X_train.shape[1], hparams["num_hidden_units"], hparams['dropout_rate'], num_layers).to(device)
    optimizer = create_optimizer(hparams["optimizer"], hparams["learning_rate"], hparams.get("momentum", 0),
                                 hparams.get("weight_decay"), model.parameters())

    model.train()
    # criterion = nn.MSELoss()
    criterion = nn.L1Loss()  # mae
    optimizer_name = hparams["optimizer"]
    num_epochs = hparams["num_epochs"]
    best_model_state_dict = None
    bestmodel_avg_mae_score = float('inf')  # Initialize the variable for the best MAE score
    bestmodel_avg_mse_score = float('inf')  # Initialize the variable for the best MSE score
    bestmodel_r2_score = float('-inf')  # Initialize the variable for the best R² score

    best_val_loss = np.inf  # Best validation loss
    sum_val_loss = 0.0
    num_epochs_processed = 0
    l1_lambda = hparams.get("l1_lambda", 0)  # L1 regularization coefficient
    for epoch in range(num_epochs):

        epoch_sum_val_loss = 0.0
        epoch_total_samples = 0
        mae_accum = 0
        mse_accum = 0
        model.train()
        # Training step
        loss = None  # Define loss outside the inner loop
        for X_batch, y_batch in train_loader:
            if optimizer_name == "LBFGS":
                def closure():
                    nonlocal loss  # Refer to the outer scope's loss variable
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    # outputs_scaled = scaler_y.transform(outputs.detach().cpu().numpy())
                    # outputs_tensor = torch.tensor(outputs_scaled, dtype=torch.float32).to(device)
                    # outputs = outputs.squeeze(1)
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
        if lr_scheduler is not None:
            if isinstance(lr_scheduler, StepLR) or isinstance(lr_scheduler, ExponentialLR):
                lr_scheduler.step()
        model.eval()
        # Validation step


        r2_accum = 0  # Initialize the variable for accumulating R² values

        for X_val_batch, y_val_batch in val_loader:
            with torch.no_grad():
                val_outputs = model(X_val_batch)
                val_loss = criterion(val_outputs, y_val_batch)
                # print("Outputs: ",val_outputs[-5:].tolist())
                # print("Truth: ",y_val_batch[-5:].tolist())
                mae_score = mae_metric(val_outputs, y_val_batch)  # Assuming you have mae_metric defined
                mse_score = mse_metric(val_outputs, y_val_batch)  # Assuming you have mse_metric defined
                r2_score = r2_metric(val_outputs, y_val_batch)  # Assuming you have r2_metric defined
                r2_accum += r2_score.item() * len(y_val_batch)

                mae_accum += mae_score.item() * len(y_val_batch)
                mse_accum += mse_score.item() * len(y_val_batch)

                # Add L1 regularization to validation loss
                l1_val_reg = torch.tensor(0.).to(device)
                for param in model.parameters():
                    l1_val_reg += torch.norm(param, 1)
                val_loss += l1_lambda * l1_val_reg

                epoch_sum_val_loss += val_loss.item() * len(y_val_batch)  # Multiply by batch size
                epoch_total_samples += len(y_val_batch)
        epoch_avg_r2 = r2_accum / epoch_total_samples  # Calculate average R² for the epoch

            # Calculate average validation loss for this epoch
        epoch_avg_val_loss = epoch_sum_val_loss / epoch_total_samples
        epoch_avg_mae = mae_accum / epoch_total_samples
        epoch_avg_mse = mse_accum / epoch_total_samples
        sum_val_loss += epoch_avg_val_loss
        num_epochs_processed += 1
        if epoch_avg_val_loss < best_val_loss:
            best_val_loss = epoch_avg_val_loss
            bestmodel_avg_mae_score = epoch_avg_mae
            bestmodel_avg_mse_score = epoch_avg_mse
            bestmodel_r2_score = epoch_avg_r2  # Update the best R² score

            best_model_state_dict = copy.deepcopy(model.state_dict())
            patience_counter = 0  # Reset the early stopping counter
        else:
            patience_counter += 1

        # After your epoch loop, return both the best_model_state_dict and overall_avg_val_loss
        overall_avg_val_loss = sum_val_loss / num_epochs_processed
        if isinstance(lr_scheduler, ReduceLROnPlateau):
            # Step the learning rate scheduler
            lr_scheduler.step(epoch_avg_val_loss)
            patience_counter = 0  # Reset patience counter if validation loss improves
        else:

            patience_counter += 1

        if isinstance(lr_scheduler, ReduceLROnPlateau):
            lr_scheduler.step(epoch_avg_val_loss)
            patience_counter = 0  # Reset patience counter if validation loss improves

        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break
    overall_avg_val_loss = sum_val_loss / num_epochs_processed


    # print(f"VALIDATION Epoch: {epoch + 1}, Training Loss: {loss}, Validation Loss: {val_loss_avg} ")
    # if r2_avg > 0:
    #     play_sound()

    return bestmodel_avg_mae_score, bestmodel_avg_mse_score, best_model_state_dict, overall_avg_val_loss, bestmodel_r2_score


def create_optimizer(optimizer_name, learning_rate, momentum, weight_decay, model_parameters):
    if optimizer_name == "SGD":
        return torch.optim.SGD(model_parameters, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == "Adam":
        return torch.optim.Adam(model_parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "AdamW":
        return torch.optim.AdamW(model_parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "RMSprop":
        return torch.optim.RMSprop(model_parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "Adagrad":
        return torch.optim.Adagrad(model_parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "Adamax":
        return torch.optim.Adamax(model_parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "Adadelta":
        return torch.optim.Adadelta(model_parameters, lr=learning_rate, weight_decay=weight_decay)
    elif optimizer_name == "LBFGS":
        return torch.optim.LBFGS(model_parameters, lr=learning_rate, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


def objective(trial):
    print(datetime.now())

    # print(len(X_val_tensor))
    learning_rate = trial.suggest_float("learning_rate", .00001, 0.01, log=True)  # 0003034075497582067
    num_epochs = trial.suggest_int("num_epochs", 5, 1000)  # 3800 #230  291  400-700
    # batch_size = trial.suggest_int("batch_size", 20, 3000)  # 10240  3437
    batch_size = trial.suggest_int('batch_size', 20, len(X_val) - 1)
    dropout_rate = trial.suggest_float("dropout_rate", 0, .5)  # 30311980533100547  16372372692286732
    num_hidden_units = trial.suggest_int("num_hidden_units", 100, 4000)  # 2560 #83 125 63  #7000
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1)  # Adding L2 regularization parameter
    l1_lambda = trial.suggest_float("l1_lambda", 1e-5, 1e-1)  # l1 regssss
    lr_scheduler_name = trial.suggest_categorical('lr_scheduler', ['StepLR', 'ExponentialLR', 'ReduceLROnPlateau'])

    optimizer_name = trial.suggest_categorical("optimizer", ["SGD", "Adam", "AdamW", "RMSprop"])
    num_layers = trial.suggest_int("num_layers", 1, 5)  # example

    if optimizer_name == "SGD":
        momentum = trial.suggest_float('momentum', 0, 0.9)  # Only applicable if using SGD with momentum
    else:

        momentum = 0  # Default value if not using SGD
    patience = None  # Define a default value outside the conditional block
    step_size = None  # Define a default value for step_size as well
    gamma = None
    # Inside the objective function, create the model first
    model = RegressionNN(X_train.shape[1], num_hidden_units, dropout_rate, num_layers).to(device)

    # Then pass the model's parameters to create_optimizer
    optimizer = create_optimizer(optimizer_name, learning_rate, momentum, weight_decay, model.parameters())

    # Now proceed with the rest of the code
    ...

    if lr_scheduler_name == 'StepLR':
        step_size = trial.suggest_int('step_size', 5, 50)
        gamma = trial.suggest_float('gamma', 0.1, 1)

        scheduler = StepLR(optimizer, step_size, gamma)  # Use optimizer, not optimizer_name
    elif lr_scheduler_name == 'ExponentialLR':
        gamma = trial.suggest_float('gamma', 0.1, 1)
        scheduler = ExponentialLR(optimizer, gamma)  # Use optimizer
    elif lr_scheduler_name == 'ReduceLROnPlateau':
        patience = trial.suggest_int('patience', 5, 20)
        scheduler = ReduceLROnPlateau(optimizer, patience=patience, mode='min')  # Use optimizer

    # Call the train_model function with the current hyperparameters
    bestmodel_avg_mae_score, bestmodel_avg_mse_score, best_model_state_dict, overall_avg_val_loss, bestmodel_r2_score = train_model(

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
        X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor
    )
    print("best mae score: ", bestmodel_avg_mae_score, "best mse score: ", bestmodel_avg_mse_score, "smallest val loss: ",
          overall_avg_val_loss, "best r2 avg:", bestmodel_r2_score)

    return overall_avg_val_loss  # Note this is actually criterion, which is currently mae.
    #    # return prec_score  # Optuna will try to maximize this value


# #TODO Comment out to skip the hyperparameter selection.  Swap "best_params".
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)
best_params = study.best_params
print(best_params)
##TODO
# best_params = {'learning_rate': 1.1292399886886521e-05, 'num_epochs': 969, 'batch_size': 2905,
#                'dropout_rate': 0.4993665871002279, 'num_hidden_units': 2946, 'weight_decay': 0.0003679260243350177,
#                'l1_lambda': 0.0017721772314088363, 'lr_scheduler': 'ReduceLROnPlateau', 'optimizer': 'Adam',
#                'num_layers': 5, 'patience': 20}
## Train the model with the best hyperparameters
print("~~~~training model using best params.~~~~")
bestmodel_avg_mae_score, bestmodel_avg_mse_score, best_model_state_dict, overall_avg_val_loss, bestmodel_r2_score = train_model(
    best_params, X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor)
model_up_nn = RegressionNN(X_train.shape[1], best_params["num_hidden_units"],
                           best_params["dropout_rate"], best_params["num_layers"]).to(device)
# Load the saved state_dict into the model
model_up_nn.load_state_dict(best_model_state_dict)
model_up_nn.eval()

predicted_values = model_up_nn(X_test_tensor)

# predicted_values_scaled = scaler_y.transform(predicted_values.detach().cpu().numpy())
# predicted_values_tensor = torch.tensor(predicted_values_scaled, dtype=torch.float).to(device)

print("MIN YTEST: ", min(y_test_tensor), " MAX YTEST: ", max(y_test_tensor))
print("MIN pred: ", min(predicted_values), " MAX pred: ", max(predicted_values))
print(predicted_values, y_test_tensor)

# Calculate R2 score
r2 = r2_metric(predicted_values, y_test_tensor)
mae = mae_metric(predicted_values, y_test_tensor)
mse = mse_metric(predicted_values, y_test_tensor)

print(f"R2 Score: {r2}", f"mae Score: {mae}",f"mse Score: {mse}")
# print('selected features: ',selected_features)

input_val = input("Would you like to save these models? y/n: ").upper()
if input_val == "Y":
    model_summary = input("Save this set of models as: ")
    model_directory = os.path.join("../../Trained_Models", f"{model_summary}")
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    model_filename_up = os.path.join(model_directory, "target_up.pth")
    save_dict = {
        'model_class': model_up_nn.__class__.__name__,
        'features': Chosen_Predictor,
        'input_dim': X_train.shape[1],
        "l1_lambda": best_params.get('l1_lambda'),
        "learning_rate": best_params.get('learning_rate'),
        "optimizer": best_params.get('optimizer'),
        "num_layers": best_params.get('num_layers'),
        "momentum": best_params.get('momentum'),
        "num_epochs": best_params.get('num_epochs'),
        "batch_size": best_params.get('batch_size'),
        "dropout_rate": best_params.get('dropout_rate'),
        "num_hidden_units": best_params.get('num_hidden_units'),
        "weight_decay": best_params.get('weight_decay'),
        "lr_scheduler": best_params.get('scheduler'),
        "patience": best_params.get('patience'),
        "gamma": best_params.get('gamma'),
        "step_size": best_params.get('step_size'),
        'scaler_X_min': scaler_X.min_,
        'scaler_X_scale': scaler_X.scale_,
        'scaler_y_min': scaler_y.min_,
        'scaler_y_scale': scaler_y.scale_,
        'model_state_dict': model_up_nn.state_dict(),
    }

    # Remove keys with None values
    save_dict = {key: value for key, value in save_dict.items() if value is not None}

    # Save the dictionary if it's not empty
    if save_dict:
        torch.save(save_dict, model_filename_up)

with open(f"../../Trained_Models/{model_summary}/info.txt", "w") as info_txt:
    info_txt.write("This file contains information about the model.\n\n")
    info_txt.write(
        f"File analyzed: {DF_filename}\nCells_Foward_to_check: {cells_forward_to_check}\n\n"
    )
    info_txt.write(
        f"Metrics for Target_Up:bestmodel mae score:  {bestmodel_avg_mae_score} bestmodel mse score:  {bestmodel_avg_mse_score} overall val loss best modle: {overall_avg_val_loss} bestmodel r2: {bestmodel_r2_score}")
    info_txt.write(
        f"Predictors: {Chosen_Predictor}\n\n\n"
        f"Best Params: {best_params}\n\n\n")
