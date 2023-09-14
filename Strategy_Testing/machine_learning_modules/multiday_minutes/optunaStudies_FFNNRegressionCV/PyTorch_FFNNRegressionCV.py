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

DF_filename = r"../../../../data/historical_multiday_minute_DF/SPY_historical_multiday_min.csv"
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
    periods=cells_forward_to_check - 10 * -1) * 100
# 8.31 got rid of rolling ave.
# ml_dataframe['Target_Change'] = ml_dataframe['Target_Change'].rolling(window=10,min_periods=1).mean()
ml_dataframe.dropna(subset=Chosen_Predictor + ["Target_Change"], inplace=True)
ml_dataframe.reset_index(drop=True, inplace=True)
ml_dataframe['LastTradeTime'] = ml_dataframe['LastTradeTime'].apply(
    lambda x: datetime.strptime(str(x), '%y%m%d_%H%M') if not pd.isna(x) else np.nan)
ml_dataframe['LastTradeTime'] = ml_dataframe['LastTradeTime'].apply(lambda x: x.timestamp())

y_change = ml_dataframe["Target_Change"].values.reshape(-1, 1)
# for col in ml_dataframe.columns:
#     finite_max = ml_dataframe.loc[ml_dataframe[col] != np.inf, col].max()
#     ml_dataframe.loc[ml_dataframe[col] == np.inf, col] = finite_max
X = ml_dataframe[Chosen_Predictor].copy()
# threshold = 1e10
# X[Chosen_Predictor] = np.clip(X[Chosen_Predictor], -threshold, threshold)
for column in X.columns:
    # Handle positive infinite values
    finite_max = X.loc[X[column] != np.inf, column].max()

    # Multiply by 1.5, considering the sign of the finite_max
    finite_max_adjusted = finite_max * 1.5 if finite_max > 0 else finite_max / 1.5

    X.loc[X[column] == np.inf, column] = finite_max_adjusted

    # Handle negative infinite values
    finite_min = X.loc[X[column] != -np.inf, column].min()

    # Multiply by 1.5, considering the sign of the finite_min
    finite_min_adjusted = finite_min * 1.5 if finite_min < 0 else finite_min / 1.5

    X.loc[X[column] == -np.inf, column] = finite_min_adjusted

nan_indices = np.argwhere(np.isnan(X.to_numpy()))  # Convert DataFrame to NumPy array
inf_indices = np.argwhere(np.isinf(X.to_numpy()))  # Convert DataFrame to NumPy array
neginf_indices = np.argwhere(np.isneginf(X.to_numpy()))  # Convert DataFrame to NumPy array

print("NaN values found at indices:" if len(nan_indices) > 0 else "No NaN values found.")
print("Infinite values found at indices:" if len(inf_indices) > 0 else "No infinite values found.")
print("Negative Infinite values found at indices:" if len(neginf_indices) > 0 else "No negative infinite values found.")

# Replace infinite values with large_number, -infinite values with small_number


test_set_percentage = 0.1  # Specify the percentage of the data to use as a test set
split_index = int(len(X) * (1 - test_set_percentage))
entire_scaler_X = RobustScaler()
entire_scaler_y = RobustScaler()
entire_scaler_X = entire_scaler_X.fit(X)
entire_scaler_y = entire_scaler_y.fit(y_change)

X_test = X[split_index:]
y_test = y_change[split_index:]
X = X[:split_index]
y_change = y_change[:split_index]
test_scaler_X = RobustScaler()
test_scaler_y = RobustScaler()
test_scaler_X.fit(X)
X_test_scaled = test_scaler_X.transform(X_test)
test_scaler_y.fit(y_change)
y_test_scaled = test_scaler_y.transform(y_test)
for column in Chosen_Predictor:
    print(f"The data type of column {column} is {ml_dataframe[column].dtype}")


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


def train_model(hparams, X, y_change, trial=None):
    X_np = X.to_numpy()
    tscv = TimeSeriesSplit(n_splits=5)

    total_mae, total_mse, total_r2, total_avg_val_loss = 0, 0, 0, 0

    num_folds = 0
    total_best_fold_val_loss = 0
    best_model_state_dict_across_folds = None
    best_val_loss_across_folds = float('inf')

    for fold, (train_index, val_index) in enumerate(tscv.split(X)):
        # Data preparation
        X_train, X_val = X_np[train_index], X_np[val_index]
        y_train, y_val = y_change[train_index], y_change[val_index]

        scaler_y = RobustScaler()
        scaler_X = RobustScaler()
        X_train_scaled = scaler_X.fit_transform(X_train)
        y_train_scaled = scaler_y.fit_transform(y_train)
        X_val_scaled = scaler_X.transform(X_val)
        y_val_scaled = scaler_y.transform(y_val)

        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32).to(device)

        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=hparams["batch_size"], shuffle=False, drop_last=True)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=hparams["batch_size"])

        # Model, optimizer, and loss initialization
        model = RegressionNN(X_train.shape[1], hparams["num_hidden_units"], hparams['dropout_rate'],
                             hparams["num_layers"]).to(device)
        optimizer = create_optimizer(hparams["optimizer"], hparams["learning_rate"], hparams.get("momentum", 0),
                                     hparams.get("weight_decay"), model.parameters())
        criterion = nn.L1Loss()
        l1_lambda = hparams.get("l1_lambda", 0)

        # Learning rate scheduler
        if hparams.get("lr_scheduler") == 'StepLR':
            lr_scheduler = StepLR(optimizer, hparams["step_size"], hparams["gamma"])
        elif hparams.get("lr_scheduler") == 'ExponentialLR':
            lr_scheduler = ExponentialLR(optimizer, hparams["gamma"])
        elif hparams.get("lr_scheduler") == 'ReduceLROnPlateau':
            lr_scheduler = ReduceLROnPlateau(optimizer, patience=hparams["patience"], mode='min')

        # Training and validation
        best_fold_val_loss = np.inf
        sum_val_loss, num_epochs_processed = 0.0, 0
        early_stop_rounds = 10 # Number of epochs with no improvement to wait before stopping
        early_stop_rounds_counter = 0
        #TODO maybe jsut initialize to None
        fold_bestmodel_avg_mae_score = float('inf')  # Initialize to a large value
        fold_bestmodel_avg_mse_score = float('inf')  # Initialize to a large value
        fold_bestmodel_r2_score = float('-inf')  # Initialize to a small value

        l1_lambda = hparams.get("l1_lambda", 0)  # L1 regularization coefficient
        for epoch in range(hparams["num_epochs"]):
            # print(epoch)
            model.train()
            for X_batch, y_batch in train_loader:
                if hparams["optimizer"] == "LBFGS":
                    def closure():
                        nonlocal loss
                        optimizer.zero_grad()
                        outputs = model(X_batch)
                        loss = criterion(outputs, y_batch)
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
                    l1_reg = torch.tensor(0., requires_grad=True).to(device)
                    for param in model.parameters():
                        l1_reg += torch.norm(param, 1)
                    loss += l1_lambda * l1_reg
                    loss.backward()
                    optimizer.step()
            if lr_scheduler is not None and (
                    isinstance(lr_scheduler, StepLR) or isinstance(lr_scheduler, ExponentialLR)):
                lr_scheduler.step()

                # Validation step
            model.eval()
            epoch_sum_val_loss_accum, epoch_total_samples, mae_accum, mse_accum = 0.0, 0, 0, 0
            all_val_outputs, all_val_labels = [], []

            for X_val_batch, y_val_batch in val_loader:
                with torch.no_grad():
                    val_outputs = model(X_val_batch)
                    all_val_outputs.extend(val_outputs.tolist())
                    all_val_labels.extend(y_val_batch.tolist())
                    val_loss = criterion(val_outputs, y_val_batch)
                    # print(val_loss)
                    mae_score = mae_metric(val_outputs, y_val_batch)
                    mse_score = mse_metric(val_outputs, y_val_batch)
                    mae_accum += mae_score.item() * len(y_val_batch)
                    mse_accum += mse_score.item() * len(y_val_batch)
                    epoch_sum_val_loss_accum += val_loss.item() * len(y_val_batch)
                    epoch_total_samples += len(y_val_batch)

            epoch_avg_val_loss = epoch_sum_val_loss_accum / epoch_total_samples
            sum_val_loss += epoch_avg_val_loss
            num_epochs_processed += 1

            if epoch_avg_val_loss < best_fold_val_loss:
                best_fold_val_loss = epoch_avg_val_loss
                fold_bestmodel_avg_mae_score = mae_accum / epoch_total_samples

                best_fold_model_state_dict = copy.deepcopy(model.state_dict())
                fold_bestmodel_avg_mse_score = mse_accum / epoch_total_samples
                fold_bestmodel_r2_score = r2_metric(torch.tensor(all_val_outputs).to(device),
                                                    torch.tensor(all_val_labels).to(device))
                early_stop_rounds_counter = 0


            else:
                early_stop_rounds_counter += 1

            if early_stop_rounds_counter >= early_stop_rounds:
                print(f"Early stopping triggered at epoch {epoch + 1}")
                break
            # if fold_avg_val_loss < best_fold_val_loss:
            #     best_fold_val_loss = fold_avg_val_loss
            #     best_fold_model_state_dict = copy.deepcopy(model.state_dict())
            if num_epochs_processed == 0:
                fold_avg_val_loss = 0
            else:
                fold_avg_val_loss = sum_val_loss / num_epochs_processed
            if fold_bestmodel_avg_mae_score != float('inf'):
                total_mae += fold_bestmodel_avg_mae_score
            if fold_bestmodel_avg_mse_score != float('inf'):
                total_mse += fold_bestmodel_avg_mse_score
            if fold_bestmodel_r2_score != float('-inf'):
                total_r2 += fold_bestmodel_r2_score
            num_folds += 1
        if best_fold_val_loss < best_val_loss_across_folds:
            best_val_loss_across_folds = best_fold_val_loss
            best_model_state_dict_across_folds = best_fold_model_state_dict

                # torch.save(best_fold_model_state_dict, f"model_weights_trial_{trial.number}.pt")
        total_best_fold_val_loss += best_fold_val_loss


        bestmodel_avg_mae = total_mae / num_folds
        bestmodel_avg_mse = total_mse / num_folds
        bestmodel_avg_r2 = total_r2 / num_folds
        # bestmodel_avg_val_loss = total_avg_val_loss / num_folds
    avg_best_fold_val_loss = total_best_fold_val_loss / num_folds  # Compute the average best fold val loss across all folds

    print('bestavg__fold_val_loss', avg_best_fold_val_loss, 'bestmodel_avg_mse', bestmodel_avg_mse,
              'bestmodel_avg_mae', bestmodel_avg_mae, 'bestmodel_avg_r2', bestmodel_avg_r2)
    return bestmodel_avg_mae, bestmodel_avg_mse, best_model_state_dict_across_folds, avg_best_fold_val_loss, bestmodel_avg_r2
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
    num_epochs = trial.suggest_int("num_epochs", 5, 300)  # 3800 #230  291  400-700
    # batch_size = trial.suggest_int("batch_size", 20, 3000)  # 10240  3437
    batch_size = trial.suggest_int('batch_size', 20, 1000)  # 2000
    dropout_rate = trial.suggest_float("dropout_rate", 0, .5)  # 30311980533100547  16372372692286732
    num_hidden_units = trial.suggest_int("num_hidden_units", 100, 4000)  # 2560 #83 125 63  #7000
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1)  # Adding L2 regularization parameter
    l1_lambda = trial.suggest_float("l1_lambda", 1e-5, 1e-1)  # l1 regssss

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", ])  # "AdamW", "RMSprop""SGD"
    num_layers = trial.suggest_int("num_layers", 1, 5)  # example

    if optimizer_name == "SGD":
        momentum = trial.suggest_float('momentum', 0, 0.9)  # Only applicable if using SGD with momentum
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

    lr_scheduler_name = trial.suggest_categorical('lr_scheduler', ['StepLR', 'ExponentialLR', 'ReduceLROnPlateau'])

    if lr_scheduler_name == 'StepLR':
        step_size = trial.suggest_int('step_size', 5, 50)
        gamma = trial.suggest_float('gamma', 0.1, 1)
    elif lr_scheduler_name == 'ExponentialLR':
        gamma = trial.suggest_float('gamma', 0.1, 1)
    elif lr_scheduler_name == 'ReduceLROnPlateau':
        patience = trial.suggest_int('patience', 5, 20)

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
            "lr_scheduler": lr_scheduler_name,  # Pass the scheduler instance here
            "patience": patience,  # Pass the value directly
            "gamma": gamma,  # Pass the value directly
            "step_size": step_size,  # Add step_size parameter for StepLR
            # Add more hyperparameters as needed

        },
        X, y_change, trial=trial
    )
    # print("best mae score: ", bestmodel_avg_mae_score, "best mse score: ", bestmodel_avg_mse_score,
    #       "smallest val loss: ",
    #       overall_avg_val_loss, "best r2 avg:", bestmodel_r2_score)

    return bestmodel_r2_score  # Note this is actually criterion, which is currently mae.



    #    # return prec_score  # Optuna will try to maximize this value

# TODO Comment out to skip the hyperparameter selection.  Swap "best_params".
try:

    study = optuna.load_study(study_name='SPY_FFNNRegressionCV_R2_take2', storage='sqlite:///SPY_FFNNRegressionCV_R2_take2.db')
    best_trial = study.best_trial

    print("Study Loaded.")
    print(f"Best trial was {best_trial.number} with value: {best_trial.value}.")
    print(f"Best parameters were: {study.best_trial.params}")

except KeyError:
    study = optuna.create_study(direction="maximize", study_name='SPY_FFNNRegressionCV_R2_take2',
                            storage='sqlite:///SPY_FFNNRegressionCV_R2_take2.db')
"Keyerror, new optuna study created."  #
study.optimize(objective, n_trials=10000)
best_params = study.best_params

# # ##TODO#
# best_params = {'batch_size': 442, 'dropout_rate': 0.23625354932872586, 'l1_lambda': 0.00829443884457374,
#                'learning_rate': 0.000121861043679494, 'lr_scheduler': 'ReduceLROnPlateau', 'num_epochs': 245,
#                'num_hidden_units': 2269, 'num_layers': 4, 'optimizer': 'Adam', 'patience': 9,
#                'weight_decay': 0.040409278725667123}
# best_params={'batch_size': 701, 'dropout_rate': 0.2526422228728652, 'gamma': 0.19252043462862822, 'l1_lambda': 0.0018668238209024065, 'learning_rate': 0.006068311706135157, 'lr_scheduler': 'ExponentialLR', 'num_epochs': 186, 'num_hidden_units': 991, 'num_layers': 2, 'optimizer': 'Adam', 'weight_decay': 0.05058457883947225}

## Train the model with the best hyperparameters
print("~~~~training model using best params.~~~~")
bestmodel_avg_mae_score, bestmodel_avg_mse_score, best_model_state_dict, overall_avg_val_loss, bestmodel_r2_score = train_model(
    best_params, X, y_change)
model_up_nn = RegressionNN(X.shape[1], best_params["num_hidden_units"],
                           best_params["dropout_rate"], best_params["num_layers"]).to(device)


# Load the saved state_dict into the model
model_up_nn.load_state_dict(best_model_state_dict)
model_up_nn.eval()



# TODO scaled or unscaled y?
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
predicted_values = model_up_nn(X_test_tensor)

y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).to(device)
# predicted_values_tensor = torch.tensor(predicted_values_scaled, dtype=torch.float).to(device)

print("MIN YTEST: ", min(y_test_tensor), " MAX YTEST: ", max(y_test_tensor))
print("MIN pred: ", min(predicted_values), " MAX pred: ", max(predicted_values))
print(predicted_values, y_test_tensor)
#
# # Calculate R2 score
r2 = r2_metric(predicted_values, y_test_tensor)
mae = mae_metric(predicted_values, y_test_tensor)
mse = mse_metric(predicted_values, y_test_tensor)

print(f"Test R2 Score: {r2}", f"Test mae Score: {mae}", f"Test mse Score: {mse}")

input_val = input("Would you like to save these models? y/n: ").upper()
if input_val == "Y":
    model_summary = input("Save this set of models as: ")
    model_directory = os.path.join("../../../Trained_Models", f"{model_summary}")
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    model_filename_up = os.path.join(model_directory, "tar"
                                                      "get_up.pth")
    save_dict = {
        'model_class': model_up_nn.__class__.__name__,
        'features': Chosen_Predictor,
        'input_dim': X.shape[1],
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
        'scaler_X':entire_scaler_X,
        # 'scaler_X_min': scaler_X.min_,
        # 'scaler_X_scale': scaler_X.scale_,
        'scaler_y':entire_scaler_y,
        # 'scaler_y_min': scaler_y.min_,
        # 'scaler_y_scale': scaler_y.scale_,
        'model_state_dict': model_up_nn.state_dict(),
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
        f"Metrics for Target_Up:bestmodel mae score:  {bestmodel_avg_mae_score} bestmodel mse score:  {bestmodel_avg_mse_score} overall val loss best modle: {overall_avg_val_loss} bestmodel r2: {bestmodel_r2_score}")
    info_txt.write(
        f"Predictors: {Chosen_Predictor}\n\n\n"
        f"Best Params: {best_params}\n\n\n")
