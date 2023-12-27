import os
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torchmetrics
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from torchmetrics import Precision, Accuracy, Recall, F1Score

# with open('yaml_config\s/config.yaml', 'r') as file:
#     config = yaml.safe_load(file)
#
# # Example replacements
# DF_filename = config['data']['df_filename']
# Chosen_Predictor = config['model']['chosen_predictors']
# cells_forward_to_check = config['training']['cells_forward_to_check']
# threshold_cells_up = config['training']['threshold_cells_up']
# percent_down = config['training']['percent_down']
# anticondition_threshold_cells = config['training']['anticondition_threshold_cells']
# theshhold_down = config['training']['threshold_down']
# n_trials = config['optuna']['n_trials']
# print(Chosen_Predictor)

DF_filename = r"C:\Users\del_p\PycharmProjects\StockAlgoV2\data\historical_multiday_minute_DF\SPY_historical_multiday_min.csv"
# TODO add early stop or no?
# from tensorflow.keras.callbacks import EarlyStopping
ml_dataframe = pd.read_csv(DF_filename)
for col in ml_dataframe.columns:
    ml_dataframe[col] = pd.to_numeric(ml_dataframe[col], errors='coerce')

#FEATURE SET 1?
# Chosen_Predictor= ['Bonsai Ratio', 'ITM PCR-Vol','Net_IV','NIV 2Higher Strike', 'NIV 2Lower Strike', 'Net ITM IV']

set_best_params_manually = {'batch_size': 2295, 'dropout_rate': 0.1956805168069912, 'learning_rate': 0.0006924438743970371, 'n_layers': 2, 'n_units_l0': 1681, 'n_units_l1': 271, 'optimizer': 'Adam', 'positivecase_weight_up': 1.0812601618252304}
# Best Params: {'learning_rate': 0.002973181466202932, 'num_epochs': 365, 'batch_size': 2500, 'optimizer': 'Adam', 'dropout_rate': 0.05, 'num_hidden_units': 2350}


#FEATURE SET 2
# by looking at corr table and eliminating features that have different signs (- or +) for correlating 10,15,20,30min later price, and 15,30 min max change.
# That gave me this froim te list above:
# Chosen_Predictor = ["Current Stock Price",
#                     "Current SP % Change(LAC)",
#                     "Bonsai Ratio",
#                     "Bonsai Ratio 2",
#                     "B1/B2",
#                     "PCR-Vol",
#                     "PCRv @CP Strike",
#                     "PCRv Up1",
#                     "PCRv Up2",
#                     "PCRv Up3",
#                     "PCRv Up4",
#                     "PCRv Down2",
#                     "PCRoi Up4",
#                     "PCRoi Down3",
#                     "ITM PCR-Vol",
#                     "ITM PCRv Up1",
#                     "ITM PCRv Up2",
#                     "ITM PCRv Up3",
#                     "ITM PCRv Up4",
#                     "ITM PCRv Down1",
#                     "ITM PCRv Down2",
#
#                     "ITM PCRv Down3",
#                     "ITM PCRv Down4",
#                     "Net_IV",
#                     "Net ITM IV",
#                     "Net IV MP",
#                     "Net IV LAC",
#                     "Net_IV/OI",

#                     "Closest Strike to CP",
#
#                     ]
# TODO scale  based on data ranges/types
# Chosen_Predictor = [
#     'Bonsai Ratio','Bonsai Ratio 2','PCRv Up1', 'PCRv Down1','ITM PCR-Vol', 'Net IV LAC',
# ]
#Feature set 4
# Chosen_Predictor = [  'Bonsai Ratio','Bonsai Ratio 2', 'PCRv Down1', 'PCRv Down2',
#                    'PCRoi Down3', 'ITM PCR-Vol', 'ITM PCRv Up1', 'ITM PCRv Down1',
#                     'ITM PCRv Down2', 'Net_IV', 'Net ITM IV',  'NIV Current Strike',
#                       'RSI', 'RSI2', 'RSI14', 'AwesomeOsc',
#                     'AwesomeOsc5_34']
#FEATRUE SET 5, original feature set form 3hrptminclass
Chosen_Predictor = [
    'Current SP % Change(LAC)','B1/B2', 'B2/B1',  'PCRv @CP Strike','PCRoi @CP Strike','PCRv Up1', 'PCRv Down1','PCRoi Up4','PCRoi Down3' ,'ITM PCR-Vol','ITM PCR-OI', 'Net IV LAC',
    'RSI14', 'AwesomeOsc5_34',


]
study_name = ('_3hr_40pt_down_FeatSet5')
n_trials =1000
cells_forward_to_check = 60*3
percent_down = .4  # as percent


threshold_cells_up = cells_forward_to_check * 0.2
#The anticondition is when the price goes below the 1st price.  The threshold is how many cells can be anticondition, and still have True label.
anticondition_threshold_cells = cells_forward_to_check * .1  # was .7
theshhold_down = 0.5  ###TODO these dont do any

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('device: ', device)

print(ml_dataframe.columns)
# ml_dataframe['LastTradeTime'] = ml_dataframe['LastTradeTime'].apply(
#     lambda x: datetime.strptime(str(x), '%y%m%d_%H%M') if not pd.isna(x) else np.nan)
# ml_dataframe['LastTradeTime'] = ml_dataframe['LastTradeTime'].apply(lambda x: x.timestamp())
# ml_dataframe['LastTradeTime'] = ml_dataframe['LastTradeTime'] / (60 * 60 * 24 * 7)
ml_dataframe['ExpDate'] = ml_dataframe['ExpDate'].astype(float)

ml_dataframe.dropna(subset=Chosen_Predictor, inplace=True)
length = ml_dataframe.shape[0]
print("Length of ml_dataframe:", length)
ml_dataframe["Target_Up"] = 0
ml_dataframe = ml_dataframe.copy()
targetUpCounter = 0
anticondition_UpCounter = 0
for i in range(1, cells_forward_to_check + 1):
    shifted_values = ml_dataframe["Current Stock Price"].shift(-i)
    condition_met_up = shifted_values < (
                ml_dataframe["Current Stock Price"] - (ml_dataframe["Current Stock Price"] * (percent_down / 100)))
    anticondition_up = shifted_values >= ml_dataframe["Current Stock Price"]
    targetUpCounter += condition_met_up.astype(int)
    anticondition_UpCounter += anticondition_up.astype(int)
ml_dataframe["Target_Up"] = (
        (targetUpCounter >= threshold_cells_up) & (anticondition_UpCounter <= anticondition_threshold_cells)
).astype(int)
ml_dataframe.dropna(subset=["Target_Up"], inplace=True)
y_up = ml_dataframe["Target_Up"]
X = ml_dataframe[Chosen_Predictor]

# Reset index
X.reset_index(drop=True, inplace=True)

# Function to replace infinities and adjust extrema by column in a DataFrame
# TODO love this implimetnation, send to a utillity to use by all scripts.
def replace_infinities_and_scale(df, factor=1.5):
    for col in df.columns:
        # Replace infinities with NaN, then calculate max and min
        max_val = df[col].replace([np.inf, -np.inf], np.nan).max()
        min_val = df[col].replace([np.inf, -np.inf], np.nan).min()

        # Scale max and min values by a factor based on their sign
        max_val = max_val * factor if max_val >= 0 else max_val / factor
        min_val = min_val * factor if min_val < 0 else min_val / factor

        # Replace infinities with the scaled max and min values
        df[col].replace([np.inf, -np.inf], [max_val, min_val], inplace=True)
        print(f"Column: {col}, Min/Max values: {min_val}, {max_val}")

# Function to convert scaled data to tensors
def convert_to_tensor(scaler, X_train, X_val,  device):
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)


    return (
        torch.tensor(X_train_scaled, dtype=torch.float32).to(device),
        torch.tensor(X_val_scaled, dtype=torch.float32).to(device),
    )


# # # TODO#shuffle trur or false?
X_temp,X_test, y_up_temp,y_up_test = train_test_split(
    X, y_up, test_size=0.05,random_state=None, shuffle=False
)

# Split the temp set into validation and test sets
X_train,X_val , y_up_train, y_up_val  = train_test_split(
    X_temp, y_up_temp, test_size=0.3, random_state=None, shuffle=False
)
# Concatenate train and validation sets
X_trainval = pd.concat([X_train, X_val], ignore_index=True)
y_trainval = pd.concat([y_up_train, y_up_val], ignore_index=True)

# Replace infinities and adjust extrema in the training, validation, and test sets
replace_infinities_and_scale(X_train)
replace_infinities_and_scale(X_val)
replace_infinities_and_scale(X_test)

# Replace infinities and adjust extrema in the concatenated train and validation set
replace_infinities_and_scale(X_trainval)

# Fit a robust scaler on the concatenated train and validation set and transform it
finalscaler_X = RobustScaler()
X_trainval_scaled = finalscaler_X.fit_transform(X_trainval)
X_test_scaled = finalscaler_X.transform(X_test)
X_trainval_tensor = torch.tensor(X_trainval_scaled, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

y_trainval_tensor = torch.tensor(y_trainval.values, dtype=torch.float32).to(device)
y_up_test_tensor = torch.tensor(y_up_test.values, dtype=torch.float32).to(device)

y_up_train_tensor = torch.tensor(y_up_train.values, dtype=torch.float32).to(device)
y_up_val_tensor = torch.tensor(y_up_val.values, dtype=torch.float32).to(device)

# Create a scaler object and convert datasets to tensors
scaler = RobustScaler()
X_train_tensor, X_val_tensor = convert_to_tensor(scaler, X_train, X_val, device)

# Print lengths of datasets
print(f"Train length: {len(X_train_tensor)}, Validation length: {len(X_val_tensor)}, Test length: {len(X_test_tensor)}")

# Calculate the number of positive and negative samples in each set
num_positive_up_train = y_up_train_tensor.sum().item()
num_negative_up_train = (y_up_train_tensor == 0).sum().item()
num_positive_up_val = y_up_val_tensor.sum().item()
num_negative_up_val = (y_up_val_tensor == 0).sum().item()
num_positive_up_test = y_up_test_tensor.sum().item()
num_negative_up_test = (y_up_test_tensor == 0).sum().item()

# Calculate the number of positive and negative samples in the combined train and validation set
num_negative_up_trainval = num_negative_up_train + num_negative_up_val
num_positive_up_trainval = num_positive_up_train + num_positive_up_val



def print_dataset_statistics(stage, num_positive, num_negative):
    ratio = num_positive / num_negative if num_negative else float('inf')  # Avoid division by zero
    print(f"{stage} ratio of pos/neg up: {ratio:.2f}")
    print(f"{stage} num_positive_up: {num_positive}")
    print(f"{stage} num_negative_up: {num_negative}\n")


print_dataset_statistics("Train", num_positive_up_train, num_negative_up_train)
print_dataset_statistics("Validation", num_positive_up_val, num_negative_up_val)
print_dataset_statistics("Test", num_positive_up_test, num_negative_up_test)

def plot_learning_curves(train_losses, val_losses):
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Learning Curves')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
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


def feature_importance(model, X_val, y_val):
    model.eval()
    with torch.no_grad():
        baseline_output = model(X_val)
        baseline_metric = f1_score(y_val.cpu().numpy(), (baseline_output > 0.5).cpu().numpy())

    importances = {}
    for i, col in enumerate(Chosen_Predictor):  # Assuming Chosen_Predictor contains feature names
        temp_val = X_val.clone()
        temp_val[:, i] = torch.randperm(temp_val[:, i].size(0))

        with torch.no_grad():
            shuff_output = model(temp_val)
            shuff_metric = f1_score(y_val.cpu().numpy(), (shuff_output > 0.5).cpu().numpy())

        drop_in_metric = baseline_metric - shuff_metric
        importances[col] = drop_in_metric

    return importances


def train_model(hparams, X_train, y_train, X_val, y_val):
    positivecase_weight_up = hparams["positivecase_weight_up"]
    weight_positive_up = (num_negative_up_train / num_positive_up_train) * positivecase_weight_up
    best_model_state_dict = None
    model = DynamicNNwithDropout(X_train.shape[1], hparams['layers'], hparams['dropout_rate']).to(device)

    model.train()

    weight = torch.Tensor([weight_positive_up]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    # criterion = nn.BCELoss(weight=weight)
    optimizer_name = hparams["optimizer"]
    learning_rate = hparams["learning_rate"]

    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    elif optimizer_name == "Adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)

    # num_epochs = hparams["num_epochs"]
    batch_size = hparams["batch_size"]
    f1 = torchmetrics.F1Score(num_classes=2, average='binary', task='binary').to(device)
    prec = Precision(num_classes=2, average='binary', task='binary').to(device)
    recall = Recall(num_classes=2, average='binary', task='binary').to(device)

    best_f1_score = 0.0  # Track the best F1 score
    best_prec_score = 0.0  # Track the best F1 score
    sum_f1_score = 0.0
    sum_prec_score = 0.0
    sum_recall_score = 0.0  # Initialize sum of recall scores

    epochs_sum = 0
    best_epoch = 0  # Initialize variable to save the best epoch

    best_val_loss = float('inf')  # Initialize best validation loss
    patience = 20  # Early stopping patience; how many epochs to wait
    counter = 0  # Initialize counter for early stopping
    train_losses, val_losses = [], []

    for epoch in range(1000):

        # Training step
        model.train()
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i: i + batch_size]
            y_batch = y_train[i: i + batch_size]

            # Skip the batch if it has only one sample. works well when the occasional skipping of small batches won't significantly impact the overall training process,
            if X_batch.shape[0] <= 1:
                continue

            y_batch = y_batch.unsqueeze(1)  # was wrong shape?
            optimizer.zero_grad()

            train_output= model(X_batch)
            train_loss = criterion(train_output, y_batch)


            train_loss.backward()
            optimizer.step()
            train_losses.append(train_loss.item())

        model.eval()
        # Validation step
        with torch.no_grad():

            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val.unsqueeze(1))
            val_losses.append(val_loss.item())

            # Compute F1 score and Precision score
            # # print(val_outputs.max,val_outputs.min)
            # print("Min:", val_outputs.min().item())
            # print("Max:", val_outputs.max().item())
            val_predictions = (val_outputs > theshhold_down).float().squeeze(1)
            F1Score = f1(val_predictions, y_val)  # computing F1 score
            PrecisionScore = prec(val_predictions, y_val)  # computing Precision score
            # PrecisionScore2 = (val_predictions * y_val).sum() / (val_predictions.sum() + 1e-7)
            RecallScore = recall(val_predictions, y_val)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state_dict = model.state_dict()
            counter = 0  # Reset counter when validation loss improves
            best_epoch = epoch  # Save the epoch where the best F1 score was found



        else:
            counter += 1  # Increment counter if validation loss doesn't improve

        sum_f1_score += F1Score.item()
        sum_prec_score += PrecisionScore.item()
        sum_recall_score += RecallScore.item()  # Add to sum of recall scores
        epochs_sum += 1
        # Early stopping
        if counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            model.load_state_dict(best_model_state_dict)  # Load the best model
            break
        # model.train()
        # print( f"VALIDATION Epoch: {epoch + 1}, PrecisionScore: {PrecisionScore.item()}, Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}, F1 Score: {F1Score.item()} ")
    # print(best_epoch_val_preds.sum(),y_val.sum())
    # Calculate average scores
    avg_val_f1_score = sum_f1_score / epoch
    avg_val_precision_score = sum_prec_score / epoch
    avg_val_recall_score = sum_recall_score / epoch  # Calculate average recall score

    test_outputs = model(X_test_tensor)
    # print(test_outputs)
    test_predictions = (test_outputs > theshhold_down).float().squeeze(1)
    # print(test_predictions)
    testF1Score = f1(test_predictions, y_up_test_tensor)  # computing F1 score
    testPrecisionScore = prec(test_predictions, y_up_test_tensor)
    testRecallScore = recall(test_predictions, y_up_test_tensor)

    print('val avg prec/f1/recall:  ', avg_val_precision_score, avg_val_f1_score, avg_val_recall_score)
    print('test prec/f1/recall: ', testPrecisionScore.item(), testF1Score.item(), testRecallScore.item())

    return best_val_loss, avg_val_f1_score, avg_val_precision_score, best_model_state_dict, testF1Score, testPrecisionScore, best_epoch,val_loss
    # Return the best F1 score after all epochs


def train_final_model(hparams, Xtrainval, ytrainval):
    positivecase_weight_up = hparams["positivecase_weight_up"]
    weight_positive_up = (num_negative_up_trainval / num_positive_up_trainval) * positivecase_weight_up
    model = DynamicNNwithDropout(X_train.shape[1], hparams['layers'], hparams['dropout_rate']).to(device)

    model.train()
    weight = torch.Tensor([weight_positive_up]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    optimizer_name = hparams["optimizer"]
    learning_rate = hparams["learning_rate"]

    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    elif optimizer_name == "Adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)

    num_epochs = hparams["num_epochs"]
    batch_size = hparams["batch_size"]

    for epoch in range(num_epochs):
        # Training step
        model.train()
        for i in range(0, len(Xtrainval), batch_size):
            X_batch = Xtrainval[i: i + batch_size]
            y_batch = ytrainval[i: i + batch_size]

            y_batch = y_batch.unsqueeze(1)  # was wrong shape?
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    best_model_state_dict = model.state_dict()

    return best_model_state_dict

# Best Params: {'learning_rate': 0.002973181466202932, 'num_epochs': 365, 'batch_size': 2500, 'optimizer': 'Adam', 'dropout_rate': 0.05, 'num_hidden_units': 2350}

# Define Optuna Objective
def objective(trial):
        # Define the hyperparameter search space
    learning_rate = trial.suggest_float("learning_rate", 0.00001, .1, log=True)  # 0003034075497582067
    num_epochs = trial.suggest_int("num_epochs", 100, 1000)  # 3800 #230  291
    batch_size = trial.suggest_int("batch_size", 50, 3500)  # 10240  3437
        # Add more parameters as needed
    #     # TODO the rounds with SGD seemed to be closer val/test. values.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD","RMSprop", "Adagrad"])  # ,"RMSprop", "Adagrad"
    dropout_rate = trial.suggest_float("dropout_rate", 0, .5)  # 30311980533100547  16372372692286732
    # using layers now instead of setting num_hidden.
    n_layers = trial.suggest_int("n_layers", 1, 6)
    layers = []
    for i in range(n_layers):
        layers.append(trial.suggest_int(f"n_units_l{i}", 32, 2500))
    # num_hidden_units = trial.suggest_int("num_hidden_units", 50, 3500)#2560 #83 125 63
    positivecase_weight_up = trial.suggest_float("positivecase_weight_up", 1,
                                                 20)  # 1.2 gave me .57 precisoin #was 20 and 18 its a multiplier

    # Call the train_model function with the current hyperparameters
    best_val_loss, f1_score, prec_score, best_model_state_dict, testF1Score, testPrecisionScore, best_epoch,val_loss = train_model(
        {
            "learning_rate": learning_rate,
            "optimizer": optimizer_name,  # Include optimizer name here
            # "num_epochs": num_epochs,why use epochs? im tuning it using ealry stopping and passing that to final.
            "batch_size": batch_size,
            "dropout_rate": dropout_rate,
            # "num_hidden_units": num_hidden_units,
            "positivecase_weight_up": positivecase_weight_up,
            "layers": layers
            # Add more hyperparameters as needed
        },
        X_train_tensor, y_up_train_tensor, X_val_tensor, y_up_val_tensor
    )
#TODO Include Regularization Hyperparameters: If overfitting is a concern, consider including L1/L2 regularization hyperparameters in your tuning.
    # Plot the learning curves TODO
    # plot_learning_curves(train_losses, val_losses)
    alpha = .5

    blended_score = (alpha * (1 - prec_score)) + ((1 - alpha) * (1 - f1_score)) + (alpha * (1 - testPrecisionScore)) + (
                (1 - alpha) * (1 - testF1Score))

    # return best_val_loss
    return (val_loss+(1-testPrecisionScore))/2
    # return prec_score  # Optuna will try to maximize this value


##Comment out to skip the hyperparameter selection.  Swap "best_params".
try:
    study = optuna.load_study(study_name=f'{study_name}',
                              storage=f'sqlite:///{study_name}.db')
    print("Study Loaded.")
    try:
        best_params_up = study.best_params
        best_trial_up = study.best_trial
        best_value_up = study.best_value
        print("Best Value_up:", best_value_up)
        print(best_params_up)
        print("Best Trial_up:", best_trial_up)
    except Exception as e:
        print(e)
except KeyError:
    study = optuna.create_study(direction="minimize", study_name=f'{study_name}',
                                storage=f'sqlite:///{study_name}.db')
"Keyerror, new optuna study created."  #

# TODO changed trials from 100
study.optimize(objective, n_trials=n_trials)  # You can change the number of trials as needed
best_params = study.best_params
# best_params = set_best_params_manually
# best_params={'batch_size': 824, 'dropout_rate': 0.025564321641021875, 'learning_rate': 0.009923900109174951, 'num_epochs': 348, 'num_hidden_units': 886, 'optimizer': 'Adam'}
# best_params={'batch_size': 881, 'dropout_rate': 0.32727848596144893, 'learning_rate': 0.0006858665963457134, 'n_layers': 2, 'n_units_l0': 110, 'n_units_l1': 29, 'num_epochs': 308, 'optimizer': 'Adam', 'positivecase_weight_up': 1.0015016778402561}
print("Best Hyperparameters:", best_params)

n_layers = best_params['n_layers']
layers = [best_params[f'n_units_l{i}'] for i in range(n_layers)]
best_params['layers'] = layers
## Train the model with the best hyperparameters

(best_val_loss, best_f1_score, best_prec_score, best_model_state_dict, testF1Score, testPrecisionScore,
 best_epoch,val_loss) = train_model(
    best_params, X_train_tensor, y_up_train_tensor, X_val_tensor, y_up_val_tensor)
best_params['num_epochs'] = best_epoch
#TODO must be better way to get
n_layers = best_params['n_layers']
layers = [best_params[f'n_units_l{i}'] for i in range(n_layers)]
best_params['layers'] = layers
(best_model_state_dict) = train_final_model(best_params, X_trainval_tensor, y_trainval_tensor)

finalmodel = DynamicNNwithDropout(X_train.shape[1], best_params['layers'], best_params['dropout_rate']).to(device)
# Load the saved state_dict into the model
finalmodel.load_state_dict(best_model_state_dict)
finalmodel.eval()
feature_imp = feature_importance(finalmodel, X_trainval_tensor, y_trainval_tensor)
print("Feature Importances:", feature_imp)
predicted_probabilities_up = finalmodel(X_test_tensor).detach().cpu().numpy()
# print("predicted_prob up:",predicted_probabilities_up)
predicted_probabilities_up = (predicted_probabilities_up > theshhold_down).astype(int)
# print("predicted_prob up:",predicted_probabilities_up)
predicted_up_tensor = torch.tensor(predicted_probabilities_up, dtype=torch.float32).squeeze().to(device)
num_positives_up = np.sum(predicted_probabilities_up)

task = "binary"
# move metric to same device as tensors
precision_up = Precision(num_classes=2, average='binary', task=task).to(device)(predicted_up_tensor, y_up_test_tensor)
accuracy_up = Accuracy(num_classes=2, average='binary', task=task).to(device)(predicted_up_tensor, y_up_test_tensor)
recall_up = Recall(num_classes=2, average='binary', task=task).to(device)(predicted_up_tensor, y_up_test_tensor)
f1_up = F1Score(num_classes=2, average='binary', task=task).to(device)(predicted_up_tensor, y_up_test_tensor)

# Print Number of Positive and Negative Samples
num_positive_samples_up = sum(y_up_test_tensor)
num_negative_samples_up = len(y_up_test_tensor) - num_positive_samples_up

print("Metrics for Target_Up:", "\n")
print("Precision:", precision_up)
print("Accuracy:", accuracy_up)
print("Recall:", recall_up)
print("F1-Score:", f1_up, "\n")

print("Best Hyperparameters:", best_params)
print(f"Number of positive predictions for 'up': {sum(x[0] for x in predicted_probabilities_up)}")
print("Number of Positive Samples(Target_Up):", num_positive_samples_up)
print("Number of Total Samples(Target_Up):", num_positive_samples_up + num_negative_samples_up)

# TODO figure out why
# Number of positive predictions for 'up': 0
# Number of positive predictions for 'down': 0
# Number of Positive Samples (Target_Up): tensor(111., device='cuda:0')
# Number of Negative Samples (Target_Up): tensor(1231., device='cuda:0')
# Number of Positive Samples (Target_Down): tensor(79., device='cuda:0')
# Number of Negative Samples (Target_Down): tensor(1263., device='cuda:0')
# Metrics for Target_Up:
# Precision: tensor(0.5714, device='cuda:0')
# Accuracy: tensor(0.9195, device='cuda:0')
# Recall: tensor(0.1081, device='cuda:0')
# F1-Score: tensor(0.1818, device='cuda:0') """

# Save the models using joblib
input_val = input("Would you like to save these models? y/n: ").upper()
if input_val == "Y":
    model_summary = input("Save this set of models as: ")
    model_directory = os.path.join("../../../../../Trained_Models", f"{model_summary}")
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    model_filename_up = os.path.join(model_directory, "target_up.pth")

    torch.save({'features': Chosen_Predictor,
                'input_dim': X_train_tensor.shape[1],
                'dropout_rate': best_params["dropout_rate"],
                'layers': best_params["layers"],
                'model_state_dict': finalmodel.state_dict(),
                'scaler_X': finalscaler_X
                }, model_filename_up)
    # Save the scaler

    # Generate the function definition
    function_def = f"""
def {model_summary}(new_data_df):
    checkpoint = torch.load(f'{{base_dir}}/{model_summary}/target_up.pth', map_location=torch.device('cpu'))
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

    tempdf = pd.DataFrame(scaler_X.transform(tempdf), columns=features)
    input_tensor = torch.tensor(tempdf.values, dtype=torch.float32)
    predictions = loaded_model(input_tensor)
    predictions_prob = torch.sigmoid(predictions)
    predictions_numpy = predictions_prob.detach().numpy()
    prediction_series = pd.Series(predictions_numpy.flatten(), index=tempdf.index)

    result = new_data_df.copy()
    result["Predictions"] = np.nan
    result.loc[prediction_series.index, "Predictions"] = prediction_series.values
    return result
    """

    # Append the new function definition to pytorch_trained_minute_models.py
    with open('../../../../../Trained_Models/pytorch_trained_minute_models.py', 'a') as file:
        file.write(function_def)
    with open(f"../../../../../Trained_Models/{model_summary}/info.txt", "w") as info_txt:
        info_txt.write("This file contains information about the model.\n\n")
        info_txt.write(
            f"File analyzed: {DF_filename}\nStudy Name: {study_name}\nCells_Foward_to_check: {cells_forward_to_check}\n\n"
        )

        info_txt.write(
            f"Metrics for Target_Up:\nPrecision: {precision_up}\nAccuracy: {accuracy_up}\nRecall: {recall_up}\nF1-Score: {f1_up}\n"
        )
        info_txt.write(
            f"Predictors: {Chosen_Predictor}\n\n\n"
            f"Best Params: {best_params}\n\n\n"
            f"study_name: {study_name}\n\n\n"
            f"Number of Positive Samples (Target_Up): {num_positive_samples_up}\nNumber of Negative Samples (Target_Up): {num_negative_samples_up}\n"
            f"Threshold(sensitivity): {theshhold_down}\n"
            f"Target Underlying Percentage: {percent_down}\n"
            f"Threshhold positive condition cells: {threshold_cells_up}"
            f"anticondition_cells_threshold: {anticondition_threshold_cells}\n")
