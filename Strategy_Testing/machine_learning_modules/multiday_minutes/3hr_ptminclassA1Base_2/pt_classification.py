from datetime import datetime
from joblib import dump

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from torchmetrics import Precision, Accuracy, Recall, F1Score
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
import os
import optuna

DF_filename = r"../../../../data/historical_multiday_minute_DF/older/SPY_historical_multiday_minprior_231002.csv"
#TODO add early stop or no?
# from tensorflow.keras.callbacks import EarlyStopping
ml_dataframe = pd.read_csv(DF_filename)

# Chosen_Predictor = ['ExpDate','LastTradeTime','Current Stock Price','Current SP % Change(LAC)','Bonsai Ratio','Bonsai Ratio 2','B1/B2','B2/B1','PCR-Vol','PCRv @CP Strike','PCRoi @CP Strike','PCRv Up1','PCRv Up4','PCRv Down1','PCRv Down2','PCRv Down4',"PCRoi Up1",'PCRoi Up4','PCRoi Down1','PCRoi Down2','PCRoi Down3','ITM PCR-Vol','ITM PCRv Up1','ITM PCRv Up4','ITM PCRv Down1','ITM PCRv Down2','ITM PCRv Down3','ITM PCRv Down4','ITM PCRoi Up1','ITM PCRoi Up2','ITM PCRoi Up3','ITM PCRoi Up4','ITM PCRoi Down2','ITM PCRoi Down3','ITM PCRoi Down4','ITM OI','Total OI','Net_IV','Net ITM IV','Net IV MP','Net IV LAC','NIV Current Strike','NIV 1Lower Strike','NIV 2Higher Strike','NIV 2Lower Strike','NIV 3Higher Strike','NIV 3Lower Strike','NIV 4Lower Strike','NIV highers(-)lowers1-2','NIV 1-4 % from mean','Net_IV/OI','Net ITM_IV/ITM_OI','Closest Strike to CP','RSI','RSI2','RSI14','AwesomeOsc','AwesomeOsc5_34']

Chosen_Predictor = [
    'Current SP % Change(LAC)','B1/B2', 'B2/B1',  'PCRv @CP Strike','PCRoi @CP Strike','PCRv Up1', 'PCRv Down1','PCRoi Up4','PCRoi Down3' ,'ITM PCR-Vol','ITM PCR-OI', 'Net IV LAC',
    'RSI14', 'AwesomeOsc5_34',
]

study_name='_shuffled_blendedscore_2hrptminclassA1_withweightmultipliler_scaled_X_231013'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('device: ',device)
print(ml_dataframe.columns)
ml_dataframe['LastTradeTime'] = ml_dataframe['LastTradeTime'].apply(
    lambda x: datetime.strptime(str(x), '%y%m%d_%H%M') if not pd.isna(x) else np.nan)
ml_dataframe['LastTradeTime'] = ml_dataframe['LastTradeTime'].apply(lambda x: x.timestamp())
ml_dataframe['LastTradeTime'] = ml_dataframe['LastTradeTime'] / (60 * 60 * 24 * 7)
ml_dataframe['ExpDate'] = ml_dataframe['ExpDate'].astype(float)

cells_forward_to_check =2*60
threshold_cells_up = cells_forward_to_check * 0.4
percent_up =   .25  #as percent
anticondition_threshold_cells_up = cells_forward_to_check * .7 #was .7


theshhold_up = 0.5 ###TODO these dont do any

ml_dataframe.dropna(subset=Chosen_Predictor, inplace=True)
length = ml_dataframe.shape[0]
print("Length of ml_dataframe:", length)
ml_dataframe["Target_Up"] = 0
targetUpCounter = 0
anticondition_UpCounter = 0
for i in range(1, cells_forward_to_check + 1):
    shifted_values = ml_dataframe["Current Stock Price"].shift(-i)
    condition_met_up = shifted_values > (ml_dataframe["Current Stock Price"] + (ml_dataframe["Current Stock Price"]*(percent_up/100)))
    anticondition_up = shifted_values <= ml_dataframe["Current Stock Price"]
    targetUpCounter += condition_met_up.astype(int)
    anticondition_UpCounter += anticondition_up.astype(int)
ml_dataframe["Target_Up"] = (
    (targetUpCounter >= threshold_cells_up) & (anticondition_UpCounter <= anticondition_threshold_cells_up)
    )   .astype(int)
ml_dataframe.dropna(subset=["Target_Up"], inplace=True)
y_up = ml_dataframe["Target_Up"]
X = ml_dataframe[Chosen_Predictor]

# Reset index
X.reset_index(drop=True, inplace=True)


# # # TODO#shuffle trur or false?
X_train, X_temp, y_up_train, y_up_temp = train_test_split(
    X, y_up, test_size=0.3, random_state=None, shuffle=True
)

# Split the temp set into validation and test sets
X_val, X_test, y_up_val, y_up_test = train_test_split(
    X_temp, y_up_temp, test_size=0.5, random_state=None, shuffle=False
)
# # Generate indices for the entire dataset
# total_indices = np.arange(len(X))
#
# # Calculate lengths of training, validation, and test sets
# train_len = int(0.6 * len(X))  # 60% for training
# val_len = int(0.2 * len(X))    # 20% for validation
# # Remaining 20% for test
#
# # Shuffle only the training indices
# train_indices = np.random.permutation(total_indices[:train_len])
#
# # Keep the rest as is
# val_indices = total_indices[train_len:train_len + val_len]
# test_indices = total_indices[train_len + val_len:]
#
# # Extract sets using the indices
# X_train = X.iloc[train_indices]
# y_up_train = y_up.iloc[train_indices]
#
# X_val = X.iloc[val_indices]
# y_up_val = y_up.iloc[val_indices]
#
# X_test = X.iloc[test_indices]
# y_up_test = y_up.iloc[test_indices]

import numpy as np


X_trainval = pd.concat([X_train, X_val], ignore_index=True)
y_trainval = pd.concat([y_up_train, y_up_val], ignore_index=True)

for col in X_train.columns:
    max_val = X_train[col].replace([np.inf, -np.inf], np.nan).max()
    min_val = X_train[col].replace([np.inf, -np.inf], np.nan).min()


    # Adjust max_val based on its sign
    max_val = max_val * 1.5 if max_val >= 0 else max_val / 1.5

    # Adjust min_val based on its sign
    min_val = min_val * 1.5 if min_val < 0 else min_val / 1.5
    print("min/max values ", min_val, max_val)
    # Apply the same max_val and min_val to training, validation, and test sets
    # Apply the same max_val and min_val to training, validation, and test sets
    X_train[col].replace([np.inf, -np.inf], [max_val, min_val], inplace=True)
    X_val[col].replace([np.inf, -np.inf], [max_val, min_val], inplace=True)  # Include this
    X_test[col].replace([np.inf, -np.inf], [max_val, min_val], inplace=True)

for col in X_trainval.columns:
    max_val = X_trainval[col].replace([np.inf, -np.inf], np.nan).max()
    min_val = X_trainval[col].replace([np.inf, -np.inf], np.nan).min()
    # Adjust max_val based on its sign
    max_val = max_val * 1.5 if max_val >= 0 else max_val / 1.5
    # Adjust min_val based on its sign
    min_val = min_val * 1.5 if min_val < 0 else min_val / 1.5
    print("min/max values ", min_val, max_val)
    # Apply the same max_val and min_val to training, validation, and test sets
    X_trainval[col].replace([np.inf, -np.inf], [max_val, min_val], inplace=True)
finalscaler_X = RobustScaler()
X_trainval_scaled = finalscaler_X.fit_transform(X_trainval)
X_trainval_tensor = torch.tensor(X_trainval_scaled, dtype=torch.float32).to(device)
y_trainval_tensor = torch.tensor(y_trainval.values, dtype=torch.float32).to(device)



# Create a scaler object
scaler = RobustScaler()
# Fit the scaler to the training data and then transform both training and test data
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
# X_train_scaled = np.array(X_train.values)
# X_val_scaled = np.array(X_val.values)
# X_test_scaled = np.array(X_test.values)
# Now convert them to torch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)

y_up_train_tensor = torch.tensor(y_up_train.values, dtype=torch.float32).to(device)
y_up_val_tensor = torch.tensor(y_up_val.values, dtype=torch.float32).to(device)
y_up_test_tensor = torch.tensor(y_up_test.values, dtype=torch.float32).to(device)


print("train length:",len(X_train_tensor),"val length:", len(X_val_tensor),"test length:",len(X_test_tensor))


num_positive_up_train = sum(y_up_train_tensor)
num_negative_up_train = len(y_up_train_tensor) - num_positive_up_train
num_positive_up_val = sum(y_up_val_tensor)
num_negative_up_val = len(y_up_val_tensor) - num_positive_up_val
num_positive_up_test = sum(y_up_test_tensor)
num_negative_up_test = len(y_up_test_tensor) - num_positive_up_test
weight_negative_up = 1.0

num_negative_up_trainval = num_negative_up_train+num_negative_up_val
num_positive_up_trainval= num_positive_up_train+num_positive_up_val

print("train ratio of pos/neg up:", num_positive_up_train/num_negative_up_train)
print('train num_positive_up:', num_positive_up_train)
print('train num_negative_up:', num_negative_up_train)
print("val ratio of pos/neg up:", num_positive_up_val/num_negative_up_val)
print('val num_positive_up:', num_positive_up_val)
print('val num_negative_up:', num_negative_up_val)
print("test ratio of pos/neg up:", num_positive_up_test/num_negative_up_test)
print('test num_positive_up:', num_positive_up_test)
print('test num_negative_up:', num_negative_up_test)

class BinaryClassificationNNwithDropout(nn.Module):
    def __init__(self, input_dim, num_hidden_units, dropout_rate):
        super(BinaryClassificationNNwithDropout, self).__init__()
        self.layer1 = nn.Linear(input_dim, num_hidden_units)
        self.layer2 = nn.Linear(num_hidden_units, int(num_hidden_units/2))
        self.layer3 = nn.Linear(int(num_hidden_units/2), int(num_hidden_units/4))
        self.output_layer = nn.Linear(int(num_hidden_units/4), 1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate) # Dropout layer

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.dropout(x) # Apply dropout after the activation
        x = self.activation(self.layer2(x))
        x = self.dropout(x) # Apply dropout after the activation
        x = self.activation(self.layer3(x))
        x = self.dropout(x) # Apply dropout after the activation
        x = self.output_layer(x)  # Removed sigmoid here
        return x


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
    # best_epoch_val_preds = None
    # model = BinaryClassificationNNwithDropout(X_train.shape[1], hparams["num_hidden_units"], hparams['dropout_rate']).to(device)
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

    num_epochs = hparams["num_epochs"]
    batch_size = hparams["batch_size"]
    f1 = torchmetrics.F1Score(num_classes=2, average='weighted', task='binary').to(device)
    prec = Precision(num_classes=2, average='weighted', task='binary').to(device)
    recall = Recall(num_classes=2, average='weighted', task='binary').to(device)

    best_f1_score = 0.0  # Track the best F1 score
    best_prec_score = 0.0  # Track the best F1 score
    sum_f1_score = 0.0
    sum_prec_score = 0.0
    sum_recall_score = 0.0  # Initialize sum of recall scores

    epochs_sum =0
    best_epoch = 0  # Initialize variable to save the best epoch

    best_val_loss = float('inf')  # Initialize best validation loss
    patience = 20  # Early stopping patience; how many epochs to wait
    counter = 0  # Initialize counter for early stopping

    for epoch in range(num_epochs):
        # Training step
        model.train()
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i : i + batch_size]
            y_batch = y_train[i : i + batch_size]

            y_batch = y_batch.unsqueeze(1)  #was wrong shape?
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()


        model.eval()
    # Validation step
        with torch.no_grad():
            y_val = y_val.reshape(-1, 1)
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            # Compute F1 score and Precision score
            # # print(val_outputs.max,val_outputs.min)
            # print("Min:", val_outputs.min().item())
            # print("Max:", val_outputs.max().item())
            val_predictions = (val_outputs > theshhold_up).float()
            F1Score = f1(val_predictions,y_val)  # computing F1 score
            PrecisionScore = prec(val_predictions,y_val )  # computing Precision score
            # PrecisionScore2 = (val_predictions * y_val).sum() / (val_predictions.sum() + 1e-7)
            RecallScore = recall(val_predictions, y_val)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state_dict = model.state_dict()
            counter = 0  # Reset counter when validation loss improves
        else:
            counter += 1  # Increment counter if validation loss doesn't improve

        if F1Score > best_f1_score:
            best_model_state_dict = model.state_dict()
            best_f1_score = F1Score.item()
            best_epoch = epoch  # Save the epoch where the best F1 score was found

        if PrecisionScore > best_prec_score:
            # torch.save(model.state_dict(), 'best_model.pth')
            # best_epoch_val_preds = val_predictions
            best_prec_score = PrecisionScore.item()

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
    avg_val_f1_score = sum_f1_score / num_epochs
    avg_val_precision_score = sum_prec_score / num_epochs
    avg_val_recall_score = sum_recall_score / num_epochs  # Calculate average recall score

    test_outputs = model(X_test_tensor)
    # print(test_outputs)
    test_predictions = (test_outputs > theshhold_up).float().squeeze(1)
    # print(test_predictions)
    testF1Score = f1(test_predictions, y_up_test_tensor)  # computing F1 score
    testPrecisionScore = prec(test_predictions, y_up_test_tensor)
    testRecallScore = recall(test_predictions, y_up_test_tensor)

    print('val avg prec/f1/recall:  ', avg_val_precision_score, avg_val_f1_score, avg_val_recall_score)
    print('test prec/f1/recall: ', testPrecisionScore.item(), testF1Score.item(), testRecallScore.item())

    return best_val_loss,avg_val_f1_score,avg_val_precision_score,best_model_state_dict,testF1Score,testPrecisionScore,best_epoch
    # Return the best F1 score after all epochs
def train_final_model(hparams, Xtrainval, ytrainval):
    positivecase_weight_up = hparams["positivecase_weight_up"]
    weight_positive_up = (num_negative_up_trainval / num_positive_up_trainval) * positivecase_weight_up
    best_model_state_dict = None
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
            X_batch = Xtrainval[i : i + batch_size]
            y_batch = ytrainval[i : i + batch_size]

            y_batch = y_batch.unsqueeze(1)  #was wrong shape?
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()


    best_model_state_dict = model.state_dict()

    return best_model_state_dict
# Define Optuna Objective
def objective(trial):
    # Define the hyperparameter search space
    learning_rate = trial.suggest_float("learning_rate",  1e-05,0.01,log=True)#0003034075497582067
    num_epochs = trial.suggest_int("num_epochs", 50, 3000)#3800 #230  291
    batch_size = trial.suggest_int("batch_size", 20,2500)#10240  3437
    # Add more parameters as needed

    optimizer_name = trial.suggest_categorical("optimizer", [ "Adam", ])#"SGD","RMSprop", "Adagrad"
    dropout_rate = trial.suggest_float("dropout_rate", 0,.4)# 30311980533100547  16372372692286732
    #using layers now instead of setting num_hidden.
    n_layers = trial.suggest_int("n_layers", 1, 4)
    layers = []
    for i in range(n_layers):
        layers.append(trial.suggest_int(f"n_units_l{i}", 4, 128))
    # num_hidden_units = trial.suggest_int("num_hidden_units", 50, 3500)#2560 #83 125 63
    positivecase_weight_up = trial.suggest_float("positivecase_weight_up", 1,10)  # 1.2 gave me .57 precisoin #was 20 and 18 its a multiplier

    # Call the train_model function with the current hyperparameters
    best_val_loss,f1_score, prec_score,best_model_state_dict,testF1Score,testPrecisionScore,best_epoch = train_model(
        {
            "learning_rate": learning_rate,
            "optimizer": optimizer_name,  # Include optimizer name here
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "dropout_rate": dropout_rate,
            # "num_hidden_units": num_hidden_units,
            "positivecase_weight_up": positivecase_weight_up,
            "layers":layers
            # Add more hyperparameters as needed
        },
        X_train_tensor, y_up_train_tensor, X_val_tensor, y_up_val_tensor
    )
    alpha=.5

    blended_score = (alpha * (1 - prec_score)) + ((1 - alpha) * (1-f1_score)) + (alpha * (1 - testPrecisionScore)) + ((1 - alpha) * (1-testF1Score))

    # return best_val_loss
    return blended_score
    # return prec_score  # Optuna will try to maximize this value

##Comment out to skip the hyperparameter selection.  Swap "best_params".
# study = optuna.create_study(direction="maximize")  # We want to maximize the F1-Score
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
study.optimize(objective, n_trials=10000)  # You can change the number of trials as needed
best_params = study.best_params
# best_params = set_best_params_manually
# best_params={'batch_size': 824, 'dropout_rate': 0.025564321641021875, 'learning_rate': 0.009923900109174951, 'num_epochs': 348, 'num_hidden_units': 886, 'optimizer': 'Adam'}
best_params={'batch_size': 881, 'dropout_rate': 0.32727848596144893, 'learning_rate': 0.0006858665963457134, 'n_layers': 2, 'n_units_l0': 110, 'n_units_l1': 29, 'num_epochs': 308, 'optimizer': 'Adam', 'positivecase_weight_up': 1.0015016778402561}
print("Best Hyperparameters:", best_params)

n_layers = best_params['n_layers']
layers = [best_params[f'n_units_l{i}'] for i in range(n_layers)]
best_params['layers'] = layers
## Train the model with the best hyperparameters

(best_val_loss,best_f1_score,best_prec_score,best_model_state_dict,testF1Score,testPrecisionScore,best_epoch) = train_model(
    best_params, X_train_tensor, y_up_train_tensor,X_val_tensor,y_up_val_tensor)
best_params['num_epochs'] = best_epoch


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
predicted_probabilities_up = (predicted_probabilities_up > theshhold_up).astype(int)
# print("predicted_prob up:",predicted_probabilities_up)
predicted_up_tensor = torch.tensor(predicted_probabilities_up, dtype=torch.float32).squeeze().to(device)

num_positives_up = np.sum(predicted_probabilities_up)
task = "binary"
precision_up = Precision(num_classes=2, average='weighted', task='binary').to(device)(predicted_up_tensor,y_up_test_tensor )  # move metric to same device as tensors
accuracy_up = Accuracy(num_classes=2, average='weighted', task=task).to(device)(predicted_up_tensor,y_up_test_tensor )
recall_up = Recall(num_classes=2, average='weighted', task=task).to(device)(predicted_up_tensor,y_up_test_tensor )
f1_up = F1Score(num_classes=2, average='weighted', task=task).to(device)(predicted_up_tensor,y_up_test_tensor )


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
print("Number of Total Samples(Target_Up):", num_positive_samples_up+num_negative_samples_up)

#TODO figure out why
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
    model_directory = os.path.join("../../../Trained_Models", f"{model_summary}")
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    model_filename_up = os.path.join(model_directory, "target_up.pth")

    torch.save({'features':Chosen_Predictor,
        'input_dim': X_train_tensor.shape[1],
                'dropout_rate':best_params["dropout_rate"],
                'layers': best_params["layers"],
                'model_state_dict': finalmodel.state_dict(),
                'scaler_X':finalscaler_X
    }, model_filename_up)
    # Save the scaler
with open(f"../../../Trained_Models/{model_summary}/info.txt", "w") as info_txt:
    info_txt.write("This file contains information about the model.\n\n")
    info_txt.write(
        f"File analyzed: {DF_filename}\nCells_Foward_to_check: {cells_forward_to_check}\n\n"
    )
    info_txt.write(
        f"Metrics for Target_Up:\nPrecision: {precision_up}\nAccuracy: {accuracy_up}\nRecall: {recall_up}\nF1-Score: {f1_up}\n"
    )
    info_txt.write(
        f"Predictors: {Chosen_Predictor}\n\n\n"
        f"Best Params: {best_params}\n\n\n"
        f"Number of Positive Samples (Target_Up): {num_positive_samples_up}\nNumber of Negative Samples (Target_Up): {num_negative_samples_up}\n"
        f"Threshold Up (sensitivity): {theshhold_up}\n"
        f"Target Underlying Percentage Up: {percent_up}\n"
        f"Anticondition: {anticondition_UpCounter}\n")

