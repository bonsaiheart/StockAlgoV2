import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from sklearn.preprocessing import StandardScaler
from torchmetrics import Precision, Accuracy, Recall, F1Score
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
import os
import optuna

DF_filename = r"../../../../data/historical_multiday_minute_DF/SPY_historical_multiday_min.csv"
#TODO add early stop or no?
# from tensorflow.keras.callbacks import EarlyStopping

Chosen_Predictor = ['Bonsai Ratio','Bonsai Ratio 2','B1/B2','B2/B1','PCRoi Up1','PCRoi Down1','ITM PCR-OI','ITM PCRoi Up1','ITM PCRoi Down1','ITM PCRoi Down2','ITM PCRoi Down3','ITM PCRoi Down4','ITM Contracts %','Net ITM IV','NIV highers(-)lowers1-4','Net_IV/OI','Net ITM_IV/ITM_OI']
# early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
# model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: ',device)
ml_dataframe = pd.read_csv(DF_filename)
print(ml_dataframe.columns)
# ##had highest corr for 3-5 hours with these:
# Chosen_Predictor = ['Bonsai Ratio','Bonsai Ratio 2','ITM PCRv','ITM PCRoi Up1', 'RSI14','AwesomeOsc5_34', 'Net IV']
# set_best_params_manually={'learning_rate': 0.002973181466202932, 'num_epochs': 365, 'batch_size': 2500, 'optimizer': 'Adam', 'dropout_rate': 0.05, 'num_hidden_units': 2350}
# Chosen_Predictor = ['Bonsai Ratio','Bonsai Ratio 2' ]
# set_best_params_manually={'learning_rate': 1.4273231212290852e-04, 'num_epochs': 339, 'batch_size': 3000, 'optimizer': 'SGD', 'dropout_rate': 0.2, 'num_hidden_units': 2000}
# set_best_params_manually={'learning_rate': 0.00007, 'num_epochs':400, 'batch_size': 2500, 'optimizer': 'SGD', 'dropout_rate': 0., 'num_hidden_units': 2000}
trainsizepercent = .8
valsizepercent = .15

cells_forward_to_check =3*60
threshold_cells_up = cells_forward_to_check * 0.1
percent_up =   .4  #as percent
anticondition_threshold_cells_up = cells_forward_to_check * .6 #was .7

#TODO th
positivecase_weight_up = 1 #1.2 gave me .57 precisoin #was 20 and 18 its a multiplier

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
X = ml_dataframe[Chosen_Predictor].copy()

# Reset index
X.reset_index(drop=True, inplace=True)
# Split the data into training, validation, and test sets first
trainsizepercent = .7
valsizepercent = .15
train_size = int(len(X) * trainsizepercent)
val_size = train_size + int(len(X) * valsizepercent)

X_train = X.iloc[:train_size].copy()
X_val = X.iloc[train_size:val_size].copy()
X_test = X.iloc[val_size:].copy()

# Handle inf and -inf values based on training set
for col in X_train.columns:
    max_val = X_train[col].replace([np.inf, -np.inf], np.nan).max()
    min_val = X_train[col].replace([np.inf, -np.inf], np.nan).min()

    # Adjust max_val based on its sign
    max_val = max_val * 1.5 if max_val >= 0 else max_val / 1.5

    # Adjust min_val based on its sign
    min_val = min_val * 1.5 if min_val < 0 else min_val / 1.5

    # Apply the same max_val and min_val to training, validation, and test sets
    X_train[col].replace([np.inf, -np.inf], [max_val, min_val], inplace=True)
    X_val[col].replace([np.inf, -np.inf], [max_val, min_val], inplace=True)
    X_test[col].replace([np.inf, -np.inf], [max_val, min_val], inplace=True)

# Reset the index
X_train.reset_index(drop=True, inplace=True)
X_val.reset_index(drop=True, inplace=True)
X_test.reset_index(drop=True, inplace=True)
scaler = StandardScaler()

# Fit the scaler to the training data and transform all sets
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Convert these scaled DataFrames back to DataFrames if needed
X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_val = pd.DataFrame(X_val_scaled, columns=X_val.columns)
X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Now convert to torch tensors and move to device
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val.values, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)

print("train length:",len(X_train_tensor),"val length:", len(X_val_tensor),"test length:",len(X_test_tensor))

# Split target data
y_up_train = y_up[:train_size]
y_up_val = y_up[train_size:val_size]
y_up_test = y_up[val_size:]

# Convert to torch tensors and move to device
y_up_train_tensor = torch.tensor(y_up_train.values, dtype=torch.float32).to(device)
y_up_val_tensor = torch.tensor(y_up_val.values, dtype=torch.float32).to(device)
y_up_test_tensor = torch.tensor(y_up_test.values, dtype=torch.float32).to(device)

num_positive_up_train = sum(y_up_train_tensor)
num_negative_up_train = len(y_up_train_tensor) - num_positive_up_train
num_positive_up_train = sum(y_up_train_tensor)
num_negative_up_train = len(y_up_train_tensor) - num_positive_up_train
num_positive_up_val = sum(y_up_val_tensor)
num_negative_up_val = len(y_up_val_tensor) - num_positive_up_val
num_positive_up_test = sum(y_up_test_tensor)
num_negative_up_test = len(y_up_test_tensor) - num_positive_up_test
weight_negative_up = 1.0
weight_positive_up = (num_negative_up_train / num_positive_up_train) * positivecase_weight_up

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
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.dropout(x) # Apply dropout after the activation
        x = self.activation(self.layer2(x))
        x = self.dropout(x) # Apply dropout after the activation
        x = self.activation(self.layer3(x))
        x = self.dropout(x) # Apply dropout after the activation
        x = self.sigmoid(self.output_layer(x))
        return x


def train_model(hparams, X_train, y_train, X_val, y_val):
    best_model_state_dict = None
    # best_epoch_val_preds = None
    model = BinaryClassificationNNwithDropout(X_train.shape[1], hparams["num_hidden_units"], hparams['dropout_rate']).to(device)
    model.train()

    weight = torch.Tensor([weight_positive_up]).to(device)
    # criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    criterion = nn.BCELoss(weight=weight)
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

    best_val_f1_score = 0.0  # Track the best F1 score
    best_val_prec_score = 0.0  # Track the best F1 score

    for epoch in range(num_epochs):
        # Training step
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i : i + batch_size]
            y_batch = y_train[i : i + batch_size]

            y_batch = y_batch.unsqueeze(1)  #was wrong shape?
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()


        # model.eval()
    # Validation step
        with torch.no_grad():
            y_val = y_val.reshape(-1, 1)
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)
            # Compute F1 score and Precision score
            # print(val_outputs.max,val_outputs.min)
            # print("Min:", val_outputs.min().item())
            # print("Max:", val_outputs.max().item())
            val_predictions = (val_outputs > theshhold_up).float()
            valF1Score = f1(val_predictions,y_val)  # computing F1 score
            valPrecisionScore = prec(val_predictions,y_val )  # computing Precision score
            # PrecisionScore2 = (val_predictions * y_val).sum() / (val_predictions.sum() + 1e-7)
        if valF1Score > best_val_f1_score:
            best_model_state_dict = model.state_dict()
            best_val_f1_score = valF1Score.item()

        if valPrecisionScore > best_val_prec_score:
            # torch.save(model.state_dict(), 'best_model.pth')
            # best_epoch_val_preds = val_predictions
            best_val_prec_score = valPrecisionScore.item()
        # model.train()
        test_predicted_probabilities_up = model(X_test_tensor).detach().cpu().numpy()
        # print("predicted_prob up:",predicted_probabilities_up)
        test_predicted_probabilities_up = (test_predicted_probabilities_up > theshhold_up).astype(int)
        # print("predicted_prob up:",predicted_probabilities_up)
        test_predicted_up_tensor = torch.tensor(test_predicted_probabilities_up, dtype=torch.float32).squeeze().to(device)

        num_positives_up = np.sum(test_predicted_probabilities_up)
        task = "binary"
        test_precision_up = Precision(num_classes=2, average='weighted', task='binary').to(device)(test_predicted_up_tensor,
                                                                                              y_up_test_tensor)  # move metric to same device as tensors
        test_accuracy_up = Accuracy(num_classes=2, average='macro', task=task).to(device)(test_predicted_up_tensor,
                                                                                     y_up_test_tensor)
        test_recall_up = Recall(num_classes=2, average='macro', task=task).to(device)(test_predicted_up_tensor, y_up_test_tensor)
        test_f1_up = F1Score(num_classes=2, average='macro', task=task).to(device)(test_predicted_up_tensor, y_up_test_tensor)

        # print( f"VALIDATION Epoch: {epoch + 1}, PrecisionScore: {PrecisionScore.item()}, Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}, F1 Score: {F1Score.item()} ")
    # print(best_epoch_val_preds.sum(),y_val.sum())
    return best_val_f1_score,best_val_prec_score,best_model_state_dict,test_precision_up,test_f1_up # Return the best F1 score after all epochs
# Define Optuna Objective
def objective(trial):
    # Define the hyperparameter search space
    learning_rate = trial.suggest_float("learning_rate",  1e-05,0.01,log=True)#0003034075497582067
    num_epochs = trial.suggest_int("num_epochs", 100, 500)#3800 #230  291
    batch_size = trial.suggest_int("batch_size", 20,3000)#10240  3437
    # Add more parameters as needed
    optimizer_name = trial.suggest_categorical("optimizer", ["SGD", "Adam", "RMSprop", "Adagrad"])
    dropout_rate = trial.suggest_float("dropout_rate", 0,.4)# 30311980533100547  16372372692286732
    num_hidden_units = trial.suggest_int("num_hidden_units", 500, 2500)#2560 #83 125 63


    # Call the train_model function with the current hyperparameters
    f1_score, prec_score,best_model_state_dict,test_precision_up,test_f1_up = train_model(
        {
            "learning_rate": learning_rate,
            "optimizer": optimizer_name,  # Include optimizer name here
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "dropout_rate": dropout_rate,
            "num_hidden_units": num_hidden_units,
            # Add more hyperparameters as needed
        },
        X_train_tensor, y_up_train_tensor, X_val_tensor, y_up_val_tensor
    )
    alpha = .5
    print("precision :",prec_score,' f1 :',f1_score,"test prec: ",test_precision_up,"test f1: ",test_f1_up)
    # Blend the scores using alpha
    blended_score = alpha * (1 - prec_score) + (1 - alpha) * f1_score
    blended_test_score = alpha * (1 - test_precision_up) + (1 - alpha) * test_f1_up
    print('Blended test score: ',blended_test_score)
    return (blended_score+blended_test_score)/2
    #
    # return prec_score  # Optuna will try to maximize this value
# Precision:  1.0 F1 :  0.2857142857142857 AUC : 0.9978723404255319
# [I 2023-10-02 16:26:55,124] Trial 47 finished with value: 0.7611955420466058 and parameters: {'max_depth': 21, 'min_samples_split': 5, 'n_estimators': 416, 'min_samples_leaf': 1}. Best is trial 47 with value: 0.7611955420466058.
# Precision:  0.3333333333333333 F1 :  0.2222222222222222 AUC : 0.9848977889027951
#Comment out to skip the hyperparameter selection.  Swap "best_params".
study = optuna.create_study(direction="minimize")  # We want to maximize the F1-Score
# TODO changed trials from 100
study.optimize(objective, n_trials=100)  # You can change the number of trials as needed
best_params = study.best_params
# best_params = set_best_params_manually

print("Best Hyperparameters:", best_params)

## Train the model with the best hyperparameters
(best_f1_score,best_prec_score,best_model_state_dict,test_precision_up,test_f1_up) = train_model(
    best_params, X_train_tensor, y_up_train_tensor, X_val_tensor, y_up_val_tensor)

print("val F1-Score:", best_f1_score)
print("val Precision-Score:", best_prec_score)

model_up_nn = BinaryClassificationNNwithDropout(X_train_tensor.shape[1],best_params["num_hidden_units"],best_params["dropout_rate"]).to(device)
# Load the saved state_dict into the model
model_up_nn.load_state_dict(best_model_state_dict)
model_up_nn.eval()
predicted_probabilities_up = model_up_nn(X_test_tensor).detach().cpu().numpy()
# print("predicted_prob up:",predicted_probabilities_up)
predicted_probabilities_up = (predicted_probabilities_up > theshhold_up).astype(int)
# print("predicted_prob up:",predicted_probabilities_up)
predicted_up_tensor = torch.tensor(predicted_probabilities_up, dtype=torch.float32).squeeze().to(device)

num_positives_up = np.sum(predicted_probabilities_up)
task = "binary"
precision_up = Precision(num_classes=2, average='weighted', task='binary').to(device)(predicted_up_tensor,y_up_test_tensor )  # move metric to same device as tensors
accuracy_up = Accuracy(num_classes=2, average='macro', task=task).to(device)(predicted_up_tensor,y_up_test_tensor )
recall_up = Recall(num_classes=2, average='macro', task=task).to(device)(predicted_up_tensor,y_up_test_tensor )
f1_up = F1Score(num_classes=2, average='macro', task=task).to(device)(predicted_up_tensor,y_up_test_tensor )


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
    model_directory = os.path.join("../../Trained_Models", f"{model_summary}")
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    model_filename_up = os.path.join(model_directory, "target_up.pth")

    torch.save({'features':Chosen_Predictor,
        'input_dim': X_train_tensor.shape[1],
        'num_hidden_units': best_params["num_hidden_units"],
        'model_state_dict': model_up_nn.state_dict(),
    }, model_filename_up)

with open(f"../../Trained_Models/{model_summary}/info.txt", "w") as info_txt:
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
        f"Anticondition: {anticondition_UpCounter}\n"
        f"Weight multiplier: {positivecase_weight_up}")

