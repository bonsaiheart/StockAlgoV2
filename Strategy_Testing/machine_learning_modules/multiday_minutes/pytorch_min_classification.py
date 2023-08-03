import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from torchmetrics import Precision, Accuracy, Recall, F1Score
from sklearn.model_selection import train_test_split
import numpy as np
import joblib
import os
import optuna

DF_filename = r"../../../data/historical_multiday_minute_DF/Copy of SPY_historical_multiday_min.csv"
#TODO add early stop or no?
# from tensorflow.keras.callbacks import EarlyStopping
#
# early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
#
# model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: ',device)
ml_dataframe = pd.read_csv(DF_filename)
print(ml_dataframe.columns)
##had highest corr for 3-5 hours with these:
# Chosen_Predictor = ['Bonsai Ratio','Bonsai Ratio 2','PCRoi Up1','ITM PCRoi Up1', 'RSI14','AwesomeOsc5_34', 'Net IV LAC']

Chosen_Predictor = [
    "Bonsai Ratio",
    "Bonsai Ratio 2",
    "B1/B2", 'ITM PCR-Vol',
    "PCRv Up3", "PCRv Up2",
    "PCRv Down3", "PCRv Down2",
'ITM PCRoi Up1','ITM PCRoi Down1',
    "ITM PCRv Up3", 'Net_IV', 'Net ITM IV',
    "ITM PCRv Down3",
    "ITM PCRv Up4", "ITM PCRv Down2", "ITM PCRv Up2",
    "ITM PCRv Down4",
    "RSI14",
    "AwesomeOsc5_34",
    "RSI",
    "RSI2",
    "AwesomeOsc",
]
# Chosen_Predictor = [ 'Current Stock Price',
#         'Maximum Pain', 'Bonsai Ratio',
#        'Bonsai Ratio 2', 'B1/B2', 'B2/B1', 'PCR-Vol',
#          'PCRv Up1', 'PCRv Up2',
#        'PCRv Up3', 'PCRv Up4', 'PCRv Down1', 'PCRv Down2', 'PCRv Down3',
#        'PCRv Down4', 'PCRoi Up1',
#        'PCRoi Down1',     'ITM PCR-Vol', 'ITM PCR-OI', 'ITM PCRv Up2',
#        'ITM PCRv Up3', 'ITM PCRv Up4',  'ITM PCRv Down2',
#        'ITM PCRv Down3', 'ITM PCRv Down4', 'ITM PCRoi Up2',
#        'ITM PCRoi Up3', 'ITM PCRoi Up4', 'ITM PCRoi Down2',
#        'ITM PCRoi Down3', 'ITM PCRoi Down4', 'ITM OI',
#        'ITM Contracts %', 'Net_IV', 'Net ITM IV',
#        'NIV 2Higher Strike', 'NIV 2Lower Strike', 'NIV 3Higher Strike',
#        'NIV 3Lower Strike', 'NIV 4Higher Strike', 'NIV 4Lower Strike',
#        'NIV highers(-)lowers1-2', 'NIV highers(-)lowers1-4',
#        'NIV 1-2 % from mean', 'NIV 1-4 % from mean', 'Net_IV/OI',
#        'Net ITM_IV/ITM_OI', 'RSI', 'AwesomeOsc',
#        'RSI14', 'RSI2', 'AwesomeOsc5_34']
##changed from %change LAC to factoring in % change of stock price.
cells_forward_to_check = 2*60
threshold_cells_up = cells_forward_to_check * 0.1
threshold_cells_down = cells_forward_to_check * 0.1
percent_up =   .5   /100
percent_down = .5  /100
anticondition_threshold_cells_up = cells_forward_to_check * 1  #was .7
anticondition_threshold_cells_down = cells_forward_to_check * 1
positivecase_weight_up = 20  #was 20 and 18
positivecase_weight_down = 20

# num_features_up = '8'
# num_features_down = '8'
threshold_up = 0.5
threshold_down = 0.5

ml_dataframe.dropna(subset=Chosen_Predictor, inplace=True)
length = ml_dataframe.shape[0]
print("Length of ml_dataframe:", length)

ml_dataframe["Target_Down"] = 0
ml_dataframe["Target_Up"] = 0
targetUpCounter = 0
targetDownCounter = 0
anticondition_UpCounter = 0
anticondition_DownCounter = 0
for i in range(1, cells_forward_to_check + 1):
    shifted_values = ml_dataframe["Current Stock Price"].shift(-i)
    condition_met_up = shifted_values > (ml_dataframe["Current Stock Price"] + (ml_dataframe["Current Stock Price"]*percent_up))
    anticondition_up = shifted_values <= ml_dataframe["Current Stock Price"]

    condition_met_down = (
        ml_dataframe["Current Stock Price"].shift(-i) < (ml_dataframe["Current Stock Price"] - (ml_dataframe["Current Stock Price"]*percent_down))
    )
    anticondition_down = shifted_values >= ml_dataframe["Current Stock Price"]

    targetUpCounter += condition_met_up.astype(int)
    targetDownCounter += condition_met_down.astype(int)

    anticondition_UpCounter += anticondition_up.astype(int)
    anticondition_DownCounter += anticondition_down.astype(int)
    ml_dataframe["Target_Up"] = (
        (targetUpCounter >= threshold_cells_up) & (anticondition_UpCounter <= anticondition_threshold_cells_up)
    ).astype(int)

    ml_dataframe["Target_Down"] = (
        (targetDownCounter >= threshold_cells_down) & (anticondition_DownCounter <= anticondition_threshold_cells_down)
    ).astype(int)

ml_dataframe.dropna(subset=["Target_Up", "Target_Down"], inplace=True)
y_up = ml_dataframe["Target_Up"]
y_down = ml_dataframe["Target_Down"]
X = ml_dataframe[Chosen_Predictor]
X.reset_index(drop=True, inplace=True)


X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
y_up_tensor = torch.tensor(y_up.values, dtype=torch.float32).to(device)
y_down_tensor = torch.tensor(y_down.values, dtype=torch.float32).to(device)

train_size = int(len(X_tensor) * 0.7)  # Let's say we'll use 70% of data for training
val_size = train_size + int(len(X_tensor) * 0.15)  # And 15% for validation

X_train_tensor = X_tensor[:train_size]
X_val_tensor = X_tensor[train_size:val_size]
X_test_tensor = X_tensor[val_size:]

y_up_train_tensor = y_up_tensor[:train_size]
y_up_val_tensor = y_up_tensor[train_size:val_size]
y_up_test_tensor = y_up_tensor[val_size:]

y_down_train_tensor = y_down_tensor[:train_size]
y_down_val_tensor = y_down_tensor[train_size:val_size]
y_down_test_tensor = y_down_tensor[val_size:]

num_positive_up = sum(y_up_train_tensor)
num_negative_up = len(y_up_train_tensor) - num_positive_up
weight_negative_up = 1.0
weight_positive_up = (num_negative_up / num_positive_up) * positivecase_weight_up

num_positive_down = sum(y_down_train_tensor)
print('num_positive_down:', num_positive_down)
print('num_positive_up:', num_positive_up)
num_negative_down = len(y_down_train_tensor) - num_positive_down
weight_negative_down = 1.0
weight_positive_down = (num_negative_down / num_positive_down) * positivecase_weight_down


custom_weights_up = {0: weight_negative_up, 1: weight_positive_up}
custom_weights_down = {0: weight_negative_down, 1: weight_positive_down}

class BinaryClassificationNN(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassificationNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 32)
        self.output_layer = nn.Linear(32, 1)
        self.activation = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.activation(self.layer2(x))
        x = self.activation(self.layer3(x))
        x = self.sigmoid(self.output_layer(x))
        return x


# def evaluate_model(model, X_test, y_test):
#     model.eval()
#     with torch.no_grad():
#         y_pred = model(X_test)
#         loss = criterion(y_pred, y_test.unsqueeze(1))  # Assuming y_test is a tensor, need to add a dimension for BCELoss
#         y_pred_binary = (y_pred > 0.5).float()  # Convert to binary predictions using a threshold of 0.5
#         accuracy = (y_pred_binary == y_test.unsqueeze(1)).float().mean().item()
#         precision = Precision(y_test, y_pred_binary)
#         recall = Recall(y_test, y_pred_binary)
#         f1 = F1Score(y_test, y_pred_binary)
#
#     return loss.item(), accuracy, precision, recall, f1
#
# # Evaluate the model_up_nn
#
#
# print(f"Loss up: {loss_up}, Accuracy up: {accuracy_up}, Precision up: {precision_up}, Recall up: {recall_up}, F1 up: {f1_up}")
# print(f"Loss down: {loss_down}, Accuracy down: {accuracy_down}, Precision down: {precision_down}, Recall down: {recall_down}, F1 down: {f1_down}")

def train_model(hparams, X_train, y_train, X_val, y_val):
    model = BinaryClassificationNN(X_train.shape[1]).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=hparams["learning_rate"])
    num_epochs = hparams["num_epochs"]
    batch_size = hparams["batch_size"]
    f1 = torchmetrics.F1Score(num_classes=2, average='macro', task='binary').to(device)

    best_f1_score = 0.0  # Track the best F1 score

    for epoch in range(num_epochs):
        # Training step
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i : i + batch_size]
            y_batch = y_train[i : i + batch_size]
            y_batch = y_batch.reshape(-1, 1)  #was wrong shape?

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

    # Validation step
        with torch.no_grad():
            y_val = y_val.reshape(-1, 1)
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)

            # Compute F1 score
            val_predictions = (val_outputs > 0.5).float()  # thresholding
            F1Score = f1(y_val, val_predictions)  # computing F1 score
        if F1Score > best_f1_score:
            best_f1_score = F1Score

        print(
            f"Epoch: {epoch + 1}, Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}, F1 Score: {F1Score.item()}")

    return best_f1_score  # Return the best F1 score after all epochs
# Define Optuna Objective
def objective(trial):
    # Define the hyperparameter search space
    learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1e-1)
    num_epochs = trial.suggest_int("num_epochs", 10, 3800)
    batch_size = trial.suggest_int("batch_size", 16, 10240)
    # Add more parameters as needed
    dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.5)
    num_hidden_units = trial.suggest_int("num_hidden_units", 16, 2560)

    # Call the train_model function with the current hyperparameters
    f1_score = train_model(
        {
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "dropout_rate": dropout_rate,
            "num_hidden_units": num_hidden_units,
            # Add more hyperparameters as needed
        },
        X_train_tensor, y_up_train_tensor, X_test_tensor, y_up_test_tensor
    )

    return f1_score  # Optuna will try to maximize this value

study = optuna.create_study(direction="maximize")  # We want to maximize the F1-Score
study.optimize(objective, n_trials=100)  # You can change the number of trials as needed
# Access the best hyperparameters found by Optuna
best_params = study.best_params
print("Best Hyperparameters:", best_params)


# Train the model with the best hyperparameters
best_f1_score = train_model(
    best_params, X_train_tensor, y_up_train_tensor, X_test_tensor, y_up_test_tensor
)
print("Best F1-Score:", best_f1_score)
model_up_nn = BinaryClassificationNN(X_train_tensor.shape[1]).to(device)
model_down_nn = BinaryClassificationNN(X_train_tensor.shape[1]).to(device)

criterion = nn.BCELoss()
optimizer_up = optim.Adam(model_up_nn.parameters())
optimizer_down = optim.Adam(model_down_nn.parameters())


model_up_nn.eval()
model_down_nn.eval()

predicted_probabilities_up = model_up_nn(X_test_tensor).detach().cpu().numpy()
predicted_probabilities_down = model_down_nn(X_test_tensor).detach().cpu().numpy()

threshold_up_formatted = int(threshold_up * 10)
threshold_down_formatted = int(threshold_down * 10)

predicted_up = model_up_nn(X_test_tensor).detach().cpu().numpy()
predicted_up_tensor = torch.tensor(predicted_up, dtype=torch.float32).squeeze().to(device)
predicted_down = model_down_nn(X_test_tensor).detach().cpu().numpy()
predicted_down_tensor = torch.tensor(predicted_down, dtype=torch.float32).squeeze().to(device)


task = "binary"
precision_up = Precision(num_classes=2, average='micro', task=task).to(device)(predicted_up_tensor, y_up_test_tensor)  # move metric to same device as tensors
accuracy_up = Accuracy(num_classes=2, average='micro', task=task).to(device)(predicted_up_tensor, y_up_test_tensor)
recall_up = Recall(num_classes=2, average='micro', task=task).to(device)(predicted_up_tensor, y_up_test_tensor)
f1_up = F1Score(num_classes=2, average='micro', task=task).to(device)(predicted_up_tensor, y_up_test_tensor)

precision_down = Precision(num_classes=2, average='micro', task=task).to(device)(predicted_down_tensor, y_down_test_tensor)
accuracy_down = Accuracy(num_classes=2, average='micro', task=task).to(device)(predicted_down_tensor, y_down_test_tensor)
recall_down = Recall(num_classes=2, average='micro', task=task).to(device)(predicted_down_tensor, y_down_test_tensor)
f1_down = F1Score(num_classes=2, average='micro', task=task).to(device)(predicted_down_tensor, y_down_test_tensor)

# Print Number of Positive and Negative Samples
num_positive_samples_up = sum(y_up_train_tensor)
num_negative_samples_up = len(y_up_train_tensor) - num_positive_samples_up
print("Number of Positive Samples (Target_Up):", num_positive_samples_up)
print("Number of Negative Samples (Target_Up):", num_negative_samples_up)

num_positive_samples_down = sum(y_down_train_tensor)
num_negative_samples_down = len(y_down_train_tensor) - num_positive_samples_down
print("Number of Positive Samples (Target_Down):", num_positive_samples_down)
print("Number of Negative Samples (Target_Down):", num_negative_samples_down)

print("Metrics for Target_Up:", "\n")
print("Precision:", precision_up)
print("Accuracy:", accuracy_up)
print("Recall:", recall_up)
print("F1-Score:", f1_up, "\n")

print("Metrics for Target_Down:", "\n")
print("Precision:", precision_down)
print("Accuracy:", accuracy_down)
print("Recall:", recall_down)
print("F1-Score:", f1_down, "\n")
# loss_up, accuracy_up, precision_up, recall_up, f1_up = evaluate_model(model_up_nn, X_test_tensor, y_up_test_tensor)
# loss_down, accuracy_down, precision_down, recall_down, f1_down = evaluate_model(model_down_nn, X_test_tensor, y_down_test_tensor)
#
# print(f"Loss up: {loss_up}, Loss down: {loss_down}, metric up: {metric_up}, metric down: {metric_down}")

# Save the models using joblib
input_val = input("Would you like to save these models? y/n: ").upper()
if input_val == "Y":
    model_summary = input("Save this set of models as: ")
    model_directory = os.path.join("../../Trained_Models", f"{model_summary}")
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    model_filename_up = os.path.join(model_directory, "target_up")
    model_filename_down = os.path.join(model_directory, "target_down")
    model_up_nn.save(model_filename_up)
    model_down_nn.save(model_filename_down)

with open(f"../../Trained_Models/{model_summary}/info.txt", "w") as info_txt:
    info_txt.write("This file contains information about the model.\n\n")
    info_txt.write(
        f"File analyzed: {DF_filename}\nCells_Foward_to_check: {cells_forward_to_check}\n\n"
    )
    info_txt.write(
        f"Metrics for Target_Up:\nPrecision: {precision_up}\nAccuracy: {accuracy_up}\nRecall: {recall_up}\nF1-Score: {f1_up}\n"
    )
    info_txt.write(
        f"Metrics for Target_Down:\nPrecision: {precision_down}\nAccuracy: {accuracy_down}\nRecall: {recall_down}\nF1-Score: {f1_down}\n"
    )
    info_txt.write(
        f"Predictors: {Chosen_Predictor}\n\n\n"
        f"Number of Positive Samples (Target_Up): {num_positive_samples_up}\nNumber of Negative Samples (Target_Up): {num_negative_samples_up}\n"
        f"Number of Positive Samples (Target_Down): {num_positive_samples_down}\nNumber of Negative Samples (Target_Down): {num_negative_samples_down}\n"
        f"Threshold Up (sensitivity): {threshold_up}\nThreshold Down (sensitivity): {threshold_down}\n"
        f"Target Underlying Percentage Up: {percent_up}\nTarget Underlying Percentage Down: {percent_down}\n"
        f"Anticondition Up: {anticondition_UpCounter}\nAnticondition Down: {anticondition_DownCounter}\n"
        f"Weight multiplier Up: {positivecase_weight_up}\nWeight multiplier Down: {positivecase_weight_down}\n")

