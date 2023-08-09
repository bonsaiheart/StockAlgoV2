import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from torchmetrics import Precision, Accuracy, Recall, F1Score
import numpy as np
import os
import optuna

DF_filename = r"../../../data/historical_multiday_minute_DF/Copy of SPY_historical_multiday_min.csv"
#TODO add early stop or no?
# early_stopping = EarlyStopping(patience=5, restore_best_weights=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: ',device)
ml_dataframe = pd.read_csv(DF_filename)
##had highest corr for 3-5 hours with these:
# Chosen_Predictor = ['Bonsai Ratio','Bonsai Ratio 2']
set_best_params_manually = {'learning_rate': 0.000114, 'num_epochs': 100, 'batch_size': 3000, 'dropout_rate': 0.30311980533100547, 'num_hidden_units': 1000}
# Best Hyperparameters: {'learning_rate': 0.0003034075497582067, 'num_epochs': 291, 'batch_size': 3437, 'dropout_rate': 0.16372372692286732, 'num_hidden_units': 63}
Chosen_Predictor = [
    "Bonsai Ratio",
    "Bonsai Ratio 2",
    "B1/B2",
    # "PCRv Up3", "PCRv Up2",
    # "PCRv Down3", "PCRv Down2",
    #
    #  'ITM PCR-Vol',"ITM PCRv Up3",'ITM PCRoi Up1','ITM PCRoi Down1', 'Net_IV', 'Net ITM IV',
    # "ITM PCRv Down3",
    # "ITM PCRv Up4", "ITM PCRv Down2", "ITM PCRv Up2",
    # "ITM PCRv Down4",
    # "RSI14",
    # "AwesomeOsc5_34",
    # "RSI",
    # "RSI2",
    # "AwesomeOsc",
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
cells_forward_to_check = 1*60
threshold_cells_up = cells_forward_to_check * 0.15
threshold_cells_down = cells_forward_to_check * 0.15
percent_up =   .2  #as percent
percent_down = .15 #as percent
anticondition_threshold_cells_up = cells_forward_to_check * 1 #was .7
anticondition_threshold_cells_down = cells_forward_to_check * 1

#TODO these do nothing?.
positivecase_weight_up = 1 #1.2 gave me .57 precisoin #was 20 and 18 its a multiplier

# num_features_up = '8'

#TODO look at validation, i was printing all of the val data sizes just like the test nad train.
threshold_up = 0.5 ###TODO these dont do anything...
threshold_down = 0.8

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

    condition_met_down = (
        ml_dataframe["Current Stock Price"].shift(-i) < (ml_dataframe["Current Stock Price"] - (ml_dataframe["Current Stock Price"]*(percent_down/100)))
    )
    # print( "shifted and current prices",ml_dataframe["Current Stock Price"].shift(-i),(ml_dataframe["Current Stock Price"]))
    anticondition_down = shifted_values >= ml_dataframe["Current Stock Price"]

    targetUpCounter += condition_met_up.astype(int)


    anticondition_UpCounter += anticondition_up.astype(int)
ml_dataframe["Target_Up"] = (
    (targetUpCounter >= threshold_cells_up) & (anticondition_UpCounter <= anticondition_threshold_cells_up)
    )   .astype(int)


ml_dataframe.dropna(subset=["Target_Up"], inplace=True)
y_up = ml_dataframe["Target_Up"]
X = ml_dataframe[Chosen_Predictor]
X.reset_index(drop=True, inplace=True)


X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
y_up_tensor = torch.tensor(y_up.values, dtype=torch.float32).to(device)

train_size = int(len(X_tensor) * 0.8)  # Let's say we'll use 70% of data for training
val_size = train_size + int(len(X_tensor) * 0.1)  # And 15% for validation

X_train_tensor = X_tensor[:train_size]
X_val_tensor = X_tensor[train_size:val_size]
X_test_tensor = X_tensor[val_size:]

y_up_train_tensor = y_up_tensor[:train_size]
y_up_val_tensor = y_up_tensor[train_size:val_size]
y_up_test_tensor = y_up_tensor[val_size:]



num_positive_up_train = sum(y_up_train_tensor)
num_negative_up_train = len(y_up_train_tensor) - num_positive_up_train

num_positive_up_test = sum(y_up_test_tensor)
num_negative_up_test = len(y_up_test_tensor) - num_positive_up_test

weight_negative_up = 1.0
weight_positive_up = (num_negative_up_train / num_positive_up_train) * positivecase_weight_up


# print('num_positive_down:', num_positive_down)
print("test ratio of pos/neg up:", num_positive_up_test/num_negative_up_test)
print('test num_positive_up:', num_positive_up_test)
print('test num_negative_up:', num_negative_up_test)





# custom_weights_up = {0: weight_negative_up, 1: weight_positive_up}
# custom_weights_down = {0: weight_negative_down, 1: weight_positive_down}



print("test num positive up:", num_positive_up_test)

class BinaryClassificationNN(nn.Module):
    def __init__(self, input_dim,num_hidden_units):
        super(BinaryClassificationNN, self).__init__()
        self.layer1 = nn.Linear(input_dim, num_hidden_units)
        self.layer2 = nn.Linear(num_hidden_units, num_hidden_units)
        self.layer3 = nn.Linear(num_hidden_units, num_hidden_units)
        self.output_layer = nn.Linear(num_hidden_units, 1)
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
    model = BinaryClassificationNN(X_train.shape[1],hparams["num_hidden_units"]).to(device)

    #NOTE - swap this criterion with the weight and criterion below it to use the custom weights.
    criterion = nn.BCELoss()
    # weight = torch.Tensor([weight_positive_up]).to(device)
    # criterion = nn.BCEWithLogitsLoss(pos_weight=weight)

   ## # criterion = nn.BCELoss(weight=weight)
   ## # Create tensors for the weights
   ## # weight_positive_up_tensor = torch.tensor([weight_positive_up]).to(device)
   ## # weight_positive_down_tensor = torch.tensor([weight_positive_down]).to(device)

   ## # Create loss functions
   ## # criterion_down = nn.BCEWithLogitsLoss(pos_weight=weight_positive_down_tensor)

    optimizer = optim.Adagrad(model.parameters(), lr=hparams["learning_rate"])
    num_epochs = hparams["num_epochs"]
    batch_size = hparams["batch_size"]
    f1 = torchmetrics.F1Score(num_classes=2, average='macro', task='binary').to(device)
    prec = torchmetrics.Precision(num_classes=2, average='macro', task='binary').to(device)

    best_f1_score = 0.0  # Track the best F1 score
    best_prec_score = 0.0  # Track the best F1 score

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

            # Compute F1 score and Precision score
            # output_predictions = (outputs > 0.5).float()  # thresholding
            # TrainF1Score = f1(y_batch, output_predictions)  # computing F1 score
            # TrainPrecisionScore = prec(y_batch, output_predictions)  # computing Precision score

        model.eval()
    # Validation step
        with torch.no_grad():
            y_val = y_val.reshape(-1, 1)
            val_outputs = model(X_val)
            val_loss = criterion(val_outputs, y_val)

            # Compute F1 score and Precision score
            val_predictions = (val_outputs > threshold_up).float()
            F1Score = f1(val_predictions,y_val)  # computing F1 score
            PrecisionScore = prec(val_predictions,y_val )  # computing Precision score

            # Print the precision score for each epoch

        # print(F1Score,best_f1_score,"that was F1Score,best_f1_score.")
        if F1Score > best_f1_score:

            best_f1_score = F1Score.item()
            torch.save(model.state_dict(), 'best_model.pth')

        if PrecisionScore > best_prec_score:

            best_prec_score = PrecisionScore.item()



        print( f"VALIDATION Epoch: {epoch + 1}, PrecisionScore: {PrecisionScore}, Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}, F1 Score: {F1Score.item()}")

    return best_f1_score,best_prec_score  # Return the best F1 score after all epochs
# Define Optuna Objective
def objective(trial):
    # Define the hyperparameter search space
    learning_rate = trial.suggest_loguniform("learning_rate",  8e-05,0.00035)#0003034075497582067
    num_epochs = trial.suggest_int("num_epochs", 100, 500)#3800 #230  291
    batch_size = trial.suggest_int("batch_size", 1500, 4000)#10240  3437
    # Add more parameters as needed
    dropout_rate = trial.suggest_float("dropout_rate", 0.25,.26)# 30311980533100547  16372372692286732
    num_hidden_units = trial.suggest_int("num_hidden_units", 2000, 2500)#2560 #83 125 63
#Best Hyperparameters: {'learning_rate': 3.6620502192811457e-05, 'num_epochs': 189, 'batch_size': 83, 'dropout_rate': 0.30311980533100547, 'num_hidden_units': 125}
# Best Hyperparameters: {'learning_rate': 0.0003034075497582067, 'num_epochs': 291 , 'batch_size': 3437, 'dropout_rate': 0.16372372692286732, 'num_hidden_units': 63}
#Best Hyperparameters: {'learning_rate': 3.1839682972478313e-05, 'num_epochs': 264, 'batch_size': 2372, 'dropout_rate': 0.2746085482997296, 'num_hidden_units': 487}
#1h4 .4%. Best Hyperparameters: {'learning_rate': 3.2948293353814504e-05, 'num_epochs': 300, 'batch_size': 3070, 'dropout_rate': 0.25704385354888537, 'num_hidden_units': 227}
#1hr .2 Best Hyperparameters: {'learning_rate': 3.340616659933517e-05, 'num_epochs': 257, 'batch_size': 3315, 'dropout_rate': 0.2948078986925536, 'num_hidden_units': 453}
#4hr Best Hyperparameters: {'learning_rate': 0.0003391103649807255, 'num_epochs': 449, 'batch_size': 1238, 'dropout_rate': 0.2521489441443193, 'num_hidden_units': 120}
#Best Hyperparameters: {'learning_rate': 7.782115520036349e-05, 'num_epochs': 100, 'batch_size': 1622, 'dropout_rate': 0.2589106386785677, 'num_hidden_units': 985}

    # Call the train_model function with the current hyperparameters
    f1_score = train_model(
    # prec_score = train_model(

        {
            "learning_rate": learning_rate,
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "dropout_rate": dropout_rate,
            "num_hidden_units": num_hidden_units,
            # Add more hyperparameters as needed
        },
        X_train_tensor, y_up_train_tensor, X_val_tensor, y_up_val_tensor
    )
    return f1_score  # Optuna will try to maximize this value
    #
    # return prec_score  # Optuna will try to maximize this value

##Comment out to skip the hyperparameter selection.  Swap "best_params".
# study = optuna.create_study(direction="maximize")  # We want to maximize the F1-Score
# #TODO changed trials from 100
# study.optimize(objective, n_trials=1)  # You can change the number of trials as needed
# best_params = study.best_params

best_params = set_best_params_manually
# Best Hyperparameters: {'learning_rate': 0.0003034075497582067, 'num_epochs': 291, 'batch_size': 3437, 'dropout_rate': 0.16372372692286732, 'num_hidden_units': 63}

print("Best Hyperparameters:", best_params)


## Train the model with the best hyperparameters
# best_f1_score = train_model(
#     best_params, X_train_tensor, y_up_train_tensor, X_test_tensor, y_up_test_tensor
# )
# Train the model with the best hyperparameters
(best_f1_score,best_prec_score) = train_model(
    best_params, X_train_tensor, y_up_train_tensor, X_val_tensor, y_up_val_tensor)

print("test F1-Score:", best_f1_score)
print("test Precision-Score:", best_prec_score)

model_up_nn = BinaryClassificationNN(X_train_tensor.shape[1],best_params["num_hidden_units"]).to(device)
# Load the saved state_dict into the model
model_up_nn.load_state_dict(torch.load('best_model.pth'))





model_up_nn.eval()

predicted_probabilities_up = model_up_nn(X_test_tensor).detach().cpu().numpy()
threshold_up_formatted = int(threshold_up * 10)

#####################################################################################################################################################
# Get binary predictions by applying threshold

predicted_up = (predicted_probabilities_up > threshold_up_formatted).astype(int)
predicted_up_tensor = torch.tensor(predicted_up, dtype=torch.float32).squeeze().to(device)
# Count the number of positive predictions
num_positives_up = np.sum(predicted_up)

print(f"Number of positive predictions for 'up': {sum(x[0] for x in predicted_up)}")
######################################################################################################################################################
task = "binary"
precision_up = Precision(num_classes=2, average='macro', task=task).to(device)( predicted_up_tensor,y_up_test_tensor)  # move metric to same device as tensors
accuracy_up = Accuracy(num_classes=2, average='macro', task=task).to(device)( predicted_up_tensor,y_up_test_tensor)
recall_up = Recall(num_classes=2, average='macro', task=task).to(device)( predicted_up_tensor,y_up_test_tensor)
f1_up = F1Score(num_classes=2, average='macro', task=task).to(device)( predicted_up_tensor,y_up_test_tensor)


# Print Number of Positive and Negative Samples
num_positive_samples_up = sum(y_up_test_tensor)
num_negative_samples_up = len(y_up_test_tensor) - num_positive_samples_up
print("Number of Positive Samples(Target_Up):", num_positive_samples_up)
print("Number of Negative Samples(Target_Up):", num_negative_samples_up)

print("Metrics for Target_Up:", "\n")
print("Precision:", precision_up)
print("Accuracy:", accuracy_up)
print("Recall:", recall_up)
print("F1-Score:", f1_up, "\n")
print("Best Hyperparameters:", best_params)

#TODO figure out why

# Save the models using joblib
input_val = input("Would you like to save these models? y/n: ").upper()
if input_val == "Y":
    model_summary = input("Save this set of models as: ")
    model_directory = os.path.join("../../Trained_Models", f"{model_summary}")
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    model_filename_up = os.path.join(model_directory, "target_up")
    torch.save(model_up_nn.state_dict(), model_filename_up)

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
        f"Threshold Up (sensitivity): {threshold_up}\n"
        f"Target Underlying Percentage Up: {percent_up}\n\n"
        f"Anticondition Up: {anticondition_UpCounter}\n"
        f"Weight multiplier Up: {positivecase_weight_up}\n")
