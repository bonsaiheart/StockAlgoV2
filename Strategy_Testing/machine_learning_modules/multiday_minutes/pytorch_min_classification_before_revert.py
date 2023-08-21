import datetime

import pandas as pd
import torch
import torch.nn as nn
import torchmetrics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torchmetrics import Precision, Accuracy, Recall, F1Score
import numpy as np
import os
import optuna

DF_filename = r"../../../data/historical_multiday_minute_DF/SPY_historical_multiday_min.csv"
#TODO add early stop or no?
# from tensorflow.keras.callbacks import EarlyStopping

Chosen_Predictor =  [
     'Bonsai Ratio',
       'Bonsai Ratio 2',
    'B1/B2', 'B2/B1', 'PCR-Vol', 'PCR-OI',]
  #      'PCRv @CP Strike', 'PCRoi @CP Strike', 'PCRv Up1', 'PCRv Up2',
  #      'PCRv Up3', 'PCRv Up4', 'PCRv Down1', 'PCRv Down2', 'PCRv Down3',
  #      'PCRv Down4', 'PCRoi Up1', 'PCRoi Up2', 'PCRoi Up3', 'PCRoi Up4',
  #      'PCRoi Down1', 'PCRoi Down2', 'PCRoi Down3', 'PCRoi Down4',
  #      'ITM PCR-Vol', 'ITM PCR-OI', 'ITM PCRv Up1', 'ITM PCRv Up2',
  #      'ITM PCRv Up3', 'ITM PCRv Up4', 'ITM PCRv Down1', 'ITM PCRv Down2',
  #      'ITM PCRv Down3', 'ITM PCRv Down4', 'ITM PCRoi Up1', 'ITM PCRoi Up2',
  #      'ITM PCRoi Up3', 'ITM PCRoi Up4', 'ITM PCRoi Down1', 'ITM PCRoi Down2',
  #      'ITM PCRoi Down3', 'ITM PCRoi Down4',
  #     'Net_IV', 'Net ITM IV',
  #      'NIV Current Strike', 'NIV 1Higher Strike', 'NIV 1Lower Strike',
  #      'NIV 2Higher Strike', 'NIV 2Lower Strike', 'NIV 3Higher Strike',
  #      'NIV 3Lower Strike', 'NIV 4Higher Strike', 'NIV 4Lower Strike',
  #      'NIV highers(-)lowers1-2', 'NIV highers(-)lowers1-4',
  #      'NIV 1-2 % from mean', 'NIV 1-4 % from mean',
  # 'RSI', 'AwesomeOsc',
  #      'RSI14', 'RSI2', 'AwesomeOsc5_34']
# early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
# model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: ',device)
ml_dataframe = pd.read_csv(DF_filename)
print(ml_dataframe.columns)
# ##had highest corr for 3-5 hours with these:
# Chosen_Predictor = ['Bonsai Ratio','Bonsai Ratio 2','ITM PCR-Vol','ITM PCRoi Up1', 'RSI14','AwesomeOsc5_34', 'Net_IV']

# set_best_params_manually={'learning_rate': 0.002973181466202932, 'num_epochs': 365, 'batch_size': 2500, 'optimizer': 'Adam', 'dropout_rate': 0.05, 'num_hidden_units': 2350}
#[I 2023-08-10 08:48:53,618] Trial 60 finished with value: 0.45271560549736023 and parameters: {'learning_rate': 0.003634678222007879, 'num_epochs': 267, 'batch_size': 1148, 'optimizer': 'Adam', 'dropout_rate': 0.39750214588898447, 'num_hidden_units': 1528}. Best is trial 60 with value: 0.45271560549736023.
# set_best_params_manually={'learning_rate': 1.6814691444050638e-05, 'num_epochs': 345, 'batch_size': 1337, 'optimizer': 'Adadelta','momentum':.5, 'dropout_rate': 0.01120678984579504, 'num_hidden_units': 5000}
#TODO# do the above setiings. it was best 50 from 0-70 trials  [I 2023-08-10 08:09:18,938] Trial 72 finished with value: 0.674614429473877 and parameters: {'learning_rate': 0.0005948477674326639, 'num_epochs': 208, 'batch_size': 679, 'optimizer': 'Adam', 'dropout_rate': 0.16972190725289144, 'num_hidden_units': 749}. Best is trial 44 with value: 0.3507848083972931.
set_best_params_manually={'learning_rate': 0.001621715398308046, 'num_epochs': 617, 'batch_size': 2250, 'optimizer': 'Adadelta', 'dropout_rate': 0.13908048750415472, 'num_hidden_units': 2037}

trainsizepercent = .79
valsizepercent = .2  #test is leftovers
cells_forward_to_check =1*60  #rows to check(minutes in this case)
threshold_cells_up = cells_forward_to_check * 0.5 #how many rows must achieve target %
percent_up =   .25  #target percetage.
anticondition_threshold_cells_up = cells_forward_to_check * .2#was .7
positivecase_weight_up = 1
threshold_up = 0.5 ###At positive prediction = >X
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
X.reset_index(drop=True, inplace=True)
large_number = 1e8  # or another value that makes sense in your context

X = X.replace([np.inf], large_number)

# Convert to torch tensors
X_tensor = torch.tensor(X.values, dtype=torch.float32).to(device)
y_up_tensor = torch.tensor(y_up.values, dtype=torch.float32).to(device)
# Split into training, validation, and test sets
train_size = int(len(X_tensor) * trainsizepercent)
val_size = train_size + int(len(X_tensor) * valsizepercent)
X_train_tensor = X_tensor[:train_size].cpu().numpy()
X_val_tensor = X_tensor[train_size:val_size].cpu().numpy()
X_test_tensor = X_tensor[val_size:].cpu().numpy()
# Create a scaler object
nan_indices = np.argwhere(np.isnan(X))
inf_indices = np.argwhere(np.isinf(X))


if len(nan_indices) > 0:
    print("NaN values found at indices:", nan_indices)
else:
    print("No NaN values found.")

if len(inf_indices) > 0:
    print("Infinite values found at indices:", inf_indices)
else:
    print("No infinite values found.")
#


scaler = MinMaxScaler()
# Fit the scaler to the training data and then transform both training and test data
X_train_scaled = scaler.fit_transform(X_train_tensor)
X_val_scaled = scaler.transform(X_val_tensor)
X_test_scaled = scaler.transform(X_test_tensor)

# Converting the scaled data back to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
print("train length:",len(X_train_tensor),"val length:", len(X_val_tensor),"test length:",len(X_test_tensor))
# Split target data
y_up_train_tensor = y_up_tensor[:train_size]
y_up_val_tensor = y_up_tensor[train_size:val_size]
y_up_test_tensor = y_up_tensor[val_size:]
y_up_numpy = y_up_tensor.cpu().numpy()
y_up_train_numpy = y_up_numpy[:train_size]
# from sklearn.feature_selection import SelectKBest, chi2
# k = 5
# # Create and fit selector
# selector = SelectKBest(chi2, k=k)
# selector.fit(X_train_scaled, y_change_train_numpy)
#
# # Get columns to keep
# cols = selector.get_support(indices=True)
#
# # Use the selected columns to create new data arrays with only the selected features
# X_train_selected = X_train_scaled[:, cols]
# X_val_selected = X_val_scaled[:, cols]
# X_test_selected = X_test_scaled[:, cols]
#
# # Converting the selected data back to PyTorch tensors
# X_train_tensor = torch.tensor(X_train_selected, dtype=torch.float32).to(device)
# X_val_tensor = torch.tensor(X_val_selected, dtype=torch.float32).to(device)
# X_test_tensor = torch.tensor(X_test_selected, dtype=torch.float32).to(device)
# # Number of top features to select
#
# # Get the column names
# selected_features = [Chosen_Predictor[i] for i in cols]
# print('selected features: ',selected_features)
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
class BinaryClassificationLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout_rate):
        super(BinaryClassificationLSTM, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout_rate)
        self.output_layer = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        h_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c_0 = torch.zeros(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.sigmoid(self.output_layer(out[:, -1, :]))
        return out

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


class ClassNN_BatchNorm_DropoutA1(nn.Module):
    def __init__(self, input_dim, num_hidden_units, dropout_rate):
        super(ClassNN_BatchNorm_DropoutA1, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, num_hidden_units),
            nn.BatchNorm1d(num_hidden_units),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(num_hidden_units, int(num_hidden_units / 2)),
            nn.BatchNorm1d(int(num_hidden_units / 2)),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(int(num_hidden_units / 2), int(num_hidden_units / 4)),
            nn.BatchNorm1d(int(num_hidden_units / 4)),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Linear(int(num_hidden_units / 4), 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)


#at alpha =0, it will focus on precision. at alpha=1 it will soley focus f1
def custom_loss(outputs, targets, alpha=0.7):
    epsilon = 1e-7
    # Apply Sigmoid to transform outputs into probabilities
    # Smooth approximation of True Positives, False Positives, False Negatives
    TP = (outputs * targets).sum()
    FP = ((1 - targets) * outputs).sum()
    FN = (targets * (1 - outputs)).sum()
    precision = TP / (TP + FP + epsilon)
    recall = TP / (TP + FN + epsilon)
    f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
    # Combine F1 Score and Precision with weight alpha
    loss = alpha * (1 - f1_score) + (1 - alpha) * (1 - precision)
    return loss
def train_model(hparams, X_train, y_train, X_val, y_val):


    smallest_custom_loss = float('inf')
    best_model_state_dict = None
    # best_epoch_val_preds = None
    model = ClassNN_BatchNorm_DropoutA1(X_train.shape[1], hparams["num_hidden_units"], hparams['dropout_rate']).to(device)
    model.train()
    weight = torch.Tensor([weight_positive_up]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    # criterion = nn.BCELoss(weight=weight)
    # criterion = custom_loss()
    optimizer_name = hparams["optimizer"]
    learning_rate = hparams["learning_rate"]
    momentum = hparams.get("momentum", 0)  # Use the momentum value if provided

    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    elif optimizer_name == "Adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    elif optimizer_name == "Adamax":
        optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)
    elif optimizer_name == "Adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(), lr=learning_rate)
    elif optimizer_name == "LBFGS":
        optimizer = torch.optim.LBFGS(model.parameters(), lr=learning_rate)
    # Add any other optimizers here
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")
    num_epochs = hparams["num_epochs"]
    batch_size = hparams["batch_size"]
    f1 = torchmetrics.F1Score(num_classes=2, average='weighted', task='binary').to(device)
    prec = Precision(num_classes=2, average='weighted', task='binary').to(device)
    best_f1_score = 0.0  # Track the best F1 score
    best_prec_score = 0.0  # Track the best F1 score
    for epoch in range(num_epochs):

        model.train()
        # Training step
        for i in range(0, len(X_train), batch_size):
            X_batch = X_train[i : i + batch_size]
            y_batch = y_train[i : i + batch_size]
            y_batch = y_batch.unsqueeze(1)  #was wrong shape?
            if optimizer_name == "LBFGS":
                def closure():
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    # loss = criterion(outputs, y_batch)
                    loss = custom_loss(outputs, y_batch)
                    loss.backward()
                    return loss

                optimizer.step(closure)
            else:
                optimizer.zero_grad()
                outputs = model(X_batch)
                # loss = criterion(outputs, y_batch)
                loss = custom_loss(outputs, y_batch)
                loss.backward()
                optimizer.step()
        model.eval()
        # Validation step
        val_loss_accum = 0
        F1Score_accum = 0
        PrecisionScore_accum = 0
        num_batches_processed = 0  # Counter for the actual number of batches processed

        for i in range(0, len(X_val), batch_size):
            X_val_batch = X_val[i: i + batch_size]
            y_val_batch = y_val[i: i + batch_size]
            y_val_batch = y_val_batch.unsqueeze(1)
            with torch.no_grad():
                val_outputs = model(X_val_batch)
                # val_loss = criterion(val_outputs, y_val_batch)
                val_loss = custom_loss(val_outputs, y_val_batch)
                val_predictions = (val_outputs > threshold_up).float()
                F1Score_batch = f1(val_predictions, y_val_batch)  # computing F1 score for the batch
                PrecisionScore_batch = prec(val_predictions, y_val_batch)  # computing Precision score for the batch

                # Accumulate the loss and metrics over the validation set
                val_loss_accum += val_loss.item()
                F1Score_accum += F1Score_batch.item()
                PrecisionScore_accum += PrecisionScore_batch.item()
                num_batches_processed += 1  # Increment the counter

        # Compute the average loss and metrics over the validation set using the actual number of batches processed
        val_loss_avg = val_loss_accum / num_batches_processed
        F1Score_avg = F1Score_accum / num_batches_processed
        PrecisionScore_avg = PrecisionScore_accum / num_batches_processed
        # You can then use these averages for further processing

        # Validation step
        # with torch.no_grad():
        #     y_val = y_val.reshape(-1, 1)
        #     val_outputs = model(X_val)
        #   # @Customlossfunction
        #     # val_loss = custom_loss(val_outputs, y_val, alpha=0.5)
        #     val_loss = criterion(val_outputs, y_val)
        #     # Compute F1 score and Precision score
        #     # print(val_outputs.max,val_outputs.min)
        #     # print("Min:", val_outputs.min().item() ,"Max:", val_outputs.max().item())
        #     val_predictions = (val_outputs > theshhold_up).float()
        #     F1Score = f1(val_predictions,y_val)  # computing F1 score
        #     PrecisionScore = prec(val_predictions,y_val )  # computing Precision score
        #     # PrecisionScore2 = (val_predictions * y_val).sum() / (val_predictions.sum() + 1e-7)
        #     # Check if this validation loss is smaller than the current smallest custom loss
        if val_loss_avg < smallest_custom_loss:
            smallest_custom_loss = val_loss_avg
            best_f1_score = F1Score_avg
            best_model_state_dict = model.state_dict()
            best_prec_score = PrecisionScore_avg
        # if F1Score_avg > best_f1_score:
        #     best_f1_score = F1Score_avg
        #     best_model_state_dict = model.state_dict()
        #     best_prec_score = PrecisionScore_avg
        # if PrecisionScore_avg > best_prec_score:
        # best_f1_score = F1Score_avg
        # best_model_state_dict = model.state_dict()
        # best_prec_score = PrecisionScore_avg

        # print( f"VALIDATION Epoch: {epoch + 1}, PrecisionScore: {PrecisionScore_avg}, Training Loss: {loss.item()}, Validation Loss: {val_loss_avg}, F1 Score: {F1Score_avg} ")
    # p
    # rint(best_epoch_val_preds.sum(),y_va
    # l.sum())
    return best_f1_score,best_prec_score,best_model_state_dict,smallest_custom_loss # Return the best F1 score after all epochs
# Define Optuna Objective
def objective(trial):
    print(datetime.datetime.now())
    # print(len(X_val_tensor))
    learning_rate = trial.suggest_float("learning_rate",  .00001,0.001,log=True)#0003034075497582067
    num_epochs = trial.suggest_int("num_epochs", 200 , 500)#3800 #230  291  400-700
    batch_size = trial.suggest_int("batch_size", 1400,5000)#10240  3437
    # batch_size = trial.suggest_int('batch_size', 1400, len(X_val_tensor) - 1)
    dropout_rate = trial.suggest_float("dropout_rate", 0, .5)  # 30311980533100547  16372372692286732
    num_hidden_units = trial.suggest_int("num_hidden_units", 100, 7000)  # 2560 #83 125 63
    optimizer_name = trial.suggest_categorical("optimizer",["SGD", "Adam",   "Adadelta"   ,"Adamax", "Adagrad", "LBFGS","RMSprop",])

    if optimizer_name == "SGD":
        momentum = trial.suggest_float('momentum', 0, 0.9)  # Only applicable if using SGD with momentum
    else:

        momentum = 0  # Default value if not using SGD

    #TODO add learning rate scheduler
    """from torch.optim.lr_scheduler import StepLR
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Training loop
    for epoch in range(num_epochs):
        # Training code here
        # ...
    
        # Validation code here (if you have it)
        # ...
    
        # Step the scheduler
        scheduler.step()"""
    # Call the train_model function with the current hyperparameters
    f1_score, prec_score,best_model_state_dict,smallest_custom_loss = train_model(
        {
            "learning_rate": learning_rate,
            "optimizer": optimizer_name,
            "momentum": momentum,# Include optimizer name here
            "num_epochs": num_epochs,
            "batch_size": batch_size,
            "dropout_rate": dropout_rate,
            "num_hidden_units": num_hidden_units,
            # Add more hyperparameters as needed
        },
        X_train_tensor, y_up_train_tensor, X_val_tensor, y_up_val_tensor
    )
    print("prec score: ",prec_score,"f1: ",f1_score
          )

    return smallest_custom_loss
    #    # return prec_score  # Optuna will try to maximize this value

######################################################################################################################
#TODO Comment out to skip the hyperparameter selection.  Swap "best_params".
study = optuna.create_study(direction="minimize")  # We want to maximize the custom loss score.
study.optimize(objective, n_trials=100)  # You can change the number of trials as needed
best_params = study.best_params

#TODO
# best_params = set_best_params_manually
######################################################################################################################

## Train the model with the best hyperparameters
print("~~~~training model using best params.~~~~")
(best_f1_score,best_prec_score,best_model_state_dict,smallest_custom_loss) = train_model(
    best_params, X_train_tensor, y_up_train_tensor, X_val_tensor, y_up_val_tensor)
model_up_nn = ClassNN_BatchNorm_DropoutA1(X_train_tensor.shape[1],best_params["num_hidden_units"],best_params["dropout_rate"]).to(device)
# Load the saved state_dict into the model
model_up_nn.load_state_dict(best_model_state_dict)
model_up_nn.eval()
predicted_probabilities_up = model_up_nn(X_test_tensor).detach().cpu().numpy()
predicted_probabilities_up = (predicted_probabilities_up > threshold_up).astype(int)
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
print("val F1-Score:", best_f1_score)
print("val Precision-Score:", best_prec_score)
print("Metrics for Test Target_Up:", "\n")
print("Precision:", precision_up)
print("Accuracy:", accuracy_up)
print("Recall:", recall_up)
print("F1-Score:", f1_up, "\n")
print("Best Hyperparameters:", best_params)
print(f"Number of positive predictions"
      f" for 'up': {sum(x[0] for x in predicted_probabilities_up)}")
print("Number of Positive Samples(Target_Up):", num_positive_samples_up)
print("Number of Total Samples(Target_Up):", num_positive_samples_up+num_negative_samples_up)
# print('selected features: ',selected_features)

input_val = input("Would you like to save these models? y/n: ").upper()
if input_val == "Y":
    model_summary = input("Save this set of models as: ")
    model_directory = os.path.join("../../Trained_Models", f"{model_summary}")
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    model_filename_up = os.path.join(model_directory, "target_up.pth")

    torch.save({        'model_class': model_up_nn.__class__.__name__, # Save the class name
'features':Chosen_Predictor,
        'input_dim': X_train_tensor.shape[1],
                'dropout_rate':best_params["dropout_rate"],
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
        f"Metrics for Target_Up:\nPrecision: {precision_up}\nAccuracy: {accuracy_up}\nRecall: {recall_up}\nF1-Score: {f1_up}\n"
    )
    info_txt.write(
        f"Predictors: {Chosen_Predictor}\n\n\n"
        f"Best Params: {best_params}\n\n\n"
        f"Number of Positive Samples (Target_Up): {num_positive_samples_up}\nNumber of Negative Samples (Target_Up): {num_negative_samples_up}\n"
        f"Threshold Up (sensitivity): {threshold_up}\n"
        f"Target Underlying Percentage Up: {percent_up}\n"
        f"Anticondition: {anticondition_UpCounter}\n"
        f"Weight multiplier: {positivecase_weight_up}")

