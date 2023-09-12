import copy
import datetime
import os
from datetime import datetime
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics import Accuracy, Precision, Recall, F1Score

DF_filename = r"../../../../data/historical_multiday_minute_DF/older/SPY_historical_multiday_min.csv"
# TODO add early stop or no?
# from tensorflow.keras.callbacks import EarlyStopping

# Chosen_Predictor = [
#     'Bonsai Ratio',
#     'Bonsai Ratio 2',
#     'B1/B2', 'B2/B1', 'PCR-Vol', 'PCR-OI',
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: ', device)
ml_dataframe = pd.read_csv(DF_filename)
print(ml_dataframe.columns)
# ##had highest corr for 3-5 hours with these:
# Chosen_Predictor = ['Bonsai Ratio','Bonsai Ratio 2','ITM PCR-Vol','ITM PCRoi Up1', 'RSI14','AwesomeOsc5_34', 'Net_IV']
ml_dataframe['LastTradeTime'] = ml_dataframe['LastTradeTime'].apply(
    lambda x: datetime.strptime(str(x), '%y%m%d_%H%M') if not pd.isna(x) else np.nan)
ml_dataframe['LastTradeTime'] = ml_dataframe['LastTradeTime'].apply(lambda x: x.timestamp())
ml_dataframe['ExpDate'] = ml_dataframe['ExpDate'].astype(float)
# TODO# do the above setiings. it was best 50 from 0-70 trials  [I 2023-08-10 08:09:18,938] Trial 72 finished with value: 0.674614429473877 and parameters: {'learning_rate': 0.0005948477674326639, 'num_epochs': 208, 'batch_size': 679, 'optimizer': 'Adam', 'dropout_rate': 0.16972190725289144, 'num_hidden_units': 749}. Best is trial 44 with value: 0.3507848083972931.
set_best_params_manually = {'batch_size': 2454, 'dropout_rate': 0.23497234236258646, 'l1_lambda': 0.060447826084657486, 'learning_rate': 0.0002075166931095152, 'lr_scheduler': 'ReduceLROnPlateau', 'num_epochs': 300, 'num_hidden_units': 1938, 'num_layers': 5, 'optimizer': 'AdamW', 'lrpatience': 14, 'weight_decay': 0.05054688467780508}

cells_forward_to_check = 3 * 60  # rows to check(minutes in this case)
threshold_cells_up = cells_forward_to_check * 0.5  # how many rows must achieve target %
percent_up = .35  # target percetage.
anticondition_threshold_cells_up = cells_forward_to_check * .2  # was .7
positivecase_weight = 1  #was 1.2 on 8/6
threshold_up = 0.5  ###At positive prediction = >X
ml_dataframe.dropna(subset=Chosen_Predictor, inplace=True)
length = ml_dataframe.shape[0]
ml_dataframe["Target"] = 0
target_Counter = 0
anticondition_UpCounter = 0
for i in range(1, cells_forward_to_check + 1):
    shifted_values = ml_dataframe["Current Stock Price"].shift(-i)
    condition_met_up = shifted_values > (
            ml_dataframe["Current Stock Price"] + (ml_dataframe["Current Stock Price"] * (percent_up / 100)))
    anticondition_up = shifted_values <= ml_dataframe["Current Stock Price"]
    target_Counter += condition_met_up.astype(int)
    anticondition_UpCounter += anticondition_up.astype(int)
ml_dataframe["Target"] = (
        (target_Counter >= threshold_cells_up) & (anticondition_UpCounter <= anticondition_threshold_cells_up)
).astype(int)
ml_dataframe.dropna(subset=["Target"], inplace=True)
y = ml_dataframe["Target"].copy()
X = ml_dataframe[Chosen_Predictor].copy()
X.reset_index(drop=True, inplace=True)
y.reset_index(drop=True, inplace=True)

largenumber = 1e5
X[Chosen_Predictor] = np.clip(X[Chosen_Predictor], -largenumber, largenumber)

nan_indices = np.argwhere(np.isnan(X.to_numpy()))  # Convert DataFrame to NumPy array
inf_indices = np.argwhere(np.isinf(X.to_numpy()))  # Convert DataFrame to NumPy array
neginf_indices = np.argwhere(np.isneginf(X.to_numpy()))  # Convert DataFrame to NumPy array
print("NaN values found at indices:" if len(nan_indices) > 0 else "No NaN values found.")
print("Infinite values found at indices:" if len(inf_indices) > 0 else "No infinite values found.")
print("Negative Infinite values found at indices:" if len(neginf_indices) > 0 else "No negative infinite values found.")

#
test_set_percentage = 0.2  # Specify the percentage of the data to use as a test set
split_index = int(len(X) * (1 - test_set_percentage))

X_test = X[split_index:]
y_test = y[split_index:]
X = X[:split_index]
y = y[:split_index]
# Fit the scaler on the entire training data
scaler_X_trainval = RobustScaler().fit(X)
scaler_y_trainval = RobustScaler().fit(y.values.reshape(-1, 1))



class BinaryClassificationNNwithDropout(nn.Module):
    def __init__(self, input_dim, num_hidden_units, dropout_rate):
        super(BinaryClassificationNNwithDropout, self).__init__()
        self.layer1 = nn.Linear(input_dim, num_hidden_units)
        self.layer2 = nn.Linear(num_hidden_units, int(num_hidden_units / 2))
        self.layer3 = nn.Linear(int(num_hidden_units / 2), int(num_hidden_units / 4))
        self.output_layer = nn.Linear(int(num_hidden_units / 4), 1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)  # Dropout layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.activation(self.layer1(x))
        x = self.dropout(x)  # Apply dropout after the activation
        x = self.activation(self.layer2(x))
        x = self.dropout(x)  # Apply dropout after the activation
        x = self.activation(self.layer3(x))
        x = self.dropout(x)  # Apply dropout after the activation
        x = self.sigmoid(self.output_layer(x))
        return x


class ClassNN_BatchNorm_DropoutA1(nn.Module):
    def __init__(self, input_dim, num_hidden_units, dropout_rate, num_layers=None):
        super(ClassNN_BatchNorm_DropoutA1, self).__init__()

        if num_layers is None:
            num_layers = 3  # Default value

        layers = []
        layers.append(nn.Linear(input_dim, num_hidden_units))
        layers.append(nn.BatchNorm1d(num_hidden_units))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(num_hidden_units, int(num_hidden_units / 2)))
            layers.append(nn.BatchNorm1d(int(num_hidden_units / 2)))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            num_hidden_units = int(num_hidden_units / 2)

        layers.append(nn.Linear(num_hidden_units, 1))
        layers.append(nn.Sigmoid())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


def create_optimizer(optimizer_name, learning_rate, momentum,  model_parameters,weight_decay=0,):
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


# at alpha =0, it will focus on precision. at alpha=1 it will soley focus f1
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


f1 = F1Score(num_classes=2, average='weighted', task='binary').to(device)
prec = Precision(num_classes=2, average='weighted', task='binary').to(device)
recall = Recall(num_classes=2, average='weighted', task='binary').to(device)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_metrics(outputs, labels, threshold=0.5):
    predicted_labels = (outputs > threshold).long()
    f1_score_value = f1(predicted_labels, labels)
    precision_value = prec(predicted_labels, labels)
    recall_value = recall(predicted_labels, labels)
    return f1_score_value, precision_value, recall_value



def train_model(hparams, X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    X_np = X.to_numpy()
    best_model_state_dict = None

    best_f1, best_precision, best_recall, best_val_loss = 0, 0, 0, float('inf')
    total_f1, total_precision, total_recall, total_val_loss = 0, 0, 0, 0
    num_folds = 0
    total_num_epochs = 0  # Initialize before entering the fold loop
    for fold, (train_index, val_index) in enumerate(kf.split(X_np)):
        X_train, X_val = X_np[train_index], X_np[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Fit the scaler on the training data of the current fold
        scaler_X_fold = RobustScaler().fit(X_train)
        scaler_y_fold = RobustScaler().fit(y_train.values.reshape(-1, 1))

        # Transform the training and validation data of the current fold
        X_train_scaled = scaler_X_fold.transform(X_train)
        X_val_scaled = scaler_X_fold.transform(X_val)
        y_train_scaled = scaler_y_fold.transform(y_train.values.reshape(-1, 1))
        y_val_scaled = scaler_y_fold.transform(y_val.values.reshape(-1, 1))


        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train_scaled, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val_scaled, dtype=torch.float32).to(device)
        batch_size = hparams["batch_size"]
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        patience = 5  # Number of epochs with no improvement to wait before stopping
        patience_counter = 0
        num_layers = hparams["num_layers"]
        # print( hparams['dropout_rate'],num_layers)
        model = ClassNN_BatchNorm_DropoutA1(X_train.shape[1], hparams["num_hidden_units"], hparams['dropout_rate'],
                                            num_layers).to(
            device)
        optimizer = create_optimizer(hparams["optimizer"], hparams["learning_rate"], hparams.get("momentum", 0),
                                      model.parameters(),hparams.get("weight_decay",0))
        model.train()
        weight = torch.Tensor([positivecase_weight]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
        # criterion = nn.BCELoss(weight=weight)
        # criterion = custom_loss()
        optimizer_name = hparams["optimizer"]
        num_epochs = hparams["num_epochs"]

        if hparams.get("lr_scheduler") == 'StepLR':
            lr_scheduler = StepLR(optimizer, hparams["step_size"], hparams["gamma"])
        elif hparams.get("lr_scheduler") == 'ExponentialLR':
            lr_scheduler = ExponentialLR(optimizer, hparams["gamma"])
        elif hparams.get("lr_scheduler") == 'ReduceLROnPlateau':
            lr_scheduler = ReduceLROnPlateau(optimizer, patience=hparams["lrpatience"], mode='min')
        l1_lambda = hparams.get("l1_lambda", 0)  # L1 regularization coefficient
        epoch_best_precision = 0
        epoch_best_recall = 0
        epoch_best_accuracy = 0
        epoch_best_val_loss = 10000000
        epoch_best_f1 = 0
        for epoch in range(num_epochs):
            all_val_outputs = []
            all_val_labels = []
            model.train()
            # Training step
            loss = None  # Define loss outside the inner loop
            for X_batch, y_batch in train_loader:
                if optimizer_name == "LBFGS":
                    def closure():
                        nonlocal loss  # Refer to the outer scope's loss variable
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

            epoch_sum_val_loss = 0.0
            epoch_total_samples = 0
            all_val_outputs = []
            all_val_labels = []
            for X_val_batch, y_val_batch in val_loader:
                with torch.no_grad():
                    val_outputs = model(X_val_batch)
                    all_val_outputs.append(val_outputs.flatten())  # Append the tensor
                    all_val_labels.append(y_val_batch.flatten())  # Append the tensor

                    # all_val_labels.extend(y_val_batch.tolist())
                    val_loss = criterion(val_outputs, y_val_batch)
                    # Add L1 regularization to validation loss
                    l1_val_reg = torch.tensor(0.).to(device)
                    for param in model.parameters():
                        l1_val_reg += torch.norm(param, 1)
                    val_loss += l1_lambda * l1_val_reg

                    epoch_sum_val_loss += val_loss.item() * len(y_val_batch)  # Multiply by batch size
                    epoch_total_samples += len(y_val_batch)
            all_val_outputs_tensor = torch.cat(all_val_outputs, dim=0)
            all_val_labels_tensor = torch.cat(all_val_labels, dim=0)

            all_val_outputs_bin = (all_val_outputs_tensor > 0.5).long()
            all_val_labels_bin = (all_val_labels_tensor > 0.5).long()
            epoch_avg_f1 = f1(all_val_labels_bin, all_val_outputs_bin)
            epoch_avg_precision = prec(all_val_labels_bin, all_val_outputs_bin)
            epoch_avg_val_loss = epoch_sum_val_loss / epoch_total_samples
            epoch_avg_recall = recall(all_val_labels_bin, all_val_outputs_bin)

            # Update best scores and early stopping counter
            if epoch_avg_val_loss < best_val_loss:
                best_val_loss = epoch_avg_val_loss
                best_f1 = epoch_avg_f1
                best_precision = epoch_avg_precision
                best_recall = epoch_avg_recall
                best_model_state_dict = copy.deepcopy(model.state_dict())

                patience_counter = 0


            else:
                patience_counter += 1
            if isinstance(lr_scheduler, ReduceLROnPlateau):
                lr_scheduler.step(epoch_avg_val_loss)

                # Only reset the counter if the validation loss improved

            if patience_counter >= patience:
                print(f"Early stop"
                      f"ping triggered at epoch {epoch + 1}")
                break
            total_f1 += epoch_best_f1
            total_precision += epoch_best_precision
            total_recall += epoch_best_recall
            total_val_loss += epoch_avg_val_loss
            num_folds += 1
            total_num_epochs += 1  # Increment for each epoch run
        avg_val_loss_per_epoch_across_folds = total_val_loss / total_num_epochs
        avg_f1 = total_f1 / num_folds
        avg_precision = total_precision / num_folds
        avg_recall = total_recall / num_folds
        avg_val_loss = total_val_loss / num_folds

        return {
            'best_f1': best_f1,
            'best_precision': best_precision,
            'best_recall': best_recall,
            'best_val_loss': best_val_loss,
            'avg_f1': avg_f1,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'avg_val_loss': avg_val_loss,
            'avg_val_loss_per_epoch_across_folds': avg_val_loss_per_epoch_across_folds,
            'best_model_state_dict': best_model_state_dict
        }


# Define Optuna Objective
def objective(trial):
    print(datetime.now())

    # print(len(X_val_tensor))
    learning_rate = trial.suggest_float("learning_rate", .00001, 0.01, log=True)  # 0003034075497582067
    num_epochs = trial.suggest_int("num_epochs", 5, 3800) #230  291  400-700
    # batch_size = trial.suggest_int("batch_size", 20, 3000)  # 10240  3437
    batch_size = trial.suggest_int('batch_size', 20, 2500)  # 2000
    num_hidden_units = trial.suggest_int("num_hidden_units", 800,4000)  # 2560 #83 125 63  #7000
    # weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1,log=True)  # Adding L2 regularization parameter
    l1_lambda = trial.suggest_float("l1_lambda", 1e-5, 1e-1,log=True)  # l1 regssss

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW", "RMSprop","SGD"])  #
    num_layers = trial.suggest_int("num_layers", 1, 5)  # example
    if num_layers == 1:
        dropout_rate = 0
    else:
        dropout_rate = trial.suggest_float("dropout_rate", 0, 0.5)

    if optimizer_name == "SGD":
        momentum = trial.suggest_float('momentum', 0, 0.9)  # Only applicable if using SGD with momentum
    else:

        momentum = 0  # Default value if not using SGD
    lrpatience = None  # Define  default value outside the conditional block
    step_size = None  # Define  default value for step_size
    gamma = None



    lr_scheduler_name = trial.suggest_categorical('lr_scheduler', ['StepLR', 'ExponentialLR', 'ReduceLROnPlateau'])

    if lr_scheduler_name == 'StepLR':
        step_size = trial.suggest_int('step_size', 5, 50)
        gamma = trial.suggest_float('gamma', 0.1, 1)
    elif lr_scheduler_name == 'ExponentialLR':
        gamma = trial.suggest_float('gamma', 0.1, 1)
    elif lr_scheduler_name == 'ReduceLROnPlateau':
        lrpatience = trial.suggest_int('lrpatience', 5, 20)

    # Call the train_model function with the current hyperparameters
    result = train_model(

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
            # "weight_decay": weight_decay,
            "lr_scheduler": lr_scheduler_name,  # Pass the scheduler instance here
            "lrpatience": lrpatience,  # Pass the value directly
            "gamma": gamma,  # Pass the value directly
            "step_size": step_size,  # Add step_size parameter for StepLR
            # Add more hyperparameters as needed
        },
        X, y
    )
    best_f1 = result['best_f1']
    best_precision = result['best_precision']
    best_recall = result['best_recall']
    best_val_loss = result['best_val_loss']
    avg_f1 = result['avg_f1']
    avg_precision = result['avg_precision']
    avg_recall = result['avg_recall']
    avg_val_loss = result['avg_val_loss']
    avg_val_loss_per_epoch_across_folds = result['avg_val_loss_per_epoch_across_folds']
    best_model_state_dict = result['best_model_state_dict']

    print("best f1 score: ", best_f1, "best precision score: ", best_precision,
          "best val loss: ",

          best_val_loss)
    alpha = .3
    combined_metric = (alpha * (1 - best_f1)) + ((1 - alpha) * (1 - best_precision))
    return best_val_loss  # Note this is actually criterion, which is currently mae.
    #    # return prec_score  # Optuna will try to maximize this value


##TODO Comment out to skip the hyperparameter selection.  Swap "best_params".
try:
    study = optuna.load_study(study_name='SPY_best_val_loss_allfeatures_l1ONLY_3hr35percent',
                              storage='sqlite:///SPY_best_val_loss_allfeatures_l1ONLY_3hr35percent.db')
    print("Study Loaded.")
    try:
        best_params = study.best_params
        best_trial = study.best_trial
        best_value = study.best_value
        print("Best Value:", best_value)

        print(best_params)
        print("Best Trial:", best_trial)

    except Exception as e:
        print(e)
except KeyError:
    study = optuna.create_study(direction="minimize", study_name='SPY_best_val_loss_allfeatures_l1ONLY_3hr35percent',
                                storage='sqlite:///SPY_best_val_loss_allfeatures_l1ONLY_3hr35percent.db')
"Keyerror, new optuna study created."  #

study.optimize(objective, n_trials=5000)
best_params = study.best_params
#
# best_params ={'batch_size': 1197, 'dropout_rate': 0.4608394623321738, 'l1_lambda': 0.01320220981011121, 'learning_rate': 1.1625919878731402e-05, 'lr_scheduler': 'ReduceLROnPlateau', 'lrpatience': 10, 'num_epochs': 211, 'num_hidden_units': 114, 'num_layers': 5, 'optimizer': 'RMSprop', 'weight_decay': 0.00013649093677743602}
# best_params = {'batch_size': 972, 'dropout_rate': 0.23030333490770447, 'gamma': 0.3089135336987861, 'l1_lambda': 0.0950910207258489, 'learning_rate': 1.5716591458439578e-05, 'lr_scheduler': 'StepLR', 'num_epochs': 153, 'num_hidden_units': 2520, 'num_layers': 5, 'optimizer': 'Adam', 'step_size': 45, 'weight_decay': 0.09863209738072187}
#####################################################################################################
# ################

## Train the model with the best hyperparameters
print("~~~~training model using best params.~~~~")
test_result = train_model(
    best_params, X, y)
model_up_nn = ClassNN_BatchNorm_DropoutA1(X.shape[1], best_params["num_hidden_units"],
                                          best_params["dropout_rate"], best_params["num_layers"]
).to(device)
test_best_f1 = test_result['best_f1']
test_best_precision = test_result['best_precision']
test_best_recall = test_result['best_recall']
test_best_val_loss = test_result['best_val_loss']
test_avg_f1 = test_result['avg_f1']
test_avg_precision = test_result['avg_precision']
test_avg_recall = test_result['avg_recall']
test_avg_val_loss = test_result['avg_val_loss']
test_avg_val_loss_per_epoch_across_folds = test_result['avg_val_loss_per_epoch_across_folds']
best_model_state_dict = test_result['best_model_state_dict']
# Load saved state_dict into the model
model_up_nn.load_state_dict(best_model_state_dict)
model_up_nn.eval()
X_test_scaled = scaler_X_trainval.transform(X_test)

y_test_scaled = scaler_y_trainval.transform(y_test.values.reshape(-1, 1))
# TODO scaled or unscaled y?
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test_scaled, dtype=torch.float32).squeeze().to(device)

predicted_probabilities = model_up_nn(X_test_tensor).detach().cpu().numpy()
predicted_probabilities = (predicted_probabilities > threshold_up).astype(int)
predicted_up_tensor = torch.tensor(predicted_probabilities, dtype=torch.float32).squeeze().to(device)
num_positives_up = np.sum(predicted_probabilities)

task = "binary"
precision_up = Precision(num_classes=2, average='weighted', task='binary').to(device)(predicted_up_tensor,
                                                                                      y_test_tensor)  # move metric to same device as tensors
accuracy_up = Accuracy(num_classes=2, average='weighted', task=task).to(device)(predicted_up_tensor, y_test_tensor)
recall_up = Recall(num_classes=2, average='weighted', task=task).to(device)(predicted_up_tensor, y_test_tensor)
f1_up = F1Score(num_classes=2, average='weighted', task=task).to(device)(predicted_up_tensor, y_test_tensor)
# Print Number of Positive and Negative Samples
num_positive_samples = sum(y_test)
# num_negative_samples_up = len(y_up_test_tensor) - num_positive_samples_up

print("Metrics for Test Target_Up:", "\n")
print("Test Precision:", precision_up)
print("Test Accuracy:", accuracy_up)
print("Test Recall:", recall_up)
print("Test F1-Score:", f1_up, "\n")
print("Best Hyperparameters:", best_params)
print(f"Test Number of positive predictions: {num_positives_up}")
print("Number of Positive Samples(Target):", num_positive_samples)
# print("Number of Total Samples(Target_Up):", num_positive_samples_up + num_negative_samples_up)
# print('selected features: ',selected_features)

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
        "lrpatience": best_params.get('lrpatience'),
        "gamma": best_params.get('gamma'),
        "step_size": best_params.get('step_size'),
        'scaler_X': scaler_X_trainval,
        # 'scaler_X_min': scaler_X.min_,
        # 'scaler_X_scale': scaler_X.scale_,
        'scaler_y': scaler_y_trainval,
        # 'scaler_y_min': scaler_y.min_,
        # 'scaler_y_scale': scaler_y.scale_,
        'model_state_dict': model_up_nn.state_dict(),
    }

    # Remove keys with None values
    save_dict = {key: value for key, value in save_dict.items() if value is not None}

    # Save the dict if exist
    if save_dict:
        torch.save(save_dict, model_filename_up)

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
        # f"Number of Positive Samples (Target_Up): {num_positive_samples_up}\nNumber of Negative Samples (Target_Up): {num_negative_samples_up}\n"
        f"Threshold Up (sensitivity): {threshold_up}\n"
        f"Target Underlying Percentage Up: {percent_up}\n"
        f"Anticondition: {anticondition_UpCounter}\n"
        f"Weight multiplier: {positivecase_weight}")
