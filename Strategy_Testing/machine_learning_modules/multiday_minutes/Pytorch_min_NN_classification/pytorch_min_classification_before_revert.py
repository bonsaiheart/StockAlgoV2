import copy
import datetime
import os
from datetime import datetime
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold, TimeSeriesSplit, train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.utils import compute_class_weight
from torch.optim.lr_scheduler import StepLR, ExponentialLR, ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from torchmetrics import Accuracy, Precision, Recall, F1Score

DF_filename = r"../../../../data/historical_multiday_minute_DF/older/SPY_historical_multiday_min.csv"
# Chosen_Predictor = ['ExpDate', 'LastTradeTime', 'Current Stock Price',
#                     'Current SP % Change(LAC)', 'Maximum Pain', 'Bonsai Ratio',
#                     'Bonsai Ratio 2', 'B1/B2', 'B2/B1', 'PCR-Vol', 'PCR-OI',
#                     'PCRv @CP Strike', 'PCRoi @CP Strike', 'PCRv Up1', 'PCRv Up2',
#                     'PCRv Up3', 'PCRv Up4', 'PCRv Down1', 'PCRv Down2', 'PCRv Down3',
#                     'PCRv Down4', 'PCRoi Up1', 'PCRoi Up2', 'PCRoi Up3', 'PCRoi Up4',
#                     'PCRoi Down1', 'PCRoi Down2', 'PCRoi Down3', 'PCRoi Down4',
#                     'ITM PCR-Vol', 'ITM PCR-OI', 'ITM PCRv Up1', 'ITM PCRv Up2',
#                     'ITM PCRv Up3', 'ITM PCRv Up4', 'ITM PCRv Down1', 'ITM PCRv Down2',
#                     'ITM PCRv Down3', 'ITM PCRv Down4', 'ITM PCRoi Up1', 'ITM PCRoi Up2',
#                     'ITM PCRoi Up3', 'ITM PCRoi Up4', 'ITM PCRoi Down1', 'ITM PCRoi Down2',
#
#                     'ITM PCRoi Down3', 'ITM PCRoi Down4', 'ITM OI', 'Total OI',
#                     'ITM Contracts %', 'Net_IV', 'Net ITM IV', 'Net IV MP', 'Net IV LAC',
#                     'NIV Current Strike', 'NIV 1Higher Strike', 'NIV 1Lower Strike',
#                     'NIV 2Higher Strike', 'NIV 2Lower Strike', 'NIV 3Higher Strike',
#                     'NIV 3Lower Strike', 'NIV 4Higher Strike', 'NIV 4Lower Strike',
#                     'NIV highers(-)lowers1-2', 'NIV highers(-)lowers1-4',
#                     'NIV 1-2 % from mean', 'NIV 1-4 % from mean', 'Net_IV/OI',
#                     'Net ITM_IV/ITM_OI', 'Closest Strike to CP', 'RSI', 'AwesomeOsc',
#                     'RSI14', 'RSI2', 'AwesomeOsc5_34']
# ##had highest corr for 3-5 hours with these:
Chosen_Predictor = ['LastTradeTime', 'Current Stock Price', 'Current SP % Change(LAC)', 'Maximum Pain','Bonsai Ratio','Bonsai Ratio 2','ITM PCR-Vol','ITM PCRoi Up1', 'RSI14','AwesomeOsc5_34', 'Net_IV']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device: ', device)
ml_dataframe = pd.read_csv(DF_filename)
print(ml_dataframe.columns)

ml_dataframe['LastTradeTime'] = ml_dataframe['LastTradeTime'].apply(
    lambda x: datetime.strptime(str(x), '%y%m%d_%H%M') if not pd.isna(x) else np.nan)
ml_dataframe['LastTradeTime'] = ml_dataframe['LastTradeTime'].apply(lambda x: x.timestamp())
ml_dataframe['LastTradeTime'] = ml_dataframe['LastTradeTime'] / (60 * 60 * 24 * 7)

ml_dataframe['ExpDate'] = ml_dataframe['ExpDate'].astype(float)

cells_forward_to_check = 3 * 60  # rows to check(minutes in this case)
threshold_cells_up = cells_forward_to_check * 0.5  # how many rows must achieve target %
percent_up = .4  # target percetage.
anticondition_threshold_cells_up = cells_forward_to_check * .7  # was .7


positivecase_weight = 1  # Your desired multiplier

threshold_up = 0.5  ###At positive prediction = >X
patience = 200

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



# largenumber = 1e5
# X[Chosen_Predictor] = np.clip(X[Chosen_Predictor], -largenumber, largenumber)

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
        # layers.append(nn.Sigmoid())

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
# def custom_loss(outputs, targets, alpha=0.7):
#     epsilon = 1e-7
#     # Apply Sigmoid to transform outputs into probabilities
#     # Smooth approximation of True Positives, False Positives, False Negatives
#     TP = (outputs * targets).sum()
#     FP = ((1 - targets) * outputs).sum()
#     FN = (targets * (1 - outputs)).sum()
#     precision = TP / (TP + FP + epsilon)
#     recall = TP / (TP + FN + epsilon)
#     f1_score = 2 * (precision * recall) / (precision + recall + epsilon)
#     # print(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1_score}")
#
#     # Combine F1 Score and Precision with weight alpha
#     loss = alpha * (1 - f1_score) + (1 - alpha) * (1 - precision)
#     # print(loss)
#     return loss


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



def train_model_with_time_series_cv(hparams, X, y):
    tscv = TimeSeriesSplit(n_splits=2)
    X_np = X.to_numpy()
    best_model_state_dict = None

    best_f1, best_precision, best_recall, best_val_loss = 0, 0, 0, float('inf')
    total_f1, total_precision, total_recall, total_val_loss = 0, 0, 0, 0
    num_folds = 0
    total_num_epochs = 0  # Initialize before entering the fold loop
    for fold, (train_index, val_index) in enumerate(tscv.split(X_np)):
        X_train, X_val = X_np[train_index], X_np[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Fit the scaler on the training data of
        # the current fold
        scaler_X_fold = RobustScaler().fit(X_train)
        # scaler_y_fold = RobustScaler().fit(y_train.values.reshape(-1, 1))

        # Transform the training and validation data of the current fold
        X_train_scaled = scaler_X_fold.transform(X_train)
        X_val_scaled = scaler_X_fold.transform(X_val)
        # y_train_scaled = scaler_y_fold.transform(y_train.values.reshape(-1, 1))
        # y_val_scaled = scaler_y_fold.transform(y_val.values.reshape(-1, 1))
        #

        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val.values.reshape(-1, 1), dtype=torch.float32).to(device)
        batch_size = hparams["batch_size"]
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        num_layers = hparams["num_layers"]
        # print( hparams['dropout_rate'],num_layers)
        model = ClassNN_BatchNorm_DropoutA1(X_train.shape[1], hparams["num_hidden_units"],  hparams.get('dropout_rate', 0),
                                            num_layers).to(
            device)
        optimizer = create_optimizer(hparams["optimizer"], hparams["learning_rate"], hparams.get("momentum", 0),
                                      model.parameters(),hparams.get("weight_decay",0))
        model.train()
        # Calculate class weights
        class_weights = compute_class_weight('balanced', classes=[0, 1], y=y_train)

        # Get the weight for the positive class (class 1)
        balanced_weight = class_weights[1]

        # Now, you can multiply this balanced weight by your desired multiplier (positivecase_weight)
        final_weight = balanced_weight * positivecase_weight

        weight = torch.Tensor([final_weight]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
        # criterion = nn.BCELoss(weight=weight)
        # criterion = custom_loss
        optimizer_name = hparams["optimizer"]
        num_epochs = hparams["num_epochs"]

        if hparams.get("lr_scheduler") == 'StepLR':
            lr_scheduler = StepLR(optimizer, hparams["step_size"], hparams["gamma"])
        elif hparams.get("lr_scheduler") == 'ExponentialLR':
            lr_scheduler = ExponentialLR(optimizer, hparams["gamma"])
        elif hparams.get("lr_scheduler") == 'ReduceLROnPlateau':
            lr_scheduler = ReduceLROnPlateau(optimizer, patience=hparams["lrpatience"], mode='min')
        else:
            lr_scheduler = None
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
                    #     loss = criterion(outputs, y_batch)
                    #
                    #     l1_reg = torch.tensor(0., requires_grad=True).to(device)
                    #     for param in model.parameters():
                    #         l1_reg += torch.norm(param, 1)
                    #     loss += l1_lambda * l1_reg
                    #     loss.backward()
                    #     return loss
                    #
                    # optimizer.step(closure)
                else:
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                    # print(outputs)
                    loss = criterion(outputs, y_batch)
                    # Add L1 regularization to loss
                    l1_reg = torch.tensor(0., requires_grad=True).to(device)
                    for param in model.parameters():
                        l1_reg += torch.norm(param, 1)
                    loss += l1_lambda * l1_reg
                    loss.backward()
                    optimizer.step()
                    # if lr_scheduler is not None:
                    #     if isinstance(lr_scheduler, StepLR) or isinstance(lr_scheduler, ExponentialLR):
                    #         lr_scheduler.step()
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
            # if isinstance(lr_scheduler, ReduceLROnPlateau):
            #     lr_scheduler.step(epoch_avg_val_loss)

                # Only reset the counter if the validation loss improved
            patience = 20
            if patience_counter >= patience:
                print(f"Early stopping triggered at fold {fold+1},  epoch {epoch + 1}")
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

def train_model_full_dataset(hparams, X, y):
    X_np = X.to_numpy()
    y_np = y.to_numpy()

    # Fit and transform the data

    train_index, val_index = train_test_split(np.arange(len(X)), test_size=0.2)
    X_train, X_val = X_np[train_index], X_np[val_index]
    y_train, y_val = y_np[train_index], y_np[val_index]
    scaler_X = RobustScaler().fit(X_train)
    X_train_scaled = scaler_X.transform(X_train)
    X_val_scaled = scaler_X.transform(X_val)

    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32).to(device)
    X_test_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32).to(device)

    # Create DataLoader
    batch_size = hparams["batch_size"]
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # Initialize model and optimizer
    model = ClassNN_BatchNorm_DropoutA1(X_np.shape[1], hparams["num_hidden_units"], hparams.get('dropout_rate', 0), hparams["num_layers"]).to(device)
    optimizer = create_optimizer(hparams["optimizer"], hparams["learning_rate"], hparams.get("momentum", 0), model.parameters(), hparams.get("weight_decay", 0))

    # Loss function
    class_weights = compute_class_weight('balanced', classes=[0, 1], y=y_train)
    balanced_weight = class_weights[1]
    final_weight = balanced_weight * positivecase_weight
    weight = torch.Tensor([final_weight]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
    optimizer_name = hparams["optimizer"]

    l1_lambda = hparams.get("l1_lambda", 0)

    if hparams.get("lr_scheduler") == 'StepLR':
        lr_scheduler = StepLR(optimizer, hparams["step_size"], hparams["gamma"])
    elif hparams.get("lr_scheduler") == 'ExponentialLR':
        lr_scheduler = ExponentialLR(optimizer, hparams["gamma"])
    elif hparams.get("lr_scheduler") == 'ReduceLROnPlateau':
        lr_scheduler = ReduceLROnPlateau(optimizer, patience=hparams["lrpatience"], mode='min')
    else:
        lr_scheduler = None
    # Training loop
    num_epochs = hparams["num_epochs"]
    best_val_loss = float('inf')  # Initialize with a high value

    for epoch in range(num_epochs):
        all_val_outputs = []
        all_val_labels = []
        model.train()
        # Training step
        loss = None  # Define loss outside the inner loop
        for X_batch, y_batch in loader:
            if optimizer_name == "LBFGS":
                def closure():
                    nonlocal loss  # Refer to the outer scope's loss variable
                    optimizer.zero_grad()
                    outputs = model(X_batch)
                #     loss = criterion(outputs, y_batch)
                #
                #     l1_reg = torch.tensor(0., requires_grad=True).to(device)
                #     for param in model.parameters():
                #         l1_reg += torch.norm(param, 1)
                #     loss += l1_lambda * l1_reg
                #     loss.backward()
                #     return loss
                #
                # optimizer.step(closure)
            else:
                optimizer.zero_grad()
                outputs = model(X_batch)
                # print(outputs)
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
            validation_dataset_size = len(val_loader.dataset)
            # print(f"Validation dataset size: {validation_dataset_size}")
            # if batch_size >= validation_dataset_size:
            #     print("Warning: Batch size is greater than or equal to the size of the validation dataset!")
            # else:
            #     print("Batch size is appropriate.")
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
        # Early stopping and other hyperparameters
        patience_counter = 0
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
        # if isinstance(lr_scheduler, ReduceLROnPlateau):
        #     lr_scheduler.step(epoch_avg_val_loss)

        # Only reset the counter if the validation loss improved
        if patience_counter >= patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    X_test_scaled = scaler_X.transform(X_test)
    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).squeeze().to(device)

    predicted_probabilities = model(X_test_tensor).detach().cpu().numpy()
    predicted_binary = (predicted_probabilities > threshold_up).astype(int)
    predicted_up_tensor = torch.tensor(predicted_binary, dtype=torch.float32).squeeze().to(device)
    num_positives_up = np.sum(predicted_binary)
    task = "binary"
    precision_up = Precision(num_classes=2, average='binary', task=task).to(device)(predicted_up_tensor, y_test_tensor)
    accuracy_up = Accuracy(num_classes=2, average='binary', task=task).to(device)(predicted_up_tensor, y_test_tensor)
    recall_up = Recall(num_classes=2, average='binary', task=task).to(device)(predicted_up_tensor, y_test_tensor)
    f1_up = F1Score(num_classes=2, average='binary', task=task).to(device)(predicted_up_tensor, y_test_tensor)
    # Print Number of Positive and Negative Samples
    num_positive_samples = sum(y_test)
    # num_negative_samples_up = len(y_up_test_tensor) - num_positive_samples_up
    print('Total outcomes: ', len(y_test))
    print(f"Test Number of positive predictions: {num_positives_up}")
    print('# True positive samples: ',y_test_tensor.sum())

    print("\nMetrics for Test Target_Up:", "\n")
    print("Test Precision:", precision_up)
    print("Test Accuracy:", accuracy_up)
    print("Test Recall:", recall_up)
    print("Test F1-Score:", f1_up, "\n")

    # print("Number of Total Samples(Target_Up):", num_positive_samples_up + num_negative_samples_up)
    # print('selected features: ',selected_features)

    result_dict = {
        'model': model,
        'scaler_X': scaler_X,
        'precision_up': precision_up.item(),
        'accuracy_up': accuracy_up.item(),
        'recall_up': recall_up.item(),
        'f1_up': f1_up.item(),
        'num_positives_up': num_positives_up,
        'num_positive_samples': num_positive_samples,
        'len_y_test': len(y_test)
    }

    return result_dict
# Define Optuna Objective
def objective(trial):
    print(datetime.now())

    # Initialize variables to None
    step_size = None
    gamma = None
    lrpatience = None
    learning_rate = trial.suggest_float("learning_rate", .00001, 0.01, log=True)  # 0003034075497582067
    num_epochs = trial.suggest_int("num_epochs", 5, 500) #230  291  400-700
    batch_size = trial.suggest_int('batch_size', 2, 2400)  # 2000make sure its smaller than val dataset

    num_hidden_units = trial.suggest_int("num_hidden_units", 800,4000)  # 2560 #83 125 63  #7000
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-1,log=True)  # Adding L2 regularization parameter
    l1_lambda = trial.suggest_float("l1_lambda", 1e-2, 1e-1,log=True)  # l1 regssss

    optimizer_name = trial.suggest_categorical("optimizer", ["Adam","SGD" ])  #"AdamW", "RMSprop","SGD"
    num_layers = trial.suggest_int("num_layers", 1, 5)  # example
    if num_layers == 1:
        dropout_rate = 0
    else:
        dropout_rate = trial.suggest_float("dropout_rate", 0.4, 0.8)
    #
    if optimizer_name == "SGD":
        momentum = trial.suggest_float('momentum', 0, 0.9)  # Only applicable if using SGD with momentum
    else:

        momentum = None  # Default value if not using SGD


    lr_scheduler_name = trial.suggest_categorical('lr_scheduler', ['StepLR', 'ReduceLROnPlateau'])

    if lr_scheduler_name == 'StepLR':
        step_size = trial.suggest_int('step_size', 5, 50)
        gamma = trial.suggest_float('gamma', 0.1, 1)

    elif lr_scheduler_name == 'ReduceLROnPlateau':
        lrpatience = trial.suggest_int('lrpatience', 5, 20)

    # Call the train_model function with the current hyperparameters
    result = train_model_full_dataset(

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
            "lrpatience": lrpatience,  # Pass the value directly
            "gamma": gamma,  # Pass the value directly
            "step_size": step_size,  # Add step_size parameter for StepLR
            # Add more hyperparameters as needed
        },
        X, y
    )
    # best_f1 = result['best_f1']
    # best_precision = result['best_precision']
    # best_recall = result['best_recall']
    # best_val_loss = result['best_val_loss']
    # avg_f1 = result['avg_f1']
    # avg_precision = result['avg_precision']
    # avg_recall = result['avg_recall']
    # avg_val_loss = result['avg_val_loss']
    # avg_val_loss_per_epoch_across_folds = result['avg_val_loss_per_epoch_across_folds']
    # best_model_state_dict = result[('best_model_state_dict')]
    #
    # print("best f1 score: ", best_f1, "best precision score: ", best_precision,
    #       "best val loss: ", best_val_loss)
    # alpha = .3
    # return avg_val_loss  # Note this is actually criterion, which is currently mae.
    #    # return prec_score  # Optuna will try to maximize this value
    precision_up = result['precision_up']
    accuracy_up = result['accuracy_up']
    recall_up = result['recall_up']
    f1_up = result['f1_up']
    alpha = .5

    combined_metric = (alpha * (1 - f1_up)) + ((1 - alpha) * (1 - precision_up))

    return combined_metric
##TODO a
try:


    study = optuna.load_study(study_name='study3',
                              storage='sqlite:///study3.db')
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
    study = optuna.create_study(direction="minimize", study_name='study3',
                                storage='sqlite:///study3.db')
"Keyerror, new optuna study created."  #

study.optimize(objective, n_trials=5000)
best_params = study.best_params_
# best_params = {'batch_size': 20, 'l1_lambda': 0.025672338776057218, 'learning_rate': 0.006407862762170161, 'num_epochs': 347, 'num_hidden_units': 1800, 'num_layers': 2, 'optimizer': 'Adam', 'weight_decay': 0.00028501994741074124}# best_params ={'batch_size': 1197, 'dropout_rate': 0.4608394623321738, 'l1_lambda': 0.01320220981011121, 'learning_rate': 1.1625919878731402e-05, 'lr_scheduler': 'ReduceLROnPlateau', 'lrpatience': 10, 'num_epochs': 211, 'num_hidden_units': 114, 'num_layers': 5, 'optimizer': 'RMSprop', 'weight_decay': 0.00013649093677743602}
# best_params = best_params = {
#     'batch_size': 32,
#     'l1_lambda': 0.05,
#     'learning_rate': 1e-04,
#     'lr_scheduler': 'ReduceLROnPlateau',
#     'lrpatience': 10,  # Add this line
#     'num_epochs': 100,
#     'num_hidden_units': 1024,
#     'num_layers': 3,
#     'optimizer': 'Adam',
#     'weight_decay': 0.05
# }#####################################################################################################
# # ################
# best_params = {'batch_size': 2065, 'l1_lambda': 0.024106635698321235, 'learning_rate': 0.0032431984870763698, 'num_epochs': 328, 'num_hidden_units': 907, 'num_layers': 1, 'optimizer': 'Adam', 'weight_decay': 0.00011742699581949843}
## Train the model with the best hyperparameters
print("~~~~training model using best params.~~~~")
result = train_model_full_dataset(best_params, X, y)

# Access the metrics, model, and scaler
trained_model = result['model']
scaler_X = result['scaler_X']
precision_up = result['precision_up']
accuracy_up = result['accuracy_up']
recall_up = result['recall_up']
f1_up = result['f1_up']
num_positives_up = result['num_positives_up']
num_positive_samples = result['num_positive_samples']
len_y_test = result['len_y_test']

# Now you can print or use these values as needed
# print("Test Precision:", precision_up)
# print("Test Accuracy:", accuracy_up)
# print("Test Recall:", recall_up)
# print("Test F1-Score:", f1_up)



input_val = input("Would you like to save these models? y/n: ").upper()
if input_val == "Y":
    model_summary = input("Save this set of models as: ")
    model_directory = os.path.join("../../../Trained_Models", f"{model_summary}")
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    model_filename_up = os.path.join(model_directory, "tar"
                                                      "get_up.pth")
    save_dict = {
        'model_class': trained_model.__class__.__name__,
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
        'scaler_X': scaler_X,
        'model_state_dict': trained_model.state_dict(),
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
