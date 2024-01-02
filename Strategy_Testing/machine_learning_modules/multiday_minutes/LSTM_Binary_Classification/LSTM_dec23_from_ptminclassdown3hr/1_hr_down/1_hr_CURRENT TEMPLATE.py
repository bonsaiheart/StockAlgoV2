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
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics import Precision, Accuracy, Recall, F1Score
import torch.nn.functional as F
import re

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
# ml_dataframe=ml_dataframe[:15000]
# FEATURE SET 1?
# Chosen_Predictor= ['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'B2/B1', 'PCRoi Up1', 'PCRoi Down1', 'ITM PCR-OI', 'ITM PCRoi Up1', 'ITM PCRoi Down1', 'ITM PCRoi Down2', 'ITM PCRoi Down3', 'ITM PCRoi Down4', 'ITM Contracts %', 'Net ITM IV', 'NIV highers(-)lowers1-4', 'Net_IV/OI', 'Net ITM_IV/ITM_OI']
# Feature set 2
# Chosen_Predictor = [
#     'Current SP % Change(LAC)','B1/B2', 'B2/B1',  'PCRv @CP Strike','PCRoi @CP Strike','PCRv Up1', 'PCRv Down1','PCRoi Up4','PCRoi Down3' ,'ITM PCR-Vol','ITM PCR-OI', 'Net IV LAC',
#     'RSI14', 'AwesomeOsc5_34',
# ]
# Feat. Set 3
Chosen_Predictor = [
    "Bonsai Ratio",
    "Bonsai Ratio 2",
    "B1/B2",
    "B2/B1",
    "PCRoi Up1",
    "PCRoi Down1",
    "ITM PCR-OI",
    "ITM PCRoi Up1",
    "ITM PCRoi Down1",
    "ITM Contracts %",
    "Net ITM IV",
    "NIV highers(-)lowers1-4",
]

# feat. set 4
Chosen_Predictor = [
    "Bonsai Ratio",
    "Bonsai Ratio 2",
    "PCRv Up1",
    "PCRv Down1",
    "ITM PCR-Vol",
    "Net IV LAC",
]
# Best Params: {'learning_rate': 0.002973181466202932, 'num_epochs': 365, 'batch_size': 2500, 'optimizer': 'Adam', 'dropout_rate': 0.05, 'num_hidden_units': 2350}

# TODO scale Predictors based on data ranges/types

study_name = "_1hr_20pt_down_FeatSet4"
n_trials = 10000
cells_forward_to_check = 60 * 1
percent_down = 0.2  # as percent

threshold_cells_up = cells_forward_to_check * 0.3
# The anticondition is when the price goes below the 1st price.  The threshold is how many cells can be anticondition, and still have True label.
anticondition_threshold_cells = cells_forward_to_check * 0.2  # was .7
theshhold_down = 0.5  ###TODO these dont do any

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("device: ", device)

# print(ml_dataframe.columns)
ml_dataframe["LastTradeTime"] = ml_dataframe["LastTradeTime"].apply(
    lambda x: datetime.strptime(str(x), "%y%m%d_%H%M") if not pd.isna(x) else np.nan
)
ml_dataframe["LastTradeTime"] = ml_dataframe["LastTradeTime"].apply(
    lambda x: x.timestamp()
)
ml_dataframe["LastTradeTime"] = ml_dataframe["LastTradeTime"] / (60 * 60 * 24 * 7)
ml_dataframe["ExpDate"] = ml_dataframe["ExpDate"].astype(float)

ml_dataframe.dropna(subset=Chosen_Predictor, inplace=True)
length = ml_dataframe.shape[0]
ml_dataframe["Target_Up"] = 0
targetUpCounter = 0
anticondition_UpCounter = 0
for i in range(1, cells_forward_to_check + 1):
    shifted_values = ml_dataframe["Current Stock Price"].shift(-i)
    condition_met_up = shifted_values < (
        ml_dataframe["Current Stock Price"]
        - (ml_dataframe["Current Stock Price"] * (percent_down / 100))
    )
    anticondition_up = shifted_values >= ml_dataframe["Current Stock Price"]
    targetUpCounter += condition_met_up.astype(int)
    anticondition_UpCounter += anticondition_up.astype(int)
ml_dataframe["Target_Up"] = (
    (targetUpCounter >= threshold_cells_up)
    & (anticondition_UpCounter <= anticondition_threshold_cells)
).astype(int)
ml_dataframe.dropna(subset=["Target_Up"], inplace=True)
y_up = ml_dataframe["Target_Up"]
X = ml_dataframe[Chosen_Predictor]

# Reset index
X.reset_index(drop=True, inplace=True)

# # make test set
X_temp, X_test, y_up_temp, y_up_test = train_test_split(
    X, y_up, test_size=0.05, random_state=None, shuffle=False
)

# Split the temp set into validation and TRAIN sets
X_train, X_val, y_up_train, y_up_val = train_test_split(
    X_temp, y_up_temp, test_size=0.15, random_state=None, shuffle=False
)


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
X_trainval_tensor = torch.tensor(X_trainval_scaled, dtype=torch.float32).to(device)
y_trainval_tensor = torch.tensor(y_trainval.values, dtype=torch.float32).to(device)


# Function to convert scaled data to tensors
def convert_to_tensor(scaler, X_train, X_val, X_test, device):
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    return (
        torch.tensor(X_train_scaled, dtype=torch.float32).to(device),
        torch.tensor(X_val_scaled, dtype=torch.float32).to(device),
        torch.tensor(X_test_scaled, dtype=torch.float32).to(device),
    )


# TODO may need to change the way i throw away parts of sequences... maybe these should be created, THEN split into test/train/val
def create_sequences(data, seq_length, targets=None):
    sequences = []
    target_seq = []
    min_length = len(data)
    # print(targets)
    if targets is not None:
        min_length = min(len(data), len(targets))

    for i in range(min_length - seq_length):
        seq = data[i : i + seq_length]
        sequences.append(seq)
        if targets is not None:
            label = targets[i + seq_length - 1]
            target_seq.append(label)

    if not sequences:
        return torch.empty(0), torch.empty(0)
    else:
        sequences = torch.stack(sequences)
        if targets is not None:
            target_seq = torch.tensor(target_seq, dtype=torch.float32).to(device)
            return sequences, target_seq
        else:
            return sequences


# Create a scaler object and convert datasets to tensors
scaler = RobustScaler()
X_train_tensor, X_val_tensor, X_test_tensor = convert_to_tensor(
    scaler, X_train, X_val, X_test, device
)
X_test_tensor = X_test_tensor.to(device)

X_train_tensor = X_train_tensor.to(device)
X_val_tensor = X_val_tensor.to(device)

y_up_train_tensor = torch.tensor(y_up_train.values, dtype=torch.float32).to(device)
y_up_val_tensor = torch.tensor(y_up_val.values, dtype=torch.float32).to(device)
y_up_test_tensor = torch.tensor(y_up_test.values, dtype=torch.float32).to(device)

# print(type(y_up_test_tensor),type(y_up_test))
# Print lengths of datasets
print(
    f"Train length: {len(X_train_tensor)}, Validation length: {len(X_val_tensor)}, Test length: {len(X_test_tensor)}"
)

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
    ratio = (
        num_positive / num_negative if num_negative else float("inf")
    )  # Avoid division by zero
    print(f"{stage} ratio of pos/neg up: {ratio:.2f}")
    print(f"{stage} num_positive_up: {num_positive}")
    print(f"{stage} num_negative_up: {num_negative}\n")


print_dataset_statistics("Train", num_positive_up_train, num_negative_up_train)
print_dataset_statistics("Validation", num_positive_up_val, num_negative_up_val)
print_dataset_statistics("Test", num_positive_up_test, num_negative_up_test)


def plot_learning_curves(train_losses, val_losses):
    plt.figure(figsize=(12, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.title("Learning Curves")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()


class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout):
        print("HIDDENDEMSA", hidden_dims)
        # print(input_dim)
        # print("hiddendims"  ,hidden_dims)
        super(LSTMModel, self).__init__()
        self.layers = nn.ModuleList()
        self.hidden_dims = hidden_dims
        # Adjust dropout if only one layer is present
        if len(hidden_dims) == 1:
            dropout = 0  # Set dropout to 0 for a single-layer LSTM

        # Create LSTM layers with varying number of neurons
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            lstm_layer = nn.LSTM(prev_dim, hidden_dim, num_layers=1, batch_first=True)
            self.layers.append(lstm_layer)
            if dropout > 0:
                self.layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim

        # Output layer
        self.fc = nn.Linear(hidden_dims[-1], output_dim)

    def forward(self, x):
        # Initializing hidden state and cell state for each layer
        h0 = [
            torch.zeros(1, x.size(0), hidden_dim).to(x.device)
            for hidden_dim in self.hidden_dims
        ]
        c0 = [
            torch.zeros(1, x.size(0), hidden_dim).to(x.device)
            for hidden_dim in self.hidden_dims
        ]

        # Forward pass through each layer
        for i, layer in enumerate(self.layers):
            if isinstance(layer, nn.LSTM):
                ###THe //2 is to account for the dropout layer following each layer.
                x, (h0[i // 2], c0[i // 2]) = layer(x, (h0[i // 2], c0[i // 2]))

        # Applying the fully connected layer to the output of the last LSTM layer
        out = self.fc(x[:, -1, :])  # Taking the output of the last time step
        # return out
        return torch.sigmoid(out)  # Apply sigmoid to ensure output is in [0, 1]


def feature_importance(model, X_val, y_val, sequence_length):
    model.eval()

    # Create sequences for validation data
    X_val_seq, y_val_seq = create_sequences(X_val, sequence_length, y_val)

    with torch.no_grad():
        # Get baseline predictions
        baseline_output = model(X_val_seq)
        # Convert outputs to probabilities
        baseline_prob = torch.sigmoid(baseline_output).cpu().numpy()
        # Flatten if necessary and calculate the baseline metric
        baseline_metric = f1_score(
            y_val_seq.cpu().numpy(), (baseline_prob > 0.5).flatten()
        )

    importances = {}
    for i, col in enumerate(Chosen_Predictor):
        temp_val_seq = X_val_seq.clone()

        # Shuffle the feature across all sequences
        for seq in temp_val_seq:
            seq[:, i] = seq[torch.randperm(seq.size(0)), i]

        with torch.no_grad():
            # Get predictions for shuffled data
            shuff_output = model(temp_val_seq)
            shuff_prob = torch.sigmoid(shuff_output).cpu().numpy()
            # Calculate metric for shuffled data
            shuff_metric = f1_score(
                y_val_seq.cpu().numpy(), (shuff_prob > 0.5).flatten()
            )

        # Calculate drop in metric
        drop_in_metric = baseline_metric - shuff_metric
        importances[col] = drop_in_metric

    return importances


f1 = torchmetrics.F1Score(num_classes=2, average="weighted", task="binary").to(device)
accuracy = Accuracy(num_classes=2, average="weighted", task="binary").to(device)
prec = Precision(num_classes=2, average="weighted", task="binary").to(device)
recall = Recall(num_classes=2, average="weighted", task="binary").to(device)
micro_f1 = torchmetrics.F1Score(num_classes=2, average="micro", task="binary").to(
    device
)
micro_accuracy = Accuracy(num_classes=2, average="micro", task="binary").to(device)
micro_prec = Precision(num_classes=2, average="micro", task="binary").to(device)
micro_recall = Recall(num_classes=2, average="micro", task="binary").to(device)


def train_model(hparams, X_train, y_train, X_val, y_val):
    positivecase_weight_up = hparams["positivecase_weight_up"]
    sequence_length = hparams["sequence_length"]
    batch_size = hparams["batch_size"]
    best_model_state_dict = None
    X_train_seq, y_train_seq = create_sequences(X_train, sequence_length, y_train)

    X_val_seq, y_val_seq = create_sequences(X_val, sequence_length, y_val)

    train_dataset = TensorDataset(X_train_seq, y_train_seq)
    val_dataset = TensorDataset(X_val_seq, y_val_seq)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(hparams)
    hidden_dims = []
    pattern = re.compile(r"dims_layer_(\d+)")

    # Find all matching keys and their corresponding 'i' values to get the hidden dims for each layer
    i_values = []
    for key in hparams:
        # print(key)
        match = pattern.match(key)
        if match:
            i_values.append(int(match.group(1)))

    # Ensure the i_values list is unique and sorted
    i_values = sorted(set(i_values))

    # Now use these i_values to build hidden_dims
    for i in i_values:
        hidden_dims.append(hparams[f"dims_layer_{i}"])
    model = LSTMModel(
        X_train_seq.shape[2], hidden_dims, 1, hparams.get("dropout_rate", 0)
    ).to(device)

    model.train()
    weight_positive_up = (
        num_negative_up_train / num_positive_up_train
    ) * positivecase_weight_up
    weight = torch.Tensor([weight_positive_up]).to(device)
    # criterion = nn.BCEWithLogitsLoss(pos_weight=weight) have to undo signgmoid out i think.
    criterion = nn.BCELoss()
    optimizer_name = hparams.get("optimizer", "Adam")
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
    # Changed avg. from weighted

    # Gradient accumulation settings
    accumulation_steps = 4  # Define how many steps to accumulate gradients
    optimizer.zero_grad()  # Initialize gradient to zero
    best_f1_score = 0.0  # Track the best F1 score
    best_prec_score = 0.0  # Track the best F1 score
    sum_f1_score = 0.0
    sum_prec_score = 0.0
    sum_recall_score = 0.0  # Initialize sum of recall scores
    sum_micro_f1_score = 0.0
    sum_micro_prec_score = 0.0
    sum_micro_recall_score = 0.0

    epochs_sum = 0
    best_epoch = 0  # Initialize variable to save the best epoch

    best_val_loss = float("inf")  # Initialize best validation loss
    patience = 5  # Early stopping patience; how many epochs to wait was 20 changed to 5
    counter = 0  # Initialize counter for early stopping
    train_losses, val_losses = [], []

    for epoch in range(100000):
        print(epoch)
        f1.reset()
        prec.reset()
        recall.reset()
        micro_f1.reset()
        micro_accuracy.reset()
        micro_prec.reset()
        micro_recall.reset()
        # Training step
        model.train()

        for step, (X_batch, y_batch) in enumerate(train_loader):
            # Forward pass
            # if X_batch.shape[0] <= 1:
            #     continue
            train_output = model(X_batch)
            train_output = train_output.squeeze(1)
            train_loss = criterion(train_output, y_batch)
            train_loss = train_loss / accumulation_steps  # Normalize the loss
            train_loss.backward()  # Accumulate gradients

            # Perform optimization step every 'accumulation_steps'
            if (step + 1) % accumulation_steps == 0 or step + 1 == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
                train_losses.append(train_loss.item())

        model.eval()
        val_loss = 0.0  # Initialize validation loss

        val_pred_output_epoch = []
        val_losses_epoch = []
        val_targets_epoch = []
        f1.reset()
        prec.reset()
        recall.reset()

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                val_outputs = model(X_batch)
                val_outputs = val_outputs.squeeze(1)

                val_batch_loss = criterion(val_outputs, y_batch)
                val_losses_epoch.append(val_batch_loss.item())
                val_loss += val_batch_loss.item()  # Accumulate batch loss
                val_probs = torch.sigmoid(val_outputs)
                val_pred_output_epoch.append(val_probs)
                val_targets_epoch.append(y_batch)
            # # print(val_outputs.max,val_outputs.min)
            # print("Min:", val_outputs.min().item())
            # print("Max:", val_outputs.max().item())

        # val_loss /= len(val_loader)  # Calculate average batch loss

        val_outputs_epoch = torch.cat(val_pred_output_epoch, dim=0)
        val_targets_epoch = torch.cat(val_targets_epoch, dim=0)

        val_losses_epoch = torch.tensor(val_losses_epoch)

        val_predictions_all = (val_outputs_epoch > theshhold_down).float()
        F1Score = f1(val_predictions_all, val_targets_epoch)
        PrecisionScore = prec(val_predictions_all, val_targets_epoch)
        RecallScore = recall(val_predictions_all, val_targets_epoch)
        micro_F1Score = micro_f1(val_predictions_all, val_targets_epoch)
        micro_PrecisionScore = micro_prec(val_predictions_all, val_targets_epoch)
        micro_RecallScore = micro_recall(val_predictions_all, val_targets_epoch)
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
            best_prec_score = PrecisionScore.item()

        sum_f1_score += F1Score.item()
        sum_prec_score += PrecisionScore.item()
        sum_recall_score += RecallScore.item()  # Add to sum of recall scores
        sum_micro_f1_score += micro_F1Score.item()
        sum_micro_prec_score += micro_PrecisionScore.item()
        sum_micro_recall_score += (
            micro_RecallScore.item()
        )  # Add to sum of recall scores
        epochs_sum += 1

        # Calculate epoch validation loss
        # epoch_val_loss = sum(val_losses_epoch)

        # Early stopping
        if counter >= patience:
            print(f"Early stopping triggered at epoch {epoch}.")
            model.load_state_dict(best_model_state_dict)  # Load the best model
            break
        # print( f"VALIDATION Epoch: {epoch + 1}, PrecisionScore: {PrecisionScore.item()}, Training Loss: {loss.item()}, Validation Loss: {val_loss.item()}, F1 Score: {F1Score.item()} ")
    # Calculate average scores
    avg_val_micro_f1_score = sum_micro_f1_score / epochs_sum
    avg_val_micro_precision_score = sum_micro_prec_score / epochs_sum
    avg_val_micro_recall_score = (
        sum_micro_recall_score / epochs_sum
    )  # Calculate average recall score
    avg_val_f1_score = sum_f1_score / epochs_sum
    avg_val_precision_score = sum_prec_score / epochs_sum
    avg_val_recall_score = (
        sum_recall_score / epochs_sum
    )  # Calculate average recall score
    f1.reset()
    prec.reset()
    recall.reset()
    micro_f1.reset()
    micro_accuracy.reset()
    micro_prec.reset()
    micro_recall.reset()

    X_test_seq, y_test_seq = create_sequences(
        X_test_tensor, sequence_length, y_up_test_tensor
    )
    test_loader = DataLoader(
        TensorDataset(X_test_seq, y_test_seq), batch_size=batch_size, shuffle=False
    )
    test_pred_output_epoch = []
    test_targets_epoch = []

    with torch.no_grad():
        model.eval()
        for X_batch, y_batch in test_loader:
            test_outputs = model(X_batch)
            test_outputs = test_outputs.squeeze(1)
            test_probs = torch.sigmoid(test_outputs)
            test_predictions = (test_probs > theshhold_down).float()

            test_pred_output_epoch.append(test_predictions)
            test_targets_epoch.append(y_batch)
    # TODO MAKE THE BATCH EQUAL TO THE DAILY MINUTES, PREDICT NEXT DAYS OPEN.

    # Concatenate all predictions and targets
    test_predictions_all = torch.cat(test_pred_output_epoch, dim=0)
    test_targets_all = torch.cat(test_targets_epoch, dim=0)
    print(
        "test_targets/test_preds",
        test_targets_all.sum().item(),
        test_predictions_all.sum().item(),
    )
    # Calculate test metrics
    testF1Score = f1(test_predictions_all, test_targets_all)
    testPrecisionScore = prec(test_predictions_all, test_targets_all)
    testRecallScore = recall(test_predictions_all, test_targets_all)
    test_micro_F1Score = micro_f1(test_predictions_all, test_targets_all)
    test_micro_PrecisionScore = micro_prec(test_predictions_all, test_targets_all)
    test_micro_RecallScore = micro_recall(test_predictions_all, test_targets_all)
    # Aggregate and print scores
    test_micro_F1Score_avg = test_micro_F1Score.item()
    test_micro_PrecisionScore_avg = test_micro_PrecisionScore.item()
    test_micro_RecallScore_avg = test_micro_RecallScore.item()
    testF1Score_avg = testF1Score.item()
    testPrecisionScore_avg = testPrecisionScore.item()
    testRecallScore_avg = testRecallScore.item()

    # print(test_outputs)

    print(
        "val avg prec/f1/recall:  ",
        avg_val_precision_score,
        avg_val_f1_score,
        avg_val_recall_score,
    )
    print(
        "micro val avg prec/f1/recall:  ",
        avg_val_micro_precision_score,
        avg_val_micro_f1_score,
        avg_val_micro_recall_score,
    )
    print(
        "Test avg Precision/F1/Recall:",
        test_micro_PrecisionScore_avg,
        test_micro_F1Score_avg,
        test_micro_RecallScore_avg,
    )

    print(
        "Test avg Precision/F1/Recall:",
        testPrecisionScore_avg,
        testF1Score_avg,
        testRecallScore_avg,
    )
    # TODO make return best not avg of epochs.. i am trying to frind the best last epoch vslues. maybe use best val loss?
    return (
        best_val_loss,
        avg_val_f1_score,
        avg_val_precision_score,
        best_model_state_dict,
        testF1Score_avg,
        testPrecisionScore_avg,
        best_epoch,
        train_losses,
        val_losses,
    )
    # Return the best F1 score after all epochs


def train_final_model(best_params, Xtrainval, ytrainval):
    print("TRAINING FINAL MODEL")
    sequence_length = best_params["sequence_length"]
    Xtrainval_seq, ytrainval_seq = create_sequences(
        Xtrainval, sequence_length, ytrainval
    )
    positivecase_weight_up = best_params["positivecase_weight_up"]
    weight_positive_up = (
        num_negative_up_trainval / num_positive_up_trainval
    ) * positivecase_weight_up
    batch_size = best_params["batch_size"]
    trainval_dataset = TensorDataset(Xtrainval_seq, ytrainval_seq)
    # hidden_dims = best_params['hidden_dims_list']
    hidden_dims = []
    pattern = re.compile(r"dims_layer_(\d+)")

    # Find all matching keys and their corresponding 'i' values
    i_values = []
    for key in best_params:
        # print(key)
        match = pattern.match(key)
        if match:
            i_values.append(int(match.group(1)))

    # Ensure the i_values list is unique and sorted
    i_values = sorted(set(i_values))

    # Now use these i_values to build hidden_dims
    for i in i_values:
        hidden_dims.append(best_params[f"dims_layer_{i}"])
    # Create data loaders
    train_loader = DataLoader(trainval_dataset, batch_size=batch_size, shuffle=True)
    model = LSTMModel(
        input_dim=Xtrainval_seq.shape[2],
        hidden_dims=hidden_dims,
        output_dim=1,
        dropout=best_params.get("dropout_rate", 0),
    ).to(device)

    model.train()
    weight = torch.Tensor([weight_positive_up]).to(device)
    # criterion = nn.BCEWithLogitsLoss(pos_weight=weight) sigmoid thing
    criterion = nn.BCELoss()
    optimizer_name = best_params.get("optimizer", "Adam")
    learning_rate = best_params["learning_rate"]

    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    elif optimizer_name == "Adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)

    num_epochs = best_params["num_epochs"]

    # Gradient accumulation settings
    accumulation_steps = 4  # Define your accumulation steps here
    for epoch in range(num_epochs):
        for step, (X_batch, y_batch) in enumerate(train_loader):
            # if X_batch.shape[0] <= 1:
            #     continue
            train_output = model(X_batch)
            train_output = train_output.squeeze(1)

            train_loss = criterion(train_output, y_batch)
            train_loss = train_loss / accumulation_steps  # Normalize the loss
            train_loss.backward()  # Accumulate gradients

            if (step + 1) % accumulation_steps == 0 or step + 1 == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

    best_model_state_dict = model.state_dict()
    return model, best_model_state_dict


# Best Params: {'learning_rate': 0.002973181466202932, 'num_epochs': 365, 'batch_size': 2500, 'optimizer': 'Adam', 'dropout_rate': 0.05, 'num_hidden_units': 2350}


# Define Optuna Objective
def objective(trial):
    # Define the hyperparameter search space
    # learning_rate = trial.suggest_float("learning_rate", .0005, 0.007, log=True)  # 0003034075497582067
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-1, log=True)
    # num_epochs = trial.suggest_int("num_epochs", 100, 1000)  # 3800 #230  291
    batch_size = trial.suggest_int("batch_size", 2, 4096, log=True)

    # Add more parameters as needed
    sequence_length = trial.suggest_int(
        "sequence_length", 5, cells_forward_to_check, step=5
    )
    # optimizer_name = trial.suggest_categorical("optimizer", ["Adam","SGD"])  # ,"RMSprop", "Adagrad" "SGD"
    optimizer_name = "Adam"
    n_layers = trial.suggest_int("n_layers", 1, 5)
    # n_layers = 1
    if n_layers != 1:
        dropout_rate = trial.suggest_float(
            "dropout_rate", 0, 0.2
        )  # 30311980533100547  16372372692286732
    else:
        dropout_rate = 0
    positivecase_weight_up = trial.suggest_float(
        "positivecase_weight_up", 1, 20
    )  # 1.2 gave me .57 precisoin #was 20 and 18 its a multiplier
    hparams = {
        "learning_rate": learning_rate,
        "optimizer": optimizer_name,  # Include optimizer name here
        # "num_epochs": num_epochs,why use epochs? im tuning it using ealry stopping and passing that to final.
        "batch_size": batch_size,
        "dropout_rate": dropout_rate,
        "sequence_length": sequence_length,
        "positivecase_weight_up": positivecase_weight_up,
        # Add more hyperparameters as needed
    }
    for layer in range(n_layers):
        hparams[f"dims_layer_{layer}"] = trial.suggest_categorical(
            f"dims_layer_{layer}", [32, 64, 128, 256, 512, 1024]
        )
    # Call the train_model function with the current hyperparameters
    (
        best_val_loss,
        f1_score,
        prec_score,
        best_model_state_dict,
        testF1Score,
        testPrecisionScore,
        best_epoch,
        train_losses,
        val_losses,
    ) = train_model(
        hparams, X_train_tensor, y_up_train_tensor, X_val_tensor, y_up_val_tensor
    )

    # Plot the learning curves TODO
    # plot_learning_curves(train_losses, val_losses)
    alpha = 0.5

    blended_score = (
        (alpha * (1 - prec_score))
        + ((1 - alpha) * (1 - f1_score))
        + (alpha * (1 - testPrecisionScore))
        + ((1 - alpha) * (1 - testF1Score))
    )

    # return best_val_loss
    return blended_score
    # return prec_score  # Optuna will try to maximize this value


##Comment out to skip the hyperparameter selection.  Swap "best_params".
try:
    study = optuna.load_study(
        study_name=f"{study_name}", storage=f"sqlite:///{study_name}.db"
    )
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
    study = optuna.create_study(
        direction="minimize",
        study_name=f"{study_name}",
        storage=f"sqlite:///{study_name}.db",
    )
"Keyerror, new optuna study created."  #

# TODO changed trials from 100
study.optimize(
    objective, n_trials=n_trials
)  # You can change the number of trials as needed
best_params = study.best_params
# best_params ={'learning_rate': 0.00001, 'optimizer': 'Adam', 'batch_size': 3, 'dropout_rate': 0, 'sequence_length': 180, 'positivecase_weight_up': 1, 'dims_layer_0': 1024,'dims_layer_1': 1024,'dims_layer_2': 1024}
print("Best Hyperparameters:", best_params)

## Train the model with the best hyperparameters
torch.cuda.empty_cache()
(
    best_val_loss,
    best_f1_score,
    best_prec_score,
    best_model_state_dict,
    testF1Score,
    testPrecisionScore,
    best_epoch,
    train_losses,
    val_losses,
) = train_model(
    best_params, X_train_tensor, y_up_train_tensor, X_val_tensor, y_up_val_tensor
)
best_params["num_epochs"] = best_epoch
torch.cuda.empty_cache()

final_model, best_model_state_dict = train_final_model(
    best_params, X_trainval_tensor, y_trainval_tensor
)
# model = LSTMModel(input_dim=X_train_tensor.shape[2], hidden_dims=hidden_dims, output_dim=1, dropout=dropout_rate).to(
#     device)
torch.cuda.empty_cache()

# Load the saved state_dict into the model
final_model.eval()
feature_imp = feature_importance(
    final_model,
    X_trainval_tensor,
    y_trainval_tensor,
    sequence_length=best_params["sequence_length"],
)
print("Feature Importances:", feature_imp)
X_test_seq, y_up_test_seq = create_sequences(
    X_test_tensor, best_params["sequence_length"], y_up_test_tensor
)

with torch.no_grad():
    model_output = final_model(X_test_seq)
    predicted_probabilities_up = torch.sigmoid(model_output).cpu().numpy().flatten()
predicted_probabilities_up = (predicted_probabilities_up > theshhold_down).astype(int)
predicted_up_tensor = (
    torch.tensor(predicted_probabilities_up, dtype=torch.float32).squeeze().to(device)
)
predicted_binary_up = predicted_up_tensor > theshhold_down

num_positives_up = np.sum(predicted_probabilities_up)
task = "binary"

precision_up = prec(
    predicted_binary_up, y_up_test_seq
)  # move metric to same device as tensors
accuracy_up = accuracy(predicted_binary_up, y_up_test_seq)
recall_up = recall(predicted_binary_up, y_up_test_seq)
f1_up = f1(predicted_binary_up, y_up_test_seq)

precision_up_micro = micro_prec(
    predicted_binary_up, y_up_test_seq
)  # move metric to same device as tensors
accuracy_up_micro = micro_accuracy(predicted_binary_up, y_up_test_seq)
recall_up_micro = micro_recall(predicted_binary_up, y_up_test_seq)
f1_up_micro = micro_f1(predicted_binary_up, y_up_test_seq)
# Print Number of Positive and Negative Samples
num_positive_samples_up = sum(y_up_test_seq)
num_negative_samples_up = len(y_up_test_seq) - num_positive_samples_up

print("Metrics for Target_Up:", "\n")
print("Precision:", precision_up)
print("Accuracy:", accuracy_up)
print("Recall:", recall_up)
print("F1-Score:", f1_up, "\n")
print("Precision_micro:", precision_up_micro)
print("Accuracy_micro:", accuracy_up_micro)
print("Recall_micro:", recall_up_micro)
print("F1-Score_micro:", f1_up_micro, "\n")

print("Best Hyperparameters:", best_params)
print(
    f"Number of positive predictions for 'up': {sum(x.item() for x in predicted_binary_up)}"
)
print("Number of Positive Samples(Target_Up):", num_positive_samples_up)
print(
    "Number of Total Samples(Target_Up):",
    num_positive_samples_up + num_negative_samples_up,
)

# Save the models using joblib
input_val = input("Would you like to save these models? y/n: ").upper()
if input_val == "Y":
    model_summary = input("Save this set of models as: ")
    model_directory = os.path.join("../../../../../Trained_Models", f"{model_summary}")
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    model_filename_up = os.path.join(model_directory, "target_up.pth")

    torch.save(
        {
            "features": Chosen_Predictor,
            "input_dim": X_trainval_tensor.shape[2],
            "sequence_length": best_params["sequence_length"],
            "dropout_rate": best_params["dropout_rate"],
            "layers": best_params["layers"],
            "model_state_dict": final_model.state_dict(),
            "scaler_X": finalscaler_X,
        },
        model_filename_up,
    )
    # Save the scaler
    # TODO
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
    with open(
        "../../../../../Trained_Models/pytorch_trained_minute_models.py", "a"
    ) as file:
        file.write(function_def)
    with open(
        f"../../../../../Trained_Models/{model_summary}/info.txt", "w"
    ) as info_txt:
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
            f"anticondition_cells_threshold: {anticondition_threshold_cells}\n"
        )


# prediction workflow?
# # Preprocess new data (new_data_df)
# # Assume new_data_df is a DataFrame with the same structure as your training data
# new_data_df_processed = preprocess_data(new_data_df)  # Implement this function based on your preprocessing steps
#
# # Scale the features
# scaled_features = finalscaler_X.transform(new_data_df_processed[Chosen_Predictor].values)
#
# # Create sequences - adjust create_sequences function if necessary
# X_new, _ = create_sequences(scaled_features, None, sequence_length)
#
# # Load the model
# model = LSTMModel(...)  # use the same parameters as during training
# model.load_state_dict(torch.load(model_filename_up))  # Load the saved model
# model.eval()
#
# # Make predictions
# with torch.no_grad():
#     predictions = model(torch.tensor(X_new, dtype=torch.float32))
#     predictions = torch.sigmoid(predictions)
#     predicted_classes = (predictions > theshhold_down).int()
#
# # predicted_classes now contains the predicted labels for your new data


# TODO or

# # Reverse the DataFrame order
# reversed_df = ml_dataframe.iloc[::-1].reset_index(drop=True)
# # Create sequences from the reversed data
# X_sequences = create_sequences(X_test_tensor, sequence_length)
#
# # Now the first sequence in X_sequences corresponds to the most recent data
# # Use the first sequence for the latest data
# latest_sequence = X_sequences[0].unsqueeze(0)  # Add a batch dimension
#
# # Make prediction
# final_model.eval()
# with torch.no_grad():
#     latest_output = final_model(latest_sequence)
#     latest_prediction_prob = torch.sigmoid(latest_output).cpu().numpy().flatten()
