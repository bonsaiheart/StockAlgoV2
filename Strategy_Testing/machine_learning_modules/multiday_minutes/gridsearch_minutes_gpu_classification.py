import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, cross_val_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

from tensorflow.keras.metrics import Precision

import numpy as np
import joblib
import os

DF_filename = r"../../../data/historical_multiday_minute_DF/Copy of SPY_historical_multiday_min.csv"
#TODO add early stop or no?
# from tensorflow.keras.callbacks import EarlyStopping
#
# early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
#
# model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stopping])

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
percent_up = .01  #.01 = 1%
percent_down = .01
anticondition_threshold_cells_up = cells_forward_to_check * 1  #was .7
anticondition_threshold_cells_down = cells_forward_to_check * 1
positivecase_weight_up = 20  #was 20 and 18
positivecase_weight_down = 20

# num_features_up = '8'
# num_features_down = '8'
threshold_up = 0.9
threshold_down = 0.9

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

X_train, X_test, y_up_train, y_up_test, y_down_train, y_down_test = train_test_split(
    X, y_up, y_down, test_size=0.2, random_state=None
)

num_positive_up = sum(y_up_train)
num_negative_up = len(y_up_train) - num_positive_up
weight_negative_up = 1.0
weight_positive_up = (num_negative_up / num_positive_up) * positivecase_weight_up

num_positive_down = sum(y_down_train)
print('num_positive_down:', num_positive_down)
print('num_positive_up:', num_positive_up)
num_negative_down = len(y_down_train) - num_positive_down
weight_negative_down = 1.0
weight_positive_down = (num_negative_down / num_positive_down) * positivecase_weight_down


custom_weights_up = {0: weight_negative_up, 1: weight_positive_up}
custom_weights_down = {0: weight_negative_down, 1: weight_positive_down}



# Define your neural network model using TensorFlow
model_up_nn = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])
model_down_nn = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the models with binary cross-entropy loss for binary classification
model_up_nn.compile(optimizer=Adam(), loss='binary_crossentropy',               metrics=[Precision()])

model_down_nn.compile(optimizer=Adam(), loss='binary_crossentropy',               metrics=[Precision()])


# Train the models on your data
model_up_nn.fit(X_train, y_up_train, epochs=50, batch_size=32)
model_down_nn.fit(X_train, y_down_train, epochs=50, batch_size=32)

# Use the trained models for prediction
predicted_probabilities_up = model_up_nn.predict(X_test)
predicted_probabilities_down = model_down_nn.predict(X_test)

threshold_up_formatted = int(threshold_up * 10)
threshold_down_formatted = int(threshold_down * 10)

predicted_up = (predicted_probabilities_up[:, 0] > threshold_up_formatted / 10).astype(int)
predicted_down = (predicted_probabilities_down[:, 0] > threshold_down_formatted / 10).astype(int)

precision_up = precision_score(y_up_test, predicted_up)
accuracy_up = accuracy_score(y_up_test, predicted_up)
recall_up = recall_score(y_up_test, predicted_up)
f1_up = f1_score(y_up_test, predicted_up)

precision_down = precision_score(y_down_test, predicted_down)
accuracy_down = accuracy_score(y_down_test, predicted_down)
recall_down = recall_score(y_down_test, predicted_down)
f1_down = f1_score(y_down_test, predicted_down)


# ... (Previous code)

# Print Selected Features


# Print Number of Positive and Negative Samples
num_positive_samples_up = sum(y_up_train)
num_negative_samples_up = len(y_up_train) - num_positive_samples_up
print("Number of Positive Samples (Target_Up):", num_positive_samples_up)
print("Number of Negative Samples (Target_Up):", num_negative_samples_up)

num_positive_samples_down = sum(y_down_train)
num_negative_samples_down = len(y_down_train) - num_positive_samples_down
print("Number of Positive Samples (Target_Down):", num_positive_samples_down)
print("Number of Negative Samples (Target_Down):", num_negative_samples_down)

# Print Feature Importance


# ... (Rest of the code)

# Write Additional Information to info.txt


# ... (Rest of the code)

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
# Assuming `model` is your trained Keras model and `X_test` and `y_test` are your testing data and labels
loss_down, metric_down = model_down_nn.evaluate(X_test, y_down_test, verbose=0)
# Assuming `model` is your trained Keras model and `X_test` and `y_test` are your testing data and labels
loss_up, metric_up = model_up_nn.evaluate(X_test, y_up_test, verbose=0)
print(f"Loss up: {loss_up}, Loss down: {loss_down}, metric up: {metric_up}, metric down: {metric_down}")

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
        f"Weight multiplier Up: {positivecase_weight_up}\nWeight multiplier Down: {positivecase_weight_down}\n"
    )