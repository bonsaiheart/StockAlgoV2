import joblib
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFECV
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, TimeSeriesSplit, cross_val_score
import os
import numpy as np
import sys
from skopt import BayesSearchCV, Optimizer

DF_filename = "../../../data/historical_multiday_minute_DF/SPY_historical_multiday_min.csv"
ml_dataframe = pd.read_csv(DF_filename)

# Chosen_Predictor = ['Bonsai Ratio','Bonsai Ratio 2','B1/B2','B2/B1','ITM PCR-Vol','ITM PCR-OI','ITM PCRv Up2','ITM PCRv Down2','ITM PCRoi Up2','ITM PCRoi Down2','Net_IV','Net ITM IV','NIV 2Higher Strike','NIV 2Lower Strike','NIV highers(-)lowers1-4','NIV 1-4 % from mean','RSI','AwesomeOsc']
Chosen_Predictor = [
    "Bonsai Ratio",
    "Bonsai Ratio 2",
    "B1/B2",
    "PCRv Up4",
    "PCRv Down4",
    "ITM PCRv Up4",
    "ITM PCRv Down4",
    "RSI14",
    "AwesomeOsc5_34",
    "RSI",
    "RSI2",
    "AwesomeOsc",
]
##had highest corr for 3-5 hours with these:
# Chosen_Predictor = ['Bonsai Ratio','Bonsai Ratio 2','PCRoi Up1', 'B1/B2', 'PCRv Up4']
cells_forward_to_check = 45
##this many cells must meet the percentup/down requiremnet.
threshold_cells_up = cells_forward_to_check * 0.6
threshold_cells_down = cells_forward_to_check * 0.6
#TODO add Beta to the percent, to make it more applicable across tickers.
percent_up = 0.5
percent_down = -0.5
###this many cells cannot be < current price for up, >
# current price for down.
anticondition_threshold_cells_up = cells_forward_to_check * 0.6
anticondition_threshold_cells_down = cells_forward_to_check * 0.6

####multiplier for positive class weight.  It is already "balanced".  This should put more importance on the positive cases.
positivecase_weight_up = 20

positivecase_weight_down = 20


# num_features_up = 3
# num_features_down = 3
##probablility threshhold.
threshold_up = 0.7
threshold_down = 0.7

###35,5,80   6/3/80


parameters = {
    "max_depth": (30,40,50,60,80, 100 ),  # 50//70/65  100      up 65/3/1400  down 85/5/1300         71123 for 15 min  100/80
    # ###up 100/2/1300,down 80/3/1000
    "min_samples_split": (2, 3, 4,6,),  # 5//5/2     5                      71123                  for 15   2, 3,
    "n_estimators": (800,900,1000 ,1250,1500 ),  # 1300//1600/1300/1400/1400  71123for 15 ,1000, 1300, ,
}
#30cells - up80.4.900 down  80.2.1300
#45 cells Target_Up: {'max_depth': 40, 'min_samples_split': 2, 'n_estimators': 900}Down: {'max_depth': 30, 'min_samples_split': 2, 'n_estimators': 800}
#60cells up=60.2.800  down= 60.2.1250 1 hr=Target_Up: {'max_depth': 30, 'min_samples_split': 2, 'n_estimators': 800}Target_Down: {'max_depth': 30, 'min_samples_split': 4, 'n_estimators': 800}
##TODO make param_up/param_down.  up = 'max_depth': 40, 'min_samples_split': 7, 'n_estimators': 1000
#down=max_depth': 90, 'min_samples_split': 2, 'n_estimators': 1450
####TODO REMEMBER I MADE LOTS OF CHANGES DEBUGGING 7/5/23
ml_dataframe.dropna(subset=Chosen_Predictor, inplace=True)
threshold_up_formatted = int(threshold_up * 10)
threshold_down_formatted = int(threshold_down * 10)
Chosen_Predictor_nobrackets = [
    x.replace("/", "").replace(",", "_").replace(" ", "_").replace("-", "") for x in Chosen_Predictor
]
Chosen_Predictor_formatted = "_".join(Chosen_Predictor_nobrackets)
length = ml_dataframe.shape[0]
print("Length of ml_dataframe:", length)

# Number of cells to check
ml_dataframe["Target_Down"] = 0  # Initialize "Target_Down" column with zeros
ml_dataframe["Target_Up"] = 0
targetUpCounter = 0
targetDownCounter = 0
anticondition_UpCounter = 0
anticondition_DownCounter = 0
for i in range(1, cells_forward_to_check + 1):
    shifted_values = ml_dataframe["Current SP % Change(LAC)"].shift(-i)
    condition_met_up = shifted_values > ml_dataframe["Current SP % Change(LAC)"] + percent_up
    anticondition_up = shifted_values <= ml_dataframe["Current SP % Change(LAC)"]

    condition_met_down = (
        ml_dataframe["Current SP % Change(LAC)"].shift(-i) < ml_dataframe["Current SP % Change(LAC)"] + percent_down
    )
    anticondition_down = shifted_values >= ml_dataframe["Current SP % Change(LAC)"]

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
# Reset the index of your DataFrame
X.reset_index(drop=True, inplace=True)

# ml_dataframe.to_csv("Current_ML_DF_FOR_TRAINING.csv")

# weight_negative = 1.0
# weight_positive = 5.0  # Assigning a higher weight to the positive class (you can adjust this value based on your needs)
X_train, X_test, y_up_train, y_up_test, y_down_train, y_down_test = train_test_split(
    X, y_up, y_down, test_size=0.2, random_state=None
)
feature_selector_up = SelectKBest(score_func=mutual_info_classif)
# Calculate class weights
num_positive_up = sum(y_up_train)  # Number of positive cases in the training set
num_negative_up = len(y_up_train) - num_positive_up  # Number of negative cases in the training set
weight_negative_up = 1.0
weight_positive_up = (num_negative_up / num_positive_up) * positivecase_weight_up

num_positive_down = sum(y_down_train)  # Number of positive cases in the training set
num_negative_down = len(y_down_train) - num_positive_down  # Number of negative cases in the training set
weight_negative_down = 1.0
weight_positive_down = (num_negative_down / num_positive_down) * positivecase_weight_down
print(
    "num_positive_up= ",
    num_positive_up,
    "num_negative_up= ",
    num_negative_up,
    "num_positive_down= ",
    num_positive_down,
    "negative_down= ",
    num_negative_down,
)
print("weight_positve_up: ", weight_positive_up, "//weight_negative_up: ", weight_negative_up)
print("weight_positve_down: ", weight_positive_down, "//weight_negative_down: ", weight_negative_down)
# Define custom class weights as a dictionary
custom_weights_up = {
    0: weight_negative_up,
    1: weight_positive_up,
}  # Assign weight_negative to class 0 and weight_positive to class 1
custom_weights_down = {
    0: weight_negative_down,
    1: weight_positive_down,
}  # Assign weight_negative to class 0 and weight_positive to class 1
# Create RandomForestClassifier with custom class weights
model_up = RandomForestClassifier(class_weight=custom_weights_up)
model_down = RandomForestClassifier(class_weight=custom_weights_down)
###25/50      ###2/20   ###100/40
# model = RandomForestClassifier(random_state=None, class_weight="balanced")
X_train_selected_up = feature_selector_up.fit_transform(X_train, y_up_train)
X_test_selected_up = feature_selector_up.transform(X_test)
# Feature selection for Target_Down
feature_selector_down = SelectKBest(score_func=mutual_info_classif)
X_train_selected_down = feature_selector_down.fit_transform(X_train, y_down_train)
X_test_selected_down = feature_selector_down.transform(X_test)
print("Shape of X_test_selected_up:", X_test_selected_up.shape)
print("Shape of X_test_selected_down:", X_test_selected_down.shape, "\n")
print("Shape of X_train_selected_up:", X_train_selected_up.shape)
print("Shape of X_train_selected_down:", X_train_selected_down.shape, "\n")
tscv = TimeSeriesSplit(n_splits=5)

# scoring = 'precision'
grid_search_up = GridSearchCV(estimator=model_up, param_grid=parameters, cv=tscv, scoring="precision")

print("Performing GridSearchCV UP...\n")
grid_search_up.fit(X_train_selected_up, y_up_train)
best_features_up = [Chosen_Predictor[i] for i in feature_selector_up.get_support(indices=True)]
print("Best features for Target_Up:", best_features_up)

print("Best parameters for Target_Up:", grid_search_up.best_params_)
print("Best score for Target_Up:", grid_search_up.best_score_)
best_param_up = (
    f"Best parameters for Target_Up: {grid_search_up.best_params_}. Best precision: {grid_search_up.best_score_}"
)
model_up = grid_search_up.best_estimator_
X_train_selected_up[np.isinf(X_train_selected_up) & (X_train_selected_up > 0)] = sys.float_info.max
X_train_selected_up[np.isinf(X_train_selected_up) & (X_train_selected_up < 0)] = -sys.float_info.max
importance_tuples = [
    (feature, importance) for feature, importance in zip(Chosen_Predictor, model_up.feature_importances_)
]
importance_tuples = sorted(importance_tuples, key=lambda x: x[1], reverse=True)
for feature, importance in importance_tuples:
    print(f"model up {feature}: {importance}")

selected_features_up = feature_selector_up.get_support(indices=True)
feature_names_up = X_train.columns[selected_features_up]
print("Selected Features Up:", feature_names_up)

grid_search_down = GridSearchCV(estimator=model_down, param_grid=parameters, cv=tscv, scoring="precision")
print("Performing GridSearchCV DOWN...", "\n")
grid_search_down.fit(X_train_selected_down, y_down_train)
best_features_down = [Chosen_Predictor[i] for i in feature_selector_down.get_support(indices=True)]
print("Best features for Target_Down:", best_features_down)
selected_features_down = feature_selector_down.get_support(indices=True)

print("Best parameters for Target_Down:", grid_search_down.best_params_)
print("Best score for Target_Down:", grid_search_down.best_score_, "\n")
best_param_down = (
    f"Best parameters for Target_Down: {grid_search_down.best_params_}. Best precision: {grid_search_down.best_score_}"
)
model_down = grid_search_down.best_estimator_

importance_tuples = [
    (feature, importance) for feature, importance in zip(Chosen_Predictor, model_down.feature_importances_)
]
importance_tuples = sorted(importance_tuples, key=lambda x: x[1], reverse=True)

for feature, importance in importance_tuples:
    print(f"Modle down {feature}: {importance}")
selected_features_down = feature_selector_down.get_support(indices=True)
feature_names_down = X_train.columns[selected_features_down]
print("Selected Features Down:", feature_names_down)
# Use the selected features for prediction
predicted_probabilities_up = model_up.predict_proba(X_test_selected_up)
predicted_probabilities_down = model_down.predict_proba(X_test_selected_down)
print("Shape of predicted_probabilities_up:", predicted_probabilities_up.shape)

###predict
predicted_up = (predicted_probabilities_up[:, 1] > threshold_up).astype(int)
predicted_down = (predicted_probabilities_down[:, 1] > threshold_down).astype(int)

precision_up = precision_score(y_up_test, predicted_up)
accuracy_up = accuracy_score(y_up_test, predicted_up)
recall_up = recall_score(y_up_test, predicted_up)
f1_up = f1_score(y_up_test, predicted_up)

precision_down = precision_score(y_down_test, predicted_down)
accuracy_down = accuracy_score(y_down_test, predicted_down)
recall_down = recall_score(y_down_test, predicted_down)
f1_down = f1_score(y_down_test, predicted_down)

cv_scores_up = cross_val_score(model_up, X_test_selected_up, y_up_test, cv=tscv)
cv_scores_down = cross_val_score(model_down, X_test_selected_down, y_down_test, cv=tscv)
print("Metrics for Target_Up:", "\n")
print("Precision:", precision_up)
print("Accuracy:", accuracy_up)
print("Recall:", recall_up)
print("F1-Score:", f1_up, "\n")
print("Cross-validation scores for Target_Up:", cv_scores_up)
print("Mean cross-validation score for Target_Up:", cv_scores_up.mean(), "\n")

print("Metrics for Target_Down:", "\n")
print("Precision:", precision_down)
print("Accuracy:", accuracy_down)
print("Recall:", recall_down)
print("F1-Score:", f1_down, "\n")

print("Cross-validation scores for Target_Down:", cv_scores_down)
print("Mean cross-validation score for Target_Down:", cv_scores_down.mean(), "\n")


def save_file_with_shorter_name(data, file_path):
    try:
        # Attempt to save the file with the original name
        with open(file_path, "w") as file:
            file.write(data)
        print("File saved successfully:", file_path)
    except OSError as e:
        if e.errno == 63:  # File name too long error
            print("File name is too long. Please enter a shorter file name:")
            new_file_name = input()
            new_file_path = os.path.join(os.path.dirname(file_path), new_file_name)
            save_file_with_shorter_name(data, new_file_path)
        else:
            print("Error occurred while saving the file:", e)


input_val = input("Would you like to save these models? y/n: ").upper()
if input_val == "Y":
    model_summary = input("Save this set of models as: ")
    model_directory = os.path.join("../../Trained_Models", f"{model_summary}")
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
        print(model_directory)
    model_filename_up = os.path.join(model_directory, "target_up.joblib")
    model_filename_down = os.path.join(model_directory, "target_down.joblib")

    joblib.dump(model_up, model_filename_up)
    joblib.dump(model_down, model_filename_down)

    with open(f"../../Trained_Models/{model_summary}/info.txt", "w") as info_txt:
        info_txt.write("This file contains information about the model.\n\n")
        info_txt.write(
            f"File analyzed: {DF_filename}\nCells_Foward_to_check: {cells_forward_to_check}\n\nBest parameters for Target_Up: {grid_search_up.best_params_}. \nBest precision: {grid_search_down.best_score_}\nBest parameters for Target_Down: {grid_search_down.best_params_}. \nBest precision: {grid_search_down.best_score_}\n\nMetrics for Target_Up:\nPrecision: {precision_up}\nAccuracy: {accuracy_up}\nRecall: {recall_up}\nF1-Score: {f1_up}\nCross-validation scores for Target_Up: {cv_scores_up}\nMean cross-validation score for Target_Up: {cv_scores_up.mean()}\n\nMetrics for Target_Down:\nPrecision: {precision_down}\nAccuracy: {accuracy_down}\nRecall: {recall_down}\nF1-Score: {f1_down}\nCross-validation scores for Target_Down: {cv_scores_down}\nMean cross-validation score for Target_Down: {cv_scores_down.mean()}\n\n"
        )
        info_txt.write(
            f"Predictors: {Chosen_Predictor}\n\nSelected Features Up:, {selected_features_up}\nSelected Features Down:, {selected_features_down}\n\nBest_Predictors_Selected Up: {best_features_up}\nBest_Predictors_Selected Down: {best_features_down}\n\nThreshold Up(sensitivity): {threshold_up}\nThreshold Down(sensitivity): {threshold_down}\nTarget Underlying Percentage Up: {percent_up}\nTarget Underlying Percentage Down: {percent_down}\n"
        )
else:
    exit()
