###for gpu acceleration
import os
from collections import Counter
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, cross_validate
import warnings
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.utils import compute_class_weight

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)
DF_filename = "../../../data/historical_multiday_minute_DF/SPY_historical_multiday_min.csv"
ml_dataframe = pd.read_csv(DF_filename)
"""TODO  as of scikit-learn version 0.23, you can now add feature names to your datasets, which will prevent this warning from occurring. Here is how you can do it:
# 
# If you're using a numpy array, you should first convert it to a pandas DataFrame. For example:
# 
# python
# Copy code
# import pandas as pd
# 
# # Assuming X is your feature matrix and feature_names is a list of your feature names
# X_df = pd.DataFrame(X, columns=feature_names)
# Now, when you call your RandomForestClassifier's fit method, you can use this DataFrame:
# 
# python
# Copy code
# clf = RandomForestClassifier()
# clf.fit(X_df, y)
# By doing this, your RandomForestClassifier will have the feature names and the warning should no longer occur.
"""

# Chosen_Predictor = ['Bonsai Ratio','Bonsai Ratio 2','B1/B2','B2/B1','ITM PCR-Vol','ITM PCR-OI','ITM PCRv Up2','ITM PCRv Down2','ITM PCRoi Up2','ITM PCRoi Down2','Net_IV','Net ITM IV','NIV 2Higher Strike','NIV 2Lower Strike','NIV highers(-)lowers1-4','NIV 1-4 % from mean','RSI','AwesomeOsc']
Chosen_Predictor = ['LastTradeTime', 'Current Stock Price',
    "Bonsai Ratio",
    "Bonsai Ratio 2",
    "B1/B2",
    "PCRv Up3", "PCRv Up2",
    "PCRv Down3", "PCRv Down2",
    "PCRv Up4",
    "PCRv Down4",
    "ITM PCRv Up3",
    "ITM PCRv Down3", "ITM PCRv Up4", "ITM PCRv Down2", "ITM PCRv Up2",
    "ITM PCRv Down4",
    "RSI14",
    "AwesomeOsc5_34",
    "RSI",
    "RSI2",
    "AwesomeOsc",
]
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
#                     'ITM PCRoi Down3', 'ITM PCRoi Down4', 'ITM OI', 'Total OI',
#                     'ITM Contracts %', 'Net_IV', 'Net ITM IV', 'Net IV MP', 'Net IV LAC',
#                     'NIV Current Strike', 'NIV 1Higher Strike', 'NIV 1Lower Strike',
#                     'NIV 2Higher Strike', 'NIV 2Lower Strike', 'NIV 3Higher Strike',
#                     'NIV 3Lower Strike', 'NIV 4Higher Strike', 'NIV 4Lower Strike',
#                     'NIV highers(-)lowers1-2', 'NIV highers(-)lowers1-4',
#                     'NIV 1-2 % from mean', 'NIV 1-4 % from mean', 'Net_IV/OI',
#                     'Net ITM_IV/ITM_OI', 'Closest Strike to CP', 'RSI', 'AwesomeOsc',
#                     'RSI14', 'RSI2', 'AwesomeOsc5_34']
##had highest corr for 3-5 hours with these:
# Chosen_Predictor = ['Bonsai Ratio','Bonsai Ratio 2','PCRoi Up1', 'B1/B2', 'PCRv Up4']
ml_dataframe['LastTradeTime'] = ml_dataframe['LastTradeTime'].apply(
    lambda x: datetime.strptime(str(x), '%y%m%d_%H%M') if not pd.isna(x) else np.nan)
ml_dataframe['LastTradeTime'] = ml_dataframe['LastTradeTime'].apply(lambda x: x.timestamp())
ml_dataframe['LastTradeTime'] = ml_dataframe['LastTradeTime'] / (60 * 60 * 24 * 7)
ml_dataframe['ExpDate'] = ml_dataframe['ExpDate'].astype(float)

##this many cells must meet the percentup/down requiremnet.
cells_forward_to_check = 180
threshold_cells_up = cells_forward_to_check * 0.7
threshold_cells_down = cells_forward_to_check * 0.7
# TODO add Beta to the percent, to make it more applicable across tickers.
percent_up = 0.5 / 100
percent_down = 0.5 / 100
###this many cells cannot be < current price for up, ># current price for down.
anticondition_threshold_cells_up = cells_forward_to_check * 0.6
anticondition_threshold_cells_down = cells_forward_to_check * 0.6
####multiplier for positive class weight.  It is already "balanced".  This should put more importance on the positive cases.
positivecase_weight_up_multiplier = 1  ###changed these from 20 7/12
positivecase_weight_down_multiplier = 1  # was 10 for both 9/26/23
num_features_up = 7
num_features_down = 7
##to binary threshhold.
threshold_up = 0.5
threshold_down = 0.5
parameters = {
    "max_depth": (10,20, 30,40),  # 50//70/65  100      up 65/3/1400  down 85/5/1300         71123 for 15 min  100/80
    # ###up 100/2/1300,down 80/3/1000
    "min_samples_split": (2, 5,10),  # 5//5/2     5                      71123                  for 15   2, 3,
    "n_estimators": (800,1500,)  # 1300//1600/1300/1400/1400  71123for 15 ,1000, 1300, ,
}
# 120 cells own: {'max_depth': 30, 'min_samples_split': 3, 'n_estimators': 900}Up: {'max_depth': 30, 'min_samples_split': 2, 'n_estimators': 800}
# 30cells - up80.4.900 down  80.2.1300
# 45 cells Target_Up: {'max_depth': 40, 'min_samples_split': 2, 'n_estimators': 900}Down: {'max_depth': 30, 'min_samples_split': 2, 'n_estimators': 800}
# 60cells up=60.2.800  down= 60.2.1250 up=30.2.1250  Down: 50: 2: 900}
##TODO make param_up/param_down.  up = 'max_depth': 40, 'min_samples_split': 7, 'n_estimators': 1000
# down=max_depth': 90, 'min_samples_split': 2, 'n_estimators': 1450
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
    shifted_values = ml_dataframe["Current Stock Price"].shift(-i)
    condition_met_up = shifted_values > (
            ml_dataframe["Current Stock Price"] + (ml_dataframe["Current Stock Price"] * percent_up))
    anticondition_up = shifted_values <= ml_dataframe["Current Stock Price"]

    condition_met_down = (
            ml_dataframe["Current Stock Price"].shift(-i) < (
            ml_dataframe["Current Stock Price"] - (ml_dataframe["Current Stock Price"] * percent_down))
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
            (targetDownCounter >= threshold_cells_down) & (
            anticondition_DownCounter <= anticondition_threshold_cells_down)
    ).astype(int)

ml_dataframe.dropna(subset=["Target_Up", "Target_Down"], inplace=True)
y_up = ml_dataframe["Target_Up"]
y_down = ml_dataframe["Target_Down"]
X = ml_dataframe[Chosen_Predictor].copy()
for column in X.columns:
    # Handle positive infinite values
    finite_max = X.loc[X[column] != np.inf, column].max()

    # Multiply by 1.5, considering the sign of the finite_max
    finite_max_adjusted = finite_max * 1.5 if finite_max > 0 else finite_max / 1.5

    X.loc[X[column] == np.inf, column] = finite_max_adjusted

    # Handle negative infinite values
    finite_min = X.loc[X[column] != -np.inf, column].min()

    # Multiply by 1.5, considering the sign of the finite_min
    finite_min_adjusted = finite_min * 1.5 if finite_min < 0 else finite_min / 1.5

    X.loc[X[column] == -np.inf, column] = finite_min_adjusted

# Reset the index of your DataFrame
X.reset_index(drop=True, inplace=True)






# TODO add this stuff in
# best_score = 0
# best_k = 0
#
# # Loop through different k values to find the best one
# for k in range(1, X.shape[1] + 1):
#     feature_selector_down = SelectKBest(score_func=mutual_info_classif, k=k)
#     X_kbest = feature_selector_down.fit_transform(X, y_down)
#
#     model_down = RandomForestClassifier()
#
#     # Use cross-validation to evaluate the model
#     cv_scores = cross_val_score(model_down, X_kbest, y_down, cv=5, scoring='f1')
#     mean_score = np.mean(cv_scores)
#
#     if mean_score > best_score:
#         best_score = mean_score
#         best_k = k
#
# print(f"Best number of features: {best_k}, Best Score: {best_score}")
# Use the first 3 splits for hyperparameter tuning
tscv = TimeSeriesSplit(n_splits=5)

#manual split
train_size = int(len(X) * 0.8)  # Assuming 80% train, 20% test split

X_train, X_test = X[:train_size], X[train_size:]
y_up_train, y_up_test = y_up[:train_size], y_up[train_size:]
y_down_train, y_down_test = y_down[:train_size], y_down[train_size:]
model_up = RandomForestClassifier()
print('Performing Gridsearch Up...')
# grid_search_up = GridSearchCV(estimator=model_up, param_grid=parameters,
#                               cv=TimeSeriesSplit(n_splits=2).split(X_train), scoring="f1")
# grid_search_up.fit(X_train, y_up_train)
# best_params_up = grid_search_up.best_params_
best_params_up =  {"max_depth": 100, "min_samples_split": 2000, "n_estimators": 3000  # 1300//1600/1300/1400/1400  71123for 15 ,1000, 1300, ,
}
# Perform GridSearchCV for 'down' model
model_down = RandomForestClassifier()
print('Performing Gridsearch Down...')

# grid_search_down = GridSearchCV(estimator=model_down, param_grid=parameters,
#                                 cv=TimeSeriesSplit(n_splits=2).split(X_train), scoring="f1")
# grid_search_down.fit(X_train, y_down_train)
# best_params_down = grid_search_down.best_params_
best_params_down =  {"max_depth": 100, "min_samples_split": 2000, "n_estimators": 3000  # 1300//1600/1300/1400/1400  71123for 15 ,1000, 1300, ,
                     }
print('best params up: ',best_params_up,'best params down: ',best_params_down)
# for train_index, val_index in tscv.split(X_train):
#     X_train_fold, X_val_fold = X_train.iloc[train_index], X_train.iloc[val_index]
#     y_up_train_fold, y_up_val_fold = y_up_train.iloc[train_index], y_up_train.iloc[val_index]
#     y_down_train_fold, y_down_val_fold = y_down_train.iloc[train_index], y_down_train.iloc[val_index]
# #TODO change test to val
#     # Select features for Target_Up
#     feature_selector_up = SelectKBest(score_func=mutual_info_classif, k=num_features_up)
#     # Select features for Target_Down
#     feature_selector_down = SelectKBest(score_func=mutual_info_classif, k=num_features_down)
#
#     X_train_selected_up = feature_selector_up.fit_transform(X_train_fold, y_up_train_fold)
#     X_test_selected_up = feature_selector_up.transform(X_val_fold)
#     X_train_selected_down = feature_selector_down.fit_transform(X_train_fold, y_down_train_fold)
#     X_test_selected_down = feature_selector_down.transform(X_val_fold)
#
#     best_features_up = [Chosen_Predictor[i] for i in feature_selector_up.get_support(indices=True)]
#     print("Best features for Target_Up:", best_features_up)
#     best_features_down = [Chosen_Predictor[i] for i in feature_selector_down.get_support(indices=True)]
#     print("Best features for Target_Down:", best_features_down)
#     # Class weights for Target_Up
#     num_positive_up = sum(y_up_train_fold)
#     num_negative_up = len(y_up_train_fold) - num_positive_up
#     weight_negative_up = 1.0
#     weight_positive_up = (num_negative_up / num_positive_up) * positivecase_weight_up_multiplier
#     custom_weights_up = {0: weight_negative_up, 1: weight_positive_up}
#     # Class weights for Target_Down
#     num_positive_down = sum(y_down_train_fold)
#     num_negative_down = len(y_down_train_fold) - num_positive_down
#     weight_negative_down = 1.0
#     weight_positive_down = (num_negative_down / num_positive_down) * positivecase_weight_down_multiplier
#     custom_weights_down = {0: weight_negative_down, 1: weight_positive_down}
#
#     print(
#         "num_positive_up= ",
#         num_positive_up,
#         "num_negative_up= ",
#         num_negative_up,
#         "num_positive_down= ",
#         num_positive_down,
#         "negative_down= ",
#         num_negative_down,
#     )
#     print("weight_positve_up: ", weight_positive_up, "//weight_negative_up: ", weight_negative_up)
#     print("weight_positve_down: ", weight_positive_down, "//weight_negative_down: ", weight_negative_down)
#
#     # Feature selection for Target_Up
#
#     # Assign weight_negative to class 0 and weight_positive to class 1
#
#     # Train with best hyperparameters
#     model_up = RandomForestClassifier(**best_params_up, class_weight=custom_weights_up)
#     model_up.fit(X_train_selected_up, y_up_train_fold)  # Use the selected features for training
#
#     model_down = RandomForestClassifier(**best_params_down, class_weight=custom_weights_down)
#     model_down.fit(X_train_selected_down, y_down_train_fold)
#
#     predicted_probabilities_up = model_up.predict_proba(X_test_selected_up)
#     predicted_up = (predicted_probabilities_up[:, 1] > threshold_up).astype(int)
#
#     predicted_probabilities_down = model_down.predict_proba(X_test_selected_down)
#     predicted_down = (predicted_probabilities_down[:, 1] > threshold_down).astype(int)
#
#     importance_tuples = [
#         (feature, importance) for feature, importance in zip(Chosen_Predictor, model_up.feature_importances_)
#     ]
#     importance_tuples = sorted(importance_tuples, key=lambda x: x[1], reverse=True)
#     for feature, importance in importance_tuples:
#         print(f"model up {feature}: {importance}")
#
#     importance_tuples = [
#         (feature, importance) for feature, importance in zip(Chosen_Predictor, model_down.feature_importances_)
#     ]
#     importance_tuples = sorted(importance_tuples, key=lambda x: x[1], reverse=True)
#
#     for feature, importance in importance_tuples:
#         print(f"Modle down {feature}: {importance}")
#
#     precision_up = precision_score(y_up_val_fold, predicted_up)
#     accuracy_up = accuracy_score(y_up_val_fold, predicted_up)
#     recall_up = recall_score(y_up_val_fold, predicted_up)
#     f1_up = f1_score(y_up_val_fold, predicted_up)
#
#     precision_down = precision_score(y_down_val_fold, predicted_down)
#     accuracy_down = accuracy_score(y_down_val_fold, predicted_down)
#     recall_down = recall_score(y_down_val_fold, predicted_down)
#     f1_down = f1_score(y_down_val_fold, predicted_down)
#
#     # scoring = ['precision', 'f1']
#     # cv_results_up = cross_validate(model_up, X_train_selected_up, y_up_train_fold, cv=tscv, scoring=scoring)
#     # cvprecision_scores_up = cv_results_up['test_precision']
#     # cvf1_scores_up = cv_results_up['test_f1']
#     # cv_results_down = cross_validate(model_down, X_train_selected_down, y_down_train_fold, cv=tscv, scoring=scoring)
#     # cvprecision_scores_down = cv_results_down['test_precision']
#     # cvf1_scores_down = cv_results_down['test_f1']
#     # print(
#     #     f"cvprec_up: {cvprecision_scores_up}   cvf1_up: {cvf1_scores_up}   cvprec_down: {cvprecision_scores_down}   cvf1_down: {cvf1_scores_down}")
#     # cv_scores_up = cross_val_score(model_up, X_test_selected_up, y_up_val_fold, cv=tscv)
#     # cv_scores_down = cross_val_score(model_down, X_test_selected_down, y_down_val_fold, cv=tscv)
#     print("Metrics for Target_Up:", "\n")
#     print("Precision:", precision_up)
#     print("Accuracy:", accuracy_up)
#     print("Recall:", recall_up)
#     print("F1-Score:", f1_up, "\n")
#     # print("Cross-validation scores for Target_Up:", cv_scores_up)
#     # print("Mean cross-validation score for Target_Up:", cv_scores_up.mean(), "\n")
#
#     print("Metrics for Target_Down:", "\n")
#     print("Precision:", precision_down)
#     print("Accuracy:", accuracy_down)
#     print("Recall:", recall_down)
#     print("F1-Score:", f1_down, "\n")
#
#     # print("Cross-validation scores for Target_Down:", cv_scores_down)
#     # print("Mean cross-validation score for Target_Down:", cv_scores_down.mean(), "\n")
# # 1. Use the entire training set for feature selection
# X_train_selected_up = feature_selector_up.fit_transform(X_train, y_up_train)
# X_train_selected_down = feature_selector_down.fit_transform(X_train, y_down_train)
#
# X_test_selected_up = feature_selector_up.transform(X_test)
# X_test_selected_down = feature_selector_down.transform(X_test)

# 2. Train the model on the entire training set using the best features and hyperparameters
# Calculate balanced class weights for 'up' model
class_weights_up = compute_class_weight('balanced', classes=[0, 1], y=y_up_train)
custom_weights_up = {0: class_weights_up[0], 1: class_weights_up[1] * positivecase_weight_up_multiplier}

# Calculate balanced class weights for 'down' model
class_weights_down = compute_class_weight('balanced', classes=[0, 1], y=y_down_train)
custom_weights_down = {0: class_weights_down[0], 1: class_weights_down[1] * positivecase_weight_down_multiplier}

# Train the 'up' model with custom weights
final_model_up = RandomForestClassifier(**best_params_up, class_weight=custom_weights_up)
final_model_up.fit(X_train, y_up_train)
print('yupsum',y_up_test.sum())
print('ydownsum',y_down_test.sum())

# Train the 'down' model with custom weights
final_model_down = RandomForestClassifier(**best_params_down, class_weight=custom_weights_down)
final_model_down.fit(X_train, y_down_train)

# 3. Evaluate the final model on the test set
predicted_probabilities_up_final = final_model_up.predict_proba(X_test)
predicted_up_final = (predicted_probabilities_up_final[:, 1] > threshold_up).astype(int)

predicted_probabilities_down_final = final_model_down.predict_proba(X_test)
predicted_down_final = (predicted_probabilities_down_final[:, 1] > threshold_down).astype(int)
accuracy_up = accuracy_score(y_up_test, predicted_up_final)
precision_up = precision_score(y_up_test, predicted_up_final)
recall_up = recall_score(y_up_test, predicted_up_final)
f1_up = f1_score(y_up_test, predicted_up_final)
print('leny',len(y_up_test),'yuptestsum',y_up_test.sum(),'ydowntestsum',y_down_test.sum())
print('predictedupsum',predicted_up_final.sum(),'predicteddownsum',predicted_down_final.sum())
print("Metrics for Target_Up:")
print(f"Accuracy: {accuracy_up:.4f}")
print(f"Precision: {precision_up:.4f}")
print(f"Recall: {recall_up:.4f}")
print(f"F1 Score: {f1_up:.4f}")
print("\n")

# Evaluate and print metrics for the "down" model
accuracy_down = accuracy_score(y_down_test, predicted_down_final)
precision_down = precision_score(y_down_test, predicted_down_final)
recall_down = recall_score(y_down_test, predicted_down_final)
f1_down = f1_score(y_down_test, predicted_down_final)

print("Metrics for Target_Down:")
print(f"Accuracy: {accuracy_down:.4f}")
print(f"Precision: {precision_down:.4f}")
print(f"Recall: {recall_down:.4f}")
print(f"F1 Score: {f1_down:.4f}")
# This code will print the accuracy, precision, recall, and F1 score for both the "up" and "down"



# # You can then compute the performance metrics (like accuracy, F1 score, etc.) using `predicted_up_final` and `predicted_down_final`
# def preprocess_dataframe(df, chosen_predictor):
#     # Your preprocessing code here
#     return df
#
# def calculate_thresholds(cells_forward_to_check):
#     # Your threshold calculation code here
#     return threshold_cells_up, threshold_cells_down
#
# def set_target_columns(df, cells_forward_to_check, percent_up, percent_down):
#     # Your target column setting code here
#     return df
#
# def handle_infinite_values(X):
#     # Your infinite value handling code here
#     return X
#
# def perform_grid_search(X_train, y_up_train, y_down_train):
#     # Your GridSearch code here
#     return best_params_up, best_params_down
#
# def feature_selection_and_model_training(X_train, y_up_train, y_down_train, best_params_up, best_params_down):
#     # Your feature selection and model training code here
#     return model_up, model_down
#
# def evaluate_model(model, X_test, y_test, threshold):
#     # Your model evaluation code here
#     return accuracy, precision, recall, f1
#
# if __name__ == "__main__":
#     # Load your dataframe
#     ml_dataframe = ...
#
#     # Preprocess the dataframe
#     ml_dataframe = preprocess_dataframe(ml_dataframe, Chosen_Predictor)
#
#     # Calculate thresholds
#     threshold_cells_up, threshold_cells_down = calculate_thresholds(cells_forward_to_check)
#
#     # Set target columns
#     ml_dataframe = set_target_columns(ml_dataframe, cells_forward_to_check, percent_up, percent_down)
#
#     # Handle infinite values
#     X = handle_infinite_values(X)
#
#     # Perform grid search
#     best_params_up, best_params_down = perform_grid_search(X_train, y_up_train, y_down_train)
#
#     # Feature selection and model training
#     model_up, model_down = feature_selection_and_model_training(X_train, y_up_train, y_down_train, best_params_up, best_params_down)
#
#     # Evaluate the model
#     accuracy_up, precision_up, recall_up, f1_up = evaluate_model(model_up, X_test_selected_up, y_up_test, threshold_up)
#     accuracy_down, precision_down, recall_down, f1_down = evaluate_model(model_down, X_test_selected_down, y_down_test, threshold_down)
#
#     print("Metrics for Target_Up:", accuracy_up, precision_up, recall_up, f1_up)
#     print("Metrics for Target_Down:", accuracy_down, precision_down, recall_down, f1_down)

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
            f"File analyzed: {DF_filename}\nCells_Foward_to_check: {cells_forward_to_check}\n\nBest parameters for Target_Up: {best_params_up}. \n\nBest parameters for Target_Down: {best_params_down}. \n\n\nMetrics for Target_Up:\nPrecision: {precision_up}\nAccuracy: {accuracy_up}\nRecall: {recall_up}\nF1-Score: {f1_up}\nCross-validation scores for Target_Up: \nMean cross-validation score for Target_Up: \n\nMetrics for Target_Down:\nPrecision: {precision_down}\nAccuracy: {accuracy_down}\nRecall: {recall_down}\nF1-Score: {f1_down}\nCross-validation scores for Target_Down: \nMean cross-validation score for Target_Down: \n\n"
        )
        # info_txt.write(
        #     f"Predictors: {Chosen_Predictor}\n\n\n\nBest_Predictors_Selected Up: {best_features_up}\nBest_Predictors_Selected Down: {best_features_down}\n\nThreshold Up(sensitivity): {threshold_up}\nThreshold Down(sensitivity): {threshold_down}\nTarget Underlying Percentage Up: {percent_up}\nTarget Underlying Percentage Down: {percent_down}\n\nAnticondition Up: {anticondition_up}\nTAnticondition Down: {anticondition_down}\n\nWeight multiplier Up: {positivecase_weight_up_multiplier}\nWeight multiplier Down: {positivecase_weight_down_multiplier}\n"
        # )
else:
    exit()
