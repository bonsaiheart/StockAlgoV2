import json
from datetime import datetime
import optuna
import joblib
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFECV
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, TimeSeriesSplit, cross_val_score
import os
import numpy as np
import sys
from skopt import BayesSearchCV, Optimizer

DF_filename = r"../../../../data/historical_multiday_minute_DF/SPY_historical_multiday_min.csv"
ml_dataframe = pd.read_csv(DF_filename)

# Chosen_Predictor = [ 'ITM PCRv Up2','ITM PCRv Down2','ITM PCRoi Up2','ITM PCRoi Down2','Net_IV','Net ITM IV','NIV 2Higher Strike','NIV 2Lower Strike','NIV highers(-)lowers1-4','NIV 1-4 % from mean','RSI','AwesomeOsc']
Chosen_Predictor =  [
   'LastTradeTime','Bonsai Ratio','Bonsai Ratio 2', 'B1/B2', 'PCRv Up4', 'PCRv Down4','ITM PCRv Up2','ITM PCRv Down2',
    'ITM PCRv Up4','ITM PCRv Down4',
      'RSI14','AwesomeOsc5_34','RSI','RSI2',
                     'AwesomeOsc'
      ]

##had highest corr for 3-5 hours with these:
# Chosen_Predictor = ['Bonsai Ratio','Bonsai Ratio 2','PCRoi Up1', 'B1/B2', 'PCRv Up4']
ml_dataframe['LastTradeTime'] = ml_dataframe['LastTradeTime'].apply(
    lambda x: datetime.strptime(str(x), '%y%m%d_%H%M') if not pd.isna(x) else np.nan)
ml_dataframe['LastTradeTime'] = ml_dataframe['LastTradeTime'].apply(lambda x: x.timestamp())
ml_dataframe['LastTradeTime'] = ml_dataframe['LastTradeTime'] / (60 * 60 * 24 * 7)

ml_dataframe['ExpDate'] = ml_dataframe['ExpDate'].astype(float)
study_name = '30minutes'
cells_forward_to_check = 30
threshold_cells_up = cells_forward_to_check * .7
threshold_cells_down = cells_forward_to_check * .7
# num_features_up = 3
# num_features_down = 3
threshold_up = 0.5
threshold_down = 0.5
percent_up = .2
percent_down = -.2
num_features_to_select=8
####TODO REMEMBER I MADE LOTS OF CHANGES DEBUGGING 7/5/23
ml_dataframe.dropna(subset= Chosen_Predictor, inplace=True)
threshold_up_formatted = int(threshold_up * 10)
threshold_down_formatted = int(threshold_down * 10)
Chosen_Predictor_nobrackets = [x.replace('/', '').replace(',', '_').replace(' ', '_').replace('-', '') for x in
                               Chosen_Predictor]
Chosen_Predictor_formatted = "_".join(Chosen_Predictor_nobrackets)
length = ml_dataframe.shape[0]
print("Length of ml_dataframe:", length)

 # Number of cells to check
ml_dataframe["Target_Down"] = 0  # Initialize "Target_Down" column with zeros
ml_dataframe["Target_Up"] = 0
targetUpCounter =0
targetDownCounter=0
for i in range(1, cells_forward_to_check+1):
    shifted_values = ml_dataframe["Current Stock Price"].shift(-i)
    condition_met_up = shifted_values > (
            ml_dataframe["Current Stock Price"] + (ml_dataframe["Current Stock Price"] * (percent_up / 100)))
    condition_met_down = shifted_values < (
            ml_dataframe["Current Stock Price"] + (ml_dataframe["Current Stock Price"] * (percent_down / 100)))
    targetUpCounter += condition_met_up.astype(int)
    targetDownCounter += condition_met_down.astype(int)
    ml_dataframe["Target_Down"] = (targetDownCounter >= threshold_cells_down).astype(int)
    ml_dataframe["Target_Up"] = (targetUpCounter >= threshold_cells_up).astype(int)

ml_dataframe.dropna(subset= ['Target_Up','Target_Down'], inplace=True)
y_up = ml_dataframe["Target_Up"].copy()
y_down = ml_dataframe["Target_Down"].copy()
X = ml_dataframe[Chosen_Predictor].copy()
# Reset the index of your DataFrame
X.reset_index(drop=True, inplace=True)



# # Get column names
# col_names = X.columns

# Print indices along with column names
# if len(nan_indices) > 0:
#     print("NaN values found at indices:")
#     for i, j in nan_indices:
#         print(f"Row: {i}, Column: {col_names[j]}")
# else:
#     print("No NaN values found.")
#
# if len(inf_indices) > 0:
#     print("Infinite values found at indices:")
#     for i, j in inf_indices:
#         print(f"Row: {i}, Column: {col_names[j]}")
# else:
#     print("No infinite values found.")
#
# if len(neginf_indices) > 0:
#     print("Negative Infinite values found at indices:")
#     for i, j in neginf_indices:
#         print(f"Row: {i}, Column: {col_names[j]}")
# else:
#     print("No negative infinite values found.")

# weight_negative = 1.0
# weight_positive = 5.0  # Assigning a higher weight to the positive class (you can adjust this value based on your needs)
X_train, X_test, y_up_train, y_up_test, y_down_train, y_down_test = (train_test_split
                                                                     (X, y_up, y_down, test_size=0.2, random_state=None))
print('y_up_testsum ', y_up_test.sum(),'lenYtestup ',len(y_up_test))
print('y_down_testsum ', y_down_test.sum(),'lenYtestdown ',len(y_down_test))
# Handle inf and -inf values based on the training set
min_max_dict = {}

for col in X_train.columns:
    max_val = X_train[col].replace([np.inf, -np.inf], np.nan).max()
    min_val = X_train[col].replace([np.inf, -np.inf], np.nan).min()

    # Adjust max_val based on its sign
    max_val = max_val * 1.5 if max_val >= 0 else max_val / 1.5

    # Adjust min_val based on its sign
    min_val = min_val * 1.5 if min_val < 0 else min_val / 1.5
    print("min/max values ",min_val,max_val)
    # Apply the same max_val and min_val to training, validation, and test sets
    X_train[col].replace([np.inf, -np.inf], [max_val, min_val], inplace=True)
    X_test[col].replace([np.inf, -np.inf], [max_val, min_val], inplace=True)
    min_max_dict[col] = {'min_val': min_val, 'max_val': max_val}





num_positive_up = sum(y_up_train)
num_negative_up = len(y_up_train) - num_positive_up
weight_negative_up = 1.0
weight_positive_up = num_negative_up / num_positive_up
# Calculate class weights
num_positive_up = sum(y_up_train)  # Number of positive cases in the training set
num_negative_up = len(y_up_train) - num_positive_up  # Number of negative cases in the training set
weight_negative_up = 1.0
weight_positive_up = num_negative_up / num_positive_up

num_positive_down = sum(y_down_train)  # Number of positive cases in the training set
num_negative_down = len(y_down_train) - num_positive_down  # Number of negative cases in the training set
weight_negative_down = 1.0
weight_positive_down = num_negative_down / num_positive_down
print('num_positive_up= ',num_positive_up,'num_negative_up= ',num_negative_up,'num_positive_down= ',num_positive_down,'negative_down= ',num_negative_down)
print('weight_positve_up: ',weight_positive_up,'//weight_negative_up: ',weight_negative_up)
print('weight_positve_down: ',weight_positive_down,'//weight_negative_down: ',weight_negative_down)
# Define custom class weights as a dictionary
custom_weights_up = {0: weight_negative_up,1: weight_positive_up}  # Assign weight_negative to class 0 and weight_positive to class 1
custom_weights_down = {0: weight_negative_down, 1: weight_positive_down}  # Assign weight_negative to class 0 and weight_positive to class 1
# Create RandomForestClassifier with custom class weights
tscv = TimeSeriesSplit(n_splits=3)


def objective_up(trial, X_train_selected, y_train, X_test_selected, y_test):
    # Define hyperparameter search space
    max_depth = trial.suggest_int('max_depth',12, 15)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 6)
    n_estimators = trial.suggest_int('n_estimators', 450, 600)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 3)

    # Create and train model
    model_up = RandomForestClassifier(max_depth=max_depth, min_samples_split=min_samples_split, n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,class_weight=custom_weights_up)
    model_up.fit(X_train_selected, y_train)
    y_pred = model_up.predict(X_test_selected)
    y_prob = model_up.predict_proba(X_test_selected)[:, 1]  # Probability estimates of the positive class

    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    alpha=.5
    auc = roc_auc_score(y_test, y_prob)
    beta = 0.3  # You can adjust this weight as well
    print('Precision: ',precision,'F1 : ',f1,'AUC :',auc)
    # Blend the scores using alpha
    cv_prec_scores_up = cross_val_score(model_up, X_test_selected_up, y_up_test, cv=tscv, scoring='precision')
    cv_f1_scores_up = cross_val_score(model_up, X_test_selected_up, y_up_test, cv=tscv, scoring='f1')

    print("Cross-validation prec,f1 scores for Target_up:", cv_prec_scores_up,cv_f1_scores_up)
    print("Mean cross-validation score for Target_up:", cv_prec_scores_up.mean(),cv_f1_scores_up.mean(), '\n')
    # blended_score = alpha * (1 - precision) + (1 - alpha) * cv_scores_down.mean()
    blended_score = alpha * (0.5 * (1 - precision) + 0.5 * (1 - f1)) + (1 - alpha) * (
                0.5 * (1 - cv_prec_scores_up.mean()) + 0.5 * (1 - cv_f1_scores_up.mean()))

    return blended_score
    return blended_score


def objective_down(trial, X_train_selected, y_train, X_test_selected, y_test):
    max_depth = trial.suggest_int('max_depth',12, 15)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 6)
    n_estimators = trial.suggest_int('n_estimators', 450, 600)


    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 3)

    # Create and train model
    model_down = RandomForestClassifier(max_depth=max_depth, min_samples_split=min_samples_split,
                                        n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                        class_weight=custom_weights_down)
    model_down.fit(X_train_selected, y_train)
    y_pred = model_down.predict(X_test_selected)
    y_prob = model_down.predict_proba(X_test_selected)[:, 1]  # Probability estimates of the positive class

    precision = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    alpha = .5
    auc = roc_auc_score(y_test, y_prob)
    beta = 0.3  # You can adjust this weight as well
    print('Precision: ', precision, 'F1 : ', f1, 'AUC :', auc)
    # Blend the scores using alpha
    blended_score = alpha * (1 - precision) + (1 - alpha) * f1
    # blended_score = (f1 + precision + auc) / 3
    cv_prec_scores_down = cross_val_score(model_down, X_test_selected_down, y_down_test, cv=tscv, scoring='precision')
    cv_f1_scores_down = cross_val_score(model_down, X_test_selected_down, y_down_test, cv=tscv, scoring='f1')

    print("Cross-validation prec,f1 scores for Target_Down:", cv_prec_scores_down,cv_f1_scores_down)
    print("Mean cross-validation score for Target_Down:", cv_prec_scores_down.mean(),cv_f1_scores_down.mean(), '\n')
    # blended_score = alpha * (1 - precision) + (1 - alpha) * cv_scores_down.mean()
    blended_score = alpha * (0.5 * (1 - precision) + 0.5 * (1 - f1)) + (1 - alpha) * (
                0.5 * (1 - cv_prec_scores_down.mean()) + 0.5 * (1 - cv_f1_scores_down.mean()))

    return blended_score
# Feature selection and transformation
feature_selector_down = SelectKBest(score_func=mutual_info_classif,k=num_features_to_select)
feature_selector_up = SelectKBest(score_func=mutual_info_classif,k=num_features_to_select)
X_train_selected_up = feature_selector_up.fit_transform(X_train, y_up_train)
X_test_selected_up = feature_selector_up.transform(X_test)
X_train_selected_down = feature_selector_down.fit_transform(X_train, y_down_train)
X_test_selected_down = feature_selector_down.transform(X_test)
best_features_up = [Chosen_Predictor[i] for i in feature_selector_up.get_support(indices=True)]
selected_features_up = feature_selector_up.get_support(indices=True)
feature_names_up = X_train.columns[selected_features_up]
print("Selected Features Up:", feature_names_up)
print("Best features for Target_Up:", best_features_up)

best_features_down = [Chosen_Predictor[i] for i in feature_selector_down.get_support(indices=True)]
selected_features_down = feature_selector_down.get_support(indices=True)
feature_names_down = X_train.columns[selected_features_down]
print("Selected Features Down:", feature_names_down)
print("Best features for Target_Down:", best_features_down)
# Optuna optimization
try:


    study_up = optuna.load_study(study_name=f'{study_name}_up',
                              storage=f'sqlite:///{study_name}_up.db')
    print("Study Loaded.")
    try:
        best_params_up = study_up.best_params
        best_trial_up = study_up.best_trial
        best_value_up = study_up.best_value
        print("Best Value_up:", best_value_up)

        print(best_params_up)

        print("Best Trial_up:", best_trial_up)

    except Exception as e:
        print(e)
except KeyError:
    study_up = optuna.create_study(direction="minimize", study_name=f'{study_name}_up',
                                storage=f'sqlite:///{study_name}_up.db')
"Keyerror, new optuna study created."  #
study_up.optimize(lambda trial: objective_up(trial, X_train_selected_up, y_up_train, X_test_selected_up, y_up_test), n_trials=2)

# Results
best_params_up = study_up.best_params
best_score_up = study_up.best_value
print(f"Best parameters up: {best_params_up}")
print(f"Best score up: {best_score_up}")
try:


    study_down = optuna.load_study(study_name=f'{study_name}_down',
                              storage=f'sqlite:///{study_name}_down.db')

    print("Study Loaded.")
    try:
        best_params_down = study_down.best_params
        best_trial_down = study_down.best_trial
        best_value_down = study_down.best_value
        print("Best Value_down:", best_value_down)

        print(best_params_down)

        print("Best Trial_down:", best_trial_down)

    except Exception as e:
        print(e)
except KeyError:
    study_down = optuna.create_study(direction="minimize", study_name=f'{study_name}_down',
                                storage=f'sqlite:///{study_name}_down.db')
"Keyerror, new optuna study created."  #
study_down.optimize(lambda trial: objective_down(trial, X_train_selected_down, y_down_train, X_test_selected_down, y_down_test), n_trials=2)

best_params_down = study_down.best_params
best_score_down = study_down.best_value

print(f"Best parameters down: {best_params_down}")
print(f"Best score down: {best_score_down}")
model_up = RandomForestClassifier(**best_params_up, class_weight=custom_weights_up)
model_down = RandomForestClassifier(**best_params_down, class_weight=custom_weights_down)

# Fit the models on the selected features
model_up.fit(X_train_selected_up, y_up_train)
model_down.fit(X_train_selected_down, y_down_train)


print("Shape of X_test_selected_up:", X_test_selected_up.shape)
print("Shape of X_test_selected_down:", X_test_selected_down.shape,'\n')
print("Shape of X_train_selected_up:", X_train_selected_up.shape)
print("Shape of X_train_selected_down:", X_train_selected_down.shape,'\n')



X_train_selected_up[np.isinf(X_train_selected_up) & (X_train_selected_up > 0)] = sys.float_info.max
X_train_selected_up[np.isinf(X_train_selected_up) & (X_train_selected_up < 0)] = -sys.float_info.max
importance_tuples_up = [(feature, importance) for feature, importance in zip(Chosen_Predictor, model_up.feature_importances_)]
importance_tuples_up = sorted(importance_tuples_up, key=lambda x: x[1], reverse=True)
for feature, importance in importance_tuples_up:
    print(f"model up {feature}: {importance}")

importance_tuples_down = [(feature, importance) for feature, importance in zip(Chosen_Predictor, model_down.feature_importances_)]
importance_tuples_down = sorted(importance_tuples_down, key=lambda x: x[1], reverse=True)

for feature, importance in importance_tuples_down:
    print(f"Model down {feature}: {importance}")

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

cv_prec_score_up = cross_val_score(model_up, X_test_selected_up, y_up_test, cv=tscv, scoring='precision')
cv_prec_score_down = cross_val_score(model_down, X_test_selected_down, y_down_test, cv=tscv, scoring='precision')
cv_f1_score_up = cross_val_score(model_up, X_test_selected_up, y_up_test, cv=tscv, scoring='f1')
cv_f1_score_down = cross_val_score(model_down, X_test_selected_down, y_down_test, cv=tscv, scoring='f1')
print("Metrics for Target_Up:",'\n')
print("Precision:", precision_up)
print("Accuracy:", accuracy_up)
print("Recall:", recall_up)
print("F1-Score:", f1_up,'\n')
print("Cross-validation scores for Target_Up:(prec/f1)", cv_prec_score_up,cv_f1_score_up)
print("Mean cross-validation score for Target_Up:", cv_prec_score_up.mean(),cv_f1_score_up.mean(),'\n')

print("Metrics for Target_Down:",'\n')
print("Precision:", precision_down)
print("Accuracy:", accuracy_down)
print("Recall:", recall_down)
print("F1-Score:", f1_down,'\n')

print("Cross-validation scores for Target_Down:(prec/f1)", cv_prec_score_down,cv_f1_score_down)
print("Mean cross-validation score for Target_Down:(prec/f1)", cv_prec_score_down.mean(),cv_f1_score_down.mean(),'\n')



def save_file_with_shorter_name(data, file_path):
    try:
        # Attempt to save the file with the original name
        with open(file_path, 'w') as file:
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


input_val = input("Would you li"
                  "ke to save these models? y/n: ").upper()
if input_val == "Y":
    model_summary = input("Save this set of models as: ")
    model_directory = os.path.join("..","..", "..", "Trained_Models", model_summary)

    print("Current working directory:", os.getcwd())

    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
        print(f"Directory created at: {model_directory}")

    model_filename_up = os.path.join(model_directory, "target_up.joblib")
    model_filename_down = os.path.join(model_directory, "target_down.joblib")

    joblib.dump(model_up, model_filename_up)
    joblib.dump(model_down, model_filename_down)

    with open(
            f"../../../Trained_Models/{model_summary}/info.txt","w") as info_txt:
        info_txt.write("This file contains information about the model.\n\n")
        info_txt.write(
            f"File analyzed: {DF_filename}\nCells_Foward_to_check: {cells_forward_to_check}\n\nBest parameters for Target_Up: {best_params_down}. \nBest precision: {best_score_down}\nBest parameters for Target_Down: {best_params_down}. \nBest precision: {best_score_down}\n\nMetrics for Target_Up:\nPrecision: {precision_up}\nAccuracy: {accuracy_up}\nRecall: {recall_up}\nF1-Score: {f1_up}\nCross-validation scores for Target_Up:(prec/f1) {cv_prec_score_up,cv_f1_score_up}\nMean cross-validation score for Target_Up:(prec/f1) {cv_prec_score_up.mean(),cv_f1_score_up.mean()}\n\nMetrics for Target_Down:\nPrecision: {precision_down}\nAccuracy: {accuracy_down}\nRecall: {recall_down}\nF1-Score: {f1_down}\nCross-validation scores for Target_Down:(prec/f1) {cv_prec_score_down,cv_f1_score_down}\nMean cross-validation score for Target_Down:(prec/f1) {cv_prec_score_down.mean(),cv_f1_score_down.mean()}\n\n")
        info_txt.write(
            f"Min value//max value used for -inf/inf:{min_val, max_val}Predictors: {Chosen_Predictor}\n\nSelected Features Up:, {selected_features_up}\nSelected Features Down:, {selected_features_down}\n\nBest_Predictors_Selected Up: {best_features_up}\nBest_Predictors_Selected Down: {best_features_down}\n\nThreshold Up(sensitivity): {threshold_up}\nThreshold Down(sensitivity): {threshold_down}\nTarget Underlying Percentage Up: {percent_up}\nTarget Underlying Percentage Down: {percent_down}\n")
    with open( f"../../../Trained_Models/{model_summary}/min_max_values.json", 'w') as f:
        json.dump(min_max_dict, f)
else:

    exit()
