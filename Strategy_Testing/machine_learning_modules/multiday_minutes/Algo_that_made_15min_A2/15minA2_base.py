import json
import os
from datetime import datetime
import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score

DF_filename = r"../../../../data/historical_multiday_minute_DF/older/SPY_historical_multiday_minprior_231002.csv"
ml_dataframe = pd.read_csv(DF_filename)
# TODO train with thhe best parasm listed below, it had great metrics even on test.
# Chosen_Predictor = [ 'ITM PCRv Up2','ITM PCRv Down2','ITM PCRoi Up2','ITM PCRoi Down2','Net_IV','Net ITM IV','NIV 2Higher Strike','NIV 2Lower Strike','NIV highers(-)lowers1-4','NIV 1-4 % from mean','RSI','AwesomeOsc']
Chosen_Predictor = [
    'LastTradeTime', 'Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'PCRv Up4', 'PCRv Down4', 'ITM PCRv Up2',
    'ITM PCRv Down2',
    'ITM PCRv Up4', 'ITM PCRv Down4',
    'RSI14', 'AwesomeOsc5_34', 'RSI', 'RSI2',
    'AwesomeOsc'
]
##had highest corr for 3-5 hours with these:
# Chosen_Predictor = ['Bonsai Ratio','Bonsai Ratio 2','PCRoi Up1', 'B1/B2', 'PCRv Up4']
ml_dataframe['LastTradeTime'] = ml_dataframe['LastTradeTime'].apply(
    lambda x: datetime.strptime(str(x), '%y%m%d_%H%M') if not pd.isna(x) else np.nan)
ml_dataframe['LastTradeTime'] = ml_dataframe['LastTradeTime'].apply(lambda x: x.timestamp())
ml_dataframe['LastTradeTime'] = ml_dataframe['LastTradeTime'] / (60 * 60 * 24 * 7)

ml_dataframe['ExpDate'] = ml_dataframe['ExpDate'].astype(float)
study_name = 'SPY_1hour_prior231002'
num_features_to_select = 8
cells_forward_to_check = 60*1

threshold_cells_up = cells_forward_to_check * .75
threshold_cells_down = cells_forward_to_check * .75

threshold_up = 0.5
threshold_down = 0.5

percent_up = .15
percent_down = -.15


ml_dataframe.dropna(subset=Chosen_Predictor, inplace=True)
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
targetUpCounter = 0
targetDownCounter = 0
for i in range(1, cells_forward_to_check + 1):
    shifted_values = ml_dataframe["Current Stock Price"].shift(-i)
    condition_met_up = shifted_values > (
            ml_dataframe["Current Stock Price"] + (ml_dataframe["Current Stock Price"] * (percent_up / 100)))
    condition_met_down = shifted_values < (
            ml_dataframe["Current Stock Price"] + (ml_dataframe["Current Stock Price"] * (percent_down / 100)))
    targetUpCounter += condition_met_up.astype(int)
    targetDownCounter += condition_met_down.astype(int)
    ml_dataframe["Target_Down"] = (targetDownCounter >= threshold_cells_down).astype(int)
    ml_dataframe["Target_Up"] = (targetUpCounter >= threshold_cells_up).astype(int)

ml_dataframe.dropna(subset=['Target_Up', 'Target_Down'], inplace=True)
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


X_train, X_temp, y_up_train, y_up_temp, y_down_train, y_down_temp = train_test_split(
    X, y_up, y_down, test_size=0.2, random_state=None
)

# Then, split the temporary set into validation and test sets
X_val, X_test, y_up_val, y_up_test, y_down_val, y_down_test = train_test_split(
    X_temp, y_up_temp, y_down_temp, test_size=0.5, random_state=None
)
print('y_up_valsum ', y_up_val.sum(), 'lenYvalup ', len(y_up_val))
print('y_down_valsum ', y_down_val.sum(), 'lenYvaldown ', len(y_down_val))

print('y_up_testsum ', y_up_test.sum(), 'lenYtestup ', len(y_up_test))
print('y_down_testsum ', y_down_test.sum(), 'lenYtestdown ', len(y_down_test))
# Handle inf and -inf values based on the training set

for col in X_train.columns:
    max_val = X_train[col].replace([np.inf, -np.inf], np.nan).max()
    min_val = X_train[col].replace([np.inf, -np.inf], np.nan).min()

    # Adjust max_val based on its sign
    max_val = max_val * 1.5 if max_val >= 0 else max_val / 1.5

    # Adjust min_val based on its sign
    min_val = min_val * 1.5 if min_val < 0 else min_val / 1.5
    print("min/max values ", min_val, max_val)
    # Apply the same max_val and min_val to training, validation, and test sets
    # Apply the same max_val and min_val to training, validation, and test sets
    X_train[col].replace([np.inf, -np.inf], [max_val, min_val], inplace=True)
    X_val[col].replace([np.inf, -np.inf], [max_val, min_val], inplace=True)  # Include this
    X_test[col].replace([np.inf, -np.inf], [max_val, min_val], inplace=True)


num_positive_up = sum(y_up_train)
num_negative_up = len(y_up_train) - num_positive_up
weight_negative_up = 1.0
weight_positive_up = num_negative_up / num_positive_up
# Calculate class weights


num_positive_down = sum(y_down_train)  # Number of positive cases in the training set
num_negative_down = len(y_down_train) - num_positive_down  # Number of negative cases in the training set
weight_negative_down = 1.0
weight_positive_down = num_negative_down / num_positive_down
# print('num_positive_up= ', num_positive_up, 'num_negative_up= ', num_negative_up, 'num_positive_down= ',
#       num_positive_down, 'negative_down= ', num_negative_down)
print('weight_positve_up: ', weight_positive_up, '//weight_negative_up: ', weight_negative_up)
print('weight_positve_down: ', weight_positive_down, '//weight_negative_down: ', weight_negative_down)
# Define custom class weights as a dictionary
custom_weights_up = {0: weight_negative_up,
                     1: weight_positive_up}  # Assign weight_negative to class 0 and weight_positive to class 1
custom_weights_down = {0: weight_negative_down,
                       1: weight_positive_down}  # Assign weight_negative to class 0 and weight_positive to class 1
# Create RandomForestClassifier with custom class weights
tscv = TimeSeriesSplit(n_splits=3)


def objective_up(trial, X_train_selected, y_train, X_val_selected_up, y_val):
    # Define hyperparameter search space
    max_depth = trial.suggest_int('max_depth', 5, 25)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 8)

    # Create and train model
    model_up = RandomForestClassifier(max_depth=max_depth, min_samples_split=min_samples_split,
                                      n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                      class_weight=custom_weights_up)
    model_up.fit(X_train_selected, y_train)
    y_pred = model_up.predict(X_val_selected_up)
    y_prob = model_up.predict_proba(X_val_selected_up)[:, 1]  # Probability estimates of the positive class

    precision = precision_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    traincv_prec_scores_up = cross_val_score(model_up, X_train_selected_up, y_up_train, cv=tscv, scoring='precision')
    traincv_f1_scores_up = cross_val_score(model_up, X_train_selected_up, y_up_train, cv=tscv, scoring='f1')
    print("TRAINCross-validation prec,f1 scores for Target_up:", traincv_prec_scores_up, traincv_f1_scores_up)
    print("TRAINMean cross-validation score for Target_up:", traincv_prec_scores_up.mean(), traincv_f1_scores_up.mean(),
          '\n')
    alpha = .5
    auc = roc_auc_score(y_val, y_prob)
    beta = 0.3  # You can adjust this weight as well
    print('Precision: ', precision, 'F1 : ', f1, 'AUC :', auc)
    # Blend the scores using alpha
    val_cv_prec_scores_up = cross_val_score(model_up, X_val_selected_up, y_up_val, cv=tscv, scoring='precision')
    val_cv_f1_scores_up = cross_val_score(model_up, X_val_selected_up, y_up_val, cv=tscv, scoring='f1')

    print("ValCross-validation prec,f1 scores for Target_up:", val_cv_prec_scores_up, val_cv_f1_scores_up)
    print("ValMean cross-validation score for Target_up:", val_cv_prec_scores_up.mean(), val_cv_f1_scores_up.mean(), '\n')
    # blended_score = alpha * (1 - precision) + (1 - alpha) * cv_scores_down.mean()
    blended_score = alpha * (0.5 * (1 - traincv_prec_scores_up.mean()) + 0.5 * (1 - traincv_f1_scores_up.mean())) + (
                1 - alpha) * (
                            0.5 * (1 - val_cv_prec_scores_up.mean()) + 0.5 * (1 - val_cv_f1_scores_up.mean()))

    return blended_score


def objective_down(trial, X_train_selected, y_train, X_val_selected_down, y_val):
    max_depth = trial.suggest_int('max_depth', 5, 25)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
    n_estimators = trial.suggest_int('n_estimators', 100, 1000)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 8)

    # Create and train model

    model_down = RandomForestClassifier(max_depth=max_depth, min_samples_split=min_samples_split,
                                        n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
                                        class_weight=custom_weights_down)
    model_down.fit(X_train_selected, y_train)
    y_pred = model_down.predict(X_val_selected_down)
    y_prob = model_down.predict_proba(X_val_selected_down)[:, 1]  # Probability estimates of the positive class

    precision = precision_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    traincv_prec_scores_down = cross_val_score(model_down, X_train_selected_down, y_down_train, cv=tscv,
                                               scoring='precision')
    traincv_f1_scores_down = cross_val_score(model_down, X_train_selected_down, y_down_train, cv=tscv, scoring='f1')
    print("TRAINCross-validation prec,f1 scores for Target_up:", traincv_prec_scores_down, traincv_f1_scores_down)
    print("TRAINMean cross-validation score for Target_up:", traincv_prec_scores_down.mean(),
          traincv_f1_scores_down.mean(), '\n')
    alpha = .5
    auc = roc_auc_score(y_val, y_prob)
    beta = 0.3  # You can adjust this weight as well
    print('Precision: ', precision, 'F1 : ', f1, 'AUC :', auc)
    # Blend the scores using alpha
    blended_score = alpha * (1 - precision) + (1 - alpha) * f1
    # blended_score = (f1 + precision + auc) / 3
    cv_prec_scores_down = cross_val_score(model_down, X_val_selected_down, y_down_val, cv=tscv, scoring='precision')
    cv_f1_scores_down = cross_val_score(model_down, X_val_selected_down, y_down_val, cv=tscv, scoring='f1')

    print("VALCross-validation prec,f1 scores for Target_Down:", cv_prec_scores_down, cv_f1_scores_down)
    print("VALMean cross-validation score for Target_Down:", cv_prec_scores_down.mean(), cv_f1_scores_down.mean(), '\n')
    # blended_score = alpha * (1 - precision) + (1 - alpha) * cv_scores_down.mean()
    blended_score = alpha * (
                0.5 * (1 - traincv_prec_scores_down.mean()) + 0.5 * (1 - traincv_f1_scores_down.mean())) + (
                                1 - alpha) * (
                            0.5 * (1 - cv_prec_scores_down.mean()) + 0.5 * (1 - cv_f1_scores_down.mean()))

    return blended_score


# Feature selection and transformation
feature_selector_down = SelectKBest(score_func=mutual_info_classif, k=num_features_to_select)
feature_selector_up = SelectKBest(score_func=mutual_info_classif, k=num_features_to_select)
X_train_selected_up = feature_selector_up.fit_transform(X_train, y_up_train)
X_val_selected_up = feature_selector_up.transform(X_val)

X_test_selected_up = feature_selector_up.transform(X_test)
X_train_selected_down = feature_selector_down.fit_transform(X_train, y_down_train)
X_val_selected_down = feature_selector_down.transform(X_val)

X_test_selected_down = feature_selector_down.transform(X_test)
best_features_up = [Chosen_Predictor[i] for i in feature_selector_up.get_support(indices=True)]
print("Best features for Target_Up:", best_features_up)

best_features_down = [Chosen_Predictor[i] for i in feature_selector_down.get_support(indices=True)]
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
study_up.optimize(lambda trial: objective_up(trial, X_train_selected_up, y_up_train, X_val_selected_up, y_up_val),
                  n_trials=100)
# best_params_up = {'max_depth': 13, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 519}
# Results   best_params_up ={'max_depth': 13, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 519}
# best_params_up = study_up.best_params
# best_score_up = study_up.best_value
# print(f"Best parameters up: {best_params_up}")
# print(f"Best score up: {best_score_up}")
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
study_down.optimize(
    lambda trial: objective_down(trial, X_train_selected_down, y_down_train, X_val_selected_down, y_down_val),
    n_trials=100)
# best_params_down = {'max_depth': 13, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 519}

# best_params_down = study_down.best_params
# best_score_down = study_down.best_value

print(f"Best parameters down: {best_params_down}")
# print(f"Best score down: {best_score_down}")
model_up = RandomForestClassifier(**best_params_up, class_weight=custom_weights_up)
model_down = RandomForestClassifier(**best_params_down, class_weight=custom_weights_down)
# Concatenate training and validation sets for 'up'
X_train_val_selected_up = np.vstack((X_train_selected_up, X_val_selected_up))
y_train_val_up = np.concatenate((y_up_train, y_up_val))

# Concatenate training and validation sets for 'down'
X_train_val_selected_down = np.vstack((X_train_selected_down, X_val_selected_down))
y_train_val_down = np.concatenate((y_down_train, y_down_val))
# Fit the models on the selected features
model_up.fit(X_train_val_selected_up, y_train_val_up)
model_down.fit(X_train_val_selected_down, y_train_val_down)



importance_tuples_up = [(feature, importance) for feature, importance in
                        zip(Chosen_Predictor, model_up.feature_importances_)]
importance_tuples_up = sorted(importance_tuples_up, key=lambda x: x[1], reverse=True)
for feature, importance in importance_tuples_up:
    print(f"model up {feature}: {importance}")

importance_tuples_down = [(feature, importance) for feature, importance in
                          zip(Chosen_Predictor, model_down.feature_importances_)]
importance_tuples_down = sorted(importance_tuples_down, key=lambda x: x[1], reverse=True)

for feature, importance in importance_tuples_down:
    print(f"Model down {feature}: {importance}")

# Use the selected features for prediction
predicted_probabilities_up = model_up.predict_proba(X_test_selected_up)
predicted_probabilities_down = model_down.predict_proba(X_test_selected_down)

###predict
predicted_up = (predicted_probabilities_up[:, 1] > threshold_up).astype(int)
predicted_down = (predicted_probabilities_down[:, 1] > threshold_down).astype(int)

test_precision_up = precision_score(y_up_test, predicted_up)
test_accuracy_up = accuracy_score(y_up_test, predicted_up)
test_recall_up = recall_score(y_up_test, predicted_up)
test_f1_up = f1_score(y_up_test, predicted_up)

test_precision_down = precision_score(y_down_test, predicted_down)
test_accuracy_down = accuracy_score(y_down_test, predicted_down)
test_recall_down = recall_score(y_down_test, predicted_down)
test_f1_down = f1_score(y_down_test, predicted_down)
tscv1 = TimeSeriesSplit(n_splits=5)

test_cv_prec_score_up = cross_val_score(model_up, X_test_selected_up, y_up_test, cv=tscv1, scoring='precision')
test_cv_prec_score_down = cross_val_score(model_down, X_test_selected_down, y_down_test, cv=tscv1, scoring='precision')
test_cv_f1_score_up = cross_val_score(model_up, X_test_selected_up, y_up_test, cv=tscv1, scoring='f1')
test_cv_f1_score_down = cross_val_score(model_down, X_test_selected_down, y_down_test, cv=tscv1, scoring='f1')
print("test Metrics for Target_Up:", '\n')
print("testPrecision:", test_precision_up)
print("testAccuracy:", test_accuracy_up)
print("testRecall:", test_recall_up)
print("testF1-Score:", test_f1_up, '\n')
print("testCross-validation scores for Target_Up:(prec/f1)", test_cv_prec_score_up, test_cv_f1_score_up)
print("testMean cross-validation score for Target_Up:", test_cv_prec_score_up.mean(), test_cv_f1_score_up.mean(), '\n')

print("test Metrics for Target_Down:", '\n')
print("testPrecision:", test_precision_down)
print("testAccuracy:", test_accuracy_down)
print("testRecall:", test_recall_down)
print("testF1-Score:", test_f1_down, '\n')

print("test Cross-validation scores for Target_Down:(prec/f1)", test_cv_prec_score_down, test_cv_f1_score_down)
print("test Mean cross-validation score for Target_Down:(prec/f1)", test_cv_prec_score_down.mean(),
      test_cv_f1_score_down.mean(), '\n')



#scaling for final model
min_max_dict = {}
for col in X.columns:
    max_val = X[col].replace([np.inf, -np.inf], np.nan).max()
    min_val = X[col].replace([np.inf, -np.inf], np.nan).min()

    # Adjust max_val based on its sign
    max_val = max_val * 1.5 if max_val >= 0 else max_val / 1.5
    # Adjust min_val based on its sign
    min_val = min_val * 1.5 if min_val < 0 else min_val / 1.5
    print("min/max values ", min_val, max_val)
    X[col].replace([np.inf, -np.inf], [max_val, min_val], inplace=True)

    min_max_dict[col] = {'min_val': min_val, 'max_val': max_val}
X_selected_up = feature_selector_up.transform(X)
X_selected_down = feature_selector_down.transform(X)

model_up.fit(X_selected_up, y_up)
model_down.fit(X_selected_down, y_down)

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
    model_directory = os.path.join("..", "..", "..", "Trained_Models", model_summary)

    print("Current working directory:", os.getcwd())

    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
        print(f"Directory created at: {model_directory}")

    model_filename_up = os.path.join(model_directory, "target_up.joblib")
    model_filename_down = os.path.join(model_directory, "target_down.joblib")

    joblib.dump(model_up, model_filename_up)
    joblib.dump(model_down, model_filename_down)

    with open(
            f"../../../Trained_Models/{model_summary}/info.txt", "w") as info_txt:
        info_txt.write("This file contains information about the model.\n\n")
        info_txt.write(
            f"File analyzed: {DF_filename}\nCells_Foward_to_check: {cells_forward_to_check}\n\nBest parameters for Target_Up: {best_params_down}. \n\nBest parameters for Target_Down: {best_params_down}. \n\ntest_Metrics for Target_Up:\nPrecision: {test_precision_up}\nAccuracy: {test_accuracy_up}\nRecall: {test_recall_up}\nF1-Score: {test_f1_up}\ntest_Cross-validation scores for Target_Up:(prec/f1) {test_cv_prec_score_up, test_cv_f1_score_up}\nMean test_cross-validation score for Target_Up:(prec/f1) {test_cv_prec_score_up.mean(), test_cv_f1_score_up.mean()}\n\ntest_Metrics for Target_Down:\nPrecision: {test_precision_down}\nAccuracy: {test_accuracy_down}\nRecall: {test_recall_down}\nF1-Score: {test_f1_down}\ntest_Cross-validation scores for Target_Down:(prec/f1) {test_cv_prec_score_down, test_cv_f1_score_down}\nMean test_cross-validation score for Target_Down:(prec/f1) {test_cv_prec_score_down.mean(), test_cv_f1_score_down.mean()}\n\n")
        info_txt.write(
            f"Min value//max value used for -inf/inf:{min_val, max_val}Predictors: {Chosen_Predictor}\n\nBest_Predictors_Selected Up: {best_features_up}\nBest_Predictors_Selected Down: {best_features_down}\n\nThreshold Up(sensitivity): {threshold_up}\nThreshold Down(sensitivity): {threshold_down}\nTarget Underlying Percentage Up: {percent_up}\nTarget Underlying Percentage Down: {percent_down}\n")
    with open(f"../../../Trained_Models/{model_summary}/min_max_values.json", 'w') as f:
        json.dump(min_max_dict, f)
    with open(f"../../../Trained_Models/{model_summary}/features_up.json", 'w') as f2:
        json.dump(best_features_up, f2)
    with open(f"../../../Trained_Models/{model_summary}/features_down.json", 'w') as f3:
        json.dump(best_features_down, f3)
else:

    exit()
