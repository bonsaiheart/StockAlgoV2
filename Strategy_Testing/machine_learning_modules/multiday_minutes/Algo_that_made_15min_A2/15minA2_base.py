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
     'Bonsai Ratio',  'PCRv Up4', 'PCRv Down4', 'ITM PCRv Up2',
    'ITM PCRv Down2',
    'ITM PCRv Up4', 'ITM PCRv Down4',
    'RSI14', 'AwesomeOsc5_34', 'RSI', 'RSI2',
    'AwesomeOsc'

]
# Chosen_Predictor = ['ExpDate','LastTradeTime','Current Stock Price','Current SP % Change(LAC)','Bonsai Ratio','Bonsai Ratio 2','B1/B2','B2/B1','PCR-Vol','PCRv @CP Strike','PCRoi @CP Strike','PCRv Up1','PCRv Up4','PCRv Down1','PCRv Down2','PCRv Down4',"PCRoi Up1",'PCRoi Up4','PCRoi Down1','PCRoi Down2','PCRoi Down3','ITM PCR-Vol','ITM PCRv Up1','ITM PCRv Up4','ITM PCRv Down1','ITM PCRv Down2','ITM PCRv Down3','ITM PCRv Down4','ITM PCRoi Up1','ITM PCRoi Up2','ITM PCRoi Up3','ITM PCRoi Up4','ITM PCRoi Down2','ITM PCRoi Down3','ITM PCRoi Down4','ITM OI','Total OI','Net_IV','Net ITM IV','Net IV MP','Net IV LAC','NIV Current Strike','NIV 1Lower Strike','NIV 2Higher Strike','NIV 2Lower Strike','NIV 3Higher Strike','NIV 3Lower Strike','NIV 4Lower Strike','NIV highers(-)lowers1-2','NIV 1-4 % from mean','Net_IV/OI','Net ITM_IV/ITM_OI','Closest Strike to CP','RSI','RSI2','RSI14','AwesomeOsc','AwesomeOsc5_34']
#
#had highest corr for 3-5 hours with these:
Chosen_Predictor = ['Bonsai Ratio','Bonsai Ratio 2','PCRoi Up1', 'B1/B2', 'PCRv Up4']
ml_dataframe['LastTradeTime'] = ml_dataframe['LastTradeTime'].apply(
    lambda x: datetime.strptime(str(x), '%y%m%d_%H%M') if not pd.isna(x) else np.nan)
ml_dataframe['LastTradeTime'] = ml_dataframe['LastTradeTime'].apply(lambda x: x.timestamp())
ml_dataframe['LastTradeTime'] = ml_dataframe['LastTradeTime'] / (60 * 60 * 24 * 7)
ml_dataframe['ExpDate'] = ml_dataframe['ExpDate'].astype(float)
ml_dataframe.dropna(subset=Chosen_Predictor, inplace=True)

study_name = 'SPY_3hr_prior231002_auc_a2'
percent_up = .3
percent_down = -.3
num_features_to_select = 8
cells_forward_to_check = 60*4

threshold_cells_up = cells_forward_to_check * .2
threshold_cells_down = cells_forward_to_check * .2
# pos_weight_multiplier = 1



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


def calculate_metrics(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return precision, accuracy, recall, f1
def calculate_class_weights(y,pos_weight_multiplier ):
    # Calculate the number of positive and negative cases
    num_positive = sum(y)
    num_negative = len(y) - num_positive

    # Calculate class weights
    weight_negative = 1.0
    weight_positive = num_negative / num_positive if num_positive > 0 else 1.0
    weight_positive = weight_positive*pos_weight_multiplier

    # Define custom class weights as a dictionary
    class_weights = {0: weight_negative, 1: weight_positive}

    return class_weights

def select_features(X, y, num_features_to_select, feature_names):
    # Perform feature selection using mutual information
    feature_selector = SelectKBest(score_func=mutual_info_classif, k='all')

    # Fit the feature selector on the data
    X_selected = feature_selector.fit_transform(X, y)

    # Get the indices of the selected features
    selected_feature_indices = feature_selector.get_support(indices=True)

    # Get the names of the selected features
    selected_feature_names = [feature_names[i] for i in selected_feature_indices]

    # Convert X_selected back to a DataFrame
    X_selected_df = pd.DataFrame(X_selected, columns=selected_feature_names)

    return X_selected_df, selected_feature_names


X_train, X_temp, y_up_train, y_up_temp, y_down_train, y_down_temp = train_test_split(
    X, y_up, y_down, test_size=0.15, random_state=None, shuffle=False
)
# Then, split the temporary set into validation and test sets
X_val, X_test, y_up_val, y_up_test, y_down_val, y_down_test = train_test_split(
    X_temp, y_up_temp, y_down_temp, test_size=0.5, random_state=None, shuffle=False
)
# # Generate indices for the entire dataset
# total_indices = np.arange(len(X))
#
# # Calculate lengths of training, validation, and test sets
# train_len = int(0.6 * len(X))  # 60% for training
# val_len = int(0.2 * len(X))    # 20% for validation
# # Remaining 20% for test
#
# # Shuffle only the training indices
# train_indices = np.random.permutation(total_indices[:train_len])
#
# # Keep the rest as is
# val_indices = total_indices[train_len:train_len + val_len]
# test_indices = total_indices[train_len + val_len:]
#
# # Extract sets using the indices
# X_train = X.iloc[train_indices]
# y_up_train = y_up.iloc[train_indices]
# y_down_train = y_down.iloc[train_indices]
#
# X_val = X.iloc[val_indices]
# y_up_val = y_up.iloc[val_indices]
# y_down_val = y_down.iloc[val_indices]
#
# X_test = X.iloc[test_indices]
# y_up_test = y_up.iloc[test_indices]
# y_down_test = y_down.iloc[test_indices]

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






# Define custom class weights as a dictionary

# Create RandomForestClassifier with custom class weights
tscv = TimeSeriesSplit(n_splits=3)
X_train_selected_up, selected_feature_names_up = select_features(X_train, y_up_train, num_features_to_select, Chosen_Predictor)
print(type(X_train_selected_up))
X_val_selected_up = X_val[selected_feature_names_up]
X_test_selected_up = X_test[selected_feature_names_up]

# Call the select_features function for the 'down' case
X_train_selected_down, selected_feature_names_down = select_features(X_train, y_down_train, num_features_to_select, Chosen_Predictor)
X_val_selected_down = X_val[selected_feature_names_down]
X_test_selected_down = X_test[selected_feature_names_down]

print("Best features for Target_Up:", selected_feature_names_up)
print("Best features for Target_Down:", selected_feature_names_down)
# Get the number of selected features for the 'up' case
n_features_up = X_train_selected_up.shape[1]

# Get the number of selected features for the 'down' case
n_features_down = X_train_selected_down.shape[1]

def objective_up(trial, X_train_selected, y_train, X_val_selected_up, y_val):
    # Define hyperparameter search space
    max_depth = trial.suggest_int('max_depth', 1, 50)  # Broadened range
    min_samples_split = trial.suggest_int('min_samples_split', 2, 50)  # Broadened range
    n_estimators = trial.suggest_int('n_estimators', 50, 2000)  # Broadened range
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)  # Broadened range

    # Additional hyperparameters
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
    max_features_value = n_features_up // 3  # Integer division
    max_features = trial.suggest_categorical('max_features', ['auto', 'sqrt', max_features_value, 'log2'])
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])
    positivecase_weight_up_multiplier = trial.suggest_float("positivecase_weight_up", 1, 10)

    custom_weights_up = calculate_class_weights(y_up_train,positivecase_weight_up_multiplier)

    # Create and train model
    # Create and train model
    model_up = RandomForestClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,  # New hyperparameter
        max_features=max_features,  # New hyperparameter
        bootstrap=bootstrap,  # New hyperparameter
        class_weight=custom_weights_up
    )

    model_up.fit(X_train_selected, y_train)
    y_pred = model_up.predict(X_val_selected_up)
    y_prob = model_up.predict_proba(X_val_selected_up)[:, 1]  # Probability estimates of the positive class

    precision = precision_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    traincv_prec_scores_up = cross_val_score(model_up, X_train_selected_up, y_up_train, cv=tscv, scoring='precision')
    traincv_f1_scores_up = cross_val_score(model_up, X_train_selected_up, y_up_train, cv=tscv, scoring='f1')
    print("TRAINCross-validation prec,f1 scores for Target_up:", traincv_prec_scores_up, traincv_f1_scores_up)
    # print("TRAINMean cross-validation score for Target_up:", traincv_prec_scores_up.mean(), traincv_f1_scores_up.mean(),
    #       )
    alpha = .5
    auc = roc_auc_score(y_val, y_prob)
    beta = 0.3  # You can adjust this weight as well
    print('Precision: ', precision, 'F1 : ', f1, 'AUC :', auc)
    # Blend the scores using alpha
    val_cv_prec_scores_up = cross_val_score(model_up, X_val_selected_up, y_up_val, cv=tscv, scoring='precision')
    val_cv_f1_scores_up = cross_val_score(model_up, X_val_selected_up, y_up_val, cv=tscv, scoring='f1')
    test_cv_prec_scores_up = cross_val_score(model_up, X_test_selected_up, y_up_test, cv=tscv, scoring='precision')
    test_cv_f1_scores_up = cross_val_score(model_up, X_test_selected_up, y_up_test, cv=tscv, scoring='f1')
    print("ValCross-validation prec,f1 scores for Target_up:", val_cv_prec_scores_up, val_cv_f1_scores_up)
    # print("ValMean cross-validation score for Target_up:", val_cv_prec_scores_up.mean(), val_cv_f1_scores_up.mean())
    # blended_score = alpha * (1 - precision) + (1 - alpha) * cv_scores_down.mean()
    blended_score = alpha * (0.5 * (1 - traincv_prec_scores_up.mean()) + 0.5 * (1 - traincv_f1_scores_up.mean())) + (
                1 - alpha) * (
                            0.5 * (1 - val_cv_prec_scores_up.mean()) + 0.5 * (1 - val_cv_f1_scores_up.mean()))
    print("testCross-validation prec,f1 scores for Target_up:", test_cv_prec_scores_up, test_cv_f1_scores_up)
    # print("testMean cross-validation score for Target_up:", test_cv_prec_scores_up.mean(), test_cv_f1_scores_up.mean())
    blended_score = precision+f1
    return auc


def objective_down(trial, X_train_selected, y_train, X_val_selected_down, y_val):
    max_depth = trial.suggest_int('max_depth', 1, 50)  # Broadened range
    min_samples_split = trial.suggest_int('min_samples_split', 2, 50)  # Broadened range
    n_estimators = trial.suggest_int('n_estimators', 50, 2000)  # Broadened range
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 20)  # Broadened range

    # Additional hyperparameters
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])
    max_features_value = n_features_down // 3  # Integer division
    max_features = trial.suggest_categorical('max_features', ['auto', 'sqrt', max_features_value, 'log2'])
    bootstrap = trial.suggest_categorical('bootstrap', [True, False])
    positivecase_weight_down_multiplier = trial.suggest_float("positivecase_weight_down", 1, 10)
    custom_weights_down = calculate_class_weights(y_down_train,positivecase_weight_down_multiplier)

    # Create and train model
    # Create and train model
    model_down = RandomForestClassifier(
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        criterion=criterion,  # New hyperparameter
        max_features=max_features,  # New hyperparameter
        bootstrap=bootstrap,  # New hyperparameter
        class_weight=custom_weights_down
    )

    model_down.fit(X_train_selected, y_train)
    y_pred = model_down.predict(X_val_selected_down)
    y_prob = model_down.predict_proba(X_val_selected_down)[:, 1]  # Probability estimates of the positive class

    precision = precision_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    traincv_prec_scores_down = cross_val_score(model_down, X_train_selected_down, y_down_train, cv=tscv,
                                               scoring='precision')
    traincv_f1_scores_down = cross_val_score(model_down, X_train_selected_down, y_down_train, cv=tscv, scoring='f1')
    # print("TRAINCross-validation prec,f1 scores for Target_up:", traincv_prec_scores_down, traincv_f1_scores_down)
    # print("TRAINMean cross-validation score for Target_up:", traincv_prec_scores_down.mean(),
    #       traincv_f1_scores_down.mean(), )
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
    # blended_score = alpha * (
    #             0.5 * (1 - traincv_prec_scores_down.mean()) + 0.5 * (1 - traincv_f1_scores_down.mean())) + (
    #                             1 - alpha) * (
    #                         0.5 * (1 - cv_prec_scores_down.mean()) + 0.5 * (1 - cv_f1_scores_down.mean()))
    blended_score = precision+f1

    return auc


# Feature selection and transformation



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
    study_up = optuna.create_study(direction="maximize", study_name=f'{study_name}_up',
                                   storage=f'sqlite:///{study_name}_up.db')
"Keyerror, new optuna study created."  #

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
study_up.optimize(lambda trial: objective_up(trial, X_train_selected_up, y_up_train, X_val_selected_up, y_up_val),
                  n_trials=1000)
study_down.optimize(
    lambda trial: objective_down(trial, X_train_selected_down, y_down_train, X_val_selected_down, y_down_val),
    n_trials=1000)
# best_params_up = {'max_depth': 13, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 519}
# best_params_up=study_up.best_params
# best_value_up=study_up.best_value
# best_params_down=study_down.best_params
# best_value_down=study_down.best_value
# print(f"Best parameters up: {best_params_up}")
# print(f"Best score up: {best_value_up}")
#
# best_params_down = {'max_depth': 13, 'min_samples_leaf': 1, 'min_samples_split': 3, 'n_estimators': 519}
# print(f"Best parameters down: {best_params_down}")
# print(f"Best score down: {best_value_down}")

model_up = RandomForestClassifier(**best_params_up, class_weight=custom_weights_up)
model_down = RandomForestClassifier(**best_params_down, class_weight=custom_weights_down)
# Concatenate training and validation sets for 'up'
X_train_val_selected_up =pd.concat([X_train_selected_up, X_val_selected_up], axis=0)
y_train_val_up = np.concatenate((y_up_train, y_up_val))
# Concatenate training and validation sets for 'down'
X_train_val_selected_down = pd.concat([X_train_selected_down, X_val_selected_down], axis=0)
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
predicted_up = model_up.predict(X_test_selected_up)
predicted_down = model_down.predict(X_test_selected_down)



test_precision_up, test_accuracy_up, test_recall_up, test_f1_up = calculate_metrics(y_up_test, predicted_up)
test_precision_down, test_accuracy_down, test_recall_down, test_f1_down = calculate_metrics(y_down_test, predicted_down)

tscv1 = TimeSeriesSplit(n_splits=5)

test_cv_prec_score_up = cross_val_score(model_up, X_test_selected_up, y_up_test, cv=tscv1, scoring='precision')
test_cv_prec_score_down = cross_val_score(model_down, X_test_selected_down, y_down_test, cv=tscv1, scoring='precision')
test_cv_f1_score_up = cross_val_score(model_up, X_test_selected_up, y_up_test, cv=tscv1, scoring='f1')
test_cv_f1_score_down = cross_val_score(model_down, X_test_selected_down, y_down_test, cv=tscv1, scoring='f1')
print("test Metrics for Target_Up:", '\n')
print("testPrecision:", test_precision_up)
print("testAccuracy:", test_accuracy_up)
print("testF1-Score:", test_f1_up, '\n')
print("testCross-validation scores for Target_Up:(prec/f1)", test_cv_prec_score_up, test_cv_f1_score_up)
print("testMean cross-validation score for Target_Up:", test_cv_prec_score_up.mean(), test_cv_f1_score_up.mean(), '\n')
print("test Metrics for Target_Down:", '\n')
print("testPrecision:", test_precision_down)
print("testAccuracy:", test_accuracy_down)
print("testF1-Score:", test_f1_down, '\n')
print("test Cross-validation scores for Target_Down:(prec/f1)", test_cv_prec_score_down, test_cv_f1_score_down)
print("test Mean cross-validation score for Target_Down:(prec/f1)", test_cv_prec_score_down.mean(), test_cv_f1_score_down.mean(), '\n')


final_weights_up = calculate_class_weights(y_up)
final_weights_down = calculate_class_weights(y_down)
model_up = RandomForestClassifier(**best_params_up, class_weight=final_weights_up)
model_down = RandomForestClassifier(**best_params_down, class_weight=final_weights_down)
#scaling for final model
final_min_max_dict = {}
for col in X.columns:
    final_max_val = X[col].replace([np.inf, -np.inf], np.nan).max()
    final_min_val = X[col].replace([np.inf, -np.inf], np.nan).min()

    # Adjust max_val based on its sign
    final_max_val = final_max_val * 1.5 if final_max_val >= 0 else final_max_val / 1.5
    # Adjust min_val based on its sign
    final_min_val = final_min_val * 1.5 if final_min_val < 0 else final_min_val / 1.5
    print("min/max values ", final_min_val, final_max_val)
    X[col].replace([np.inf, -np.inf], [final_max_val, final_min_val], inplace=True)

    final_min_max_dict[col] = {'min_val': final_min_val, 'max_val': final_max_val}

X_selected_up = X[selected_feature_names_up]
X_selected_down = X[selected_feature_names_down]
model_up.fit(X_selected_up, y_up)
model_down.fit(X_selected_down, y_down)


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
            f"Min value//max value used for -inf/inf:{min_val, max_val}Predictors: {Chosen_Predictor}\n\nBest_Predictors_Selected Up: {selected_feature_names_up}\nBest_Predictors_Selected Down: {selected_feature_names_down}\n\nTarget Underlying Percentage Up: {percent_up}\nTarget Underlying Percentage Down: {percent_down}\n")
    with open(f"../../../Trained_Models/{model_summary}/min_max_values.json", 'w') as f:
        json.dump(final_min_max_dict, f)
    with open(f"../../../Trained_Models/{model_summary}/features_up.json", 'w') as f2:
        json.dump(selected_feature_names_up, f2)
    with open(f"../../../Trained_Models/{model_summary}/features_down.json", 'w') as f3:
        json.dump(selected_feature_names_down, f3)
else:

    exit()
