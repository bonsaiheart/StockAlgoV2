import datetime
import os
import warnings
from datetime import datetime
from sklearn.metrics import roc_auc_score, log_loss

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from joblib import dump
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn.model_selection import KFold, TimeSeriesSplit

# Filter out specific warning messages
warnings.filterwarnings("ignore", category=Warning)
warnings.filterwarnings("ignore", category=Warning, message="No further splits with positive gain, best gain: -inf")
OPTUNA_EARLY_STOPING = 10

class EarlyStoppingExceeded(optuna.exceptions.OptunaError):
    early_stop = OPTUNA_EARLY_STOPING
    early_stop_count = 0
    best_score = None

def early_stopping_opt(study, trial):
    # Determine if the study is maximizing or minimizing
    is_maximize = study.direction == optuna.study.StudyDirection.MAXIMIZE

    # Initialize best_score if it's None
    if EarlyStoppingExceeded.best_score is None:
        EarlyStoppingExceeded.best_score = study.best_value

    # Check if the new value is better than the best score
    if (is_maximize and study.best_value > EarlyStoppingExceeded.best_score) or \
       (not is_maximize and study.best_value < EarlyStoppingExceeded.best_score):
        EarlyStoppingExceeded.best_score = study.best_value
        EarlyStoppingExceeded.early_stop_count = 0
    else:
        if EarlyStoppingExceeded.early_stop_count > EarlyStoppingExceeded.early_stop:
            EarlyStoppingExceeded.early_stop_count = 0
            EarlyStoppingExceeded.best_score = None
            raise EarlyStoppingExceeded()
        else:
            EarlyStoppingExceeded.early_stop_count += 1

    print(f'EarlyStop counter: {EarlyStoppingExceeded.early_stop_count}, Best score: {study.best_value} and {EarlyStoppingExceeded.best_score}')
    return
# This modification checks the study's direction and adjusts the comparison logic accordingly. If the study is maximizing, it checks if the new value is greater than the best score. If the study is minimizing, it checks if the new value is less than the best score.






# Your code here
# ...
from warnings import simplefilter

simplefilter("ignore", category=RuntimeWarning)
# Restore the warning filter (if needed)
# warnings.filterwarnings("default", category=Warning)

DF_filename = r"../../../../data/historical_multiday_minute_DF/older/SPY_historical_multiday_min.csv"
# TODO add early stop or no?
# from tensorflow.keras.callbacks import EarlyStopping
'''use lasso?# Sample data
data = load_iris()
X, y = data.data, data.target

# Lasso for feature selection
alpha_value = 0.01  # adjust based on your needs
lasso = Lasso(alpha=alpha_value)
lasso.fit(X, y)

# Features with non-zero coefficients
selected_features = np.where(lasso.coef_ != 0)[0]

# Train Random Forest on selected features
rf = RandomForestClassifier()
rf.fit(X[:, selected_features], y)'''
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
#

# ##had highest corr for 3-5 hours with these:
# Chosen_Predictor = ['Bonsai Ratio','Bonsai Ratio 2','ITM PCR-Vol','ITM PCRoi Up1', 'RSI14','AwesomeOsc5_34', 'Net_IV']
ml_dataframe = pd.read_csv(DF_filename)
print(ml_dataframe.columns)
# ##had highest corr for 3-5 hours with these:
Chosen_Predictor = ['Bonsai Ratio','Bonsai Ratio 2','ITM PCR-Vol','ITM PCRoi Up1', 'RSI14','AwesomeOsc5_34', 'Net_IV']
ml_dataframe['LastTradeTime'] = ml_dataframe['LastTradeTime'].apply(
    lambda x: datetime.strptime(str(x), '%y%m%d_%H%M') if not pd.isna(x) else np.nan)
ml_dataframe['LastTradeTime'] = ml_dataframe['LastTradeTime'].apply(lambda x: x.timestamp())
ml_dataframe['ExpDate'] = ml_dataframe['ExpDate'].astype(float)

cells_forward_to_check = 3 * 60  # rows to check(minutes in this case)
threshold_cells_up = cells_forward_to_check * 0.5  # how many rows must achieve target %
percent_up = .35  # target percetage.
anticondition_threshold_cells_up = cells_forward_to_check * .2  # was .7
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

X_test = X[split_index:].reset_index(drop=True)
y_test = y[split_index:].reset_index(drop=True)
X = X[:split_index].reset_index(drop=True)
y = y[:split_index].reset_index(drop=True)
print("Xlength: ",len(X), "XTestlen: ",len(X_test),"positive in y: ",y.sum(),"positive in ytest: ",y_test.sum())
# Fit the scaler on the entire training data
# scaler_X_trainval = RobustScaler().fit(X)
# scaler_y_trainval = RobustScaler().fit(y.values.reshape(-1, 1))

'''Metrics & Model Selection: You're storing the best model based on the F1 score. This is okay if F1 is the most important metric for your problem. If not, you might want to adjust this logic. Also, you could consider saving the models from all the folds and using a voting mechanism for predictions if you want to leverage the power of ensemble predictions'''


def train_model(param_dict, X, y, final_classifier=None):
    # Extract hyperparameters from param_dict
    num_leaves = param_dict['num_leaves']
    max_depth = param_dict['max_depth']
    learning_rate = param_dict['learning_rate']
    n_estimators = param_dict['n_estimators']
    min_child_samples = param_dict['min_child_samples']
    min_child_weight = param_dict['min_child_weight']
    subsample = param_dict['subsample']
    colsample_bytree = param_dict['colsample_bytree']
    reg_alpha = param_dict['reg_alpha']
    reg_lambda = param_dict['reg_lambda']
    max_bin = param_dict['max_bin']
    feature_fraction = param_dict['feature_fraction']

    scale_pos_weight = param_dict['scale_pos_weight']
    boosting_type = param_dict['boosting_type']
    # if param_dict['boosting_type'] != 'goss':
    #     # If boosting type is 'goss', remove bagging-related parameters
    #     bagging_fraction = param_dict['bagging_fraction']
    #     bagging_freq = param_dict['bagging_freq']

    device = "gpu"  # Use GPU
    best_model = None
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    tscv = TimeSeriesSplit(n_splits=5)  # No shuffle or random state, because it's a time series
    X_np = X.to_numpy()
    auc_scores = []  # To store AUC scores for each fold
    log_loss_scores = []  # To store log loss scores for each fold

    best_f1, best_precision, best_recall,best_accuracy,best_auc = 0, 0, 0, 0,0
    total_f1, total_precision, total_recall = 0, 0, 0
    num_folds = 0
    total_accuracy = 0

    for train_index, val_index in tscv.split(X_np):
        X_train, X_val = X_np[train_index], X_np[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Create LightGBM datasets
        train_data = lgb.Dataset(X_train, label=y_train)
        valid_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

        # Create and train the LightGBM model
        if boosting_type == "dart":
            drop_rate = param_dict['drop_rate']
            lgb_params = {
                # 'early_stopping_rounds':50,  # Specify early stopping rounds !not available in dart
                'verbose': -1,  # Set verbose to -1 to suppress most LightGBM output

                'num_leaves': num_leaves,
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                # 'n_estimators': n_estimators,
                'min_child_samples': min_child_samples,
                'min_child_weight': min_child_weight,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
                'reg_alpha': reg_alpha,
                'reg_lambda': reg_lambda,
                'max_bin': max_bin,
                'feature_fraction': feature_fraction,
                'bagging_fraction': param_dict['bagging_fraction'],
                'bagging_freq': param_dict['bagging_freq'],
                'scale_pos_weight': scale_pos_weight,
                'boosting_type': boosting_type,
                'drop_rate': drop_rate,
                'device': device
            }
        elif boosting_type == "goss":
            lgb_params = {
                'early_stopping_rounds': 50,  # Specify early stopping rounds
                'verbose': -1,  # Set verbose to -1 to suppress most LightGBM output

                'num_leaves': num_leaves,
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                # 'n_estimators': n_estimators,
                'min_child_samples': min_child_samples,
                'min_child_weight': min_child_weight,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
                'reg_alpha': reg_alpha,
                'reg_lambda': reg_lambda,
                'max_bin': max_bin,
                'feature_fraction': feature_fraction,
                # 'bagging_fraction': bagging_fraction,
                # 'bagging_freq': bagging_freq,
                'scale_pos_weight': scale_pos_weight,
                'boosting_type': boosting_type,
                'device': device
            }
        else:
            lgb_params = {
                'early_stopping_rounds': 10,  # Specify early stopping rounds
                'verbose': -1,  # Set verbose to -1 to suppress most LightGBM output

                'num_leaves': num_leaves,
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                # 'n_estimators': n_estimators,
                'min_child_samples': min_child_samples,
                'min_child_weight': min_child_weight,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
                'reg_alpha': reg_alpha,
                'reg_lambda': reg_lambda,
                'max_bin': max_bin,
                'feature_fraction': feature_fraction,
                'bagging_fraction': param_dict['bagging_fraction'],
                'bagging_freq': param_dict['bagging_freq'],
                'scale_pos_weight': scale_pos_weight,
                'boosting_type': boosting_type,
                'device': device    ,
                'objective': 'binary'
            }

        if final_classifier is not None:
            lgb_classifier = final_classifier  # Use the final classifier
        else:
            # Create and train the LightGBM model with hyperparameters
            lgb_classifier = lgb.train(

                params=lgb_params,
                train_set=train_data,
                num_boost_round=n_estimators,
                valid_sets=[valid_data],
            )

        # Predict on validation data
        y_pred = lgb_classifier.predict(X_val, num_iteration=lgb_classifier.best_iteration)
        # print(y_pred[0],y_pred[10]
        # y_pred[20],y_pred[-1])

        # Convert predictions to binary values (0 or 1)
        y_pred_binary = (y_pred > 0.5).astype(int)
        auc = roc_auc_score(y_val, y_pred)
        logloss = log_loss(y_val, y_pred)
        # Compute metrics
        f1 = f1_score(y_val, y_pred_binary)
        precision = precision_score(y_val, y_pred_binary)
        recall = recall_score(y_val, y_pred_binary)
        accuracy = accuracy_score(y_val, y_pred_binary)

        auc_scores.append(auc)

        log_loss_scores.append(logloss)
        # Update best scores

        total_f1 += f1
        total_accuracy += accuracy_score(y_val, y_pred_binary)

        total_precision += precision
        total_recall += recall
        num_folds += 1
    avg_accuracy = total_accuracy / num_folds  # Compute average accuracy after the loop
    avg_logloss = sum(log_loss_scores) / {num_folds}
    avg_f1 = total_f1 / num_folds
    avg_precision = total_precision / num_folds
    avg_recall = total_recall / num_folds
    avg_auc = sum(auc_scores) / num_folds
    if avg_auc >best_auc:
        print("best model assigned")
        best_f1 = avg_f1
        best_precision = avg_precision
        best_recall = avg_recall
        best_accuracy = avg_accuracy
        best_model = lgb_classifier
    print(f'avg_accuracy: {best_accuracy},avg_f1: {best_f1}, avg_precision: {best_precision}, avg_recall: {best_recall},avg_auc: {avg_auc},  avg_logloss: {avg_logloss}) / {num_folds}, ')
    return {
    'avg_accuracy': best_accuracy,  # Add avg_accuracy to your return dictionary

        'avg_f1': best_f1,
        'avg_precision': best_precision,
        'avg_recall': best_recall,
        'avg_auc': avg_auc,
        'avg_logloss': sum(log_loss_scores) / num_folds,
        'best_model': best_model
    }

# Define Optuna Objective
def objective(trial):
    print(datetime.now())
    try:

        num_leaves = trial.suggest_int("num_leaves", 25, 128)
        max_depth = trial.suggest_int("max_depth", 5, 20)  # -1 means no limit
        learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True)
        n_estimators = trial.suggest_int("n_estimators", 100, 2000)
        min_child_samples = trial.suggest_int("min_child_samples", 5, 100)
        min_child_weight = trial.suggest_float("min_child_weight", 1e-5, 1e-1, log=True)
        subsample = trial.suggest_float("subsample", 0.5, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0)
        reg_alpha = trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True)
        reg_lambda = trial.suggest_float("reg_lambda", 1e-5, 10.0, log=True)
        max_bin = trial.suggest_int("max_bin", 2, 128)

        feature_fraction = trial.suggest_float("feature_fraction", 0.5, 1.0)
        bagging_fraction = trial.suggest_float("bagging_fraction", 0.5, 1.0)
        bagging_freq = trial.suggest_int("bagging_freq", 0, 10)
        scale_pos_weight = trial.suggest_float("scale_pos_weight", 0.1, 20.0)
        # boosting_type = "gbdt"
        boosting_type = trial.suggest_categorical("boosting_type", ["goss"])# "gbdt", "dart",
        if boosting_type == "dart":
            drop_rate = trial.suggest_float("drop_rate", 0.1, 0.5)
            param_dict = {
                'num_leaves': num_leaves,
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'n_estimators': n_estimators,
                'min_child_samples': min_child_samples,
                'min_child_weight': min_child_weight,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
                'reg_alpha': reg_alpha,
                'reg_lambda': reg_lambda,
                'max_bin': max_bin,
                'feature_fraction': feature_fraction,
                'bagging_fraction': bagging_fraction,
                'bagging_freq': bagging_freq,
                'scale_pos_weight': scale_pos_weight,
                'boosting_type': boosting_type,
                'drop_rate': drop_rate

            }
        else:
            param_dict = {
                'num_leaves': num_leaves,
                'max_depth': max_depth,
                'learning_rate': learning_rate,
                'n_estimators': n_estimators,
                'min_child_samples': min_child_samples,
                'min_child_weight': min_child_weight,
                'subsample': subsample,
                'colsample_bytree': colsample_bytree,
                'reg_alpha': reg_alpha,
                'reg_lambda': reg_lambda,
                'max_bin': max_bin,
                'feature_fraction': feature_fraction,
                'bagging_fraction': bagging_fraction,
                'bagging_freq': bagging_freq,
                'scale_pos_weight': scale_pos_weight,
                'boosting_type': boosting_type, }
        results = train_model(param_dict, X, y)
        avg_f1 = results['avg_f1']
        avg_precision = results['avg_precision']
        avg_logloss = results['avg_logloss']
        avg_auc = results['avg_auc']


        alpha = .3
        combined_metric = (alpha * (1 - avg_f1)) + ((1 - alpha) * (1 - avg_precision))
        return avg_auc
    except Exception as e:
        # Handle the exception (e.g., print an error message)
        print(f"Trial failed with exception: {str(e)}")
        # Return a value indicating trial failure (e.g., -1)
        return None
###TODOI found a solution. If I remove the num_iteration from the parameter dict, and directly use num_boost_round in lightgbm.train it don't complain.
##TODO Comment out to skip the hyperparameter selection.  Swap "best_params".
while True:
    try:
        study = optuna.load_study(study_name='SPY_best_auc_a_GOSS_3hr35percent',
                                  storage='sqlite:///SPY_best_auc_a_GOSS_3hr35percent.db')
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
        study = optuna.create_study(direction="maximize", study_name='SPY_best_auc_a_GOSS_3hr35percent',
                                    storage='sqlite:///SPY_best_auc_a_GOSS_3hr35percent'
                                            '.db')
    "Keyerror, new optuna study created."  #
    #TODO add a second loop of test, wehre if it doesnt achieve x score, the trial fails.)
    try:
        study.optimize(objective, n_trials=1000,)#callbacks=[early_stopping_opt]

    except EarlyStoppingExceeded:
        print(f'EarlyStopping Exceeded: No new best scores on iters {OPTUNA_EARLY_STOPPING}')

    best_params = study.best_params
    # show failed trials
    # for trial in study.trials:
    #     if trial.value is None:
    #         print("Trial failed with NaN:", trial.params)

    # best_params ={'bagging_fraction': 0.7468662275929627, 'bagging_freq': 1, 'boosting_type': 'gbdt', 'colsample_bytree': 0.6148210766873191, 'feature_fraction': 0.7967793478163084, 'learning_rate': 0.010193878127533921, 'max_bin': 242, 'max_depth': 40, 'min_child_samples': 5, 'min_child_weight': 0.0008649048921128245, 'n_estimators': 814, 'num_leaves': 130, 'reg_alpha': 0.16115108601553896, 'reg_lambda': 6.788655218754687e-05, 'scale_pos_weight': 8.567473838835983, 'subsample': 0.7913698005108732}
    # best_params = {'batch_size': 972, 'dropout_rate': 0.23030333490770447, 'gamma': 0.3089135336987861, 'l1_lambda': 0.0950910207258489, 'learning_rate': 1.5716591458439578e-05, 'lr_scheduler': 'StepLR', 'num_epochs': 153, 'num_hidden_units': 2520, 'num_layers': 5, 'optimizer': 'Adam', 'step_size': 45, 'weight_decay': 0.09863209738072187}
    #####################################################################################################
    # ################

    full_data = lgb.Dataset(X.to_numpy(), label=y.to_numpy())
    if best_params['boosting_type'] == 'goss':
        # If boosting type is 'goss', remove bagging-related parameters
        best_params.pop('bagging_fraction', None)
        best_params.pop('bagging_freq', None)

    final_lgb_classifier = lgb.train(
        params=best_params,  # These should be the best parameters you found
        train_set=full_data,
        num_boost_round=best_params['n_estimators'] # Or however many you deem fit
    )

    print("~~~~training model using best params.~~~~")
    # best_params["boosting_type"] = "gbdt"#TODO remove this

    results = train_model(best_params, X, y, final_classifier=final_lgb_classifier)

    trained_model = results['best_model']


    # plt.barh(range(X.shape[1]), feature_importances[sorted_idx])
    # plt.yticks(range(X.shape[1]), X.columns[sorted_idx])
    # plt.xlabel("Random Forest Feature Importance")
    # plt.show()
    # #
    # selector = SelectFromModel(RandomForestClassifier, threshold=0.1)
    # X_new = selector.transform(X_test)
    # Now use trained_rf to predict on your test data
    if results['best_model'] == None:
        continue
    else:
        # X_test_scaled = scaler_X_trainval.transform(X_test)
        threshold = 0.5
        y_test_pred = trained_model.predict(X_test)
        # print(y_test_pred[0],y_test_pred[10],y_test_pred[20],y_test_pred[-1])
        binary_predictions = (y_test_pred > threshold).astype(int)

        # indexes = np.where(y_test == 1)[0]
        # values_at_indexes = y_test_pred[indexes]
        # print(values_at_indexes)

        # Compute metrics
        f1 = f1_score(y_test, binary_predictions)
        precision = precision_score(y_test, binary_predictions)
        recall = recall_score(y_test, binary_predictions)
        accuracy = accuracy_score(y_test, binary_predictions)
        if f1 < 0.75 and precision < .75:
            print("Test F1 score and prec. are not above 0.75. Restarting optimization.")
            print("Test Metrics:")
            print("F1:", f1)
            print("Precision:", precision)

            print("Recall:", recall)
            print("Accuracy:", accuracy)
            continue  # Restart the optimization loop
        else:
            print("Test F1 score is above 0.8. Optimization complete.")
            feature_importances = trained_model.feature_importance(
                importance_type='split')  # or 'gain' for gain-based importance

            sorted_idx = np.argsort(feature_importances)
            sorted_features = X.columns[sorted_idx]

            plt.figure(figsize=(10, 15))  # Increased from (10, 8) to (10, 15)
            plt.barh(range(X.shape[1]), feature_importances[sorted_idx])
            plt.yticks(range(X.shape[1]), sorted_features, fontsize=8)
            plt.xlabel("Feature Importance")
            plt.title("Feature Importances")
            plt.show()
            sorted_idx = np.argsort(feature_importances)
            break  # Exi

    # Print results
    print("Test Metrics:")
    print("F1:", f1)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Accuracy:", accuracy)

    # Print Number of Positive and Negative Samples
    num_positive_samples = sum(y_test)
    # num_negative_samples_up = len(y_up_test_tensor) - num_positive_samples_up

    # print("Number of Total Samples(Target_Up):", num_positive_samples_up + num_negative_samples_up)


    input_val = input("Would you like to save these models? y/n: ").upper()
    if input_val == "Y":
        model_summary = input("Save this set of models as: ")
        model_directory = os.path.join("../../../Trained_Models", f"{model_summary}")
        if not os.path.exists(model_directory):
            os.makedirs(model_directory)

        # Saving using joblib
        model_filename_up = os.path.join(model_directory, "target_up_rf_model.joblib")
        dump(trained_model, model_filename_up)

        # Save other info
        with open(f"../../../Trained_Models/{model_summary}/info.txt", "w") as info_txt:
            info_txt.write("This file contains information about the model.\n\n")
            info_txt.write(f"File analyzed: {DF_filename}\n\n")
            info_txt.write(f"Metrics:\nPrecision: {precision}\nAccuracy: {accuracy}\nRecall: {recall}\nF1-Score: {f1}\n")
            info_txt.write(f"Predictors: {Chosen_Predictor}\n\n\nBest Params: {best_params}\n")
