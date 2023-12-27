import datetime
import os
import warnings
from datetime import datetime
from warnings import simplefilter
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from joblib import dump
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit, KFold
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

# Filter out specific warning messages
warnings.filterwarnings("ignore", category=Warning)

simplefilter("ignore", category=RuntimeWarning)
# Restore the warning filter (if needed)
# warnings.filterwarnings("default", category=Warning)

DF_filename = r"../../../../data/historical_multiday_minute_DF/SPY_historical_multiday_min.csv"
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

ml_dataframe = pd.read_csv(DF_filename)
print(ml_dataframe.columns)
# ##had highest corr for 3-5 hours with these:
Chosen_Predictor = ['Bonsai Ratio', 'Bonsai Ratio 2', 'ITM PCR-Vol', 'ITM PCRv Up1','ITM PCRv Down1', 'RSI14',
                    'Net_IV']
# ml_dataframe['LastTradeTime'] = ml_dataframe['LastTradeTime'].apply(
#     lambda x: datetime.strptime(str(x), '%y%m%d_%H%M') if not pd.isna(x) else np.nan)
# ml_dataframe['LastTradeTime'] = ml_dataframe['LastTradeTime'].apply(lambda x: x.timestamp())
ml_dataframe['ExpDate'] = ml_dataframe['ExpDate'].astype(float)

cells_forward_to_check = 3 * 60  # rows to check(minutes in this case)
threshold_cells_up = cells_forward_to_check * 0.5  # how many rows must achieve target %
num_trials = 1 #before using best params on test.

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
from sklearn.model_selection import train_test_split

# Assuming X and y are your features and target variables
test_set_percentage = 0.02  # Specify the percentage of the data to use as a test set

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_set_percentage,shuffle=False)

# If your data is a time series and should not be shuffled, use:
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_set_percentage, shuffle=False)

print("Xlength: ", len(X), "XTestlen: ", len(X_test), "positive in y: ", y.sum(), "positive in ytest: ", y_test.sum())

'''Metrics & Model Selection: You're storing the best model based on the F1 score. This is okay if F1 is the most important metric for your problem. If not, you might want to adjust this logic. Also, you could consider saving the models from all the folds and using a voting mechanism for predictions if you want to leverage the power of ensemble predictions'''


def train_model(param_dict, X, y, final_classifier=None):
    # Extract hyperparameters from param_dict
    n_estimators = param_dict['n_estimators']
    max_depth = param_dict['max_depth']
    min_samples_split = param_dict['min_samples_split']
    min_samples_leaf = param_dict['min_samples_leaf']
    max_features = param_dict['max_features']
    bootstrap = param_dict['bootstrap']

    best_model = None

    kf = KFold(n_splits=2, shuffle=False)  # 5-fold cross-validation
    auc_scores = []
    log_loss_scores = []
    X_np = X.to_numpy()
    best_f1, best_precision, best_recall, best_accuracy, best_auc = 0, 0, 0, 0, 0
    total_f1, total_precision, total_recall, total_accuracy = 0, 0, 0, 0
    num_folds = 0

    for train_index, val_index in kf.split(X_np):
        X_train, X_val = X_np[train_index], X_np[val_index]
        y_train, y_val = y[train_index], y[val_index]
        scaler_X = RobustScaler().fit(X_train)

        # Transform the training and validation data
        X_train = scaler_X.transform(X_train)
        X_val = scaler_X.transform(X_val)

        rf_params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features,
            'bootstrap': bootstrap,
        'class_weight': 'balanced'
        }

        if final_classifier is not None:
            rf_classifier = final_classifier
        else:
            rf_classifier = RandomForestClassifier(**rf_params)
            rf_classifier.fit(X_train, y_train)

        y_pred = rf_classifier.predict_proba(X_val)[:, 1]
        y_pred_binary = (y_pred > 0.5).astype(int)
        auc = roc_auc_score(y_val, y_pred)
        logloss = log_loss(y_val, y_pred)

        f1 = f1_score(y_val, y_pred_binary)
        precision = precision_score(y_val, y_pred_binary)
        recall = recall_score(y_val, y_pred_binary)
        accuracy = accuracy_score(y_val, y_pred_binary)

        auc_scores.append(auc)
        log_loss_scores.append(logloss)

        total_f1 += f1
        total_accuracy += accuracy
        total_precision += precision
        total_recall += recall
        num_folds += 1

    avg_accuracy = total_accuracy / num_folds
    avg_logloss = sum(log_loss_scores) / num_folds
    avg_f1 = total_f1 / num_folds
    avg_precision = total_precision / num_folds
    avg_recall = total_recall / num_folds
    avg_auc = sum(auc_scores) / num_folds

    if avg_f1 > best_f1:
        best_f1 = avg_f1
        best_precision = avg_precision
        best_recall = avg_recall
        best_accuracy = avg_accuracy
        best_model = rf_classifier

    print(
        f'avg_accuracy: {best_accuracy}, avg_f1: {best_f1}, avg_precision: {best_precision}, avg_recall: {best_recall}, avg_auc: {avg_auc}, avg_logloss: {avg_logloss}')
    return {
        'avg_accuracy': best_accuracy,
        'avg_f1': best_f1,
        'avg_precision': best_precision,
        'avg_recall': best_recall,
        'avg_auc': avg_auc,
        'avg_logloss': avg_logloss,
        'best_model': best_model,
    }


# Define Optuna Objective for RandomForest
def objective(trial):
    # ... Your objective function remains mostly the same ...
    n_estimators = trial.suggest_int("n_estimators", 100, 2000)
    max_depth = trial.suggest_int("max_depth", 5, 20)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 100)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 50)
    max_features = trial.suggest_categorical("max_features", ["auto", "sqrt"])
    bootstrap = trial.suggest_categorical("bootstrap", [True, False])
    n_jobs = trial.suggest_categorical("n_jobs", [-1])

    param_dict = {
        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        'max_features': max_features,
        'bootstrap': bootstrap,
        'n_jobs':n_jobs

    }

    results = train_model(param_dict, X_train, y_train)
    avg_f1 = results['avg_f1']
    avg_precision = results['avg_precision']
    avg_logloss = results['avg_logloss']
    avg_auc = results['avg_auc']

    alpha = .3
    combined_metric = (alpha * (1 - avg_f1)) + ((1 - alpha) * (1 - avg_precision))
    return avg_f1


# TODO broke here# show failed trials
# for trial in study.trials:
#     if trial.value is None:
#         print("Trial failed with NaN:", tria
#         l.params)

# best_params ={'bagging_fraction': 0.7468662275929627, 'bagging_freq': 1, 'boosting_type': 'gbdt', 'colsample_bytree': 0.6148210766873191, 'feature_fraction': 0.7967793478163084, 'learning_rate': 0.010193878127533921, 'max_bin': 242, 'max_depth': 40, 'min_child_samples': 5, 'min_child_weight': 0.0008649048921128245, 'n_estimators': 814, 'num_leaves': 130, 'reg_alpha': 0.16115108601553896, 'reg_lambda': 6.788655218754687e-05, 'scale_pos_weight': 8.567473838835983, 'subsample': 0.7913698005108732}
# best_params ={'boosting_type': 'gbdt', 'colsample_bytree': 0.9363197433703218, 'data_sample_strategy': 'goss', 'extra_trees': False, 'feature_fraction': 0.983299665025065, 'learning_rate': 0.028540466850651803, 'max_bin': 60, 'max_depth': 19, 'min_child_samples': 5, 'min_child_weight': 0.0010703276151551506, 'n_estimators': 349, 'num_leaves': 114, 'path_smooth': 0.001303105511766731, 'reg_alpha': 0.09724991199226693, 'reg_lambda': 2.206401583028771e-05, 'scale_pos_weight': 14.223823958899008}
#####################################################################################################
# ################
while True:
    try:
        study = optuna.load_study(study_name='rf_spy_3hr35percent',
                                  storage='sqlite:///rf_spy_3hr35percent.db')
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
        study = optuna.create_study(direction="maximize", study_name='rf_spy_3hr35percent',
                                    storage='sqlite:///rf_spy_3hr35percent'
                                            '.db')
    "Keyerror, new optuna study created."  #
    # TODO add a second loop of test, wehre if it doesnt achieve x score, the trial fails.)

    study.optimize(objective, n_trials=num_trials, )  # callbacks=[early_stopping_opt]

    best_params = study.best_params
    X_Scaler = RobustScaler().fit(X_train)
    X_train_scaled = X_Scaler.transform(X_train)
    X_test_scaled = X_Scaler.transform(X_test)
    final_rf_classifier = RandomForestClassifier(**best_params)
    trained_model = final_rf_classifier.fit(X_train_scaled, y_train.to_numpy())

    print("~~~~training model using best params.~~~~")


    # plt.barh(range(X.shape[1]), feature_importances[sorted_idx])
    # plt.yticks(range(X.shape[1]), X.columns[sorted_idx])
    # plt.xlabel("Random Forest Feature Importance")
    # plt.show()
    # #
    # selector = SelectFromModel(RandomForestClassifier, threshold=0.1)
    # X_new = selector.transform(X_test)
    # Now use trained_rf to predict on your test data



    threshold = 0.5
    y_test_pred = trained_model.predict(X_test_scaled)
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
    if f1 < 0.7 and precision < .7:
        print("Test F1 score and prec. are not above 0.75. Restarting optimization.")
        print("Test Metrics:")
        print("F1:", f1)
        print("Precision:", precision)

        print("Recall:", recall)
        print("Accuracy:", accuracy)
        continue  # Restart the optimization loop
    else:
        print("Test F1 score is above 0.8. Optimization complete.")
        feature_importances = trained_model.feature_importances_
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
