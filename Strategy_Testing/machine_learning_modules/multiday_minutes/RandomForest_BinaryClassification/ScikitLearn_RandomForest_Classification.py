from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import datetime
import os
from datetime import datetime
from joblib import dump
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel

import numpy as np
import optuna
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import RobustScaler
from torchmetrics import Accuracy, Precision, Recall, F1Score

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
Chosen_Predictor = ['ExpDate', 'LastTradeTime', 'Current Stock Price',
                    'Current SP % Change(LAC)', 'Maximum Pain', 'Bonsai Ratio',
                    'Bonsai Ratio 2', 'B1/B2', 'B2/B1', 'PCR-Vol', 'PCR-OI',
                    'PCRv @CP Strike', 'PCRoi @CP Strike', 'PCRv Up1', 'PCRv Up2',
                    'PCRv Up3', 'PCRv Up4', 'PCRv Down1', 'PCRv Down2', 'PCRv Down3',
                    'PCRv Down4', 'PCRoi Up1', 'PCRoi Up2', 'PCRoi Up3', 'PCRoi Up4',
                    'PCRoi Down1', 'PCRoi Down2', 'PCRoi Down3', 'PCRoi Down4',
                    'ITM PCR-Vol', 'ITM PCR-OI', 'ITM PCRv Up1', 'ITM PCRv Up2',
                    'ITM PCRv Up3', 'ITM PCRv Up4', 'ITM PCRv Down1', 'ITM PCRv Down2',
                    'ITM PCRv Down3', 'ITM PCRv Down4', 'ITM PCRoi Up1', 'ITM PCRoi Up2',
                    'ITM PCRoi Up3', 'ITM PCRoi Up4', 'ITM PCRoi Down1', 'ITM PCRoi Down2',

                    'ITM PCRoi Down3', 'ITM PCRoi Down4', 'ITM OI', 'Total OI',
                    'ITM Contracts %', 'Net_IV', 'Net ITM IV', 'Net IV MP', 'Net IV LAC',
                    'NIV Current Strike', 'NIV 1Higher Strike', 'NIV 1Lower Strike',
                    'NIV 2Higher Strike', 'NIV 2Lower Strike', 'NIV 3Higher Strike',
                    'NIV 3Lower Strike', 'NIV 4Higher Strike', 'NIV 4Lower Strike',
                    'NIV highers(-)lowers1-2', 'NIV highers(-)lowers1-4',
                    'NIV 1-2 % from mean', 'NIV 1-4 % from mean', 'Net_IV/OI',
                    'Net ITM_IV/ITM_OI', 'Closest Strike to CP', 'RSI', 'AwesomeOsc',
                    'RSI14', 'RSI2', 'AwesomeOsc5_34']
# ##had highest corr for 3-5 hours with these:
# Chosen_Predictor = ['Bonsai Ratio','Bonsai Ratio 2','ITM PCR-Vol','ITM PCRoi Up1', 'RSI14','AwesomeOsc5_34', 'Net_IV']
ml_dataframe = pd.read_csv(DF_filename)
print(ml_dataframe.columns)
# ##had highest corr for 3-5 hours with these:
# Chosen_Predictor = ['Bonsai Ratio','Bonsai Ratio 2','ITM PCR-Vol','ITM PCRoi Up1', 'RSI14','AwesomeOsc5_34', 'Net_IV']
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

X_test = X[split_index:]
y_test = y[split_index:]
X = X[:split_index]
y = y[:split_index]
# Fit the scaler on the entire training data
# scaler_X_trainval = RobustScaler().fit(X)
# scaler_y_trainval = RobustScaler().fit(y.values.reshape(-1, 1))


def train_model(trial, X, y):
    # Extract hyperparameters from trial object
    n_estimators = trial.suggest_int("n_estimators", 10, 1000)
    max_depth = trial.suggest_int("max_depth", 2, 50, log=True)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 15)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 15)
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2"])
    criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
    bootstrap = trial.suggest_categorical("bootstrap", [True, False])
    class_weight = trial.suggest_categorical("class_weight", [None, "balanced", "balanced_subsample"])
    # warm_start = trial.suggest_categorical("warm_start", [True, False])

    best_model = None
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    X_np = X.to_numpy()

    best_f1, best_precision, best_recall = 0, 0, 0
    total_f1, total_precision, total_recall = 0, 0, 0
    num_folds = 0

    for train_index, val_index in kf.split(X_np):
        X_train, X_val = X_np[train_index], X_np[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Fit the scaler on the training data
        # scaler_X_fold = RobustScaler().fit(X_train)
        # X_train_scaled = scaler_X_fold.transform(X_train)
        # X_val_scaled = scaler_X_fold.transform(X_val)

        # Create and train the RandomForest model
        rf_classifier = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
            max_features=max_features, criterion=criterion, bootstrap=bootstrap,
            class_weight=class_weight, warm_start=False, random_state=0
        )
        rf_classifier.fit(X_train, y_train)

        # Predict on validation data
        y_pred = rf_classifier.predict(X_val)

        # Compute metrics
        f1 = f1_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)

        # Update best scores
        if f1 > best_f1:
            best_f1 = f1
            best_precision = precision
            best_recall = recall
            best_model = rf_classifier
        total_f1 += f1
        total_precision += precision
        total_recall += recall
        num_folds += 1

    avg_f1 = total_f1 / num_folds
    avg_precision = total_precision / num_folds
    avg_recall = total_recall / num_folds

    return {
        'avg_f1': avg_f1,
        'avg_precision': avg_precision,
        'avg_recall': avg_recall,
        'best_model': best_model
    }


# Define Optuna Objective
def objective(trial):
    print(datetime.now())

    results = train_model(trial, X, y)
    avg_f1 = results['avg_f1']
    avg_precision = results['avg_precision']

    print("best f1 score: ", avg_f1, "best precision score: ", avg_precision, )

    alpha = .3
    combined_metric = (alpha * (1 - avg_f1)) + ((1 - alpha) * (1 - avg_precision))
    return avg_f1  # Optuna will try to minimize this value


##TODO Comment out to skip the hyperparameter selection.  Swap "best_params".
try:
    study = optuna.load_study(study_name='SPY_avg_f1_allfeatures_RF_class_3hr35percent',
                              storage='sqlite:///SPY_avg_f1_allfeatures_RF_class_3hr35percent.db')
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
    study = optuna.create_study(direction="maximize", study_name='SPY_avg_f1_allfeatures_RF_class_3hr35percent',
                                storage='sqlite:///SPY_avg_f1_allfeatures_RF_class_3hr35percent.db')
"Keyerror, new optuna study created."  #

study.optimize(objective, n_trials=5000)
best_params = study.best_params
#
# best_params ={'batch_size': 1197, 'dropout_rate': 0.4608394623321738, 'l1_lambda': 0.01320220981011121, 'learning_rate': 1.1625919878731402e-05, 'lr_scheduler': 'ReduceLROnPlateau', 'lrpatience': 10, 'num_epochs': 211, 'num_hidden_units': 114, 'num_layers': 5, 'optimizer': 'RMSprop', 'weight_decay': 0.00013649093677743602}
# best_params = {'batch_size': 972, 'dropout_rate': 0.23030333490770447, 'gamma': 0.3089135336987861, 'l1_lambda': 0.0950910207258489, 'learning_rate': 1.5716591458439578e-05, 'lr_scheduler': 'StepLR', 'num_epochs': 153, 'num_hidden_units': 2520, 'num_layers': 5, 'optimizer': 'Adam', 'step_size': 45, 'weight_decay': 0.09863209738072187}
#####################################################################################################
# ################
n_estimators = best_params["n_estimators"]
max_depth = best_params["max_depth"]
min_samples_split = best_params["min_samples_split"]
min_samples_leaf = best_params["min_samples_leaf"]
max_features = best_params["max_features"]

print("~~~~training model using best params.~~~~")
results = train_model(best_params, X, y)


trained_rf = results['best_model']
feature_importances = trained_rf.feature_importances_

sorted_idx = np.argsort(feature_importances)

plt.barh(range(X.shape[1]), feature_importances[sorted_idx])
plt.yticks(range(X.shape[1]), X.columns[sorted_idx])
plt.xlabel("Random Forest Feature Importance")
plt.show()
#
# selector = SelectFromModel(RandomForestClassifier, threshold=0.1)
# X_new = selector.transform(X_test)
# Now use trained_rf to predict on your test data
# X_test_scaled = scaler_X_trainval.transform(X_test)

y_pred = trained_rf.predict(X_test)

# Compute metrics
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)

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
    dump(trained_rf, model_filename_up)

    # Save other info
    with open(f"../../../Trained_Models/{model_summary}/info.txt", "w") as info_txt:
        info_txt.write("This file contains information about the model.\n\n")
        info_txt.write(f"File analyzed: {DF_filename}\n\n")
        info_txt.write(f"Metrics:\nPrecision: {precision}\nAccuracy: {accuracy}\nRecall: {recall}\nF1-Score: {f1}\n")
        info_txt.write(f"Predictors: {Chosen_Predictor}\n\n\nBest Params: {best_params}\n")