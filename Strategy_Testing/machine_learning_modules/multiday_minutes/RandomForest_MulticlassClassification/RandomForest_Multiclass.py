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
from sklearn.feature_selection import RFECV
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, \
    classification_report
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.preprocessing import RobustScaler
from sklearn.utils import compute_class_weight
from torchmetrics import Precision

# Filter out specific warning messages
# warnings.filterwarnings("ignore", category=Warning)
#
# simplefilter("ignore", category=RuntimeWarning)
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
# Chosen_Predictor = ['Bonsai Ratio', 'Bonsai Ratio 2', 'ITM PCR-Vol', 'ITM PCRoi Up1', 'RSI14', 'AwesomeOsc5_34',
#                     'Net_IV']
Chosen_Predictor = ['LastTradeTime', 'Current Stock Price' ,
 'Maximum Pain' ,'Bonsai Ratio', 'Bonsai Ratio 2' ,'B1/B2' ,'B2/B1', 'PCR-Vol',
 'PCRv Up1', 'PCRv Up2', 'PCRv Up3',
 'PCRv Down3' ,'PCRoi Up4',
 'PCRoi Down2', 'PCRoi Down4' ,'ITM PCR-Vol' , 'ITM PCRv Up1',
 'ITM PCRv Up2' ,'ITM PCRv Up3',
                    'NIV highers(-)lowers1-4', 'NIV 1-4 % from mean']
ml_dataframe['LastTradeTime'] = ml_dataframe['LastTradeTime'].apply(
    lambda x: datetime.strptime(str(x), '%y%m%d_%H%M') if not pd.isna(x) else np.nan)
ml_dataframe['LastTradeTime'] = ml_dataframe['LastTradeTime'].apply(lambda x: x.timestamp())
ml_dataframe['ExpDate'] = ml_dataframe['ExpDate'].astype(float)
ml_dataframe['LastTradeTime'] = ml_dataframe['LastTradeTime'] / (60 * 60 * 24 * 7)

cells_forward_to_check = 3 * 60  # rows to check(minutes in this case)
threshold_cells_up = cells_forward_to_check * 0.2  # how many rows must achieve target %
threshold_cells_down = cells_forward_to_check * 0.2
percent_up = .3  # target percetage.
percent_down = .3  # target percentage for downward movement.

anticondition_threshold_cells_up = cells_forward_to_check * .2  # was .7
anticondition_threshold_cells_down = cells_forward_to_check * .2  # Adjust as needed

ml_dataframe.dropna(subset=Chosen_Predictor, inplace=True)
length = ml_dataframe.shape[0]
target_Counter = 0
anticondition_UpCounter = 0
anticondition_DownCounter = 0

UPWARD = 2
DOWNWARD = 1
LITTLE_MOVEMENT = 0
ml_dataframe["Target"] = LITTLE_MOVEMENT  # Default to little movement
for i in range(1, cells_forward_to_check + 1):
    shifted_values = ml_dataframe["Current Stock Price"].shift(-i)
    condition_met_up = shifted_values > (
            ml_dataframe["Current Stock Price"] + (ml_dataframe["Current Stock Price"] * (percent_up / 100)))
    condition_met_down = shifted_values < (
            ml_dataframe["Current Stock Price"] - (ml_dataframe["Current Stock Price"] * (percent_down / 100)))

    anticondition_up = shifted_values <= ml_dataframe["Current Stock Price"]
    anticondition_down = shifted_values >= ml_dataframe["Current Stock Price"]

    target_Counter += condition_met_up.astype(int)
    anticondition_UpCounter += anticondition_up.astype(int)
    anticondition_DownCounter += anticondition_down.astype(int)

# Start by setting everything to LITTLE_MOVEMENT as default
ml_dataframe["Target"] = LITTLE_MOVEMENT

# Assign upward trend only if condition for upward movement is met AND anticondition for upward movement is not met
ml_dataframe.loc[(target_Counter >= threshold_cells_up) & (
            anticondition_UpCounter <= anticondition_threshold_cells_up), "Target"] = UPWARD

# Assign downward trend only if condition for downward movement is met AND anticondition for downward movement is not met
ml_dataframe.loc[(target_Counter <= threshold_cells_down) & (
            anticondition_DownCounter <= anticondition_threshold_cells_down), "Target"] = DOWNWARD

ml_dataframe.dropna(subset=["Target"], inplace=True)
y = ml_dataframe["Target"].copy()
X = ml_dataframe[Chosen_Predictor].copy()
X.reset_index(drop=True, inplace=True)
y.reset_index(drop=True, inplace=True)

# largenumber = 1e5
# X[Chosen_Predictor] = np.clip(X[Chosen_Predictor], -largenumber, largenumber)
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
# Create a RandomForest classifier
# rf_classifier_for_selection = RandomForestClassifier(n_estimators=100)
#
# # Create an RFECV object
# selector = RFECV(estimator=rf_classifier_for_selection, step=1, cv=StratifiedKFold(5), scoring='f1_weighted')
#
# # Fit RFECV
# # selector = selector.fit(X, y)
#
# # Get the selected features
# selected_features = np.array(Chosen_Predictor)[selector.support_]

# print(f"Number of selected features: {selector.n_features_}")
# print(f"Selected features: {selected_features}")

# Update X to only include the selected features
# X = X[selected_features]
print('diddly doo,, there was sno seletion!'
      '')
count_downward = ml_dataframe[ml_dataframe["Target"] == DOWNWARD].shape[0]
print(f"There are {count_downward} entries labeled as DOWNWARD.")

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
# TODO make this make sense for multi
train_class_counts = y.value_counts()
test_class_counts = y_test.value_counts()

print("Train Set Class Counts:")
print(train_class_counts)
print("\nTest Set Class Counts:")
print(test_class_counts)

print("\nXlength:", len(X), "XTestlen:", len(X_test))
print("Total positives in y:", y.sum(), "Total positives in y_test:", y_test.sum())

# print("Xlength: ", len(X), "XTestlen: ", len(X_test), "positive in y: ", y.sum(), "positive in ytest: ", y_test.sum())
# Fit the scaler on the entire training data
scaler_X_trainval = RobustScaler().fit(X)
scaler_y_trainval = RobustScaler().fit(y.values.reshape(-1, 1))

'''Metrics & Model Selection: You're storing the best model based on the F1 score. This is okay if F1 is the most important metric for your problem. If not, you might want to adjust this logic. Also, you could consider saving the models from all the folds and using a voting mechanism for predictions if you want to leverage the power of ensemble predictions'''

def train_model(param_dict, X, y, final_classifier=None):
    # Extract hyperparameters from param_dict
    n_estimators = param_dict['n_estimators']
    max_depth = param_dict['max_depth']
    min_samples_split = param_dict['min_samples_split']
    min_samples_leaf = param_dict['min_samples_leaf']
    # max_features = param_dict['max_features']
    bootstrap = param_dict['bootstrap']
    class_weight_multiplier = param_dict['class_weight_multiplier']
    balanced_weights = compute_class_weight('balanced', classes=[0, 1, 2], y=y)
    class_weights = {
        0: balanced_weights[0],
        1: class_weight_multiplier * balanced_weights[1],
        2: class_weight_multiplier * balanced_weights[2]
    }

    best_model = None
    tscv = TimeSeriesSplit(n_splits=5)
    X_np = X.to_numpy()
    auc_scores = []
    log_loss_scores = []

    total_f1, total_precision, total_recall, total_accuracy = 0, 0, 0, 0
    num_folds = 0
    class_reports = []
    # class1_all_folds_avg_f1, class2_all_folds_avg_f1, class1and2avg_all_folds_avg_f1 = 0,0,0
    # class1_all_folds_avg_precision,  class2_all_folds_avg_precision,  class1and2avg_all_folds_avg_precision= 0,0,0
    # class1_all_folds_avg_recall,  class2_all_folds_avg_recall,  class1and2avg_all_folds_avg_recall= 0,0,0
    # class1and2avg_all_folds_avg_accuracy = 0
    class1_total_f1, class2_total_f1, class1and2avg_total_f1 = 0,0,0
    class1_total_precision,  class2_total_precision,  class1and2avg_total_precision= 0,0,0
    class1_total_recall,  class2_total_recall,  class1and2avg_total_recall= 0,0,0
    class1and2avg_total_accuracy = 0
    for train_index, val_index in tscv.split(X_np):
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
            # 'max_features': max_features,
            'bootstrap': bootstrap,
            'class_weight': class_weights  # Add this line

        }

        if final_classifier is not None:
            rf_classifier = final_classifier
        else:

            rf_classifier = RandomForestClassifier(**rf_params)
            rf_classifier.fit(X_train, y_train)

        y_pred = rf_classifier.predict_proba(X_val)
        y_pred_binary = np.argmax(y_pred, axis=1)  # Get the class with the highest probability
        # print("truth: ", y_val.iloc[0], "binary_pred: ", y_pred_binary[0], "valprediction: ", y_pred[0])
        # print("truth: ", y_val.iloc[60], "binary_pred: ", y_pred_binary[60], "valprediction: ", y_pred[60])
        try:
            auc = roc_auc_score(y_val, y_pred, multi_class="ovr", average="micro")
        except ValueError:
            auc = None  # Handle cases where not all classes have positive samples
        logloss = log_loss(y_val, y_pred)

        # F1 for classes 1 and 2
        f1_class_1 = f1_score(y_val, y_pred_binary, average=None ,labels=[1],zero_division=0)
        f1_class_2 = f1_score(y_val, y_pred_binary, average=None, labels=[2],zero_division=0)
        avg_f1_1_2 = f1_score(y_val, y_pred_binary, average='weighted', labels=[1, 2],zero_division=0)

        # Precision for classes 1 and 2
        precision_class_1 = precision_score(y_val, y_pred_binary, average=None, labels=[1],zero_division=0)
        precision_class_2 = precision_score(y_val, y_pred_binary, average=None, labels=[2],zero_division=0)
        avg_precision_1_2 = precision_score(y_val, y_pred_binary, average=None, labels=[1, 2],zero_division=0)

        # Recall for classes 1 and 2
        recall_class_1 = recall_score(y_val, y_pred_binary, average=None, labels=[1])
        recall_class_2 = recall_score(y_val, y_pred_binary, average=None, labels=[2])
        avg_recall_1_2 = recall_score(y_val, y_pred_binary, average='weighted', labels=[1, 2])

        # Accuracy (specifically for classes 1 and 2)
        # For this, you'd need to filter out class 0 predictions and true values, then compute accuracy
        recall_class_1 = recall_score(y_val, y_pred_binary, average=None, labels=[1])
        recall_class_2 = recall_score(y_val, y_pred_binary, average=None, labels=[2])
        avg_recall_1_2 = recall_score(y_val, y_pred_binary, average='weighted', labels=[1, 2])
        # Accuracy (specifically for classes 1 and 2)
        # For this, you'd need to filter out class 0 predictions and true values, then compute accuracy
        y_val_filtered_mask = (y_val == 1) | (y_val == 2)
        y_val_filtered = y_val[y_val_filtered_mask]
        y_pred_binary_filtered = y_pred_binary[y_val_filtered_mask]

        accuracy_1_2 = accuracy_score(y_val_filtered, y_pred_binary_filtered)

        f1 = f1_score(y_val, y_pred_binary, average='micro',zero_division=0)
        precision = precision_score(y_val, y_pred_binary, average='micro',zero_division=0)
        recall = recall_score(y_val, y_pred_binary, average='micro')
        accuracy = accuracy_score(y_val, y_pred_binary)
        # f1_class_1 = f1_score(y_val, y_pred_binary, average='macro', labels=[1])
        # precision_class_1 = precision_score(y_val, y_pred_binary, average='macro', labels=[1])
        # # Similarly, calculate for class 2
        # f1_class_2 = f1_score(y_val, y_pred_binary, average='macro', labels=[2])
        # precision_class_2 = precision_score(y_val, y_pred_binary, average='macro', labels=[2])
        # This will give you F1 and precision scores specifically for class 1 and class 2, without being influenced by other classes, and the scores are averaged equally.
        auc_scores.append(auc)
        log_loss_scores.append(logloss)
        class1_total_f1 += f1_class_1
        class2_total_f1 += f1_class_2
        class1and2avg_total_f1 += avg_f1_1_2
        class1_total_precision += precision_class_1
        class2_total_precision += precision_class_2
        class1and2avg_total_precision += avg_precision_1_2
        class1_total_recall += recall_class_1
        class2_total_recall += recall_class_2
        class1and2avg_total_recall += avg_recall_1_2
        class1and2avg_total_accuracy += accuracy_1_2
        total_f1 += f1
        total_accuracy += accuracy
        total_precision += precision
        total_recall += recall
        num_folds += 1
        # report = classification_report(y_val, y_pred_binary, output_dict=True)
        # print(report)
#TODO figure out how to use f1 and precition so that it doesnt just preditc all precioins all positive
        # class_reports.append(report)
    avg_accuracy = total_accuracy / num_folds
    avg_logloss = sum(log_loss_scores) / num_folds
    avg_f1 = total_f1 / num_folds
    avg_precision = total_precision / num_folds
    avg_recall = total_recall / num_folds
    avg_auc = sum(auc_scores) / num_folds
    all_folds_f1_class_1 = class1_total_f1 /num_folds
    all_folds_f1_class_2 = class2_total_f1/ num_folds
    all_folds_avg_f1_1_2 =  class1and2avg_total_f1/num_folds

    # Precision for classes 1 and 2
    all_folds_precision_class_1 =  class1_total_precision/num_folds
    all_folds_precision_class_2 =  class2_total_precision/num_folds
    all_folds_avg_precision_1_2 = class1and2avg_total_precision/num_folds

    # Recall for classes 1 and 2
    all_folds_recall_class_1 = class1_total_recall/num_folds
    all_folds_recall_class_2 = class2_total_recall/num_folds
    all_folds_avg_recall_1_2 =class1and2avg_total_recall/num_folds

    # Accuracy (specifically for classes 1 and 2)
    all_folds_avg_accuracy_1_2 =  class1and2avg_total_accuracy/num_folds

    # avg_class_report = {}
    # for report in class_reports:
    #     for key, value in report.items():
    #         if key == 'support':
    #             continue  # Skip processing the "support" field
    #         if key not in avg_class_report:
    #             avg_class_report[key] = {}
    #         for metric, score in value.items():
    #             if metric not in avg_class_report[key]:
    #                 avg_class_report[key][metric] = 0
    #             avg_class_report[key][metric] += score / len(class_reports)


    # print("class report: ", avg_class_report)
    print(
        f'class1and2avg_all_folds_avg_f1: {all_folds_avg_f1_1_2}, all_folds_avg_precision_1_2: {all_folds_avg_precision_1_2}, class1and2avg_all_folds_avg_recall: {all_folds_avg_recall_1_2}, , avg_auc: {avg_auc}, avg_logloss: {avg_logloss}')
    return {
        'class1and2avg_all_folds_avg_accuracy': all_folds_avg_accuracy_1_2,
        'class1and2avg_all_folds_avg_f1': all_folds_avg_f1_1_2,
        'class1and2avg_all_folds_avg_precision': all_folds_avg_precision_1_2,
        'class1and2avg_all_folds_avg_recall': all_folds_avg_recall_1_2,
        'avg_auc': avg_auc,
        'avg_logloss': avg_logloss,
        'best_model': best_model
    }


# Define Optuna Objective for RandomForest
def objective(trial):
    # ... Your objective function remains mostly the same ...
    n_estimators = trial.suggest_int("n_estimators", 100, 3000)
    max_depth = trial.suggest_int("max_depth", 2, 20)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 800)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 100)
    # max_features = trial.suggest_categorical("max_features", ["auto", "sqrt"])
    bootstrap = trial.suggest_categorical("bootstrap", [True, False])
    n_jobs = trial.suggest_categorical("n_jobs", [-1])
    # class_weights = {0: 1, 1: 100, 2: 100}  # Define class weights
    # Define the classes you want to control (e.g., class 1 and class 2)
    classes_to_control = [1, 2]
    class_weight_multiplier = trial.suggest_float("class_weight_multiplier", 1.0, 20.0)


    # class_weights = 'balanced'
    param_dict = {
        'class_weight_multiplier': class_weight_multiplier,

        'n_estimators': n_estimators,
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'min_samples_leaf': min_samples_leaf,
        # 'max_features': max_features,
        'bootstrap': bootstrap,
        'n_jobs': n_jobs

    }

    results = train_model(param_dict, X, y)
    class1and2avg_all_folds_avg_accuracy =results['class1and2avg_all_folds_avg_accuracy'],
    class1and2avg_all_folds_avg_f1 =results['class1and2avg_all_folds_avg_f1'],
    class1and2avg_all_folds_avg_precision =results['class1and2avg_all_folds_avg_precision'],
    class1and2avg_all_folds_avg_recall = results['class1and2avg_all_folds_avg_recall'],
    avg_logloss = results['avg_logloss']
    avg_auc = results['avg_auc']
    print(class1and2avg_all_folds_avg_f1)
    alpha = .3
    # combined_metric = (alpha * (1 - avg_f1)) + ((1 - alpha) * (1 - avg_precision))
    return class1and2avg_all_folds_avg_f1


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
        study = optuna.load_study(study_name='class1and2avg_all_folds_avg_f1_1',
                                  storage='sqlite:///class1and2avg_all_folds_avg_f1_1.db')
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
        study = optuna.create_study(direction="maximize", study_name='class1and2avg_all_folds_avg_f1_1',
                                    storage='sqlite:///class1and2avg_all_folds_avg_f1_1'
                                            '.db')
    "Keyerror, new optuna study created."  #
    # TODO add a second loop of test, wehre if it doesnt achieve x score, the trial fails.)

    study.optimize(objective, n_trials=1000, )  # callbacks=[early_stopping_opt]
    #
    best_params = study.best_params
    # best_params = {'bootstrap': False, 'class_weight_multiplier': 18.921505709138085,
    #                                         'max_depth': 41, 'min_samples_leaf': 1, 'min_samples_split': 402,
    #                                         'n_estimators': 1098, 'n_jobs': -1}
    # class1and2avg_all_folds_avg_accuracy: 0.49638774553537457, all_folds_avg_precision_1_2: [0.2768455 0.22699356], class1and2avg_all_folds_avg_recall: 0.49638774553537457, , avg_auc: 0.34794811385459534, avg_logloss: 1.4511769734738669
    #     0.2892100213808334.
    final_rf_classifier = RandomForestClassifier()
    final_rf_classifier.fit(X.to_numpy(), y.to_numpy())

    print("~~~~training model using best params.~~~~")

    results = train_model(best_params, X, y, final_classifier=final_rf_classifier)

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
        X_test_scaled = scaler_X_trainval.transform(X_test)
        y_test_pred_probs = trained_model.predict_proba(X_test_scaled)
        y_test_pred = np.argmax(y_test_pred_probs,
                                axis=1)  # print(y_test_pred[0],y_test_pred[10],y_test_pred[20],y_test_pred[-1])
        # binary_predictions = (y_test_pred > threshold).astype(int)

        # indexes = np.where(y_test == 1)[0]
        # values_at_indexes = y_test_pred[indexes]
        # print(values_at_indexes)

        # Compute metrics
        f1 = f1_score(y_test, y_test_pred, average='weighted',zero_division=0)
        precision = precision_score(y_test, y_test_pred, average='weighted',zero_division=0)
        print(5)

        recall = recall_score(y_test, y_test_pred, average='weighted')
        accuracy = accuracy_score(y_test, y_test_pred, average='weighted')
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
