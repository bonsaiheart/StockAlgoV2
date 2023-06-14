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

# Replace positive infinity with a large finite value

processed_dir = "../dailyDF"
ticker = "SPY"
print(ticker)
list_of_df = []
ticker_dir = os.path.join(processed_dir, ticker)


DF_filename = ("../historical_minute_DF/SPY/230612_SPY.csv")
ml_dataframe = pd.read_csv(DF_filename)

Chosen_Timeframe = "1 hour later change %"
Chosen_Predictor = ['Bonsai Ratio','Bonsai Ratio 2','B1/B2','B2/B1','ITM PCR-Vol','ITM PCR-OI','ITM PCRv Up2','ITM PCRv Down2','ITM PCRoi Up2','ITM PCRoi Down2','Net_IV','Net ITM IV','NIV 2Higher Strike','NIV 2Lower Strike','NIV highers(-)lowers1-4','NIV 1-4 % from mean','RSI','AwesomeOsc']
# Chosen_Predictor = ['Bonsai Ratio']
threshold_up = 0.6
threshold_down = 0.6
percent_up = 0.2
percent_down = -0.2
# num_best_features = 1
ml_dataframe.dropna(subset=[Chosen_Timeframe] + Chosen_Predictor, inplace=True)

num_rows = len(ml_dataframe[Chosen_Timeframe].dropna())
ml_dataframe.dropna(thresh=num_rows, axis=1, inplace=True)
ml_dataframe = ml_dataframe.astype(int)
threshold_up_formatted = int(threshold_up * 10)
threshold_down_formatted = int(threshold_down * 10)

Chosen_Predictor_nobrackets = [x.replace('/', '').replace(',', '_').replace(' ', '_').replace('-', '') for x in
                               Chosen_Predictor]
Chosen_Predictor_formatted = "_".join(Chosen_Predictor_nobrackets)

Chosen_Timeframe_formatted = Chosen_Timeframe.replace(' ', '_').strip('%').replace(' ', '_').replace('%', '')


ml_dataframe["Target_Up"] = (ml_dataframe[Chosen_Timeframe] > percent_up).astype(int)
ml_dataframe["Target_Down"] = (ml_dataframe[Chosen_Timeframe] < percent_down).astype(int)


model = RandomForestClassifier(random_state=None)

parameters = {
    'n_estimators': (40,80,100,200, 300,500,1000
                     ),
    'min_samples_split': (20,50,100, 150,200,500),
    'max_depth': ( 5,10,20,50,100,150,200,250,500),
}

X = ml_dataframe[Chosen_Predictor]
# Reset the index of your DataFrame
print(X.head())
X.reset_index(drop=True, inplace=True)
print(type(ml_dataframe.columns))
# Replace positive infinity with a large finite value
# X = X.mask(np.isinf(X), sys.float_info.max)
# X = X.mask(np.isneginf(X), -sys.float_info.max)
y_up = ml_dataframe["Target_Up"]
y_down = ml_dataframe["Target_Down"]
print(type(ml_dataframe.Target_Down))
X_train, X_test, y_up_train, y_up_test, y_down_train, y_down_test = train_test_split(X, y_up, y_down, test_size=0.25,    random_state=1)
# Feature selection for Target_Up
feature_selector_up = RFECV(estimator=model, scoring='precision', step=1)

X_train_selected_up = feature_selector_up.fit_transform(X_train, y_up_train)


X_test_selected_up = feature_selector_up.transform(X_test)



# Feature selection for Target_Down
feature_selector_down = RFECV(estimator=model, scoring='precision', step=1)
X_train_selected_down = feature_selector_down.fit_transform(X_train, y_down_train)
X_test_selected_down = feature_selector_down.transform(X_test)

...

tscv = TimeSeriesSplit(n_splits=3)

bayes_search_up = BayesSearchCV(
    estimator=model,
    search_spaces=parameters,
    scoring='precision',
    cv=tscv,
    n_iter=50,  # Number of iterations for the Bayesian optimization
    random_state=1
)
# grid_search_up = GridSearchCV(estimator=model, param_grid=parameters, cv=tscv, scoring='precision')
# print("Performing GridSearchCV UP...")
# grid_search_up.fit(X_train_selected_up, y_up_train)
# best_features_up = [Chosen_Predictor[i] for i in feature_selector_up.get_support(indices=True)]
# print("Best features for Target_Up:", best_features_up)
#
# print("Best parameters for Target_Up:", grid_search_up.best_params_)
# print("Best score for Target_Up:", grid_search_up.best_score_)
# best_param_up = f"Best parameters for Target_Up: {grid_search_up.best_params_}. Best precision: {grid_search_up.best_score_}"
# model_up = grid_search_up.best_estimator_
print("Performing BayesSearchCV UP...")
# X_train_selected_up[np.isinf(X_train_selected_up) & (X_train_selected_up > 0)] = sys.float_info.max
# X_train_selected_up[np.isinf(X_train_selected_up) & (X_train_selected_up < 0)] = -sys.float_info.max
X_train_selected_up = X_train_selected_up.astype(int)

bayes_search_up.fit(X_train_selected_up, y_up_train)
best_features_up = [Chosen_Predictor[i] for i in feature_selector_up.get_support(indices=True)]
print("Best features for Target_Up:", best_features_up)
print("Best parameters for Target_Up:", bayes_search_up.best_params_)
print("Best score for Target_Up:", bayes_search_up.best_score_)
best_param_up = f"Best parameters for Target_Up: {bayes_search_up.best_params_}. Best precision: {bayes_search_up.best_score_}"
model_up = bayes_search_up.best_estimator_
importance_tuples = [(feature, importance) for feature, importance in zip(Chosen_Predictor, model_up.feature_importances_)]
importance_tuples = sorted(importance_tuples, key=lambda x: x[1], reverse=True)

for feature, importance in importance_tuples:
    print(f"{feature}: {importance}")

...

# grid_search_down = GridSearchCV(estimator=model, param_grid=parameters, cv=tscv, scoring='precision')
# print("Performing GridSearchCV DOWN...")
# grid_search_down.fit(X_train_selected_down, y_down_train)
# best_features_down = [Chosen_Predictor[i] for i in feature_selector_down.get_support(indices=True)]
# print("Best features for Target_Down:", best_features_down)
#
# print("Best parameters for Target_Down:", grid_search_down.best_params_)
# print("Best score for Target_Down:", grid_search_down.best_score_)
# best_param_down = f"Best parameters for Target_Down: {grid_search_down.best_params_}. Best precision: {grid_search_down.best_score_}"
# model_down = grid_search_down.best_estimator_
bayes_search_down = BayesSearchCV(
    estimator=model,
    search_spaces=parameters,
    scoring='precision',
    cv=tscv,
    n_iter=50,  # Number of iterations for the Bayesian optimization
    random_state=1
)

print("Performing BayesSearchCV DOWN...")
bayes_search_down.fit(X_train_selected_down, y_down_train)
best_features_down = [Chosen_Predictor[i] for i in feature_selector_down.get_support(indices=True)]
print("Best features for Target_Down:", best_features_down)

print("Best parameters for Target_Down:", bayes_search_down.best_params_)
print("Best score for Target_Down:", bayes_search_down.best_score_)
best_param_down = f"Best parameters for Target_Down: {bayes_search_down.best_params_}. Best precision: {bayes_search_down.best_score_}"
model_down = bayes_search_down.best_estimator_
importance_tuples = [(feature, importance) for feature, importance in zip(Chosen_Predictor, model_down.feature_importances_)]
importance_tuples = sorted(importance_tuples, key=lambda x: x[1], reverse=True)

for feature, importance in importance_tuples:
    print(f"{feature}: {importance}")

...

# Use the selected features for prediction
predicted_probabilities_up = model_up.predict_proba(X_test_selected_up)
predicted_probabilities_down = model_down.predict_proba(X_test_selected_down)
...


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

print("Metrics for Target_Up:")
print("Precision:", precision_up)
print("Accuracy:", accuracy_up)
print("Recall:", recall_up)
print("F1-Score:", f1_up)

print("Metrics for Target_Down:")
print("Precision:", precision_down)
print("Accuracy:", accuracy_down)
print("Recall:", recall_down)
print("F1-Score:", f1_down)


cv_scores_up = cross_val_score(model_up, X_test_selected_up, y_up_test, cv=tscv)
cv_scores_down = cross_val_score(model_down, X_test_selected_down, y_down_test, cv=tscv)

print("Cross-validation scores for Target_Up:", cv_scores_up)
print("Mean cross-validation score for Target_Up:", cv_scores_up.mean())
print("Cross-validation scores for Target_Down:", cv_scores_down)
print("Mean cross-validation score for Target_Down:", cv_scores_down.mean())


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




# Load the new data
new_data_filename = "../historical_minute_DF/TSLA/230608_TSLA.csv"
new_data = pd.read_csv(new_data_filename)

# Preprocess the new data (apply the same preprocessing steps as on training data)
# ...
new_data.dropna(subset=[Chosen_Timeframe] + Chosen_Predictor, inplace=True)

num_rows = len(new_data[Chosen_Timeframe].dropna())
new_data.dropna(thresh=num_rows, axis=1, inplace=True)
# Select the relevant features for the new datan
X_new = new_data[Chosen_Predictor]

X_new_selected_up = feature_selector_up.transform(X_new)
X_new_selected_down = feature_selector_down.transform(X_new)

# Make predictions on the new data
predicted_probabilities_up_new = model_up.predict_proba(X_new_selected_up)
predicted_probabilities_down_new = model_down.predict_proba(X_new_selected_down)

# Apply the threshold to obtain binary predictions
predicted_up_new = (predicted_probabilities_up_new[:, 1] > threshold_up).astype(int)
predicted_down_new = (predicted_probabilities_down_new[:, 1] > threshold_down).astype(int)
new_data["Target_Up"] = (new_data[Chosen_Timeframe] > percent_up).astype(int)
new_data["Target_Down"] = (new_data[Chosen_Timeframe] < percent_down).astype(int)

# Evaluate the predictions if actual target labels are available
if "Target_Up" in new_data.columns and "Target_Down" in new_data.columns:
    y_up_new = new_data["Target_Up"]
    y_down_new = new_data["Target_Down"]

    precision_up_new = precision_score(y_up_new, predicted_up_new)
    accuracy_up_new = accuracy_score(y_up_new, predicted_up_new)
    recall_up_new = recall_score(y_up_new, predicted_up_new)
    f1_up_new = f1_score(y_up_new, predicted_up_new)

    precision_down_new = precision_score(y_down_new, predicted_down_new)
    accuracy_down_new = accuracy_score(y_down_new, predicted_down_new)
    recall_down_new = recall_score(y_down_new, predicted_down_new)
    f1_down_new = f1_score(y_down_new, predicted_down_new)

    print("Metrics for Target_Up (New Data):")
    print("Precision:", precision_up_new)
    print("Accuracy:", accuracy_up_new)
    print("Recall:", recall_up_new)
    print("F1-Score:", f1_up_new)

    print("Metrics for Target_Down (New Data):")
    print("Precision:", precision_down_new)
    print("Accuracy:", accuracy_down_new)
    print("Recall:", recall_down_new)
    print("F1-Score:", f1_down_new)

input_val = input("Would you like to save these models? y/n: ").upper()
if input_val == "Y":
    model_summary = input("Save this set of models as: ")
    model_directory = os.path.join("../Trained_Models",
                                   f'{model_summary}')
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
        print(model_directory)
    model_filename_up = os.path.join(model_directory, "target_up.joblib")
    model_filename_down = os.path.join(model_directory, "target_down.joblib")

    model_filename_up = os.path.join(model_directory, "target_up.joblib")
    model_filename_down = os.path.join(model_directory, "target_down.joblib")
    joblib.dump(model_up, model_filename_up)
    joblib.dump(model_down, model_filename_down)


    with open(
            f"../Trained_Models/{model_summary}/info.txt","w") as info_txt:
        info_txt.write("This file contains information about the model.\n\n")
        info_txt.write(
            f"File analyzed: {DF_filename}\nLookahead Target: {Chosen_Timeframe}\n\nBest parameters for Target_Up: {bayes_search_up.best_params_}. \nBest precision: {bayes_search_up.best_score_}\nBest parameters for Target_Down: {bayes_search_down.best_params_}. \nBest precision: {bayes_search_down.best_score_}\n\nMetrics for Target_Up:\nPrecision: {precision_up}\nAccuracy: {accuracy_up}\nRecall: {recall_up}\nF1-Score: {f1_up}\nCross-validation scores for Target_Up: {cv_scores_up}\nMean cross-validation score for Target_Up: {cv_scores_up.mean()}\n\nMetrics for Target_Down:\nPrecision: {precision_down}\nAccuracy: {accuracy_down}\nRecall: {recall_down}\nF1-Score: {f1_down}\nCross-validation scores for Target_Down: {cv_scores_down}\nMean cross-validation score for Target_Down: {cv_scores_down.mean()}\n\n")
        info_txt.write(
            f"Predictors: {Chosen_Predictor}\n\nBest_Predictors_Selected Up: {best_features_up}\nBest_Predictors_Selected Down: {best_features_down}\n\nThreshold Up(sensitivity): {threshold_up}\nThreshold Down(sensitivity): {threshold_down}\nTarget Underlying Percentage Up: {percent_up}\nTarget Underlying Percentage Down: {percent_down}\n")
else:
    exit()
