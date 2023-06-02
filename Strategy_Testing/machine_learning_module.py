import joblib
import pandas as pd
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, cross_val_score
import os

processed_dir = "dailyDF"
ticker = "SPY"
print(ticker)
list_of_df = []
ticker_dir = os.path.join(processed_dir, ticker)

DF_filename = ("historical_minute_DF/SPY.csv")
ml_dataframe = pd.read_csv(DF_filename)

Chosen_Timeframe = "30 min later change %"
Chosen_Predictor = ["B1/B2","Bonsai Ratio","Maximum Pain"]
threshold_up = .8
threshold_down = .8
percent_up = .3
percent_down = -.3

ml_dataframe.dropna(subset=[Chosen_Timeframe] + Chosen_Predictor, inplace=True)

num_rows = len(ml_dataframe[Chosen_Timeframe].dropna())
ml_dataframe.dropna(thresh=num_rows, axis=1, inplace=True)
threshold_up_formatted = int(threshold_up * 10)
threshold_down_formatted = int(threshold_down * 10)
ml_dataframe = ml_dataframe[800:]

Chosen_Predictor_nobrackets = [x.replace('/', '').replace(',', '_').replace(' ', '_').replace('-', '') for x in Chosen_Predictor]
Chosen_Predictor_formatted = "_".join(Chosen_Predictor_nobrackets)

Chosen_Timeframe_formatted = Chosen_Timeframe.replace(' ', '_').strip('%').replace(' ', '_').replace('%', '')

ml_dataframe.to_csv("mldataframetest.csv")
ml_dataframe["Target_Up"] = (ml_dataframe[Chosen_Timeframe] > percent_up).astype(int)
ml_dataframe["Target_Down"] = (ml_dataframe[Chosen_Timeframe] < percent_down).astype(int)

ml_dataframe.to_csv('tempMLDF.csv')

model = RandomForestClassifier(random_state=1)

parameters = {
    'n_estimators': [80, 90, 100, 110, 120],
    'min_samples_split': [40, 60, 80],
    'max_depth': [None, 4, 5, 6, 7, 8],
}
print("Performing GridSearchCV...")

X = ml_dataframe[Chosen_Predictor]
y_up = ml_dataframe["Target_Up"]
y_down = ml_dataframe["Target_Down"]
X_train, X_test, y_up_train, y_up_test, y_down_train, y_down_test = train_test_split(X, y_up, y_down, test_size=0.2, random_state=1)

grid_search = GridSearchCV(estimator=model, param_grid=parameters, cv=10, scoring='precision')

grid_search.fit(X_train, y_up_train)

print("Best parameters for Target_Up:", grid_search.best_params_)
print("Best score for Target_Up:", grid_search.best_score_)
best_param_up = f"Best parameters for Target_Up: {grid_search.best_params_}. Best precision: {grid_search.best_score_}"
model_up = grid_search.best_estimator_

grid_search.fit(X_train, y_down_train)

print("Best parameters for Target_Down:", grid_search.best_params_)
print("Best score for Target_Down:", grid_search.best_score_)
best_param_down = f"Best parameters for Target_Down: {grid_search.best_params_}. Best precision: {grid_search.best_score_}"
model_down = grid_search.best_estimator_

# predicted_up = model_up.predict(X_test)
# predicted_down = model_down.predict(X_test)

predicted_probabilities_up = model_up.predict_proba(X_test)
predicted_probabilities_down = model_down.predict_proba(X_test)

threshold_up = .6
threshold_down = .7

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

# Cross-validation
cv_scores_up = cross_val_score(model_up, X, y_up, cv=10)
cv_scores_down = cross_val_score(model_down, X, y_down, cv=10)

print("Cross-validation scores for Target_Up:", cv_scores_up)
print("Mean cross-validation score for Target_Up:", cv_scores_up.mean())
print("Cross-validation scores for Target_Down:", cv_scores_down)
print("Mean cross-validation score for Target_Down:", cv_scores_down.mean())

input_val = input("Would you like to save these models? y/n: ").upper()
if input_val == "Y":
    model_directory = os.path.join("Trained_Models", f"{ticker}_{Chosen_Timeframe_formatted}{Chosen_Predictor_formatted}_threshUp{threshold_up_formatted}_threshDown{threshold_down_formatted}")
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    model_filename_up = os.path.join(model_directory, "target_up.joblib")
    model_filename_down = os.path.join(model_directory, "target_down.joblib")

    joblib.dump(model_up, model_filename_up)
    joblib.dump(model_down, model_filename_down)
    with open(f"{model_directory}/{ticker}_{Chosen_Timeframe_formatted}{Chosen_Predictor_formatted}_threshUp{threshold_up_formatted}_threshDown{threshold_down_formatted}", "w") as info_txt:
        info_txt.write("This file contains information about the model.\n\n")
        info_txt.write(f"Metrics for Target_Up:\nPrecision: {precision_up}\nAccuracy: {accuracy_up}\nRecall: {recall_up}\nF1-Score: {f1_up}\nCross-validation scores for Target_Up: {cv_scores_up}\nMean cross-validation score for Target_Up: {cv_scores_up.mean()}\n\nMetrics for Target_Down:\nPrecision: {precision_down}\nAccuracy: {accuracy_down}\nRecall: {recall_down}\nF1-Score: {f1_down}\nCross-validation scores for Target_Down: {cv_scores_down}Mean cross-validation score for Target_Down: {cv_scores_down.mean()}\n\n")
        info_txt.write(f"File analyzed: {DF_filename}\nLookahead Target: {Chosen_Timeframe}\nPredictors: {Chosen_Predictor}\nThreshold Up(sensitivity): {threshold_up}\nThreshold Down(sensitivity): {threshold_down}\nTarget Underlying Percentage Up: {percent_up}\nTarget Underlying Percentage Down: {percent_down}\n")
else:
    exit()
