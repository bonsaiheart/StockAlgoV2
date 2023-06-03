import joblib
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, TimeSeriesSplit, cross_val_score
import os

processed_dir = "dailyDF"
ticker = "SPY"
print(ticker)
list_of_df = []
ticker_dir = os.path.join(processed_dir, ticker)

DF_filename = ("historical_minute_DF/SPY.csv")
ml_dataframe = pd.read_csv(DF_filename)

Chosen_Timeframe = "2 hour later change %"
Chosen_Predictor = ["Bonsai Ratio","Bonsai Ratio 2", "B1/B2","RSI"]
threshold_up = .6
threshold_down = .6
percent_up = .1
percent_down = -.1

ml_dataframe.dropna(subset=[Chosen_Tim
                            eframe] + Chosen_Predictor, inplace=True)

num_rows = len(ml_dataframe[Chosen_Timeframe].dropna())
ml_dataframe.dropna(thresh=num_rows, axis=1, inplace=True)
threshold_up_formatted = int(threshold_up * 10)
threshold_down_formatted = int(threshold_down * 10)

# ml_dataframe = ml_dataframe[800:]

Chosen_Predictor_nobrackets = [x.replace('/', '').replace(',', '_').replace(' ', '_').replace('-', '') for x in Chosen_Predictor]
Chosen_Predictor_formatted = "_".join(Chosen_Predictor_nobrackets)

Chosen_Timeframe_formatted = Chosen_Timeframe.replace(' ', '_').strip('%').replace(' ', '_').replace('%', '')

ml_dataframe.to_csv("mldataframetest.csv")
ml_dataframe["Target_Up"] = (ml_dataframe[Chosen_Timeframe] > percent_up).astype(int)
ml_dataframe["Target_Down"] = (ml_dataframe[Chosen_Timeframe] < percent_down).astype(int)

ml_dataframe.to_csv('tempMLDF.csv')

###FOR DAILY DATA <i think<2019 missing OI
# ml_dataframe = ml_dataframe[800:]
model = RandomForestClassifier(random_state=1)

parameters = {
    'n_estimators': [1000,1250],
    'min_samples_split': [30,60,80,100],

    'max_depth': [None],
}


X = ml_dataframe[Chosen_Predictor]
y_up = ml_dataframe["Target_Up"]
y_down = ml_dataframe["Target_Down"]
X_train, X_test, y_up_train, y_up_test, y_down_train, y_down_test = train_test_split(X, y_up, y_down, test_size=0.2, random_state=1)

tscv = TimeSeriesSplit(n_splits=4)
grid_search = GridSearchCV(estimator=model, param_grid=parameters, cv=tscv, scoring='precision')
print("Performing GridSearchCV UP...")
grid_search.fit(X_train, y_up_train)

print("Best parameters for Target_Up:", grid_search.best_params_)
print("Best score for Target_Up:", grid_search.best_score_)
best_param_up = f"Best parameters for Target_Up: {grid_search.best_params_}. Best precision: {grid_search.best_score_}"
model_up = grid_search.best_estimator_
print("Performing GridSearchCV DOWN...")
grid_search.fit(X_train, y_down_train)

print(grid_search.cv_results_['params'])

# Get the corresponding scores

# Print the parameter combinations triggering the warning

print("Best parameters for Target_Down:", grid_search.best_params_)
print("Best score for Target_Down:", grid_search.best_score_)
best_param_down = f"Best parameters for Target_Down: {grid_search.best_params_}. Best precision: {grid_search.best_score_}"
model_down = grid_search.best_estimator_

# predicted_up = model_up.predict(X_test)
# predicted_down = model_down.predict(X_test)

predicted_probabilities_up = model_up.predict_proba(X_test)
predicted_probabilities_down = model_down.predict_proba(X_test)



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
cv_scores_up = cross_val_score(model_up, X, y_up, cv=tscv)
cv_scores_down = cross_val_score(model_down, X, y_down, cv=tscv)

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
# To modify your code to perform regression instead of classification, you need to make a few changes:
#
# Update the target variables: Instead of creating binary target variables (Target_Up and Target_Down) based on whether the value exceeds a certain threshold, you need to use the actual numerical values of the target variable (Chosen_Timeframe) as your regression target.
#
# Change the model: Replace RandomForestClassifier with RandomForestRegressor to use a regression model instead of a classification model.
#
# Update evaluation metrics: Since you are performing regression, you'll need to use different evaluation metrics suited for regression tasks. Some commonly used metrics for regression include mean squared error (MSE), mean absolute error (MAE), and R-squared. You can import these metrics from sklearn.metrics and use them to evaluate your regression model.
#
# Here's an updated version of your code with the necessary modifications:
#
# python
# Copy code
# import joblib
# import pandas as pd
# from sklearn.exceptions import UndefinedMetricWarning
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import GridSearchCV, train_test_split, TimeSeriesSplit, cross_val_score
# import os
#
# # Your existing code...
#
# # Update target variables
# y = ml_dataframe[Chosen_Timeframe]
#
# # Change the model
# model = RandomForestRegressor(random_state=1)
#
# # Update evaluation metrics
# def evaluate_regression(y_true, y_pred):
#     mse = mean_squared_error(y_true, y_pred)
#     mae = mean_absolute_error(y_true, y_pred)
#     r2 = r2_score(y_true, y_pred)
#     return mse, mae, r2
#
# # Rest of your code...
#
# # Fit the model
# model.fit(X_train, y_train)
#
# # Make predictions
# y_pred = model.predict(X_test)
#
# # Evaluate the model
# mse, mae, r2 = evaluate_regression(y_test, y_pred)
#
# # Print evaluation metrics
# print("Mean Squared Error:", mse)
# print("Mean Absolute Error:", mae)
# print("R-squared:", r2)
#
# # Cross-validation
# cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='neg_mean_squared_error')
# cv_scores = -cv_scores  # Multiply by -1 to get positive MSE values
#
# print("Cross-validation scores:", cv_scores)
# print("Mean cross-validation score:", cv_scores.mean())
