import joblib
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, TimeSeriesSplit, cross_val_score
import os

processed_dir = "../dailyDF"
ticker = "SPY"
print(ticker)
list_of_df = []
ticker_dir = os.path.join(processed_dir, ticker)

DF_filename = ("historical_minute_DF/SPY/230603_SPY.csv")
ml_dataframe = pd.read_csv(DF_filename)

Chosen_Timeframe = "30 min later change %"
Chosen_Predictor = ['time','Current Stock Price','Current SP % Change(LAC)','Maximum Pain','Bonsai Ratio','Bonsai Ratio 2','B1/B2','B2/B1','PCR-Vol','PCR-OI','PCRv @CP Strike','PCRoi @CP Strike','PCRv Up1','PCRv Up2','PCRv Up3','PCRv Up4','PCRv Down1','PCRv Down2','PCRv Down3','PCRv Down4','PCRoi Up1','PCRoi Up2','PCRoi Up3','PCRoi Up4','PCRoi Down1','PCRoi Down2','PCRoi Down3','PCRoi Down4','ITM PCR-Vol','ITM PCR-OI','ITM PCRv Up1','ITM PCRv Up2','ITM PCRv Up3','ITM PCRv Up4','ITM PCRv Down1','ITM PCRv Down2','ITM PCRv Down3','ITM PCRv Down4','ITM PCRoi Up1','ITM PCRoi Up2','ITM PCRoi Up3','ITM PCRoi Up4','ITM PCRoi Down1','ITM PCRoi Down2','ITM PCRoi Down3','ITM PCRoi Down4','ITM OI','Total OI','ITM Contracts %','Net_IV','Net ITM IV','Net IV MP','Net IV LAC','NIV Current Strike','NIV 1Higher Strike','NIV 1Lower Strike','NIV 2Higher Strike','NIV 2Lower Strike','NIV 3Higher Strike','NIV 3Lower Strike','NIV 4Higher Strike','NIV 4Lower Strike','NIV highers(-)lowers1-2','NIV highers(-)lowers1-4','NIV 1-2 % from mean','NIV 1-4 % from mean','Net_IV/OI','Net ITM_IV/ITM_OI','Closest Strike to CP','RSI','AwesomeOsc','Up or down','B1% Change','B2% Change']
threshold_up = 0.9
threshold_down = 0.9
percent_up = 0.2
percent_down = -0.2
num_best_features = 4
ml_dataframe.dropna(subset=[Chosen_Timeframe] + Chosen_Predictor, inplace=True)

num_rows = len(ml_dataframe[Chosen_Timeframe].dropna())
ml_dataframe.dropna(thresh=num_rows, axis=1, inplace=True)
threshold_up_formatted = int(threshold_up * 10)
threshold_down_formatted = int(threshold_down * 10)

Chosen_Predictor_nobrackets = [x.replace('/', '').replace(',', '_').replace(' ', '_').replace('-', '') for x in
                               Chosen_Predictor]
Chosen_Predictor_formatted = "_".join(Chosen_Predictor_nobrackets)

Chosen_Timeframe_formatted = Chosen_Timeframe.replace(' ', '_').strip('%').replace(' ', '_').replace('%', '')

ml_dataframe.to_csv("mldataframetest.csv")
ml_dataframe["Target_Up"] = (ml_dataframe[Chosen_Timeframe] > percent_up).astype(int)
ml_dataframe["Target_Down"] = (ml_dataframe[Chosen_Timeframe] < percent_down).astype(int)

ml_dataframe.to_csv('tempMLDF.csv')

model = RandomForestClassifier(random_state=1)

parameters = {
    'n_estimators': [20,40,60,80,100, 125],
    'min_samples_split': [10,20,30,40,80, 100],
    'max_depth': [None,10,20],
}

X = ml_dataframe[Chosen_Predictor]
y_up = ml_dataframe["Target_Up"]
y_down = ml_dataframe["Target_Down"]
X_train, X_test, y_up_train, y_up_test, y_down_train, y_down_test = train_test_split(X, y_up, y_down, test_size=0.2,
                                                                                    random_state=1)
# Feature selection for Target_Up
selector_up = SelectKBest(score_func=f_classif, k=num_best_features)  # Select 2 best features
X_train_selected_up = selector_up.fit_transform(X_train, y_up_train)
X_test_selected_up = selector_up.transform(X_test)

# Feature selection for Target_Down
selector_down = SelectKBest(score_func=f_classif, k=num_best_features)  # Select 2 best features
X_train_selected_down = selector_down.fit_transform(X_train, y_down_train)
X_test_selected_down = selector_down.transform(X_test)

...

tscv = TimeSeriesSplit(n_splits=4)
grid_search_up = GridSearchCV(estimator=model, param_grid=parameters, cv=tscv, scoring='precision')
print("Performing GridSearchCV UP...")
grid_search_up.fit(X_train_selected_up, y_up_train)
best_features_up = [Chosen_Predictor[i] for i in selector_up.get_support(indices=True)]
print("Best features for Target_Up:", best_features_up)

print("Best parameters for Target_Up:", grid_search_up.best_params_)
print("Best score for Target_Up:", grid_search_up.best_score_)
best_param_up = f"Best parameters for Target_Up: {grid_search_up.best_params_}. Best precision: {grid_search_up.best_score_}"
model_up = grid_search_up.best_estimator_

...

grid_search_down = GridSearchCV(estimator=model, param_grid=parameters, cv=tscv, scoring='precision')
print("Performing GridSearchCV DOWN...")
grid_search_down.fit(X_train_selected_down, y_down_train)
best_features_down = [Chosen_Predictor[i] for i in selector_down.get_support(indices=True)]
print("Best features for Target_Down:", best_features_down)
print(grid_search_down.cv_results_['params'])

print("Best parameters for Target_Down:", grid_search_down.best_params_)
print("Best score for Target_Down:", grid_search_down.best_score_)
best_param_down = f"Best parameters for Target_Down: {grid_search_down.best_params_}. Best precision: {grid_search_down.best_score_}"
model_down = grid_search_down.best_estimator_

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
xtestselectedupdf = pd.DataFrame(X_test_selected_up)
xtestselectedupdf.to_csv("xtestselectedup.csv", index=False)
yup = pd.DataFrame(y_up)
yup.to_csv("yup.csv", index=False)
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


input_val = input("Would you like to save these models? y/n: ").upper()
if input_val == "Y":
    model_summary = input("Save this set of models as: ")
    model_directory = os.path.join("Trained_Models",
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
            f"Trained_Models/{model_summary}/info.txt","w") as info_txt:
        info_txt.write("This file contains information about the model.\n\n")
        info_txt.write(
            f"File analyzed: {DF_filename}\nLookahead Target: {Chosen_Timeframe}\n\nBest parameters for Target_Up: {grid_search_up.best_params_}. \nBest precision: {grid_search_up.best_score_}\nBest parameters for Target_Down: {grid_search_down.best_params_}. \nBest precision: {grid_search_down.best_score_}\n\nMetrics for Target_Up:\nPrecision: {precision_up}\nAccuracy: {accuracy_up}\nRecall: {recall_up}\nF1-Score: {f1_up}\nCross-validation scores for Target_Up: {cv_scores_up}\nMean cross-validation score for Target_Up: {cv_scores_up.mean()}\n\nMetrics for Target_Down:\nPrecision: {precision_down}\nAccuracy: {accuracy_down}\nRecall: {recall_down}\nF1-Score: {f1_down}\nCross-validation scores for Target_Down: {cv_scores_down}\nMean cross-validation score for Target_Down: {cv_scores_down.mean()}\n\n")
        info_txt.write(
            f"Predictors: {Chosen_Predictor}\n\nBest_Predictors_Selected Up: {best_features_up}\nBest_Predictors_Selected Down: {best_features_down}\n\nThreshold Up(sensitivity): {threshold_up}\nThreshold Down(sensitivity): {threshold_down}\nTarget Underlying Percentage Up: {percent_up}\nTarget Underlying Percentage Down: {percent_down}\n")
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
