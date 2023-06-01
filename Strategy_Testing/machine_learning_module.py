import joblib
import pandas as pd
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
import os

processed_dir = "dailyDF"
ticker = "TSLA"
print(ticker)
list_of_df = []
ticker_dir = os.path.join(processed_dir, ticker)
ml_dataframe = pd.read_csv(f'{ticker_dir}.csv')

Chosen_Timeframe = "30 min later change %"
Chosen_Predictor = ["B1/B2","Bonsai Ratio","RSI",'ITM PCR-Vol']
threshold_up = .8
threshold_down = .8
percent_up=.1
percent_down=-.1
num_rows = len(ml_dataframe[Chosen_Timeframe].dropna())
threshold_up_formatted = int(threshold_up*10)
threshold_down_formatted = int(threshold_down*10)
print(threshold_down_formatted)
ml_dataframe.dropna(thresh=num_rows, axis=1, inplace=True)
ml_dataframe.dropna(inplace=True)
Chosen_Predictor_nobrackets = [x.replace('/', '').replace(',', '_').replace(' ', '_').replace('-', '') for x in Chosen_Predictor]
Chosen_Predictor_formatted = "_".join(Chosen_Predictor_nobrackets)

Chosen_Timeframe_formatted = Chosen_Timeframe.replace(' ', '_').strip('%').replace(' ', '_').replace('%', '')

print(f'{Chosen_Timeframe_formatted}{Chosen_Predictor_formatted}')

required_columns = ['ExpDate', 'date', 'time', 'Current Stock Price', 'Current SP % Change(LAC)', 'Bonsai Ratio', 'Bonsai Ratio 2', 'PCR-Vol', 'PCR-OI', 'ITM PCR-Vol', 'Up or down', 'B1/B2','B2/B1', 'RSI', 'AwesomeOsc', '6 hour later change %', '5 hour later change %', '4 hour later change %', '3 hour later change %', '2 hour later change %', '1 hour later change %', '45 min later change %', '30 min later change %', '20 min later change %', '15 min later change %', '10 min later change %', '5 min later change %']
existing_columns = [col for col in required_columns if col in ml_dataframe.columns]
ml_dataframe = ml_dataframe[existing_columns]

ml_dataframe["Target_Up"] = (ml_dataframe[Chosen_Timeframe] > percent_up).astype(int)
ml_dataframe["Target_Down"] = (ml_dataframe[Chosen_Timeframe] < percent_down).astype(int)

ml_dataframe.to_csv('tempMLDF.csv')

model = RandomForestClassifier(random_state=1)

parameters = {
    'n_estimators': [80, 100, 120],
    'min_samples_split': [40, 80, 100]
}

grid_search = GridSearchCV(estimator=model, param_grid=parameters, cv=5, scoring='accuracy')

###CONTROLS DATA SPLIT HERE
train = ml_dataframe.sample(frac=0.8, random_state=1)
test = ml_dataframe.drop(train.index)

predictors = Chosen_Predictor

grid_search.fit(train[predictors], train["Target_Up"])

print("Best parameters for Target_Up:", grid_search.best_params_)
print("Best score for Target_Up:", grid_search.best_score_)

model = grid_search.best_estimator_
model_directory = os.path.join("Trained_Models", f"{Chosen_Predictor_formatted}_threshUp{threshold_up_formatted}_threshDown{threshold_down_formatted}_{Chosen_Timeframe_formatted}{ticker}")
if not os.path.exists(model_directory):
    os.makedirs(model_directory)

model_filename_up = os.path.join(model_directory, "target_up.joblib")
joblib.dump(model, model_filename_up)

###ADDING probablities by replacing this line with the following 3 lines.
# predicted_up = model.predict(test[predictors])
  # Adjust the threshold as desired
predicted_probabilities_up = model.predict_proba(test[predictors])
predicted_up = (predicted_probabilities_up[:, 1] > threshold_up).astype(int)


grid_search.fit(train[predictors], train["Target_Down"])

print("Best parameters for Target_Down:", grid_search.best_params_)
print("Best score for Target_Down:", grid_search.best_score_)

model = grid_search.best_estimator_
model_filename_down = os.path.join(model_directory, "target_down.joblib")
joblib.dump(model, model_filename_down)

###SAME HERE
predicted_down = model.predict(test[predictors])
 # Adjust the threshold as desired
predicted_probabilities_down = model.predict_proba(test[predictors])
predicted_down = (predicted_probabilities_down[:, 1] > threshold_down).astype(int)
precision_up = precision_score(test["Target_Up"], predicted_up)
accuracy_up = accuracy_score(test["Target_Up"], predicted_up)
recall_up = recall_score(test["Target_Up"], predicted_up)
f1_up = f1_score(test["Target_Up"], predicted_up)

print("Metrics for Target_Up:")
print("Precision:", precision_up)
print("Accuracy:", accuracy_up)
print("Recall:", recall_up)
print("F1-Score:", f1_up)

precision_down = precision_score(test["Target_Down"], predicted_down)
accuracy_down = accuracy_score(test["Target_Down"], predicted_down)
recall_down = recall_score(test["Target_Down"], predicted_down)
f1_down = f1_score(test["Target_Down"], predicted_down)

print("Metrics for Target_Down:")
print("Precision:", precision_down)
print("Accuracy:", accuracy_down)
print("Recall:", recall_down)
print("F1-Score:", f1_down)



