import joblib
import pandas as pd
from sklearn.metrics import precision_score, accuracy_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
import os

processed_dir = "dailyDF"
ticker = "SPY"
print(ticker)
list_of_df = []
ticker_dir = os.path.join(processed_dir, ticker)
ml_dataframe = pd.read_csv(f'{ticker_dir}.csv')

Chosen_Timeframe = "1 hour later change %"
Chosen_Predictor = ["b1/b2","RSI"]
num_rows = len(ml_dataframe[Chosen_Timeframe].dropna())

ml_dataframe.dropna(thresh=num_rows, axis=1, inplace=True)
ml_dataframe.dropna(inplace=True)
Chosen_Predictor_nobrackets = ",".join([x.replace('/', '_').replace(',','_') for x in Chosen_Predictor])
Chosen_Timeframe_formatted =  Chosen_Timeframe.replace(' ', '_').strip('%')


required_columns = ['ExpDate', 'date', 'time', 'Current Stock Price', 'Current SP % Change(LAC)', 'Bonsai Ratio', 'Bonsai Ratio 2', 'PCR-Vol', 'PCR-OI', 'ITM PCR-Vol', 'Up or down', 'b1/b2', 'RSI', 'AwesomeOsc', '6 hour later change %', '5 hour later change %', '4 hour later change %', '3 hour later change %', '2 hour later change %', '1 hour later change %', '45 min later change %', '30 min later change %', '20 min later change %', '15 min later change %', '10 min later change %', '5 min later change %']
existing_columns = [col for col in required_columns if col in ml_dataframe.columns]
ml_dataframe = ml_dataframe[existing_columns]

ml_dataframe["Target_Up"] = (ml_dataframe[Chosen_Timeframe] > 0.1).astype(int)
ml_dataframe["Target_Down"] = (ml_dataframe[Chosen_Timeframe] < -0.1).astype(int)

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
if not os.path.exists("Trained_Models"):
    os.makedirs(f"Trained_Models/{Chosen_Timeframe_formatted}_{Chosen_Predictor_nobrackets}")
model_filename_up = os.path.join("Trained_Models", f"{Chosen_Timeframe_formatted}_{Chosen_Predictor_nobrackets}/target_up.joblib")
joblib.dump(model, model_filename_up)
predicted_up = model.predict(test[predictors])

grid_search.fit(train[predictors], train["Target_Down"])

print("Best parameters for Target_Down:", grid_search.best_params_)
print("Best score for Target_Down:", grid_search.best_score_)

model = grid_search.best_estimator_
model_filename_down = os.path.join("Trained_Models", f"{Chosen_Timeframe_formatted}_{Chosen_Predictor_nobrackets}/target_down.joblib")
joblib.dump(model, model_filename_down)
predicted_down = model.predict(test[predictors])

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



