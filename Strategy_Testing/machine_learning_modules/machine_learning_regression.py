import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, TimeSeriesSplit
import os

processed_dir = "dailyDF"
ticker = "SPY"
print(ticker)
list_of_df = []
ticker_dir = os.path.join(processed_dir, ticker)

DF_filename = ("../historical_minute_DF/SPY/230603_SPY.csv")
ml_dataframe = pd.read_csv(DF_filename)

Chosen_Timeframe = "15 min later change %"
Chosen_Timeframe_formatted = Chosen_Timeframe.replace(' ', '_').strip('%').replace(' ', '_').replace('%', '')

Chosen_Predictor = ['time','Current SP % Change(LAC)','Maximum Pain','Bonsai Ratio','Bonsai Ratio 2','B1/B2','B2/B1','PCR-Vol','PCR-OI','PCRv @CP Strike','PCRoi @CP Strike','PCRv Up1','PCRv Up2','PCRv Up3','PCRv Up4','PCRv Down1','PCRv Down2','PCRv Down3','PCRv Down4','PCRoi Up1','PCRoi Up2','PCRoi Up3','PCRoi Up4','PCRoi Down1','PCRoi Down2','PCRoi Down3','PCRoi Down4','ITM PCR-Vol','ITM PCR-OI','ITM PCRv Up1','ITM PCRv Up2','ITM PCRv Up3','ITM PCRv Up4','ITM PCRv Down1','ITM PCRv Down2','ITM PCRv Down3','ITM PCRv Down4','ITM PCRoi Up1','ITM PCRoi Up2','ITM PCRoi Up3','ITM PCRoi Up4','ITM PCRoi Down1','ITM PCRoi Down2','ITM PCRoi Down3','ITM PCRoi Down4','ITM OI','Total OI','ITM Contracts %','Net_IV','Net ITM IV','NIV Current Strike','NIV 1Higher Strike','NIV 1Lower Strike','NIV 2Higher Strike','NIV 2Lower Strike','NIV 3Higher Strike','NIV 3Lower Strike','NIV 4Higher Strike','NIV 4Lower Strike','NIV highers(-)lowers1-2','NIV highers(-)lowers1-4','NIV 1-2 % from mean','NIV 1-4 % from mean','Net_IV/OI','Net ITM_IV/ITM_OI','RSI','AwesomeOsc','Up or down','B1% Change','B2% Change']
threshold_up = 0.9
threshold_down = 0.9
percent_up = 0.1
percent_down = -0.1
num_best_features = 3
ml_dataframe.dropna(subset=[Chosen_Timeframe] + Chosen_Predictor, inplace=True)

num_rows = len(ml_dataframe[Chosen_Timeframe].dropna())
ml_dataframe.dropna(thresh=num_rows, axis=1, inplace=True)
threshold_up_formatted = int(threshold_up * 10)
threshold_down_formatted = int(threshold_down * 10)

Chosen_Predictor_nobrackets = [x.replace('/', '').replace(',', '_').replace(' ', '_').replace('-', '') for x in
                               Chosen_Predictor]
Chosen_Predictor_formatted = "_".join(Chosen_Predictor_nobrackets)



X = ml_dataframe[Chosen_Predictor].values

ml_dataframe["Target_Up"] = (ml_dataframe[Chosen_Timeframe] > percent_up).astype(int)
ml_dataframe["Target_Down"] = (ml_dataframe[Chosen_Timeframe] < percent_down).astype(int)
y_up = ml_dataframe["Target_Up"]
y_down = ml_dataframe["Target_Down"]
X_train, X_test, y_up_train, y_up_test, y_down_train, y_down_test = train_test_split(
    X, y_up, y_down, test_size=0.2, shuffle=False)


selector_up = SelectKBest(f_regression, k=num_best_features)
selector_down = SelectKBest(f_regression, k=num_best_features)

X_train_up = selector_up.fit_transform(X_train, y_up_train)
X_train_down = selector_down.fit_transform(X_train, y_down_train)

best_features_up = [Chosen_Predictor[i] for i in selector_up.get_support(indices=True)]
best_features_down = [Chosen_Predictor[i] for i in selector_down.get_support(indices=True)]
print(f'best features up: {best_features_up}')
print(f'best features down: {best_features_down}')
model_up = RandomForestRegressor(random_state=1)
model_down = RandomForestRegressor(random_state=1)

parameters = {
    'n_estimators': [20,40,60,80,100, 125],
    'min_samples_split': [10,20,30,40,60,80, 100],
    'max_depth': [None],
}

model_up = GridSearchCV(model_up, parameters, cv=TimeSeriesSplit(n_splits=3))
model_down = GridSearchCV(model_down, parameters, cv=TimeSeriesSplit(n_splits=3))

model_up.fit(X_train_up, y_up_train)
model_down.fit(X_train_down, y_down_train)

X_test_up = selector_up.transform(X_test)
X_test_down = selector_down.transform(X_test)

predicted_up = model_up.predict(X_test_up)
predicted_down = model_down.predict(X_test_down)

mse_up = mean_squared_error(y_up_test, predicted_up)
mae_up = mean_absolute_error(y_up_test, predicted_up)

mse_down = mean_squared_error(y_down_test, predicted_down)
mae_down = mean_absolute_error(y_down_test, predicted_down)
mse_up = mean_squared_error(y_up_test, predicted_up)
mae_up = mean_absolute_error(y_up_test, predicted_up)
rmse_up = np.sqrt(mse_up)
r2_up = r2_score(y_up_test, predicted_up)

mse_down = mean_squared_error(y_down_test, predicted_down)
mae_down = mean_absolute_error(y_down_test, predicted_down)
rmse_down = np.sqrt(mse_down)
r2_down = r2_score(y_down_test, predicted_down)
print("Best parameters for Target_Up:")
print(model_up.best_params_)

print("Best parameters for Target_Down:")
print(model_down.best_params_)
# Print the evaluation metrics
print("Metrics for Target_Up:")
print("Mean Squared Error:", mse_up)
print("Mean Absolute Error:", mae_up)
print("Root Mean Squared Error:", rmse_up)
print("R-squared:", r2_up)

print("Metrics for Target_Down:")
print("Mean Squared Error:", mse_down)
print("Mean Absolute Error:", mae_down)
print("Root Mean Squared Error:", rmse_down)
print("R-squared:", r2_down)

input_prompt1 = input("Choose directory name for models:")
model_directory = os.path.join(
    "Trained_Models", f"{ticker}_{input_prompt1}_{Chosen_Timeframe_formatted}")
os.makedirs(model_directory, exist_ok=True)
print(model_directory)
model_filename_up = os.path.join(model_directory, "target_up_regression.joblib")
model_filename_down = os.path.join(model_directory, "target_down_regression.joblib")

joblib.dump(model_up, model_filename_up)
joblib.dump(model_down, model_filename_down)

with open(
        f"{model_directory}/info.txt", "w") as info_txt:
    info_txt.write("This file contains information about the model.\n\n")
    info_txt.write(
        f"Metrics for Target_Up:\nMean Squared Error: {mse_up}\nMean Absolute Error: {mae_up}\n\nMetrics for Target_Down:\nMean Squared Error: {mse_down}\nMean Absolute Error: {mae_down}\n\n")
    info_txt.write(
        f"File analyzed: {DF_filename}\nLookahead Target: {Chosen_Timeframe}\nPredictors: {Chosen_Predictor}\nBest_Predictors_Selected Up/Down: {best_features_up}/{best_features_down}\n\nThreshold Up(sensitivity): {threshold_up}\nThreshold Down(sensitivity): {threshold_down}\nTarget Underlying Percentage Up: {percent_up}\nTarget Underlying Percentage Down: {percent_down}\n")
