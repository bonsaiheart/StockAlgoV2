import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, train_test_split, TimeSeriesSplit
import os
DF_filename = "../../../Historical_Data_Scraper/data/Historical_Processed_ChainData/SPY_w_OHLC.csv"
ml_dataframe = pd.read_csv(DF_filename)

Chosen_Predictor = ['Bonsai Ratio','Bonsai Ratio 2','B1/B2','B2/B1','ITM PCR-Vol','ITM PCR-OI','ITM PCRv Up2','ITM PCRv Down2','ITM PCRoi Up2','ITM PCRoi Down2','RSI','AwesomeOsc']

##had highest corr for 3-5 hours with these:
# Chosen_Predictor = ['Bonsai Ratio','Bonsai Ratio 2','PCRoi Up1', 'B1/B2', 'PCRv Up4']
cells_forward_to_check = 1
##this many cells must meet the percentup/down requiremnet.
threshold_cells_up = cells_forward_to_check * 0.5
threshold_cells_down = cells_forward_to_check * 0.5
#TODO add Beta to the percent, to make it more applicable across tickers.
percent_up = 1
percent_down = -1
###this many cells cannot be < current price for up, >
# current price for down.
anticondition_threshold_cells_up = cells_forward_to_check
anticondition_threshold_cells_down = cells_forward_to_check

####multiplier for positive class weight.  It is already "balanced".  This should put more importance on the positive cases.
positivecase_weight_up = 50   ###changed these from 20 7/12

positivecase_weight_down = 50
###changed these from 20 7/12


num_features_up = 5
num_features_down = 5
##probablility threshhold.
threshold_up = 0.8
threshold_down = 0.8

###35,5,80   6/3/80


parameters = {
    "max_depth": (20,40,60,80, 100 ),  # 50//70/65  100      up 65/3/1400  down 85/5/1300         71123 for 15 min  100/80
    # ###up 100/2/1300,down 80/3/1000
    "min_samples_split": (2, 3,6,),  # 5//5/2     5                      71123                  for 15   2, 3,
    "n_estimators": (800,1000 ,1500,2000,2500 ),  # 1300//1600/1300/1400/1400  71123for 15 ,1000, 1300, ,
}
#2 days(2 cells)    Target_Up:'{'max_depth': 100, 'min_samples_split': 3, 'n_estimators': 1500}Down: {'max_depth': 100, 'min_samples_split': 2, 'n_estimators': 800} up:60,2,1000 down:60,2,2000
#120 cells own: {'max_depth': 30, 'min_samples_split': 3, 'n_estimators': 900}Up: {'max_depth': 30, 'min_samples_split': 2, 'n_estimators': 800}
#30cells - up80.4.900 down  80.2.1300
##TODO make param_up/param_down.  up = 'max_depth': 40, 'min_samples_split': 7, 'n_estimators': 1000
#down=max_depth': 90, 'min_samples_split': 2, 'n_estimators': 1450
####TODO REMEMBER I MADE LOTS OF CHANGES DEBUGGING 7/5/23
ml_dataframe.dropna(subset=Chosen_Predictor, inplace=True)

threshold_up_formatted = int(threshold_up * 10)
threshold_down_formatted = int(threshold_down * 10)
Chosen_Predictor_nobrackets = [
    x.replace("/", "").replace(",", "_").replace(" ", "_").replace("-", "") for x in Chosen_Predictor
]
Chosen_Predictor_formatted = "_".join(Chosen_Predictor_nobrackets)
length = ml_dataframe.shape[0]
print("Length of ml_dataframe:", length)


# Number of cells to check
ml_dataframe["Target_Down"] = 0  # Initialize "Target_Down" column with zeros
ml_dataframe["Target_Up"] = 0
targetUpCounter = 0
targetDownCounter = 0
anticondition_UpCounter = 0
anticondition_DownCounter = 0
for i in range(1, cells_forward_to_check + 1):
    shifted_values = ml_dataframe["Open"].shift(-i)
    condition_met_up = shifted_values > ml_dataframe["Open"] + percent_up
    anticondition_up = shifted_values <= ml_dataframe["Open"]

    condition_met_down = (
        ml_dataframe["Open"].shift(-i) < ml_dataframe["Open"] + percent_down
    )
    anticondition_down = shifted_values >= ml_dataframe["Open"]

    targetUpCounter += condition_met_up.astype(int)
    targetDownCounter += condition_met_down.astype(int)

    anticondition_UpCounter += anticondition_up.astype(int)
    anticondition_DownCounter += anticondition_down.astype(int)
    ml_dataframe["Target_Up"] = (
        (targetUpCounter >= threshold_cells_up) & (anticondition_UpCounter <= anticondition_threshold_cells_up)
    ).astype(int)

    ml_dataframe["Target_Down"] = (
        (targetDownCounter >= threshold_cells_down) & (anticondition_DownCounter <= anticondition_threshold_cells_down)
    ).astype(int)

ml_dataframe.dropna(subset=["Target_Up", "Target_Down"], inplace=True)
y_up = ml_dataframe["Target_Up"]
y_down = ml_dataframe["Target_Down"]
# Reset the index of your DataFrame

X = ml_dataframe[Chosen_Predictor]
X = X.mask(np.isinf(X), 100000)
X = X.mask(np.isneginf(X), -100000)

X = X.astype("float64")
# Reset the index of your DataFrame
print(X.head())
X.reset_index(drop=True, inplace=True)
# ml_dataframe.to_csv("Current_ML_DF_FOR_TRAINING.csv")

# weight_negative = 1.0
# weight_positive = 5.0  # Assigning a higher weight to the positive class (you can adjust this value based on your needs)
X_train, X_test, y_up_train, y_up_test, y_down_train, y_down_test = train_test_split(
    X, y_up, y_down, test_size=0.2, random_state=None
)
selector_up = SelectKBest(f_regression, k=num_best_features)
selector_down = SelectKBest(f_regression, k=num_best_features)

X_train_up = selector_up.fit_transform(X_train, y_up_train)
X_train_down = selector_down.fit_transform(X_train, y_down_train)

best_features_up = [Chosen_Predictor[i] for i in selector_up.get_support(indices=True)]
best_features_down = [Chosen_Predictor[i] for i in selector_down.get_support(indices=True)]
print(f"best features up: {best_features_up}")
print(f"best features down: {best_features_down}")
model_up = RandomForestRegressor(random_state=1)
model_down = RandomForestRegressor(random_state=1)

parameters = {
    "n_estimators": [20, 40, 60, 80, 100, 125],
    "min_samples_split": [10, 20, 30, 40, 60, 80, 100],
    "max_depth": [None],
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
model_directory = os.path.join("Trained_Models", f"{ticker}_{input_prompt1}_{Chosen_Timeframe_formatted}")
os.makedirs(model_directory, exist_ok=True)
print(model_directory)
model_filename_up = os.path.join(model_directory, "target_up_regression.joblib")
model_filename_down = os.path.join(model_directory, "target_down_regression.joblib")

joblib.dump(model_up, model_filename_up)
joblib.dump(model_down, model_filename_down)

with open(f"{model_directory}/info.txt", "w") as info_txt:
    info_txt.write("This file contains information about the model.\n\n")
    info_txt.write(
        f"Metrics for Target_Up:\nMean Squared Error: {mse_up}\nMean Absolute Error: {mae_up}\n\nMetrics for Target_Down:\nMean Squared Error: {mse_down}\nMean Absolute Error: {mae_down}\n\n"
    )
    info_txt.write(
        f"File analyzed: {DF_filename}\nLookahead Target: {Chosen_Timeframe}\nPredictors: {Chosen_Predictor}\nBest_Predictors_Selected Up/Down: {best_features_up}/{best_features_down}\n\nThreshold Up(sensitivity): {threshold_up}\nThreshold Down(sensitivity): {threshold_down}\nTarget Underlying Percentage Up: {percent_up}\nTarget Underlying Percentage Down: {percent_down}\n"
    )
