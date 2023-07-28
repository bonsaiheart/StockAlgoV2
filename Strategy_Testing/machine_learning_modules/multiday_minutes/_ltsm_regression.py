import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from keras_tuner.tuners import RandomSearch

import tensorflow as tf


print(tf.config.list_physical_devices('GPU'))
# Check GPU support
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPU support detected.")
else:
    print("No GPU support detected. Make sure you have installed the GPU version of TensorFlow and have compatible hardware.")

DF_filename = "../../../data/historical_multiday_minute_DF/SPY_historical_multiday_min.csv"
ml_dataframe = pd.read_csv(DF_filename)

# Define the chosen predictors
# Chosen_Predictor = ['Maximum Pain', 'Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'B2/B1', 'PCR-Vol', 'PCR-OI',
#                     'PCRv @CP Strike', 'PCRoi @CP Strike', 'PCRv Up1', 'PCRv Up2', 'PCRv Up3', 'PCRv Up4',
#                     'PCRv @CP Strike', 'PCRoi @CP Strike', 'PCRv Up1', 'PCRv Up2', 'PCRv Up3', 'PCRv Up4',
#                     'PCRv Down1', 'PCRv Down2', 'PCRv Down3', 'PCRv Down4', 'PCRoi Up1', 'PCRoi Up2', 'PCRoi Up3',
#                     'PCRoi Up4', 'PCRoi Down1', 'PCRoi Down2', 'PCRoi Down3', 'PCRoi Down4', 'ITM PCR-Vol',
#                     'ITM PCR-OI', 'ITM PCRv Up1', 'ITM PCRv Up2', 'ITM PCRv Up3', 'ITM PCRv Up4', 'ITM PCRv Down1',
#                     'ITM PCRv Down2', 'ITM PCRv Down3', 'ITM PCRv Down4', 'ITM PCRoi Up1', 'ITM PCRoi Up2',
#                     'ITM PCRoi Up3', 'ITM PCRoi Up4', 'ITM PCRoi Down1', 'ITM PCRoi Down2', 'ITM PCRoi Down3',
#                     'ITM PCRoi Down4', 'ITM OI', 'Total OI', 'ITM Contracts %', 'Net_IV', 'Net ITM IV', 'Net IV MP',
#                     'Net IV LAC', 'NIV Current Strike', 'NIV 1Higher Strike', 'NIV 1Lower Strike', 'NIV 2Higher Strike',
#                     'NIV 2Lower Strike', 'NIV 3Higher Strike', 'NIV 3Lower Strike', 'NIV 4Higher Strike',
#                     'NIV 4Lower Strike', 'NIV highers(-)lowers1-2', 'NIV highers(-)lowers1-4', 'NIV 1-2 % from mean',
#                     'NIV 1-4 % from mean', 'Net_IV/OI', 'Net ITM_IV/ITM_OI', 'Closest Strike to CP', 'RSI', 'AwesomeOsc',
#                     'RSI14', 'RSI2', 'AwesomeOsc5_34']
Chosen_Predictor = [
    "Bonsai Ratio",
    "Bonsai Ratio 2",
    "B1/B2",
    "PCRv Up3",
    "PCRv Down3",
    "PCRv Up4",
    "PCRv Down4",
    "ITM PCRv Up3",
    "ITM PCRv Down3","ITM PCRv Up4",
    "ITM PCRv Down4",
    "RSI14",
    "AwesomeOsc5_34",
    "RSI",
    "RSI2",
    "AwesomeOsc",
]
tuner_checkpoint_dir = 'tunerA1'
num_features_to_select = "all"
sequence_length = 1000  # up to 500 I've tried.
# how many cells forward the target "current price" is.
cells_forward_to_predict = 60
ml_dataframe[f'Price % change {cells_forward_to_predict}min Later'] = (
    ml_dataframe["Current Stock Price"].pct_change(periods=-cells_forward_to_predict) * 100 )
ml_dataframe.dropna(subset=Chosen_Predictor + [f'Price % change {cells_forward_to_predict}min Later'], inplace=True)
target_column = ml_dataframe[f'Price % change {cells_forward_to_predict}min Later']
ml_dataframe.dropna(subset=Chosen_Predictor + [f'Price % change {cells_forward_to_predict}min Later'], inplace=True)

# Preprocess the data
data = ml_dataframe[Chosen_Predictor].values
target = target_column.values.reshape(-1, 1)

# Scale the data and target column
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data)
target_scaled = scaler.fit_transform(target)

# Perform feature selection
selector = SelectKBest(score_func=f_regression, k=num_features_to_select)
data_scaled = selector.fit_transform(data_scaled, target_scaled.flatten())

# Get the selected features
selected_feature_indices = selector.get_support(indices=True)
Chosen_Predictor = [Chosen_Predictor[i] for i in selected_feature_indices]

# Define the number of folds for cross-validation
n_splits = 3

# Initialize lists to store evaluation metrics
mse_train_scores = []
mae_train_scores = []
r2_train_scores = []
mse_test_scores = []
mae_test_scores = []
r2_test_scores = []

# Perform cross-validation
tscv = TimeSeriesSplit(n_splits=n_splits)

for train_index, test_index in tscv.split(data_scaled):
    # Split the data into training and testing sets
    train_data = data_scaled[train_index]
    test_data = data_scaled[test_index]
    train_target = target_scaled[train_index]
    test_target = target_scaled[test_index]


    # Create the input sequences and corresponding labels
    X_train, y_train = [], []
    for i in range(sequence_length, len(train_data)):
        X_train.append(train_data[i - sequence_length:i])
        y_train.append(train_target[i])

    X_test, y_test = [], []
    for i in range(sequence_length, len(test_data)):
        X_test.append(test_data[i - sequence_length:i])
        y_test.append(test_target[i])

    # Convert the data to numpy arrays
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    # Define the model inside the loop to ensure it's re-initialized in each iteration
    def build_model(hp):
        model = Sequential()
        model.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32), return_sequences=True,
                       input_shape=(sequence_length, len(Chosen_Predictor))))
        model.add(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32), return_sequences=False))
        model.add(Dense(units=1))
        model.compile(optimizer='adam', loss='mse')
        return model

    # Perform hyperparameter tuning with Keras Tuner
    tuner = RandomSearch(
        build_model,
        objective='val_loss',
        max_trials=10,
        directory='ITSM_Regression_Models',
        project_name=tuner_checkpoint_dir
    )

    tuner.search(X_train, y_train, epochs=100, validation_data=(X_test, y_test), callbacks=[EarlyStopping(patience=10)])

    # Get the best model and its hyperparameters
    best_model = tuner.get_best_models(1)[0]
    best_params = tuner.get_best_hyperparameters(1)[0]

    # Evaluate the model on the training data
    train_pred = best_model.predict(X_train)
    mse_train = mean_squared_error(y_train, train_pred)
    mae_train = mean_absolute_error(y_train, train_pred)
    r2_train = r2_score(y_train, train_pred)
    mse_train_scores.append(mse_train)
    mae_train_scores.append(mae_train)
    r2_train_scores.append(r2_train)

    # Evaluate the model on the testing data
    test_pred = best_model.predict(X_test)
    mse_test = mean_squared_error(y_test, test_pred)
    mae_test = mean_absolute_error(y_test, test_pred)
    r2_test = r2_score(y_test, test_pred)
    mse_test_scores.append(mse_test)
    mae_test_scores.append(mae_test)
    r2_test_scores.append(r2_test)

    # Save the f-values in a text file
    with open('info.txt', 'a') as f:
        f.write(str(tuner.oracle.get_best_trials(1)[0].score) + '\n')

importance_dict = dict(zip(selected_feature_indices, best_model.layers[0].get_weights()[0].sum(axis=0)))
sorted_importance = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
for feature, importance in sorted_importance:
    print(f"{feature}: {importance}")

print("\nBest Hyperparameters:")
print(best_params.values)
print(f'MSE train: {np.mean(mse_train_scores)} (+/- {np.std(mse_train_scores)})')
print(f'MAE train: {np.mean(mae_train_scores)} (+/- {np.std(mae_train_scores)})')
print(f'R^2 train: {np.mean(r2_train_scores)} (+/- {np.std(r2_train_scores)})')
print(f'MSE test: {np.mean(mse_test_scores)} (+/- {np.std(mse_test_scores)})')
print(f'MAE test: {np.mean(mae_test_scores)} (+/- {np.std(mae_test_scores)})')
print(f'R^2 test: {np.mean(r2_test_scores)} (+/- {np.std(r2_test_scores)})')

# Save the model, scalers, selected features, and best hyperparameters
save_model = input("Do you want to save the model? (y/n): ")
if save_model.lower() == "y":
    model_name = input("Enter a name for the model: ")

    # Create the model folder inside saved_models
    model_dir = os.path.join("saved_models", model_name)
    os.makedirs(model_dir, exist_ok=True)

    # Save the model architecture and weights
    model_json = best_model.to_json()
    with open(os.path.join(model_dir, "model_architecture.json"), "w") as json_file:
        json_file.write(model_json)
    best_model.save_weights(os.path.join(model_dir, "model_weights.h5"))

    # Save the data scaler
    with open(os.path.join(model_dir, "data_scaler.pkl"), 'wb') as file:
        pickle.dump(scaler, file)

    # Save the target scaler
    with open(os.path.join(model_dir, "target_scaler.pkl"), 'wb') as file:
        pickle.dump(target_scaler, file)

    # Save the selected features
    with open(os.path.join(model_dir, "model_info.txt"), 'w') as file:
        file.write("Cells Forward to Predict: " + str(cells_forward_to_predict) + "\n")
        file.write("Sequence Length: " + str(sequence_length) + "\n")
        file.write("Chosen Predictor Features:\n")
        file.write("\n".join(Chosen_Predictor))
        file.write('\nBest Parameters:\n')
        file.write(str(best_params.values))
        file.write(f"\nMSE train: {np.mean(mse_train_scores)} (+/- {np.std(mse_train_scores)})\n"
                   f"MAE train: {np.mean(mae_train_scores)} (+/- {np.std(mae_train_scores)})\n"
                   f"R^2 train: {np.mean(r2_train_scores)} (+/- {np.std(r2_train_scores)})\n"
                   f"MSE test: {np.mean(mse_test_scores)} (+/- {np.std(mse_test_scores)})\n"
                   f"MAE test: {np.mean(mae_test_scores)} (+/- {np.std(mae_test_scores)})\n"
                   f"R^2 test: {np.mean(r2_test_scores)} (+/- {np.std(r2_test_scores)})")

    print(f"Saved the model '{model_name}', scalers, selected features, and best hyperparameters.")
