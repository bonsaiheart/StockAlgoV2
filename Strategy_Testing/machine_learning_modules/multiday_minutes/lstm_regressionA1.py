from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam
import os
import pandas as pd
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DF_filename = r"../../../data/historical_multiday_minute_DF/Copy of SPY_historical_multiday_min.csv"

ml_dataframe = pd.read_csv(DF_filename)
print(ml_dataframe.columns)

Chosen_Predictor = [  # add your chosen predictors
    "Bonsai Ratio",
    "Bonsai Ratio 2",
    "B1/B2",
    'ITM PCR-Vol',
    "PCRv Up3",
    "PCRv Up2",
    "PCRv Down3",
    "PCRv Down2",
    'Net_IV',
    'Net ITM IV',
    "ITM PCRv Up3",
    "ITM PCRv Down3",

     "ITM PCRv Up2",
    "ITM PCRv Down2",
    "RSI14",
    "AwesomeOsc5_34",
    "RSI",
]

ml_dataframe.dropna(subset=Chosen_Predictor, inplace=True)
length = ml_dataframe.shape[0]
print("Length of ml_dataframe:", length)

ml_dataframe["Target_Down"] = ml_dataframe["Current Stock Price"].shift(-5)
ml_dataframe["Target_Up"] = ml_dataframe["Current Stock Price"].shift(-5)


ml_dataframe.dropna(subset=["Target_Up", "Target_Down"], inplace=True)
y_up = ml_dataframe["Target_Up"]
y_down = ml_dataframe["Target_Down"]
X = ml_dataframe[Chosen_Predictor]
X.reset_index(drop=True, inplace=True)

X_train, X_test, y_up_train, y_up_test, y_down_train, y_down_test = train_test_split(
    X, y_up, y_down, test_size=0.2, random_state=None
)

def create_model(learning_rate, dropout_rate, optimizer):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(dropout_rate))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1))

    # if optimizer == 'SGD':
    #     opt = SGD(learning_rate=learning_rate)
    # elif optimizer == 'RMSprop':
    #     opt = RMSprop(learning_rate=learning_rate)
    # elif optimizer == 'Adagrad':
    #     opt = Adagrad(learning_rate=learning_rate)
    # elif optimizer == 'Adadelta':
    #     opt = Adadelta(learning_rate=learning_rate)
    # elif optimizer == 'Adam':
    #     opt = Adam(learning_rate=learning_rate)
    # elif optimizer == 'Adamax':
    #     opt = Adamax(learning_rate=learning_rate)
    # elif optimizer == 'Nadam':
    #     opt = Nadam(learning_rate=learning_rate)

    model.compile(loss='mean_squared_error', optimizer=optimizer(learning_rate=learning_rate))
    return model

model = KerasRegressor(build_fn=create_model, verbose=0)

# define the grid search parameters
batch_size = [16,64 ]               #128
epochs = [50]
learning_rate = [.01, 0.1]
dropout_rate = [0.0,.2,.5]
optimizer = [ Nadam, RMSprop]
# optimizer = [RMSprop]

param_grid = dict(batch_size=batch_size,
                  epochs=epochs,
                  learning_rate=learning_rate,
                  optimizer=optimizer,
                  dropout_rate=dropout_rate)

# Use TimeSeriesSplit for cross-validation
tscv = TimeSeriesSplit(n_splits=3)

grid = GridSearchCV(estimator=model,
                    param_grid=param_grid,
                    cv=tscv,
                    n_jobs=-1,
                    verbose=1)

# Fit and search best parameters on both target up and target down models
grid_result_up = grid.fit(X_train, y_up_train)
# grid_result_down = grid.fit(X_train, y_down_train)

# summarize results
print("Best: %f using %s" % (grid_result_up.best_score_, grid_result_up.best_params_))
# print("Best: %f using %s" % (grid_result_down.best_score_, grid_result_down.best_params_))
# Define a function to create the model, required for KerasClassifier
# def create_model(learning_rate, optimizer):


# Use the best parameters to create the final model
final_model_up_nn = create_model(learning_rate=grid_result_up.best_params_['learning_rate'], optimizer=grid_result_up.best_params_['optimizer'],dropout_rate=grid_result_up.best_params_['dropout_rate'])
# final_model_down_nn = create_model(learning_rate=grid_result_down.best_params_['learning_rate'], optimizer=grid_result_down.best_params_['optimizer'],dropout_rate=grid_result_down.best_params_['dropout_rate'])

# Add early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10)

# Train the model
history_up = final_model_up_nn.fit(X_train, y_up_train, validation_split=0.2, epochs=grid_result_up.best_params_['epochs'], batch_size=grid_result_up.best_params_['batch_size'], callbacks=[early_stopping])
# history_down = final_model_down_nn.fit(X_train, y_down_train, validation_split=0.2, epochs=grid_result_down.best_params_['epochs'], batch_size=grid_result_down.best_params_['batch_size'], callbacks=[early_stopping])

# Use the trained models for prediction
predicted_values_up = final_model_up_nn.predict(X_test)
# predicted_values_down = final_model_down_nn.predict(X_test)

# Calculate MSE for both models
mse_up = mean_squared_error(y_up_test, predicted_values_up)
mae_up = mean_absolute_error(y_up_test, predicted_values_up)
r2_up = r2_score(y_up_test, predicted_values_up)

# mse_down = mean_squared_error(y_down_test, predicted_values_down)
# mae_down = mean_absolute_error(y_down_test, predicted_values_down)
# r2_down = r2_score(y_down_test, predicted_values_down)
mse_down = "none"
mae_down = "mpme"
r2_down = "none"

print("MSE for Up Model:", mse_up)
print("MAE for Up Model:", mae_up)
print("R^2 for Up Model:", r2_up)

# print("MSE for Down Model:", mse_down)
# print("MAE for Down Model:", mae_down)
# print("R^2 for Down Model:", r2_down)

input_val = input("Would you like to save these models? y/n: ").upper()
if input_val == "Y":
    model_summary = input("Save this set of models as: ")
    model_directory = os.path.join("../../Trained_Models", f"{model_summary}")
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)
    model_filename_up = os.path.join(model_directory, "target_up.h5")
    model_filename_down = os.path.join(model_directory, "target_down.h5")
    grid_result_up.best_estimator_.model.save(model_filename_up)
    # grid_result_down.best_estimator_.model.save(model_filename_down)

with open(f"../../Trained_Models/{model_summary}/info.txt", "w") as info_txt:
    info_txt.write("This file contains information about the model.\n\n")
    info_txt.write(
        f"File analyzed: {DF_filename}\nMSE for Up Model: {mse_up}\nMSE for Down Model: {mse_down}\n\MAE for Up Model: {mae_up}\nMAE for Down Model: {mae_down}\n\nR^2 for Up Model: {r2_up}\nR^2 for Down Model: {r2_down}"
    )
    info_txt.write(
        f"Predictors: {Chosen_Predictor}\n"
    )
