import os
from datetime import datetime
import numpy as np
import yaml
from sklearn.preprocessing import RobustScaler
import optuna
import pandas as pd
import torch
import torch.nn as nn
import torchmetrics
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from torchmetrics import Precision, Accuracy, Recall, F1Score
#TODO cvhange test len to 2 days.
def create_model(ticker):
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)


    # Dynamically generate study name and filename
    study_name = f"{ticker}_ptminclassA1Base_2hr50ptdown"
    df_filename = f"StockAlgoV2\\data\\historical_multiday_minute_DF\\{ticker}_historical_multiday_min.csv"

    config = config['tickers'][ticker]
    # best_params = config['best_params']

    Chosen_Predictor = config['chosen_predictors']
    theshhold_down = config['threshhold_down']
    DF_filename = config['df_filename']
    study_name =  '_2hr_50pt_down'
    # study_name= config['study_name']
    cells_forward_to_check = config['cells_forward_to_check']
    percent_down =  config['percent_down'] # as percent
    takeprofits_trailingstops = config['takeprofits_trailingstops']
    threshold_cells_up = cells_forward_to_check * config['min_cells_positive_percentage']
    anticondition_threshold_cells = cells_forward_to_check * config['max_cells_below_1st_price_percentage']


    ml_dataframe = pd.read_csv(DF_filename)

    # TODO scale Predictors based on data ranges/types


    n_trials = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("device: ", device)

    print(ml_dataframe.columns)

    last_1_months_min = 3900#7800
    ml_dataframe = ml_dataframe[:last_1_months_min]
    ml_dataframe.dropna(subset=Chosen_Predictor, inplace=True)
    length = ml_dataframe.shape[0]
    print("Length of ml_dataframe:", length)
    ml_dataframe["Target_Up"] = 0
    targetUpCounter = 0
    anticondition_UpCounter = 0
    for i in range(1, cells_forward_to_check + 1):
        shifted_values = ml_dataframe["Current Stock Price"].shift(-i)
        condition_met_up = shifted_values < (
            ml_dataframe["Current Stock Price"]
            - (ml_dataframe["Current Stock Price"] * (percent_down / 100))
        )
        anticondition_up = shifted_values >= ml_dataframe["Current Stock Price"]
        targetUpCounter += condition_met_up.astype(int)
        anticondition_UpCounter += anticondition_up.astype(int)
    ml_dataframe["Target_Up"] = (
        (targetUpCounter >= threshold_cells_up)
        & (anticondition_UpCounter <= anticondition_threshold_cells)
    ).astype(int)
    ml_dataframe.dropna(subset=["Target_Up"], inplace=True)
    y_up = ml_dataframe["Target_Up"]
    X = ml_dataframe[Chosen_Predictor]

    # Reset index
    X.reset_index(drop=True, inplace=True)


    # # # TODO#shuffle trur or false?
    X_train, X_temp, y_up_train, y_up_temp = train_test_split(
        X, y_up, test_size=0.3, random_state=None, shuffle=True
    )

    # Split the temp set into validation and test sets
    X_val, X_test, y_up_val, y_up_test   = train_test_split(
        X_temp, y_up_temp, test_size=0.5, random_state=None, shuffle=True
    )



    # def replace_infinities_and_scale(df):
    #     pass

    def replace_infinities_test(df):

        very_large_number = 1e15  # Placeholder for positive infinity
        very_small_number = -1e15  # Placeholder for negative infinity

        for col in df.columns:
            # Replace positive and negative infinity with the defined large and small numbers
            df[col].replace(
                [np.inf, -np.inf], [very_large_number, very_small_number], inplace=True
            )
    def replace_infinities_and_scale(df, factor=1.5):
        for col in df.columns:
            # Replace infinities with NaN, then calculate max and min
            max_val = df[col].replace([np.inf, -np.inf], np.nan).max()
            min_val = df[col].replace([np.inf, -np.inf], np.nan).min()
            # Check if max_val or min_val is infinity
            if np.isinf(max_val) or np.isinf(min_val):
                print(f"Column: {col}, Min/Max values: {min_val}, {max_val}")

            # # Scale max and min values by a factor based on their sign
            max_val = max_val * factor if max_val >= 0 else max_val / factor
            min_val = min_val * factor if min_val < 0 else min_val / factor
            # max_val = min_val
            # min_val = max_val

            # Replace infinities with the scaled max and min values
            df[col].replace([np.inf, -np.inf], [max_val, min_val], inplace=True)
            print(f"Column: {col}, Min/Max values: {min_val}, {max_val}")

    # Concatenate train and validation sets
    X_trainval = pd.concat([X_train, X_val], ignore_index=True)
    y_trainval = pd.concat([y_up_train, y_up_val], ignore_index=True)

    # Replace infinities and adjust extrema in the training, validation, and test sets
    replace_infinities_and_scale(X_train)
    replace_infinities_and_scale(X_val)
    replace_infinities_test(X_test)

    # Replace infinities and adjust extrema in the concatenated train and validation set
    replace_infinities_and_scale(X_trainval)

    # Fit a robust scaler on the concatenated train and validation set and transform it
    finalscaler_X = RobustScaler()
    X_trainval_scaled = finalscaler_X.fit_transform(X_trainval)
    X_trainval_tensor = torch.tensor(X_trainval_scaled, dtype=torch.float32).to(device)
    y_trainval_tensor = torch.tensor(y_trainval.values, dtype=torch.float32).to(device)


    # Function to convert scaled data to tensors
    def convert_to_tensor(scaler, X_train, X_val, X_test, device):
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        return (
            torch.tensor(X_train_scaled, dtype=torch.float32).to(device),
            torch.tensor(X_val_scaled, dtype=torch.float32).to(device),
            torch.tensor(X_test_scaled, dtype=torch.float32).to(device),
        )


    # Create a scaler object and convert datasets to tensors
    scaler = RobustScaler()
    X_train_tensor, X_val_tensor, X_test_tensor = convert_to_tensor(
        scaler, X_train, X_val, X_test, device
    )
    y_up_train_tensor = torch.tensor(y_up_train.values, dtype=torch.float32).to(device)
    y_up_val_tensor = torch.tensor(y_up_val.values, dtype=torch.float32).to(device)
    y_up_test_tensor = torch.tensor(y_up_test.values, dtype=torch.float32).to(device)

    # Print lengths of datasets
    print(
        f"Train length: {len(X_train_tensor)}, Validation length: {len(X_val_tensor)}, Test length: {len(X_test_tensor)}"
    )

    # Calculate the number of positive and negative samples in each set
    num_positive_up_train = y_up_train_tensor.sum().item()
    num_negative_up_train = (y_up_train_tensor == 0).sum().item()
    num_positive_up_val = y_up_val_tensor.sum().item()
    num_negative_up_val = (y_up_val_tensor == 0).sum().item()
    num_positive_up_test = y_up_test_tensor.sum().item()
    num_negative_up_test = (y_up_test_tensor == 0).sum().item()

    # Calculate the number of positive and negative samples in the combined train and validation set
    num_negative_up_trainval = num_negative_up_train + num_negative_up_val
    num_positive_up_trainval = num_positive_up_train + num_positive_up_val


    def print_dataset_statistics(stage, num_positive, num_negative):
        ratio = (
            num_positive / num_negative if num_negative else float("inf")
        )  # Avoid division by zero
        print(f"{stage} ratio of pos/neg up: {ratio:.2f}")
        print(f"{stage} num_positive_up: {num_positive}")
        print(f"{stage} num_negative_up: {num_negative}\n")


    print_dataset_statistics("Train", num_positive_up_train, num_negative_up_train)
    print_dataset_statistics("Validation", num_positive_up_val, num_negative_up_val)
    print_dataset_statistics("Test", num_positive_up_test, num_negative_up_test)


    class DynamicNNwithDropout(nn.Module):
        def __init__(self, input_dim, layers, dropout_rate):
            super(DynamicNNwithDropout, self).__init__()
            self.layers = nn.ModuleList()

            # Create hidden layers
            prev_units = input_dim
            for units in layers:
                self.layers.append(nn.Linear(prev_units, units))
                self.layers.append(nn.ReLU())
                self.layers.append(nn.Dropout(p=dropout_rate))
                self.layers.append(nn.BatchNorm1d(units))  # Batch normalization
                prev_units = units

            # Output layer
            self.layers.append(nn.Linear(prev_units, 1))

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x


    def feature_importance(model, X_val, y_val):
        model.eval()
        with torch.no_grad():
            baseline_output = model(X_val)
            baseline_metric = f1_score(
                y_val.cpu().numpy(), (baseline_output > 0.5).cpu().numpy()
            )

        importances = {}
        for i, col in enumerate(
            Chosen_Predictor
        ):
            temp_val = X_val.clone()
            temp_val[:, i] = torch.randperm(temp_val[:, i].size(0))

            with torch.no_grad():
                shuff_output = model(temp_val)
                shuff_metric = f1_score(
                    y_val.cpu().numpy(), (shuff_output > 0.5).cpu().numpy()
                )

            drop_in_metric = baseline_metric - shuff_metric
            importances[col] = drop_in_metric

        return importances


    def train_model(hparams, X_train, y_train, X_val, y_val):
        positivecase_weight_up = hparams["positivecase_weight_up"]
        weight_positive_up = (
            num_negative_up_train / num_positive_up_train
        ) * positivecase_weight_up
        best_model_state_dict = None
        model = DynamicNNwithDropout(
            X_train.shape[1], hparams["layers"], hparams["dropout_rate"]
        ).to(device)

        model.train()

        weight = torch.Tensor([weight_positive_up]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
        # criterion = nn.BCELoss(weight=weight)
        optimizer_name = hparams["optimizer"]
        learning_rate = hparams["learning_rate"]

        if optimizer_name == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        elif optimizer_name == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name == "RMSprop":
            optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
        elif optimizer_name == "Adagrad":
            optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)

        num_epochs = hparams["num_epochs"]
        batch_size = hparams["batch_size"]
        f1 = torchmetrics.F1Score(num_classes=2, average="weighted", task="binary").to(
            device
        )
        prec = Precision(num_classes=2, average="weighted", task="binary").to(device)
        recall = Recall(num_classes=2, average="weighted", task="binary").to(device)

        best_f1_score = 0.0  # Track the best F1 score
        best_prec_score = 0.0  # Track the best F1 score
        sum_f1_score = 0.0
        sum_prec_score = 0.0
        sum_recall_score = 0.0  # Initialize sum of recall scores

        epochs_sum = 0
        best_epoch = 0  # Initialize variable to save the best epoch

        best_val_loss = float("inf")  # Initialize best validation loss
        patience = 20  # Early stopping patience; how many epochs to wait
        counter = 0  # Initialize counter for early stopping

        for epoch in range(num_epochs):
            # Training step
            model.train()
            for i in range(0, len(X_train), batch_size):
                X_batch = X_train[i : i + batch_size]
                y_batch = y_train[i : i + batch_size]

                # Skip the batch if it has only one sample. works well when the occasional skipping of small batches won't significantly impact the overall training process,
                if X_batch.shape[0] <= 1:
                    continue

                y_batch = y_batch.unsqueeze(1)  # was wrong shape?
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            model.eval()
            # Validation step
            with torch.no_grad():
                y_val = y_val.reshape(-1, 1)
                val_outputs = model(X_val)
                val_loss = criterion(val_outputs, y_val)
                # Compute F1 score and Precision score

                val_predictions = (val_outputs > theshhold_down).float()
                F1Score = f1(val_predictions, y_val)  # computing F1 score
                PrecisionScore = prec(val_predictions, y_val)  # computing Precision score
                RecallScore = recall(val_predictions, y_val)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state_dict = model.state_dict()
                counter = 0  # Reset counter when validation loss improves
            else:
                counter += 1  # Increment counter if validation loss doesn't improve

            if F1Score > best_f1_score:
                best_model_state_dict = model.state_dict()
                best_f1_score = F1Score.item()
                best_epoch = epoch  # Save the epoch where the best F1 score was found

            if PrecisionScore > best_prec_score:
                best_prec_score = PrecisionScore.item()

            sum_f1_score += F1Score.item()
            sum_prec_score += PrecisionScore.item()
            sum_recall_score += RecallScore.item()  # Add to sum of recall scores
            epochs_sum += 1
            # Early stopping
            if counter >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                model.load_state_dict(best_model_state_dict)  # Load the best model
                break
        # Calculate average scores
        avg_val_f1_score = sum_f1_score / num_epochs
        avg_val_precision_score = sum_prec_score / num_epochs
        avg_val_recall_score = (
            sum_recall_score / num_epochs
        )  # Calculate average recall score

        test_outputs = model(X_test_tensor)
        # print(test_outputs)
        test_predictions = (test_outputs > theshhold_down).float().squeeze(1)
        # print(test_predictions)
        testF1Score = f1(test_predictions, y_up_test_tensor)  # computing F1 score
        testPrecisionScore = prec(test_predictions, y_up_test_tensor)
        testRecallScore = recall(test_predictions, y_up_test_tensor)

        print(
            "avgval avg prec/f1/recall:  ",
            avg_val_precision_score,
            avg_val_f1_score,
            avg_val_recall_score,
        )
        print(
            "test prec/f1/recall: ",
            testPrecisionScore.item(),
            testF1Score.item(),
            testRecallScore.item(),
        )

        return (
            best_val_loss,
            avg_val_f1_score,
            avg_val_precision_score,
            best_model_state_dict,
            testF1Score,
            testPrecisionScore,
            best_epoch,
        )
        # Return the best F1 score after all epochs


    def train_final_model(hparams, Xtrainval, ytrainval):
        positivecase_weight_up = hparams["positivecase_weight_up"]
        weight_positive_up = (
            num_negative_up_trainval / num_positive_up_trainval
        ) * positivecase_weight_up
        best_model_state_dict = None
        model = DynamicNNwithDropout(
            X_train.shape[1], hparams["layers"], hparams["dropout_rate"]
        ).to(device)

        model.train()
        weight = torch.Tensor([weight_positive_up]).to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=weight)
        optimizer_name = hparams["optimizer"]
        learning_rate = hparams["learning_rate"]

        if optimizer_name == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
        elif optimizer_name == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name == "RMSprop":
            optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
        elif optimizer_name == "Adagrad":
            optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)

        num_epochs = hparams["num_epochs"]
        batch_size = hparams["batch_size"]

        for epoch in range(num_epochs):
            # Training step
            model.train()
            for i in range(0, len(Xtrainval), batch_size):
                X_batch = Xtrainval[i : i + batch_size]
                y_batch = ytrainval[i : i + batch_size]

                y_batch = y_batch.unsqueeze(1)  # was wrong shape?
                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

        best_model_state_dict = model.state_dict()

        return best_model_state_dict


    # Define Optuna Objective
    def objective(trial):
        # Define the hyperparameter search space
        learning_rate = trial.suggest_float(
            "learning_rate", 0.0005, 0.007, log=True
        )  # 0003034075497582067
        num_epochs = trial.suggest_int("num_epochs", 100, 1000)  # 3800 #230  291
        batch_size = trial.suggest_int("batch_size", 1000, 3500)  # 10240  3437
        # Add more parameters as needed
        # TODO the rounds with SGD seemed to be closer val/test. values.
        optimizer_name = trial.suggest_categorical(
            "optimizer", ["Adam", "SGD"]
        )  # ,"RMSprop", "Adagrad"
        dropout_rate = trial.suggest_float(
            "dropout_rate", 0, 0.2
        )
        # using layers now instead of setting num_hidden.
        n_layers = trial.suggest_int("n_layers", 1, 4)
        layers = []
        for i in range(n_layers):
            layers.append(trial.suggest_int(f"n_units_l{i}", 32, 256))
        positivecase_weight_up = trial.suggest_float(
            "positivecase_weight_up", 1, 2
        )

        # Call the train_model function with the current hyperparameters
        (
            best_val_loss,
            f1_score,
            prec_score,
            best_model_state_dict,
            testF1Score,
            testPrecisionScore,
            best_epoch,
        ) = train_model(
            {
                "learning_rate": learning_rate,
                "optimizer": optimizer_name,  # Include optimizer name here
                "num_epochs": num_epochs,
                "batch_size": batch_size,
                "dropout_rate": dropout_rate,
                # "num_hidden_units": num_hidden_units,
                "positivecase_weight_up": positivecase_weight_up,
                "layers": layers
                # Add more hyperparameters as needed
            },
            X_train_tensor,
            y_up_train_tensor,
            X_val_tensor,
            y_up_val_tensor,
        )
        alpha = 0.5

        blended_score = (
            (alpha * (1 - prec_score))
            + ((1 - alpha) * (1 - f1_score))
            + (alpha * (1 - testPrecisionScore))
            + ((1 - alpha) * (1 - testF1Score))
        )

        # return best_val_loss
        return blended_score
        # return prec_score  # Optuna will try to maximize this value


    ##Comment out to skip the hyperparameter selection.  Swap "best_params".
    try:
        study = optuna.load_study(
            study_name=f"{study_name}", storage=f"sqlite:///{study_name}.db"
        )
        print("Study Loaded.")
        try:
            best_params_up = study.best_params
            best_trial_up = study.best_trial
            best_value_up = study.best_value
            print("Best Value_up:", best_value_up)
            print(best_params_up)
            print("Best Trial_up:", best_trial_up)
        except Exception as e:
            print(e)
    except KeyError:
        study = optuna.create_study(
            direction="minimize",
            study_name=f"{study_name}",
            storage=f"sqlite:///{study_name}.db",
        )
    "Keyerror, new optuna study created."  #

    study.optimize(
        objective, n_trials=n_trials
    )  # You can change the number of trials as needed

    best_params = study.best_params

    # best_params = set_best_params_manually
    print("Best Hyperparameters:", best_params)

    n_layers = best_params["n_layers"]
    layers = [best_params[f"n_units_l{i}"] for i in range(n_layers)]
    best_params["layers"] = layers
    ## Train the model with the best hyperparameters

    (
        best_val_loss,
        best_f1_score,
        best_prec_score,
        best_model_state_dict,
        testF1Score,
        testPrecisionScore,
        best_epoch,
    ) = train_model(
        best_params, X_train_tensor, y_up_train_tensor, X_val_tensor, y_up_val_tensor
    )
    best_params["num_epochs"] = best_epoch


    n_layers = best_params["n_layers"]
    layers = [best_params[f"n_units_l{i}"] for i in range(n_layers)]
    best_params["layers"] = layers
    (best_model_state_dict) = train_final_model(
        best_params, X_trainval_tensor, y_trainval_tensor
    )

    finalmodel = DynamicNNwithDropout(
        X_train.shape[1], best_params["layers"], best_params["dropout_rate"]
    ).to(device)
    # Load the saved state_dict into the model
    finalmodel.load_state_dict(best_model_state_dict)
    finalmodel.eval()
    feature_imp = feature_importance(finalmodel, X_trainval_tensor, y_trainval_tensor)
    print("Feature Importances:", feature_imp)
    predicted_probabilities_up = finalmodel(X_test_tensor).detach().cpu().numpy()
    predicted_probabilities_up = (predicted_probabilities_up > theshhold_down).astype(int)
    predicted_up_tensor = (
        torch.tensor(predicted_probabilities_up, dtype=torch.float32).squeeze().to(device)
    )

    num_positives_up = np.sum(predicted_probabilities_up)
    task = "binary"
    precision_up = Precision(num_classes=2, average="weighted", task="binary").to(device)(
        predicted_up_tensor, y_up_test_tensor
    )  # move metric to same device as tensors
    accuracy_up = Accuracy(num_classes=2, average="weighted", task=task).to(device)(
        predicted_up_tensor, y_up_test_tensor
    )
    recall_up = Recall(num_classes=2, average="weighted", task=task).to(device)(
        predicted_up_tensor, y_up_test_tensor
    )
    f1_up = F1Score(num_classes=2, average="weighted", task=task).to(device)(
        predicted_up_tensor, y_up_test_tensor
    )


    # Print Number of Positive and Negative Samples
    num_positive_samples_up = sum(y_up_test_tensor)
    num_negative_samples_up = len(y_up_test_tensor) - num_positive_samples_up


    print("Metrics for Target_Up:", "\n")
    print("Precision:", precision_up)
    print("Accuracy:", accuracy_up)
    print("Recall:", recall_up)
    print("F1-Score:", f1_up, "\n")

    print("Best Hyperparameters:", best_params)
    print(
        f"Number of positive predictions for 'up': {sum(x[0] for x in predicted_probabilities_up)}"
    )
    print("Number of Positive Samples(Target_Up):", num_positive_samples_up)
    print(
        "Number of Total Samples(Target_Up):",
        num_positive_samples_up + num_negative_samples_up,
    )



    # Save the models using joblib
    input_val = input("Would you like to save these models? y/n: ").upper()
    if input_val == "Y":
        current_datetime = (
            datetime.now().strftime("%y%m%d%H%M"))
        study_name=config['study_name']
        # model_summary = input("Save this set of models as: ")
        model_name=study_name +'_'+ current_datetime
        model_directory = os.path.join("../../../../../Trained_Models", f"{model_name}")

        if not os.path.exists(model_directory):
            os.makedirs(model_directory)
        model_filename_up = os.path.join(model_directory, "target_up.pth")

        torch.save(
            {
                "features": Chosen_Predictor,
                "input_dim": X_train_tensor.shape[1],
                "dropout_rate": best_params["dropout_rate"],
                "layers": best_params["layers"],
                "model_state_dict": finalmodel.state_dict(),
                "scaler_X": finalscaler_X,
            },
            model_filename_up,
        )
        # Save the scaler

        # Generate the function definition
        function_def = f"""
def {model_name}(new_data_df):
    checkpoint = torch.load(f'{{base_dir}}/{model_name}/target_up.pth', map_location=torch.device('cpu'))
    features = checkpoint['features']
    dropout_rate = checkpoint['dropout_rate']
    input_dim = checkpoint['input_dim']
    layers = checkpoint['layers']
    scaler_X = checkpoint['scaler_X']

    loaded_model = DynamicNNwithDropout(input_dim, layers, dropout_rate)
    loaded_model.load_state_dict(checkpoint['model_state_dict'])
    loaded_model.eval()

    tempdf = new_data_df.copy()
    # tempdf.dropna(subset=features, inplace=True)
    tempdf = tempdf[features]

    very_large_number = 1e15  # Placeholder for positive infinity
    very_small_number = -1e15  # Placeholder for negative infinity

    for col in tempdf.columns:
        # Replace positive and negative infinity with the defined large and small numbers. this reaplce the 1.5x multiplier logic.
        tempdf[col].replace([np.inf, -np.inf], [very_large_number, very_small_number], inplace=True)
        

    tempdf = pd.DataFrame(tempdf.values, columns=features, index=tempdf.index)

    # scale the new data features and generate predictions

    scaled_features = scaler_X.transform(tempdf)
    input_tensor = torch.tensor(scaled_features, dtype=torch.float32)
    predictions = loaded_model(input_tensor)
    predictions_prob = torch.sigmoid(predictions)
    predictions_numpy = predictions_prob.detach().numpy()
    prediction_series = pd.Series(predictions_numpy.flatten(), index=tempdf.index)

    result = new_data_df.copy()
    result["Predictions"] = np.nan
    result.loc[prediction_series.index, "Predictions"] = prediction_series
    return result["Predictions"], 0.5, 0.5, 5, 20
    """

        # Append the new function definition to pytorch_trained_minute_models.py
        with open(
            "../../../../../Trained_Models/pytorch_trained_minute_models.py", "a"
        ) as file:
            file.write(function_def)
        with open(
            f"../../../../../Trained_Models/{model_name}/info.txt", "w"
        ) as info_txt:
            info_txt.write("This file contains information about the model.\n\n")
            info_txt.write(
                f"File analyzed: {DF_filename}\nCells_Foward_to_check: {cells_forward_to_check}\n\n"
            )
            info_txt.write(
                f"Metrics for Target_Up:\nPrecision: {precision_up}\nAccuracy: {accuracy_up}\nRecall: {recall_up}\nF1-Score: {f1_up}\n"
            )
            info_txt.write(
                f"Predictors: {Chosen_Predictor}\n\n\n"
                f"Best Params: {best_params}\n\n\n"
                f"Number of Positive Samples (Target_Up): {num_positive_samples_up}\nNumber of Negative Samples (Target_Up): {num_negative_samples_up}\n"
                f"Threshold Up (sensitivity): {theshhold_down}\n"
                f"Target Underlying Percentage Up: {percent_down}\n"
                f"Anticondition: {anticondition_UpCounter}\n"
            )
    #TODO the template for model needs to look like this
    """def MSFT_2hr_50pct_Down_PTNNclass(new_data_df):
        checkpoint = torch.load(f'{base_dir}/MSFT_2hr_50pct_Down_PTNNclass/target_up.pth', map_location=torch.device('cpu'))
        features = checkpoint['features']
        dropout_rate = checkpoint['dropout_rate']
        input_dim = checkpoint['input_dim']
        layers = checkpoint['layers']
        scaler_X = checkpoint['scaler_X']
    
        loaded_model = DynamicNNwithDropout(input_dim, layers, dropout_rate)
        loaded_model.load_state_dict(checkpoint['model_state_dict'])
        loaded_model.eval()
    
        tempdf = new_data_df.copy()
        # tempdf.dropna(subset=features, inplace=True)
        tempdf = tempdf[features]
    
        very_large_number = 1e15  # Placeholder for positive infinity
        very_small_number = -1e15  # Placeholder for negative infinity
    
        for col in tempdf.columns:
            # Replace positive and negative infinity with the defined large and small numbers. this reaplce the 1.5x multiplier logic.
            tempdf[col].replace([np.inf, -np.inf], [very_large_number, very_small_number], inplace=True)
            
    
        tempdf = pd.DataFrame(tempdf.values, columns=features, index=tempdf.index)
    
        # scale the new data features and generate predictions
    
        scaled_features = scaler_X.transform(tempdf)
        input_tensor = torch.tensor(scaled_features, dtype=torch.float32)
        predictions = loaded_model(input_tensor)
        predictions_prob = torch.sigmoid(predictions)
        predictions_numpy = predictions_prob.detach().numpy()
        prediction_series = pd.Series(predictions_numpy.flatten(), index=tempdf.index)
    
        result = new_data_df.copy()
        result["Predictions"] = np.nan
        result.loc[prediction_series.index, "Predictions"] = prediction_series
        return result["Predictions"], 0.5, 0.5, 5, 20
    """
tickers = 'SPY','msft','tsla' # This can be dynamically set
for ticker in tickers:
    try:
        ticker = ticker.upper()
        create_model(ticker)
    except Exception as e:
        print(e)