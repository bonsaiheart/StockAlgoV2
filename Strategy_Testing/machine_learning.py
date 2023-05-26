
import PrivateData.tradier_info as private
from pytradier.tradier import Tradier
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import os
import matplotlib.pyplot as plt
# if os.path.exists("ML_DF.csv"):
#     ml_dataframe = pd.read_csv("ML_DF.csv", index_col=0)
from sklearn.ensemble import RandomForestClassifier
# else:
#     ml_dataframe = yf.Ticker("^GSPC")
#     ml_dataframe = ml_dataframe.history(start="2023-05-16", end="2023-05-23", interval="1m")
#     ml_dataframe.to_csv("ML_DF.csv")
# ml_dataframe.index = pd.to_datetime(ml_dataframe.index)

###TODO ENTER TICKER HERE
ml_dataframe = pd.read_csv('dailyDF/TSLA.csv')


ml_dataframe = ml_dataframe[['ExpDate', 'date', 'time', 'Current Stock Price',
       'Current SP % Change(LAC)',  'Bonsai Ratio',
       'Bonsai Ratio 2',
        'PCR-Vol', 'PCR-OI',  'ITM PCR-Vol',
       'Up or down', 'b1/b2',
       '6 hour later change %', '5 hour later change %',
       '4 hour later change %', '3 hour later change %',
       '2 hour later change %', '1 hour later change %',
       '45 min later change %', '30 min later change %',
       '20 min later change %', '15 min later change %',
       '10 min later change %', '5 min later change %']]
# ml_dataframe.plot.line(y="Close", use_index=True)
# plt.show()
# del ml_dataframe["Dividends"]
# del ml_dataframe["Stock Splits"]
# ml_dataframe["Tomorrow"] = ml_dataframe["Close"].shift(-1)

ml_dataframe.dropna(subset=["30 min later change %"], inplace=True)
ml_dataframe["Target"] = (ml_dataframe["30 min later change %"] > 0).astype(int)
# start_date = pd.to_datetime('2020-06-02 00:00:00-05:00')
print(ml_dataframe.columns)
ml_dataframe = ml_dataframe.loc[0:].copy()
print(ml_dataframe.columns)

ml_dataframe.to_csv('wheresthefnblanks.csv')
model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
print(ml_dataframe.columns)
train = ml_dataframe.iloc[:-100]
print("hello mante",train.columns)
test = ml_dataframe.iloc[-100:]
print(ml_dataframe.iloc[-100:])
print("hello mante",test.columns)
predictors = ["b1/b2"]
model.fit(train[predictors], train["Target"])
from sklearn.metrics import precision_score

preds = model.predict(test[predictors])
preds = pd.Series(preds, index=test.index)
precision_score(test["Target"], preds)
combined = pd.concat([test["Target"], preds], axis=1)
combined.plot()
# plt.show()


def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict(test[predictors])
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined


# def backtest(data, model, predictors, start=2500, step=250):
def backtest(data, model, predictors, start=100, step=10
             ):
    all_predictions = []

    for i in range(start, data.shape[0], step):
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i + step)].copy()
        predictions = predict(train, test, predictors, model)
        all_predictions.append(predictions)

    return pd.concat(all_predictions)


predictions = backtest(ml_dataframe, model, predictors)
predictions["Predictions"].value_counts()
precision_score(predictions["Target"], predictions["Predictions"])
predictions["Target"].value_counts() / predictions.shape[0]
horizons = [2, 5, 60, 250, 1000]
new_predictors = []

for horizon in horizons:
    rolling_averages = ml_dataframe.rolling(horizon).mean()

    ratio_column = f"Close_Ratio_{horizon}"
    ml_dataframe[ratio_column] = ml_dataframe["b1/b2"] / rolling_averages["b1/b2"]

    trend_column = f"Trend_{horizon}"
    ml_dataframe[trend_column] = ml_dataframe.shift(1).rolling(horizon).sum()["Target"]

    new_predictors += [ratio_column, trend_column]
ml_dataframe = ml_dataframe.dropna(subset=ml_dataframe.columns[ml_dataframe.columns != "30 min later change %"])
print("sp", ml_dataframe)
ml_dataframe.to_csv("ml_dataframe.csv")
model = RandomForestClassifier(n_estimators=100, min_samples_split=50, random_state=1)
def predict(train, test, predictors, model):
    model.fit(train[predictors], train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >=.6] = 1
    preds[preds <.6] = 0
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined
predictions = backtest(ml_dataframe, model, new_predictors)
predictions["Predictions"].value_counts()
print('predictions["Predictions"].value_counts()',predictions["Predictions"].value_counts())
precision_score(predictions["Target"], predictions["Predictions"])
print('prec_score',precision_score(predictions["Target"], predictions["Predictions"]))
predictions["Target"].value_counts() / predictions.shape[0]
print(f'predictions["Target"].value_counts() / predictions.shape[0]',predictions["Target"].value_counts() / predictions.shape[0])
predictions.to_csv("predictions.csv")
print("himnom",predictions)