BuyHistA1 = A1_Buy_historical_prediction(dailyminutes_df[['Bonsai Ratio', 'Bonsai Ratio 2', 'B1/B2', 'ITM PCR-Vol',
       'ITM PCRv Up2', 'ITM PCRv Down2', 'ITM PCRoi Up2']])
dailyminutes_df['BuyHistA1'] = BuyHistA1
SellHistA1 = A1_Sell_historical_prediction(dailyminutes_df[['B1/B2', 'ITM PCRoi Down2']])
dailyminutes_df['SellHistA1'] = SellHistA1