"""
Author : Arnaud TECHER
Date : 12/04/2024
Goal : Build a simple model of the Bitcoin Stock Market. 
We want to make an IA to make the following classification : 
at each time step, we want to do the classification :
    1 : it is an entrance or exit point to trade
    0 : it is not an entrance/exit point. Keep position.
"""
import reader_bitcoin_data
from sklearn.model_selection import train_test_split,TimeSeriesSplit
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error,r2_score,PredictionErrorDisplay
from sklearn.model_selection import learning_curve,validation_curve,LearningCurveDisplay,ValidationCurveDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
import time
import datetime



class PreprocessingStockData:

    def __init__(self,fileName):
        self.stockDataset = reader_bitcoin_data.Reader_bitcoin_dataset(fileName=fileName).bitcoinDataset
        self.Y = self.is_gradient_change_size()
        print(self.get_nbr_features())
        print(self.get_nbr_samples())
        print(self.Y.shape)

    def get_nbr_samples(self):
        return int(self.stockDataset.shape[0])
    
    def get_nbr_features(self):
        return int(self.stockDataset.shape[1]-1)
    
    def compute_sign_gradient(self):
        price = self.stockDataset["Close"].array
        dates = pd.to_datetime(self.stockDataset["Date"])
        time = np.array([(date-dates.iloc[0]).total_seconds() for date in dates])
        return np.sign(np.gradient(price,time))

    def is_gradient_change_size(self):
        """Check if there is a change of sign in the gradient array.
        If there is a change return True, and False otherwise.


        Args:
            sign_gradient (_type_): _description_

        Returns:
            _type_: _description_
        """
        # build a copy of the array
        sign_gradient = self.compute_sign_gradient()
        res = np.copy(sign_gradient)
        for i in range(sign_gradient.shape[0]):
            if i == 0:
                # by default, the first value is set to false
                res[i] = False
            else:
                res[i] = sign_gradient[i] != sign_gradient[i-1]
        return res
    
    def generate_training_test_data(self,test_size=0.1):
        X_train,X_test,y_train,y_test = train_test_split(self.stockDataset,self.Y,test_size=test_size)
        return X_train,X_test,y_train,y_test
    

class Modelisation_in_out:

    def __init__(self,X_train,X_test,y_train,y_test):
        self.X_train = X_train
        self.X_test = X_test 
        self.y_train = y_train
        self.y_test = y_test

    def logisticRegressionModel(self):
        lreg = LogisticRegression(random_state=0).fit(self.X_train,self.y_train)
        print(lreg.coef_)
        return lreg


if __name__ == "__main__":
    fileName = r"C:\Users\arnaud\Documents\02_PROJET_PERSO\02_APPRENTISSAGE_PYTHON\01_IA_PROJECT\01_TRADING\BTC_USD_1D\BTCUSD_HISTORICAL_1D_2015-01-01_2015-12-31.csv"
    preprocess = PreprocessingStockData(fileName=fileName)
    X_train,X_test,y_train,y_test = preprocess.generate_training_test_data()
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)
    model_in_out = Modelisation_in_out(X_train,X_test,y_train,y_test)
    lreg = model_in_out.logisticRegressionModel()