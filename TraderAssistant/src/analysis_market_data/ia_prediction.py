from stock_market_data import historical_market_data as hmd
from stock_market_data import preprocess_market_data as pmd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,mean_absolute_percentage_error,r2_score,PredictionErrorDisplay
from sklearn.model_selection import learning_curve,validation_curve,LearningCurveDisplay,ValidationCurveDisplay
from sklearn.ensemble import GradientBoostingRegressor


class IA_Prediction:
    """Class to make prediction of the stock market. 
    """
    
    def __init__(self,X_train,X_test,y_train,y_test,moving_average,dp):
        """Instanciate the IA_Prediction class with the train dataset, the test dataset and the cross-validation dataset

        Args:
            train_dataset (pd.DataFrame): dataFrame that contain the train dataset
            test_dataset (pd.DataFrame): dataFrame that contain the test dataset
            cv_dataset (list[tuple]): list that contains tuple of dataFrame cross-validation train data and cross-validation test data
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.moving_average = moving_average
        self.dp = dp
    
    def linear_regression_predict_trend(self):
        model_linear_reg = LinearRegression(fit_intercept=False)
        model_linear_reg.fit(self.X_train,self.y_train)
        return model_linear_reg
    
    def linear_regression_prediction_train(self,model_linear_reg):
        """Generate multi-linear regression to predict the moving average of the stock market.
        The goal is to estimate the trend of the stock market using linear regression.

        Returns:
            pd.DataFrame: contain the prediction of the stock market on the X test data
        """
        
        y_prediction = pd.Series(model_linear_reg.predict(self.X_train)[:,0],index=self.X_train.index)
        print(f"mean squared error is : {mean_squared_error(model_linear_reg.predict(self.X_train),self.y_train)}\n")
        print(f"R2 score is : {r2_score(model_linear_reg.predict(self.X_train),self.y_train)}")
        return y_prediction
    
    def linear_regression_prediction_test(self,model_linear_reg):
        """Generate multi-linear regression to predict the moving average of the stock market.
        The goal is to estimate the trend of the stock market using linear regression.

        Returns:
            pd.DataFrame: contain the prediction of the stock market on the X test data
        """
        y_prediction = pd.Series(model_linear_reg.predict(self.X_test)[:,0],index=self.X_test.index)
        print(f"mean squared error is : {mean_squared_error(model_linear_reg.predict(self.X_test),self.y_test)}\n")
        print(f"R2 score is : {r2_score(model_linear_reg.predict(self.X_test),self.y_test)}")
        return y_prediction
    
    def plot_moving_average_prediction_train_dataset(self,y_prediction):
        plt.figure()
        plt.plot(self.y_train.index,self.y_train)
        plt.plot(y_prediction.index,y_prediction)
        plt.xlabel("Date")
        plt.ylabel("Adj. Close Price")
        plt.show()
        return None
    
    def plot_moving_average_prediction_test_dataset(self,y_prediction):
        plt.figure()
        plt.plot(self.y_test.index,self.y_test)
        plt.plot(y_prediction.index,y_prediction)
        plt.xlabel("Date")
        plt.ylabel("Adj. Close Price")
        plt.show()
        return None
    
    def ia_trend_forecasting(self,model,nbr_days_to_forecast=7):
        last_day_before_forecast = self.dp.index[-1]
        days_to_forecast = pd.date_range(start=last_day_before_forecast,periods=8,freq='D')[1:]
        X_fore = self.dp.out_of_sample(nbr_days_to_forecast)
        X_fore.index = days_to_forecast
        y_forecast = pd.Series(model.predict(X_fore)[:,0],index=X_fore.index)
        return y_forecast
    
    def plot_trend_forecasting(self,y_prediction_train,y_prediction_test,y_forecast):
        plt.figure()
        plt.plot(self.y_train.index,self.y_train)
        plt.plot(y_prediction_train.index,y_prediction_train)
        plt.plot(self.y_test.index,self.y_test)
        plt.plot(y_prediction_test.index,y_prediction_test)
        plt.plot(y_forecast.index,y_forecast)
        plt.show()
        # new figure to have a zoom on the forecast zone
        plt.figure()
        plt.plot(self.y_test.index,self.y_test)
        plt.plot(y_prediction_test.index,y_prediction_test)
        plt.plot(y_forecast.index,y_forecast)
        plt.show()
    
        return None
    
    def XGBoost_prediction(self):
        """Generate XGBoost Algorithm regression to predict the stock market value

        Returns:
            pd.DataFrame: contain the prediction of the stock market of the X test data.
        """
        model_xgboost = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
        model_xgboost.fit(self.X_train,self.y_train)
        y_prediction = model_xgboost.predict(self.X_test)
        print(f"mean squared error is : {mean_squared_error(y_prediction,self.y_test)}\n")
        print(f"R2 score is : {r2_score(y_prediction,self.y_test)}")
        return y_prediction
        
    def plot_prediction_test_dataset(self,y_predictions):
        """Plot the prediction data with the real stock market value.

        Args:
            y_predictions (dict): prediction of the X test data for a given model. The prediction is a pd.DataFrame
        """
        plt.figure(figsize=(10,5))
        plt.plot(self.X_test.index, self.y_test, label='Actual')
        for model,y_pred in y_predictions.items():
            plt.plot(y_pred.index, y_pred, label=f"{model}_prediction")
        plt.xlabel('Date')
        plt.ylabel('Adj Close Price')
        plt.title('Stock Price Prediction using Linear Regression')
        plt.legend()
        plt.show()
    
        
if __name__ == "__main__":
    data = hmd.Historical_market_data("BTC-USD","1d").marketData
    preprocess_data = pmd.Preprocess_market_data(data)
    idx_train,idx_test = preprocess_data.create_train_test_indexes()
    X_train,X_test,y_train,y_test = preprocess_data.create_train_test_dataset(idx_train,idx_test)
    moving_average = preprocess_data.moving_average
    dp = preprocess_data.dp
    ia_pred = IA_Prediction(X_train,X_test,y_train,y_test,moving_average,dp) 
    linear_reg_model = ia_pred.linear_regression_predict_trend()
    print("Prediction from a linear regression : \n")
    y_prediction = {"LinearRegressor_train_dataset":ia_pred.linear_regression_prediction_train(linear_reg_model),
                    "LinearRegressor_test_dataset":ia_pred.linear_regression_prediction_test(linear_reg_model)}
    # print(y_prediction)
    # ia_pred.plot_moving_average_prediction_train_dataset(y_prediction["LinearRegressor_train_dataset"])
    # ia_pred.plot_moving_average_prediction_test_dataset(y_prediction["LinearRegressor_test_dataset"])
    print('\n')
    y_forecast = ia_pred.ia_trend_forecasting(linear_reg_model)
    ia_pred.plot_trend_forecasting(y_prediction["LinearRegressor_train_dataset"],y_prediction["LinearRegressor_test_dataset"],y_forecast)