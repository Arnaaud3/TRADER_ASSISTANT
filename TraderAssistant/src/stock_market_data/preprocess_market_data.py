from stock_market_data import historical_market_data as hmd
from mplfinance.plotting import plot
from mplfinance.plotting import make_addplot
import pandas as pd 
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.deterministic import DeterministicProcess

class Preprocess_market_data:
    """Class to preprocess market data. It requiere marketData as a pd.DataFrame to be initialize. 
    """
    def __init__(self,marketData:pd.DataFrame,flagPlot:bool=False):
        """Instanciate the Preprocess class the dataframe.

        Args:
            marketData (pd.DataFrame): contain stock market data we want to preprocess
            flagPlot (bool, optional): True if plot of the data to be generated. Defaults to False.
        """
        self.marketData = marketData
        # convert the date in the dataframe into datetime format
        self.convert_date_in_datetime()
        self.drop_useless_column()
        if flagPlot:
            self.plot_marketData()
            self.plot_moving_average_market_data()
        self.moving_average = self.moving_average_market_data()
        self.dp = self.deterministic_process()
            
    def convert_date_in_datetime(self):
        """Convert the date in the index of the dataframe into a datetime format
        """
        self.marketData.index = pd.to_datetime(self.marketData.index)
    
    def plot_marketData(self):
        """Build a candle plot with the market data contain in the dataframe
        """
        plot(self.marketData,type="candle",style='charles', title="OHLC Chart",
         ylabel='Price', ylabel_lower='Volume', volume=True,datetime_format = '%y' )

    def create_subset_dataframes(self):
        """Generate multiple subset of the dataframe

        Returns:
            dict: dict containing multitple dataset for each month
        """
        # Create a dictionary to hold each month's DataFrame
        monthly_dfs = {}
        # Group by year and month
        grouped = self.market.marketData.groupby([self.market.marketData.index.year, self.market.marketData.index.month])
        # Iterate over the grouped object and create a DataFrame for each month
        for (year, month), group in grouped:
            monthly_dfs[f'{year}-{month:02d}'] = group
        return monthly_dfs
    
    def drop_useless_column(self):
        self.marketData = self.marketData.drop(["Open","High","Low","Close","Volume"],axis=1)
        return None
    
    def create_time_feature(self):
        self.marketData['Time'] = np.arange(len(self.marketData.index))

    def moving_average_market_data(self,window=365,center=True,min_periods=183):
        """Compute moving average of the market data

        Args:
            window (int, optional): Window on which the moving average is computed. Defaults to 365.
            center (bool, optional): Centerize of not the window. Defaults to True.
            min_periods (int, optional): minimum period on which the moving average is computed. Defaults to 183.

        Returns:
            _type_: _description_
        """
        moving_average = self.marketData.rolling(
            window=window,
            center=center,
            min_periods=min_periods
        ).mean()
        return moving_average
        
    def plot_moving_average_market_data(self):
        """Plot the moving average of the market data

        Args:
            moving_average (_type_): _description_
        """
        ax = self.marketData.plot(style='.',color='0.5')
        self.moving_average.plot(ax=ax,linewidth=3,title='Moving average of the market data',legend=False)
        return None
    
    def deterministic_process(self):
        """Create trend features

        Returns:
            DeterministicProcess: _description_
        """
        dp = DeterministicProcess(
            index = self.marketData.index,
            constant=True,
            order=2,
            drop=True
        )
        return dp
    
    def create_features_in_sample(self):
        Y = self.marketData.copy()
        X = self.dp.in_sample()
        return X,Y
    
    def create_train_test_indexes(self):
        idx_train,idx_test = train_test_split(self.marketData.index, test_size=14, shuffle=False)
        return idx_train,idx_test
    
    def create_train_test_dataset(self,idx_train,idx_test):
        X,Y = self.create_features_in_sample()
        X_train,X_test = X.loc[idx_train],X.loc[idx_test]
        y_train,y_test = Y.loc[idx_train],Y.loc[idx_test]
        return X_train,X_test,y_train,y_test
    
    def create_train_test_data(self):
        """Create train test dataset in the dataset using Train Test Split from sklearn
        """
        self.marketData = self.marketData.drop(["Open","High","Low","Close","Volume"],axis=1)
        self.marketData['Time'] = np.arange(len(self.marketData.index))
        X = self.marketData.loc[:,["Time"]]
        y = self.marketData.loc[:,["Adj Close"]]
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=14,shuffle=False)
        return X_train,X_test,y_train,y_test
        
    def generate_train_test_data(self,test_size=7,n_splits=10):
        """Function to create train,test and cross validation dataset.

        Args:
            test_size (int, optional): size of the prediction we want to make. By default, prediciton is done weekly. Defaults to 7.
            n_splits (int, optional): number of split of the dataset. Defaults to 10.

        Returns:
            tuple : train dataset, test dataset and cross-validation dataset.
        """
        train_dataset = self.marketData[:-test_size]
        test_dataset = self.marketData[-test_size:]
        # create a dict to pack the cv datasets
        cv_dataset = list()
        # initialise the TimeSeriesSplit object
        tscv = TimeSeriesSplit(n_splits=n_splits,test_size=test_size)
        # create the split
        # all_split = list(tscv.split())
        for train_index,test_index in tscv.split(train_dataset):
            train_cv,test_cv = train_dataset.iloc[train_index],train_dataset.iloc[test_index]
            cv_dataset.append((train_cv,test_cv))
        return train_dataset,test_dataset,cv_dataset
        
if __name__ == '__main__':
    market = hmd.Historical_market_data("BTC-USD","1d")
    preprocess_data = Preprocess_market_data(market.marketData,flagPlot=False)
    idx_train,idx_test = preprocess_data.create_train_test_indexes()
    X_train,X_test,y_train,y_test = preprocess_data.create_train_test_dataset(idx_train,idx_test)

