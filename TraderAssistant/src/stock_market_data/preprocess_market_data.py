from stock_market_data import historical_market_data as hmd
from mplfinance.plotting import plot
from mplfinance.plotting import make_addplot
import pandas as pd 
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

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
        if flagPlot:
            self.plot_marketData()
            
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
    preprocess_data = Preprocess_market_data(market.marketData)
    train,test,cv = preprocess_data.generate_train_test_data()
        
