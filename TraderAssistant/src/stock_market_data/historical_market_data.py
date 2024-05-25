import yfinance as yf
import pandas as pd
import datetime as datetime
import os


class Historical_market_data:
    
    CURRENT_DATE = datetime.datetime.now()
    STARTING_YEAR_RECORD_MARKET_DATA = 2015
    DAY_DELTA_15MIN_DATA = 60
    
    def __init__(self,marketName:str,timeInterval:str,flagWriteData:bool=False) -> None:
        """Initializes the class to retrieve the stock market data with market name

        Args:
            marketName (str): The ticker symbol of the market.
            timeInterval (str): The interval of the data (e.g., '1d' for daily data).
            flagWriteData (bool): Whether to write the data to a CSV file. Defaults to False.
        """
        self.marketName = marketName
        self.timeInterval = timeInterval
        self.marketData = self.retrieve_market_data()
        if flagWriteData:
            self.write_market_data_to_csv()
        
        
    def retrieve_market_data(self) -> pd.DataFrame:
        """Retrieves market data from Yahoo Finance.

        Returns:
            pd.DataFrame: The retrieved market data.
        """
        market_data_list = list()
        if self.timeInterval == "1d":
            for year in range(self.STARTING_YEAR_RECORD_MARKET_DATA,self.CURRENT_DATE.year):
                start = f"{year}-01-01"
                end = f"{year}-12-31"
                market_data_list.append(yf.download(self.marketName, start=start, end=end, interval=self.timeInterval))
            try:
                market_data_list.append(yf.download(self.marketName, start=start, end=end, interval=self.timeInterval))
            except Exception as e:
                print(f"Error downloading data for {year}: {e}")
            start = f"{self.CURRENT_DATE.year}-01-01"
            end = f"{self.CURRENT_DATE.year}-{self.CURRENT_DATE.month}-{self.CURRENT_DATE.day}"
            market_data_list.append(yf.download(self.marketName,start=start,end=end,interval=self.timeInterval))
            try:
                market_data_list.append(yf.download(self.marketName, start=start, end=end, interval=self.timeInterval))
            except Exception as e:
                print(f"Error downloading data for current year: {e}")
        marketData = pd.concat(market_data_list,ignore_index=False)
        return marketData
    
    def write_market_data_to_csv(self,marketDataFolderName:str="marketDataBase") -> None:
        """Writes the market data to a CSV file.

        Args:
            marketDataFolderName (str, optional): The name of the folder to save the CSV file. Defaults to "marketDataBase".
        """
        if not os.path.exists(marketDataFolderName):
            os.mkdir(marketDataFolderName)
        self.marketData.to_csv(f"{marketDataFolderName}/{self.marketName}-{self.timeInterval.upper()}.csv")
        return None
    
if __name__ == "__main__":
    data = Historical_market_data("BTC-USD","1d").marketData
    