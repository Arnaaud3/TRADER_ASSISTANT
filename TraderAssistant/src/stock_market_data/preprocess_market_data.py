import historical_market_data as hmd
from mplfinance.plotting import plot
import pandas as pd 

class Preprocess_market_data:
    
    def __init__(self,market,flagPlot=False):
        self.market = market
        if flagPlot:
            self.convert_date_in_datetime()
            self.plot_marketData()
            
    def convert_date_in_datetime(self):
        print(self.market.marketData)
        self.market.marketData.index = pd.to_datetime(self.market.marketData.index)
    
    def plot_marketData(self):
        plot(self.market.marketData,type="candle",style='charles', title=f"{self.market.marketName} OHLC Chart",
         ylabel='Price', ylabel_lower='Volume', volume=True,datetime_format = '%y' )


if __name__ == '__main__':
    market = hmd.Historical_market_data("BTC-USD","1d")
    preprocess_data = Preprocess_market_data(market,True)