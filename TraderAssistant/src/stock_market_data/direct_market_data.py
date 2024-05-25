import yfinance as yf
import time
import mplfinance as mpf
import pandas as pd
import datetime as datetime


class Direct_market_data:
    
    def __init__(self,marketName:str,timeInterval:str,period:str,plotFlag:bool=True,updateTime:int=10):
        self.market = yf.Ticker(marketName)
        self.marketName = marketName
        self.period = period
        self.timeInterval = timeInterval
        self.fig = None
        self.ax = None
        while True:
            marketData = self.fetch_real_time_data()
            print(marketData)
            if plotFlag:
                self.plot_market_data(marketData)
            time.sleep(updateTime)
        
    def fetch_real_time_data(self) -> pd.DataFrame:
        marketData = yf.download(self.marketName,period="1d",interval="1m")
        return marketData
            
    def plot_market_data(self,marketData:pd.DataFrame):
        if self.fig is None:
            self.fig,self.ax = mpf.plot(marketData[::60],type="candle",style='charles', title=f"{self.marketName} OHLC Chart",
         ylabel='Price', ylabel_lower='Volume', volume=True,block=False,returnfig=True)
        else:
            self.ax[0].clear()
            self.fig,self.ax = mpf.plot(marketData[::60], type='candle', style='charles', ax=self.ax[0],
                                        volume=self.ax[1],block=False,returnfig=True)
            return None
        
if __name__ == "__main__":
    direct_market = Direct_market_data(marketName= "BTC-USD",timeInterval="1m",period="1d")
    