import pandas as pd
from stock_market_data import historical_market_data

class Signal_analysis:
    
    def __init__(self,marketData:pd.DataFrame):
        self.marketData = marketData
        
    def simple_moving_average(self,period):
        self.marketData["SMA"] = self.marketData["Close"].rolling(window=period).mean()
        
    def exponential_moving_average(self,com:float):
        self.marketData["EMA"] = self.marketData["Close"].ewm(com=com)
        
        
if __name__ == "__main__":
    print("hello")