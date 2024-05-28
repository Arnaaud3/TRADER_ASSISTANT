from stock_market_data import historical_market_data as hmd
from stock_market_data import preprocess_market_data as pmd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from scipy.signal import argrelextrema

class Pattern_detection:
    
    def __init__(self,marketData:pd.DataFrame):
        self.marketData = self.move_date_from_index(marketData)
        self.minimas = self.create_ascendant_trendline()
        self.maximas = self.create_descendant_trendline()

    def move_date_from_index(self,marketData):
        marketData.reset_index(drop=False,inplace=True)
        return marketData
    
    def create_ascendant_trendline(self):
        self.marketData['min'] = self.marketData.iloc[argrelextrema(self.marketData['Close'].values, np.less_equal, order=5)[0]]['Close']
        minimas = self.marketData.dropna(subset=["min"])
        # linear regression
        x = np.arange(len(self.marketData))
        x_low = x[minimas.index]
        y_low = minimas["Close"].values
        coefficient = np.polyfit(x_low,y_low,1)
        trendline = np.poly1d(coefficient)
        self.marketData["Trendline_ascendant"] = trendline(x)
        return minimas
        
    def create_descendant_trendline(self):
        self.marketData['max'] = self.marketData.iloc[argrelextrema(self.marketData['Close'].values, np.greater_equal, order=5)[0]]['Close']
        maximas = self.marketData.dropna(subset=["max"])
        # linear regression
        x = np.arange(len(self.marketData))
        x_max = x[maximas.index]
        y_max = maximas["Close"].values
        coefficient = np.polyfit(x_max,y_max,1)
        trendline = np.poly1d(coefficient)
        self.marketData["Trendline_descendant"] = trendline(x)
        return maximas
        
    def plot_trendline(self):
        # Visualiser le signal et la ligne de tendance
        plt.figure(figsize=(14, 7))
        plt.plot(self.marketData['Date'], self.marketData['Close'], label='Close_price')
        plt.scatter(self.minimas['Date'], self.minimas['Close'], color='blue', label='Low point')
        plt.scatter(self.maximas['Date'], self.maximas['Close'], color='red', label='High point')
        plt.plot(self.marketData['Date'], self.marketData['Trendline_ascendant'], color='green', linestyle='-.', label='Ligne de tendance ascendante')
        plt.plot(self.marketData['Date'], self.marketData['Trendline_descendant'], color='green', linestyle='--', label='Ligne de tendance descendante')
        plt.title('Signal with trendline')
        plt.xlabel('Date')
        plt.ylabel('Close price')
        plt.legend()
        plt.show()

        
if __name__ == "__main__":
    data = hmd.Historical_market_data("BTC-USD","1d").marketData
    pattern_detection = Pattern_detection(data)
    pattern_detection.plot_trendline()