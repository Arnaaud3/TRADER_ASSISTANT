import pandas as pd
import matplotlib.pyplot as plt
from mplfinance.plotting import plot

class Reader_bitcoin_dataset:

    def __init__(self,fileName,flagPlot = True):
        self.bitcoinDataset = self.read_bitcoinDataFile(fileName)
        if flagPlot:
            self.convert_date_in_datetime()
            self.plot_bitcoin_dataset()

    def read_bitcoinDataFile(self,fileName):
        return pd.read_csv(fileName)
    
    def convert_date_in_datetime(self):
        self.bitcoinDataset.index = pd.to_datetime(self.bitcoinDataset["Date"])

    def plot_bitcoin_dataset(self):
        plot(self.bitcoinDataset,type="candle",style='charles', title='BTCUSD OHLC Chart',
         ylabel='Price', ylabel_lower='Volume', volume=True,datetime_format = '%m' )


if __name__ == "__main__":
    fileName = "BTC_USD_1D/BTCUSD_HISTORICAL_1D_2015-01-01_2015-12-31.csv"
    reader_bitcoin_dataset = Reader_bitcoin_dataset(fileName)
    print(reader_bitcoin_dataset.bitcoinDataset.head())


    
