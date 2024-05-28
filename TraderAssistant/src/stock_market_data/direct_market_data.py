import yfinance as yf
import time
import pandas as pd
import mplfinance as mpf
import threading
import queue
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class Direct_market_data:
    """Class to fetch real-time stock market data.
    """
    def __init__(self, marketName: str, timeInterval: str, period: str, plotFlag: bool = True, updateTime: int = 10):
        """Instanciate the Direct_market_data class. 

        Args:
            marketName (str): ticker of the stock market we want to fetch data ex. "BTC-USD"
            timeInterval (str): time interval on which the data are retrieved ex. "1m","1h"
            period (str): period on which the data are retrieved ex. "1d"
            plotFlag (bool, optional): True or False to plot the stock market data or not. Defaults to True.
            updateTime (int, optional): update time to fetch the data and replot the graph in second. Defaults to 10.
        """
        self.market = yf.Ticker(marketName)
        self.marketName = marketName
        self.period = period
        self.timeInterval = timeInterval
        self.plotFlag = plotFlag
        self.updateTime = updateTime
        self.fig = None # initialise the fig for the graph
        self.ax = None  # initialise the axis of the graph
        self.running = False    # flag to know if the thread to retrieved data is currently running
        self.data_thread = None #  thread where the data will be retrieved.
        self.latest_data = None # contains the dataFrame with the latest updated data
        self.data_queue = queue.Queue()  # Queue to pass data between threads
        self.data_condition = threading.Condition() # Condition to wait the data before plotting it

    def start_fetching_data(self):
        """Start a thread to fetch the real-time data
        """
        self.running = True 
        # create a thread where the data are fetch continuously
        self.data_thread = threading.Thread(target=self._fetch_data_continuously)
        # start the thread
        self.data_thread.start()
        # start to plot the data
        if self.plotFlag:
            self._start_plotting()

    def stop_fetching_data(self):
        """Stop the thread that fetch the real-time data
        """
        self.running = False
        if self.data_thread is not None:
            self.data_thread.join()

    def _fetch_data_continuously(self):
        """Fetch the real-time stock market data constinuously. This function is run in a different threading than the main one
        """
        while self.running:
            # retrieve the data while the thread is running
            marketData = self.fetch_real_time_data()
            with self.data_condition: 
                # replace the latest data retrieved 
                self.latest_data = marketData
                self.data_queue.put(marketData)  # Put the new data in the queue
                self.data_condition.notify_all()
            print(marketData)
            time.sleep(self.updateTime)
        
    def fetch_real_time_data(self) -> pd.DataFrame:
        """Retrieve the stock market data on yahoo finance and return a dataFrame containing the data

        Returns:
            pd.DataFrame: contain latest data available on yahoo finance
        """
        marketData = yf.download(self.marketName, period=self.period, interval=self.timeInterval)
        return marketData

    def _start_plotting(self):
        """Start plotting the stock market data in a candle grah with volumes. An animation is created and updated using the method _update_plot.
        """
        with self.data_condition:
            while self.latest_data is None:
                self.data_condition.wait()  # wait until the data is available
        self.fig, self.ax = mpf.plot(self.latest_data, type="candle", style='charles', title=f"{self.marketName} OHLC Chart",
                                        ylabel='Price', ylabel_lower='Volume', volume=True, block=False, returnfig=True)
        ani = animation.FuncAnimation(self.fig, self._update_plot, interval=self.updateTime * 1000,cache_frame_data=False)
        plt.show()
        
        
    def _update_plot(self, frame):
        """Update the animation created with the new data

        Args:
            frame (animation.FuncAnimation): animation created for plotting the data
        """
        if not self.data_queue.empty():
            marketData = self.data_queue.get()
            self.ax[0].cla()
            self.ax[1].cla()
            mpf.plot(marketData, type='candle', style='charles', ax=self.ax[0],
                     volume=self.ax[1], block=False)
            self.fig.canvas.draw()

    def get_latest_data(self) -> pd.DataFrame:
        """Get the latest dataFrame containing the stock market data

        Returns:
            pd.DataFrame: contain the latest stock market data on Yahoo Finance.
        """
        return self.latest_data

if __name__ == "__main__":
    direct_market = Direct_market_data(marketName="BTC-USD", timeInterval="1m", period="1d")
    direct_market.start_fetching_data()
    
    # Run for a certain period for demonstration (e.g., 60 seconds) and then stop
    # time.sleep(60)
    # direct_market.stop_fetching_data()
    
    # # Get the latest data after stopping
    # latest_data = direct_market.get_latest_data()
    # print("Latest Data:")
    # print(latest_data)
