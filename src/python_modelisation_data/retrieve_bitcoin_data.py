import yfinance as yf
import pandas as pd
import os
import datetime as datetime


def getBitCoinData():
    intervalRequired = ["15m","1d"]
    folderNames = [f"BTC_USD_{interval.upper()}" for interval in intervalRequired]

    for folderName in folderNames:
        if not os.path.exists(folderName):
            os.mkdir(folderName)

    # Get the data for all the interval required.
    for interval in intervalRequired:
        if interval == '1d':
            for yearNum in range(2015,2024):
                if interval == '1d':
                    start = f"{yearNum}-01-01"
                    end = f"{yearNum}-12-31"
                    btc_usd = yf.download('BTC-USD', start=start, end=end, interval=interval)
                    btc_usd.to_csv(f"BTC_USD_{interval.upper()}/BTCUSD_HISTORICAL_{interval.upper()}_{start}_{end}.csv")
        elif interval == "15m":
            today = datetime.datetime.now()
            daysDelta = datetime.timedelta(days=60)
            dayStart = today-daysDelta
            start = f"{dayStart.year}-{dayStart.month}-{dayStart.day}"
            end = f"{today.year}-{today.month}-{today.day}"
            btc_usd = yf.download('BTC-USD', start=start, end=end, interval=interval)
            btc_usd.to_csv(f"BTC_USD_{interval.upper()}/BTCUSD_HISTORICAL_{interval.upper()}_{start}_{end}.csv")
        else:
            raise NotImplemented
    return None
