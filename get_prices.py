""" obtain 2-year historic prices of each of the tickers and consolidate as a pandas dataframe """

import pandas as pd
import yfinance as yf
from datetime import date
from dateutil.relativedelta import relativedelta


def yfinance_data (tickers, backtest_months, today):
    """ Get 3-year data from YFinance """

    start = (today + relativedelta(years=-3)).replace(day=1)
    #End date is yesterday. yfinance will automatically adjust to yesterday
    end = today
    actual_end = today + relativedelta(days = -1)
    #Get closing price data of each stock as pd Series
    data = {}
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        df = stock.history(start=start, end=end)
        df_time = df.loc[start:end, "Close"]
        data[ticker] = df_time
    #consolidate the series to a dataframe
    data_df = pd.DataFrame(data)

    #backtest will start from the first day of the xth month before today
    backtest_start = (today + relativedelta(months=-(backtest_months-1))).replace(day=1)

    return data_df, backtest_start, actual_end


