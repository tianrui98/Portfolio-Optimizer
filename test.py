import pytest
from get_prices import *
from get_portfolio import *
from backtest import *
from datetime import datetime

df = pd.read_csv("historic_data.csv",parse_dates=True, index_col= "Date")
today = datetime.strptime('2021-03-31', '%Y-%m-%d')
BT = Backtest(df,10000,today,12,1,"wf","mvo")

def test_get_prices () -> None:
    tickers = ['GOOG','AAPL','FB']
    data_df, start, end = yfinance_data(tickers, 12, today)
    #check  all tickers are included
    for t in tickers:
        assert(t in data_df.columns)
    #check if the end date is before today
    assert(end == today + relativedelta(days=-1) )
    #check if the start date is the start of the month, regardless holidays
    assert(str(start.date()) =="2020-04-01" )

def test_optimizer () -> None:
    """
    Check whether all three stocks have been assigned weights
    Check whether the result of allocation include all 3 stocks
    """
    import warnings
    warnings.filterwarnings("ignore")
    aum =  10000
    optimizers = ["msr","mvo","hrp"]
    for op in optimizers:
        Opt = Optimize(df, op, aum)
        ef, weights, _ = Opt.run_optimizer()
        allocation = Opt.allocate(weights)
        assert(len(weights) == 3)
        assert(len(allocation.keys()) == 3)
        assert(Opt.calculate_performance (ef) != None)

def test_slice_dataset () -> None:
    """
    Slices should be the length of backtest_months
    """
    slices = BT.slice_dataset()
    assert(len(slices) == 12)

def test_walk_forward_get_train_test () -> None:
    """
    Check if walk-forward windows are sliced correctly
    """
    slices = BT.slice_dataset()
    train_month_data, test_month_data = BT.walk_forward_get_train_test (slices)
    assert(str(min(train_month_data[0].index).date()) == "2020-04-01")
    assert(str(max(train_month_data[0].index).date()) == "2020-04-29")
    assert(str(min(train_month_data[1].index).date()) == "2020-04-01")
    assert(str(max(train_month_data[1].index).date()) == "2020-05-29")
    assert(str(min(train_month_data[10].index).date()) == "2020-04-01")
    assert(str(max(train_month_data[10].index).date()) == "2021-02-26")

def test_cross_validation_get_train_test () -> None:
    """
    Check if cross_validation windows are sliced correctly
    """
    BT = Backtest(df,10000,today,12,1,"cv","mvo")
    slices = BT.slice_dataset()

    train_month_data, test_month_data = BT.cross_validation_get_train_test (slices)
    assert(str(min(test_month_data[0][0].index).date()) == "2020-04-01")
    assert(str(max(test_month_data[0][0].index).date()) == "2020-04-29")
    assert(str(min(train_month_data[0].index).date()) == "2020-05-01")
    assert(str(max(train_month_data[0].index).date()) == "2021-03-30")

def test_test_one_set () -> None:
    """
    Check if test_one_set returns 3 performance stats
    """
    slices = BT.slice_dataset()
    train_month_data, test_month_data = BT.walk_forward_get_train_test (slices)
    for i in range(len(train_month_data)):
        train = train_month_data[i]
        test = test_month_data [i]
        performance = BT.test_one_set(train, test)
        assert(len(performance) == 3)


