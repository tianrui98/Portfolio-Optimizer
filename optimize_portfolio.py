"""
Data Science for Quantitative Finance
Group Assignment 3

Group members
Zhu Tianrui, Julian Chong, Krys Lis

Project description:
This python program  uses the PyPortfolioOpt library to find out the optimal asset allocation
for today and for a given basket of stocks.

User can run the program following this format:
optimize_portfolio.py --tickers GOOG,AAPL,FB --optimizer mvo --aum 10000 --backtest_months 12 --test_months 1 --backtest_type wf --plot_weights

The program performs a backtest for each of the ‘backtest_months’ previous months
(by using the exact same portfolio optimization parameters). It returns the realized and expected portfolio
stats and the quantities of shares that should be bought today for the given AUM.

Module description:
optimize_portfolio:
    process arguments and run program
get_prices:
    obtain historic prices of each of the tickers with yfinance, and consolidate as a pandas dataframe
get_portfolio:
    performs portfolio optimization using PyPortfolioOpt.
backtest:
    Backtest has functions for performing backtesting with the optimized portfolio.
bt_methods:
    BacktestBasics stores basic information for backtesting
    CrossValidation and WalkForward contains methods for different backtesting types

Package requirement:
pip install yfinance
pip install fix-yahoo-finance==0.1.30
pip install PyPortfolioOpt
pip install cvxpy
pip install cvxopt

"""
from pypfopt.plotting import plot_weights as pw
import argparse
from datetime import date, datetime
import pandas as pd
from get_prices import *
from get_portfolio import *
from backtest import *
import matplotlib.pyplot as plt


def set_arguments ():
    """
    Allow user to set arguments with argparse

    Return: arguments
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--tickers",
                        type = str,
                        help="The stocks' tickers, separated by comma,eg.MSFT",
                        required=True)

    parser.add_argument("--optimizer",
                        type = str,
                        help = str("Which optimizer to use? Enter msr, mvo, or hrp\n"+
                        "msr: Maximum Sharpe Ratio\n"+
                        "mvo: Mean-Variance Optimization\n"+
                        "hrp: Hierarchical Risk Parity\n"),
                        required = True)

    parser.add_argument("--aum", type = int,
                        help="The asset under management. Follow the stock's currency",
                        required=True)

    parser.add_argument("--backtest_months", type = int,
                        help="The number of months to perform backtest on.Should be multiple of test_months",
                        required=True)

    parser.add_argument("--test_months", type = int,
                        help="The number of months out of backtest_months to be used as test.",
                        required=True)

    parser.add_argument("--backtest_type",
                        type = str,
                        help = str("Which type of backtest? Enter wf, cv or cvpcv\n"+
                        "wf: Walk-Forward.\n"+
                        "cv: Cross-Validation\n"+
                        "cvpcv: Combinatorial Purged Cross-Validation\n"),
                        required = True)

    parser.add_argument("--plot_weights", action='store_true',
                        help="If called, plot portfolio weights as a horizontal bar chart.",
                        required=False)

    args = parser.parse_args()

    return args

class Args:
    """ Retrieve arguments from argparse """

    def __init__ (self, args):
        #tickers is a list of tickers
        self.tickers = [ t.upper() for t in args.tickers.split(',')]
        self.optimizer = args.optimizer
        self.aum = args.aum
        self.backtest_months = args.backtest_months
        self.test_months = args.test_months
        if self.backtest_months % self.test_months != 0:
            raise Exception("Backtest_month must be a multiple of test_months!")
        self.backtest_type = args.backtest_type
        self.plot_weights = args.plot_weights

def main (tickers,optimizer, aum, backtest_months, test_months, backtest_type, plot_weights,today = date.today()):
    """ Run the program """

    #retrieve data
    df, start, end =yfinance_data (tickers, backtest_months, today)

    #expected performance
    OP = Optimize(df, optimizer, aum)
    ef, weights ,_ = OP.run_optimizer()
    performance = OP.calculate_performance (ef)
    allocations = OP.allocate(weights)

    #backtest stats
    BT=Backtest(df,aum,today,backtest_months,test_months,backtest_type,optimizer)
    stats = BT.run_test()

    print("\n\n")
    print("Backtest Stats :")
    print("Start date: ",str(start))
    print("End date: ", str(end))
    print("\n\n")
    print("Realised Annual Return: %", str(np.round(stats[0] * 100,1)))
    print("Realised Annual Volatility : %", str(np.round(stats[1]* 100 ,1)))
    print("Realised Annual Sharpe Ratio: ", str(np.round(stats[2],2)))
    print("\n\n")
    print("Expected Annual Return: %", str(np.round(performance[0] * 100,1)))
    print("Expected Annual Volatility: %", str(np.round(performance[1]* 100,1)))
    print("Expected Annual Sharpe Ratio: ", str(np.round(performance[2],2)))
    print("\n\n")
    print("Shares Needed:")
    for stock in allocations.keys():
        print(stock, ":", str(allocations[stock]))

    if plot_weights:
        pw(weights)
        plt.show()

    return None

if __name__ == "__main__":
    args = set_arguments ()
    A = Args(args)
    main(tickers = A.tickers,
        optimizer = A.optimizer,
        aum = A.aum,
        backtest_months = A.backtest_months,
        test_months = A.test_months,
        backtest_type = A.backtest_type,
        plot_weights = A.plot_weights)

