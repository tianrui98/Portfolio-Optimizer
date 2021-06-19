# Portfolio-Optimizer

## Data Science for Quantitative Finance - Final Project

### Project description:
This python program  uses the PyPortfolioOpt library to find out the optimal asset allocation
for today and for a given basket of stocks.

User can call the program following this example:

```optimize_portfolio.py --tickers GOOG,AAPL,FB --optimizer mvo --aum 10000 --backtest_months 12 --test_months 1 --backtest_type wf --plot_weights```

The program performs a backtest for each of the ‘backtest_months’ previous months
(by using the exact same portfolio optimization parameters). It returns the realized and expected portfolio
stats and the quantities of shares that should be bought today for the given AUM.

### Project Structure:
optimize_portfolio:\n
    process arguments and run program
 
get_prices:\n
    obtain historic prices of each of the tickers with yfinance, and consolidate as a pandas dataframe
    
get_portfolio:\n
    performs portfolio optimization using PyPortfolioOpt.
    
backtest:\n
    Backtest has functions for performing backtesting with the optimized portfolio.
    
bt_methods:\n
    BacktestBasics stores basic information for backtesting
    CrossValidation and WalkForward contains methods for different backtesting types

### Package requirement:
pip install yfinance  

pip install fix-yahoo-finance==0.1.30  

pip install PyPortfolioOpt  

pip install cvxpy  

pip install cvxopt
