"""
Test investment strategy (No. of shares) on previous months
Generate test statistics based on aggregated returns
"""
from get_portfolio import *
from bt_methods import *
import pandas as pd
from pypfopt.objective_functions import portfolio_variance
from pypfopt.risk_models import CovarianceShrinkage
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
import numpy as np
import warnings
from sklearn.model_selection import LeavePOut
warnings.filterwarnings("ignore")


class Backtest(WalkForward, CrossValidation):

    def weights_to_np (self, weights):
        """
        Helper function to convert weights to np.array format
        """
        #make sure the weights array is in order of stocks
        weights_dict = dict(weights)
        weights_list = []
        for stock in weights_dict.keys():
            w = weights_dict[stock]
            weights_list.append(w)

        weights_np = np.array(weights_list)

        return weights_np

    def compute_daily_returns (self, weights, test):
        """
        Apply the portfolio weights to test data
        Return: weighted daily returns, summed weighted daily returns
        """
        weights_np = self.weights_to_np (weights)
        daily_returns = test.pct_change().dropna()
        daily_returns_weighted = np.multiply(daily_returns, weights_np)

        #sum the stock returns to get daily portfolio returns
        total_daily_returns = daily_returns_weighted.sum (axis = 1)
        return daily_returns, total_daily_returns

    def test_one_set (self,train,tests):
        """
        Perform backtesting on one training-tests set.
        Return: average performance stats of the tests.
        """
        #get weights from training data
        OP = Optimize(train,self.optimizer,self.aum)
        _, weights, _ = OP.run_optimizer()
        all_test_daily_returns = []
        all_test_total_daily_returns =[]

        #calculate daily returns on all test sets
        for test in tests:
            daily_returns, total_daily_returns= self.compute_daily_returns(weights, test)
            all_test_daily_returns.append(daily_returns)
            all_test_total_daily_returns.append(total_daily_returns)

        #calcualte the average daily returns
        average_test_daily = np.mean(np.array(all_test_daily_returns), axis=0 )
        average_test_total_daily_returns = np.mean( np.array(all_test_total_daily_returns), axis = 0)

        return average_test_daily, average_test_total_daily_returns, weights


    def slice_dataset (self):
        """
        slice dataset into x months where x = backtest_months
        Return: A list of stock price data of each month
        """
        slices = []
        period_start = self.start_date
        for i in range(self.backtest_months):
            #remove the last 1 day for purging purpose
            period_end = (period_start + relativedelta(months = 1)).replace(day=1) + relativedelta(days = -2)
            period_data = self.df[period_start:period_end]
            slices.append(period_data)
            period_start = (period_start + relativedelta(months = 1)).replace(day=1)

        return slices

    def evaluate_performance (self, all_daily_returns,all_total_daily_returns, all_weights):
        """
        Input: all_daily_returns is a list of daily returns computed from test set
        Return: Annualized return, volatility, sharpe ratio
        """
        #compute portfolio_volatility for each test
        all_portfolio_volatilities = []
        for i in range(len(all_daily_returns)):
            weights = all_weights[i]
            returns = pd.DataFrame(all_daily_returns[i])
            start = min(returns.index)
            end = max(returns.index)
            prices = self.df[start:end]
            cov_matrix = CovarianceShrinkage(prices).ledoit_wolf()
            weights_np = self.weights_to_np (weights)
            var = portfolio_variance(weights_np,cov_matrix)
            portfolio_volatility = np.sqrt(var)
            all_portfolio_volatilities.append(portfolio_volatility)

        #compute mean daily returns of every test
        all_mean_daily_returns = [np.mean(x) for x in all_total_daily_returns]

        #compute average of the stats
        average_daily_returns = np.mean(all_mean_daily_returns)
        average_portfolio_volatilities = np.mean(all_portfolio_volatilities)

        #compute daily stats from the average
        daily_portfolio_return = average_daily_returns
        daily_sharpe_ratio = (daily_portfolio_return - 0.02) / average_portfolio_volatilities
        daily_volatility = average_portfolio_volatilities

        #compute annualized stats
        annual_portfolio_return = ((daily_portfolio_return + 1)**250)-1
        annual_sharpe_ratio = daily_sharpe_ratio * (np.sqrt(250))
        annual_volatility = daily_volatility * (np.sqrt(250))

        return (annual_portfolio_return, annual_volatility, annual_sharpe_ratio)

    def run_test(self):
        if self.backtest_type == "wf":
            all_daily_returns,all_total_daily_returns,all_weights = self.walk_forward_test()
            stats = self.evaluate_performance(all_daily_returns,all_total_daily_returns,all_weights )
        elif self.backtest_type == "cv":
            all_daily_returns,all_total_daily_returns,all_weights = self.cross_validation_test()
            stats = self.evaluate_performance(all_daily_returns,all_total_daily_returns,all_weights )
        else:
            raise Exception("Test type not implemented!")
        return stats


