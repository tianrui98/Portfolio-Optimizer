from backtest import *
import pandas as pd
from dateutil.relativedelta import relativedelta
import numpy as np
import warnings
from sklearn.model_selection import LeavePOut
warnings.filterwarnings("ignore")


class BacktestBasics ():
    def __init__ (self, df, aum, today, backtest_months, test_months, backtest_type, optimizer):
        self.df = df
        self.aum = aum
        self.backtest_months = backtest_months
        self.test_months = test_months
        self.backtest_type = backtest_type
        self.today = today
        self.train_months = self.backtest_months - self.test_months
        self.optimizer = optimizer
        self.start_date = (self.today + relativedelta(months=-(self.backtest_months-1))).replace(day=1)
        self.end_date = self.today + relativedelta(days = -1)


class WalkForward (BacktestBasics):

    def walk_forward_get_train_test (self,slices):
        """
        return training and test set for walk-forward testing
        train_month_data: a list of training periods
        test_month_data: a list of lists of testing periods
        """
        train_month_data = []
        test_month_data = []
        for i in range(self.train_months):
            if i == 0:
                trains = slices[0]
            else:
                trains = slices[0:(i+1)]
                trains = pd.concat(trains, axis = 0)
            train_month_data.append(trains)
            if (self.test_months == 1):
                tests = [slices[i+1]]
            else:
                tests = slices[(i+1): (i+self.test_months+1)]
            test_month_data.append(tests)

        return train_month_data , test_month_data


    def walk_forward_test(self):
        """
        Test strategy with walk forward method
        Calculate the performance stats for all sets
        Consolidate performance stats by taking the average

        Return:
        daily returns of each test
        summed daily returns of each test
        weights used in each test
        """
        slices = self.slice_dataset()
        train_month_data, test_month_data = self.walk_forward_get_train_test(slices)

        #calculate performance stats for all train-test sets
        all_daily_returns = []
        all_total_daily_returns = []
        all_weights = []

        for i in range(len(train_month_data)):
            train = train_month_data[i]
            tests = test_month_data[i]
            daily_returns,total_daily_returns, weights = self.test_one_set(train,tests)
            all_daily_returns.append(daily_returns)
            all_total_daily_returns.append(total_daily_returns)
            all_weights.append(weights)

        return all_daily_returns,all_total_daily_returns,all_weights



class CrossValidation (BacktestBasics):

    def cross_validation_get_train_test (self,slices):
        """
        return training and test set for cross validation testing
        train_month_data: a list of training periods
        test_month_data: a list of lists of testing periods
        """
        train_month_data = []
        test_month_data = []

        lpo = LeavePOut(p=self.test_months)
        for train_index, test_index in lpo.split(slices):
            trains = [slices[x] for x in train_index]
            tests = [slices[x] for x in test_index]
            train_month_data.append(pd.concat(trains, axis = 0))
            test_month_data.append(tests)

        return train_month_data , test_month_data

    def cross_validation_test(self):
        """
        Test strategy with cross validation method
        Calculate the performance stats for all sets
        Consolidate performance stats by taking the average

        Return:
        daily returns of each test
        summed daily returns of each test
        weights used in each test
        """
        slices = self.slice_dataset()
        train_month_data, test_month_data = self.cross_validation_get_train_test(slices)

        #calculate performance stats for all train-test sets
        all_daily_returns = []
        all_total_daily_returns = []
        all_weights = []

        for i in range(len(train_month_data)):
            train = train_month_data[i]
            tests = test_month_data[i]
            daily_returns,total_daily_returns, weights = self.test_one_set(train,tests)
            all_daily_returns.append(daily_returns)
            all_total_daily_returns.append(total_daily_returns)
            all_weights.append(weights)

        return all_daily_returns,all_total_daily_returns,all_weights
