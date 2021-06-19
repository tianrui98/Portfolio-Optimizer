""" Feed pypfopt with historic data and generate optimized portfolio """

import pandas as pd
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.hierarchical_portfolio import HRPOpt
from pypfopt import risk_models
from pypfopt import expected_returns
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt import objective_functions
import sys, os

class Optimize:
    def __init__ (self, df, optimizer,aum):
        self.df = df
        self.optimizer = optimizer
        # Calculate expected returns and sample covariance
        self.mu = expected_returns.mean_historical_return(df)
        self.S = risk_models.sample_cov(df)
        self.aum = aum

    def maximum_sharpe_ratio (self):
        # Optimize for maximal Sharpe ratio
        ef = EfficientFrontier(self.mu, self.S)
        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()

        return ef, weights, cleaned_weights

    def minimum_volatility (self):
        # Optimize for minimum volatility
        ef = EfficientFrontier(self.mu, self.S)
        # ef.add_objective(objective_functions.L2_reg, gamma=0.1)
        weights = ef.min_volatility()
        cleaned_weights = ef.clean_weights()

        return ef, weights, cleaned_weights

    def hierarchical_risk_parity (self):
        # Optimize for Hierarchical Risk Parity
        #this optimizer uses historic returns rather than mean historic return
        #this optimizer does not support objective
        mu = expected_returns.returns_from_prices(self.df)
        ef = HRPOpt(mu,self.S)
        weights = ef.optimize()
        cleaned_weights = ef.clean_weights()

        return ef, weights, cleaned_weights

    def run_optimizer (self):
        #select optimizer functions
        if self.optimizer == "msr":
            ef, weights, cleaned_weights = self.maximum_sharpe_ratio ()
        if self.optimizer == "mvo":
            ef, weights, cleaned_weights = self.minimum_volatility ()
        if self.optimizer == "hrp":
            ef, weights, cleaned_weights = self.hierarchical_risk_parity ()

        return ef, weights, cleaned_weights

    def calculate_performance (self,ef):
        #Print the following stats:
        #Expected annual return
        #Annual volatility
        #Sharpe Ratio
        pf = ef.portfolio_performance(verbose=False)
        return pf

    def allocate(self,weights):

        #convert weights to No.of shares
        latest_prices = get_latest_prices(self.df)

        da = DiscreteAllocation(weights,
                                latest_prices,
                                total_portfolio_value=self.aum,
                                short_ratio = 0.0)
        allocation, leftover = da.lp_portfolio(solver = "GLPK_MI")

        #because DiscreteAllocation removes stocks with 0 shares from allocation
        #we need to add them back
        tickers = (self.df).columns
        for t in tickers:
            if t not in allocation.keys():
                allocation[t] = 0

        return allocation

