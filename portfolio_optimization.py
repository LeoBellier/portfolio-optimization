#import Libraries
from pandas_datareader import data as web 
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import matplotlib.pyplot as plt

class PortfolioReporter:
    
    def __init__(self):
        plt.style.use('fivethirtyeight')
        assets = ['FB','AMZN', 'AAPL','NFLX','GOOG']
        self.weights = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        self.df = self.get_prices(assets)
    
    def create_graphic(self, stocks):
        title = 'PortFolio Adj. Close Price History'
        for c in stocks:
            plt.plot(stocks[c], label = c)
        plt.title(title)
        plt.xlabel('Date', fontsize=18)
        plt.ylabel('Adj. Price USD($)')
        plt.legend(stocks.columns.values, loc = 'upper left')
        plt.show()
    
    def get_prices(self, stocks):
        stockdate = '2017-01-01'
        today = datetime.today().strftime('%Y-%m-%d')
        df = pd.DataFrame()
        for stock in stocks:
            df[stock] = web.DataReader(stock, data_source='yahoo', start = stockdate, end=today)['Adj Close']
        return df
    
    def get_daily_return(self):
        return self.df.pct_change()
    
    def get_cov_matrix_annual(self):
        return self.get_daily_return().cov() * 252
    
    def get_portfolio_variance(self):
        return np.dot(self.weights.T, np.dot(self.get_cov_matrix_annual(), self.weights))
    
    def get_portfolio_volatility(self):
        return np.sqrt(self.get_portfolio_variance())
    
    def get_portfolio_annual_return(self):
        return np.sum(self.get_daily_return().mean() * self.weights) * 252

class PortfolioOprtimizer:
    def __init__(self, df):
        self.df = df
    
    def get_expected_returns(self):
        mu = expected_returns.mean_historical_return(self.df)
        S = risk_models.sample_cov(self.df)
        
        ef = EfficientFrontier(mu, S)
        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        port_performance_percentage = {
            'cleanedWeights' : cleaned_weights,
            'portfolioPerformance': ef.portfolio_performance(verbose=True)
            }
        return port_performance_percentage
        
    def get_discrete_allocation(self):
        latest_prices = get_latest_prices(self.df)
        weights = self.get_expected_returns['cleanedWeights']
        da = DiscreteAllocation(weights,latest_prices, total_portfolio_value = 10000)
        discrete_allocarion_and_remaining = {
            'discreteAllocation': 'Discrete allocation: '+str(allocation),
            'fundsRemaining':  'Funds remaining: ${:.2f}'.format(leftover)
            }
        return discrete_allocarion_and_remaining

def as_percent(value):
    return str(round(value, 2) * 100) + '%'
    
po = PortfolioReporter()
print(as_percent(po.get_portfolio_variance()))
print(as_percent(po.get_portfolio_volatility()))
print(as_percent(po.get_portfolio_annual_return()))