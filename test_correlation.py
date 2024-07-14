import datetime as dt

import numpy as np
import matplotlib.pyplot as plt
import sys

from pandas_datareader import data as pdr
import yfinance as yfin

if __name__ == '__main__':

    yfin.pdr_override()


    start   = dt.datetime(2017, 1, 1)
    end     = dt.datetime(2022, 1, 1)

    tickers = ['AAPL', 'MSFT', 'IBM', '^GSPC']

    # '^GSPC' indicate the SP500
    start   = dt.datetime(2015, 12, 1)
    end     = dt.datetime(2021, 1, 1)

    data = pdr.get_data_yahoo(tickers, start, end)

    data = data['Adj Close']

    # compute log-return
    log_returns = np.log(data / data.shift())

    plt.plot(log_returns)
    plt.ylabel('Log-return')
    plt.xlabel('Time [years]')
    plt.legend(tickers)

    plt.show()


    # compute covariance, variance
    cov = log_returns.cov()
    var = log_returns['^GSPC'].var()

    # compute beta
    beta = cov.loc['AAPL', '^GSPC'] / var

    # compute log-return
    log_returns = log_returns.fillna(0)
    mkt     = log_returns['^GSPC']
    apple   = log_returns['AAPL']


    # fit APPLE vs SP500
    b, a = np.polyfit(mkt.values, apple.values, 1)

    print('b: ', b)
    print('beta: ', beta)

    beta_model =  b * log_returns['^GSPC'] + a
    plt.plot(log_returns['^GSPC'], beta_model, '-', color='r')
    plt.scatter(log_returns['^GSPC'], log_returns['AAPL'])

    plt.xlabel('Return (SP500)')
    plt.ylabel('Return (APPLE)')
    plt.legend(['Beta model', 'MKT data'])

    plt.show()
    sys.exit()

    #risk_free_return = 0.0138
    #market_return = .105
    #expected_return = risk_free_return + beta * (market_return - risk_free_return)
    #print('expected_return: ', expected_return)