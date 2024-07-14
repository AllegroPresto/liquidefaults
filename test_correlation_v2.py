import datetime as dt

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import comb

from pandas_datareader import data as pdr
import yfinance as yfin

import sys
def FQ(label):
    print ('------------- FIN QUI TUTTO OK  %s ----------' %(label))
    sys.exit()



def set_norm(tickers):

    gr_norm = {}
    n_tickers = len(tickers)

    for i in range(0, n_tickers):

        tik_ref = tickers[i]

        for j in range(1, 2):

            for k in range(0, n_tickers):

                tik_tmp = tickers[k]

                if (tik_tmp != tik_ref):

                    r_dist = np.abs(k - i)
                    try:
                        gr_norm[r_dist] = gr_norm[r_dist] + 1.0
                    except:
                        gr_norm[r_dist] = 1.0
                else:
                    continue

    return gr_norm




if __name__ == '__main__':

    display = False
    yfin.pdr_override()


    start   = dt.datetime(2017, 1, 1)
    end     = dt.datetime(2022, 1, 1)

    #tickers = ['AAPL', 'MSFT', 'IBM', 'MTA', 'GOOGL', 'TSLA', 'JPM', 'NVDA', 'DIS', 'PG', 'BABA', 'ADBE', 'WMT', 'XOM', 'PG', 'KO', 'PFE', 'DIS', 'NVDA', 'NFLX']
    #tickers = ['AAPL', 'MSFT', 'IBM', 'GOOGL', 'TSLA', 'JPM', 'NVDA', 'DIS', 'PG', 'ADBE', 'WMT', 'BABA', 'XOM', 'PG', 'KO', 'PFE', 'DIS', 'NVDA', 'NFLX']
    #tickers = ['AAPL', 'MSFT', 'IBM', 'GOOGL', 'TSLA', 'JPM', 'NVDA', 'DIS', 'PG', 'ADBE', 'WMT', 'BABA', 'XOM', 'PG', 'KO', 'PFE']
    #tickers = ['AAPL', 'MSFT', 'IBM', 'GOOGL', 'TSLA', 'JPM', 'NVDA', 'DIS', 'PG', 'ADBE', 'WMT', 'BABA', 'XOM']
    #tickers = ['AAPL', 'MSFT', 'IBM', 'GOOGL', 'TSLA', 'JPM', 'NVDA', 'DIS', 'PG', 'ADBE']

    #tickers = ['AAPL', 'MSFT', 'IBM', 'GOOGL', 'TSLA', 'JPM', 'NVDA']
    #tickers = [ 'DIS', 'PG', 'ADBE','AAPL', 'MSFT', 'IBM', 'GOOGL', 'TSLA', 'JPM', 'NVDA']

    #tickers = ['GOOGL', 'TSLA', 'JPM', 'IBM', 'NVDA', 'PG', 'ADBE', 'WMT']

    #tickers = ['AAPL', 'MSFT', 'IBM', 'MTA', 'GOOGL', 'TSLA', 'JPM', 'NVDA', 'DIS', 'PG',
    #           'BABA', 'ADBE', 'WMT', 'XOM', 'PG', 'KO', 'PFE', 'DIS', 'NVDA', 'NFLX',
    #           'AMZN', 'MTA', 'CRM', 'BAC', 'V', 'MA', 'INTC', 'PEP', 'CSCO', 'CMCSA',
    #           'PYPL', 'HD', 'VZ', 'ABT', 'MRK', 'MDT', 'UNH', 'MO', 'ORCL', 'AVGO',
    #           'MMM', 'BA', 'CAT', 'CVX', 'DOW', 'GS', 'HON', 'IBM', 'JNJ', 'MCD']

    tickers = [#'MSFT',
               'AAPL', 'IBM', 'MTA', 'GOOGL',
               'TSLA', 'JPM', 'NVDA', 'DIS', 'PG',
               'BABA', 'ADBE', 'WMT', 'XOM', 'PG', 'KO', 'PFE', 'DIS', 'NVDA', 'NFLX',
               'AMZN', 'MTA', 'CRM', 'BAC', 'V', 'MA', 'INTC', 'PEP', 'CSCO', 'CMCSA',
               'PYPL', 'HD', 'VZ', 'ABT', 'MRK', 'MDT', 'UNH', 'MO', 'ORCL', 'AVGO',
               'MMM', 'BA', 'CAT', 'CVX', 'DOW',
               'GS', 'HON', 'IBM', 'JNJ', 'MCD']


    n_tickers = len(tickers)

    # '^GSPC' indicate the SP500
    start   = dt.datetime(2018, 12, 1)
    end     = dt.datetime(2023, 1, 1)

    data = pdr.get_data_yahoo(tickers, start, end)
    data = data['Adj Close']
    #market_cap_data  = data['Market Cap']

    # compute log-return
    log_returns = np.log(data / data.shift())

    correlation_matrix = log_returns.corr()

    # Order by correlation
    average_correlation = correlation_matrix.mean().sort_values(ascending=False)
    ordered_tickers_by_correlation = average_correlation.index.tolist()



    tickers = ordered_tickers_by_correlation

    avg_corr_dict = average_correlation.to_dict()

    print('Ticker: ', tickers)
    print('average_correlation: ', average_correlation)
    print('type(average_correlation): ', type(average_correlation))
    n_tickers = len(tickers)

    #print('log_returns: ', log_returns.head())
    #print('log_returns: ', log_returns['JPM'].iloc[1])

    norm_dict = set_norm(tickers)

    gr_dict = {}
    for i in range(1, n_tickers):
        gr_dict[i] = 0.0



    n_ts     = len(data)
    dict_dist = {}

    for i in range(0, n_tickers):
    #for i in range(0, 1):
        tik_ref = tickers[i]

        for j in range(1, n_ts):
        #for j in range(1, 2):

            r_ref = log_returns[tik_ref].iloc[j]
            for k in range(0, n_tickers):

                tik_tmp = tickers[k]
                avg_corr_ref = avg_corr_dict[tik_ref]
                avg_corr_tmp = avg_corr_dict[tik_tmp]

                corr_dist =(avg_corr_tmp - avg_corr_ref)*10000.0
                corr_dist = int(corr_dist)

                if (tik_tmp != tik_ref):

                    r_tmp = log_returns[tik_tmp].iloc[j]
                    r_dist = np.abs(k - i)
                    #r_dist = corr_dist
                    dict_dist[r_dist] = corr_dist


                    #print('avg_corr_ref: ', avg_corr_ref)
                    #print('avg_corr_tmp: ', avg_corr_tmp)
                    #FQ(222) print('=====================')
                    #gr_dict[r_dist] = gr_dict[r_dist] + 1.0

                    if (r_tmp*r_ref > 0.0) and (r_ref > 0):
                    #if (r_tmp * r_ref > 0.0):

                        gr_dict[r_dist] = gr_dict[r_dist] + 1.0
                    else:
                        gr_dict[r_dist] = gr_dict[r_dist] + 0.0

                else:
                    continue


                if (display):
                    print('=======================')
                    print('tik_ref: ', tik_ref)
                    print('tik_tmp: ', tik_tmp)
                    print('r_dist: ', r_dist)
                    print('r_tmp: ', r_tmp)
                    print('r_ref: ', r_ref)
                    print('gr_dict: ', gr_dict)

    # Computo le correlazioni normali

    gr_corr_dict = {}
    for i in range(0, n_tickers):
        tik_ref = tickers[i]
        r_ref = log_returns[tik_ref]

        for k in range(0, n_tickers):
            tik_tmp = tickers[k]

            if (tik_tmp != tik_ref):
                r_tmp = log_returns[tik_tmp]
                r_dist = np.abs(k - i)

                corr_tmp = r_tmp.corr(r_ref)
                gr_corr_dict[r_dist] = corr_tmp

            else:
                continue


    dist_list =[]
    prob_list = []
    corr_list = []
    norm_list = []

    # Prima normalizzazione
    norm = 0.0
    for k in range(1, n_tickers):

        dist = n_tickers - k
        norm = norm_dict[dist]
        gr_dict[dist] =  gr_dict[dist]/norm
        gr_corr_dict[dist] =  gr_corr_dict[dist]/norm

        norm_list.append(norm)
    #


    #print('===========================')



    # Seconda normalizzazione
    sum = 0.0
    for k in  gr_dict.keys():
        sum = sum  + gr_dict[k]

    p_sum = 0.0
    for k in  gr_dict.keys():
        p_tmp =   gr_dict[k] / sum
        c_tmp =   gr_corr_dict[k]

        gr_dict[k] =  p_tmp

        corr_dist = dict_dist[k]

        #dist_list.append(k)
        dist_list.append(corr_dist)
        prob_list.append(p_tmp)
        corr_list.append(c_tmp)

        p_sum = p_sum  + p_tmp

    print('p_sum: ', p_sum)
    #print('gr_dict: ', gr_dict)

    dist_list_n = [x / 10000.0 for x in dist_list]

    plt.plot(dist_list_n, prob_list, 'o')
    #plt.plot(dist_list, corr_list, 'r--o')

    #plt.plot(dist_list, norm_list, '--k')

    plt.title('Probability of simultaneous return increase of a %s Stock ptf.'%(n_tickers))
    plt.ylabel('Event Probability')
    plt.xlabel('Relative distance (correlation)')
    plt.show()



    FQ(99)

    plt.plot(log_returns)
    plt.ylabel('Log-return')
    plt.xlabel('Time [years]')
    plt.legend(tickers)

    plt.show()





    # compute covariance, variance
    #cov = log_returns.cov()
    #var = log_returns['^GSPC'].var()

    # compute beta
    #beta = cov.loc['AAPL', '^GSPC'] / var

    # compute log-return
    #log_returns = log_returns.fillna(0)
    #mkt     = log_returns['^GSPC']
    #apple   = log_returns['AAPL']


    # fit APPLE vs SP500
    #b, a = np.polyfit(mkt.values, apple.values, 1)

    #print('b: ', b)
    #print('beta: ', beta)

    #beta_model =  b * log_returns['^GSPC'] + a
    #plt.plot(log_returns['^GSPC'], beta_model, '-', color='r')
    #plt.scatter(log_returns['^GSPC'], log_returns['AAPL'])

    #plt.xlabel('Return (SP500)')
    #plt.ylabel('Return (APPLE)')
    #plt.legend(['Beta model', 'MKT data'])

    #plt.show()
    #sys.exit()

    #risk_free_return = 0.0138
    #market_return = .105
    #expected_return = risk_free_return + beta * (market_return - risk_free_return)
    #print('expected_return: ', expected_return)