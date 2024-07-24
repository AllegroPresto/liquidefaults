import datetime as dt

import numpy as np
import matplotlib.pyplot as plt
import sys

from pandas_datareader import data as pdr
import yfinance as yfin

import sys
def FQ(label):
    print ('------------- FIN QUI TUTTO OK  %s ----------' %(label))
    sys.exit()


if __name__ == '__main__':

    time_t = 0
    # PRIMO METODO PER G(T)
    time_to_default = []
    for i in range(0,6):
        delta_t = float(i)
        time_t = time_t + delta_t
        time_to_default.append(time_t)

    print('time_to_default: ', time_to_default)
    N_sample = len(time_to_default)

    # Step 6: Compute subsequent default times for each default event
    subsequent_default_times = []
    w_data = []

    print(time_to_default[N_sample-1])
    for i in range(N_sample):
        for j in range(i + 1, N_sample):

            subsequent_default_times.append(time_to_default[j] - time_to_default[i])


            w_ = 1.0
            w_ = (N_sample - (j - i))

            w_data.append(1.0/w_)

            print(time_to_default[j] - time_to_default[i])
            print(w_)
            print('==========================================')
    print('subsequent_default_times: ', subsequent_default_times)
    #FQ(8888)
    # Step 7: Estimate g(t) - probability distribution of all subsequent default events
    #time_bins = np.arange(0.5, 5 + 1.0, 1.0)
    #print(time_bins)
    #g_t, _ = np.histogram(subsequent_default_times, bins=10 , weights = w_data, density=True)
    #time_bins_yy = time_bins/12.0
    plt.hist(subsequent_default_times, bins= 15, weights=w_data, edgecolor='black')

    # Adding titles and labels
    plt.title('Weighted Histogram')
    plt.xlabel('Data')
    plt.ylabel('Frequency')
    plt.show()

    FQ(77)

