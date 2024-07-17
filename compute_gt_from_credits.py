import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from datetime import datetime, timedelta


import sys
def FQ(label):
    print ('------------- FIN QUI TUTTO OK  %s ----------' %(label))
    sys.exit()

color_list = ['blue', 'red', 'black', 'magenta', 'green', 'brown', 'grey', 'cyan', 'blue', 'red',
                  'black', 'magenta', 'green', 'brown', 'grey', 'cyan']


def plot_histogram_line(data_1, data_2, data_3, data_4, bins=30, density=True, line_color='blue', line_width=2):
    """
    Plot a histogram with only the line connecting the bins.

    Parameters:
    - data (array-like): Input data for the histogram.
    - bins (int): Number of bins for the histogram.
    - density (bool): Whether to normalize the histogram.
    - line_color (str): Color of the line.
    - line_width (int): Width of the line.
    """
    # Generate the histogram data
    hist_1, bin_edges_1 = np.histogram(data_1, bins=bins, density=density)
    hist_2, bin_edges_2 = np.histogram(data_2, bins=bins, density=density)
    hist_3, bin_edges_3 = np.histogram(data_3, bins=bins, density=density)
    hist_4, bin_edges_4 = np.histogram(data_4, bins=bins, density=density)

    # Compute the bin centers
    bin_centers_1 = (bin_edges_1[:-1] + bin_edges_1[1:]) / 2
    bin_centers_2 = (bin_edges_2[:-1] + bin_edges_2[1:]) / 2
    bin_centers_3 = (bin_edges_3[:-1] + bin_edges_3[1:]) / 2
    bin_centers_4 = (bin_edges_4[:-1] + bin_edges_4[1:]) / 2

    # Plot the line connecting the histogram bins
    plt.plot(bin_centers_1, hist_1, color=line_color, linewidth=line_width)
    plt.plot(bin_centers_2, hist_2, color=line_color, linewidth=line_width)
    plt.plot(bin_centers_3, hist_3, color=line_color, linewidth=line_width)
    plt.plot(bin_centers_4, hist_4, color=line_color, linewidth=line_width)

    # Add labels and title
    plt.xlabel('Value')
    plt.ylabel('Density' if density else 'Frequency')
    plt.title('Histogram Line Plot')

    # Show the plot
    plt.show()






def extract_prob_g_r(data, perc_n):

    data = data.sort_values(by='TimeToDefault').reset_index(drop=True)
    time_to_default = data['TimeToDefault'].values
    num_credits = data['NumberOfCredits'].values
    N = len(time_to_default)
    N_sample = int(N*perc_n)


    n_bins_ref_ = np.minimum(int(2.0 * np.sqrt(N_sample)), 30)


    N_sample = len(time_to_default)
    print('N_sample: ', N_sample)

    # Step 6: Compute subsequent default times for each default event
    subsequent_default_times = []
    w_data = []

    for i in range(N_sample):

        if (i % 1000 == 0):
            print('i: %s/%s'%(i, N_sample))
        for j in range(i + 1, N_sample):
            subsequent_default_times.append((time_to_default[j] - time_to_default[i])/365.0)
            w_ = (N_sample - (j - i))
            w2_= num_credits[i]
            w_data.append(1.0/w_/w2_)
            #w_data.append(1)


    #time_bins = np.arange(delta_bin/2.0, time_mat + 1.0*delta_bin/2, delta_bin)
    hist_, bin_edges_ = np.histogram(subsequent_default_times, bins=n_bins_ref_, weights = w_data, density=True)

    plot_to_chk = False
    if (plot_to_chk):
        plt.hist(subsequent_default_times, bins=n_bins_ref_, weights=w_data, edgecolor='black', density=True)

        # Adding titles and labels
        plt.title('Weighted Histogram')
        plt.xlabel('Data')
        plt.ylabel('Frequency')
        plt.show()

    #delta_bin = bin_edges_[1:] - bin_edges_[:-1]
    #area_norm = np.dot(np.array(hist_), np.array(delta_bin))
    #hist_ = hist_/area_norm

    bin_centers_w_ = (bin_edges_[:-1] + bin_edges_[1:]) / 2.0

    return  hist_, bin_centers_w_, N_sample





def load_csv_files(folder_path, start_tag):
    # List to hold dataframes
    dataframes_list = []

    # Iterate through all files in the specified folder
    for file_name in os.listdir(folder_path):
        # Check if the file is a .csv and starts with 'X'
        if file_name.endswith('.csv') and file_name.startswith(start_tag):
            # Construct the full file path
            file_path = os.path.join(folder_path, file_name)
            # Load the CSV file into a DataFrame
            df = pd.read_csv(file_path)
            if (len(df)) < 2:
                print('file_path: ', file_path)
                continue
            else:
            # Append the DataFrame to the list
                dataframes_list.append(df)

    return dataframes_list


def average_distributions(distributions):
    # Number of elements in each distribution
    n = len(distributions[0])

    # Initialize the resulting distribution
    averaged_distribution = []

    # Iterate through each index position
    for i in range(n):
        # Collect non-zero values at index i from all distributions
        values = [dist[i] for dist in distributions if dist[i] != 0]

        # Calculate the average if there are non-zero values, else set to 0
        if values:
            avg = sum(values) / len(values)
        else:
            avg = 0

        # Append the average to the resulting distribution
        averaged_distribution.append(avg)

    # Print the resulting distribution
    #print(f"Averaged distribution: {averaged_distribution}")
    return averaged_distribution


if __name__ == "__main__":

    #df_out = extract_1000_events_det(start_date, end_date, num_events)

    perc_n = 1
    n_default_min = 100
    n_default_max = 1000

    n_default_min = 900
    n_default_max = 10000
    start_tag = 'AUT'
    start_tag = 'AUT'
    start_tag = 'RMB'
    start_tag = 'SME'

    gt_save = True
    #gt_save = False


    line_color = 'blue'
    line_width = 1
    density = True
    bins_ref = 30


    folder_path = r'default_events'

    data_list = load_csv_files(folder_path, start_tag)

    hist_list = []
    n_default_list = []
    #
    bin_c_list = []

    n_cr_s = 0
    for data_ in data_list:

        if (data_.shape[0] > n_default_min)  and (data_.shape[0] < n_default_max) and (data_.shape[0] > 20):
            n_cr_s = n_cr_s + 1
            hist_, bin_cent_, n_defaults_  = extract_prob_g_r(data_, perc_n)

            hist_list.append(hist_)
            bin_c_list.append(bin_cent_)
            n_default_list.append(n_defaults_)


            if (gt_save):
                file_out = r'gt_from_credits/gt_%s_ndef_%s.csv' % (start_tag, n_defaults_)

                g_t_df = pd.DataFrame({'Time (Months)': bin_cent_, 'Probability': hist_})
                g_t_df.to_csv(file_out, index=False)


    if (n_cr_s ==0):
        print('NO CREDIT SELECTED!!')
        FQ(77)


    density = False

    for i in range(0, len(hist_list)):

        hist_ = hist_list[i]
        bin_ = bin_c_list[i]
        n_def = n_default_list[i]
        label_ = ('n. def. %s'%(n_def))

        plt.plot(bin_, hist_, '--', color=color_list[i], linewidth = line_width, label=label_)

    plt.legend(loc="upper right")

    # Add labels and title
    plt.xlabel('Time to default [years]')
    plt.ylabel('Pair Probability Density')
    plt.title('%s Pair Default probability analysis'%(start_tag))

    # Show the plot
    plt.show()

