
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

def FQ(label):
    print ('------------- FIN QUI TUTTO OK  %s ----------' %(label))
    sys.exit()



if __name__ == "__main__":


    # Parameters
    directory = 'gt_via_copulas'  # Specify your folder path
    #copula = 'gaussian_exp'
    copula = 'gaussian'
    #copula = 'clayton_exp'
    #copula = 'frank_exp'
    #copula = 'tstudent_exp'

    marginal = 'exponential'

    prm_name = 'rho'
    #prm_name = 'theta'

    #flag_save = True
    flag_save = True

    start_letters = 'gt_' + copula  + '_' +  marginal# Specify the starting letters to filter files
    output_file = start_letters + '_vs_%s_.png'%(prm_name)
    # List all files in the directory
    all_files = os.listdir(directory)

    # Filter files that start with the specified letters and end with .csv
    csv_files = [f for f in all_files if f.startswith(start_letters) and f.endswith('.csv')]

    # Initialize a figure for plotting
    plt.figure(figsize=(12, 6))

    # Read and plot each CSV file
    for csv_file in csv_files:
        file_path = os.path.join(directory, csv_file)
        data = pd.read_csv(file_path)
        param_ = csv_file.split('par1')[1][1:-4]

        # Calculate the integral using the trapezoidal rule
        integral = np.trapz(data['Probability'], data['Time (Months)'])

        # Normalize the probability values so that the integral is equal to one
        data['Probability'] = data['Probability'] / integral

        #print('param_: ', param_)
        #print('param__: ', param__)

        label_ = '%s: %s'%(prm_name, param_)

        if (float(param_) not in [0.0005, 0.3]):

            # Assuming the CSV files have columns 'Time (Months)' and 'Probability'
            plt.plot(data['Time (Months)'], data['Probability'], label=label_)

    # Customize the plot
    plt.xlabel('Time (Years)')
    plt.ylabel('g(t) Probability density')
    plt.title('Probability density of defaults subsequent a default event at t=0 (g(t))')
    plt.title('g(t) for default distribution based on Gaussian copula and Exponential marginal ')
    plt.legend()
    plt.grid(True)

    # Save the plot to a file
    if (flag_save):
        plt.savefig('graph\%s'%(output_file))

    # Show the plot
    plt.show()
