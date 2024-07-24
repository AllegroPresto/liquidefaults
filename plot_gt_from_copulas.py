
import os
import pandas as pd
import matplotlib.pyplot as plt
import sys

def FQ(label):
    print ('------------- FIN QUI TUTTO OK  %s ----------' %(label))
    sys.exit()



if __name__ == "__main__":


    # Parameters
    directory = 'gt_via_copulas'  # Specify your folder path
    copula = 'gaussian_exp'
    #copula = 'clayton'
    #copula = 'frank'

    prm_name = 'rho'
    #prm_name = 'theta'

    #flag_save = True
    flag_save = True

    start_letters = 'gt_' + copula  # Specify the starting letters to filter files
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
        label_ = '%s: %s'%(prm_name, param_)

        if (float(param_) < 0.3):

            # Assuming the CSV files have columns 'Time (Months)' and 'Probability'
            plt.plot(data['Time (Months)'], data['Probability'], label=label_)

    # Customize the plot
    plt.xlabel('Time (Months)')
    plt.ylabel('g(t) Probability density')
    plt.title('Probability Distribution of defaults subsequent a default event (g(t))')
    plt.legend()
    plt.grid(True)

    # Save the plot to a file
    if (flag_save):
        plt.savefig('graph\%s'%(output_file))

    # Show the plot
    plt.show()
