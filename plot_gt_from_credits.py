
import os
import pandas as pd
import matplotlib.pyplot as plt
import sys

def FQ(label):
    print ('------------- FIN QUI TUTTO OK  %s ----------' %(label))
    sys.exit()



if __name__ == "__main__":


    # Parameters
    directory = 'gt_from_credits'  # Specify your folder path
    creditype = 'AUT'
    creditype = 'RMB'
    #creditype = 'SME'
    #creditype = 'CMR'

    min_n_def = 4
    #min_n_def = 1500

    #min_n_def = 2000
    max_n_def =10000000

    #min_n_def = 4000
    #max_n_def = 7000

    #min_n_def = 900
    #max_n_def = 10000

    #flag_save = True
    flag_save = False
    to_exclude =[4249, 328, 1128]
    to_exclude =[]

    start_letters = 'gt_' + creditype  # Specify the starting letters to filter files
    output_file = start_letters + '_' + str(min_n_def) + '_' + str(max_n_def) +'.png'
    # List all files in the directory
    all_files = os.listdir(directory)

    print('directory: ', directory)
    FQ(99)
    # Filter files that start with the specified letters and end with .csv
    csv_files = [f for f in all_files if f.startswith(start_letters) and f.endswith('.csv')]

    # Initialize a figure for plotting
    plt.figure(figsize=(12, 6))

    # Read and plot each CSV file
    for csv_file in csv_files:
        file_ = csv_file.split('ndef_')[1]
        ndef_ = file_.split('.csv')[0]

        if (int(ndef_) >= min_n_def) and (int(ndef_) <= max_n_def) and int(ndef_)  not in to_exclude:

            file_path = os.path.join(directory, csv_file)
            data = pd.read_csv(file_path)
            label_ = creditype + ': n. def. ' + ndef_
            # Assuming the CSV files have columns 'Time (Months)' and 'Probability'
            plt.plot(data['Time (Years)'], data['Probability'], label=label_)

    # Customize the plot
    plt.xlabel('Time (Years)')
    plt.ylabel('g(t) Probability density')
    plt.title('Probability Distribution of defaults subsequent a default event (g(t))')
    plt.legend()
    plt.grid(True)

    # Save the plot to a file
    if (flag_save):
        plt.savefig('graph\%s'%(output_file))

    # Show the plot
    plt.show()
