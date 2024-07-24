
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import sys
def FQ(label):
    print ('------------- FIN QUI TUTTO OK  %s ----------' %(label))
    sys.exit()

color_list = ['blue', 'red', 'black', 'magenta', 'green', 'brown', 'grey', 'cyan', 'blue', 'red',
                  'black', 'magenta', 'green', 'brown', 'grey', 'cyan']








if __name__ == "__main__":

    # Step 1: Read Data from CSV
    file_path = 'default_events/RMB_UE_549300HQGL7PSLFK4G36N201401.csv'  # Update this with your actual file path
    df = pd.read_csv(file_path)

    # Step 2: Data Cleaning
    # Assuming 'default_date' and 'default_time' columns exist



    #df['datetime'] = pd.to_datetime(df['DefaultDate'] + ' ' + df['TimeToDefault'])
    df['datetime'] = pd.to_datetime(df['DefaultDate'])

    #DefaultDate, TimeToDefault, NumberOfCredits

    # Drop the original date and time columns if they are not needed
    df.drop(['DefaultDate', 'TimeToDefault', 'NumberOfCredits'], axis=1, inplace=True)

    # Step 3: Feature Engineering
    # Sort by datetime to ensure proper calculation of time differences
    #df.sort_values('datetime', inplace=True)

    # Step 4: Correlation Analysis
    # Compute the correlation matrix
    correlation_matrix = df.corr()

    # Print the correlation matrix
    #print(correlation_matrix)
    #FQ(99)
    # Visualize the correlation matrix
    #plt.figure(figsize=(10, 8))
    #sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
    #plt.title('Correlation Matrix')
    #plt.show()


    #print('df[datetime]: ', df['datetime'])
    #FQ(99)

    # Calculate the time difference in seconds between consecutive default events
    df['time_diff_seconds'] = df['datetime'].diff().dt.total_seconds().fillna(0)

    # Step 4: Correlation Analysis
    # Compute the autocorrelation of the time differences
    autocorrelation = df['time_diff_seconds'].autocorr()

    # Print the autocorrelation
    print("Autocorrelation of time differences between default events:", autocorrelation)

    # Plot the time differences to visualize the pattern
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='datetime', y='time_diff_seconds')
    plt.title('Time Differences Between Default Events')
    plt.xlabel('Datetime')
    plt.ylabel('Time Difference (seconds)')
    plt.show()

