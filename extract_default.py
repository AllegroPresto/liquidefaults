import os
import re
import numpy as np

import sys
def FQ(label):
    print ('------------- FIN QUI TUTTO OK  %s ----------' %(label))
    sys.exit()


import pandas as pd
import os

def load_csv_files(folder_path, start_tag):
    # List to hold dataframes
    dataframes_list = []
    field_list = []

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
                field_list.append(file_name)

    return dataframes_list, field_list



def extract_default_events_(folder_path, start_tag, file_code, xlsx_path, output_csv):

    df_list, file_list = load_csv_files(folder_path, start_tag)

    # read excel scheme
    #file_code = csv_path.split('1_')[1][:3]  # Extract the three-letter code
    fields_df = pd.read_excel(xlsx_path, sheet_name='info_field_0')


    row = fields_df[fields_df['FileCode'] == file_code].iloc[0]

    origin_date_field   = f"{row['ExpSymbol']}{int(row['OriginDate'])}"
    default_date_field  = f"{row['ExpSymbol']}{int(row['DefaultDate'])}"
    prepay_date_field   = f"{row['ExpSymbol']}{int(row['PrepaymentDate'])}"
    mat_date_field      = f"{row['ExpSymbol']}{int(row['MaturityDate'])}"


    for i in range(0, len(df_list)):

        df    = df_list[i]
        file_ = file_list[i]
        #df = pd.read_csv(csv_path)

        # Filter out rows where the default date is not available or contains "ND5"
        default_events = df[df[default_date_field].notna() & ~df[default_date_field].astype(str).str.contains("ND5")]

        # Calculate the time difference

        # Convert the dates to datetime format
        default_events.loc[:, 'DefaultDate'] = pd.to_datetime(default_events[default_date_field])
        df.loc[:, 'PrepaymentDate'] = pd.to_datetime(df[prepay_date_field], errors='coerce')
        df.loc[:, 'MaturityDate'] = pd.to_datetime(df[mat_date_field], errors='coerce')

        default_events.loc[:, 'TimeToDefault'] = (default_events['DefaultDate'] - pd.to_datetime(df[origin_date_field], errors='coerce')).dt.days

        total_credits = df.shape[0]

        # Calculate the number of credits alive at each default date
        number_of_credits = []
        for default_date in default_events['DefaultDate']:
            prepayments_before_default = df.loc[df['PrepaymentDate'] < default_date].shape[0]
            maturities_before_default = df.loc[df['MaturityDate'] < default_date].shape[0]
            credits_alive = total_credits - (prepayments_before_default + maturities_before_default)
            number_of_credits.append(credits_alive)

        n_def = len(default_events['DefaultDate'])

        print('total_credits: ', total_credits)
        print('file_: ', file_)
        #print('number_of_credits: ', number_of_credits)
        print('number_of_default:  ', n_def)
        print('======================================')

        if n_def > 1:
            # Add NumberOfCredits to the result DataFrame
            default_events['NumberOfCredits'] = number_of_credits

            result_df = default_events[['DefaultDate','TimeToDefault', 'NumberOfCredits']]
            file_out = file_.split('_2022')[0][3:] + '.csv'

            output_file = os.path.join(output_csv, file_out)

            result_df.to_csv(output_file, index=False)



def extract_default_events(csv_path, xlsx_path, output_csv):

    # read excel scheme
    file_code = csv_path.split('1_')[1][:3]  # Extract the three-letter code
    fields_df = pd.read_excel(xlsx_path, sheet_name='info_field_0')


    row = fields_df[fields_df['FileCode'] == file_code].iloc[0]

    origin_date_field   = f"{row['ExpSymbol']}{int(row['OriginDate'])}"
    default_date_field  = f"{row['ExpSymbol']}{int(row['DefaultDate'])}"
    prepay_date_field   = f"{row['ExpSymbol']}{int(row['PrepaymentDate'])}"
    mat_date_field      = f"{row['ExpSymbol']}{int(row['MaturityDate'])}"

    df = pd.read_csv(csv_path)


    # Filter out rows where the default date is not available or contains "ND5"
    default_events = df[df[default_date_field].notna() & ~df[default_date_field].astype(str).str.contains("ND5")]

    # Calculate the time difference

    # Convert the dates to datetime format
    default_events.loc[:, 'DefaultDate'] = pd.to_datetime(default_events[default_date_field])
    df.loc[:, 'PrepaymentDate'] = pd.to_datetime(df[prepay_date_field], errors='coerce')
    df.loc[:, 'MaturityDate'] = pd.to_datetime(df[mat_date_field], errors='coerce')

    default_events.loc[:, 'TimeToDefault'] = (default_events['DefaultDate'] - pd.to_datetime(df[origin_date_field], errors='coerce')).dt.days

    print('df.shape: ', df.shape)
    print('default_events.shape: ', default_events.shape)



    total_credits = df.shape[0]

    # Calculate the number of credits alive at each default date
    number_of_credits = []
    for default_date in default_events['DefaultDate']:
        prepayments_before_default = df.loc[df['PrepaymentDate'] < default_date].shape[0]
        maturities_before_default = df.loc[df['MaturityDate'] < default_date].shape[0]
        credits_alive = total_credits - (prepayments_before_default + maturities_before_default)
        number_of_credits.append(credits_alive)


    # Add NumberOfCredits to the result DataFrame
    default_events['NumberOfCredits'] = number_of_credits

    result_df = default_events[['DefaultDate','TimeToDefault', 'NumberOfCredits']]



    # Write to CSV
    result_df.to_csv(output_csv, index=False)


def calculate_credit_number(row, df):
    # Current defaultDate
    current_date = row['DefaultDate']
    total_credits = len(df)

    # Count of previous prepayDates, defaultDates, and maturityDates
    prepay_count = df[df['PrepayDate'] < current_date].shape[0]
    default_count = df[df['DefaultDate'] < current_date].shape[0]
    maturity_count = df[df['MaturityDate'] < current_date].shape[0]

    # Total previous relevant dates
    total_previous_dates = prepay_count + default_count + maturity_count

    # Calculate creditNumber
    return total_credits - total_previous_dates



if __name__ == "__main__":



    xlsx_path = r'C:\Users\proprietario\Desktop\UCD\lavori\dataset_prepay\schema_templates_esma_v3.xlsx'  # Update this to your schema file path

    csv_path = r'C:\Users\proprietario\Desktop\UCD\lavori\dataset_prepay\data_to_analyze'
    output_path = r'C:\Users\proprietario\Desktop\UCD\lavori\dataset_prepay\default_events_v0'

    start_tag = 'X2'
    start_tag = 'X4'
    start_tag = 'X3'

    file_code = 'SME'
    file_code = 'AUT'
    file_code = 'CMR'

    extract_default_events_(csv_path, start_tag, file_code, xlsx_path, output_path)

