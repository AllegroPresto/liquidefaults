
import pandas as pd
import os


def load_schema(xlsx_file):
    """Load the schema from the xlsx file."""
    return pd.read_excel(xlsx_file, sheet_name='info_field_0')


def get_field_names(file_code, schema_df):
    """Construct field names for Default Date and Prepayment Date."""
    row = schema_df[schema_df['FileCode'] == file_code].iloc[0]

    default_date_field = f"{row['ExpSymbol']}{int(row['DefaultDate'])}"
    prepayment_date_field = f"{row['ExpSymbol']}{int(row['PrepaymentDate'])}"
    credit_type_field = f"{row['ExpSymbol']}"

    return default_date_field, prepayment_date_field, credit_type_field


def analyze_csv(csv_file, default_date_field, prepayment_date_field, credit_type):
    """Analyze the CSV file and calculate statistics."""
    df = pd.read_csv(csv_file)
    total_credits = len(df)
    #default_events = df[default_date_field].notna().sum()
    default_events = df[df[default_date_field].notna() & ~df[default_date_field].str.contains("ND5", na=False)].shape[0]

    #prepayment_events = df[prepayment_date_field].notna().sum()
    prepayment_events = df[df[prepayment_date_field].notna() & ~df[prepayment_date_field].str.contains("ND5", na=False)].shape[0]


    default_ratio = default_events / total_credits if total_credits > 0 else 0
    prepayment_ratio = prepayment_events / total_credits if total_credits > 0 else 0
    return total_credits, default_events, prepayment_events, default_ratio, prepayment_ratio, credit_type


def write_statistics(filename, stats):
    """Write the statistics to a file."""
    with open(filename, 'w') as f:
        f.write(f"Credit Type: {stats['credit_type']}\n")
        f.write(f"Total Credits: {stats['total_credits']}\n")
        f.write(f"Default Events: {stats['default_events']}\n")
        f.write(f"Prepayment Events: {stats['prepayment_events']}\n")
        f.write(f"Default Ratio: {stats['default_ratio']}\n")
        f.write(f"Prepayment Ratio: {stats['prepayment_ratio']}\n")


def main(root_folder, schema_file, dest_folder):
    schema_df = load_schema(schema_file)

    for filename in os.listdir(root_folder):
        print('Filename: %s'%(filename))
        if filename.startswith('1_') and filename.endswith('.csv'):
            file_code = filename[2:5]
            csv_file_path = os.path.join(root_folder, filename)

            default_date_field, prepayment_date_field, credit_type = get_field_names(file_code, schema_df)

            stats = analyze_csv(csv_file_path, default_date_field, prepayment_date_field, credit_type)

            stats_filename = os.path.join(dest_folder, f"statistics_{filename[:-4]}.txt")
            stats_data = {
                'credit_type': stats[5],
                'total_credits': stats[0],
                'default_events': stats[1],
                'prepayment_events': stats[2],
                'default_ratio': stats[3],
                'prepayment_ratio': stats[4],
            }
            write_statistics(stats_filename, stats_data)

            print(f"Statistics written to {stats_filename}")


if __name__ == "__main__":
    root_folder = r'C:\Users\proprietario\Desktop\UCD\lavori\dataset_prepay\data_to_analyze1'  # Update this to your root folder path
    schema_file = r'C:\Users\proprietario\Desktop\UCD\lavori\dataset_prepay\schema_templates_esma_v3.xlsx'  # Update this to your schema file path
    dest_folder = r'C:\Users\proprietario\Desktop\UCD\lavori\dataset_prepay\data_statistics'  # Update this to your destination folder path
    main(root_folder, schema_file, dest_folder)


    # Load the .xlsx schema file
    #xlsx_file = r'C:\Users\proprietario\Desktop\UCD\lavori\dataset_prepay\schema_templates_esma_v3.xlsx'  # Update this path
    #info_field_df = pd.read_excel(xlsx_file, sheet_name='info_field_0')

    # Directory containing the .csv files
