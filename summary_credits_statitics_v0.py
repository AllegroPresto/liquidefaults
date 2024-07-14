import os
import re
import numpy as np

import sys
def FQ(label):
    print ('------------- FIN QUI TUTTO OK  %s ----------' %(label))
    sys.exit()



def parse_statistics_file(filepath):
    """Parse a single statistics file to extract relevant data."""
    with open(filepath, 'r') as f:
        content = f.read()

    def extract_value(pattern, content, cast_type=str):
        match = re.search(pattern, content)
        if match:
            return cast_type(match.group(1))
        else:
            raise ValueError(f"Pattern {pattern} not found in {filepath}")


    # Extract relevant data using regular expressions
    file_code = extract_value(r"Credit Type: (.+)", content, str)
    total_credits = extract_value(r"Total Credits: (\d+)", content, int)
    default_events = extract_value(r"Default Events: (\d+)", content, int)
    prepayment_events = extract_value(r"Prepayment Events: (\d+)", content, int)
    default_ratio = extract_value(r"Default Ratio: (\d+\.\d{1,4}|\d+)", content, float)
    prepayment_ratio = extract_value(r"Prepayment Ratio: (\d+\.\d{1,4}|\d+)", content, float)

    return {
        'file_code': file_code,
        'total_credits': total_credits,
        'default_events': default_events,
        'prepayment_events': prepayment_events,
        'default_ratio': default_ratio,
        'prepayment_ratio': prepayment_ratio,
    }


def summarize_statistics(stats_folder, summary_file):
    """Summarize statistics from individual files and write to a summary file."""
    summary = {}

    for filename in os.listdir(stats_folder):
        if filename.startswith('statistics_') and filename.endswith('.txt'):
            filepath = os.path.join(stats_folder, filename)
            try:
                stats = parse_statistics_file(filepath)
                file_code = stats['file_code']

                if file_code not in summary:
                    summary[file_code] = {
                        'num_files': 0,
                        'total_credits': [],
                        'default_events': [],
                        'prepayment_events': [],
                        'default_ratio': [],
                        'prepayment_ratio': []
                    }
                summary[file_code]['num_files'] += 1
                summary[file_code]['total_credits'].append(stats['total_credits'])
                summary[file_code]['default_events'].append(stats['default_events'])
                summary[file_code]['prepayment_events'].append(stats['prepayment_events'])
                summary[file_code]['default_ratio'].append(stats['default_ratio'])
                summary[file_code]['prepayment_ratio'].append(stats['prepayment_ratio'])

            except ValueError as e:
                print(f"Error processing file {filename}: {e}")

    with open(summary_file, 'w') as f:
        for file_code, stats in summary.items():
            f.write(f"Asset Type: {file_code}\n")
            f.write(f"Number of Ptf: {stats['num_files']}\n")
            f.write(f"Max num. of Credits: {int(max(stats['total_credits']))}\n")
            f.write(f"Min num. of Credits: {min(stats['total_credits'])}\n")
            f.write(f"Average num. of Credits: {int(np.mean(stats['total_credits']))}\n")
            f.write(f"Max num. of Default Events: {max(stats['default_events'])}\n")
            f.write(f"Min num. of Default Events: {min(stats['default_events'])}\n")
            f.write(f"Average num. of Default Events: {int(np.mean(stats['default_events']))}\n")
            f.write(f"Max num. of Prepayment Events: {max(stats['prepayment_events'])}\n")
            f.write(f"Min num. of Prepayment Events: {min(stats['prepayment_events'])}\n")
            f.write(f"Average num. of Prepayment Events: {np.mean(stats['prepayment_events'])}\n")
            f.write(f"Max Default Ratio: {max(stats['default_ratio'])}\n")
            f.write(f"Min Default Ratio: {min(stats['default_ratio'])}\n")
            f.write(f"Average Default Ratio: {np.mean(stats['default_ratio']):.4f}\n")
            f.write(f"Max Prepayment Ratio: {max(stats['prepayment_ratio'])}\n")
            f.write(f"Min Prepayment Ratio: {min(stats['prepayment_ratio'])}\n")
            f.write(f"Average Prepayment Ratio: {np.mean(stats['prepayment_ratio']):.4f}\n")
            f.write("\n")



if __name__ == "__main__":
    stats_folder = r'C:\Users\proprietario\Desktop\UCD\lavori\dataset_prepay\data_statistics'  # Update this to your destination folder path
    summary_file = r'C:\Users\proprietario\Desktop\UCD\lavori\dataset_prepay\data_statistics\summary_statistics.txt'  # Update this to your destination folder path


    summarize_statistics(stats_folder, summary_file)
