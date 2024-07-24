import re
import pandas as pd

# The provided text data
text = """
Asset Type: AUTL
Number of Ptf: 119
Max num. of Credits: 1040540
Min num. of Credits: 10587
Average num. of Credits: 94105
Max num. of Default Events: 9147
Min num. of Default Events: 0
Average num. of Default Events: 462
Max num. of Prepayment Events: 53446
Min num. of Prepayment Events: 0
Average num. of Prepayment Events: 2282.563025210084
Max Default Ratio: 3.5109
Min Default Ratio: 0.0
Average Default Ratio: 0.0762
Max Prepayment Ratio: 8.1446
Min Prepayment Ratio: 0.0
Average Prepayment Ratio: 0.2220

Asset Type: CREL
Number of Ptf: 1
Max num. of Credits: 2342
Min num. of Credits: 2342
Average num. of Credits: 2342
Max num. of Default Events: 2320
Min num. of Default Events: 2320
Average num. of Default Events: 2320
Max num. of Prepayment Events: 239
Min num. of Prepayment Events: 239
Average num. of Prepayment Events: 239.0
Max Default Ratio: 0.9906
Min Default Ratio: 0.9906
Average Default Ratio: 0.9906
Max Prepayment Ratio: 0.102
Min Prepayment Ratio: 0.102
Average Prepayment Ratio: 0.1020

Asset Type: CMRL
Number of Ptf: 62
Max num. of Credits: 5108886
Min num. of Credits: 5532
Average num. of Credits: 218012
Max num. of Default Events: 250665
Min num. of Default Events: 0
Average num. of Default Events: 6773
Max num. of Prepayment Events: 250665
Min num. of Prepayment Events: 0
Average num. of Prepayment Events: 19355.74193548387
Max Default Ratio: 1.0
Min Default Ratio: 0.0
Average Default Ratio: 0.0660
Max Prepayment Ratio: 1.0
Min Prepayment Ratio: 0.0
Average Prepayment Ratio: 0.1366

Asset Type: LESL
Number of Ptf: 13
Max num. of Credits: 75026
Min num. of Credits: 1361
Average num. of Credits: 22942
Max num. of Default Events: 516
Min num. of Default Events: 0
Average num. of Default Events: 159
Max num. of Prepayment Events: 57532
Min num. of Prepayment Events: 0
Average num. of Prepayment Events: 4959.307692307692
Max Default Ratio: 0.0351
Min Default Ratio: 0.0
Average Default Ratio: 0.0077
Max Prepayment Ratio: 1.0
Min Prepayment Ratio: 0.0
Average Prepayment Ratio: 0.1022

Asset Type: RREL
Number of Ptf: 133
Max num. of Credits: 525641
Min num. of Credits: 802
Average num. of Credits: 20893
Max num. of Default Events: 8162
Min num. of Default Events: 0
Average num. of Default Events: 192
Max num. of Prepayment Events: 43145
Min num. of Prepayment Events: 0
Average num. of Prepayment Events: 2924.93984962406
Max Default Ratio: 1.5812
Min Default Ratio: 0.0
Average Default Ratio: 0.0407
Max Prepayment Ratio: 1.0
Min Prepayment Ratio: 0.0
Average Prepayment Ratio: 0.2481

Asset Type: CRPL
Number of Ptf: 33
Max num. of Credits: 205221
Min num. of Credits: 47
Average num. of Credits: 21328
Max num. of Default Events: 10571
Min num. of Default Events: 0
Average num. of Default Events: 738
Max num. of Prepayment Events: 205130
Min num. of Prepayment Events: 0
Average num. of Prepayment Events: 8737.818181818182
Max Default Ratio: 9.6308
Min Default Ratio: 0.0
Average Default Ratio: 0.3669
Max Prepayment Ratio: 1.0
Min Prepayment Ratio: 0.0
Average Prepayment Ratio: 0.1357
"""

# Split the text by double newlines to separate each asset block
assets_data = text.strip().split('\n\n')

# Initialize an empty list to store the structured data
assets_list = []

# Define a pattern to extract the key-value pairs
pattern = re.compile(r'([^:]+):\s*([\d.]+|\w+)')

# Iterate over each asset block
for asset in assets_data:
    # Create a dictionary to store the asset's data
    asset_dict = {}

    # Find all key-value pairs in the block
    matches = pattern.findall(asset)

    # Add each key-value pair to the dictionary
    for key, value in matches:
        # Convert numeric values to appropriate types
        if value.replace('.', '', 1).isdigit():
            if '.' in value:
                asset_dict[key.strip()] = float(value)
            else:
                asset_dict[key.strip()] = int(value)
        else:
            asset_dict[key.strip()] = value.strip()

    # Append the dictionary to the list
    assets_list.append(asset_dict)

# Create a DataFrame with the required headers
columns = [
    "N.", "Asset Class", "Symbol", "N. of Ptf.", "Max N. of Credits",
    "Min Num. of Credits", "Max. Num. default", "Min. Num. Defaults",
    "Avg. Default Ratio", "Maximum Length of credits"
]

data = []
for i, asset in enumerate(assets_list, start=1):
    data.append([
        i,
        asset.get('Asset Type', ''),
        asset.get('Asset Type', ''),
        asset.get('Number of Ptf', ''),
        asset.get('Max num. of Credits', ''),
        asset.get('Min num. of Credits', ''),
        asset.get('Max num. of Default Events', ''),
        asset.get('Min num. of Default Events', ''),
        asset.get('Average Default Ratio', ''),
        asset.get('Max num. of Credits', '')  # Assuming 'Maximum Length of credits' is same as 'Max num. of Credits'
    ])

df = pd.DataFrame(data, columns=columns)

# Save the DataFrame to an Excel file
df.to_excel('assets_data.xlsx', index=False)
