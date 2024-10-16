import os
import zipfile
import shutil
from datetime import datetime


import sys
def FQ(label):
    print ('------------- FIN QUI TUTTO OK  %s ----------' %(label))
    sys.exit()


def get_most_recent_date_folder(parent_folder):
    date_folders = [f for f in os.listdir(parent_folder) if os.path.isdir(os.path.join(parent_folder, f))]
    date_folders.sort(key=lambda date: datetime.strptime(date, "%Y-%m-%d"), reverse=True)
    return date_folders[0] if date_folders else None


def unzip_folder(zip_path, extract_to):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)


def copy_unzipped_files(source_folder, dest_folder):
    for root, _, files in os.walk(source_folder):
        for file in files:
            #if file.startswith("1_") and "Collateral" not in file:
            if file.startswith("1_"):

                shutil.copy(os.path.join(root, file), dest_folder)


def main(root_folder, dest_folder):
    root_subfolders = [f for f in os.listdir(root_folder) if os.path.isdir(os.path.join(root_folder, f))]
    k = 1
    k_tot = len(root_subfolders)
    for subfolder in root_subfolders:
        print(f"Start elaboration of subfolder n. {k, k_tot}.")
        k = k + 1

        subfolder_path = os.path.join(root_folder, subfolder)
        most_recent_date_folder = get_most_recent_date_folder(subfolder_path)

        if not most_recent_date_folder:
            print(f"No date folders found in {subfolder}.")
            continue

        most_recent_date_folder_path = os.path.join(subfolder_path, most_recent_date_folder)
        subfolders = [f for f in os.listdir(most_recent_date_folder_path) if f.startswith("1") and f.endswith(".zip")]

        if not subfolders:
            print(f"No zipped folders starting with '1' found in {most_recent_date_folder_path}.")
            continue

        source_zip_path = os.path.join(most_recent_date_folder_path, subfolders[0])
        unzipped_folder = os.path.join(most_recent_date_folder_path, "unzipped")
        os.makedirs(unzipped_folder, exist_ok=True)

        unzip_folder(source_zip_path, unzipped_folder)
        copy_unzipped_files(unzipped_folder, dest_folder)



if __name__ == "__main__":
    root_folder = r"C:\Users\proprietario\Desktop\UCD\lavori\dataset_prepay\EDW_DATA_s2\ESMA_CSVs_s"  # Change this to your root folder path
    dest_folder = r"C:\Users\proprietario\Desktop\UCD\lavori\dataset_prepay\data_to_analyze2"  # Change this to your destination folder path
    main(root_folder, dest_folder)
