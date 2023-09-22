# Prepare training dataset for our model from ucsd data and labelstudio annotations

import pandas as pd
import numpy as np
import labelstudio_tools as lst
import os
from ucsd_ecg_dataset import ECGDataset

# File containing annotations from label studio

l = lst.LabelStudioTools("label_studio_files")

df = l.convert_json_annotations_to_dataframe("./data/label_studio_annotations/export_29479_project-29479-at-2023-07-01-12-13-8a13df80.json")
df2 = l.convert_json_annotations_to_dataframe("./data/label_studio_annotations/project-1-at-2023-07-01.json")

df = pd.concat([df, df2])

# Get the dataset
ucsd_ds = ECGDataset("/media/vlermakov/data/UCSD_ECG_SOURCE_DATA")

# Iterate throug hthe whole dataset and get the data for each row in the dataframe
# and save it to a numpy file and annotations to a different corresponding numpy file

for index, row in df.iterrows():
    location = row["location"]
    timestamp = row["timestamp"]
    session_id = row["session_id"]

    print(f"Processing {location} {timestamp} {session_id}")

    # Based on the location, timestamp and session_id load data from the ECG2 and ECG3 channel files
    # that are of type "Data" into two numpy arrays
    try:
        raw_ecg_data = ucsd_ds.get_ecg_data(location,timestamp,session_id)
        np.save(f"./data/preprocessed/data_{location}_{timestamp}_{session_id}.npy",raw_ecg_data[0,:])
        np.save(f"./data/preprocessed/data_anno_{location}_{timestamp}_{session_id}.npy",sorted(row["annotations"]))
    except:
        print(f"Skipping {location} {timestamp} {timestamp}")
        continue