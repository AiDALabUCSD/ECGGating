# Prepare training dataset for our model from ucsd data and labelstudio annotations
import os
import numpy as np
import pandas as pd
import labelstudio_tools as lst

# File containing annotations from label studio
l = lst.LabelStudioTools()
df1 = l.convert_json_annotations_to_dataframe("./data/label_studio/original_annotations/export_29479_project-29479-at-2023-07-01-12-13-8a13df80.json")
df2 = l.convert_json_annotations_to_dataframe("./data/label_studio/original_annotations/project-1-at-2023-07-01.json")
df = pd.concat([df1, df2])
print(df)

# Iterate throug hthe whole dataset and get the data for each row in the dataframe
# and save it to a numpy file and annotations to a different corresponding numpy file
for index, row in df.iterrows():
    location = row["location"]
    timestamp = row["timestamp"]
    session_id = row["session_id"]

    # if the annotation file does not exist in the ./data/old_preprocessed/ folder, skip the row
    # otherwise save the annotations to the corresponding files, unless there is an error
    if os.path.exists(f"./data/old_preprocessed/data_anno_{location}_{timestamp}_{session_id}.npy"):
        print(f"Annotation Exists for: {location} {timestamp} {session_id}")
        continue
    else: 
        print(f"Processing {location} {timestamp} {session_id}")
        try: np.save(f"./data/old_preprocessed/data_anno_{location}_{timestamp}_{session_id}.npy",sorted(row["annotations"]))
        except Exception as e:
            print(f"Skipping: {location} {timestamp} {session_id}")
            print(e)
            continue