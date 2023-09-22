import numpy as np
import os
import pandas as pd
import pickle

class ECGDataset:
    fs = 1000

    # Load the data from the folder
    def __init__(self, dataset_location):
        self.dataset_location = dataset_location

        # database_annotations_path = dataset_location + '/ucsd_ecg_dataset.pkl'

        # if os.path.exists(database_annotations_path):
        #     self.ucsd_ecg_df = pd.read_pickle(database_annotations_path)
        # else:
        # Get subdirectories from the ./ucsd-ekg-data directory
        subdirs = [x[0] for x in os.walk("ucsd-ekg-data")][1:]

        self.ucsd_ecg_df = pd.DataFrame(columns=["subdir", "file", "location", "lead", "type",  "timestamp", "session_id", "category","sampling_rate", "annotations","vcgtrigger"])

        # Create a dictionary of subdirectories and files
        files = {}
        all_parts = []
        for subdir in subdirs:
            files[subdir] = [x[2] for x in os.walk(subdir)][0]

            for file in files[subdir]:
                # Split the file name into parts
                parts = file.split("_")
                # Create a dictionary of the parts
                parts_dict = {}
                parts_dict["subdir"] = subdir
                parts_dict["file"] = file
                # name of the directory is the location
                parts_dict["location"] = subdir.split("/")[-1]
                parts_dict["lead"] = parts[0][:4]
                parts_dict["type"] = parts[0][4:]
                parts_dict["timestamp"] = "_".join(parts[2:5])
                parts_dict["session_id"] = parts[5]

                all_parts.append(parts_dict)
                
        # Append the dictionary to the dataframe
        
        new_df = pd.DataFrame(all_parts)
        self.ucsd_ecg_df = pd.concat([self.ucsd_ecg_df, new_df], ignore_index=True)
        # Save the dataframe to a pickle file
        #self.ucsd_ecg_df.to_pickle(database_annotations_path)

    def get_first_datafile_info(self):
        # Get the first row of the dataframe
        first_row = self.ucsd_ecg_df.iloc[0]
        # Get the location, session_id and timestamp
        location = first_row["location"]
        timestamp = first_row["timestamp"]
        session_id = first_row["session_id"]
        
        # Return the location, session_id and timestamp
        return location, timestamp, session_id
    
    def get_trigger_detections(self, location,timestamp,session_id):
        # Get the file name
        file_name = self.dataset_location + f"/{location}/ECG3Trig_fgre_{timestamp}_{session_id}"

        print("Trigger file: ", file_name)
        
        # Load the data from the file
        data = np.loadtxt(open(file_name, "rb"), delimiter=",", skiprows=1)

        # Return the data
        return data
    
    def get_dataset_dataframe(self) -> pd.DataFrame:
        return self.ucsd_ecg_df

    # Get ECG Data. Returns a numpy array with the ECG data, with rows being separate channels
    def get_ecg_data(self, location, timestamp, session_id):
        ecg2File = self.dataset_location + f"/{location}/ECG2Data_fgre_{timestamp}_{session_id}"
        ecg3File = self.dataset_location + f"/{location}/ECG3Data_fgre_{timestamp}_{session_id}"

        # Check if the file exists
        if not os.path.exists(ecg2File) or not os.path.exists(ecg3File):
            # Throw exception
            raise IOError(f"File does not exist: {ecg2File}")
                
        raw_ecg2_data = np.loadtxt(open(ecg2File, "rb"), delimiter=",", skiprows=1)
        raw_ecg3_data = np.loadtxt(open(ecg3File, "rb"), delimiter=",", skiprows=1)

        # return the two numpy arrays concatenated as a single numpy array
        return np.stack((raw_ecg2_data, raw_ecg3_data), axis=0)

    # From ucsd_ecg_df get number of unique session_id,timestamp,location combinations
    # for lead ECG2
    def get_unique_combinations(self):
        # Get the unique combinations of session_id, timestamp and location
        unique_combinations = self.ucsd_ecg_df[self.ucsd_ecg_df["lead"] == "ECG2"][["location","timestamp", "session_id"]].drop_duplicates()
        # Return the number of unique combinations
        return unique_combinations