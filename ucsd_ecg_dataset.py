import os
import numpy as np
import pandas as pd

class ECGDataset:
    def __init__(self, dataset_location):
        """
        Initialize the ECG dataset from the folder by setting the dataset_location to the folder.

        Args:
            dataset_location (str): The overall dataset folder which contains subfolders (locations) with 
            files (ECGs, PPGs, Triggers) in each subfolder.
        """
        self.dataset_location = dataset_location


    def get_first_datafile_info(self):
        """
        Get the location, session_id and timestamp of the first row of the dataframe.
        
        Returns:
            location (str): The location of the ecg script data from title of the file
            timestamp (str): The timestamp of the ecg script data from title of the file
            session_id (str): The session_id of the ecg script data from title of the file
        """
        # Get the dataframe
        ucsd_ecg_df = self.get_dataset_dataframe()
        first_row = ucsd_ecg_df.iloc[0]
        location = first_row["location"]
        timestamp = first_row["timestamp"]
        session_id = first_row["session_id"]
        return location, timestamp, session_id


    def get_trigger_detections(self, location, timestamp, session_id, lead = 'ECG3', iStart=None, iStop=None):
        """
        Get the VCG trigger detections from the Trig file and return as a numpy array.

        Args:
            location (str): The location of the ecg script triggers from title of the file
            timestamp (str): The timestamp of the ecg script triggers from title of the file
            session_id (str): The session_id of the ecg script triggers from title of the file
            lead (str): The lead of the ecg script triggers from title of the file. Default is 'ECG3'
            iStart (float): The starting index of the signal to load. Default is None.
            iStop (float): The stopping index of the signal to load. Default is None.

        Returns:
            final_data (np.array): A numpy array with the trigger detections scaled to 250Hz and cropped.
        """
        # Check if the session_id has _ecg3 at the end and remove it, use str() to convert to string if it's not
        if str(session_id).endswith("_ecg3"): session_id = session_id[:-5]

        # Get the file name from the folder and load the raw trigger data
        file_name = os.path.join(self.dataset_location , f"{location}/{lead}Trig_fgre_{timestamp}_{session_id}")
        # print("Trigger file: ", file_name)
        raw_data = np.loadtxt(open(file_name, "rb"), delimiter=",", skiprows=1)

        # Scale the trigger data to match 250Hz sampling rate from 1000Hz (1000/4 = 250)
        scaled_data = raw_data / 4.0

        # Set the start and stop indices of the triggers
        if(iStart == None): iStart = 0
        else: iStart = max(iStart, 0)
        if(iStop == None): iStop = max(scaled_data) 
        else: iStop = min(iStop, max(scaled_data))

        # Crop the triggers to the specified start and stop indices
        final_data = np.array([a - iStart for a in scaled_data if (a >= iStart and a <= iStop)])
        return final_data
    

    def get_pulse_detections(self, location, timestamp, session_id, iStart=None, iStop=None):
        """
        Get the PPG pulse detections from the PPGTrig file and return as a numpy array.

        Args:
            location (str): The location of the ecg script pulse triggers from title of the file
            timestamp (str): The timestamp of the ecg script pulse triggers from title of the file
            session_id (str): The session_id of the ecg script pulse triggers from title of the file

        Returns:
            data (np.array): A numpy array with the pulse detections
        """
        # Check if the session_id has _ecg3 at the end and remove it
        if session_id.endswith("_ecg3"): session_id = session_id[:-5]

        # Get the file name from the local folder and load the data
        file_name = os.path.join(self.dataset_location , f"{location}/PPGTrig_fgre_{timestamp}_{session_id}")
        print("Pulse Trigger file: ", file_name)

        raw_data = np.loadtxt(open(file_name, "rb"), delimiter=",", skiprows=1)

        # Scale the trigger data to match 250Hz sampling rate from 100Hz ((100/4)*10 = 250)
        scaled_data = (raw_data / 4.0) * 10.0

        # Set the start and stop indices of the triggers
        if(iStart == None): iStart = 0
        else: iStart = max(iStart, 0)
        if(iStop == None): iStop = max(scaled_data) 
        else: iStop = min(iStop, max(scaled_data))

        # Crop the triggers to the specified start and stop indices
        final_data = np.array([a - iStart for a in scaled_data if (a >= iStart and a <= iStop)])
        return final_data
    

    def get_dataset_dataframe(self) -> pd.DataFrame:
        """
        Get the dataframe with the all the ECGs or Trigger files in the folder.
        Create a pandas DataFrame with the information for each file saved in the rows of the dataframe.
        
        Returns:
            ucsd_ecg_df (pd.DataFrame): A pandas DataFrame with the dataset information
        """
        # # Load the dataframe annotations from the pickle file if it exists
        # if os.path.exists(database_annotations_path):
        #     ucsd_ecg_df = pd.read_pickle(database_annotations_path)
        # else:
        #     database_annotations_path = dataset_location + '/ucsd_ecg_dataset.pkl'

        # Get subdirectories from the ECG dataset_location (main folder)
        subdirs = [x[0] for x in os.walk(self.dataset_location)][1:] # skip the main folder

        ucsd_ecg_df = pd.DataFrame(columns=["subdir", "file", "location", "lead", "type",  
                                                 "timestamp", "session_id", "category",
                                                 "sampling_rate", "annotations","vcgtrigger"])

        # Create a dictionary of subdirectories (locations) and files (ECGs, PPGs, Triggers)
        files = {}
        all_parts = []
        for subdir in subdirs:
            # Get the files in the subdirectory
            files[subdir] = [x[2] for x in os.walk(subdir)][0]

            for file in files[subdir]:
                try:
                    # Split the file name into parts as separated by "_"
                    parts = file.split("_")

                    # Create a dictionary of the parts of the file name
                    parts_dict = {}
                    parts_dict["subdir"] = subdir
                    parts_dict["file"] = file

                    # name of the directory is the location, which is the last part of the path
                    parts_dict["location"] = subdir.split("/")[-1]
                    # lead is the first 4 characters of the first part of file name (ECG2 or ECG3 or PPG)
                    parts_dict["lead"] = parts[0][:4]
                    # type is the next 4 characters of the first part of file name (Data or Trig)
                    parts_dict["type"] = parts[0][4:]
                    # timestamp is the 3rd (year), 4th (month), 5th (day) parts of the file name
                    parts_dict["timestamp"] = "_".join(parts[2:5])
                    # session_id is the final part of the file name
                    parts_dict["session_id"] = parts[5]

                    all_parts.append(parts_dict)

                except Exception as e: 
                    print(f"Error processing: {file}, error: {e}")
                    continue

        # Append the dictionary to the dataframe such that each row is a file
        new_df = pd.DataFrame(all_parts)
        ucsd_ecg_df = pd.concat([ucsd_ecg_df, new_df], ignore_index=True)

        ## Save the dataframe to a pickle file
        # ucsd_ecg_df.to_pickle(database_annotations_path)
        return ucsd_ecg_df

    
    def get_ecg_data(self, location, timestamp, session_id):
        """
        Read the ECG data from both ECG leads and return together as a stacked numpy array.

        Args:
            location (str): The location of the ecg script data from title of the file
            timestamp (str): The timestamp of the ecg script data from title of the file
            session_id (str): The session_id of the ecg script data from title of the file
        
        Returns:
            ecg_data (np.array): A numpy array with the ECG data, with rows being channels (ECG2 and ECG3)
        """
        # Get the file names from the folder
        ecg2File = os.path.join(self.dataset_location , f"{location}/ECG2Data_fgre_{timestamp}_{session_id}")
        ecg3File = os.path.join(self.dataset_location , f"{location}/ECG3Data_fgre_{timestamp}_{session_id}")

        # Check if the ecg2File or ecg3File exist
        if not os.path.exists(ecg2File): raise IOError(f"File does not exist: {ecg2File}")
        if not os.path.exists(ecg3File): raise IOError(f"File does not exist: {ecg3File}")

        # Vlad skipped the first row, so I will do the same
        # however, in these files the first row is not a header, it has data  
        raw_ecg2_data = np.loadtxt(open(ecg2File, "rb"), delimiter=",", skiprows=1)
        raw_ecg3_data = np.loadtxt(open(ecg3File, "rb"), delimiter=",", skiprows=1)
        
        # return the two numpy arrays concatenated as a single numpy array
        return np.stack((raw_ecg2_data, raw_ecg3_data), axis=0)


class ProcessedDataset:
    def __init__(self, dataset_location):
        """
        Initialize the processed dataset from the folder by setting the dataset_location to the folder.
        
        Args:
            dataset_location (str): The location of the dataset folder
        """
        self.dataset_location = dataset_location


    def get_dataset_dataframe(self) -> pd.DataFrame:
        """
        Get the dataframe with the all the ECGs or Trigger files in the folder.
        Create a pandas DataFrame with the information for each file saved in the rows of the dataframe.
        
        Returns:
            ucsd_ecg_df (pd.DataFrame): A pandas DataFrame with the dataset information
        """
        ucsd_ecg_df = pd.DataFrame(columns=["file", "location", "timestamp", "session_id","lead"])

        # Create a list to store all parts dictionaries
        all_parts = []

        # Get files from the processed ECG dataset_location 
        files = [x[2] for x in os.walk(self.dataset_location)][0]

        for file in files:
            try:
                # Split the file name into parts as separated by "_"
                # Skip if it's an annotation file
                parts = file.split("_")
                if parts[1] == "anno": continue

                parts_dict = {}
                parts_dict["file"] = file
                # location is the second part of the processed file name
                parts_dict["location"] = parts[1]
                # timestamp is the 3rd (year), 4th (month), 5th (day) parts of the file name
                parts_dict["timestamp"] = "_".join(parts[2:5])
                # session_id is the first three characters of the 6th part of the file name
                # lead is the 7th part of the file name before the extension or defaults to 'ecg2' 
                if len(parts) > 6:
                    parts_dict["session_id"] = parts[5]
                    parts_dict["lead"] = parts[6].split('.')[0]
                else:
                    parts_dict["session_id"] = parts[5].split('.')[0] 
                    parts_dict["lead"] = "ecg2"

                all_parts.append(parts_dict)

            except Exception as e:
                print(f"Error processing: {file}, error: {e}")
                continue

        # Append the dictionary to the dataframe such that each row is a file
        new_df = pd.DataFrame(all_parts)
        ucsd_ecg_df = pd.concat([ucsd_ecg_df, new_df], ignore_index=True)
        return ucsd_ecg_df
    

class ScanArchiveDataset:
    def __init__(self, dataset_location):
        """
        Initialize the ScanArchiveDataset class with the dataset_location.
        
        Args:
            dataset_location (str): The overall dataset folder which contains subfolders (locations) with 
            more subfolders (exam) that each have ScanArchive files them.
        """
        self.dataset_location = dataset_location


    def get_dataset_dataframe(self) -> pd.DataFrame:
        """
        Get the dataframe with the all the ScanArchive files in the folder.
        Create a pandas DataFrame with the dataset information for each file in the dataset_location.
        
        Returns:
            ucsd_ecg_df (pd.DataFrame): A pandas DataFrame with the dataset information
        """
        # Get subdirectories from the ECG dataset_location (main folder)
        subdirs = [x[0] for x in os.walk(self.dataset_location)][1:] # skip the main folder

        ucsd_ecg_df = pd.DataFrame(columns=["subdir", "file", "exam", "series", "location", "timestamp", "session_id"])

        # Create a dictionary of subdirectories (locations) and files
        files = {}
        all_parts = []
        for subdir in subdirs:
            # Get the files in the subdirectory
            files[subdir] = [x[2] for x in os.walk(subdir)][0]

            for file in files[subdir]:
                try:
                    # Split the file and subdir name into parts as separated by "_" or "/"
                    parts = file.split("_")
                    subdir_parts = subdir.split("/")

                    # Create a dictionary of the parts of the file name
                    parts_dict = {}
                    parts_dict["subdir"] = subdir
                    parts_dict["file"] = file

                    # Extract location, exam, timestamp, and session_id from the subdir and file name
                    if 'Exam' in subdir_parts[-1]:
                        parts_dict["location"] = subdir_parts[-2]  # hillcrestmr1
                        parts_dict["exam"] = subdir_parts[-1].split("-")[0]  # Exam43171
                        parts_dict["series"] = subdir_parts[-1].split("-")[-1]  # Series4
                    elif 'archive' in subdir_parts[-3]:
                        parts_dict["location"] = subdir_parts[-4]  # mractri
                        parts_dict["exam"] = subdir_parts[-2]  # Exam75
                        parts_dict["series"] = subdir_parts[-1]  # Series1
                    else:
                        parts_dict["location"] = subdir_parts[-3]  # thorntonmr
                        parts_dict["exam"] = subdir_parts[-2]  # Exam17884
                        parts_dict["series"] = subdir_parts[-1]  # Series1

                    parts_dict["timestamp"] = parts[2] + "_" + parts[3][:6]  # 20231213_170441
                    parts_dict["session_id"] = parts[3][6:9]  # 172, ignore the last 3 digits (.h5)

                    all_parts.append(parts_dict)

                except Exception as e: 
                    print(f"Error processing: {file}, error: {e}")
                    continue

        # Append the dictionary to the dataframe such that each row is a file
        new_df = pd.DataFrame(all_parts)
        ucsd_ecg_df = pd.concat([ucsd_ecg_df, new_df], ignore_index=True)
        return ucsd_ecg_df