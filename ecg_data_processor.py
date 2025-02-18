import os
import glob
import json
import numpy as np
import pandas as pd
from ucsd_ecg_dataset import ECGDataset

class GEwaveforms:
    def __init__(self, raw_directory = './data/GE_3T_waveforms/', processed_directory = './data/new_processed/') -> None:
        """
        Initialize the GEwaveforms class to read GE provided data.

        Args:
            raw_directory (str): The directory where the raw ecg files are located.
            processed_directory (str): The directory where the processed ecg files will be saved.
        """
        self.directory = raw_directory
        self.json_files = glob.glob(os.path.join(raw_directory, '*.json'))
        self.json_files = [os.path.basename(file) for file in self.json_files]
        self.out_directory = processed_directory

    def get_trigger_detections(self, file_name):
        """
        Get the trigger detection of a testcase from the processed-directory if they exist.

        Args:
            file_name (str): The name of the patient to get the trigger data from.

        Returns:
            data (numpy.ndarray): The trigger data of the patient.
        """
        file_path = os.path.join(self.out_directory, f'GE_{file_name}_trigger_data.npy')
        if os.path.exists(file_path): data = np.load(file_path)
        else: data = None
        return data
    
    def process_trigger_files(self):
        """
        Process the trigger files and save them as numpy arrays if they don't
        already exist in the processed-directory.

        Returns: 
            different_values (dict): Nested dictionary containing trigger time:voltage key:values 
            pairs that are not the same as the first trigger time:voltage key:value pair.
        """
        different_values = {}
        for file in self.json_files:
            with open(self.directory+file) as f: data = json.load(f)
            # print(data.keys()) 
            # print('trigger:', type(data['trigger']))
            # print('trigger length:', len(data['trigger'].keys())) 
            # print('trigger first key:', list(data['trigger'].keys())[0]) 
            # print('trigger first key value:', data['trigger']['0'])

            different_values[file] = {}
            if all(value == data['trigger']['0'] for value in data['trigger'].values()):
                print("All trigger values (mV) are the same.")
            else:
                print("All trigger values (mV) are not the same.")
                for key, value in data['trigger'].items():
                    if value != data['trigger']['0']:
                        different_values[file][key] = value
                print('different keys length:', len(different_values[file].keys()))
            
            trigger = data['trigger']
            # Turn keys (time) to integeres and sort the keys only if they are in different_values[file]
            # meaning they have a different value (voltage) from the first time:voltage key:value pair
            sorted_keys = sorted(int(key) for key in trigger.keys() if key in different_values[file])
            trigger_signal = np.array(sorted_keys)

            file_name = os.path.splitext(file)[0]
            file_path = os.path.join(self.out_directory, f'GE_{file_name}_trigger_data.npy')
            if not os.path.exists(file_path):
                np.save(file_path, trigger_signal)
                print(f'GE_{file_name}_trigger_data.npy saved successfully')
            else: print(f'GE_{file_name}_trigger_data.npy already exists')

        return different_values
    
    def process_ecg2_raw_files(self):
        """
        Process the ecg2_raw files and save them as numpy arrays.
        """
        for file in self.json_files:
            with open(self.directory+file) as f: data = json.load(f)
            # print(data.keys()) 
            # print('ecg2_raw:', type(data['ecg2_raw']))
            # print('ecg2_raw length:', len(data['ecg2_raw'].keys())) 
            # print('ecg2_raw first key:', list(data['ecg2_raw'].keys())[0]) 
            # print('ecg2_raw first key value:', data['ecg2_raw']['0'])

            ecg2_raw = data['ecg2_raw']
            # Turn keys (time) to integeres and sort the keys to list its corresponding values (voltage)
            sorted_voltage_values = [ecg2_raw[str(i)] for i in sorted(map(int, ecg2_raw.keys()))]
            ecg2_raw_signal = np.array(sorted_voltage_values)

            file_name = os.path.splitext(file)[0]
            file_path = os.path.join(self.out_directory, f'GE_{file_name}_ECG2_data.npy')
            if not os.path.exists(file_path):
                np.save(file_path, ecg2_raw_signal)
                print(f'GE_{file_name}_ECG2_data.npy saved successfully')
            else: print(f'GE_{file_name}_ECG2_data.npy already exists')
    

class UCSDwaveforms:
    def __init__(self, raw_directory = './data/ucsd/', processed_directory = './data/new_processed/') -> None:
        """
        Initialize the UCSDwaveforms class to read new UCSD data.
        This is similar to ECGDataset class from the ./ucsd_ecg_dataset.py file.

        Args:
            raw_directory (str): The directory where the raw ecg files are located.
            processed_directory (str): The directory where the processed ecg files will be saved.
        """
        self.directory = raw_directory
        self.ucsd_ds = ECGDataset(raw_directory)
        self.out_directory = processed_directory

    def process_ecg_raw_files(self, ECG = 'ECG2'):
        """
        Process the raw ecg files and save them as numpy arrays if they don't
        already exist in the new_processed or the old_preprocessed directory.

        Args:
            ECG (str): The ECG channel to process. Either 'ECG2' or 'ECG3'.

        Creates three csv files in the processed-directory:
            - newly_processed_files.csv: Contains the names of the files that were processed.
            - errored_processing_files.csv: Contains the names of the files that were not processed.
            - previously_processed_files.csv: Contains the names of the files that were already processed.
        """
        # Initialize lists to store file information
        saved_files = []
        errored_files = []
        processed_files = []

        # Get the dataset
        ucsd_ds = self.ucsd_ds

        # Get the dataframe of files and their locations
        df = ucsd_ds.get_dataset_dataframe()
        print(df.head())

        for index, row in df.iterrows():
            location = row["location"]
            timestamp = row["timestamp"]
            session_id = row["session_id"]
                
            if ECG == 'ECG3': file_name = f"data_{location}_{timestamp}_{session_id}_ecg3.npy"
            else: file_name = f"data_{location}_{timestamp}_{session_id}.npy"
                
            if file_name not in os.listdir("/home/aminm/Documents/ECG_Gating/ECG_Gating_MasterFolder/data/old_preprocessed/") and file_name not in os.listdir(self.out_directory):
                try:
                    raw_ecg_data = ucsd_ds.get_ecg_data(location,timestamp,session_id)
                    if ECG == 'ECG2': np.save(f"{self.out_directory + file_name}",raw_ecg_data[0,:]) # Save the first channel (ECG2)
                    if ECG == 'ECG3': np.save(f"{self.out_directory + file_name}",raw_ecg_data[1,:]) # Save the second channel (ECG3)
                    saved_files.append(file_name)  # Add to saved files list
                except:
                    errored_files.append(file_name)  # Add to errored files list
                    continue
            else: processed_files.append(file_name)  # Add to processed files list

        # Convert lists to dataframes and save as CSV
        pd.DataFrame(saved_files, columns=['file_name']).to_csv(f'{self.out_directory}newly_processed_files.csv', index=False)
        pd.DataFrame(errored_files, columns=['file_name']).to_csv(f'{self.out_directory}errored_processing_files.csv', index=False)
        pd.DataFrame(processed_files, columns=['file_name']).to_csv(f'{self.out_directory}previously_processed_files.csv', index=False)