from ecg_data_processor import GEwaveforms
from ecg_data_processor import UCSDwaveforms

"""
This script processes the raw waveform data from the GE and UCSD 
datasets as a .npy file with the same name as the raw data file.
and saves the processed data in the ./data/new_processed directory. 
"""

# TRIG only for GE dataset saved as GE_{file_name}_trigger_data.npy
GE = False
TRIG = False 

UCSD = True
ECG2 = True # saved as data_{location}_{timestamp}_{session_id}.npy
ECG3 = False # saved as data_{location}_{timestamp}_{session_id}_ecg3.npy

if(GE):
    raw_directory = "./data/GE_3T_waveforms/"
    processed_directory = "./data/new_processed/"
    waveforms = GEwaveforms(raw_directory,processed_directory)
    
elif(UCSD):
    raw_directory = "./data/ucsd"
    processed_directory = "./data/new_processed/"
    waveforms = UCSDwaveforms(raw_directory,processed_directory)

if(ECG2): waveforms.process_ecg_raw_files(ECG = 'ECG2')
elif(ECG3): waveforms.process_ecg_raw_files(ECG = 'ECG3')
elif(GE and TRIG): waveforms.process_trigger_files()
