from new_ecg_dataset import GEwaveforms
from new_ecg_dataset import UCSDwaveforms

GE = False
UCSD = True

if(GE):
    raw_directory = "./data/GE_3T_waveforms/"
    processed_directory = "./data/new_processed/"
    waveforms = GEwaveforms(raw_directory,processed_directory)
elif(UCSD):
    raw_directory = "./data/ucsd/"
    processed_directory = "./data/new_processed/"
    waveforms = UCSDwaveforms(raw_directory,processed_directory)

ECG2 = False
ECG3 = True
TRIG = False

if(ECG2):
    waveforms.process_ecg_raw_files(ECG = 'ECG2')
elif(ECG3):
    waveforms.process_ecg_raw_files(ECG = 'ECG3')
elif(TRIG):
    waveforms.process_trigger_files()
