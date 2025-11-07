# Deep Learning R-Wave Detection for Electrocardiographic Gating in Cardiac MRI

This repository contains code for developing an R-wave detection convolutional neural network. 

Published article: https://doi.org/10.1148/ryct.250104

## File Descriptions

- `README.md`: This file, providing an overview of the project and its components.
- `requirements.yml`: Conda environment configuration file listing all dependencies.
- `ecg_data_processor.py`: Processes raw ECG data from different sources and saves the processed data for further analysis.
- `ecg_dataset_manager.py`: Manages various ECG datasets, including loading signals and annotations, and preparing data for model training.
- `ucsd_ecg_dataset.py`: Manages the UCSD ECG dataset, including loading and processing trigger data files.
- `process_ecg_waveforms.py`: Format raw ECG waveform data and saves the processed data using ecg_data_processor.py.
- `process_ecg_annotations.py`: Prepares the training dataset from UCSD data and Label Studio annotations.
- `train_detector.py`: Script to train the deep learning model for QRS detection.
- `deep_qrs_detector.py`: Defines a deep learning model for detecting QRS complexes in ECG signals and includes methods for training and evaluating the model.
- `qrs_detection_timer.py`: Measures the execution time of different QRS detection methods.
- `labelstudio_tools.py`: Tools for handling Label Studio annotations and converting them to dataframes.
- `beat_classification_functions.py`: Functions for classifying ECG beats based on detected R-peaks and RR-intervals.
- `beat_classifier.py`: Script to classify ECG beats algorithmically and save the results.


