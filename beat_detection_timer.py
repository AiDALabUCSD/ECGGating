import time
import numpy as np
import pandas as pd
import biosppy.signals as bsp

import tensorflow as tf

from ecg_dataset_manager import DatasetManager
from deep_qrs_detector import DeepQRSDetector
from deep_qrs_predictor import DeepQRSPredictor
from beat_classification import ECGBeatClassifier

run_parameters = {}

run_parameters['EVALUATE_HAMILTON'] = True
run_parameters['EVALUATE_CNN_DETECTOR'] = True
run_parameters['EVALUATE_CNN_PREDICTOR'] = True

run_parameters["DETECTION_CONFIDENCE_THRESHOLD"] = 30.0, # <-- 30.0 datapoints = 120 milliseconds
run_parameters["PEAK_COOLDOWN_THRESHOLD"] = 50.0, # <-- 50.0 datapoints = 200 milliseconds

# Load preprocessed UCSD, MIT, and NEW datasets without JA dataset (middle)
UCSD_PROCESSED_DATASET_LOCATION = './data/old_preprocessed/'
MITDB_DATASET_LOCATION = './data/mitdb/raw/'
NEW_PROCESSED_DATASET_LOCATION = './data/new_processed/'

dm = DatasetManager(UCSD_PROCESSED_DATASET_LOCATION,'',MITDB_DATASET_LOCATION, NEW_PROCESSED_DATASET_LOCATION)

# Set GPU memory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# # Use MirroredStrategy for multi-GPU training
# strategy = tf.distribute.MirroredStrategy()
# with strategy.scope():
#     cnn_predictor = DeepQRSPredictor('./models/cnn_predictor.h5',run_parameters['DETECTION_CONFIDENCE_THRESHOLD'],run_parameters['PEAK_COOLDOWN_THRESHOLD'])
#     cnn_detector = DeepQRSDetector('./models/cnn_detector.h5')

cnn_predictor = DeepQRSPredictor('./models/cnn_predictor.h5',run_parameters['DETECTION_CONFIDENCE_THRESHOLD'],run_parameters['PEAK_COOLDOWN_THRESHOLD'])
cnn_detector = DeepQRSDetector('./models/cnn_detector.h5')

# Load the dataframe
df = pd.read_csv('./data/detection_timing.csv', dtype=str)

# Prepare a list to hold all the records
records = []

for index, row in df.iterrows():            
    database = 'UCSD'
    testcase = row['testcase']
    print(testcase)

    # Load signal and annotations
    unfiltered_ecg, annotations = dm.LoadSignalAndAnnotations(testcase, database)

    # Initialize record dictionary
    record = {'testcase': testcase}

    # Hamilton Detector
    if run_parameters['EVALUATE_HAMILTON']:
        start_time = time.time()
        hamilton_peaks = bsp.ecg.hamilton_segmenter(unfiltered_ecg, sampling_rate=250)
        record['Hamilton_Time'] = time.time() - start_time
        hamilton_array = np.array(hamilton_peaks).squeeze().astype(np.int32)
        classifier = ECGBeatClassifier(unfiltered_ecg, hamilton_array, p=0.1, p_short=0.75, p_long=1.20)
        _, total_beat_count = classifier.beat_breakdown()
        record['Hamilton_Beats'] = total_beat_count

    # CNN Detector
    if run_parameters['EVALUATE_CNN_DETECTOR']:
        start_time = time.time()
        qrs_peaks_detector, _ = cnn_detector.detect_peaks(unfiltered_ecg)
        record['Detector_Time'] = time.time() - start_time
        cnn_detector_peaks_array = np.array(qrs_peaks_detector).astype(np.int32)
        classifier = ECGBeatClassifier(unfiltered_ecg, cnn_detector_peaks_array, p=0.1, p_short=0.75, p_long=1.20)
        _, total_beat_count = classifier.beat_breakdown()
        record['Detector_Beats'] = total_beat_count

    # CNN Predictor
    if run_parameters['EVALUATE_CNN_PREDICTOR']:
        start_time = time.time()
        qrs_peaks_predictor, _, _ = cnn_predictor.detect_peaks(unfiltered_ecg, return_confidence_intervals=True)
        record['Predictor_Time'] = time.time() - start_time
        cnn_predictor_peaks_array = np.array(qrs_peaks_predictor).astype(np.int32)
        classifier = ECGBeatClassifier(unfiltered_ecg, cnn_predictor_peaks_array, p=0.1, p_short=0.75, p_long=1.20)
        _, total_beat_count = classifier.beat_breakdown()
        record['Predictor_Beats'] = total_beat_count

    # Append the record to the list
    records.append(record)

# Convert records to a DataFrame and save to CSV
results_df = pd.DataFrame(records)
results_df.to_csv('./results/detection_time_performance2.csv', index=False)