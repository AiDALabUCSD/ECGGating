import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from ucsd_ecg_dataset import ECGDataset
from ecg_dataset_manager import DatasetManager
from beat_classification_functions import ECGBeatClassifier


UCSD_PROCESSED_DATASET_LOCATION = './data/old_preprocessed/'
NEW_PROCESSED_DATASET_LOCATION = './data/new_processed/'
dm = DatasetManager(UCSD_PROCESSED_DATASET_LOCATION, '', '', NEW_PROCESSED_DATASET_LOCATION)


def trim_detections(detections_array, ecg_signal):
    """Context window for both of the CNNs is 2.048 seconds. CNN detector may detect in the last 64 ms of this window.
    CNN predictor context window overlaps with its prediction window by 256 ms (64 samples). Therefore, this removes 
    the first 2.048 - 0.256 = 1.792 seconds of the signal and detections (448 samples). Additionally, it removes any 
    detections that are longer than the length of the signal, corresponding to added 4 seconds (1000 samples).
    
    Args:
        detections_array (np.array): Any of the detection_arrays from the dataset
        ecg_signal (int): The original length ECG_signal 
    
    Returns:
        detections_array (list): List with the first 448 samples and any detections longer than signal length removed.
    """
    if detections_array is not None: detections_array = [x - 448 for x in detections_array if 448 < x < len(ecg_signal) + 448]
    return detections_array


def classify_cnn(cnn_detector, row, database, Start, fs):
    """Classify the beats from the ECG signal using the CNN detector and classifier.
    
    Args:
        cnn_detector (DeepQRSDetector): The CNN detector model
        row (pd.Series): The row of the DataFrame containing the ECG data
        database (str): The database to load the signal and annotations from
        Start (int): The start time of the ECG signal
        fs (int): The sampling frequency of the ECG signal
        
    Returns:
        cnn_record (dict): The classification results of the CNN detector and classifier
        annotations_not_found (bool): True if no annotations are found for the testcase
    """
    # Load signal and cnn_record if no annotations are found for the testcase
    downsampled_ecg, _, annotations_not_found = dm.LoadSignalAndAnnotations(testcase=row['ecg_scan'], database=database, ECG=row['ecg_lead'], iStart=Start*fs)

    # Initialize cnn_record dictionary with all keys set to 0
    cnn_record = {
        'ecg_scan': row['ecg_scan'],
        'ecg_lead': row['ecg_lead'],
        'trigger_file': row['trigger_file'],
        'cnn_beat_count': 0,
        'cnn_mean_RR': 0,
        'cnn_std_RR': 0,
        'cnn_Sinus-Sinus': 0,
        'cnn_Interrupted-Sinus': 0,
        'cnn_maybe Interrupted-Sinus': 0,
        'cnn_PAC': 0,
        'cnn_maybe PAC': 0,
        'cnn_PVC': 0,
        'cnn_maybe PVC': 0,
        'cnn_Post-Pause Sinus': 0,
        'cnn_maybe Post-Pause Sinus': 0,
        'cnn_Bigeminy Sinus': 0,
        'cnn_maybe Bigeminy Sinus': 0,
        'cnn_Undetected R-peaks': 0,
        'cnn_Misfire Detections': 0,
        'cnn_Other': 0}

    # Detect QRS peaks using the CNN detector and classify the beats
    qrs_peaks_detector, _ = cnn_detector.detect_peaks(downsampled_ecg)
    cnn_detector_peaks_array = np.array(trim_detections(qrs_peaks_detector,downsampled_ecg[448:])).astype(np.int32)
    cnn_classifier = ECGBeatClassifier(downsampled_ecg, cnn_detector_peaks_array, p=0.1, p_short=0.75, p_long=1.20)
    cnn_beat_percentages, cnn_total_beat_count, cnn_meanRR, cnn_stdRR = cnn_classifier.beat_breakdown()

    # Add the word 'cnn_' to the beginning of each key in cnn_beat_percentages and update the cnn_record with the values
    cnn_beat_percentages = {f'cnn_{key}': value for key, value in cnn_beat_percentages.items()}
    cnn_record['cnn_beat_count'] = cnn_total_beat_count
    cnn_record['cnn_mean_RR'] = cnn_meanRR
    cnn_record['cnn_std_RR'] = cnn_stdRR
    cnn_record.update(cnn_beat_percentages)
    return cnn_record, annotations_not_found


def classify_vcg(row, database, Start, fs):
    """Classify the beats from the ECG signal using the VCG triggers.
    
    Args:
        row (pd.Series): The row of the DataFrame containing the ECG data
        database (str): The database to load the signal and annotations from
        Start (int): The start time of the ECG signal
        fs (int): The sampling frequency of the ECG signal
        
    Returns:
        vcg_record (dict): The classification results of the VCG triggers
        annotations_not_found (bool): True if no annotations are found for the testcase
"""
    # Load signal and cnn_record if no annotations are found for the testcase
    downsampled_ecg, _, annotations_not_found = dm.LoadSignalAndAnnotations(testcase=row['ecg_scan'], database=database, ECG=row['ecg_lead'], iStart=Start*fs)

    # Initialize vcg_record dictionary with all keys set to 0
    vcg_record = {
        'ecg_scan': row['ecg_scan'],
        'trigger_file': row['trigger_file'],
        'vcg_beat_count': 0,
        'vcg_mean_RR': 0,
        'vcg_std_RR': 0,
        'vcg_Sinus-Sinus': 0,
        'vcg_Interrupted-Sinus': 0,
        'vcg_maybe Interrupted-Sinus': 0,
        'vcg_PAC': 0,
        'vcg_maybe PAC': 0,
        'vcg_PVC': 0,
        'vcg_maybe PVC': 0,
        'vcg_Post-Pause Sinus': 0,
        'vcg_maybe Post-Pause Sinus': 0,
        'vcg_Bigeminy Sinus': 0,
        'vcg_maybe Bigeminy Sinus': 0,
        'vcg_Undetected R-peaks': 0,
        'vcg_Misfire Detections': 0,
        'vcg_Other': 0}
    
    # Load the QRS detection (vcg triggers: ECG3Triggers or ECG2Triggers) from Fourier NAS or local directory
    vcg_ds = ECGDataset('/home/aminm/Documents/Fourier/repository/ekgfree/ECGRespData')
    try: scaled_vcg = vcg_ds.get_trigger_detections(row['location'],row['ecg_timestamp'],row['ecg_session_id'],row['ecg_lead'],Start*fs)
    except FileNotFoundError:
        vcg_ds = ECGDataset('./data/ucsd')
        scaled_vcg = vcg_ds.get_trigger_detections(row['location'],row['ecg_timestamp'],row['ecg_session_id'],row['ecg_lead'],Start*fs)

    # Classify the beats from the VCG triggers
    scaled_vcg = np.array(trim_detections(scaled_vcg, downsampled_ecg[448:])).astype(np.int32)
    vcg_classifier = ECGBeatClassifier(downsampled_ecg, scaled_vcg, p=0.1, p_short=0.75, p_long=1.20)
    vcg_beat_percentages, vcg_total_beat_count, vcg_meanRR, vcg_stdRR = vcg_classifier.beat_breakdown()

    # Add the word 'vcg_' to the beginning of each key in cnn_beat_percentages and update the vcg_record with the values
    vcg_beat_percentages = {f'vcg_{key}': value for key, value in vcg_beat_percentages.items()}
    vcg_record['vcg_beat_count'] = vcg_total_beat_count
    vcg_record['vcg_mean_RR'] = vcg_meanRR
    vcg_record['vcg_std_RR'] = vcg_stdRR
    vcg_record.update(vcg_beat_percentages)
    return vcg_record, annotations_not_found


def classify_annotation(row, database, Start, fs):
    """Classify the beats from the ECG signal using the annotations.
    
    Args:
        row (pd.Series): The row of the DataFrame containing the ECG data
        database (str): The database to load the signal and annotations from
        Start (int): The start time of the ECG signal
        fs (int): The sampling frequency of the ECG signal
        
    Returns:
        anno_record (dict): The classification results of the annotations
        annotations_not_found (bool): True if no annotations are found for the testcase
    """
    # Load signal and anno_record if no annotations are found for the testcase
    downsampled_ecg, annotations, annotations_not_found = dm.LoadSignalAndAnnotations(testcase=row['scan'], database=database, ECG='ECG2', iStart=Start*fs)
    annotations_array = np.array(annotations).astype(np.int32)

    # Initialize anno_record dictionary with all keys set to 0
    anno_record = {
        'pid':row['pid'],
        'scan': row['scan'], 
        'dl-type': row['dl-type'],
        'mri_datetime': row['mri_datetime'],
        'ecg_datetime': row['ecg_datetime'],
        'time_difference': row['time_difference'],
        'anno_beat_count': 0,
        'anno_mean_RR': 0,
        'anno_std_RR': 0,
        'anno_Sinus-Sinus': 0,
        'anno_Interrupted-Sinus': 0,
        'anno_maybe Interrupted-Sinus': 0,
        'anno_PAC': 0,
        'anno_maybe PAC': 0,
        'anno_PVC': 0,
        'anno_maybe PVC': 0,
        'anno_Post-Pause Sinus': 0,
        'anno_maybe Post-Pause Sinus': 0,
        'anno_Bigeminy Sinus': 0,
        'anno_maybe Bigeminy Sinus': 0,
        'anno_Undetected R-peaks': 0,
        'anno_Misfire Detections': 0,
        'anno_Other': 0}

    # Classify the annotated beats
    anno_classifier = ECGBeatClassifier(downsampled_ecg, annotations_array, p=0.1, p_short=0.75, p_long=1.20)
    anno_beat_percentages, anno_total_beat_count, anno_meanRR, anno_stdRR = anno_classifier.beat_breakdown()

    # Add the word 'anno_' to the beginning of each key in anno_beat_percentages and update the anno_record with the values
    anno_beat_percentages = {f'anno_{key}': value for key, value in anno_beat_percentages.items()}
    anno_record['anno_beat_count'] = anno_total_beat_count
    anno_record['anno_mean_RR'] = anno_meanRR
    anno_record['anno_std_RR'] = anno_stdRR
    anno_record.update(anno_beat_percentages)
    return anno_record, annotations_not_found


# Main function to classify ECG beats
def main(classification_type, start_time, csv_path) -> None:
    """Call the appropriate classification function based on the classification type, then save the results to a CSV file.
    
    Args:
        classification_type (str): The type of classification to perform [cnn, vcg, or annotation]
        csv_path (str): The path to the CSV file containing the ECG data
        start_time (int): The time in seconds to start the ECG signal
    """
    # Load the CSV file and replace NaN values with 0
    df = pd.read_csv(csv_path, dtype=str)
    df = df.where(pd.notnull(df), 0)
    
    # Prepare a list to hold all the classifications
    classifications = []
    unfound_annotations = []
    
    # Define the sampling frequency and the start of the ECG signal
    fs = 250
    Start = int(start_time)

    # For CNN: import the CNN detector GPU model and keep each unique lead of each ECG scan
    if classification_type == 'cnn':
        from deep_qrs_detector import DeepQRSDetector
        cnn_detector = DeepQRSDetector('./models/cnn_detector.h5')

        df_unique = df.drop_duplicates(subset=['ecg_scan', 'ecg_lead'])

        # Iterate through the rows of the DataFrame and classify the beats using CNN
        for _, row in tqdm(df_unique.iterrows(), total=df_unique['ecg_scan'].count(), desc="Processing ECG scans (CNN)"):
            record, annotations_not_found = classify_cnn(cnn_detector, row, 'New_Dataset' if row['pid'] == 0 else 'UCSD', Start, fs)
            if record: classifications.append(record)
            if annotations_not_found: unfound_annotations.append({'ecg_scan': row['ecg_scan']})


    # For VCG: keep only unique trigger files, since we want to classify each trigger file
    elif classification_type == 'vcg':
        df_unique = df.drop_duplicates(subset=['trigger_file'])
        df_unique = df_unique[df_unique['trigger_file'] != 0]

        # Iterate through the rows of the DataFrame and classify the beats using VCG
        for _, row in tqdm(df_unique.iterrows(), total=df_unique['ecg_scan'].count(), desc="Processing ECG scans (VCG)"):
            record, annotations_not_found = classify_vcg(row, 'New_Dataset' if row['pid'] == 0 else 'UCSD', Start, fs)
            if record: classifications.append(record)
            if annotations_not_found: unfound_annotations.append({'ecg_scan': row['ecg_scan']})

    # For Annotations: iterate through the rows of the DataFrame and classify the beats using annotations
    elif classification_type == 'annotation':
        for _, row in tqdm(df.iterrows(), total=df['scan'].count(), desc="Processing ECG scans (Annotation)"):
            record, annotations_not_found = classify_annotation(row, 'UCSD', Start, fs)
            if record: classifications.append(record)
            if annotations_not_found: unfound_annotations.append({'scan': row['scan']})

    else: raise ValueError("Invalid classification type. Choose from 'cnn', 'vcg', or 'annotation'.")

    # Convert records to a DataFrame and save to CSV
    results_df = pd.DataFrame(classifications)
    results_df.to_csv(f'./results/{classification_type}_ecg_classifications_{Start}start.csv', index=False)

    # Save the unfound annotations to a CSV
    if unfound_annotations:
        unfound_annotations_df = pd.DataFrame(unfound_annotations)
        unfound_annotations_df.to_csv(f'./results/{classification_type}_unfound_annotations_{Start}start.csv', index=False)


# Run the main function if the script is executed
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Classify ECG beats.')
    parser.add_argument('type', choices=['cnn', 'vcg', 'annotation'], help='Type of classification to perform')
    parser.add_argument('start', type=int, help='Time in seconds to start the ECG signal')
    parser.add_argument('csv_path', help='Path to the CSV file containing ECG data')

    args = parser.parse_args()
    main(args.type, args.start, args.csv_path)