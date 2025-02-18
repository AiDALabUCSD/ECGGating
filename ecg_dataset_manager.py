import numpy as np
import pandas as pd
import wfdb
from scipy.signal import resample

class DatasetManager:
    """
    Initialize the class to manage ECG datasets for deep learning model training and testing.

    Args:
        UCSD_PROCESSED_DATASET_LOCATION (str): The location of the preprocessed UCSD dataset.
        JA_PROCESSED_DATASET_LOCATION (str): The location of the preprocessed JA dataset.
        MITDB_DATASET_LOCATION (str): The location of the raw MITDB dataset.
        NEW_PROCESSED_DATASET_LOCATION (str): The location of the new preprocessed dataset
    """
    def __init__(self,UCSD_PROCESSED_DATASET_LOCATION = './data/old_preprocessed/', 
                 JA_PROCESSED_DATASET_LOCATION  = '/content/drive/MyDrive/ECG Data/JA/runtime_data/', # we dont use this
                 MITDB_DATASET_LOCATION = './data/mitdb/raw/', 
                 NEW_PROCESSED_DATASET_LOCATION = './data/new_processed/') -> None:
        self.UCSD_PROCESSED_DATASET_LOCATION = UCSD_PROCESSED_DATASET_LOCATION
        self.JA_PROCESSED_DATASET_LOCATION  = JA_PROCESSED_DATASET_LOCATION
        self.MITDB_DATASET_LOCATION = MITDB_DATASET_LOCATION
        self.NEW_PROCESSED_DATASET_LOCATION = NEW_PROCESSED_DATASET_LOCATION


    def LoadSignalAndAnnotations(self, testcase, database="UCSD", ECG ='ECG2', iStart=None, iStop=None, TARGET_FREQUENCY=250.0):
        """
        Load and downsample the signal and annotations for a given testcase from the specified database.
        Downsample the signal and annotations to the specified TARGET_FREQUENCY, defailt is 250 Hz.
        Crop the signal to the specified start and stop indices, default is the entire signal.
        
        Args:
            testcase (str): The name of the testcase to load.
            database (str): The name of the database to load the testcase from:(UCSD, JA, MITDB, New_Dataset).
            TARGET_FREQUENCY (float): The frequency to downsample the signal and annotations to. Default is 250 Hz.
            iStart (float): The starting index of the signal to load.
            iStop (float): The stopping index of the signal to load.
            ECG (str): The ECG signal to load. Either 'ECG2' or 'ECG3'.
        
        Returns:
            raw_signal (np.ndarray): The raw ECG signal.
            annotations (np.ndarray): The annotations for the ECG signal.
            annotations_not_found (bool): A flag to indicate if the annotation is missing.
        """
        annotations_not_found = False

        if(database == 'UCSD'):
            # Downsample signal and annotation from 1000 Hz to 250 Hz
            # ECG3 could be in either dataset, check old UCSD dataset first then new if not found
            try: p_signal = np.load(f'{self.UCSD_PROCESSED_DATASET_LOCATION}data_{testcase}.npy')[::int(1000.0/TARGET_FREQUENCY)]
            except FileNotFoundError: p_signal = np.load(f'{self.NEW_PROCESSED_DATASET_LOCATION}data_{testcase}.npy')[::int(1000.0/TARGET_FREQUENCY)]
            # Annotations could be missing, if missing set annotations_not_found flag to True
            try:
                annotations = np.load(f'{self.UCSD_PROCESSED_DATASET_LOCATION}data_anno_{testcase}.npy')
                annotations = [int(a * (TARGET_FREQUENCY / 1000.0)) for a in annotations]
            except FileNotFoundError:
                annotations_not_found = True
                annotations = None

        elif(database == "JA"):
            #testcase = '11_sitting_chest_strap_V2_V1_Ermakov.npy'
            p_signal = np.load(f'{self.JA_PROCESSED_DATASET_LOCATION}data_{testcase}')
            annotations = np.load(f'{self.JA_PROCESSED_DATASET_LOCATION}data_anno_{testcase}')
        
        elif(database == "MITDB"):
            # Load the MITDB dataset and annotations
            record = wfdb.rdrecord(f'{self.MITDB_DATASET_LOCATION}{testcase}', channels=[0, 1])
            annotations = wfdb.rdann(f'{self.MITDB_DATASET_LOCATION}{testcase}', 'atr')

            # Get the first channel of the signal and the sample annotations
            p_signal = record.p_signal[:,0]
            annotations = annotations.sample

            # Downsample the signal and annotations from 360 Hz to 250 Hz
            p_signal = resample(p_signal, int(len(p_signal)*TARGET_FREQUENCY/360.0))
            annotations = [round((a * TARGET_FREQUENCY)/360.0) for a in annotations]
            
        elif(database == "New_Dataset"):
            # Downsample signal and annotation from 1000 Hz to 250 Hz
            if 'Case' in testcase: p_signal = np.load(f'{self.NEW_PROCESSED_DATASET_LOCATION}GE_{testcase}_ECG2_data.npy')[::int(1000.0/TARGET_FREQUENCY)]
            else:
                # ECG3 should only be in the new dataset
                if ECG == 'ECG3': p_signal = np.load(f'{self.NEW_PROCESSED_DATASET_LOCATION}data_{testcase}_ecg3.npy')[::int(1000.0/TARGET_FREQUENCY)]
                elif ECG == 'ECG2':
                    # ECG2 could be in either dataset, check new dataset first then UCSD if not found
                    try: p_signal = np.load(f'{self.NEW_PROCESSED_DATASET_LOCATION}data_{testcase}.npy')[::int(1000.0/TARGET_FREQUENCY)]
                    except FileNotFoundError: p_signal = np.load(f'{self.UCSD_PROCESSED_DATASET_LOCATION}data_{testcase}.npy')[::int(1000.0/TARGET_FREQUENCY)]
                        # Annotations could be missing, if missing set annotations_not_found flag to True
            try:
                annotations = np.load(f'{self.UCSD_PROCESSED_DATASET_LOCATION}data_anno_{testcase}.npy')
                annotations = [int(a * (TARGET_FREQUENCY / 1000.0)) for a in annotations]
            except FileNotFoundError:
                try:
                    annotations = np.load(f'{self.NEW_PROCESSED_DATASET_LOCATION}data_anno_{testcase}.npy')
                    annotations = [int(a * (TARGET_FREQUENCY / 1000.0)) for a in annotations]
                except FileNotFoundError:
                    annotations_not_found = True
                    annotations = None

        else: raise Exception("No such dataset.")

        # Set the start and stop indices of the signal and annotations to load
        if(iStart == None): iStart = 0
        else: iStart = max(iStart, 0)
        if(iStop == None): iStop = len(p_signal)
        else: iStop = min(iStop, len(p_signal))

        # Crop the signal and annotations to the specified start and stop indices 
        p_signal = p_signal[iStart:iStop]
        raw_signal = p_signal.squeeze()

        # Adjust the annotations to the new start and stop indices
        if annotations is not None: annotations = [a - iStart for a in annotations if (a >= iStart and a <= iStop)]
        return raw_signal, annotations, annotations_not_found


    def LoadTestcaseData(self, testcase, model_class, database="JA", iStart=None, iStop=None, step_interval=1):
        """
        Load the ECG signal and annotations for a given testcase from the specified database.
        Prepare the data for the model by calling the prepare_data method of the model class.
        
        Args:
            testcase (str): The name of the testcase to load.
            database (str): The name of the database to load the testcase from:(UCSD, JA, MITDB, New_Dataset).
            model_class (class): The class of the deep learning model to prepare the data for.
            iStart (float): The starting index of the signal to load.
            iStop (float): The stopping index of the signal to load.
            step_interval (int): The step interval to use when preparing the data.
            
        Returns:
            dataX (): The input data for the model.
            dataY (): The output data for the model.
        """
        # print(f"Loading {testcase} DB: {database}")
        # Load the signal and annotations for the testcase
        raw_signal, annotations = self.LoadSignalAndAnnotations(testcase, database)

        # Set the start and stop indices to the beginning and end of the signal if not specified
        if(iStart == None): iStart = 0
        else: iStart = max(iStart, 0)
        if(iStop == None): iStop = len(raw_signal)
        else: iStop = min(iStop, len(raw_signal))

        # Crop the signal and annotations to the specified start and stop indices
        raw_signal = raw_signal[iStart:iStop]
        raw_signal = raw_signal.squeeze()

        # Adjust the annotations to the new start and stop indices
        if annotations is not None:
            annotations = [a - iStart for a in annotations if (a >= iStart and a <= iStop)]

        # Prepare the data for the model using the model class
        dataX, dataY = model_class.prepare_data(step_interval,raw_signal,annotations)
        return dataX, dataY
    

    def data_generator(self, csv_path, model_class, batch_size=32, type='training'):
        """
        A generator function to yield batches of data for the model.
        Yield allows the function to return an iterator to its previous state.
        
        Args:
            csv_path (str): The path to the CSV file containing the testcases.
            model_class (class): The class of the model to prepare the data for.
            batch_size (int): The batch size to use.
            type (str): The type of data to load: (training, validation, testing).
            
        Yields:
            X_t (np.ndarray): The input data for the model.
            Y_t (np.ndarray): The output data for the model.
        """
        df = pd.read_csv(csv_path)
        
        # Filter the dataframe by the type of data to load
        file_indexes = df[df['type'] == type].index.tolist()

        # Iterate through the rows of the dataframe and get the minimum and maximum values for the dataset 
        # to then yield batches of data for the model
        for idx in file_indexes:
            row = df.iloc[idx]
            minimum = row['min'] if 'min' in df.columns and not pd.isnull(row['min']) else 0
            maximum = row['max'] if 'max' in df.columns and not pd.isnull(row['max']) else 240000000

            # MITDB has a different maximum value
            if(row['dataset'] == 'MITDB'): maximum = 400000000

            # Load the data for the testcase
            try: X_t, Y_t = self.LoadTestcaseData(model_class, row['scan'], row['dataset'], minimum, maximum)
            except: continue
            
            # Yield batches of data for the model
            for i in range(0, len(X_t), batch_size): yield X_t[i:i+batch_size], Y_t[i:i+batch_size]


    def get_dataset(self, model_class, csv_path, batch_size=32, type='training'):
        """
        Get the dataset for the model.
        
        Args:
            model_class (class): The class of the model to prepare the data for.
            csv_path (str): The path to the CSV file containing the testcases.
            batch_size (int): The batch size to use.
            type (str): The type of data to load: (training, validation, testing).
            
        Returns:
            dataset (tf.data.Dataset): The dataset for the model.
        """
        import tensorflow as tf

        # Define output shapes and types for the dataset
        shape_of_X = model_class.input_shape
        shape_of_Y = model_class.output_shape
        
        # Create a dataset from the generator function
        dataset = tf.data.Dataset.from_generator(
            lambda: self.data_generator(model_class,csv_path, batch_size, type),
            output_signature=(
                tf.TensorSpec(shape=shape_of_X, dtype=tf.float32),
                tf.TensorSpec(shape=shape_of_Y, dtype=tf.float32)))
        
        return dataset.shuffle(batch_size*10, reshuffle_each_iteration=True).prefetch(tf.data.experimental.AUTOTUNE)

