import numpy as np
import wfdb
from scipy.signal import resample
import pandas as pd
import tensorflow as tf

# Define a class to manage ECG datasets
class DatasetManager:
    def __init__(self,UCSD_PROCESSED_DATASET_LOCATION = '/content/drive/MyDrive/ECG Data/UCSDData/',JA_PROCESSED_DATASET_LOCATION  = '/content/drive/MyDrive/ECG Data/JA/runtime_data/',MITDB_DATASET_LOCATION = '/content/drive/MyDrive/ECG Data/mitdb/raw/') -> None:
        self.UCSD_PROCESSED_DATASET_LOCATION = UCSD_PROCESSED_DATASET_LOCATION
        self.JA_PROCESSED_DATASET_LOCATION  = JA_PROCESSED_DATASET_LOCATION
        self.MITDB_DATASET_LOCATION = MITDB_DATASET_LOCATION

    def LoadSignalAndAnnotations(self,testcase,database = "UCSD", TARGET_FREQUENCY = 250.0, iStart=None,iStop=None,just_train = False,step_interval = 1):
        if(database == 'UCSD'):
            p_signal = np.load(f'{self.UCSD_PROCESSED_DATASET_LOCATION}data_{testcase}.npy')[::int(1000.0/TARGET_FREQUENCY)] # Downsample from 1000 Hz to 250
            annotations = np.load(f'{self.UCSD_PROCESSED_DATASET_LOCATION}data_anno_{testcase}.npy')
            annotations = [int(a * (TARGET_FREQUENCY / 1000.0)) for a in annotations]
        elif(database == "JA"):
            #testcase = '11_sitting_chest_strap_V2_V1_Ermakov.npy'

            p_signal = np.load(f'{self.JA_PROCESSED_DATASET_LOCATION}data_{testcase}')

            annotations = np.load(f'{self.JA_PROCESSED_DATASET_LOCATION}data_anno_{testcase}')
        elif(database == "MITDB"):
            record = wfdb.rdrecord(f'{self.MITDB_DATASET_LOCATION}{testcase}', channels=[0, 1])
            annotations = wfdb.rdann(f'{self.MITDB_DATASET_LOCATION}{testcase}', 'atr')

            p_signal = record.p_signal[:,0]
            annotations = annotations.sample

            # Generate a test signal with a sampling frequency of 360 Hz
            # Downsample the signal to 250 Hz
            new_signal = resample(p_signal, int(len(p_signal)*TARGET_FREQUENCY/360.0))
            # Need to scale down signal an annotations from 360 to 250 Hz
            # Generate a test signal with a sampling frequency of 360 Hz
            # Downsample the signal to 250 Hz
            p_signal = resample(p_signal, int(len(p_signal)*TARGET_FREQUENCY/360.0))

            annotations = [ round((a * TARGET_FREQUENCY)/360.0) for a in annotations]
        else:
            raise Exception("No such dataset.")

        if(iStart == None):
            iStart = 0
        if(iStop == None):
            iStop = len(p_signal)
            
        p_signal = p_signal[iStart:iStop]

        annotations = [a - iStart for a in annotations if (a >= iStart and a < iStop)]

        raw_signal = p_signal.squeeze()

        return raw_signal, annotations
        
    def LoadTestcaseData(self, model_class, testcase,database = "JA", iStart=None,iStop=None,step_interval = 1):
        #print(f"Loading {testcase} DB: {database}")

        raw_signal, annotations = self.LoadSignalAndAnnotations(testcase, database)

        if(iStart == None):
            iStart = 0
        if(iStop == None):
            iStop = len(raw_signal)

        raw_signal = raw_signal[iStart:iStop]
        annotations = [a - iStart for a in annotations if (a >= iStart and a < iStop)]
        raw_signal = raw_signal.squeeze()

        dataX, dataY = model_class.prepare_data(step_interval,raw_signal,annotations)
        
        return dataX, dataY
    
    def data_generator(self, model_class, csv_path, batch_size=32, type='training'):
        df = pd.read_csv(csv_path)
        
        file_indexes = df[df['type'] == type].index.tolist()

        
        for idx in file_indexes:
            row = df.iloc[idx]
            minimum = row['min'] if 'min' in df.columns and not pd.isnull(row['min']) else 0
            maximum = row['max'] if 'max' in df.columns and not pd.isnull(row['max']) else 240000000

            if(row['dataset'] == 'MITDB'):
                maximum = 400000000

            try:
                X_t, Y_t = self.LoadTestcaseData(model_class, row['scan'], row['dataset'], minimum, maximum)
            except:
                continue
            
            for i in range(0, len(X_t), batch_size):
                yield X_t[i:i+batch_size], Y_t[i:i+batch_size]

    def get_dataset(self, model_class, csv_path, batch_size=32, type='training'):
        # Define output shapes and types for the dataset
        shape_of_X = model_class.input_shape
        shape_of_Y = model_class.output_shape
        
        dataset = tf.data.Dataset.from_generator(
            lambda: self.data_generator(model_class,csv_path, batch_size, type)
            ,
            output_signature=(
                tf.TensorSpec(shape=shape_of_X, dtype=tf.float32),
                tf.TensorSpec(shape=shape_of_Y, dtype=tf.float32)
            )
        )
        
        return dataset.shuffle(batch_size*10, reshuffle_each_iteration=True).prefetch(tf.data.experimental.AUTOTUNE)

