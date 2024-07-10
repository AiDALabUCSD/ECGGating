import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K

from keras.models import Sequential
from keras.layers import MaxPooling1D, Flatten, Dense, Dropout
from keras.layers import Dense, Conv1D


DETECTION_WINDOW = 64 # <-- 64 samples = 256 ms overlap between context and prediction window
SAMPLES_PER_SLICE = 512 # <-- 512 samples = 2.048 s
PREDICTION_WINDOW = SAMPLES_PER_SLICE*2 # <-- 1024 samples = 4.096 s
WINDOWS_PER_SLICE = 1
# WINDOWS_PER_SLICE = int(SAMPLES_PER_SLICE/DETECTION_WINDOW)

OFFSET_FROM_END = 8 # <-- 8 samples = 32 ms
OFFSET_FROM_FRONT = SAMPLES_PER_SLICE - OFFSET_FROM_END - DETECTION_WINDOW * WINDOWS_PER_SLICE
# OFFSET_FROM_FRONT = 0

SAMPLING_FREQUENCY = 250

PREDICTED_PEAKS = 2
QUANTILES_PREDICTED = 3

QUANTILE_LOW = 0
QUANTILE_MID = 1
QUANTILE_HIGH = 2

QUANTILE_LOW_THRESHOLD = 0.05
QUANTILE_MID_THRESHOLD = 0.5
QUANTILE_HIGH_THRESHOLD = 0.95


SAMPLE_STEP = 10

NORM_JITTER = 10E-3 # sec

def quantile_loss(y_true, y_pred):
    y_true = tf.cast(y_true,tf.float32)
    y_pred = tf.cast(y_pred,tf.float32)

    q = np.array([QUANTILE_LOW_THRESHOLD,QUANTILE_MID_THRESHOLD,QUANTILE_HIGH_THRESHOLD]*PREDICTED_PEAKS)
    err = (y_true-y_pred)
    return K.mean(K.maximum(q*err, (q-1)*err), axis=-1)

# Define a class to detect QRS complexes in ECG signals
class DeepQRSPredictor:
    # Static variables
    input_shape = (None,SAMPLES_PER_SLICE,1)
    output_shape = (None,PREDICTED_PEAKS*QUANTILES_PREDICTED)


    def __init__(self,model_file = None, detection_confidence_threshold = 18.0, peak_cooldown_threshold = 30.0) -> None:
        self.fs = 250 # Default sampling rate is 250 Hz

        if(model_file is None):
            model_file = '/content/drive/MyDrive/ECG Data/Training/retrain_rand_splt_1024_mixed_best.h5'

        # Load the model
        self.model = keras.models.load_model(model_file,custom_objects={'quantile_loss': quantile_loss})

        self.DETECTION_CONFIDENCE_THRESHOLD = detection_confidence_threshold
        self.PEAK_COOLDOWN_THRESHOLD = peak_cooldown_threshold
        self.SAMPLES_PER_SLICE = SAMPLES_PER_SLICE
        self.PREDICTION_WINDOW = PREDICTION_WINDOW
        self.DETECTION_WINDOW = DETECTION_WINDOW
        self.OFFSET_FROM_END = OFFSET_FROM_END
        self.OFFSET_FROM_FRONT = OFFSET_FROM_FRONT
        self.SAMPLING_FREQUENCY = SAMPLING_FREQUENCY
        self.PREDICTED_PEAKS = PREDICTED_PEAKS
        self.QUANTILES_PREDICTED = QUANTILES_PREDICTED
        self.QUANTILE_LOW = QUANTILE_LOW
        self.QUANTILE_MID = QUANTILE_MID
        self.QUANTILE_HIGH = QUANTILE_HIGH
        self.QUANTILE_LOW_THRESHOLD = QUANTILE_LOW_THRESHOLD
        self.QUANTILE_MID_THRESHOLD = QUANTILE_MID_THRESHOLD
        self.QUANTILE_HIGH_THRESHOLD = QUANTILE_HIGH_THRESHOLD
        self.SAMPLE_STEP = SAMPLE_STEP
        
    
        
    def detect_peaks(self,raw_ecg_signal, sampling_rate = None, return_confidence_intervals = False):
        if(sampling_rate is None):
            sampling_rate = self.fs

        # Prepare the signal for the model
        ecg_signal_x_norm, _ = DeepQRSPredictor.prepare_data(SAMPLE_STEP, raw_ecg_signal)

        # Detect peaks
        p_output = self.model.predict(ecg_signal_x_norm,verbose=0)

        # process the output of the model to get the peaks
        qrs_peaks,mean_prediction_offset = self.process_output(p_output,raw_ecg_signal)

        if(return_confidence_intervals):
            return qrs_peaks,mean_prediction_offset, p_output
        else:
            return qrs_peaks,mean_prediction_offset
    
    def estimate_loss(self, raw_ecg_signal, annotations, sampling_rate = None):
        if(sampling_rate is None):
            sampling_rate = self.fs

        # Prepare the signal for the model
        ecg_signal_x_norm, ecg_signal_y = DeepQRSPredictor.prepare_data(SAMPLE_STEP, raw_ecg_signal, annotations)

        # Detect peaks
        p_output = self.model.predict(ecg_signal_x_norm,verbose=0)

        #loss = quantile_loss(ecg_signal_y, p_output).numpy()

        # Evaluate the model on the test set
        loss, accuracy = self.model.evaluate(ecg_signal_x_norm, ecg_signal_y,batch_size=2048)

        return loss
    
    @staticmethod
    def normalize(raw_ecg_signal):
        normalized_signal = (raw_ecg_signal - np.mean(raw_ecg_signal,axis=(1,2),keepdims=True))/np.std(raw_ecg_signal,axis=(1,2),keepdims=True)
        return normalized_signal
    
    @staticmethod
    def prepare_data(step, unfiltered_ecg, annotations = None):
        dataX,dataY = DeepQRSPredictor.create_windowed_dataset(step,unfiltered_ecg, annotations)

        # Normalize the signal
        dataX = DeepQRSPredictor.normalize(dataX)
        
        return dataX, dataY
    
    @staticmethod
    def create_windowed_dataset(step, signal, annotations = None):
        annotationLabels = []

        if(annotations is not None):
            # Create a numpy array of zeros for the annotations
            annotations_array = np.zeros(signal.shape)
            # Mark the annotation index to make it easier to generate labels
            for idx in annotations:
                annotations_array[idx] = 1
                
        # Create X and Y lists to hold the windowed signal and annotations
        X, Y = [], []

        # Slide a SAMPLES_PER_SLICE-pixel window through the signal with stride 1
        # Stop one SAMPLES_PER_SLICE short, to do prediction for all slices
        for iSignal in range(0,len(signal)-(SAMPLES_PER_SLICE+PREDICTION_WINDOW),step):
            # Add the slice of the signal in the window to X
            X.append(signal[iSignal:iSignal+SAMPLES_PER_SLICE].reshape(-1,1))
            
            if(annotations is not None):
                #annotationsWindow = annotations_array[iSignal:iSignal+SAMPLES_PER_SLICE]
                # break up annotations array in to 64 sample windows
                # for each 64 sample window add a label and location of the peak
                # in the sample as integer (we can use in in MSE loss later)
                
                label = np.zeros(PREDICTED_PEAKS*QUANTILES_PREDICTED)
                
                nextPeaks = annotations_array[iSignal+SAMPLES_PER_SLICE-DETECTION_WINDOW:iSignal+SAMPLES_PER_SLICE+PREDICTION_WINDOW-DETECTION_WINDOW]

                nextPeaks = np.nonzero(nextPeaks)[0]
                
                label_index = 0 #WINDOWS_PER_SLICE*2
                
                # save location offset of next peak and 5 and 95 % quantiles
                for iNextPeak in range(0,min(len(nextPeaks),PREDICTED_PEAKS)):
                    label[label_index+QUANTILE_LOW] = nextPeaks[iNextPeak] / PREDICTION_WINDOW
                    label[label_index+QUANTILE_MID] = nextPeaks[iNextPeak] / PREDICTION_WINDOW
                    label[label_index+QUANTILE_HIGH] = nextPeaks[iNextPeak] / PREDICTION_WINDOW
                    label_index += QUANTILES_PREDICTED
                
                # Add the slice of the annotations in the window to Y
                Y.append(label)

        # Convert X and Y to numpy arrays and return them
        if(annotations is not None):
            return np.array(X), np.array(Y)
        else:
            return np.array(X), None
        
    def process_output(self, p_output, unfiltered_ecg):
        peaks_candidates = []
        
        for iSlice,p in enumerate(p_output):
          for iPeak in range(PREDICTED_PEAKS):
              predictionX = SAMPLES_PER_SLICE - DETECTION_WINDOW + PREDICTION_WINDOW*p_output[iSlice,iPeak*QUANTILES_PREDICTED+QUANTILE_MID]
              predictionQuantileLow = SAMPLES_PER_SLICE - DETECTION_WINDOW + PREDICTION_WINDOW*p_output[iSlice,iPeak*QUANTILES_PREDICTED+QUANTILE_LOW]
              predictionQuantileHigh = SAMPLES_PER_SLICE - DETECTION_WINDOW + PREDICTION_WINDOW*p_output[iSlice,iPeak*QUANTILES_PREDICTED+QUANTILE_HIGH]
              
              confidenceRange = predictionQuantileHigh-predictionQuantileLow
              medianIndex = round((iSlice*SAMPLE_STEP) + predictionX)

              if(confidenceRange < self.DETECTION_CONFIDENCE_THRESHOLD and medianIndex < len(unfiltered_ecg)):
                  peaks_candidates.append((medianIndex,predictionX,iSlice))
                  
        peaks_candidates.sort(key = lambda x: x[2])

        qrs_peaks = []
        peak_offsets = []

        for peak in peaks_candidates:
          # Check if the peak is at least 30 samples away from all other peaks
          if(len(qrs_peaks) == 0 or min(abs(np.array(qrs_peaks) - peak[0])) > self.PEAK_COOLDOWN_THRESHOLD):
              qrs_peaks.append(peak[0])
              peak_offsets.append(peak[1])

        return qrs_peaks,np.mean(peak_offsets)
        
    @staticmethod
    def build_cnn_predictor_network(cnn_layers = 10, dense_layers_1=1024,dense_layers_2=1024,dropouts_1=0.2,dropouts_2=0.5,dropouts_3=0.5):
        # Define the Sequential model
        ecg_model = Sequential()

        # Add the convolutional layers
        ecg_model.add(Conv1D(64, 3,  activation='relu', padding='same', input_shape=(SAMPLES_PER_SLICE,1)))
        ecg_model.add(Conv1D(64, 3,  activation='relu', padding='same'))
        ecg_model.add(MaxPooling1D(2, strides=2))
        ecg_model.add(Dropout(dropouts_1))

        ecg_model.add(Conv1D(128, 3, activation='relu', padding='same'))
        ecg_model.add(Conv1D(128, 3, activation='relu', padding='same'))
        ecg_model.add(MaxPooling1D(2, strides=2))
        ecg_model.add(Dropout(dropouts_1))

        ecg_model.add(Conv1D(256, 3, activation='relu', padding='same'))
        ecg_model.add(Conv1D(256, 3, activation='relu', padding='same'))

        if(cnn_layers >= 13):
            ecg_model.add(Conv1D(256, 3, activation='relu', padding='same'))
            ecg_model.add(MaxPooling1D(2, strides=2))
            ecg_model.add(Dropout(dropouts_1))

            ecg_model.add(Conv1D(512, 3, activation='relu', padding='same'))
            ecg_model.add(Conv1D(512, 3, activation='relu', padding='same'))

        if(cnn_layers >= 12):
            ecg_model.add(Conv1D(512, 3, activation='relu', padding='same'))
            ecg_model.add(MaxPooling1D(2, strides=2))
            ecg_model.add(Dropout(dropouts_1))

        ecg_model.add(Conv1D(512, 3, activation='relu', padding='same'))
        ecg_model.add(Conv1D(512, 3, activation='relu', padding='same'))
        if(cnn_layers >= 11):
            ecg_model.add(Conv1D(512, 3, activation='relu', padding='same'))
        if(cnn_layers >= 14):
            ecg_model.add(Conv1D(512, 3, activation='relu', padding='same'))


        ecg_model.add(MaxPooling1D(2, strides=2))

        # Add the fully connected layers
        ecg_model.add(Flatten())
        ecg_model.add(Dropout(dropouts_2))
        ecg_model.add(Dense(dense_layers_1, activation='relu'))
        ecg_model.add(Dropout(dropouts_2))
        ecg_model.add(Dense(dense_layers_2, activation='relu'))
        ecg_model.add(Dropout(dropouts_3))

        prediction = Dense(PREDICTED_PEAKS*QUANTILES_PREDICTED, activation='linear')(ecg_model.output)
        
        return tf.keras.models.Model(inputs=ecg_model.inputs,outputs=prediction)

    @staticmethod
    def train_model(train_dataset, val_dataset, run_config, optimizer, callbacks):

        model = DeepQRSPredictor.build_cnn_predictor_network(run_config["cnn_layers"],run_config["dense_neurons_1"],run_config["dense_neurons_2"],run_config["dropout_1"],run_config["dropout_2"],run_config["dropout_3"])
        
        model.compile(loss=quantile_loss, optimizer=optimizer, metrics='accuracy')
        
        model.fit(train_dataset, validation_data=val_dataset, epochs=run_config["epochs"],batch_size=run_config["batch_size"],callbacks=callbacks)

        return model
