import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_probability as tfp


from keras.models import Sequential
from keras.layers import MaxPooling1D, Flatten, Dense, Dropout,Concatenate
from keras.layers import Dense, Conv1D


DETECTION_WINDOW = 16 # <-- 16 samples = 64ms
SAMPLES_PER_SLICE = 512 # <-- 512 samples = 2.048s
WINDOWS_PER_SLICE = 1
# WINDOWS_PER_SLICE = int(SAMPLES_PER_SLICE/DETECTION_WINDOW)

OFFSET_FROM_END = 0
OFFSET_FROM_FRONT = SAMPLES_PER_SLICE - OFFSET_FROM_END - DETECTION_WINDOW * WINDOWS_PER_SLICE

SAMPLING_FREQUENCY = 250

SAMPLE_STEP = 10

NORM_JITTER = 10E-3 # sec


def ecg_detection_loss(y_true,y_pred):
    y_true = tf.cast(y_true,tf.float32)
    y_pred = tf.cast(y_pred,tf.float32)

    # return ecg_detection_loss_bin(y_true, y_pred,6)

    accuracy_lambda = 5.0
    precision_lambda = 0.002
    
    # Split the y_pred tensor into binary and continuous parts
    binary_pred, continuous_pred = tf.split(y_pred, num_or_size_splits=[WINDOWS_PER_SLICE, WINDOWS_PER_SLICE], axis=-1)

    # Compute binary cross-entropy loss on the binary part
    bce_loss = tf.keras.losses.BinaryCrossentropy()(y_true[:, :WINDOWS_PER_SLICE], binary_pred)

    mse_loss = tf.reduce_mean(tf.square(tf.multiply(y_true[:,:WINDOWS_PER_SLICE], continuous_pred - y_true[:, WINDOWS_PER_SLICE:])))

    # Combine the losses
    loss = bce_loss + accuracy_lambda * mse_loss# + precision_lambda * mse_loss_rounded

    return loss

def ecg_accuracy(y_true, y_pred):
    y_true = tf.cast(y_true,tf.float32)
    y_pred = tf.cast(y_pred,tf.float32)

    #return ecg_accuracy_bin(y_true, y_pred,6)
    # Convert true labels from float to integer
    y_true = tf.cast(y_true > 0.5, tf.int32)
    
    # Convert predicted probabilities to integer labels
    y_pred = tf.round(y_pred)
    y_pred = tf.cast(y_pred, tf.int32)
    
    # Compare true and predicted labels
    correct_predictions = tf.equal(y_true[:,0:WINDOWS_PER_SLICE], y_pred[:,0:WINDOWS_PER_SLICE])
    
    # Compute accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    
    return accuracy


# The jitter which gives a 50% performance. That's a 1/4 of the average
# RMSSD which is 40ms.
NORM_JITTER = 10E-3 # sec

def ja_metric(y_true, y_pred):
    y_true = tf.cast(y_true,tf.float32)
    y_pred = tf.cast(y_pred,tf.float32)

    #return ja_metric_bin(y_true, y_pred,6)
    binary_pred, continuous_pred = tf.split(y_pred, num_or_size_splits=[WINDOWS_PER_SLICE, WINDOWS_PER_SLICE], axis=-1)

    # Compare true and predicted labels
    correct_predictions = tf.equal(tf.cast(tf.round(y_true[:,0:WINDOWS_PER_SLICE]),tf.int32), tf.cast(tf.round(binary_pred),tf.int32))

    # Compute accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))


    # Compute MSE loss on the continuous part where the binary value is true (i.e., > 0.5)
    true_binary = tf.cast(y_true[:, :WINDOWS_PER_SLICE] > 0.5, tf.float32)

    differences_for_jitter = tf.abs(tf.divide(tf.multiply(true_binary, tf.round(continuous_pred * tf.constant(DETECTION_WINDOW,tf.float32)) - tf.round(y_true[:, WINDOWS_PER_SLICE:] * tf.constant(DETECTION_WINDOW,tf.float32))),SAMPLING_FREQUENCY))
    jitter = tfp.stats.percentile(differences_for_jitter, 50.0, interpolation='midpoint')

    jitter_score = 1 / ( 1 + (jitter / NORM_JITTER) )

    return jitter_score * accuracy

def ja_loss(y_true, y_pred):
    y_true = tf.cast(y_true,tf.float32)
    y_pred = tf.cast(y_pred,tf.float32)

    return 1.0 - ecg_detection_loss(y_true,y_pred)

def qrs_jitter(y_true, y_pred):
    y_true = tf.cast(y_true,tf.float32)
    y_pred = tf.cast(y_pred,tf.float32)

    #return qrs_jitter_bin(y_true, y_pred,6)
    # Split the y_pred tensor into binary and continuous parts
    binary_pred, continuous_pred = tf.split(y_pred, num_or_size_splits=[WINDOWS_PER_SLICE, WINDOWS_PER_SLICE], axis=-1)
    
    # Compute MSE loss on the continuous part where the binary value is true (i.e., > 0.5)
    true_binary = tf.cast(y_true[:, :WINDOWS_PER_SLICE] > 0.5, tf.float32)
    
    mse_loss = tf.reduce_mean(tf.square(tf.multiply(true_binary, continuous_pred - y_true[:, WINDOWS_PER_SLICE:])))
    
    return mse_loss

# Define a class to detect QRS complexes in ECG signals
class DeepQRSDetector:# Static variables
    input_shape = (None,SAMPLES_PER_SLICE,1)
    output_shape = (None,2)

    def __init__(self,model_file=None) -> None:
        self.fs = 250 # Default sampling rate is 250 Hz

        if(model_file is None):
          model_file = '/content/drive/MyDrive/ECG Data/Training/transfermitbih_retrained_ucsd_mractri_512_16_8_best.h5'
          
        # Load the model
        #self.model = keras.models.load_model(model_file,custom_objects={'ecg_detection_loss': ecg_detection_loss, 'ja_metric': ja_metric,'ecg_accuracy': ecg_accuracy,'qrs_jitter': qrs_jitter})

        self.model = keras.models.load_model(model_file,custom_objects={'ecg_detection_loss': ecg_detection_loss, 'ja_metric': ja_metric,'ecg_accuracy': ecg_accuracy,'qrs_jitter': qrs_jitter})
    
    def detect_peaks(self, raw_ecg_signal, sampling_rate = None):
        if(sampling_rate is None):
            sampling_rate = self.fs

        # Prepare the signal for the model
        ecg_signal_x_norm, _ = self.prepare_data(SAMPLE_STEP,raw_ecg_signal)

        # Detect peaks
        p_output = self.model.predict(ecg_signal_x_norm,verbose=0)

        # process the output of the model to get the peaks
        qrs_peaks,mean_prediction_offset = self.process_output(p_output,raw_ecg_signal)
        
        return qrs_peaks,mean_prediction_offset
    
    def process_output(self, p_output, unfiltered_ecg):
        peaks_candidates = []

        for iSlice,p in enumerate(p_output):
            for iDetection in range(0,WINDOWS_PER_SLICE):
                if(p[iDetection]>0.5):
                    #peaks_candidates.append((p[iDetection],(iSlice*SAMPLE_STEP) + OFFSET_FROM_FRONT +(DETECTION_WINDOW * iDetection) + DETECTION_WINDOW*p[WINDOWS_PER_SLICE+iDetection]))
                    predictionX = OFFSET_FROM_FRONT +(DETECTION_WINDOW * iDetection) + DETECTION_WINDOW*p[WINDOWS_PER_SLICE+iDetection]
                    medianIndex = (iSlice*SAMPLE_STEP) + predictionX
                    peaks_candidates.append((medianIndex,predictionX,iSlice))

        peaks_candidates.sort(key = lambda x: x[2])

        qrs_peaks = []
        peak_offsets = []

        for peak in peaks_candidates:
            # Check if the peak is at least 30 samples away from oll other peaks
            if(len(qrs_peaks) == 0 or min(abs(np.array(qrs_peaks) - peak[0])) > 30):
                qrs_peaks.append(peak[0])
                peak_offsets.append(peak[1])
                
        return qrs_peaks, np.mean(peak_offsets)
    
    def estimate_loss(self, raw_ecg_signal, annotations, sampling_rate = None):
        if(sampling_rate is None):
            sampling_rate = self.fs

        # Prepare the signal for the model
        ecg_signal_x_norm, ecg_signal_y = DeepQRSDetector.prepare_data(SAMPLE_STEP, raw_ecg_signal, annotations)

        # Detect peaks
        #p_output = self.model.predict(ecg_signal_x_norm,verbose=0)

        #loss = quantile_loss(ecg_signal_y, p_output).numpy()

        # Evaluate the model on the test set
        loss, ja_metric, ecg_accuracy = self.model.evaluate(ecg_signal_x_norm, ecg_signal_y,batch_size=2048)

        return loss
    

    @staticmethod
    def normalize(raw_ecg_signal):
        normalized_signal = (raw_ecg_signal - np.mean(raw_ecg_signal,axis=(1,2),keepdims=True))/np.std(raw_ecg_signal,axis=(1,2),keepdims=True)
        return normalized_signal
    
    @staticmethod
    def prepare_data(step, unfiltered_ecg, annotations = None):
        dataX,dataY = DeepQRSDetector.create_windowed_dataset(step,unfiltered_ecg, annotations)

        # Normalize the signal
        dataX = DeepQRSDetector.normalize(dataX)
        
        return dataX, dataY
    
    @staticmethod
    def create_windowed_dataset(step, signal, annotations = None):
        if(annotations is not None):
            # Create a numpy array of zeros for the annotations
            annotations_array = np.zeros(signal.shape)
            # Mark the annotation index to make it easier to generate labels
            for idx in annotations:
                annotations_array[idx] = 1
            
        # Create X and Y lists to hold the windowed signal and annotations
        X, Y = [], []

        # Slide a SAMPLES_PER_SLICE-pixel window through the signal with stride 1
        for iSignal in range(0,len(signal)-SAMPLES_PER_SLICE-1,step):
            # Add the slice of the signal in the window to X
            X.append(signal[iSignal:iSignal+SAMPLES_PER_SLICE].reshape(-1,1))
            
            if(annotations is not None):
                annotationsWindow = annotations_array[iSignal:iSignal+SAMPLES_PER_SLICE]
                # break up annotations array in to 64 sample windows
                # for each 64 sample window add a label and location of the peak
                # in the sample as integer (we can use in in MSE loss later)
                label = np.zeros(int(WINDOWS_PER_SLICE*2))
                label_index = 0
                for iAnnotation in range(iSignal+OFFSET_FROM_FRONT,iSignal+SAMPLES_PER_SLICE-OFFSET_FROM_END,DETECTION_WINDOW):
                    annotationsWindow = annotations_array[iAnnotation:iAnnotation+DETECTION_WINDOW]
                    
                    if(label_index >= 0 and label_index < 15):
                        if(np.sum(annotationsWindow) > 0):
                            label[label_index] = 1.0
                            label[WINDOWS_PER_SLICE + label_index] = float(np.argmax(annotationsWindow) / float(DETECTION_WINDOW))

                    label_index += 1
                    
                # Add the slice of the annotations in the window to Y
                Y.append(label)

        # Convert X and Y to numpy arrays and return them
        if(annotations is not None):
            return np.array(X), np.array(Y)
        else:
            return np.array(X), None
    
    @staticmethod
    def build_cnn_detector_network(cnn_layers = 10, dense_layers_1=1024,dense_layers_2=1024,dropouts_1=0.2,dropouts_2=0.5,dropouts_3=0.5):
        # Define the Sequential model
        ecg_model = Sequential()

        # Add the convolutional layers

        ecg_model.add(Conv1D(64, 3,  activation='relu', padding='same', input_shape=(SAMPLES_PER_SLICE,1)))
        ecg_model.add(Conv1D(64, 3,  activation='relu', padding='same'))
        ecg_model.add(MaxPooling1D(2, strides=2))
        ecg_model.add(Dropout(0.2))

        ecg_model.add(Conv1D(128, 3, activation='relu', padding='same'))
        ecg_model.add(Conv1D(128, 3, activation='relu', padding='same'))
        ecg_model.add(MaxPooling1D(2, strides=2))
        ecg_model.add(Dropout(0.2))

        ecg_model.add(Conv1D(256, 3, activation='relu', padding='same'))
        ecg_model.add(Conv1D(256, 3, activation='relu', padding='same'))
        if(cnn_layers >= 13):
            ecg_model.add(Conv1D(256, 3, activation='relu', padding='same'))

        ecg_model.add(MaxPooling1D(2, strides=2))
        ecg_model.add(Dropout(0.2))

        ecg_model.add(Conv1D(512, 3, activation='relu', padding='same'))
        ecg_model.add(Conv1D(512, 3, activation='relu', padding='same'))
        if(cnn_layers >= 12):
            ecg_model.add(Conv1D(512, 3, activation='relu', padding='same'))
        ecg_model.add(MaxPooling1D(2, strides=2))
        ecg_model.add(Dropout(0.2))

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

        detection = Dense(WINDOWS_PER_SLICE, activation='sigmoid')(ecg_model.output)
        location = Dense(WINDOWS_PER_SLICE, activation='relu')(ecg_model.output)

        combined_output = Concatenate(name='combined_output')([detection,location])

        return tf.keras.models.Model(inputs=ecg_model.inputs,outputs=combined_output) 
    
    @staticmethod
    def train_model(train_dataset, val_dataset, run_config, optimizer, callbacks):
        model = DeepQRSDetector.build_cnn_detector_network(run_config["cnn_layers"],run_config["dense_neurons_1"],run_config["dense_neurons_2"],run_config["dropout_1"],run_config["dropout_2"],run_config["dropout_3"])

        model.compile(loss=ecg_detection_loss, optimizer=optimizer, metrics=[ecg_accuracy])

        model.fit(train_dataset, validation_data=val_dataset, epochs=run_config["epochs"],batch_size=run_config["batch_size"],callbacks=callbacks)

        return model