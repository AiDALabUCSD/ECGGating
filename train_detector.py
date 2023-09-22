import logging, os

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from keras.callbacks import ReduceLROnPlateau
import keras
import tensorflow as tf
from tensorflow.keras import mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)


from ecg_dataset_manager import DatasetManager
from deep_qrs_detector import DeepQRSDetector

UCSD_PROCESSED_DATASET_LOCATION = '/media/vlermakov/data/UCSDData/'
MITDB_DATASET_LOCATION = '/media/vlermakov/data/mitdb/raw/'
JA_PROCESSED_DATASET_LOCATION  = '/content/drive/MyDrive/ECG Data/JA/runtime_data/'

dm = DatasetManager(UCSD_PROCESSED_DATASET_LOCATION,JA_PROCESSED_DATASET_LOCATION,MITDB_DATASET_LOCATION)


class config:
    model_path =  './models/'
    exp_name =  'random_ucsd_train_detector_512_16'
    verbose =  1
    save_best_only =  True
    logs_path = './logs/'
    patience = 12

def define_model_callbacks():
    
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   patience=config.patience,
                                                   verbose=1
                                                  )
    check_point = keras.callbacks.ModelCheckpoint(os.path.join(config.model_path, config.exp_name+"_best.keras"),
                                                  verbose=config.verbose,
                                                  save_best_only=config.save_best_only
                                                  )
    lrate_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                                        min_delta=0.0001, min_lr=1e-6, verbose=1)

    callbacks = [early_stopping,check_point,lrate_scheduler]

    return callbacks

def build_optimizer(lr = 0.0001, opt_type = "adam"):

  if(opt_type == "adam"):
    return tf.keras.mixed_precision.LossScaleOptimizer(keras.optimizers.Adam(learning_rate=lr))
  elif(opt_type == "sgd"):
    return tf.keras.mixed_precision.LossScaleOptimizer(keras.optimizers.SGD(learning_rate=lr))


run_config = {
      "architecture": "VGG 16 Detector",
      "dataset": "FullTrainValidation",
      'batch_size': 4096,
      'cnn_layers': 10,
      'dense_neurons_1': 1024,
      'dense_neurons_2': 1024,
      'dropout_1': 0.20,
      'dropout_2': 0.50,
      'dropout_3': 0.50,
      'epochs': 150,
      'learning_rate': 0.0005,
      'optimizer': "adam"
    }

csv_path = "./data/training_and_evaluation_dataset.csv"

train_dataset = dm.get_dataset(DeepQRSDetector,csv_path, run_config['batch_size'], type='training')
val_dataset = dm.get_dataset(DeepQRSDetector,csv_path, run_config['batch_size'], type='validation')

model = DeepQRSDetector.train_model(train_dataset, val_dataset, run_config, build_optimizer(run_config["learning_rate"],run_config["optimizer"]),define_model_callbacks())

model.save(config.model_path, config.exp_name+"_best_finished.keras")