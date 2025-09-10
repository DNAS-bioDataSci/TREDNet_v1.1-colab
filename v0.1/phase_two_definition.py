import os
import numpy as np
from keras.models import load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import Adadelta
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten
from keras.layers import BatchNormalization
from keras.regularizers import l1_l2
from keras.constraints import max_norm
from sklearn import metrics

EPOCH = 200
BATCH_SIZE = 64
GPUS = 4
MAX_NORM = 1

save_dir = "/data/Dcode/common/CenTRED/TREDNet/v0.1/"

def define_model():

    model = Sequential()
    model.add(Conv1D(input_shape=(4560, 1),
                     filters=64,
                     kernel_size=4,
                     strides=1,
                     activation="relu",
                     kernel_regularizer=l1_l2(0.00001, 0.001),
                     kernel_constraint=max_norm(MAX_NORM)))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.4))
    model.add(Conv1D(input_shape=(None, 64),
                     filters=128,
                     kernel_size=2,
                     strides=1,
                     activation="relu",
                     kernel_regularizer=l1_l2(0.00001, 0.001),
                     kernel_constraint=max_norm(MAX_NORM)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(units=100, activation="relu"))
    model.add(Dense(units=50, activation="relu"))
    model.add(Dense(units=1, activation="sigmoid"))

    weights_file = os.path.join(save_dir, "phase_two_weights.hdf5")
    model_file = os.path.join(save_dir, "phase_two_model.hdf5")

    model.save(model_file)

if __name__ == "__main__":
    define_model()	
