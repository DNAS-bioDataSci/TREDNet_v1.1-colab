import os
import sys
import numpy as np
import random
import time
import glob
from Bio import SeqIO
from pybedtools import BedTool
from keras.models import load_model
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.optimizers import Adadelta
from sklearn import metrics
import h5py
import tensorflow as tf
import keras
from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten
from keras.layers import BatchNormalization
from keras.constraints import max_norm
import datetime


train_chromosomes = ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr10", "chr11", "chr12", "chr13",
                     "chr14", "chr15", "chr16", "chr17", "chr18", "chr19", "chr20", "chr21", "chr22", "chrX", "chrY"]
validation_chromosomes = ["chr7"]
test_chromosomes = ["chr8", "chr9"]

BIN_LENGTH = 200
INPUT_LENGTH = 2001
EPOCH = 20		#########change for final run to 200
BATCH_SIZE = 64
GPUS = 4
MAX_NORM = 1

nucleotides = ['A', 'C', 'G', 'T']

#replaced by sh file
MODEL_DIR = "sourcemodelI"
FASTA_FILE = "sourcefasta"
#source_input="sourceinput/EID"
source_input="sourceinput/"
#SAVE_DIR = "sourceoutput/EID"
SAVE_DIR = "dl_dir"

###############################################################################################################################################
def create_dataset():

    print("running create_dataset")
    positive_bed_file = os.path.join(source_input + "EID_positive_1kb.bed")
    print(os.path.join(source_input + "EID_positive_1kb.bed"))
    negative_bed_file = os.path.join(source_input + "EID_control_1kb.bed")
    print(os.path.join(source_input + "EID_control_1kb.bed"))
    print(os.path.join(SAVE_DIR, "phase_two_dataset.hdf5"))
    dataset_save_file = os.path.join(SAVE_DIR, "phase_two_dataset.hdf5")

    create_dataset_for_phase_two(positive_bed_file, negative_bed_file, dataset_save_file)
	
###############################################################################################################################################
def get_chrom2seq(capitalize=True):

    chrom2seq = {}
    for seq in SeqIO.parse(FASTA_FILE, "fasta"):
        chrom2seq[seq.description.split()[0]] = seq.seq.upper() if capitalize else seq.seq

    return chrom2seq

###############################################################################################################################################
def seq2one_hot(seq):
    alphabet = np.array(['A','C','G','T'])
    seq = seq.upper()
    char_to_idx = {ch: i for i, ch in enumerate(alphabet)}
    idx = np.array([char_to_idx.get(ch, -1) for ch in seq])
    one_hot = np.zeros((len(seq), 4), dtype=int)
    valid = idx >= 0
    one_hot[np.arange(len(seq))[valid], idx[valid]] = 1
    return one_hot


###############################################################################################################################################
def get_phase_one_model():

    print("running get phase 1 model")
    phase_one_model_file = os.path.join(MODEL_DIR, "phase_one_model.h5")
    phase_one_weights_file = os.path.join(MODEL_DIR, "phase_one_weights.h5")

    model = load_model(phase_one_model_file)
    model.load_weights(phase_one_weights_file)

    return model

###############################################################################################################################################
def create_dataset_for_phase_two(positive_bed_file, negative_bed_file, dataset_save_file):

    print("running create_dataset_forphase2")
    chrom2seq = get_chrom2seq()
        # return chrom2seq

    model = get_phase_one_model()
        # return model

    print ("Generating the positive dataset")

    pos_beds = list(BedTool(positive_bed_file))
    for r in pos_beds:
        r.start -= 501
        r.stop += 500
    neg_beds = list(BedTool(negative_bed_file))
    for r in neg_beds:
        r.start -= 501
        r.stop += 500

    pos_train_bed = [r for r in pos_beds if r.chrom in train_chromosomes]
    pos_val_bed = [r for r in pos_beds if r.chrom in validation_chromosomes]
    pos_test_bed = [r for r in pos_beds if r.chrom in test_chromosomes]

    pos_train_data = []
    pos_val_data = []
    pos_test_data = []

    for bed_list, data_list in zip([pos_train_bed, pos_val_bed, pos_test_bed],
                                   [pos_train_data, pos_val_data, pos_test_data]):

        for r in bed_list:
            _seq = chrom2seq[r.chrom][r.start:r.stop]
            if not len(_seq) == 2001:
                continue
            _vector = seq2one_hot(_seq)
            data_list.append(_vector)

    print (len(pos_train_data))
    print (len(pos_val_data))
    print (len(pos_test_data))
    
    print ("Generating the negative dataset")

    neg_train_bed = [r for r in neg_beds if r.chrom in train_chromosomes]
    neg_val_bed = [r for r in neg_beds if r.chrom in validation_chromosomes]
    neg_test_bed = [r for r in neg_beds if r.chrom in test_chromosomes]

    neg_train_data = []
    neg_val_data = []
    neg_test_data = []
    
    for bed_list, data_list in zip([neg_train_bed, neg_val_bed, neg_test_bed],
                                   [neg_train_data, neg_val_data, neg_test_data]):
        for r in bed_list:
            _seq = chrom2seq[r.chrom][r.start:r.stop]
            if not len(_seq) == 2001:
                continue
            _vector = seq2one_hot(_seq)
            data_list.append(_vector)

    print (len(neg_train_data))
    print (len(neg_val_data))
    print (len(neg_test_data))

    print ("Merging positive and negative to single matrices")

    pos_train_data_matrix = np.zeros((len(pos_train_data), INPUT_LENGTH, 4))
    for i in range(len(pos_train_data)):
        pos_train_data_matrix[i, :, :] = pos_train_data[i]
    pos_val_data_matrix = np.zeros((len(pos_val_data), INPUT_LENGTH, 4))
    for i in range(len(pos_val_data)):
        pos_val_data_matrix[i, :, :] = pos_val_data[i]
    pos_test_data_matrix = np.zeros((len(pos_test_data), INPUT_LENGTH, 4))
    for i in range(len(pos_test_data)):
        pos_test_data_matrix[i, :, :] = pos_test_data[i]

    neg_train_data_matrix = np.zeros((len(neg_train_data), INPUT_LENGTH, 4))
    for i in range(len(neg_train_data)):
        neg_train_data_matrix[i, :, :] = neg_train_data[i]
    neg_val_data_matrix = np.zeros((len(neg_val_data), INPUT_LENGTH, 4))
    for i in range(len(neg_val_data)):
        neg_val_data_matrix[i, :, :] = neg_val_data[i]
    neg_test_data_matrix = np.zeros((len(neg_test_data), INPUT_LENGTH, 4))
    for i in range(len(neg_test_data)):
        neg_test_data_matrix[i, :, :] = neg_test_data[i]

    test_data = np.vstack((pos_test_data_matrix, neg_test_data_matrix))
    test_labels = np.concatenate((np.ones(len(pos_test_data)), np.zeros(len(neg_test_data))))
    train_data = np.vstack((pos_train_data_matrix, neg_train_data_matrix))
    train_labels = np.concatenate((np.ones(len(pos_train_data)), np.zeros(len(neg_train_data))))
    val_data = np.vstack((pos_val_data_matrix, neg_val_data_matrix))
    val_labels = np.concatenate((np.ones(len(pos_val_data)), np.zeros(len(neg_val_data))))

    test_data = model.predict(test_data)
    train_data = model.predict(train_data)
    val_data = model.predict(val_data)


    print ("Saving to file:", dataset_save_file)

    with h5py.File(dataset_save_file, "w") as of:
        of.create_dataset(name="test_data", data=test_data, compression="gzip")
        of.create_dataset(name="test_labels", data=test_labels, compression="gzip")
        of.create_dataset(name="train_data", data=train_data, compression="gzip")
        of.create_dataset(name="train_labels", data=train_labels, compression="gzip")
        of.create_dataset(name="val_data", data=val_data, compression="gzip")
        of.create_dataset(name="val_labels", data=val_labels, compression="gzip")

###############################################################################################################################################
def train_model():

    print("running train model()")
    import argparse
    import os
    import sys

    data_file = os.path.join(SAVE_DIR, "phase_two_dataset.hdf5")
    phase_two_model_file = os.path.join(MODEL_DIR, "phase_two_model.hdf5")
    model = load_model(phase_two_model_file)
    print("Loading the dataset from:", data_file)
    data = load_dataset(data_file)
    print("Launching the training of model")
    print("Model files and performance evaluation results will be written in:")
    print("        " + SAVE_DIR)
    run_model(data, model, SAVE_DIR)

###############################################################################################################################################
def load_dataset(data_file):

    print("running load dataset()")
    data = {}

    with h5py.File(data_file, "r") as inf:
        for _key in inf:
            data[_key] = inf[_key][()]

    data["train_data"] = data["train_data"][..., np.newaxis]
    data["test_data"] = data["test_data"][..., np.newaxis]
    data["val_data"] = data["val_data"][..., np.newaxis]

    return data

###############################################################################################################################################
def run_model(data, model, save_dir):
    
    print("running run_model()")
    weights_file = os.path.join(SAVE_DIR, "phase_two_weights.hdf5")
    model_file = os.path.join(SAVE_DIR, "phase_two_model.hdf5")
	
    
    model.save(model_file)

    # Adadelta is recommended to be used with default values
    opt = Adadelta()

    # parallel_model = ModelMGPU(model, gpus=GPUS)
    parallel_model = model
    parallel_model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])

    X_train = data["train_data"]
    Y_train = data["train_labels"]
    X_validation = data["val_data"]
    Y_validation = data["val_labels"]
    X_test = data["test_data"]
    Y_test = data["test_labels"]
    
    _callbacks = []
    checkpoint = ModelCheckpoint(filepath=weights_file, save_best_only=True)
    _callbacks.append(checkpoint)
    earlystopping = EarlyStopping(monitor="val_loss", patience=15)
    _callbacks.append(earlystopping)

    parallel_model.fit(X_train,
                       Y_train,
                       batch_size=BATCH_SIZE * GPUS,
                       epochs=EPOCH,
                       validation_data=(X_validation, Y_validation),
                       shuffle=True,
                       callbacks=_callbacks)


    Y_pred = parallel_model.predict(X_test)

    auc = metrics.roc_auc_score(Y_test, Y_pred)

    with open(os.path.join(save_dir, "auc.txt"), "w") as of:
        of.write("AUC: %f\n" % auc)

    [fprs, tprs, thrs] = metrics.roc_curve(Y_test, Y_pred[:, 0])

    sort_ix = np.argsort(np.abs(fprs - 0.1))
    fpr10_thr = thrs[sort_ix[0]]

    sort_ix = np.argsort(np.abs(fprs - 0.05))
    fpr5_thr = thrs[sort_ix[0]]

    sort_ix = np.argsort(np.abs(fprs - 0.03))
    fpr3_thr = thrs[sort_ix[0]]

    sort_ix = np.argsort(np.abs(fprs - 0.01))
    fpr1_thr = thrs[sort_ix[0]]

    with open(os.path.join(save_dir, "fpr_threshold_scores.txt"), "w") as of:
        of.write("10 \t %f\n" % fpr10_thr)
        of.write("5 \t %f\n" % fpr5_thr)
        of.write("3 \t %f\n" % fpr3_thr)
        of.write("1 \t %f\n" % fpr1_thr)

    with open(os.path.join(save_dir, "roc_values.txt"), "w") as of:
        of.write("FPR\tTPR\tTHR\n")
        for fpr, tpr, thr in zip(fprs, tprs, thrs):
            of.write("%f\t%f\t%f\n" % (fpr, tpr, thr))

    [pr, rc, thresholds] = metrics.precision_recall_curve(Y_test, Y_pred)
    auprc = metrics.auc(rc, pr)

    with open(os.path.join(SAVE_DIR, "prc.txt"), "w") as of:
        of.write("PRC: %f\n" % auprc)
        
    with open(os.path.join(SAVE_DIR, "prc_values.txt"), "w") as of:
        of.write("PR\tRC\tTHR\n")
        for pr_, rc_, thr in zip(pr, rc, thresholds):
            of.write("%f\t%f\t%f\n" % (pr_, rc_, thr))

###############################################################################################################################################
if __name__ == "__main__":

    create_dataset()   
    train_model()
