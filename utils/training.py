from tensorflow.keras import backend as K
import numpy as np
import os
import ast
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
import itertools
from sklearn.metrics import roc_curve, auc
from utils.config import dataset_dir
from utils.config import full_task_list, task_index_dict, modality_list
from utils.config import dims_dict
from utils.config import decimals

def remove_unused_modalities(training_dataset: dict, modality: str):
    """
    Removes the modalities that won't be used from the dev set
    :param training_dataset: the dev dataset dict
    :param modality: the modality used
    :return: the training dataset without unnecessary modalities
    """
    unused_modality_list = [x for x in modality_list if x != modality]
    for subject in list(training_dataset.keys()):
        for session in list(training_dataset[subject].keys()):
            for task in full_task_list:
                for unused_modality in unused_modality_list:
                    del training_dataset[subject][session][task][unused_modality]
    return training_dataset

def remove_unused_tasks(training_dataset: dict, task: str):
    """
    Removes the modalities that won't be used from the dev set
    :param training_dataset: the dev dataset dict
    :param task: the task used
    :return: the training dataset without unnecessary modalities
    """
    unused_full_task_list = [x for x in full_task_list if x != task]
    for subject in list(training_dataset.keys()):
        for session in list(training_dataset[subject].keys()):
            for unused_task in unused_full_task_list:
                del training_dataset[subject][session][unused_task]
    return training_dataset


def cross_task_dataset(modality: str, operational_mode: str, task_list: list, set_name: str):
    """
    For a given modality, it returns validation or test set considering all tasks
    :param modality: modality
    :param operational_mode: 'enrolment' or 'verification'
    :param task_list: list of tasks to consider
    :param set_name: can be "Test" or "Val"
    :return: dictionary containing validation data
    """
    by_modality = {}
    for task in task_list:
        by_modality[task] = {}
        task_specific_dataset = np.load(dataset_dir + '{}Set_Task'.format(set_name) + task_index_dict[task] + '_' + task + '_{}_preprocessed_data.npy'.format(operational_mode), allow_pickle=True).item()
        for session in list(task_specific_dataset.keys()):
            by_modality[task][session] = task_specific_dataset[session][task][modality]
    return by_modality



def downsample(sample: np.ndarray, ratio: int):
    """
    :param sample: the biometric sample in its entire length
    :param ratio: the downsampling ratio
    :return: the downsampled sample
    """
    ds_sample = sample[::ratio,:]
    return ds_sample


def pad_or_slice(sample: np.ndarray, sequence_len: int, random_offset=True):
    """
    :param sample: the biometric sample in its entire length
    :param sequence_len: the model input sequence length
    :param random_offset: whether to apply a random offset or not
    :return: a sequence len-long sample, concatenated with zeros if shorter than sequence len,
    or with random initial point if longer than sequence len
    """

    N = len(sample)
    if N <= sequence_len:
        sample = np.concatenate((sample, np.zeros((sequence_len, np.shape(sample)[1]))), axis=0)
        return sample[:sequence_len]
    else:
        if random_offset:
            offset = np.random.randint(N-sequence_len)
        else:
            offset = 0
        return sample[offset:offset+sequence_len]



def add_fft(sample: np.ndarray, modality):
    dims = dims_dict[modality]
    tmp = sample[:, :dims]
    b = np.zeros(np.shape(tmp))
    for axis in range(np.shape(tmp)[1]):
        b[:, axis] = np.abs(np.fft.fft(tmp[:, axis]))
    for axis in range(np.shape(tmp)[1]):
        mean = np.mean(b[:, axis])
        std = np.std(b[:, axis])
        b[:, axis] = b[:, axis]-mean
        b[:, axis] = b[:, axis]/(std + 10E-8)
    b = np.round(b, decimals)
    sample = np.concatenate((sample, b), axis=-1)
    return sample

def compute_eer(labels, scores):
    """
    labels: 1D np.array, 0 = impostor comparison, 1 = genuine comparison
    scores: 1D np.array
    """
    fmr, tmr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnmr = 1 - tmr
    eer_index = np.nanargmin(np.abs((fnmr - fmr)))
    eer_threshold = thresholds[eer_index]
    eer = np.mean((fmr[eer_index], fnmr[eer_index]))
    return eer, eer_threshold


def compute_auc(labels, scores):
    """
    :param labels: 1D np array, 0 = impostor comparison, 1 = genuine comparison
    :param scores: 1D np array, same size as labels
    :return: np float in range 0-1
    """
    fmr, tmr, _ = roc_curve(labels, scores, pos_label=1)
    return auc(fmr, tmr)


def triplet_loss(inputs, alpha=1.5):
    Positive_output, Negative_output, Anchor_output = inputs

    # Pueden ser comparaciones genuinas o impostoras
    distance_p1_up1 = K.square(Anchor_output - Positive_output)
    distance_n1_up1 = K.square(Anchor_output - Negative_output)

    distance_p1_up1 = K.sqrt(K.sum(distance_p1_up1, axis=-1, keepdims=True))
    distance_n1_up1 = K.sqrt(K.sum(distance_n1_up1, axis=-1, keepdims=True))

    loss_1 = K.maximum(0.0, (distance_p1_up1 ** 2) - (
                distance_n1_up1 ** 2) + alpha)
    return K.sum(loss_1)


def prepare_data(data, idxs):
    data_list = []
    for user_key in data.item():
        if user_key in idxs:
            data_list.append(data.item()[user_key])
    return np.asarray(data_list)

