import numpy as np
import json, os
from utils.config import raw_data_dir
from utils.config import raw_test_data_dir
from utils.config import dev_set_session_list
from utils.config import modality_list
from utils.config import full_task_list
from utils.config import task_index_dict
from utils.config import preproc_output_dir
from utils.config import decimals


def downsample(array):
    """
    This operation is carried out to avoid having data captured at different acquisition frequencies
    :param array: background data sample passed by the preprocess_data function
    :return: downsampled data depending on initial frequency
    """
    intervals = []
    for i in range(np.shape(array)[0]-1):
        intervals.append(array[i+1][0]-array[i][0])
    instant_freq = np.array([1E9/x for x in intervals])
    instant_freq = instant_freq[(instant_freq <= 300.0)]
    freq = np.mean(instant_freq)
    if freq < 75.0:
       ratio = 1
    if freq >= 75.0 and freq < 150:
       ratio = 2
    if freq >= 150.0:
       ratio = 4
    array = array[::ratio]
    return array


def preprocess_sample(array, task, modality):
    """
    :param array: the modality-, task-, session-, subject- specific np 2D array of data (sample)
    :param task: depending on the task data are processed differently
    :param modality: depending on the modality data are processed differently
    :return: preprocessed sample
    """
    if modality != 'touch':  # this means that it is background sensors
        """
        In this case we concatenate first- and second-order derivatives to the x, y, z normalized data
        """
        columns = 3
        # array = downsample(array)
        # add first derivative
        for i in range(1, columns+1):
            first_derivative_i = np.reshape(np.gradient(array[:, i]), (np.shape(np.gradient(array[:, i]))[0], 1))
            array = np.concatenate((array,first_derivative_i), axis=1)
        # add second derivative
        for ii in range(columns+1,2*columns+1):
            second_derivative_i = np.reshape(np.gradient(array[:, ii]), (np.shape(np.gradient(array[:, ii]))[0], 1))
            array = np.concatenate((array, second_derivative_i), axis=1)
        # normalize
        for i in range(1, 3*columns+1):
            mean = np.mean(array[:, i])
            std = np.std(array[:, i])
            array[:, i] = (array[:, i]-mean)/std
        array = np.round(array[:, 1:], decimals)
        return array
    if modality == 'touch' and task == 'keystroke':  # keystroke data
        """
        In this case we compute normalized ASCII code and inter-press time
        """
        array_2 = np.reshape(np.array([float(x) / 255 for x in array[:, 1]])[:-1], (np.shape(array[:, 1])[0] - 1, 1))
        interpress = np.zeros((np.shape(array[:, 0])[0] - 1, 1))
        for i in range(np.shape(array[:, 0])[0] - 1):
            interpress[i] = array[i + 1, 0] - array[i, 0]
        interpress = interpress / 1E9
        array_2 = np.round(np.concatenate((interpress, array_2), axis=1), decimals)
        return array_2
    if modality == 'touch' and task != 'keystroke':  # touchscreen gestures apart from keystroke data
        """
        In this case we concatenate first- and second-order derivatives to the x, y normalized screen coordinate data
        """
        columns = 2
        # normalize
        array = np.delete(array, 3, 1)
        # add first derivative
        for i in range(1, columns + 1):
            first_derivative_i = np.reshape(np.gradient(array[:, i]), (np.shape(np.gradient(array[:, i]))[0], 1))
            array = np.concatenate((array, first_derivative_i), axis=1)
        # add second derivative
        for ii in range(columns + 1, 2 * columns + 1):
            second_derivative_i = np.reshape(np.gradient(array[:, ii]), (np.shape(np.gradient(array[:, ii]))[0], 1))
            array = np.concatenate((array, second_derivative_i), axis=1)
        for i in range(1, columns + 1):
            mean = np.mean(array[:, i])
            std = np.std(array[:, i])
            array[:, i] = (array[:, i] - mean) / std
        array = np.round(array[:, 1:], decimals)
        return array


def preprocess(dataset_name):
    """
    Takes the initial competition data and transforms them into a format for training the models
    Saves them into newly created directory: 'preprocessed_data/'
    :param dataset_name: can be 'DevSet' or 'ValSet'
    :return: 0
    """
    assert dataset_name == 'DevSet' or dataset_name == 'ValSet' or dataset_name == 'TestSet', "The dataset name needs to be 'DevSet' or 'ValSet' or 'TestSet'"
    if dataset_name == 'DevSet':
        filename = dataset_name
        with open(raw_data_dir + '{}.json'.format(filename)) as json_file:
            dataset = json.load(json_file)
        del json_file

        subject_list = list(dataset.keys())
        # Preprocessing data
        i = 0  # to measure advancement state in iterations
        L = len(subject_list)*len(dev_set_session_list)*len(full_task_list)*len(modality_list)  # total number of iterations
        preproc_set = {}
        for subject in subject_list:
            preproc_set[subject] = {}
            for session in dev_set_session_list:
                preproc_set[subject][session] = {}
                for task in full_task_list:
                    preproc_set[subject][session][task] = {}
                    for modality in modality_list:
                        preproc_set[subject][session][task][modality] = {}
                        print(dataset_name + ": preprocessing data: %.2f%%" % (100 * i / L), end='\r')
                        preproc_set[subject][session][task][modality] = preprocess_sample(np.array(dataset[subject][session][task][modality]), task, modality)
                        i = i + 1

        os.makedirs(preproc_output_dir, exist_ok=True)
        saving_dir = preproc_output_dir + '{}_preprocessed_data.npy'.format(dataset_name)
        np.save(saving_dir, preproc_set)

    if dataset_name == 'ValSet':
        sample_type_list = ['enrolment', 'verification']
        i = 0  # to measure advancement state in iterations
        for sample_type in sample_type_list:
            for task in full_task_list:
                filename = dataset_name + '_Task' + task_index_dict[task] + '_' + task + '_' + sample_type
                with open(raw_data_dir + '{}.json'.format(filename)) as json_file:
                    dataset = json.load(json_file)
                del json_file

                session_list = list(dataset.keys())

                # Preprocessing data
                L = len(sample_type_list) * len(session_list) * len(full_task_list) * len(modality_list)  # total number of iterations
                preproc_set = {}
                for session in session_list:
                    preproc_set[session] = {}
                    preproc_set[session][task] = {}
                    for modality in modality_list:
                        preproc_set[session][task][modality] = {}
                        print(dataset_name + ": preprocessing data: %.2f%%" % (100 * i / L), end='\r')
                        preproc_set[session][task][modality] = preprocess_sample(
                            np.array(dataset[session][task][modality]), task, modality)
                        i = i + 1

                os.makedirs(preproc_output_dir, exist_ok=True)
                saving_dir = preproc_output_dir + filename + '_preprocessed_data.npy'
                np.save(saving_dir, preproc_set)

    if dataset_name == 'TestSet':
        sample_type_list = ['enrolment', 'verification']
        i = 0  # to measure advancement state in iterations
        for sample_type in sample_type_list:
            for task in full_task_list:
                filename = dataset_name + '_Task' + task_index_dict[task] + '_' + task + '_' + sample_type
                with open(raw_test_data_dir + '{}.json'.format(filename)) as json_file:
                    dataset = json.load(json_file)
                del json_file

                session_list = list(dataset.keys())

                # Preprocessing data
                L = len(sample_type_list) * len(session_list) * len(full_task_list) * len(modality_list)  # total number of iterations
                preproc_set = {}
                for session in session_list:
                    preproc_set[session] = {}
                    preproc_set[session][task] = {}
                    for modality in modality_list:
                        preproc_set[session][task][modality] = {}
                        print(dataset_name + ": preprocessing data: %.2f%%" % (100 * i / L), end='\r')
                        preproc_set[session][task][modality] = preprocess_sample(
                            np.array(dataset[session][task][modality]), task, modality)
                        i = i + 1

                os.makedirs(preproc_output_dir, exist_ok=True)
                saving_dir = preproc_output_dir + filename + '_preprocessed_data.npy'
                np.save(saving_dir, preproc_set)
    return 0
