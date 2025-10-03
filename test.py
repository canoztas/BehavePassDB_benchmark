import ast
import os
import shutil

import tensorflow
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import itertools


from utils.config import full_task_list
from utils.config import model_dir
from utils.config import raw_test_data_dir
from utils.config import task_index_dict
from utils.config import modality_list
from utils.config import trained_model_list
from utils.config import best_epochs
from utils.config import sequence_len
from utils.config import keystroke_sequence_len
from utils.config import touch_sequence_len
from utils.config import tap_sequence_len
from utils.config import increments

from utils.config import model_modalities
from utils.config import decimals
from utils.config import competition_prediction_dir

from utils.training import cross_task_dataset
from utils.training import add_fft
# from utils.training import downsample
from utils.training import pad_or_slice

from utils.training import compute_eer
from utils.training import compute_auc

from utils.test import create_folder_name

comparisons = {}
for task in full_task_list:
    comparison_file = raw_test_data_dir + 'Comparisons_TestSet_Task{}_{}.txt'.format(task_index_dict[task], task)
    with open(comparison_file) as fp:
        lines = fp.read().splitlines()
        comparisons[task] = [x.split(' ') for x in lines if x]


# # Slightly preprocessing test set

test_datasets = {}
for modality in modality_list:
    test_datasets[modality] = [cross_task_dataset(modality,'enrolment', full_task_list, set_name='Test'), cross_task_dataset(modality, 'verification', full_task_list, set_name='Test')]


models = {}
for model in trained_model_list:
    # models[model] = tensorflow.keras.models.load_model(model_dir + model + '/' + model + '_' + best_epochs[model] + '.h5')
    models[model] = tensorflow.keras.models.load_model(model_dir + 'pretrained/' + model + '.h5')

samples_by_model = {}
samples_ids_by_model = {}
for modality in [x for x in modality_list if 'sensor' in x]:
    samples_by_model[modality] = []
    samples_ids_by_model[modality] = []
    for task in full_task_list:
        for sample_type in [0, 1]:  # 0 for enrolment, 1 for verification
            for session in list(test_datasets[modality][sample_type][task].keys()):
                samples = test_datasets[modality][sample_type][task][session]
                # samples = downsample(samples, ratio=4)
                uncap_num_samples = max(1, int((len(samples)-len(samples) % sequence_len) / increments[modality]))
                num_samples = np.min([50, uncap_num_samples])
                for i in range(num_samples):
                    sample = samples[i*increments[modality]:i*increments[modality]+sequence_len]
                    sample = pad_or_slice(sample, sequence_len, random_offset=False)
                    sample = add_fft(sample, modality)
                    samples_by_model[modality].append(sample)
                    samples_ids_by_model[modality].append([sample_type, task, session, i])



for modality in [x for x in modality_list if 'touch' in x]:
    samples_by_model[modality] = []
    samples_ids_by_model[modality] = []
    for task in [x for x in full_task_list if x != 'keystroke']:
        if task != 'tap':
            sl = touch_sequence_len
        else:
            sl = tap_sequence_len
        for sample_type in [0, 1]:  # 0 for enrolment, 1 for verification
            for session in list(test_datasets[modality][sample_type][task].keys()):
                samples = test_datasets[modality][sample_type][task][session]
                uncap_num_samples = max(1, int((len(samples)-len(samples) % sl) / increments[task]))
                num_samples = np.min([50, uncap_num_samples])
                for i in range(num_samples):
                    sample = samples[i*increments[task]:i*increments[task]+sl]
                    sample = pad_or_slice(sample, sl, random_offset=False)
                    sample = add_fft(sample, modality)
                    samples_by_model[modality].append(sample)
                    samples_ids_by_model[modality].append([sample_type, task, session, i])
    task = 'keystroke'
    for sample_type in [0, 1]:  # 0 for enrolment, 1 for verification
        for session in list(test_datasets[modality][sample_type][task].keys()):
            samples = test_datasets[modality][sample_type][task][session]
            uncap_num_samples = max(1, int((len(samples) - len(samples) % sl) / increments[task]))
            num_samples = np.min([50, uncap_num_samples])
            for i in range(num_samples):
                sample = samples[i * increments[task]:i * increments[task] + sl]
                sample = pad_or_slice(sample, keystroke_sequence_len, random_offset=False)
                samples_by_model[modality].append(sample)
                samples_ids_by_model[modality].append([sample_type, task, session, i])

predictions = {}
for model in [x for x in trained_model_list if 'sensor' in x]:
    print('Predicting:', model)
    tothemodel = np.array(samples_by_model[model_modalities[model]])
    predictions[model] = models[model].predict(tothemodel, batch_size = 2048)

predictions_idxs = {}
for task in full_task_list:
    task_idxs = [samples_ids_by_model['touch'].index(x) for x in samples_ids_by_model['touch'] if x[1] == task]
    predictions_idxs[task] = dict(zip(task_idxs, [x for x in range(len(task_idxs))]))
    tothemodel = np.array([samples_by_model['touch'][i] for i in task_idxs])
    model = 'touch_' + task + '_model'
    print('Predicting:', model)
    predictions[model] = models[model].predict(tothemodel, batch_size = 2048)

np.save('predictions.npy', predictions)
predictions = np.load('predictions.npy', allow_pickle=True).item()


embeddings = {}
for task in list(comparisons.keys()):
    embeddings[task] = {}
    for comparison in list(comparisons[task]):
        e = comparison[0]
        v = comparison[1]
        embeddings[task][e] = {}
        embeddings[task][v] = {}
        for modality in [x for x in modality_list if x != 'touch']:
            idx_e_samples = [x[3] for x in samples_ids_by_model[modality] if x[:3] == [0, task, e]]
            embeddings[task][e][modality] = {}
            for i in idx_e_samples:
                e_idx = samples_ids_by_model[modality].index([0, task, e, i])
                embeddings[task][e][modality][i] = predictions[modality + '_model'][e_idx]
            idx_v_samples = [x[3] for x in samples_ids_by_model[modality] if x[:3] == [1, task, v]]
            embeddings[task][v][modality] = {}
            for i in idx_v_samples:
                v_idx = samples_ids_by_model[modality].index([1, task, v, i])
                embeddings[task][v][modality][i] = predictions[modality + '_model'][v_idx]

            embeddings[task][e][modality], embeddings[task][v][modality] = np.array(list(embeddings[task][e][modality].values())), np.array(list(embeddings[task][v][modality].values()))
        modality = 'touch'
        idx_e_samples = [x[3] for x in samples_ids_by_model[modality] if x[:3] == [0, task, e]]
        embeddings[task][e][modality] = {}
        for i in idx_e_samples:
            e_idx = samples_ids_by_model[modality].index([0, task, e, i])
            embeddings[task][e][modality][i] = predictions[modality + '_' + task + '_model'][predictions_idxs[task][e_idx]]

        idx_v_samples = [x[3] for x in samples_ids_by_model[modality] if x[:3] == [1, task, v]]
        embeddings[task][v][modality] = {}
        for i in idx_v_samples:
            v_idx = samples_ids_by_model[modality].index([1, task, v, i])
            embeddings[task][v][modality][i] = predictions[modality + '_' + task + '_model'][predictions_idxs[task][v_idx]]

        embeddings[task][e][modality], embeddings[task][v][modality] = np.array(list(embeddings[task][e][modality].values())), np.array(list(embeddings[task][v][modality].values()))

np.save('embeddings.npy', embeddings)

embeddings = np.load('embeddings.npy', allow_pickle=True).item()

u_distances = {}
for task in full_task_list:
    u_distances[task] = {}
    for comparison in comparisons[task]:
        u_distances[task][str(comparison)] = {}
        e, v = comparison[0], comparison[1]
        for modality in modality_list:
            u_distances[task][str(comparison)][modality] = np.mean(euclidean_distances(embeddings[task][e][modality], embeddings[task][v][modality]))


combos = [item for items in [list(itertools.combinations(modality_list, x)) for x in range(1,1+len(modality_list))] for item in items]

m_distances = {}
for task in full_task_list:
    m_distances[task] = {}
    for combo in combos:
        spec_modality_list = list(combo)
        m_distances[task][str(spec_modality_list)] = {}
        for comparison in list(u_distances[task].keys()):
            summa = 0
            for modality in spec_modality_list:
                summa = summa + u_distances[task][comparison][modality]
            m_distances[task][str(spec_modality_list)][comparison] = np.round(summa, decimals)

# In case you do not have test set labels (currently not released)
# This part generates the scores as zip files that can be submitted to CodaLab
os.makedirs(competition_prediction_dir, exist_ok=True)
for combo in combos:
    tmp_folder_name = create_folder_name(list(combo))
    tmp_dir_name = competition_prediction_dir + '/{}/'.format(tmp_folder_name)
    os.makedirs(tmp_dir_name, exist_ok=True)

    for task in full_task_list:
        tmp_dist = list(m_distances[task][str(list(combo))].values())
        with open(tmp_dir_name + 'task{}_predictions.txt'.format(task_index_dict[task]), 'w') as f:
            for line in tmp_dist:
                f.write(f"{line}\n")
    shutil.make_archive(competition_prediction_dir + tmp_folder_name, 'zip', tmp_dir_name)
for folder in next(os.walk(competition_prediction_dir))[1]:
    shutil.rmtree(competition_prediction_dir + folder)


# In case you have test set labels (currently not released)
labels = {}
for task in full_task_list:
    labels[task] = {}
    with open(raw_test_data_dir + 'task%d_labels.txt' % int(task_index_dict[task])) as f:
        lines = f.read().splitlines()
    labels[task]['literal'] = lines
    labels[task]['general'] = np.array([0 if x == 'genuine' else 1 for x in lines])
    labels[task]['random'] = np.array([0 if x == 'genuine' else 1 for x in [x for x in lines if x != 'skilled']])
    labels[task]['skilled'] = np.array([0 if x == 'genuine' else 1 for x in [x for x in lines if x != 'random']])

scenarios = ['general', 'random', 'skilled']
eers, aucs = {}, {}
for scenario in scenarios:
    eers[scenario], aucs[scenario] = {}, {}
    for task in full_task_list:
        eers[scenario][task], aucs[scenario][task] = {}, {}
        for combo in m_distances[task]:
            eers[scenario][task][combo], aucs[scenario][task][combo] = {}, {}
            if scenario == 'general':
                scores = np.array(list(m_distances[task][combo].values()))
            if scenario == 'random':
                idxs = np.array([i for i in range(len(labels[task]['literal'])) if labels[task]['literal'][i] != 'skilled'])
                scores = np.array(list(m_distances[task][combo].values()))[idxs]
            if scenario == 'skilled':
                idxs = np.array([i for i in range(len(labels[task]['literal'])) if labels[task]['literal'][i] != 'random'])
                scores = np.array(list(m_distances[task][combo].values()))[idxs]
            eer = np.round(100*compute_eer(labels[task][scenario], scores)[0], decimals)
            auc = np.round(100*compute_auc(labels[task][scenario], scores), decimals)
            eers[scenario][task][combo], aucs[scenario][task][combo] = eer, auc

sorted_eers, sorted_aucs = {}, {}
for scenario in list(eers.keys()):
    sorted_eers[scenario], sorted_aucs[scenario] = {}, {}
    for task in list(eers[scenario].keys()):
        sorted_eers[scenario][task] = {k: v for k, v in sorted(eers[scenario][task].items(), key=lambda item: item[1])}
        sorted_aucs[scenario][task] = {k: v for k, v in sorted(aucs[scenario][task].items(), key=lambda item: -item[1])}



