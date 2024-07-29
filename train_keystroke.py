import itertools
from utils.training import cross_task_dataset, compute_eer, compute_auc, triplet_loss, remove_unused_tasks, pad_or_slice, remove_unused_modalities
import random, os
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
import time
import tensorflow
from tensorflow.keras import backend
from tensorflow.keras import optimizers
from tensorflow.keras import Model
from tensorflow.keras.layers import Dropout, Input, LSTM, BatchNormalization, Masking
import os
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

from utils.config import model_dir
from utils.config import train_info_log_dir
from utils.config import train_info_log_filename
from utils.config import dataset_dir
from utils.config import batch_size
from utils.config import keystroke_sequence_len
from utils.config import used_task_list
from utils.config import dev_set_session_list
from utils.config import task_index_dict
from utils.config import val_label_dir
from utils.config import batches_per_epoch
from utils.config import units
from utils.config import decimals
from utils.config import raw_data_dir
from utils.config import epochs
from utils.config import loss_log_filename
from utils.config import model_name
from utils.config import modality



os.makedirs(model_dir + model_name + '/', exist_ok=True)
os.makedirs(train_info_log_dir, exist_ok=True)

num_features = 2

training_dataset = np.load(dataset_dir + 'DevSet_preprocessed_data.npy', allow_pickle=True).item()
training_dataset = remove_unused_modalities(training_dataset, modality)
training_dataset = remove_unused_tasks(training_dataset, used_task_list[0])

# Slightly preprocessing validation set
val_datasets = [cross_task_dataset(modality,'enrolment', used_task_list, set_name='Val'), cross_task_dataset(modality, 'verification', used_task_list, set_name='Val')]
val_label_dict, val_comp_dict = {}, {}
for task in used_task_list:
    val_label_file = val_label_dir + 'task{}_labels.txt'.format(task_index_dict[task])
    with open(val_label_file) as fp:
        val_label_dict[task] = fp.read().split('\n')
        val_label_dict[task] = val_label_dict[task][:-1]
        val_label_dict[task] = [0 if x == 'genuine' else 1 for x in val_label_dict[task]]
    val_label_dict[task] = np.ravel(val_label_dict[task])
for task in used_task_list:
    val_comparison_file = raw_data_dir + 'Comparisons_ValSet_Task{}_{}_updated.txt'.format(task_index_dict[task], task)
    with open(val_comparison_file) as fp:
        val_comp_dict[task] = [x.split(' ') for x in fp.read().split('\n')]
    val_comp_dict[task] = val_comp_dict[task][:-1]


def training_data_generator_triplets(dataset: dict):
    '''
    :param dataset: the training dataset with only the specific sensor
    :yield: the model input
    '''

    # Create empty arrays to contain batch of features and labels#
    batch_features_positive = np.zeros((batch_size, keystroke_sequence_len, num_features))
    batch_features_anchor = np.zeros((batch_size, keystroke_sequence_len, num_features))
    batch_features_negative = np.zeros((batch_size, keystroke_sequence_len, num_features))
    subject_list = list(dataset.keys())
    while True:
        for i in range(batch_size):
            task = random.choice(used_task_list)  # randomly select task to consider for specific sample
            genuine_subject = random.choice(subject_list)
            genuine_session_idx = random.choice(dev_set_session_list)
            anchor_session_idx = random.choice(dev_set_session_list)
            while genuine_session_idx == anchor_session_idx:  # make sure session is not the same as genuine
                anchor_session_idx = random.choice(dev_set_session_list)

            impostor_subject = random.choice(subject_list)
            while genuine_subject == impostor_subject:  # make sure subject is not the same as genuine
                impostor_subject = random.choice(subject_list)
            impostor_session_idx = random.choice(dev_set_session_list)

            # Pad or slice to obtain constant length
            genuine_sample = pad_or_slice(dataset[genuine_subject][genuine_session_idx][task][modality], keystroke_sequence_len)
            anchor_sample = pad_or_slice(dataset[genuine_subject][anchor_session_idx][task][modality], keystroke_sequence_len)
            impostor_sample = pad_or_slice(dataset[impostor_subject][impostor_session_idx][task][modality], keystroke_sequence_len)

            batch_features_positive[i] = genuine_sample
            batch_features_anchor[i] = anchor_sample
            batch_features_negative[i] = impostor_sample

        yield ({'Positive_input': batch_features_positive,
                'Negative_input': batch_features_negative,
                'Anchor_input': batch_features_anchor})



def val_data_generator_triplets(datasets: list, comparisons: dict, labels: dict):
    '''
    :param datasets: validation datasets made of enrolment and probe samples for each task
    :param comparisons: the list of comparisons for each task
    :param labels: the labels for each task
    :yield: the model input for validation
    '''

    # Initialize empty arrays to contain batch of features
    batch_features_positive = np.zeros((batch_size, keystroke_sequence_len, num_features))
    batch_features_anchor = np.zeros((batch_size, keystroke_sequence_len, num_features))
    batch_features_negative = np.zeros((batch_size, keystroke_sequence_len, num_features))

    while True:
        for i in range(batch_size):
            task = random.choice(used_task_list)
            comparison_idx = random.choice(np.arange(len(labels[task])))
            while labels[task][comparison_idx] == 1:
                comparison_idx = random.choice(np.arange(len(labels[task])))
            genuine_session_idx = comparisons[task][comparison_idx][0]
            anchor_session_idx = comparisons[task][comparison_idx][1]
            impostor_comparison_idxs = [comparisons[task].index(x) for x in list(comparisons[task]) if (x[0] == genuine_session_idx or x[1] == genuine_session_idx) and comparisons[task].index(x) != comparison_idx]
            impostor_comparison_idxs = [x for x in impostor_comparison_idxs if labels[task][x] == 1]
            impostor_comparison_idx = random.choice(impostor_comparison_idxs)
            impostor_session_idx = [x for x in comparisons[task][impostor_comparison_idx] if x != genuine_session_idx][0]

            genuine_sample = pad_or_slice(datasets[0][task][genuine_session_idx], keystroke_sequence_len)
            anchor_sample = pad_or_slice(datasets[1][task][anchor_session_idx], keystroke_sequence_len)
            impostor_sample = pad_or_slice(datasets[1][task][impostor_session_idx], keystroke_sequence_len)

            batch_features_positive[i] = genuine_sample
            batch_features_anchor[i] = anchor_sample
            batch_features_negative[i] = impostor_sample

        yield ({'Positive_input': batch_features_positive,
                'Negative_input': batch_features_negative,
                'Anchor_input': batch_features_anchor})


log_list, loss_list = [], []

class Predictor_verification(tensorflow.keras.callbacks.Callback):
    def __init__(self, rnn, val_dataset: list, training_dataset: dict, comparisons: dict):
        '''
        :param rnn: the model (tensorflow.keras.engine.functional.Functional)
        :param val_dataset: validation datasets made of enrolment and probe samples for each task
        :param training_dataset: the training dataset (development set)
        :param comparisons: the list of comparisons for each task
        '''
        self.single_model = rnn
        self.val_dataset_enrolment = val_dataset[0]
        self.val_dataset_verification = val_dataset[1]
        self.training_dataset = training_dataset
        self.val_comp_dict = comparisons

        self.val_label_dict = {}
        self.num_val_rep = 100
        for task in used_task_list:
            val_label_file = val_label_dir + 'task{}_labels.txt'.format(task_index_dict[task])
            with open(val_label_file) as fp:
                self.val_label_dict[task] = fp.read().split('\n')
                self.val_label_dict[task] = self.val_label_dict[task][:-1]
                # self.val_label_dict[task] = [[0 for y in range(self.num_val_rep)] if x == 'genuine' else [1 for y in range(self.num_val_rep)] for x in self.val_label_dict[task]]
                self.val_label_dict[task]=  [0 if x == 'genuine' else 1 for x in self.val_label_dict[task]]
            self.val_label_dict[task] = np.ravel(self.val_label_dict[task])

        self.val_EER, self.val_AUC = {}, {}
        for task in used_task_list:
            self.val_EER[task], self.val_AUC[task] = [], []
        self.best_val_AUC = 0
        self.dev_set_enrolment_session_list = list(itertools.product(list(training_dataset.keys()), dev_set_session_list[:2]))
        self.dev_set_verification_session_list = list(itertools.product(list(training_dataset.keys()), dev_set_session_list[2:]))

        self.num_rep = 10

        self.train_EER, self.train_AUC = {}, {}
        for task in used_task_list:
            self.train_EER[task], self.train_AUC[task] = [], []


    def on_epoch_end(self, epoch, x):
        '''
        :param epoch:
        :param x:
        :return:
        '''
        # Competition validation set part
        for task in used_task_list:
            sample_list = [[], []]
            for comp in self.val_comp_dict[task]:
                for n in range(self.num_val_rep):
                    enrolment_session, verification_session = self.val_dataset_enrolment[task][comp[0]], self.val_dataset_verification[task][comp[1]]
                    enrolment_sample, verification_sample = pad_or_slice(enrolment_session, keystroke_sequence_len), pad_or_slice(verification_session, keystroke_sequence_len)
                    sample_list[0].append(enrolment_sample)
                    sample_list[1].append(verification_sample)
            enrolment_samples, verification_samples = np.array(sample_list[0]), np.array(sample_list[1])
            enrolment_embeddings, verification_embeddings = self.single_model.predict(enrolment_samples, batch_size=len(enrolment_samples), verbose=0), self.single_model.predict(verification_samples, batch_size=len(verification_samples), verbose=0)

            val_scores = np.sqrt(np.add.reduce(np.square(enrolment_embeddings - verification_embeddings), 1))
            val_scores_mean = np.mean(np.reshape(val_scores, (len(self.val_comp_dict[task]), self.num_val_rep)), axis=1)

            self.val_EER[task].append(np.round(100*compute_eer(self.val_label_dict[task], val_scores_mean)[0], decimals))
            self.val_AUC[task].append(np.round(100*compute_auc(self.val_label_dict[task], val_scores_mean), decimals))
        last_epoch_mean_AUC = np.mean([self.val_AUC[task][-1] for task in used_task_list])
        if last_epoch_mean_AUC >= self.best_val_AUC:
            self.best_val_AUC = last_epoch_mean_AUC
            # for file in os.listdir(model_dir):
            #     if file[:-6] == model_name:
            #         os.remove(model_dir + file)
        epoch_idx = "00" + str(int(epoch) + 1)
        epoch_idx = epoch_idx[-3:]
        saving_name = model_name + '_' + epoch_idx
        self.single_model.save(model_dir + model_name + '/' + saving_name + '.h5')
        print('\nVal Set - Epoch {} - EER by task (%): '.format(str(epoch+1)) + ''.join([task + ' ' + str(self.val_EER[task][-1]) + '\t' for task in used_task_list]) + '- AUC by task (%): '.format(str(epoch+1)) + ''.join([task + ' ' + str(self.val_AUC[task][-1]) + '\t' for task in used_task_list]))

        # Training set part

        for task in used_task_list:
            multiple_enrolment_samples, multiple_verification_samples = [], []
            for n in range(self.num_rep):
                enrolment_sample_dict, verification_sample_dict = {}, {}
                for item in self.dev_set_enrolment_session_list:
                    enrolment_sample_dict[item] = pad_or_slice(self.training_dataset[item[0]][item[1]][task][modality], keystroke_sequence_len)
                for item in self.dev_set_verification_session_list:
                    verification_sample_dict[item] = pad_or_slice(self.training_dataset[item[0]][item[1]][task][modality], keystroke_sequence_len)
                enrolment_samples, verification_samples = np.array(list(enrolment_sample_dict.values())), np.array(list(verification_sample_dict.values()))
                multiple_enrolment_samples.append(enrolment_samples)
                multiple_verification_samples.append(verification_samples)
            multiple_enrolment_samples = np.array(multiple_enrolment_samples)
            multiple_enrolment_samples = np.reshape(multiple_enrolment_samples, (np.shape(multiple_enrolment_samples)[0]*np.shape(multiple_enrolment_samples)[1], np.shape(multiple_enrolment_samples)[2], np.shape(multiple_enrolment_samples)[3]))
            multiple_verification_samples = np.array(multiple_verification_samples)
            multiple_verification_samples = np.reshape(multiple_verification_samples, (np.shape(multiple_verification_samples)[0]*np.shape(multiple_verification_samples)[1], np.shape(multiple_verification_samples)[2], np.shape(multiple_verification_samples)[3]))
            enrolment_embeddings, verification_embeddings = self.single_model.predict(multiple_enrolment_samples, batch_size=len(multiple_enrolment_samples), verbose=0), self.single_model.predict(multiple_verification_samples, batch_size=len(multiple_verification_samples), verbose=0)
            distr_g, distr_i = [], []
            for n in range(self.num_rep):
                enrolment_embedding_dict, verification_embedding_dict = {}, {}
                for i in range(len(list(enrolment_sample_dict.keys()))):
                    enrolment_embedding_dict[self.dev_set_enrolment_session_list[i]] = enrolment_embeddings[i+n*len(list(enrolment_sample_dict.keys()))]
                    verification_embedding_dict[self.dev_set_verification_session_list[i]] = verification_embeddings[i+n*len(list(enrolment_sample_dict.keys()))]
                for subject in list(training_dataset.keys()):
                    enrolment_samples = np.array([enrolment_embedding_dict[(subject, dev_set_session_list[0])], enrolment_embedding_dict[(subject, dev_set_session_list[1])]])
                    verification_samples_g = np.array([verification_embedding_dict[(subject, dev_set_session_list[2])], verification_embedding_dict[(subject, dev_set_session_list[3])]])
                    distr_g.append(np.mean(euclidean_distances(enrolment_samples, verification_samples_g), axis=0))
                    for impostor in list(training_dataset.keys()):
                        if impostor != subject:
                            verification_samples_i = np.array([verification_embedding_dict[(impostor, dev_set_session_list[2])], verification_embedding_dict[(impostor, dev_set_session_list[3])]])
                            distr_i.append(np.mean(euclidean_distances(enrolment_samples, verification_samples_i), axis=0))
            distr_g, distr_i = np.ravel(distr_g), np.ravel(distr_i)
            scores = np.concatenate((distr_g, distr_i))
            labels = np.array([0 for x in distr_g] + [1 for x in distr_i])
            eer, auc = np.round(compute_eer(labels, scores)[0]*100, decimals), np.round(compute_auc(labels, scores)*100, decimals)
            self.train_EER[task].append(eer)
            self.train_AUC[task].append(auc)

        print('Train Set - Epoch {} - EER by task (%): '.format(str(epoch+1)) + ''.join([task + ' ' + str(self.train_EER[task][-1]) + '\t' for task in used_task_list]) + '- AUC by task (%): '.format(str(epoch+1)) + ''.join([task + ' ' + str(self.train_AUC[task][-1]) + '\t' for task in used_task_list]))

        # save files with stats
        log_list.append([[self.train_EER[task][-1] for task in used_task_list],
                    [self.train_AUC[task][-1] for task in used_task_list], [self.val_EER[task][-1] for task in used_task_list],
                    [self.val_AUC[task][-1] for task in used_task_list]])
        with open(train_info_log_dir + train_info_log_filename, "w") as output:
            output.write(str(log_list))

        if epoch > 0:
            loss_list = [triplet_model.history.history['loss'], triplet_model.history.history['val_loss']]
            with open(train_info_log_dir + loss_log_filename, "w") as output:
                output.write(str(loss_list))

# Defining the model
X_input = Input(shape=(keystroke_sequence_len, num_features), dtype='float32')
X = Masking(mask_value=0.0)(X_input)
X = BatchNormalization()(X)
X = LSTM(units=units, recurrent_dropout=0.2, return_sequences=True)(X)
X = Dropout(0.5)(X)
X = BatchNormalization()(X)
X = LSTM(units=units, recurrent_dropout=0.2, return_sequences=False)(X)
rnn = Model(X_input, X, name='basic')

# Making it triple for triplet loss
Positive_input = Input(shape=(keystroke_sequence_len, num_features), name='Positive_input')
Negative_input = Input(shape=(keystroke_sequence_len, num_features), name='Negative_input')
Anchor_input = Input(shape=(keystroke_sequence_len, num_features), name='Anchor_input')
Positive_output = rnn(Positive_input)
Negative_output = rnn(Negative_input)
Anchor_output = rnn(Anchor_input)
inputs = [Positive_input, Negative_input, Anchor_input]
outputs = [Positive_output, Negative_output, Anchor_output]
triplet_model = Model(inputs, outputs)
triplet_model.add_loss(backend.sum(triplet_loss(outputs)))

opt = optimizers.Adam(learning_rate=0.005, beta_1=0.9, beta_2=0.999, epsilon=10 ** -8, decay=0.0)
triplet_model.compile(loss=None, optimizer=opt)


predictor = Predictor_verification(rnn, val_datasets, training_dataset, val_comp_dict)

start = time.time()
# Launching training
history_rnn = triplet_model.fit(training_data_generator_triplets(training_dataset),
                                                steps_per_epoch=batches_per_epoch,
                                                epochs=epochs,
                                                verbose=1,
                                                callbacks=[predictor],
                                                validation_data=val_data_generator_triplets(val_datasets, val_comp_dict, val_label_dict),
                                                validation_steps=batches_per_epoch)
end = time.time()

# Saving final log
loss_list = [triplet_model.history.history['loss'], triplet_model.history.history['val_loss']]
with open(train_info_log_dir + loss_log_filename, "w") as output:
    output.write(str(loss_list))
print("Training time [minutes]: %.2f" % ((end-start)/60))


