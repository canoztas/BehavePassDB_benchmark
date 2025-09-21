# For sensors, indicate the sensor as modality. The train_touch_task value does not matter as
# the data for training will be taken from all tasks for the specific sensor

# For touch models, modality must be equal to "touch" and then the specific you must indicate
# the specific "train_touch_task", i.e., 'readtext', 'gallery', 'tap'

# For keystroke, modality = 'touch', train_touch_task = 'keystroke'

modality = 'touch'   # valid for train_background_sensors only
train_touch_task = 'keystroke'  # valid for touch only

if modality[:6] == 'sensor':
    model_name = modality + '_model'
if modality == 'touch':
    model_name = modality + '_' + train_touch_task + '_model'

raw_data_dir = '/content/drive/MyDrive/MobileB2C_Ongoing_BehavePassDB/MobileB2C_BehavePassDB_DevSet_ValSet/'
raw_test_data_dir = '/content/drive/MyDrive/MobileB2C_Ongoing_BehavePassDB/MobileB2C_BehavePassDB_TestSet/'


# Misc parameters
dims_dict = {  # dimensions of the initial signals
    'sensor_acc': 3,
    'touch': 2,
    'sensor_magn': 3,
    'sensor_accl': 3,
    'sensor_gyro': 3,
    'sensor_grav': 3
}

preproc_output_dir = '/content/BehavePassDB_benchmark/preprocessed_data/'
# General
task_index_dict = {
    'keystroke': '1',
    'readtext': '2',
    'gallery': '3',
    'tap': '4',
}

task_color_list = {
    'keystroke': 'orange',
    'readtext': 'blue',
    'gallery': 'm',
    'tap': 'green',
    'mean': 'k',
    'loss': 'red'
}

if modality[:6] == 'sensor':
    used_task_list = ['keystroke', 'readtext', 'gallery', 'tap']
if modality == 'touch':
    used_task_list = [train_touch_task]

full_task_list = ['keystroke', 'readtext', 'gallery', 'tap']
modality_list = ['sensor_acc', 'sensor_grav', 'sensor_gyro', 'sensor_accl', 'sensor_magn', 'touch']
dev_set_session_list = ['g1', 'g2', 'g3', 'g4']

decimals = 4


# Training
dataset_dir = '/content/BehavePassDB_benchmark/preprocessed_data/'
val_label_dir = raw_data_dir + 'ValSet_labels/'
batch_size = 512
epochs = 150
ds_ratio = 1
batches_per_epoch = 100
NUM_SESSIONS_GALLERY = 2
NUM_SESSIONS_USER = 4
num_features = dims_dict[modality]*4
SAVE_NEW_BEST_MODEL = True  # to run training without saving any model, set to False
model_dir = 'models/'
train_info_log_dir = 'logs/'

train_info_log_filename = model_name + '_train_recognition.txt'
loss_log_filename = model_name + '_train_loss.txt'
fig_dir = '/content/BehavePassDB_benchmark/figures/'

# Model parameters
units = 64
sequence_len = 150
keystroke_sequence_len = 50
touch_sequence_len = 100
tap_sequence_len = 20

trained_model_list = ['sensor_acc_model', 'sensor_accl_model', 'sensor_grav_model',
                      'sensor_gyro_model', 'sensor_magn_model', 'touch_gallery_model',
                      'touch_readtext_model', 'touch_tap_model', 'touch_keystroke_model']
best_epochs_list = ['001', '001', '001', '001', '001', '001', '001', '001', '001']
best_epochs = dict(zip(trained_model_list, best_epochs_list))

model_modalities_list = ['sensor_acc', 'sensor_accl', 'sensor_grav',
                      'sensor_gyro', 'sensor_magn', 'touch',
                      'touch', 'touch', 'touch']

model_modalities = dict(zip(trained_model_list, model_modalities_list))

increments = {
    'keystroke': 20,
    'readtext': 10,
    'gallery': 10,
    'tap': 10,
    'sensor_acc': 50,
    'sensor_grav': 50,
    'sensor_gyro': 50,
    'sensor_accl': 50,
    'sensor_magn': 50,
}


# Test
competition_prediction_dir = '/content/BehavePassDB_benchmark/MobileB2C_predictions/'
