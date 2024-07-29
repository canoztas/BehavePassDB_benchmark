import matplotlib.pyplot as plt
import numpy as np
import ast
from utils.config import train_info_log_dir
from utils.config import train_info_log_filename
from utils.config import loss_log_filename
from utils.config import model_name
from utils.config import used_task_list
from utils.config import task_color_list
from utils.config import fig_dir

import os

os.makedirs(fig_dir, exist_ok=True)

fig, ax = plt.subplots(2,1,figsize=(15, 10))

with open(train_info_log_dir + train_info_log_filename, 'r') as f:
    res = ast.literal_eval(f.read())
res = np.array(res)

with open(train_info_log_dir + loss_log_filename, 'r') as f:
    loss = ast.literal_eval(f.read())
training_loss = np.array(loss)[0]
validation_loss = np.array(loss)[1]
num_epochs = len(training_loss)

for idx in range(len(used_task_list)):
    ax[0].plot(res[:, 2, idx], color=task_color_list[used_task_list[idx]], linewidth=1.5, linestyle = 'dotted')
    ax[0].plot(res[:, 0, idx], label = 'Training Set ' + used_task_list[idx], color=task_color_list[used_task_list[idx]], linewidth=1.5)
mean_res = np.mean(res, axis=2)
ax[0].plot(mean_res[:, 2], linewidth=1.5, color=task_color_list['mean'], linestyle = 'dotted')
ax[0].plot(mean_res[:, 0], label = 'Training Set Mean', linewidth=1.5, color=task_color_list['mean'])


ax[0].set_ylabel('EER [%]', fontsize=20)
ax[0].grid()
# ax[0].set_ylim([40,55])
ax[0].set_xticks(np.arange(num_epochs), labels=[str(x) for x in range(1, num_epochs+1)])
ax[0].legend(fontsize=10)
ax[0].set_title('Task: {}. Solid line: training set. Dotted line: validation set'.format(model_name))



ax[1].plot(training_loss, label = 'Training Set Loss', linewidth=1.5, color=task_color_list['loss'])
ax[1].plot(validation_loss, label = 'Validation Set Loss', linewidth=1.5, color=task_color_list['loss'], linestyle='dotted')
ax[1].set_ylabel('Loss', fontsize=20)
ax[1].set_xlabel('Epochs', fontsize=20)
ax[1].grid()
ax[1].set_xticks(np.arange(num_epochs), labels=[str(x) for x in range(1, num_epochs+1)])
ax[1].legend(fontsize=10)


fig.savefig(fig_dir + model_name + '_training_summary.pdf', dpi=300)
# fig.show()

