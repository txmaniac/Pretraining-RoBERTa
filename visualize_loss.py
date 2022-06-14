from cProfile import label
from matplotlib.font_manager import json_load
import matplotlib.pyplot as plt
import json
import numpy as np

file = json_load('trainer_state_145.json')

log_contents = file['log_history']

train_loss = []
eval_loss = []

for i,item in enumerate(log_contents):
    if i%2==0:
        train_loss.append(item['loss'])
    else:
        eval_loss.append(item['eval_loss'])

data_dict = {
    'train_loss': np.array(train_loss),
    'valid_loss': np.array(eval_loss)
}

plt.title('Loss plot from training logs')
plt.xlabel('Training steps x1000')
plt.ylabel('Loss')

plt.plot(data_dict['train_loss'], color='r', label='training loss')
plt.plot(data_dict['valid_loss'], color='b', label='validation loss')

plt.legend()
plt.show()
