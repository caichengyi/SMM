import os
import numpy as np
import torch
import random


#Edit the Path
data_path = '/data'
results_path = '/data'


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_config(pretrained):
    '''

    Args:
        dataset: string of the dataset name
        pretrained: string of the pretrained model's name

    Returns:
        attribute_layers: the layer number of the attribute network
        epochs: number of training epochs
        lr: learning rate of reprogramming
        attr_lr: learning rate of attribute network
        attr_gamma: weight decay of attribute network
    '''
    epochs = 200
    lr = 0.01

    if pretrained == 'ViT_B32':
        attribute_layers = 6
        attr_lr = 0.001
        attr_gamma = 1
    else:
        attribute_layers = 5
        attr_lr = 0.01
        attr_gamma = 0.1

    return attribute_layers, epochs, lr, attr_lr, attr_gamma