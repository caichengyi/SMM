import os
from torch.utils.data import DataLoader
from torchvision import datasets

from .const import GTSRB_LABEL_MAP


def refine_classnames(class_names):
    for i, class_name in enumerate(class_names):
        class_names[i] = class_name.lower().replace('_', ' ').replace('-', ' ')
    return class_names

def prepare_additive_data(dataset, data_path, preprocess, test_process=None, batch_size=256, shuffle=True):
    data_path = os.path.join(data_path, dataset)
    if not test_process:
        test_process = preprocess
    if dataset == "cifar10":
        train_data = datasets.CIFAR10(root = data_path, train = True, download = False, transform = preprocess)
        test_data = datasets.CIFAR10(root = data_path, train = False, download = False, transform = test_process)
        class_names = refine_classnames(test_data.classes)
        loaders = {
            'train': DataLoader(train_data, batch_size, shuffle = shuffle, num_workers=8),
            'test': DataLoader(test_data, batch_size, shuffle = False, num_workers=8),
        }
    elif dataset == "cifar100":
        train_data = datasets.CIFAR100(root = data_path, train = True, download = False, transform = preprocess)
        test_data = datasets.CIFAR100(root = data_path, train = False, download = False, transform = test_process)
        class_names = refine_classnames(test_data.classes)
        loaders = {
            'train': DataLoader(train_data, batch_size, shuffle = shuffle, num_workers=8),
            'test': DataLoader(test_data, batch_size, shuffle = False, num_workers=8),
        }
    elif dataset == "svhn":
        train_data = datasets.SVHN(root = data_path, split="train", download = False, transform = preprocess)
        test_data = datasets.SVHN(root = data_path, split="test", download = False, transform = test_process)
        class_names = [f'{i}' for i in range(10)]
        loaders = {
            'train': DataLoader(train_data, batch_size, shuffle = shuffle, num_workers=8),
            'test': DataLoader(test_data, batch_size, shuffle = False, num_workers=8),
        }
    elif dataset == "gtsrb":
        train_data = datasets.GTSRB(root = data_path, split="train", download = True, transform = preprocess)
        test_data = datasets.GTSRB(root = data_path, split="test", download = True, transform = test_process)
        class_names = refine_classnames(list(GTSRB_LABEL_MAP.values()))
        loaders = {
            'train': DataLoader(train_data, batch_size, shuffle = shuffle, num_workers=8),
            'test': DataLoader(test_data, batch_size, shuffle = False, num_workers=8),
        }
    else:
        raise NotImplementedError(f"{dataset} not supported")

    return loaders, class_names
