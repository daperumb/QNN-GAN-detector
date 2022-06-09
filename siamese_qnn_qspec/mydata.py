from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torch
import os
import random
import cv2
import scipy.io as sio
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import core_qnn.quaternion_layers as qnn
import core_qnn.quaternion_ops as qops


class MyData(Dataset):
    def __init__(self, path, start, end):
        self.path = path
        self.img = os.listdir(path)[start:end]

    def __getitem__(self, index):
        img_path = os.path.join(self.path, self.img[index])
        data = sio.loadmat(img_path)
        data = np.stack(data['qImage'][0][0])
        data = torch.as_tensor(data, dtype=torch.float)
        data = qops.q_normalize(data)
        return data

    def __len__(self):
        return len(self.img)


def make_loader(path, start, end, batch_size=8, shuffle=False, drop_last=True):
    my_set = MyData(path=path, start=start, end=end)
    my_loader = DataLoader(my_set, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return my_loader
