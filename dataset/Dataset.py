import os

import torch
import torchvision.transforms as transforms

'''
구현중
'''

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, mode):
        super().__init__()
        self.data_dir = data_dir

    def __getitem__(self, idx):
        return 0

    def __len__(self):
        return 0