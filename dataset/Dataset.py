import os

import torch
import torchvision.transforms as transforms
from PIL import Image

class MTPDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, mode):
        super().__init__()
        self.data_dir = data_dir
        self.mode = mode
        self.data_path = os.path.join(data_dir,mode)
        self.fn_list = open(os.path.join(self.data_path,'fn_list.txt')).read().split('\n')

        self.transform = transforms.Compose([transforms.Resize(224),transforms.ToTensor(), 
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    def __getitem__(self, idx):
        fn = self.fn_list[idx]
        image = Image.open(os.path.join(self.data_path,'image',fn + '.jpg'))
        image = self.transform(image)
        agent_state_vector = torch.load(os.path.join(self.data_path,'state',fn + '.state'))
        ground_truth = torch.load(os.path.join(self.data_path, 'traj', fn+'.traj'))

        data = {'image' : image, 'agent_state_vector' : agent_state_vector, 'ground_truth' : ground_truth}

        return data

    def __len__(self):
        return len(self.fn_list)