import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from dataset.Dataset import MTPDataset
from model.MTP import MTP
import model.Backbone
from loss import MTPLoss
from configs import ConfigParameters
import util 

configs = ConfigParameters()

## define train hyperparameters
mode = configs.mode
gpu_id = configs.gpu_id
train_continue = configs.train_continue

lr = configs.lr
batch_size = configs.batch_size
num_epoch = configs.num_epoch

num_modes = configs.num_modes

data_dir = configs.data_dir
ckpt_dir = configs.ckpt_dir
log_dir = configs.log_dir
result_dir = configs.result_dir

if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

device = torch.device('cuda:{}'.format(gpu_id) if torch.cuda.is_available() else 'cpu')

## create dataloader

dataset_train = MTPDataset(data_dir=data_dir, mode = 'train')
dataset_val = MTPDataset(data_dir=data_dir, mode = 'val')

loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False)


## model
Backbone = model.Backbone.MobileNetBackbone()
network = MTP(Backbone, num_modes).to(device)

## loss & optimizer
criterion = MTPLoss(num_modes)
optimizer = optim.Adam(network.parameters(), lr=lr)

# variable setting
num_data_train = len(dataset_train)
num_data_val = len(dataset_val)

num_batch_train = np.ceil(num_data_train / batch_size)
num_batch_val = np.ceil(num_data_val / batch_size)

## Tensorboard
writer_train = SummaryWriter(log_dir=os.path.join(log_dir, 'train'))
writer_val = SummaryWriter(log_dir=os.path.join(log_dir, 'val'))

## extra function
fn_tonumpy = lambda x: x.to('cpu').detach().numpy()


st_epoch = 0


if mode == 'train':
    if train_continue == 'on':
        network, optimizer, st_epoch = util.load(ckpt_dir=ckpt_dir, net=network, optim=optimizer)
        
    ## train
    for epoch in range(st_epoch + 1, num_epoch + 1):
        network.train()
        loss_arr = []

        for batch, data in enumerate(loader_train):

            # forward pass
            img = util.NaN2Zero(data['image']).to(device)
            state = util.NaN2Zero(data['agent_state_vector']).to(device)
            gt = util.NaN2Zero(data['ground_truth']).to(device)

            output = network(img, state)
        
            # backward pass
            optimizer.zero_grad()

            loss = criterion(output, gt.unsqueeze(1))
            loss.backward()

            optimizer.step()

            loss_arr += [loss.item()]
            
            if batch % 10 == 0:
                print("TRAIN: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                        (epoch, num_epoch, batch, num_batch_train, np.mean(loss_arr)))

                # use Tensorboard          
                id = num_batch_train * (epoch - 1) + batch
                writer_train.add_scalars('output',fn_tonumpy(output), id)
                writer_train.add_scalars('gt',fn_tonumpy(gt.unsqueeze(1)), id)

        writer_train.add_scalar('loss', np.mean(loss_arr), epoch)

        ## validation
        with torch.no_grad():
            network.eval()
            loss_arr = []

            for batch, data in enumerate(loader_val):
                img = util.NaN2Zero(data['image']).to(device)
                state = util.NaN2Zero(data['agent_state_vector']).to(device)
                gt = util.NaN2Zero(data['ground_truth']).to(device)

                output = network(img, state)

                loss = criterion(output, gt.unsqueeze(1))
                loss_arr += [loss.item()]

                if batch % 10 == 0:
                    print("VALID: EPOCH %04d / %04d | BATCH %04d / %04d | LOSS %.4f" %
                      (epoch, num_epoch, batch, num_batch_val, np.mean(loss_arr)))
                    writer_val.add_scalars('output',fn_tonumpy(output), id)
                    writer_val.add_scalars('gt',fn_tonumpy(gt.unsqueeze(1)), id)

        writer_val.add_scalar('loss', np.mean(loss_arr), epoch)

        if epoch % 20 == 0:
            util.save(ckpt_dir=ckpt_dir, net=network, optim=optimizer, epoch=epoch)
    
    writer_train.close()
    writer_val.close()