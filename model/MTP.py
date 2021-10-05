import torch
from torch import nn
from torch.nn import functional as f

from model.Backbone import calculate_backbone_feature_dim

ASV_DIM = 3  # velocity, acceleration, and heading change rate

'''
논문에서는 mobilenet v2를 backbone으로 사용하고 mode는 3일때 가장 우수, 6 seonds동안 prediction
10Hz 사용하며 base CNN 이후 fully connected layer의 개수는 4096, 3 X 300 X 300 rasterized image - 0.2m resolution
'''
 

class MTP(nn.Module):
    def __init__(self, Backbone, num_modes, seconds=6, Hz = 10, n_hidden_layers = 4096, input_shape = (3,500,500), training = True):
        self.Backbone = Backbone
        self.num_modes = num_modes
        self.training = training
        Backbone_feature_dim = calculate_backbone_feature_dim(Backbone, input_shape)

        self.fc1 = nn.Linear(Backbone_feature_dim + ASV_DIM, n_hidden_layers)   # CNN 이후의 fully connected layer와 state vector를 concat 후 fc 통과

        H = seconds * Hz

        self.fc2 = nn.Linear(n_hidden_layers, int(num_modes * (2*H + 1)))       # output size = M x (2H + 1) 


    def forward(self, image, agent_state_vector):
        '''
        backbone을 통과한 결과와 state vector를 concat 후 fc layer를 두번 통과한다.
        output은 모드에 따른 x,y 좌표와 해당 모드의 확률이다.
        '''
        Backbone_output = self.Backbone(image)

        concat_output = torch.cat([Backbone_output, agent_state_vector], dim = 1)

        fc1_output = self.fc1(concat_output)
        fc2_output = self.fc2(fc1_output)

        mode_probabilities = fc2_output[:, -self.num_modes:].clone()            # mode의 확률
        traj_output = fc2_output[:, :-self.num_modes]                           # 2H개의 trajectory

        if not self.training:
            mode_probabilities = f.softmax(mode_probabilities, dim=-1)              # cross entropy를 사용하므로, inference시에만 mode 확률 output이 합이 1이 되도록 softmax 통과

        output = torch.cat((traj_output,mode_probabilities), 1)
        
        return output