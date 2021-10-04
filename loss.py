import random
import math

import torch
from torch import nn
from torch.nn import functional as f


class MTPLoss:
    def __init__(self, num_modes, alpha = 1., angle_threshold = 5.):
        '''
        논문에서 실험적으로 alpha = 1, angle_threshold = 5degree를 사용. 
        5 degree 차이 이내면 gt mode로 선정
        '''
        self.num_modes = num_modes
        self.alpha = alpha
        self.angle_threshold = angle_threshold


    # output을 trajectory와 mode로 나눠서 return
    def _get_trajectory_and_modes(self, output):
        mode_output = output[:, -self.num_modes:].clone()

        trajectory_shape = (output.shape[0], self.num_modes, -1, 2)                             #[batch_size, num_modes, num_timesteps, 2]
        trajectory_output = output[:, :-self.num_modes].clone().reshape(trajectory_shape)       #[batch_size, num_modes]

        return trajectory_output, mode_output


    @staticmethod
    # 두 trajectory 사이의 각도를 return
    def _angle_between(ref_traj, traj_to_compare):                                                         

        EPSILON = 1e-5
        
        # 두 trajectory중 하나라도 nan이 존재하면 180 - E return
        if torch.isnan(traj_to_compare[-1]).any() or torch.isnan(ref_traj[-1]).any():  
            return 180. - EPSILON

        traj_norms_product = float(torch.norm(ref_traj[-1]) * torch.norm(traj_to_compare[-1]))

        # |ref|*|compare| 이 0에 가까우면 0 return
        if math.isclose(traj_norms_product, 0):
            return 0.

        dot_product = float(ref_traj[-1].dot(traj_to_compare[-1]))
        angle = math.degrees(math.acos(max(min(dot_product / traj_norms_product, 1), -1)))              #angle = acos((v1*v2)/(|v1||v2|))

        if angle >= 180:
            return angle - EPSILON

        return angle

    @staticmethod
    def _compute_ave_l2_norms(tensor):                                                           
        l2_norms = torch.norm(tensor, p=2, dim=2)
        avg_distance = torch.mean(l2_norms)
        return avg_distance.item()

    # 모드에 따라 gt와의 각도 계산후  [[각도, 모드],..] return
    def _compute_angles_from_ground_truth(self, target, trajectory_output):
        angles_from_ground_truth = []
        for mode, mode_trajectory in enumerate(trajectory_output):

            angle = self._angle_between(target[0], mode_trajectory)

            angles_from_ground_truth.append((angle, mode))
        return angles_from_ground_truth                                                                            


    # threshold보다 작은 angle을 가진 모드의 trajectory 중 가장 적은 l2 norm을 가지고 있는 모드를 gt 모드로 선정
    def _compute_best_mode(self, angles_from_ground_truth, target, trajectory_output) -> int:
        angles_from_ground_truth = sorted(angles_from_ground_truth)
        max_angle_below_thresh_idx = -1

        for angle_idx, (angle, mode) in enumerate(angles_from_ground_truth):
            if angle <= self.angle_threshold:                                                                       
                max_angle_below_thresh_idx = angle_idx
            else:
                break

        if max_angle_below_thresh_idx == -1:
            best_mode = random.randint(0, self.num_modes - 1)                                                       


        else:
            distances_from_ground_truth = []

            for angle, mode in angles_from_ground_truth[:max_angle_below_thresh_idx + 1]:
                norm = self._compute_ave_l2_norms(target - trajectory_output[mode, :, :])

                distances_from_ground_truth.append((norm, mode))

            distances_from_ground_truth = sorted(distances_from_ground_truth)
            best_mode = distances_from_ground_truth[0][1]                                                          

        return best_mode


    def __call__(self, predictions, targets):
   
        batch_losses = torch.Tensor().requires_grad_(True).to(predictions.device)          
        trajectory_output, mode_output = self._get_trajectory_and_modes(predictions)                                            # [batch_size, n_modes, n_timesteps, 2] [batch_size, num_modes]

        for batch_idx in range(predictions.shape[0]):

            ## gt와 prediction의 angle을 계산한 후 가장 적합한 mode를 gt로 선정
            angles = self._compute_angles_from_ground_truth(targets[batch_idx], trajectory_output[batch_idx])
            best_mode = self._compute_best_mode(angles, targets[batch_idx], trajectory_output[batch_idx])
            best_mode_trajectory = trajectory_output[batch_idx, best_mode, :].unsqueeze(0)

            ## regression loss : mode가 일치한 trajectory만을 이용해서 l2 loss 계산
            regression_loss = f.mse_loss(best_mode_trajectory, targets[batch_idx])

            ## classfication loss : cross entropy
            mode_probabilities = mode_output[batch_idx].unsqueeze(0)
            best_mode_target = torch.tensor([best_mode], device=predictions.device)                                 # compute_best_mode를 이용해 target의 모드를 결정하고 이를 gt로 이용                   
            classification_loss = f.cross_entropy(mode_probabilities, best_mode_target)                             # mode에대한 classification loss

            ## MTP loss
            loss = classification_loss + self.alpha * regression_loss
            
            batch_losses = torch.cat((batch_losses, loss.unsqueeze(0)), 0)

        avg_loss = torch.mean(batch_losses)

        return avg_loss
