import random
import math

import torch
from torch import nn
from torch.nn import functional as f

# class MTPLoss:
#     """ Computes the loss for the MTP model. """

#     def __init__(self,
#                  num_modes: int,
#                  regression_loss_weight: float = 1.,
#                  angle_threshold_degrees: float = 5.):
#         """
#         Inits MTP loss.
#         :param num_modes: How many modes are being predicted for each agent.
#         :param regression_loss_weight: Coefficient applied to the regression loss to
#             balance classification and regression performance.
#         :param angle_threshold_degrees: Minimum angle needed between a predicted trajectory
#             and the ground to consider it a match.
#         """
#         self.num_modes = num_modes
#         self.num_location_coordinates_predicted = 2  # We predict x, y coordinates at each timestep.
#         self.regression_loss_weight = regression_loss_weight
#         self.angle_threshold = angle_threshold_degrees

#     def _get_trajectory_and_modes(self,
#                                   model_prediction: torch.Tensor):
#         """
#         Splits the predictions from the model into mode probabilities and trajectory.
#         :param model_prediction: Tensor of shape [batch_size, n_timesteps * n_modes * 2 + n_modes].
#         :return: Tuple of tensors. First item is the trajectories of shape [batch_size, n_modes, n_timesteps, 2].
#             Second item are the mode probabilities of shape [batch_size, num_modes].
#         """
#         mode_probabilities = model_prediction[:, -self.num_modes:].clone()

#         desired_shape = (model_prediction.shape[0], self.num_modes, -1, self.num_location_coordinates_predicted)
#         trajectories_no_modes = model_prediction[:, :-self.num_modes].clone().reshape(desired_shape)

#         return trajectories_no_modes, mode_probabilities

#     @staticmethod
#     def _angle_between(ref_traj: torch.Tensor,
#                        traj_to_compare: torch.Tensor) -> float:
#         """
#         Computes the angle between the last points of the two trajectories.
#         The resulting angle is in degrees and is an angle in the [0; 180) interval.
#         :param ref_traj: Tensor of shape [n_timesteps, 2].
#         :param traj_to_compare: Tensor of shape [n_timesteps, 2].
#         :return: Angle between the trajectories.
#         """

#         EPSILON = 1e-5

#         if (ref_traj.ndim != 2 or traj_to_compare.ndim != 2 or
#                 ref_traj.shape[1] != 2 or traj_to_compare.shape[1] != 2):
#             raise ValueError('Both tensors should have shapes (-1, 2).')

#         if torch.isnan(traj_to_compare[-1]).any() or torch.isnan(ref_traj[-1]).any():
#             return 180. - EPSILON

#         traj_norms_product = float(torch.norm(ref_traj[-1]) * torch.norm(traj_to_compare[-1]))

#         # If either of the vectors described in the docstring has norm 0, return 0 as the angle.
#         if math.isclose(traj_norms_product, 0):
#             return 0.

#         # We apply the max and min operations below to ensure there is no value
#         # returned for cos_angle that is greater than 1 or less than -1.
#         # This should never be the case, but the check is in place for cases where
#         # we might encounter numerical instability.
#         dot_product = float(ref_traj[-1].dot(traj_to_compare[-1]))
#         angle = math.degrees(math.acos(max(min(dot_product / traj_norms_product, 1), -1)))

#         if angle >= 180:
#             return angle - EPSILON

#         return angle

#     @staticmethod
#     def _compute_ave_l2_norms(tensor: torch.Tensor) -> float:
#         """
#         Compute the average of l2 norms of each row in the tensor.
#         :param tensor: Shape [1, n_timesteps, 2].
#         :return: Average l2 norm. Float.
#         """
#         l2_norms = torch.norm(tensor, p=2, dim=2)
#         avg_distance = torch.mean(l2_norms)
#         return avg_distance.item()

#     def _compute_angles_from_ground_truth(self, target: torch.Tensor,
#                                           trajectories: torch.Tensor):
#         """
#         Compute angle between the target trajectory (ground truth) and the predicted trajectories.
#         :param target: Shape [1, n_timesteps, 2].
#         :param trajectories: Shape [n_modes, n_timesteps, 2].
#         :return: List of angle, index tuples.
#         """
#         angles_from_ground_truth = []
#         for mode, mode_trajectory in enumerate(trajectories):
#             # For each mode, we compute the angle between the last point of the predicted trajectory for that
#             # mode and the last point of the ground truth trajectory.
#             angle = self._angle_between(target[0], mode_trajectory)

#             angles_from_ground_truth.append((angle, mode))
#         return angles_from_ground_truth

#     def _compute_best_mode(self,
#                            angles_from_ground_truth,
#                            target: torch.Tensor, trajectories: torch.Tensor) -> int:
#         """
#         Finds the index of the best mode given the angles from the ground truth.
#         :param angles_from_ground_truth: List of (angle, mode index) tuples.
#         :param target: Shape [1, n_timesteps, 2]
#         :param trajectories: Shape [n_modes, n_timesteps, 2]
#         :return: Integer index of best mode.
#         """

#         # We first sort the modes based on the angle to the ground truth (ascending order), and keep track of
#         # the index corresponding to the biggest angle that is still smaller than a threshold value.
#         angles_from_ground_truth = sorted(angles_from_ground_truth)
#         max_angle_below_thresh_idx = -1
#         for angle_idx, (angle, mode) in enumerate(angles_from_ground_truth):
#             if angle <= self.angle_threshold:
#                 max_angle_below_thresh_idx = angle_idx
#             else:
#                 break

#         # We choose the best mode at random IF there are no modes with an angle less than the threshold.
#         if max_angle_below_thresh_idx == -1:
#             best_mode = random.randint(0, self.num_modes - 1)

#         # We choose the best mode to be the one that provides the lowest ave of l2 norms between the
#         # predicted trajectory and the ground truth, taking into account only the modes with an angle
#         # less than the threshold IF there is at least one mode with an angle less than the threshold.
#         else:
#             # Out of the selected modes above, we choose the final best mode as that which returns the
#             # smallest ave of l2 norms between the predicted and ground truth trajectories.
#             distances_from_ground_truth = []

#             for angle, mode in angles_from_ground_truth[:max_angle_below_thresh_idx + 1]:
#                 norm = self._compute_ave_l2_norms(target - trajectories[mode, :, :])

#                 distances_from_ground_truth.append((norm, mode))

#             distances_from_ground_truth = sorted(distances_from_ground_truth)
#             best_mode = distances_from_ground_truth[0][1]

#         return best_mode

#     def __call__(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
#         """
#         Computes the MTP loss on a batch.
#         The predictions are of shape [batch_size, n_ouput_neurons of last linear layer]
#         and the targets are of shape [batch_size, 1, n_timesteps, 2]
#         :param predictions: Model predictions for batch.
#         :param targets: Targets for batch.
#         :return: zero-dim tensor representing the loss on the batch.
#         """

#         batch_losses = torch.Tensor().requires_grad_(True).to(predictions.device)
#         trajectories, modes = self._get_trajectory_and_modes(predictions)

#         for batch_idx in range(predictions.shape[0]):

#             angles = self._compute_angles_from_ground_truth(target=targets[batch_idx],
#                                                             trajectories=trajectories[batch_idx])

#             best_mode = self._compute_best_mode(angles,
#                                                 target=targets[batch_idx],
#                                                 trajectories=trajectories[batch_idx])

#             best_mode_trajectory = trajectories[batch_idx, best_mode, :].unsqueeze(0)

#             regression_loss = f.smooth_l1_loss(best_mode_trajectory, targets[batch_idx])

#             mode_probabilities = modes[batch_idx].unsqueeze(0)
#             best_mode_target = torch.tensor([best_mode], device=predictions.device)
#             classification_loss = f.cross_entropy(mode_probabilities, best_mode_target)

#             loss = classification_loss + self.regression_loss_weight * regression_loss

#             deg = abs(math.atan( targets[batch_idx][0][-1][1]/targets[batch_idx][0][-1][0])*180/math.pi)
#             deg_weight = math.exp(deg/20)
#             # loss = loss * deg_weight

#             batch_losses = torch.cat((batch_losses, loss.unsqueeze(0)), 0)

#         avg_loss = torch.mean(batch_losses)

#         return avg_loss
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
