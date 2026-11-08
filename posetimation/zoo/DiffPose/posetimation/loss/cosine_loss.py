"""
    Author: Haoming Chen
    E-mail: chenhaomingbob@163.com
    Time: 2021/4/10
    Description
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class StructureCosineSimilarity(nn.Module):
    def __init__(self, use_target_weight: bool = True, batch_average: bool = True):
        super(StructureCosineSimilarity, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight
        self.batch_average = batch_average
        self.limbs = np.array(
            [(0, 1), (0, 2), (1, 5), (1, 6), (5, 7), (7, 9), (6, 8), (8, 10), (11, 12), (11, 13), (12, 14),
             (13, 15), (14, 16)])
        self.num_limb = len(self.limbs)

    def forward(self, preds: Tensor, targets: Tensor, target_weights: Tensor):
        """
            preds & target : 2D coordinates sequence
        """
        limb_target_weights = self._cal_limb_weights(target_weights)
        preds_cos_similarity = self._cal_cosine_similarity(preds)
        targets_cos_similarity = self._cal_cosine_similarity(targets)

        if self.use_target_weight:
            loss = self.criterion(torch.mul(preds_cos_similarity, limb_target_weights),
                                  torch.mul(targets_cos_similarity, limb_target_weights))
        else:
            loss = self.criterion(preds_cos_similarity, targets_cos_similarity)

        if not self.batch_average:
            batch_size = targets.shape[0]
            loss *= batch_size

        return loss

    def _cal_cosine_similarity(self, joints: Tensor) -> Tensor:
        """
            calculate a single skeleton structure cosine, which should supported the batch operation
            joints type:torch.Tensor
        """
        limb_endpoint_a_joint = joints[:, self.limbs[:, 0], :2]
        limb_endpoint_b_joint = joints[:, self.limbs[:, 1], :2]
        cosine_similarity = F.cosine_similarity(limb_endpoint_a_joint, limb_endpoint_b_joint, dim=-1)

        return cosine_similarity
        # if not torch.is_tensor(joints):
        #     joints = torch.tensor(joints)
        # joints = joints[:, :, :2]
        # cos_value_list = []
        # for pair in cosine_joint_pair:
        #     j1, j2 = pair
        #     cos_value = F.cosine_similarity(joints[:, j1, :], joints[:, j2, :])  # [batch, value-1-pair]
        #     cos_value_list.append(cos_value)
        # concat_tensor = torch.cat(cos_value_list)
        # return concat_tensor

    def _cal_limb_weights(self, target_weights: Tensor) -> Tensor:
        if target_weights.ndim == 3:
            assert target_weights.shape[2] == 1
            target_weights = target_weights.squeeze(2)

        limb_endpoint_a, limb_endpoint_b = self.limbs[:, 0], self.limbs[:, 1]
        limb_endpoint_a_weight = target_weights[:, limb_endpoint_a]  # endpoint weight = 1 if endpoint exist
        limb_endpoint_b_weight = target_weights[:, limb_endpoint_b]

        limb_weights = (limb_endpoint_a_weight + limb_endpoint_b_weight) / 2
        limb_weights = limb_weights.type(torch.int8)  # Both endpoint a and endpoint b are exist, then limb_weight = 1
        return limb_weights
        # batch_size = target_weight.shape[0]
        # target_weight_array = torch.ones(batch_size, self.pairs, 1)
        # target_weight_batch_list = []
        # for joint_index, pair in enumerate(self.limbs):
        #     j1, j2 = pair
        #     for batch_idx in range(batch_size):
        #         if target_weight[batch_idx, j1, :] == 1 and target_weight[batch_idx, j2, :] == 1:
        #             target_weight_array[batch_idx, joint_index, :] = 1
        #         else:
        #             target_weight_array[batch_idx, joint_index, :] = 0
        # for channel in range(self.pairs):
        #     target_weight_batch_list.append(target_weight_array[:, channel, :])
        # concat_weight = torch.cat(target_weight_batch_list)
        # return concat_weight


if __name__ == '__main__':

    out = torch.tensor(np.random.rand(16, 17, 3))
    target = torch.tensor(np.random.rand(16, 17, 3))
    target_weight = torch.tensor(np.random.rand(16, 17, 1))
    scs = StructureCosineSimilarity(True)
    loss = scs(out, target, target_weight)
    print(loss)
