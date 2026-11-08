import torch
import torch.nn as nn


class IntegralMSELoss(nn.Module):
    def __init__(self, size_average=True):
        super(IntegralMSELoss, self).__init__()
        self.size_average = size_average

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        # num_joints = int(output.size(1) / 3)#####
        num_joints = output.size(1)
        # output = output.reshape(batch_size, num_joints, 3)######
        pred_coordinate = output[:, :, :2]
        target = target[:, :, :2]######
        # output = output.reshape(batch_size, num_joints, 3)
        out = (pred_coordinate - target) ** 2
        out = out * target_weight
        out = out.sum() / num_joints  # my modify
        if self.size_average:
            return out.sum() / batch_size
        else:
            return out.sum()


class IntegralL1Loss(nn.Module):
    def __init__(self, size_average=True):
        super(IntegralL1Loss, self).__init__()
        self.size_average = size_average

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        # output = output.reshape(batch_size, num_joints, 3)
        pred_coordinate = output[:, :, :2]
        target = target[:, :, :2]#####
        out = torch.abs(pred_coordinate - target)
        out = out * target_weight
        out = out.sum() / num_joints  # my modify
        if self.size_average:
            return out.sum() / batch_size
        else:
            return out.sum()
