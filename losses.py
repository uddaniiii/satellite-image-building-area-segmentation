import torch
import torch.nn as nn

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, input, target):
        logpt = torch.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        focal_loss = -1 * self.alpha * (1 - pt) ** self.gamma * logpt

        if self.reduction == 'mean':
            focal_loss = focal_loss.mean()
        elif self.reduction == 'sum':
            focal_loss = focal_loss.sum()

        return focal_loss

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, input, target):
        smooth = 1.0
        iflat = input.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()
        return 1.0 - ((2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))