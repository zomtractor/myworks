import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from pytorch_msssim import ssim
import lpips
import numpy as np
from focal_frequency_loss import FocalFrequencyLoss as FFL

import model


# ========== L1 Charbonnier Loss ==========
class L1CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(L1CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        return torch.mean(torch.sqrt((x - y) ** 2 + self.eps))


# ========== Focal Frequency Loss ==========
class FocalFrequencyLoss(FFL):
    def __init__(self, loss_weight=1.0, alpha=1.0):
        super(FocalFrequencyLoss, self).__init__(loss_weight=loss_weight, alpha=alpha)


# ========== SSIM Loss ==========
class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()

    def forward(self, x, y):
        return 1 - ssim(x, y, data_range=1.0, size_average=True)

# ========== Color Consistency Loss ==========
class ColorConsistencyLoss(nn.Module):
    def __init__(self, alpha=0.1):
        super(ColorConsistencyLoss, self).__init__()
        self.alpha = alpha

    def forward(self, pred, target):
        # 计算预测图像和目标图像的颜色直方图
        pred_hist = torch.histc(pred, bins=256, min=0, max=1)
        target_hist = torch.histc(target, bins=256, min=0, max=1)

        # 计算颜色直方图之间的L1距离
        hist_diff = torch.abs(pred_hist - target_hist)

        # 计算颜色一致性损失
        loss = self.alpha * torch.sum(hist_diff)

        return loss

class LPIPSLoss(nn.Module):
    def __init__(self,net='vgg'):
        super(LPIPSLoss, self).__init__()
        self.vgg = lpips.LPIPS(net=net)  # Perceptual VGG Loss
        self.vgg.eval()  # VGG loss 不更新
        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, pred, target):
        return self.vgg(pred * 2 - 1, target * 2 - 1).mean()



# ========== 综合损失 ==========
class CombinedLoss(nn.Module):
    def __init__(self, loss_dict):
        super(CombinedLoss, self).__init__()
        self.losses = nn.ModuleDict()
        self.weights = {}
        self.cumulative_loss = {}
        self.cumulative_loss['total'] = 0.0

        for loss_name, loss_cfg in loss_dict.items():
            # Extract weight (required) and remove it from config
            loss_cfg = loss_cfg.copy()
            weight = loss_cfg.pop("loss_weight")
            self.weights[loss_name] = weight
            self.cumulative_loss[loss_name] = 0.0
            # Dynamically get the loss class from torch.nn
            loss_class = getattr(model, loss_name, None)
            if loss_class is None:
                raise ValueError(f"Loss class '{loss_name}' not found in torch.nn")

            # Initialize the loss with remaining parameters
            self.losses[loss_name] = loss_class(**loss_cfg)
            print(f'Initialized {loss_name} with weight {weight}')

    def forward(self, input, target):
        total_loss = 0.0
        for name, loss_fn in self.losses.items():
            loss_val = loss_fn(input, target)
            total_loss += self.weights[name] * loss_val
            self.cumulative_loss[name] += loss_val.item()
        self.cumulative_loss['total'] += total_loss.item()
        return total_loss

    def clear_cumulative_loss(self):
        """Clear cumulative loss values."""
        for name in self.cumulative_loss.keys():
            self.cumulative_loss[name] = 0.0

    def print_cumulative_loss(self):
        """Print cumulative loss values."""
        for name, value in self.cumulative_loss.items():
            print(f"{name}: {value:.4f}", end=',')
        print()

    def merge(self, other):
        """Merge another CombinedLoss instance into this one."""
        if not isinstance(other, CombinedLoss):
            raise ValueError("Can only merge with another CombinedLoss instance.")
        for name in self.cumulative_loss.keys():
            self.cumulative_loss[name] += other.cumulative_loss[name]
        return self


if __name__ == '__main__':
    with open('../config.yaml', 'r') as config:
        opt = yaml.safe_load(config)
    loss_dict = opt['TRAINING']['LOSS']
    criterion = CombinedLoss(loss_dict)
    gt = torch.randn(1, 3, 256, 256)
    input = torch.randn(1, 3, 256, 256)

    net = nn.Sequential(
        nn.Conv2d(3,5,3,1,1),
        nn.BatchNorm2d(5),
        nn.ReLU(inplace=True),
        nn.Conv2d(5,3,3,1,1)
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    for i in range(1000):
        prediction = net(input)
        loss = criterion(prediction, gt)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        criterion.print_cumulative_loss()


