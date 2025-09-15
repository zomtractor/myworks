import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_msssim import ssim
import lpips
import numpy as np
from focal_frequency_loss import FocalFrequencyLoss as FFL


# ========== L1 Charbonnier Loss ==========
class L1CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(L1CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        return torch.mean(torch.sqrt((x - y) ** 2 + self.eps))


# ========== Focal Frequency Loss ==========

# ========== SSIM Loss ==========
class SSIMLoss(nn.Module):
    def __init__(self):
        super(SSIMLoss, self).__init__()

    def forward(self, x, y):
        return 1 - ssim(x, y, data_range=1.0, size_average=True)


# ========== 综合损失 ==========
class CombinedLoss(nn.Module):
    def __init__(self, weights=[1.0, 1.0, 1.0, 0.01]):
        """
        weights: [w_char, w_ssim, w_vgg, w_freq]
        """
        super(CombinedLoss, self).__init__()
        assert len(weights) == 4, "Must provide 4 weights"
        self.weights = weights

        self.l1_char = L1CharbonnierLoss()
        self.ssim = SSIMLoss()
        self.vgg = lpips.LPIPS(net='vgg')  # Perceptual VGG Loss
        self.vgg.eval()  # VGG loss 不更新
        for param in self.vgg.parameters():
            param.requires_grad = False

        self.freq = FFL(loss_weight=1.0, alpha=1.0)

    def forward(self, pred, target):
        pred_norm = pred.clamp(0, 1)
        target_norm = target.clamp(0, 1)

        l_char = self.l1_char(pred_norm, target_norm)
        l_ssim = self.ssim(pred_norm, target_norm)
        l_freq = self.freq(pred_norm, target_norm)
        l_vgg = self.vgg(pred_norm * 2 - 1, target_norm * 2 - 1).mean()

        total_loss = (
                self.weights[0] * l_char +
                self.weights[1] * l_ssim +
                self.weights[2] * l_freq +
                self.weights[3] * l_vgg
        )

        return total_loss, {
            "charbonnier": l_char.item(),
            "ssim": l_ssim.item(),
            "freq": l_freq.item(),
            "vgg": l_vgg.item()
        }
