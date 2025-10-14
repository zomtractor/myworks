from utils.utils import L1_Charbonnier_loss, img_pad, calculate_metrics, SSIM_loss, VGGLoss
import torch
SSIMLoss = SSIM_loss().cuda()
CharLoss = L1_Charbonnier_loss().cuda()
VGGLoss = VGGLoss().cuda()
criterion = torch.nn.L1Loss()