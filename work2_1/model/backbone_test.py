import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch import optim
from torch.utils.data import DataLoader, Dataset


# -------------------------
# Util: create gaussian map from center + sigma
# -------------------------
def gaussian_2d(shape, center, sigma):
    """
    shape: (H, W)
    center: (cy, cx) floats (in pixels)
    sigma: float (pixels)
    returns: (H, W) gaussian map, sum normalized to 1 (or not, choose later)
    """
    H, W = shape
    ys = torch.arange(0, H, dtype=torch.float32, device=center.device)
    xs = torch.arange(0, W, dtype=torch.float32, device=center.device)
    yy = ys.view(H, 1)
    xx = xs.view(1, W)
    cy, cx = center
    gauss = torch.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (sigma ** 2 + 1e-8)))
    return gauss

def normalize_map(m):
    s = m.sum(dim=[-2, -1], keepdim=True)
    return m / (s + 1e-6)

# -------------------------
# Light Detector Head
# -------------------------
class LightDetectorHead(nn.Module):
    """
    Predicts:
      - heatmap: probability of light center (B,1,H,W)
      - sigma_map: per-pixel sigma (B,1,H,W), positive
    Usage: attach to high-res feature (e.g., decoder output)
    """
    def __init__(self, in_channels, mid=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.heat = nn.Conv2d(mid, 1, 1)      # raw logits -> sigmoid
        self.sigma = nn.Conv2d(mid, 1, 1)     # raw sigma -> softplus
        # optional intensity head
        self.intensity = nn.Conv2d(mid, 1, 1)

    def forward(self, x):
        f = self.conv(x)
        heat = torch.sigmoid(self.heat(f))            # (B,1,H,W)
        sigma = F.softplus(self.sigma(f)) + 1e-4      # enforce >0
        intensity = F.softplus(self.intensity(f))     # >0
        return heat, sigma, intensity

# -------------------------
# Mask Refinement Module (ALMR)
# -------------------------
class MaskRefinementModule(nn.Module):
    """
    Predicts a mask for strong repair region, and refines it by gaussian smoothing.
    Input: feature map (B,C,H,W)
    Output: mask (B,1,H,W) in [0,1]
    """
    def __init__(self, in_channels, mid=64, kernel_size=21, sigma=3.0):
        super().__init__()
        self.pred = nn.Sequential(
            nn.Conv2d(in_channels, mid, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, 1, 1)
        )
        # precompute gaussian blur kernel (separable)
        self.register_buffer("gauss_kernel", self._make_gauss_kernel(kernel_size, sigma))

    def _make_gauss_kernel(self, k, sigma):
        ax = torch.arange(-(k // 2), k // 2 + 1, dtype=torch.float32)
        kernel1d = torch.exp(-0.5 * (ax / sigma) ** 2)
        kernel1d = kernel1d / kernel1d.sum()
        # outer product
        kernel2d = kernel1d[:, None] * kernel1d[None, :]
        kernel2d = kernel2d.unsqueeze(0).unsqueeze(0)  # shape (1,1,k,k)
        return kernel2d

    def forward(self, x):
        raw = self.pred(x)              # (B,1,H,W)
        raw = torch.sigmoid(raw)
        # smooth with group conv (same kernel across batch)
        B = raw.shape[0]
        k = self.gauss_kernel.shape[-1]
        pad = k // 2
        # conv2d expects (B, C, H, W); we use groups=B to apply per-sample
        kernel = self.gauss_kernel.expand(B, 1, k, k)
        smoothed = F.conv2d(F.pad(raw, (pad, pad, pad, pad), mode='reflect'), kernel, groups=B)
        # optionally sharpen edges: mask = mask * (1 - alpha * laplacian(mask))  (omitted for brevity)
        return smoothed.clamp(0., 1.)

# -------------------------
# Light Consistency Loss
# -------------------------
class LightConsistencyLoss(nn.Module):
    """
    Compare predicted light distribution (from heat * sigma) to GT light mask (or GT gaussian).
    Accepts:
      - pred_heat: (B,1,H,W)
      - pred_sigma: (B,1,H,W)
      - gt_mask: (B,1,H,W) binary mask or soft mask
    Produces scalar loss.
    """
    def __init__(self, reduction='mean', alpha_center=1.0, alpha_map=1.0):
        super().__init__()
        self.reduction = reduction
        self.alpha_center = alpha_center
        self.alpha_map = alpha_map

    def forward(self, pred_heat, pred_sigma, gt_mask):
        B, _, H, W = pred_heat.shape
        device = pred_heat.device

        # 1) Fit GT center & sigma (moment-based)
        # compute gt centroid (normalized to pixels)
        gt_sum = (gt_mask + 1e-8).sum(dim=[-2, -1])  # (B,1)
        ys = torch.arange(0, H, dtype=torch.float32, device=device).view(1,1,H,1)
        xs = torch.arange(0, W, dtype=torch.float32, device=device).view(1,1,1,W)
        cy = (gt_mask * ys).sum(dim=[-2,-1]) / gt_sum  # (B,1)
        cx = (gt_mask * xs).sum(dim=[-2,-1]) / gt_sum  # (B,1)
        # gt_sigma: second moment
        ys2 = (ys - cy.view(B,1,1,1))**2
        xs2 = (xs - cx.view(B,1,1,1))**2
        gt_var = (gt_mask * (ys2 + xs2)).sum(dim=[-2,-1]) / gt_sum   # (B,1)
        gt_sigma = torch.sqrt(gt_var + 1e-6).view(B,1)               # (B,1)

        # 2) Predicted center: compute centroid from pred_heat
        ph_sum = (pred_heat + 1e-8).sum(dim=[-2, -1])
        pcy = (pred_heat * ys).sum(dim=[-2,-1]) / ph_sum
        pcx = (pred_heat * xs).sum(dim=[-2,-1]) / ph_sum
        # predicted sigma: average over heat-weighted sigma_map
        psigma = (pred_sigma * pred_heat).sum(dim=[-2,-1]) / ph_sum
        psigma = psigma.view(B,1)

        # 3) center loss (pixel distance)
        center_dist = torch.sqrt((pcy - cy).pow(2) + (pcx - cx).pow(2)).view(B)
        center_loss = center_dist.mean()

        # 4) map loss: build gaussian maps from predicted center and sigma & compare to gt gaussian (from gt_mask)
        pred_gauss_maps = []
        gt_gauss_maps = []
        for b in range(B):
            cy_b = pcy[b,0]
            cx_b = pcx[b,0]
            sig_b = psigma[b,0].clamp(min=1.0)   # avoid zero
            pg = gaussian_2d((H, W), torch.tensor([cy_b, cx_b], device=device), sig_b)
            pg = pg / (pg.sum() + 1e-9)
            pred_gauss_maps.append(pg)
            # GT gaussian: approximate from GT centroid and sigma
            gcy = cy[b,0]; gcx = cx[b,0]; gs = gt_sigma[b,0].clamp(min=1.0)
            gg = gaussian_2d((H, W), torch.tensor([gcy, gcx], device=device), gs)
            gg = gg / (gg.sum() + 1e-9)
            gt_gauss_maps.append(gg)
        pred_gauss = torch.stack(pred_gauss_maps, dim=0).unsqueeze(1)  # (B,1,H,W)
        gt_gauss = torch.stack(gt_gauss_maps, dim=0).unsqueeze(1)

        map_loss = F.l1_loss(pred_gauss, gt_gauss, reduction=self.reduction)

        loss = self.alpha_center * center_loss + self.alpha_map * map_loss
        return loss, {'center_loss': center_loss.item(), 'map_loss': map_loss.item()}

# -------------------------
# Frequency Loss (optional)
# -------------------------
def dct_2d(x):
    # naive DCT-II via matrix multiply (works for small HxW or you can use fft approximations)
    # For brevity, use torch.fft. Note: you may prefer to use DCT from scipy or implement cleanly.
    X = torch.fft.fft2(x)
    mag = torch.abs(X)
    return mag

class FrequencyLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target, mask=None):
        # pred/target: (B, C, H, W)
        # Only compute difference in frequency magnitude on masked region
        pred_f = dct_2d(pred)
        tgt_f = dct_2d(target)
        diff = torch.abs(pred_f - tgt_f)
        if mask is not None:
            diff = diff * mask
            denom = mask.sum() + 1e-6
            return diff.sum() / denom
        return diff.mean()


class SyntheticLightDataset(Dataset):
    def __init__(self, num_samples=100, image_size=(64, 64)):
        self.num_samples = num_samples
        self.image_size = image_size
        self.H, self.W = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 生成随机输入图像
        input_image = torch.randn(3, self.H, self.W)

        # 生成GT光照掩码（高斯分布）
        cy = torch.randint(10, self.H - 10, (1,)).item()
        cx = torch.randint(10, self.W - 10, (1,)).item()
        sigma = torch.randint(5, 15, (1,)).item()

        # 创建坐标网格
        y = torch.arange(self.H, dtype=torch.float32).view(self.H, 1)
        x = torch.arange(self.W, dtype=torch.float32).view(1, self.W)

        # 生成高斯分布
        gt_mask = torch.exp(-0.5 * (((y - cy) / sigma) ** 2 + ((x - cx) / sigma) ** 2))
        gt_mask = gt_mask.unsqueeze(0)  # (1, H, W)

        return input_image, gt_mask


class SimpleLightNet(nn.Module):
    def __init__(self, input_channels=3, output_channels=1):
        super(SimpleLightNet, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, 3, padding=1)
        self.heat_conv = nn.Conv2d(16, output_channels, 1)
        self.sigma_conv = nn.Conv2d(16, output_channels, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        heat = self.sigmoid(self.heat_conv(x))  # 输出在0-1之间
        sigma = self.sigmoid(self.sigma_conv(x)) * 10 + 1  # 输出在1-11之间
        return heat, sigma
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 创建数据
    dataset = SyntheticLightDataset(num_samples=100, image_size=(64, 64))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 创建模型和损失函数
    model = SimpleLightNet().to(device)
    criterion = LightConsistencyLoss(alpha_center=1.0, alpha_map=1.0)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    model.train()
    for epoch in range(5):  # 只训练5个epoch进行测试
        total_loss = 0
        total_center_loss = 0
        total_map_loss = 0

        for batch_idx, (images, gt_masks) in enumerate(dataloader):
            images = images.to(device)
            gt_masks = gt_masks.to(device)

            # 前向传播
            pred_heat, pred_sigma = model(images)

            # 计算损失
            loss, loss_dict = criterion(pred_heat, pred_sigma, gt_masks)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计损失
            total_loss += loss.item()
            total_center_loss += loss_dict['center_loss']
            total_map_loss += loss_dict['map_loss']

            if batch_idx % 5 == 0:
                print(f'Epoch: {epoch}, Batch: {batch_idx}, Loss: {loss.item():.4f}, '
                      f'Center Loss: {loss_dict["center_loss"]:.4f}, '
                      f'Map Loss: {loss_dict["map_loss"]:.4f}')

        # 打印epoch统计
        avg_loss = total_loss / len(dataloader)
        avg_center_loss = total_center_loss / len(dataloader)
        avg_map_loss = total_map_loss / len(dataloader)
        print(f'Epoch {epoch} Summary: Total Loss: {avg_loss:.4f}, '
              f'Center Loss: {avg_center_loss:.4f}, Map Loss: {avg_map_loss:.4f}')
        print('-' * 50)

    print("训练测试完成！损失函数运行正常。")
