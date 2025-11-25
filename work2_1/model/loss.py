import lpips
import model
import torch
import torch.nn as nn
import yaml
from focal_frequency_loss import FocalFrequencyLoss as FFL
from pytorch_msssim import ms_ssim
import torch.nn.functional as F
from torchvision.models import vgg19

# ========== L1 Charbonnier Loss ==========
class L1CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(L1CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y, *args):
        return torch.mean(torch.sqrt((x - y) ** 2 + self.eps))


# ========== Focal Frequency Loss ==========
class FocalFrequencyLoss(FFL):
    def __init__(self, loss_weight=1.0, alpha=1.0):
        super(FocalFrequencyLoss, self).__init__(loss_weight=loss_weight, alpha=alpha)

    def forward(self, x, y, *args):
        return super(FocalFrequencyLoss, self).forward(x, y)


# ========== SSIM Loss ==========
class SSIMLoss(nn.Module):
    def __init__(self,weights=None):
        super(SSIMLoss, self).__init__()
        self.weights = weights

    def forward(self, x, y, *args):
        return 1 - ms_ssim(x, y, data_range=1.0, size_average=True,weights=self.weights)

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

# class LPIPSLoss(nn.Module):
#     def __init__(self,net='vgg'):
#         super(LPIPSLoss, self).__init__()
#         self.vgg = lpips.LPIPS(net=net)  # Perceptual VGG Loss
#         self.vgg.eval()  # VGG loss 不更新
#         for param in self.vgg.parameters():
#             param.requires_grad = False
#
#     def forward(self, pred, target, *args):
#         return self.vgg(pred * 2 - 1, target * 2 - 1).mean()


class LPIPSLoss(nn.Module):
    def __init__(self, layers=[2,7,12,21,30],layer_weight=[1/2.6,1/4.8,1/3.7,1/5.6,10/1.5]):
        super(LPIPSLoss, self).__init__()
        vgg = vgg19(pretrained=True)
        model = nn.Sequential(*list(vgg.features)[:31])
        model = model.cuda()
        model = model.eval()
        # Freeze VGG19 #
        for param in model.parameters():
            param.requires_grad = False

        self.vgg = model
        self.mae_loss = nn.L1Loss()
        self.selected_feature_index = layers
        self.layer_weight = layer_weight

    def extract_feature(self, x):
        selected_features = []
        for i, model in enumerate(self.vgg):
            x = model(x)
            if i in self.selected_feature_index:
                selected_features.append(x.clone())
        return selected_features

    def forward(self, source, target):
        source_feature = self.extract_feature(source)
        target_feature = self.extract_feature(target)
        len_feature = len(source_feature)
        perceptual_loss = 0
        for i in range(len_feature):
            perceptual_loss += self.mae_loss(source_feature[i], target_feature[i]) * self.layer_weight[i]
        return perceptual_loss

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

class ParamAwareLightLoss(nn.Module):
    def __init__(self, w_recon=1.0, w_alpha=0.5, w_param=0.5):
        super().__init__()
        self.w_recon = w_recon
        self.w_alpha = w_alpha
        self.w_param = w_param

    def forward(self, light_map, gt_light, alpha, params):
        B, _, H, W = light_map.shape
        gt_gray = (0.2989 * gt_light[:,0,:,:] + 0.5870 * gt_light[:,1,:,:] + 0.1140 * gt_light[:,2,:,:]).unsqueeze(1)

        # === (1) Light reconstruction ===
        l1 = torch.mean(torch.abs(light_map - gt_light))
        L_light_recon = l1

        # === (2) Alpha mask consistency ===
        gt_norm = torch.norm(gt_gray, dim=1, keepdim=True)
        L_alpha = torch.mean(torch.abs(alpha - gt_norm))

        # === (3) Explicit param-level losses ===
        x_pred, y_pred = params[..., 0], params[..., 1]
        a_pred, b_pred = params[..., 2], params[..., 3]
        intensity_pred = params[..., 5]
        falloff_pred = params[..., 10]

        # 光源中心 from gt
        with torch.no_grad():
            gt_center = self._center_of_mass(gt_norm)
            gt_area = (gt_norm > 0.3).float().mean(dim=[2,3])
            gt_intensity = gt_gray.mean(dim=[2,3])

        L_center = F.l1_loss(x_pred.mean(-1), gt_center[..., 0]) + F.l1_loss(y_pred.mean(-1), gt_center[..., 1])
        A_pred = torch.pi * a_pred.mean(-1) * b_pred.mean(-1)
        L_size = F.l1_loss(torch.log(A_pred + 1e-6), torch.log(gt_area + 1e-6))
        L_intensity = F.l1_loss(intensity_pred.mean(-1), gt_intensity)

        L_param_struct = L_center + 0.3 * L_size + 0.2 * L_intensity

        L_total = (
            self.w_recon * L_light_recon +
            self.w_alpha * L_alpha +
            self.w_param * L_param_struct
        )

        return L_total

    def _center_of_mass(self, mask):
        B, _, H, W = mask.shape
        y, x = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        x = x.to(mask.device).float() / W
        y = y.to(mask.device).float() / H
        weight = mask / (mask.sum(dim=[2,3], keepdim=True) + 1e-6)
        cx = (weight * x).sum(dim=[2,3])
        cy = (weight * y).sum(dim=[2,3])
        return torch.stack([cx, cy], dim=-1)


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

    def forward(self, input, target, *args):
        total_loss = 0.0
        for name, loss_fn in self.losses.items():
            loss_val = loss_fn(input, target,*args)
            total_loss += self.weights[name] * loss_val
            self.cumulative_loss[name] += loss_val.item()
        self.cumulative_loss['total'] += total_loss.item()
        return total_loss

    def clear_cumulative_loss(self):
        """Clear cumulative loss values."""
        for name in self.cumulative_loss.keys():
            self.cumulative_loss[name] = 0.0

    def print_cumulative_loss(self, title='default'):
        print(f"loss of {title}",end=':')
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

    def logging(self, writer,epoch):
        """Log cumulative loss values to TensorBoard."""
        for name, value in self.cumulative_loss.items():
            writer.add_scalar(f'loss/{name}', value,epoch)



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


