# smfr_net.py
# PyTorch implementation of SMFR-Net (from the uploaded paper)
# - Implements: FDM, MGDC, CSAM, SMEBlock, SMDBlock, SMFRNet
# - Composite loss wrapper (L1 + perceptual (optional) + MS-SSIM (optional))

import torch
import torch.nn as nn
import torch.nn.functional as F

from model import LayerNorm


# ------------------ Utilities ------------------
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ------------------ Basic building blocks ------------------
class SimpleGate(nn.Module):
    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b

class SCA(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        gap = x.mean(dim=[2,3], keepdim=True)
        w = self.sigmoid(self.conv(gap))
        return x * w

class CSAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channel_att = SCA(channels)
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=True)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        mc = self.channel_att(x)
        avg_pool = x.mean(dim=1, keepdim=True)
        max_pool,_ = x.max(dim=1, keepdim=True)
        cat = torch.cat([avg_pool, max_pool], dim=1)
        ms = self.sigmoid(self.spatial_conv(cat))
        ms_applied = x * ms
        out = mc + ms_applied
        return out

# ------------------ MGDC ------------------
class MGDC(nn.Module):
    def __init__(self, channels, groups=4, dilations=(1,3,5)):
        super().__init__()
        self.channels = channels
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        branches = []
        for d in dilations:
            branches.append(nn.Conv2d(channels, channels, kernel_size=3, padding=d, dilation=d, groups=groups, bias=True))
        self.branches = nn.ModuleList(branches)
        self.act = nn.PReLU(num_parameters=channels)
        self.ln = nn.LayerNorm(channels, eps=1e-6)
    def forward(self, x):
        x = x + self.beta
        outs = [self.act(conv(x)) for conv in self.branches]
        y = sum(outs)
        b,c,h,w = y.shape
        y_perm = y.permute(0,2,3,1).contiguous()
        y_norm = self.ln(y_perm).permute(0,3,1,2).contiguous()
        return y_norm

# ------------------ Frequency Domain Modulation (FDM) ------------------
class FDAM(nn.Module):
    def __init__(self, hidden=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=3, padding=1)
        )
    def forward(self, Dfreq):
        return self.net(Dfreq)

class FDM(nn.Module):
    def __init__(self, channels, gamma=0.1):
        super().__init__()
        self.channels = channels
        self.gamma = gamma
        self.channel_conv = nn.Conv2d(channels, channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, 2*channels, kernel_size=1, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(2*channels, channels, kernel_size=1, bias=True)
        )
        self.fdam = FDAM(hidden=8)
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def _make_distance_map(self, H, W, device, dtype):
        ys = torch.linspace(-1.0, 1.0, steps=H, device=device, dtype=dtype)
        xs = torch.linspace(-1.0, 1.0, steps=W, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing='ij')
        dist = torch.sqrt(xx*xx + yy*yy)
        dist = dist / (dist.max() + 1e-12)
        return dist.unsqueeze(0).unsqueeze(0)

    def forward(self, x):
        B,C,H,W = x.shape
        Xf = torch.fft.rfft2(x, dim=(-2,-1), norm='ortho')
        M = torch.abs(Xf)
        phase = torch.angle(Xf)
        gap = M.mean(dim=[2,3], keepdim=True)
        w = self.sigmoid(self.channel_conv(gap))
        M_hat = M * w
        M_processed = self.mlp(M_hat)
        Dfreq = self._make_distance_map(H, W, x.device, x.dtype)  # 1,1,H,W
        Wfreq = self.fdam(Dfreq)  # 1,1,H,W
        Wfreq_bc = Wfreq.expand(B, -1, -1, -1)  # B,1,H,W
        Wfreq_bc = Wfreq_bc.repeat(1, C, 1, 1)   # B,C,H,W
        M_out = M_processed * (1.0 + self.gamma * Wfreq_bc)
        complex_spec = M_out * torch.exp(1j * phase)
        x_freq = torch.fft.irfft2(complex_spec, dim=(-2,-1), norm='ortho').real
        gate = torch.sigmoid(self.alpha)
        x_out = gate * x_freq + (1.0 - gate) * x
        return x_out

# ------------------ SMEBlock (encoder block) ------------------
class SMEBlock(nn.Module):
    def __init__(self, channels, groups=4):
        super().__init__()
        self.ln1 = LayerNorm(channels)
        self.attn = FDM(channels)
        self.ln2 = LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            MGDC(channels, groups=groups),
            SimpleGate(),
            CSAM(channels),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        )

        self.beta = nn.Parameter(torch.tensor(1.0))
        self.lam = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        z = self.ln1(x)
        z = self.attn(z)
        res = z * self.beta

        z2 = self.ln2(res)
        z2 = self.ffn(z2)
        z2 = z2 * self.lam
        return res+z2

# ------------------ SMDBlock (decoder block) ------------------
class SMDBlock(nn.Module):
    def __init__(self, channels, groups=4):
        super().__init__()
        self.ln1 = LayerNorm(channels)
        self.attn = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            MGDC(channels, groups=groups),
            SimpleGate(),
            SCA(channels),
            nn.Conv2d(channels, channels, kernel_size=1, bias=True)
        )
        self.ln2 = LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, bias=True),
            SimpleGate(),
            nn.Conv2d(channels//2 if channels%2==0 else (channels+1)//2, channels, kernel_size=1, bias=True)
        )
        self.beta = nn.Parameter(torch.tensor(1.0))
        self.lam = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        z = self.ln1(x)
        z = self.attn(z)
        res = z * self.beta

        z2 = self.ln2(res)
        z2 = self.ffn(z2)
        z2 = z2 * self.lam
        return res+z2

# ------------------ Encoder / Decoder and Whole Model ------------------
class Downsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=True)
    def forward(self, x):
        return self.conv(x)

class UpsamplePixelShuffle(nn.Module):
    def __init__(self, in_ch, out_ch, scale=2):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch * (scale**2), kernel_size=3, padding=1, bias=True)
        self.ps = nn.PixelShuffle(scale)
    def forward(self, x):
        return self.ps(self.conv(x))

class SMFRNet(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, encoder_blocks=(1,2,3), decoder_blocks=(3,1,1), groups=4):
        super().__init__()
        self.input_conv = nn.Conv2d(in_ch, base_ch, kernel_size=3, padding=1, bias=True)
        self.enc_blocks = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = base_ch
        for idx, num in enumerate(encoder_blocks):
            blocks = nn.Sequential(*[SMEBlock(ch, groups=groups) for _ in range(num)])
            self.enc_blocks.append(blocks)
            if idx < len(encoder_blocks)-1:
                self.downs.append(Downsample(ch, ch*2))
                ch = ch*2
        self.bottleneck = SMEBlock(ch, groups=groups)
        self.ups = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for idx, num in enumerate(decoder_blocks):
            self.ups.append(UpsamplePixelShuffle(ch, ch//2, scale=2))
            ch = ch//2
            blocks = nn.Sequential(*[SMDBlock(ch, groups=groups) for _ in range(num)])
            self.dec_blocks.append(blocks)
        self.output_conv = nn.Conv2d(base_ch, in_ch, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        x0 = self.input_conv(x)
        enc_feats = []
        out = x0
        for i, blocks in enumerate(self.enc_blocks):
            out = blocks(out)
            enc_feats.append(out)
            if i < len(self.downs):
                out = self.downs[i](out)
        out = self.bottleneck(out)
        for i, (up, dec) in enumerate(zip(self.ups, self.dec_blocks)):
            out = up(out)
            skip = enc_feats[-(i+1)]
            if out.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=out.shape[2:], mode='bilinear', align_corners=False)
            out = out + skip
            out = dec(out)
        res = self.output_conv(out)
        return res

# ------------------ Composite loss wrapper ------------------
class CompositeLoss(nn.Module):
    def __init__(self, device='cpu', lambda_pixel=0.5, lambda_perc=0.5, lambda_msssim=0.2):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.lambda_pixel = lambda_pixel
        self.lambda_perc = lambda_perc
        self.lambda_msssim = lambda_msssim
        try:
            from torchvision.models import vgg19, VGG19_Weights
            weights = VGG19_Weights.DEFAULT
            vgg = vgg19(weights=weights).features.eval().to(device)
            for p in vgg.parameters(): p.requires_grad = False
            self.vgg = vgg
            self.vgg_layers = [2,7,12,21,30]
        except Exception as e:
            self.vgg = None
            print("Warning: torchvision VGG19 not available; perceptual loss disabled.", e)
        try:
            from pytorch_msssim import ms_ssim
            self.ms_ssim_fn = ms_ssim
        except Exception:
            self.ms_ssim_fn = None
            print("Note: pytorch_msssim not found. MS-SSIM term will be skipped unless installed.")

    def forward(self, pred_img, gt_img):
        l_pixel = self.l1(pred_img, gt_img)
        l_perc = 0.0
        if self.vgg is not None:
            loss = 0.0
            x_p = pred_img
            x_t = gt_img
            for idx, layer in enumerate(self.vgg):
                x_p = layer(x_p)
                x_t = layer(x_t)
                if idx in self.vgg_layers:
                    loss += F.l1_loss(x_p, x_t)
            l_perc = loss
        l_msssim = 0.0
        if self.ms_ssim_fn is not None:
            l_msssim = 1.0 - self.ms_ssim_fn(pred_img, gt_img, data_range=1.0, size_average=True)
        total = self.lambda_pixel * l_pixel + self.lambda_perc * l_perc + self.lambda_msssim * l_msssim
        return total

# ------------------ Example quick usage ------------------
# from smfr_net import SMFRNet, CompositeLoss, count_parameters
# model = SMFRNet(in_ch=3, base_ch=64)  # full model (base_ch=32 for lightweight)
# print("Params: {:.3f}M".format(count_parameters(model)/1e6))
# x = torch.randn(1,3,256,256)
# res = model(x)   # predicted residual
# out = x + res    # reconstructed clean image
