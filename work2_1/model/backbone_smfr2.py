import torch
import torch.nn as nn
import torch.nn.functional as F

from model import LayerNorm


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class FDM(nn.Module):
    def __init__(self, nc, expand=2, wavelet='bior4.4', level=2):
        super(FDM, self).__init__()
        self.process = nn.Sequential(
            nn.Conv2d(nc, expand * nc, 1, 1, 0),
            nn.BatchNorm2d(expand * nc),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(expand * nc, nc, 1, 1, 0)
        )

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(nc, nc, 1, 1, 0),
            nn.Sigmoid()
        )

        self.freq_modulator = nn.Sequential(
            nn.Conv2d(1, 8, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(8, 1, 1),
            nn.Sigmoid()
        )

        self.alpha = nn.Parameter(torch.tensor([0.7]))

    def forward(self, x):
        _, _, H, W = x.shape
        x_freq = torch.fft.rfft2(x, norm='backward')
        mag = torch.abs(x_freq)
        pha = torch.angle(x_freq)

        attn = self.attention(mag)
        mag = mag * attn

        mag_processed = self.process(mag)

        h, w = mag.shape[2], mag.shape[3]
        y_coords = torch.linspace(-1, 1, h).view(-1, 1).repeat(1, w).unsqueeze(0).unsqueeze(0).to(mag.device)
        x_coords = torch.linspace(-1, 1, w).repeat(h, 1).unsqueeze(0).unsqueeze(0).to(mag.device)
        freq_dist = torch.sqrt(x_coords ** 2 + y_coords ** 2)  # 归一化的频率距离

        freq_weight = self.freq_modulator(freq_dist)

        mag_out = mag_processed * (1.0 + 0.1 * freq_weight)  # 小幅调整，最多±10%

        real = mag_out * torch.cos(pha)
        imag = mag_out * torch.sin(pha)
        x_out = torch.complex(real, imag)
        x_out = torch.fft.irfft2(x_out, s=(H, W), norm='backward')

        alpha = torch.sigmoid(self.alpha)
        return alpha * x_out + (1 - alpha) * x


class LearnableBiasnn(nn.Module):
    def __init__(self, out_chn):
        super(LearnableBiasnn, self).__init__()
        self.bias = nn.Parameter(torch.zeros([1, out_chn, 1, 1]), requires_grad=True)

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out


class RPReLU(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.move1 = nn.Parameter(torch.zeros(hidden_size))
        self.prelu = nn.PReLU(hidden_size)
        self.move2 = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x):
        out = self.prelu((x - self.move1).transpose(-1, -2)).transpose(-1, -2) + self.move2
        return out


class MGDC(nn.Module):
    def __init__(self, in_chn, dilation1=1, dilation2=3, dilation3=5, kernel_size=3, stride=1, padding=None, groups=4):
        super(MGDC, self).__init__()
        self.move = LearnableBiasnn(in_chn)

        padding1 = dilation1 * (kernel_size - 1) // 2
        padding2 = dilation2 * (kernel_size - 1) // 2
        padding3 = dilation3 * (kernel_size - 1) // 2

        self.cov1 = nn.Conv2d(in_chn, in_chn, kernel_size, stride, padding=padding1, dilation=dilation1, groups=groups,
                              bias=True)
        self.cov2 = nn.Conv2d(in_chn, in_chn, kernel_size, stride, padding=padding2, dilation=dilation2, groups=groups,
                              bias=True)
        self.cov3 = nn.Conv2d(in_chn, in_chn, kernel_size, stride, padding=padding3, dilation=dilation3, groups=groups,
                              bias=True)

        self.norm = nn.LayerNorm(in_chn)
        self.act1 = RPReLU(in_chn)
        self.act2 = RPReLU(in_chn)
        self.act3 = RPReLU(in_chn)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.move(x)

        x1 = self.cov1(x).permute(0, 2, 3, 1).flatten(1, 2)  # (B,H*W,C)
        x1 = self.act1(x1)

        x2 = self.cov2(x).permute(0, 2, 3, 1).flatten(1, 2)
        x2 = self.act2(x2)

        x3 = self.cov3(x).permute(0, 2, 3, 1).flatten(1, 2)
        x3 = self.act3(x3)

        x = self.norm(x1 + x2 + x3)
        return x.permute(0, 2, 1).view(B, C, H, W).contiguous()


class SMDBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = MGDC(in_chn=dw_channel, groups=dw_channel // 4 if dw_channel >= 4 else 1)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm(c)
        self.norm2 = LayerNorm(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = self.sca(x) * x
        x = self.conv3(x)
        x = self.dropout1(x)
        y = inp + x * self.beta
        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        x = self.dropout2(x)
        return y + x * self.gamma


class CorrectSpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_compressed = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.sigmoid(self.conv(x_compressed))
        return x * attention_map


class CSAM(nn.Module):

    def __init__(self, channels, kernel_size=3):
        super(CSAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels, kernel_size=1, padding=0, stride=1, groups=1, bias=True),
            nn.Sigmoid()
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=(kernel_size - 1) // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_attention(x)
        ca_result = x * ca

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_input = torch.cat([avg_out, max_out], dim=1)
        sa = self.spatial_attention(spatial_input)
        sa_result = x * sa

        return ca_result + sa_result


class SMEBlock(nn.Module):

    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = MGDC(in_chn=dw_channel, groups=dw_channel // 4 if dw_channel >= 4 else 1)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.csam = CSAM(channels=dw_channel // 2)

        # SimpleGate
        self.sg = SimpleGate()

        self.freq = FDM(nc=c, expand=FFN_Expand)

        self.norm1 = LayerNorm(c)
        self.norm2 = LayerNorm(c)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x_freq = self.norm1(inp)
        x_freq = self.freq(x_freq)
        y = inp + x_freq * self.beta

        x = self.norm2(y)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = self.csam(x)
        x = self.conv3(x)
        x = self.dropout1(x)

        return y + x * self.gamma


class SMFR(nn.Module):
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[1,2,3], dec_blk_nums=[3,1,1]):
        super().__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1,
                                bias=True)
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width

        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[SMEBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[SMEBlock(chan) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[SMDBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        inp = self.check_image_size(inp)

        x = self.intro(inp)

        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_blks(x)

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)

        x = self.ending(x)
        x = x + inp

        return x[:, :, :H, :W]

    def check_image_size(self, x):

        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x