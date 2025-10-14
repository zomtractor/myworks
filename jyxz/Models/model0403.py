import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import math
import numpy as np
import time
from torch import einsum

class ResBlock(nn.Module):
    def __init__(self, dim, k1, k2):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.relu0 = nn.ReLU(inplace=True)
        self.conv_spatial1 = nn.Conv2d(dim,dim,kernel_size=(k1, k2), stride=1, padding=(k1//2, k2//2), groups=dim)     
        self.conv_spatial2 = nn.Conv2d(dim,dim,kernel_size=(k2, k1), stride=1, padding=(k2//2, k1//2), groups=dim)
        self.conv1 = nn.Conv2d(dim * 2, dim, 1)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):   
        res = x
        attn = self.relu0(self.conv0(x)) + res
        attn1 = self.conv_spatial1(attn)
        attn2 = self.conv_spatial2(attn)
        attn3 = torch.cat([attn1, attn2], dim=1)
        attn2 = self.relu1(self.conv1(attn3))

        return x * attn2 + res

class LPGModule(nn.Module):
    def __init__(self, in_channels=4, feature_channels=32, num_resblocks=3, k1=1, k2=19):
        super(LPGModule, self).__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.lstm = nn.LSTM(input_size=feature_channels, hidden_size=feature_channels, batch_first=True)
        self.resblocks = nn.Sequential(*[ResBlock(feature_channels, k1, k2) for _ in range(num_resblocks)])
        self.final_conv = nn.Conv2d(feature_channels, 1, kernel_size=3, padding=1)

    def forward(self, input_image, F_prev):
        x = torch.cat([input_image, F_prev], dim=1)  # [B, 6, H, W]
        x = self.initial_conv(x)

        B, C, H, W = x.shape
        x_lstm = x.view(B, C, -1).permute(0, 2, 1)  # [B, seq_len=H*W, C]
        x_lstm_out, _ = self.lstm(x_lstm)
        x = x_lstm_out.permute(0, 2, 1).view(B, C, H, W)

        feat = self.resblocks(x)
        F_n = self.final_conv(feat)
        return F_n, feat

class LPGSequence(nn.Module):
    def __init__(self, feature_channels=32, num_resblocks=3, n=3):
        super(LPGSequence, self).__init__()
        self.lpg = LPGModule(in_channels=2, feature_channels=feature_channels, num_resblocks=num_resblocks)
        self.N = n
    def rgb_to_y_channel(self, rgb_image):
        """
        输入: rgb_image, 形状为 [B, 3, H, W]，值范围为 [0, 1] 或 [0, 255]
        输出: y_channel, 形状为 [B, 1, H, W]
        """

        # RGB to Y 公式（ITU-R BT.601）
        # Y = 0.299 * R + 0.587 * G + 0.114 * B

        R = rgb_image[:, 0:1, :, :]
        G = rgb_image[:, 1:2, :, :]
        B = rgb_image[:, 2:3, :, :]

        Y = 0.299 * R + 0.587 * G + 0.114 * B
        return Y
    def forward(self, input_image):
        """
        image_sequence: Tensor of shape [B, T, C, H, W]
        """
        y = self.rgb_to_y_channel(input_image)
        F_prev=y
        outputs = []

        for i in range(self.N):
            # print(i)
            F_n, feat = self.lpg(y, F_prev)
            outputs.append(feat)
            F_prev = F_n

        # return torch.stack(outputs, dim=1)  # shape: [B, T, feature_channels, H, W]
        return F_prev, outputs[-1]





class Uformer(nn.Module):
    def __init__(self, feature_channels=32, num_resblocks=3, n=5, **kwargs):
        super().__init__()

        self.LPG_block = LPGSequence(feature_channels=feature_channels, num_resblocks=num_resblocks, n=n)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def extra_repr(self) -> str:
        return f"embed_dim={self.embed_dim}, win_size={self.win_size}"
    
    def forward(self, x):
        # Input Projection
        mask, loc_feat = self.LPG_block(x)
        # y = self.input_proj(x)
        # y = self.pos_drop(y)
        # # Encoder
        # conv0 = self.encoderlayer_0(y)
        # # print(conv0.shape)
        # pool0 = self.dowsample_0(conv0)
        # conv1 = self.encoderlayer_1(pool0)
        # # print(conv1.shape)
        # pool1 = self.dowsample_1(conv1)
        # conv2 = self.encoderlayer_2(pool1)
        # # print(conv2.shape)
        # pool2 = self.dowsample_2(conv2)
        # conv3 = self.encoderlayer_3(pool2)
        # # print(conv3.shape)
        # pool3 = self.dowsample_3(conv3)
        # # Bottleneck
        # # print("===============")
        # conv4 = self.conv(pool3)
        # # print(conv4.shape)
        # # print("===============")

        # # Decoder
        # up0 = self.upsample_0(conv4)
        # deconv0 = torch.cat([up0, conv3], -1)
        # deconv0 = self.decoderlayer_0(deconv0)
        # # print(deconv0.shape)
        # up1 = self.upsample_1(deconv0)
        # deconv1 = torch.cat([up1, conv2], -1)
        # deconv1 = self.decoderlayer_1(deconv1)
        # # print(deconv1.shape)

        # up2 = self.upsample_2(deconv1)
        # deconv2 = torch.cat([up2, conv1], -1)
        # deconv2 = self.decoderlayer_2(deconv2)
        # # print("deconv2.shape:", deconv2.shape)
        

        # up3 = self.upsample_3(deconv2)
        # deconv3 = torch.cat([up3, conv0], -1)
        # deconv3 = self.decoderlayer_3(deconv3)
        # # print("deconv3.shape:", deconv3.shape)




        # ##
        # # m_feature = self.mask_decoderlayer(deconv3)

        # # mm = self.output_proj_mask(m_feature)
        # mm = mask
        # # print("mask.shape",mask.shape)
        # # print(deconv3.shape, m_feature.shape)
        # y_feature, m = self.mask_guide(deconv3, loc_feat)
        # yy = self.output_proj(y_feature)
        # y = yy * mm + x
        # # y=yy+x
        # yy = -yy
        # print("y.shape:", y.shape)
        # print(yy.shape)
        # print(mm.shape)
        # print(x.shape)
        # b, n, c = m_feature.shape
        # print("model show", m_deconv3.shape)
        # e = int(math.sqrt(n))
        # mask_feature_output = m_feature.transpose(-1, -2).view(b, c, e, e)
        # # Output Projection 
        # y = self.output_proj(deconv3)
        # mm = self.output_proj_mask(m_deconv3)
        # m = mm.sigmoid()
        # # mm = (mm - mm.min()) / (mm.max() - mm.min())
        # b, n, c = m_deconv3.shape
        # # print("model show", m_deconv3.shape)
        # e = int(math.sqrt(n))
        # m_deconv3 = m_deconv3.transpose(-1, -2).view(b, c, e, e)
        # # print(m_deconv3.shape)
        # print("adj model show", m_deconv3.shape)
        # # return x + y if self.dd_in == 3 else y
        # return (y * m + x, mm, m_deconv3) if self.dd_in == 3 else (y * m, mm, m_deconv3)
        # return y, mm
        return mask

    def flops(self):
        flops = 0
        # Input Projection
        flops += self.input_proj.flops(self.reso, self.reso)
        # Encoder
        flops += self.encoderlayer_0.flops() + self.dowsample_0.flops(self.reso, self.reso)
        flops += self.encoderlayer_1.flops() + self.dowsample_1.flops(self.reso // 2, self.reso // 2)
        flops += self.encoderlayer_2.flops() + self.dowsample_2.flops(self.reso // 2 ** 2, self.reso // 2 ** 2)
        flops += self.encoderlayer_3.flops() + self.dowsample_3.flops(self.reso // 2 ** 3, self.reso // 2 ** 3)

        # Bottleneck
        flops += self.conv.flops()

        # Decoder
        flops += self.upsample_0.flops(self.reso // 2 ** 4, self.reso // 2 ** 4) + self.decoderlayer_0.flops()
        flops += self.upsample_1.flops(self.reso // 2 ** 3, self.reso // 2 ** 3) + self.decoderlayer_1.flops()
        flops += self.upsample_2.flops(self.reso // 2 ** 2, self.reso // 2 ** 2) + self.decoderlayer_2.flops()
        flops += self.upsample_3.flops(self.reso // 2, self.reso // 2) + self.decoderlayer_3.flops()

        # Output Projection
        flops += self.output_proj.flops(self.reso, self.reso)
        return flops


if __name__ == "__main__":
    input_size = 512
    arch = Uformer
    depths = [1, 2, 8, 8, 4, 8, 8, 2, 1]
    model_restoration = Uformer(img_size=input_size, embed_dim=32, depths=depths,
                                win_size=8, token_projection='linear')
    # print(model_restoration)
    # from ptflops import get_model_complexity_info
    # macs, params = get_model_complexity_info(model_restoration, (3, input_size, input_size), as_strings=True,
    #                                             print_per_layer_stat=True, verbose=True)
    # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
    # print('{:<30}  {:<8}'.format('Number of parameters: ', params))
    # print('# model_restoration parameters: %.2f M' % (
    #             sum(param.numel() for param in model_restoration.parameters()) / 1e6))
    # print("number of GFLOPs: %.2f G" % (model_restoration.flops() / 1e9))
    input_ = torch.randn(1, 3, input_size, input_size)
    output_ = model_restoration(input_)
    # print(output_[0].shape)
    # print(output_[1].shape)
    # # print(output_[2].shape)
    # try:
    #     from torchviz import make_dot
    #     dot = make_dot(output_, params=dict(model_restoration.named_parameters()))
    #     dot.render('model_graph', format='png', cleanup=True)
    #     print("模型图已保存为 model_graph.png")
    # except ImportError:
    #     print("请安装 torchviz 和 graphviz 库以生成模型图。")

