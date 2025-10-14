
import torch
import torch.nn as nn

class StripConvBlock(nn.Module):
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
        attn3 = self.relu1(self.conv1(attn3))

        return x * attn3 + res

class LPGModule(nn.Module):
    def __init__(self, in_channels=4, feature_channels=32, num_resblocks=3, k1=1, k2=19):
        super(LPGModule, self).__init__()
        self.initial_conv = nn.Sequential(
            nn.Conv2d(in_channels, feature_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.lstm = nn.LSTM(input_size=feature_channels, hidden_size=feature_channels, batch_first=True)
        self.resblocks = nn.Sequential(*[StripConvBlock(feature_channels, k1, k2) for _ in range(num_resblocks)])
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
    def __init__(self, feature_channels=32, num_resblocks=3, n=5):
        super(LPGSequence, self).__init__()
        self.lpg = LPGModule(in_channels=2*feature_channels, feature_channels=feature_channels, num_resblocks=num_resblocks)
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
        F_prev = self.rgb_to_y_channel(input_image)
        outputs = []

        for i in range(self.N):
            # print(i)
            F_n, _ = self.lpg(input_image, F_prev)
            outputs.append(F_n)
            F_prev = F_n

        # return torch.stack(outputs, dim=1)  # shape: [B, T, feature_channels, H, W]
        return F_prev

if __name__ == "__main__":
    model = LPGSequence()
    input_seq = torch.randn(1, 3, 512, 512)  # [B, T, C, H, W]
    input_y = torch.randn(1, 1, 512, 512)
    output_seq = model(input_seq)  # [B, T, 64, H, W]
    print(output_seq.shape)

