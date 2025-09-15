from model import BasicDrConv, BasicConv
import torch
import torch.nn as nn
import torch.nn.functional as F


class DrExpert(nn.Module):
    def __init__(self, in_channels, direction=0):
        super().__init__()
        self.expert = nn.Sequential(
            BasicDrConv(in_channels, in_channels // 2, direction=direction),
            BasicConv(in_channels // 2, in_channels // 2),
            BasicDrConv(in_channels // 2, in_channels, kernel_length=5, relu=False, direction=direction)
        )

    def forward(self, x):
        return self.expert(x)


class ExpertGate(nn.Module):
    def __init__(self, in_channels, num_experts, topk=2):
        super().__init__()
        self.num_experts = num_experts
        self.topk = topk
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.w1 = nn.Linear(in_channels, num_experts)
        self.w2 = nn.Linear(in_channels, num_experts)

    def forward(self, x):
        b, c, _, _ = x.shape
        # 结合平均池化和最大池化的特征
        avg_feat = self.avgpool(x).view(b, c)
        max_feat = self.maxpool(x).view(b, c)
        f = avg_feat + max_feat

        # 计算门控权重
        n1 = self.w1(f)
        n2 = torch.randn_like(n1) * F.softplus(self.w2(f))  # 使用带噪声的门控
        n = n1 + n2

        # 选择topk专家
        topk_val, topk_idx = torch.topk(n, self.topk, dim=-1)
        mask = torch.full_like(n, float('-inf')).scatter_(1, topk_idx, topk_val)
        w = F.softmax(mask, dim=-1)  # [B, num_experts]
        return w, topk_idx


class DrMoE(nn.Module):
    def __init__(self, in_channels, num_expert_group=2):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.num_expert_group = num_expert_group
        self.num_experts = num_expert_group * 4
        self.experts = nn.ModuleList()
        for _ in range(num_expert_group):
            for i in range(4):
                self.experts.append(DrExpert(in_channels, direction=i))
        self.gate = ExpertGate(in_channels, self.num_experts, topk=num_expert_group)

        # 用于负载均衡的辅助损失
        self.aux_loss_coef = 0.01

    def forward(self, x):
        x = self.proj(x)
        batch_size, channels, height, width = x.shape

        # 获取门控权重和选择的专家索引
        gate_weights, expert_indices = self.gate(x)  # gate_weights: [B, num_experts], expert_indices: [B, topk]

        # 初始化输出
        output = torch.zeros_like(x)

        # 计算每个专家的输出并加权组合
        for i in range(batch_size):
            # 获取当前样本选择的专家和对应权重
            sample_expert_indices = expert_indices[i]  # [topk]
            sample_gate_weights = gate_weights[i][sample_expert_indices]  # [topk]

            # 计算每个选择的专家的输出并加权组合
            for j, expert_idx in enumerate(sample_expert_indices):
                expert_output = self.experts[expert_idx](x[i].unsqueeze(0))
                output[i] += sample_gate_weights[j] * expert_output.squeeze(0)

        # 计算辅助损失（负载均衡损失）
        # aux_loss = self._load_balancing_loss(gate_weights, expert_indices)

        # return output, aux_loss
        return output

    def _load_balancing_loss(self, gate_weights, expert_indices):
        """计算负载均衡损失，防止门控网络总是选择相同的专家"""
        batch_size, num_experts = gate_weights.shape
        topk = expert_indices.shape[1]

        # 创建专家使用掩码
        expert_mask = torch.zeros(batch_size, num_experts, device=gate_weights.device)
        expert_mask.scatter_(1, expert_indices, 1)

        # 计算每个专家的使用率
        expert_usage = expert_mask.float().mean(0)  # [num_experts]

        # 计算每个专家的平均门控权重
        expert_weights = gate_weights.mean(0)  # [num_experts]

        # 负载均衡损失
        aux_loss = torch.sum(expert_usage * expert_weights) * num_experts
        return aux_loss * self.aux_loss_coef
if __name__ == '__main__':
    from torch import nn
    from torch.nn import functional as F
    import torch

    net=DrMoE(in_channels=3,num_expert_group=2).cuda()
    lq = torch.randn(1, 3, 256, 256).cuda()
    gt = torch.randn(1, 3, 256, 256).cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters())
    for i in range(1024):
        pred = net(lq)
        optimizer.zero_grad()
        loss = criterion(pred, gt)
        loss.backward()
        print(f'{i}: {loss.item()}')
        optimizer.step()
        for name,param in net.named_parameters():
            if param.requires_grad and param.grad is None:
                print(name)
