import torch
import torch.nn as nn

class SpatialTokenShift2D(nn.Module):
    """
    VRWKV 特有的 2D 空间 Token Shift
    将通道拆分，分别向上下左右四个方向平移1个像素，实现零参数的邻域特征交融
    """
    def __init__(self, channels, shift_fraction=0.25):
        super().__init__()
        self.c_shift = int(channels * shift_fraction)

    def forward(self, x):
        # x shape: [B, C, H, W]
        B, C, H, W = x.shape
        out = torch.zeros_like(x)
        cs = self.c_shift
        
        # 四个方向平移
        out[:, :cs, 1:, :] = x[:, :cs, :-1, :]          # 向下平移
        out[:, cs:2*cs, :-1, :] = x[:, cs:2*cs, 1:, :]  # 向上平移
        out[:, 2*cs:3*cs, :, 1:] = x[:, 2*cs:3*cs, :, :-1] # 向右平移
        out[:, 3*cs:4*cs, :, :-1] = x[:, 3*cs:4*cs, :, 1:] # 向左平移
        out[:, 4*cs:, :, :] = x[:, 4*cs:, :, :]          # 保留自身不变
        return out

class RWKV_SpatialMix_2D(nn.Module):
    """
    RWKV 空间混合模块（等价于 Transformer 的 Self-Attention，但具有线性的时间/空间复杂度）
    """
    def __init__(self, channels):
        super().__init__()
        self.token_shift = SpatialTokenShift2D(channels)
        self.key = nn.Linear(channels, channels, bias=False)
        self.value = nn.Linear(channels, channels, bias=False)
        self.receptance = nn.Linear(channels, channels, bias=False)
        self.output = nn.Linear(channels, channels, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        # Token Shift 混合
        x_shifted = self.token_shift(x)
        x_mixed = x * 0.5 + x_shifted * 0.5
        
        # 拉平为序列格式进行矩阵运算: [B, H*W, C]
        x_flat = x_mixed.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        k = self.key(x_flat)
        v = self.value(x_flat)
        r = torch.sigmoid(self.receptance(x_flat))
        
        # 适用于图像的高效双向线性注意力机制机制
        k_attn = torch.softmax(k, dim=1)
        kv_context = torch.bmm(k_attn.transpose(1, 2), v) # 全局上下文融合 [B, C, C]
        out = torch.bmm(k_attn, kv_context)              # 映射回每个像素 [B, H*W, C]
        
        out = self.output(out * r)
        out = out.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x + out  # 残差连接

class RWKV_ChannelMix_2D(nn.Module):
    """
    RWKV 通道混合模块（等价于 Transformer 的 FFN 反馈网络）
    """
    def __init__(self, channels):
        super().__init__()
        self.token_shift = SpatialTokenShift2D(channels)
        self.key = nn.Linear(channels, channels * 4, bias=False)
        self.value = nn.Linear(channels * 4, channels, bias=False)
        self.receptance = nn.Linear(channels, channels, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        x_shifted = self.token_shift(x)
        x_mixed = x * 0.5 + x_shifted * 0.5
        
        x_flat = x_mixed.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        r = torch.sigmoid(self.receptance(x_flat))
        k = torch.square(torch.relu(self.key(x_flat))) # GeLU 替换近似
        kv = self.value(k)
        
        out = (r * kv).reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x + out  # 残差连接

class DehazeRWKV_Real(nn.Module):
    """
    真正的 Restore-RWKV 去雾模型主体架构
    """
    def __init__(self, hidden_dim=64, num_blocks=4):
        super().__init__()
        # 浅层特征提取
        self.in_conv = nn.Conv2d(3, hidden_dim, kernel_size=3, padding=1)
        
        # 堆叠真正的 RWKV 双模块
        self.blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.blocks.append(RWKV_SpatialMix_2D(hidden_dim))
            self.blocks.append(RWKV_ChannelMix_2D(hidden_dim))
            
        # 尾部特征还原
        self.out_conv = nn.Conv2d(hidden_dim, 3, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, 3, H, W] 输入的有雾图像
        feat = self.in_conv(x)
        
        # 经过一连串 RWKV 模块演化，提取全局长距离依赖关系
        for block in self.blocks:
            feat = block(feat)
            
        # 【神级改动】：全局跳跃连接，让模型只学如何“去雾”，不学如何“画画”
        out = self.out_conv(feat) + x 
        return self.sigmoid(out)

