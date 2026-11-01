import torch
from torch import nn
from timm.models.layers import to_2tuple, trunc_normal_
from einops.einops import rearrange


class AgentAttention(nn.Module):
    """
    简化版的AgentAttention，专注于维度匹配
    这个模块要求输入的维度是[B, N, C]，N=H*W
    """
    def __init__(self, dim, num_patches, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,
                 sr_ratio=1, agent_num=16, **kwargs):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_patches = num_patches
        self.num_heads = num_heads
        head_dim = torch.div(dim, num_heads, rounding_mode='floor')
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.agent_num = agent_num
        self.dwc = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, padding=1, groups=dim)
        
        # 简化bias参数
        self.agent_bias = nn.Parameter(torch.zeros(1, num_heads, agent_num, 1))
        self.spatial_bias = nn.Parameter(torch.zeros(1, num_heads, 1, num_patches))
        trunc_normal_(self.agent_bias, std=.02)
        trunc_normal_(self.spatial_bias, std=.02)
        
        # 固定agent_pooling
        pool_size = int(agent_num ** 0.5)
        self.pool = nn.AdaptiveAvgPool2d(output_size=(pool_size, pool_size))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, H, W):
        b, n, c = x.shape
        num_heads = self.num_heads
        head_dim = torch.div(c, num_heads, rounding_mode='floor')
        
        # 特征提取
        q = self.q(x)  # [b, n, c]
        
        # 处理KV
        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(b, c, H, W)
            x_ = self.sr(x_).reshape(b, c, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
            n_kv = x_.shape[1]  # 可能与n不同
        else:
            kv = self.kv(x).reshape(b, -1, 2, c).permute(2, 0, 1, 3)
            n_kv = n
        k, v = kv[0], kv[1]  # [b, n_kv, c]
        
        # 生成agent_tokens
        x_2d = q.reshape(b, H, W, c).permute(0, 3, 1, 2)  # [b, c, H, W]
        agent_tokens = self.pool(x_2d)  # [b, c, sqrt(agent_num), sqrt(agent_num)]
        agent_tokens = agent_tokens.reshape(b, c, -1).permute(0, 2, 1)  # [b, agent_num, c]
        
        # 多头处理
        q = q.reshape(b, n, num_heads, head_dim).permute(0, 2, 1, 3)  # [b, num_heads, n, head_dim]
        k = k.reshape(b, n_kv, num_heads, head_dim).permute(0, 2, 1, 3)  # [b, num_heads, n_kv, head_dim]
        v = v.reshape(b, n_kv, num_heads, head_dim).permute(0, 2, 1, 3)  # [b, num_heads, n_kv, head_dim]
        agent_tokens = agent_tokens.reshape(b, self.agent_num, num_heads, head_dim).permute(0, 2, 1, 3)  # [b, num_heads, agent_num, head_dim]
        
        # 打印关键维度用于调试
        #print(f"q shape: {q.shape}")
        #print(f"k shape: {k.shape}")
        #print(f"agent_tokens shape: {agent_tokens.shape}")
        
        # Agent-Key Attention，注意确保矩阵乘法的维度匹配
        # agent_tokens: [b, num_heads, agent_num, head_dim]
        # k: [b, num_heads, n_kv, head_dim]
        # agent_attn应该是 [b, num_heads, agent_num, n_kv]
        attn_scores = (agent_tokens * self.scale) @ k.transpose(-2, -1)  # [b, num_heads, agent_num, n_kv]
        
        # 添加bias，确保其形状与注意力分数匹配
        agent_bias = self.agent_bias.expand(-1, -1, -1, n_kv)
        attn_scores = attn_scores + agent_bias
        
        agent_attn = self.softmax(attn_scores)
        agent_attn = self.attn_drop(agent_attn)
        agent_v = agent_attn @ v  # [b, num_heads, agent_num, head_dim]
        
        # Query-Agent Attention
        # q: [b, num_heads, n, head_dim]
        # agent_tokens: [b, num_heads, agent_num, head_dim]
        # q_attn应该是 [b, num_heads, n, agent_num]
        q_attn_scores = (q * self.scale) @ agent_tokens.transpose(-2, -1)  # [b, num_heads, n, agent_num]
        
        # 添加bias，确保其形状与注意力分数匹配
        spatial_bias = self.spatial_bias.expand(-1, -1, n, -1)[:, :, :, :self.agent_num]
        q_attn_scores = q_attn_scores + spatial_bias
        
        q_attn = self.softmax(q_attn_scores)
        q_attn = self.attn_drop(q_attn)
        x = q_attn @ agent_v  # [b, num_heads, n, head_dim]
        
        # 重塑并投影
        x = x.transpose(1, 2).reshape(b, n, c)  # [b, n, c]
        
        # 处理残差连接
        if self.sr_ratio > 1:
            v_spatial = v.transpose(1, 2).reshape(b, n_kv, c)
            v_spatial = v_spatial.reshape(b, -1, H // self.sr_ratio, W // self.sr_ratio, c).permute(0, 4, 2, 3, 1)
            v_spatial = v_spatial.reshape(b, c, H // self.sr_ratio, W // self.sr_ratio)
            v_spatial = nn.functional.interpolate(v_spatial, size=(H, W), mode='bilinear')
        else:
            v_spatial = v.transpose(1, 2).reshape(b, n, c)
            v_spatial = v_spatial.reshape(b, H, W, c).permute(0, 3, 1, 2)
            
        # 应用深度可分离卷积并添加到输出
        v_dwc = self.dwc(v_spatial)
        v_dwc = v_dwc.permute(0, 2, 3, 1).reshape(b, n, c)
        x = x + v_dwc
        
        # 最终投影
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


if __name__ == '__main__':
    H, W = 7, 7
    x = torch.randn(4, 49, 512).cuda()
    model = AgentAttention(512, H*W).cuda()
    out = model(x, H, W)
    print(out.shape)