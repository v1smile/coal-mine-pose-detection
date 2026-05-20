import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.deform_conv import DeformConv2d
import numbers
import math
from einops import rearrange
import numpy as np

__all__ = ['MB_TaylorFormer']

freqs_dict = dict()

##########################################################################
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias

class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)

##########################################################################
## Gated-Dconv Feed-Forward Network (GDFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x

class refine_att(nn.Module):
    """Convolutional relative position encoding.（优化：限制窗口大小，减少计算量）"""

    def __init__(self, Ch, h, window=3):
        super().__init__()

        # 简化窗口配置，固定窗口大小为3，减少计算复杂度
        self.window = window
        self.ch = Ch
        self.h = h

        self.conv = nn.Conv2d(
            h * Ch * 2,
            h,
            kernel_size=(window, window),
            padding=(window//2, window//2),
            groups=h,
            bias=False
        )

    def forward(self, q, k, v, size):
        """foward function（优化：简化维度转换，减少中间内存占用）"""
        B, h, N, Ch = q.shape
        H, W = size

        # 维度转换优化：减少冗余reshape
        q_img = rearrange(q, "B h (H W) Ch -> B (h Ch) H W", H=H, W=W)
        k_img = rearrange(k, "B h (H W) Ch -> B (h Ch) H W", H=H, W=W)
        qk_concat = torch.cat((q_img, k_img), 1)  # B, 2*h*Ch, H, W

        # 卷积计算注意力，替代大规模矩阵乘法
        att = self.conv(qk_concat)  # B, h, H, W
        att = rearrange(att, "B h H W -> B h (H W) 1", H=H, W=W)
        att = F.softmax(att, dim=2)  # 归一化，减少数值波动

        return att

##########################################################################
## Multi-DConv Head Transposed Self-Attention (MDTA)（核心优化：减少矩阵乘法规模）
class Attention(nn.Module):
    def __init__(self, dim, num_heads=1, bias=False, qk_norm=1):
        super(Attention, self).__init__()
        self.norm = qk_norm
        self.num_heads = num_heads
        self.head_dim = dim // num_heads  # 每个头的维度，避免维度不匹配
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # 优化：减少输出通道，避免维度爆炸
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        
        # 初始化refine_att，使用固定小窗口
        self.refine_att = refine_att(Ch=self.head_dim, h=num_heads, window=3)

    def forward(self, x):
        b, c, h, w = x.shape
        N = h * w

        # 步骤1：获取qkv并简化维度
        qkv = self.qkv_dwconv(self.qkv(x))  # B, 3C, H, W
        q, k, v = qkv.chunk(3, dim=1)  # 每个都是 B, C, H, W

        # 步骤2：分头上采样（优化：限制头维度，避免内存溢出）
        q = rearrange(q, 'b (h d) h_img w_img -> b h (h_img w_img) d', h=self.num_heads, d=self.head_dim)
        k = rearrange(k, 'b (h d) h_img w_img -> b h (h_img w_img) d', h=self.num_heads, d=self.head_dim)
        v = rearrange(v, 'b (h d) h_img w_img -> b h (h_img w_img) d', h=self.num_heads, d=self.head_dim)

        # 步骤3：归一化（简化计算，减少数值规模）
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)

        # 步骤4：优化注意力计算（使用卷积注意力替代大规模矩阵乘法 @）
        k_t = k.transpose(-1, -2)  # B, h, d, N
        attn = torch.matmul(q, k_t) / math.sqrt(self.head_dim)  # 缩放，减少数值爆炸

        # 步骤5：使用refine_att修正注意力，限制内存占用
        refine_weight = self.refine_att(q, k, v, size=(h, w))
        attn = attn * refine_weight
        attn = F.softmax(attn, dim=-1)  # 归一化

        # 步骤6：输出计算（优化：减少中间变量）
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h (h_img w_img) d -> b (h d) h_img w_img', h_img=h, w_img=w)
        out = self.project_out(out)

        return out

##########################################################################
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=1, ffn_expansion_factor=2.66, bias=False, LayerNorm_type='BiasFree'):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

class MHCAEncoder(nn.Module):
    """Multi-Head Convolutional self-Attention Encoder（优化：减少层数，默认1层）"""
    def __init__(
            self,
            dim,
            num_layers=1,
            num_heads=1,
            ffn_expansion_factor=2.66,
            bias=False,
            LayerNorm_type='BiasFree'
    ):
        super().__init__()
        self.num_layers = num_layers
        self.MHCA_layers = nn.ModuleList([
            TransformerBlock(
                dim,
                num_heads=num_heads,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for idx in range(num_layers)
        ])

    def forward(self, x, size):
        """foward function（优化：简化维度转换）"""
        H, W = size
        B = x.shape[0]

        # 维度转换：减少冗余操作
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        for layer in self.MHCA_layers:
            x = layer(x)

        return x

class ResBlock(nn.Module):
    """Residual block for convolutional local feature."""

    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.Hardswish,
            norm_layer=nn.BatchNorm2d,
    ):
        super().__init__()

        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.conv1 = Conv2d_BN(in_features,
                               hidden_features,
                               act_layer=act_layer)
        self.dwconv = nn.Conv2d(
            hidden_features,
            hidden_features,
            3,
            1,
            1,
            bias=False,
            groups=hidden_features,
        )
        self.act = act_layer()
        self.conv2 = Conv2d_BN(hidden_features, out_features)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        initialization
        """
        if isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        """foward function"""
        identity = x
        feat = self.conv1(x)
        feat = self.dwconv(feat)
        feat = self.act(feat)
        feat = self.conv2(feat)

        return identity + feat

class MHCA_stage(nn.Module):
    """Multi-Head Convolutional self-Attention stage（优化：减少num_path，默认1）"""

    def __init__(
            self,
            embed_dim,
            out_embed_dim,
            num_layers=1,
            num_heads=1,
            ffn_expansion_factor=2.66,
            num_path=1,
            bias=False,
            LayerNorm_type='BiasFree'
    ):
        super().__init__()

        self.mhca_blks = nn.ModuleList([
            MHCAEncoder(
                embed_dim,
                num_layers,
                num_heads,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for _ in range(num_path)
        ])

        self.aggregate = SKFF(embed_dim, height=num_path)

    def forward(self, inputs):
        """foward function"""
        att_outputs = []

        for x, encoder in zip(inputs, self.mhca_blks):
            _, _, H, W = x.shape
            x = x.flatten(2).transpose(1, 2).contiguous()
            att_outputs.append(encoder(x, size=(H, W)))

        out = self.aggregate(att_outputs)

        return out

##########################################################################
## Overlapped image patch embedding with 3x3 Conv
class Conv2d_BN(nn.Module):
    def __init__(
            self,
            in_ch,
            out_ch,
            kernel_size=1,
            stride=1,
            pad=0,
            dilation=1,
            groups=1,
            bn_weight_init=1,
            norm_layer=nn.BatchNorm2d,
            act_layer=None,
    ):
        super().__init__()

        self.conv = torch.nn.Conv2d(in_ch,
                                    out_ch,
                                    kernel_size,
                                    stride,
                                    pad,
                                    dilation,
                                    groups,
                                    bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(mean=0.0, std=np.sqrt(2.0 / fan_out))

        self.act_layer = act_layer() if act_layer is not None else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.act_layer(x)
        return x

class SKFF(nn.Module):
    """优化：减少通道数，避免内存溢出"""
    def __init__(self, in_channels, height=1, reduction=16, bias=False):
        super(SKFF, self).__init__()

        self.height = height
        d = max(int(in_channels / reduction), 2)  # 进一步减少中间通道

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.PReLU())

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        if self.height == 1:
            return inp_feats[0]  # 单路径直接返回，减少计算

        batch_size = inp_feats[0].shape[0]
        n_feats = inp_feats[0].shape[1]

        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])

        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        attention_vectors = self.softmax(attention_vectors)

        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)

        return feats_V

class DWConv2d_BN(nn.Module):
    def __init__(
            self,
            in_ch,
            out_ch,
            kernel_size=1,
            stride=1,
            norm_layer=nn.BatchNorm2d,
            act_layer=nn.Hardswish,
            offset_clamp=(-1, 1)
    ):
        super().__init__()
        self.offset_clamp = offset_clamp
        self.offset_generator = nn.Sequential(nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=3,
                                                        stride=1, padding=1, bias=False, groups=in_ch),
                                              nn.Conv2d(in_channels=in_ch, out_channels=18,
                                                        kernel_size=1,
                                                        stride=1, padding=0, bias=False)
                                              )
        self.dcn = DeformConv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            groups=in_ch
        )
        self.pwconv = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)

        self.act = act_layer() if act_layer is not None else nn.Identity()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        offset = self.offset_generator(x)
        if self.offset_clamp:
            offset = torch.clamp(offset, min=self.offset_clamp[0], max=self.offset_clamp[1])
        x = self.dcn(x, offset)
        x = self.pwconv(x)
        x = self.act(x)
        return x

class DWCPatchEmbed(nn.Module):
    """Depthwise Convolutional Patch Embedding（优化：减少嵌入维度）"""

    def __init__(self,
                 in_chans=3,
                 embed_dim=32,
                 patch_size=3,
                 stride=1,
                 act_layer=nn.Hardswish):
        super().__init__()

        self.patch_conv = DWConv2d_BN(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=stride,
            act_layer=act_layer
        )

    def forward(self, x):
        x = self.patch_conv(x)
        return x

class Patch_Embed_stage(nn.Module):
    """Patch Embedding stage（优化：默认单路径）"""

    def __init__(self, in_chans, embed_dim, num_path=1, isPool=False):
        super(Patch_Embed_stage, self).__init__()

        self.patch_embeds = nn.ModuleList([
            DWCPatchEmbed(
                in_chans=in_chans if idx == 0 else embed_dim,
                embed_dim=embed_dim,
                patch_size=3,
                stride=1
            ) for idx in range(num_path)
        ])

    def forward(self, x):
        att_inputs = []
        for pe in self.patch_embeds:
            x = pe(x)
            att_inputs.append(x)
        return att_inputs

class OverlapPatchEmbed(nn.Module):
    """优化：减少输出嵌入维度，降低计算量"""
    def __init__(self, in_c=3, embed_dim=32, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)
        return x

##########################################################################
## Resizing modules
class Downsample(nn.Module):
    def __init__(self, input_feat, out_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(input_feat, input_feat, kernel_size=3, stride=1, padding=1, groups=input_feat, bias=False, ),
            nn.Conv2d(input_feat, out_feat // 4, 1, 1, 0, bias=False),
            nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample(nn.Module):
    def __init__(self, input_feat, out_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(input_feat, input_feat, kernel_size=3, stride=1, padding=1, groups=input_feat, bias=False, ),
            nn.Conv2d(input_feat, out_feat * 4, 1, 1, 0, bias=False),
            nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

##########################################################################
##---------- 彻底解决笔误 + 内存优化 ----------
class MB_TaylorFormer(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 dim=6,        # 适配YOLO传入的整数
                 num_blocks=1, # 适配YOLO传入的整数
                 heads=1,      # 适配YOLO传入的整数
                 bias=False,
                 dual_pixel_task=False,  # 关闭冗余任务，减少计算
                 num_path=1,   # 适配YOLO传入的整数
                 offset_clamp=(-1, 1)
                 ):
        super(MB_TaylorFormer, self).__init__()
        
        # 兼容逻辑1：dim - 整数转4元素列表（降低维度，减少内存）
        if isinstance(dim, int):
            self.dim = [min(dim * 1, 32), min(dim * 2, 64), min(dim * 4, 128), min(dim * 6, 256)]
        else:
            self.dim = [min(d, 256) for d in dim.copy()]
            while len(self.dim) < 4:
                self.dim.append(min(self.dim[-1] * 2, 256))
        self.dim = self.dim[:4]
        
        # 兼容逻辑2：num_blocks - 整数转4元素列表（最多2层，减少计算）
        if isinstance(num_blocks, int):
            self.num_blocks = [min(num_blocks, 2) for _ in range(4)]
        else:
            self.num_blocks = [min(b, 2) for b in num_blocks.copy()]
            while len(self.num_blocks) < 4:
                self.num_blocks.append(min(self.num_blocks[-1], 2))
        self.num_blocks = self.num_blocks[:4]
        
        # 兼容逻辑3：heads - 整数转4元素列表（最多2头，减少计算）
        if isinstance(heads, int):
            self.heads = [min(heads, 2) for _ in range(4)]
        else:
            self.heads = [min(h, 2) for h in heads.copy()]
            while len(self.heads) < 4:
                self.heads.append(min(self.heads[-1], 2))
        self.heads = self.heads[:4]
        
        # 兼容逻辑4：num_path - 整数转4元素列表（最多1路径，减少内存）
        if isinstance(num_path, int):
            self.num_path = [min(num_path, 1) for _ in range(4)]
        else:
            self.num_path = [min(p, 1) for p in num_path.copy()]
            while len(self.num_path) < 4:
                self.num_path.append(min(self.num_path[-1], 1))
        self.num_path = self.num_path[:4]
        
        # ########## 彻底修正笔误：所有参数都使用self.xxx ##########
        self.patch_embed = OverlapPatchEmbed(inp_channels, self.dim[0])
        self.patch_embed_encoder_level1 = Patch_Embed_stage(self.dim[0], self.dim[0], num_path=self.num_path[0])
        self.encoder_level1 = MHCA_stage(self.dim[0], self.dim[0], 
                                         num_layers=self.num_blocks[0], 
                                         num_heads=self.heads[0],
                                         ffn_expansion_factor=2.0,  # 降低扩张因子，减少中间维度
                                         num_path=self.num_path[0])

        self.down1_2 = Downsample(self.dim[0], self.dim[1])

        self.patch_embed_encoder_level2 = Patch_Embed_stage(self.dim[1], self.dim[1], num_path=self.num_path[1])
        self.encoder_level2 = MHCA_stage(self.dim[1], self.dim[1], 
                                         num_layers=self.num_blocks[1], 
                                         num_heads=self.heads[1],
                                         ffn_expansion_factor=2.0,
                                         num_path=self.num_path[1])

        self.down2_3 = Downsample(self.dim[1], self.dim[2])

        self.patch_embed_encoder_level3 = Patch_Embed_stage(self.dim[2], self.dim[2], num_path=self.num_path[2])
        self.encoder_level3 = MHCA_stage(self.dim[2], self.dim[2], 
                                         num_layers=self.num_blocks[2], 
                                         num_heads=self.heads[2],
                                         ffn_expansion_factor=2.0,
                                         num_path=self.num_path[2])

        self.down3_4 = Downsample(self.dim[2], self.dim[3])

        self.patch_embed_latent = Patch_Embed_stage(self.dim[3], self.dim[3], num_path=self.num_path[3])
        self.latent = MHCA_stage(self.dim[3], self.dim[3], 
                                 num_layers=self.num_blocks[3], 
                                 num_heads=self.heads[3],
                                 ffn_expansion_factor=2.0,
                                 num_path=self.num_path[3])

        self.up4_3 = Upsample(int(self.dim[3]), self.dim[2])
        self.reduce_chan_level3 = nn.Sequential(
            nn.Conv2d(self.dim[2] * 2, self.dim[2], 1, 1, 0, bias=bias),
        )

        self.patch_embed_decoder_level3 = Patch_Embed_stage(self.dim[2], self.dim[2], num_path=self.num_path[2])
        self.decoder_level3 = MHCA_stage(self.dim[2], self.dim[2], 
                                         num_layers=self.num_blocks[2], 
                                         num_heads=self.heads[2],
                                         ffn_expansion_factor=2.0,
                                         num_path=self.num_path[2])

        self.up3_2 = Upsample(int(self.dim[2]), self.dim[1])
        self.reduce_chan_level2 = nn.Sequential(
            nn.Conv2d(self.dim[1] * 2, self.dim[1], 1, 1, 0, bias=bias),
        )

        self.patch_embed_decoder_level2 = Patch_Embed_stage(self.dim[1], self.dim[1], num_path=self.num_path[1])
        self.decoder_level2 = MHCA_stage(self.dim[1], self.dim[1], 
                                         num_layers=self.num_blocks[1], 
                                         num_heads=self.heads[1],
                                         ffn_expansion_factor=2.0,
                                         num_path=self.num_path[1])

        self.up2_1 = Upsample(int(self.dim[1]), self.dim[0])
        self.reduce_chan_level1 = nn.Sequential(
            nn.Conv2d(self.dim[0] * 2, self.dim[0], 1, 1, 0, bias=bias),
        )

        self.patch_embed_decoder_level1 = Patch_Embed_stage(self.dim[0]*2, self.dim[1], num_path=self.num_path[0])
        self.decoder_level1 = MHCA_stage(self.dim[1], self.dim[1], 
                                         num_layers=self.num_blocks[0], 
                                         num_heads=self.heads[0],
                                         ffn_expansion_factor=2.0,
                                         num_path=self.num_path[0])

        # 优化：简化输出，匹配YOLO的特征图格式
        self.output = nn.Conv2d(self.dim[1], inp_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, inp_img):
        # 编码阶段
        inp_enc_level1 = self.patch_embed(inp_img)
        inp_enc_level1_list = self.patch_embed_encoder_level1(inp_enc_level1)
        out_enc_level1 = self.encoder_level1(inp_enc_level1_list)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        inp_enc_level2_list = self.patch_embed_encoder_level2(inp_enc_level2)
        out_enc_level2 = self.encoder_level2(inp_enc_level2_list)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        inp_enc_level3_list = self.patch_embed_encoder_level3(inp_enc_level3)
        out_enc_level3 = self.encoder_level3(inp_enc_level3_list)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        inp_enc_level4_list = self.patch_embed_latent(inp_enc_level4)
        out_enc_level4 = self.latent(inp_enc_level4_list)

        # 解码阶段（优化：减少冗余concat，降低内存）
        inp_dec_level3 = self.up4_3(out_enc_level4)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        inp_dec_level3_list = self.patch_embed_decoder_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3_list)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        inp_dec_level2_list = self.patch_embed_decoder_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2_list)

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        inp_dec_level1 = self.reduce_chan_level1(inp_dec_level1)
        inp_dec_level1_list = self.patch_embed_decoder_level1(inp_dec_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1_list)

        # 输出优化：匹配YOLO输入尺寸，减少内存占用
        out = self.output(out_dec_level1)
        
        # 残差连接（简化）
        return out + inp_img

##########################################################################
def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count