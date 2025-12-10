import torch
import torch.nn as nn
import torch.nn.functional as F

"""
SegFormer Architecture Implementation (Standalone PyTorch)
Based on: "SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers"
Paper: https://arxiv.org/abs/2105.15203

Dies ist die Konfiguration für SegFormer-B0 (leichtgewichtig).
"""


# --- Helper Module: Layer Scale / DropPath ---
class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


# --- Teil 1: Encoder Blöcke (Mix Transformer - MiT) ---

class OverlapPatchEmbed(nn.Module):
    """Image to Patch Embedding mit überlappenden Patches"""

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        x = self.norm(x)
        return x, H, W


class EfficientAttention(nn.Module):
    """Efficient Self-Attention mit Spatial Reduction"""

    def __init__(self, dim, num_heads=8, qkv_bias=False, sr_ratio=1, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # Spatial Reduction reduziert die Komplexität
        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MixFFN(nn.Module):
    """Mix-FeedForward Network mit Depthwise Convolution"""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.fc1(x)
        # Reshape für Conv2d
        x = x.transpose(1, 2).reshape(B, -1, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    """Ein einzelner Transformer Block"""

    def __init__(self, dim, num_heads, mlp_ratio=4., sr_ratio=1, drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = EfficientAttention(dim, num_heads=num_heads, sr_ratio=sr_ratio, dropout=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MixFFN(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class MixVisionTransformer(nn.Module):
    """Das Backbone (Encoder)"""

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dims=[32, 64, 160, 256],
                 num_heads=[1, 2, 5, 8], mlp_ratios=[4, 4, 4, 4], sr_ratios=[8, 4, 2, 1],
                 depths=[2, 2, 2, 2], drop_rate=0., drop_path_rate=0.1):
        super().__init__()

        self.depths = depths
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # Stage 1
        self.patch_embed1 = OverlapPatchEmbed(7, 4, in_chans, embed_dims[0])
        self.block1 = nn.ModuleList([
            Block(dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
                  sr_ratio=sr_ratios[0], drop=drop_rate, drop_path=dpr[i])
            for i in range(depths[0])])
        self.norm1 = nn.LayerNorm(embed_dims[0])

        # Stage 2
        self.patch_embed2 = OverlapPatchEmbed(3, 2, embed_dims[0], embed_dims[1])
        self.block2 = nn.ModuleList([
            Block(dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1],
                  sr_ratio=sr_ratios[1], drop=drop_rate, drop_path=dpr[sum(depths[:1]) + i])
            for i in range(depths[1])])
        self.norm2 = nn.LayerNorm(embed_dims[1])

        # Stage 3
        self.patch_embed3 = OverlapPatchEmbed(3, 2, embed_dims[1], embed_dims[2])
        self.block3 = nn.ModuleList([
            Block(dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2],
                  sr_ratio=sr_ratios[2], drop=drop_rate, drop_path=dpr[sum(depths[:2]) + i])
            for i in range(depths[2])])
        self.norm3 = nn.LayerNorm(embed_dims[2])

        # Stage 4
        self.patch_embed4 = OverlapPatchEmbed(3, 2, embed_dims[2], embed_dims[3])
        self.block4 = nn.ModuleList([
            Block(dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3],
                  sr_ratio=sr_ratios[3], drop=drop_rate, drop_path=dpr[sum(depths[:3]) + i])
            for i in range(depths[3])])
        self.norm4 = nn.LayerNorm(embed_dims[3])

    def forward(self, x):
        outs = []
        B = x.shape[0]

        # Stage 1
        x, H, W = self.patch_embed1(x)
        for blk in self.block1: x = blk(x, H, W)
        x_out = self.norm1(x).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x_out)

        # Stage 2
        x, H, W = self.patch_embed2(x_out)
        for blk in self.block2: x = blk(x, H, W)
        x_out = self.norm2(x).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x_out)

        # Stage 3
        x, H, W = self.patch_embed3(x_out)
        for blk in self.block3: x = blk(x, H, W)
        x_out = self.norm3(x).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x_out)

        # Stage 4
        x, H, W = self.patch_embed4(x_out)
        for blk in self.block4: x = blk(x, H, W)
        x_out = self.norm4(x).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x_out)

        return outs


# --- Teil 2: Decoder (All-MLP Decoder) ---

class SegFormerHead(nn.Module):
    def __init__(self, in_channels, embedding_dim=256, num_classes=10):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # 1x1 Convs to unify channel dimension
        self.linear_c4 = nn.Conv2d(in_channels[3], embedding_dim, 1)
        self.linear_c3 = nn.Conv2d(in_channels[2], embedding_dim, 1)
        self.linear_c2 = nn.Conv2d(in_channels[1], embedding_dim, 1)
        self.linear_c1 = nn.Conv2d(in_channels[0], embedding_dim, 1)  # stride 4

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(embedding_dim * 4, embedding_dim, 1),
            nn.BatchNorm2d(embedding_dim),
            nn.ReLU(inplace=True)
        )

        self.classifier = nn.Conv2d(embedding_dim, num_classes, 1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, features):
        c1, c2, c3, c4 = features

        # Pass through linear layers
        c4 = self.linear_c4(c4)
        c3 = self.linear_c3(c3)
        c2 = self.linear_c2(c2)
        c1 = self.linear_c1(c1)

        # Upsample inputs to the size of c1 (stride 4)
        target_size = c1.shape[2:]

        _c4 = F.interpolate(c4, size=target_size, mode='bilinear', align_corners=False)
        _c3 = F.interpolate(c3, size=target_size, mode='bilinear', align_corners=False)
        _c2 = F.interpolate(c2, size=target_size, mode='bilinear', align_corners=False)
        _c1 = c1

        # Concatenate and fuse
        x = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        # Classify
        x = self.dropout(x)
        x = self.classifier(x)
        return x


# --- Hauptmodell: SegFormer ---

class SegFormer(nn.Module):
    def __init__(self, num_classes=10):
        super(SegFormer, self).__init__()
        self.name = "segformer-b0"

        # Backbone: MiT-B0 configuration
        self.backbone = MixVisionTransformer(
            embed_dims=[32, 64, 160, 256],
            num_heads=[1, 2, 5, 8],
            mlp_ratios=[4, 4, 4, 4],
            sr_ratios=[8, 4, 2, 1],
            depths=[2, 2, 2, 2],
            drop_path_rate=0.1
        )

        # Head
        self.decode_head = SegFormerHead(
            in_channels=[32, 64, 160, 256],
            embedding_dim=256,
            num_classes=num_classes
        )

    def forward(self, x):
        # Image size for final upsampling
        H, W = x.shape[2], x.shape[3]

        # Encoder features (List of 4 feature maps)
        features = self.backbone(x)

        # Decoder
        logits = self.decode_head(features)

        # Final upsample to original image size (stride 4 -> stride 1)
        logits = F.interpolate(logits, size=(H, W), mode='bilinear', align_corners=False)

        return logits


if __name__ == '__main__':
    ### Sanity Check (gleiche Logik wie unet)
    input_tensor = torch.randn(1, 3, 1024, 2048)

    # Modell initialisieren
    model = SegFormer(num_classes=10)
    print(f"Testing {model.name}...")

    # Output size check
    output = model(input_tensor)
    print(f"Input size:  {input_tensor.shape}")
    print(f"Output size: {output.shape}")

    # Assert dimensions match
    assert output.shape[-2:] == input_tensor.shape[-2:]
    print("Test passed: Output spatial dimensions match input.")