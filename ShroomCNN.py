
import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) block that reweights channel responses
    by explicitly modeling inter channel dependencies.

    from: Hu et al., "Squeeze-and-Excitation Networks,"
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # squeeze
        y = self.avg_pool(x)
        # excite & scale
        y = self.fc(y)
        return x * y

class GeM(nn.Module):
    """
    Generalized Mean (GeM) pooling layer.

    from : Radenović et al., "Fine-Tuning CNN Image Retrieval with No Human Annotatione"
    """
    def __init__(self, p=3.0, eps=1e-6, learnable=True):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p))) if learnable else torch.tensor(float(p))
        self.eps = eps
    def forward(self, x):
        # keep p in a safe range to stabilize training
        p = torch.clamp(self.p, 0.5, 8.0)
        # guard small values and apply p-mean pooling
        x = x.clamp(min=self.eps).pow(p)
        x = F.adaptive_avg_pool2d(x, 1).pow(1.0 / p)
        return x

class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM), (simplified for speed)

    from: Woo et al., "CBAM: Convolutional Block Attention Module" 
    """
    def __init__(self, channels, reduction=16, kernel_size=5):
        super().__init__()
        # channel attention
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )
        # spatial attention
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # channel attention: refine channel weights
        ca = self.channel_attn(x)
        x = x * ca

        # spatial attention: refine spatial focus
        max_pool, _ = x.max(dim=1, keepdim=True)
        avg_pool = x.mean(dim=1, keepdim=True)
        sa = self.spatial_attn(torch.cat([max_pool, avg_pool], dim=1))
        return x * sa


class ResidualBlock(nn.Module):
    """
    Basic residual block with two 3×3 convolutions (same channels) and GELU activations.
    Optionally inserts an SE block after the second conv.:
    This is a post-activation style block (Conv → BN → GELU).
    """
    def __init__(self, channels, use_se=False, reduction=16):
        super().__init__()
        self.use_se = use_se
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        if use_se:
            self.se = SEBlock(channels, reduction)
        self.act = nn.GELU()
    def forward(self, x):
        # residual path: Conv-BN-GELU → Conv-BN → (optional SE)
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.use_se:
            out = self.se(out)
        return self.act(x + out)



class ConvAttnBlock(nn.Module):
    """
    Modular conv block:
      1) Conv → BN → GELU
      2) optional CBAM attention
      3) optional ResidualBlock (with SE)
    input:  (B, C_in, H, W)
    output: (B, C_out, H, W)
    """
    def __init__(self, in_channels, out_channels,
                 use_cbam=False, use_res=False, use_se=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        self.attn = CBAM(out_channels) if use_cbam else nn.Identity()
        self.res  = ResidualBlock(out_channels, use_se=use_se) if use_res else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.attn(x)
        x = self.res(x)
        return x


class ShroomCNNAttentive(nn.Module):
    """
    Attentive CNN backbone + classifier head, with optional CBAM/SE blocks,
    width scaling, and selectable global pooling (Avg or GeM).

    Block config format :
        ``(out_channels, do_pool, use_cbam, use_res, use_se)``

    args:
        in_ch (int, default=3): Input image channels.
        block_cfgs (list[tuple] or None): Stage configuration; if None, a 4-stage default is used.
        mlp_units (tuple[int], default=(512,)): Hidden widths for classifier MLP.
        num_classes (int, default=162): Number of output classes.
        global_pool (str, default="avg"): "avg" for AdaptiveAvgPool2d, "gem" for GeM pooling.
        width_mult (float, default=1.0): Scales all stage widths; rounded to nearest multiple of 8.

    attributes:
        feat_dim (int): Final feature dimension before the classifier.

    """
    def __init__(self, in_ch=3, block_cfgs=None,
                 mlp_units=(512,), num_classes=162, global_pool="avg", width_mult=1.0):
        super().__init__()
        # helper to scale channels and snap to a multiple of 8
        def _scale_ch(c, wm):
            return int(round(c * wm / 8.0)) * 8

        # each tuple = (out_ch, do_pool, cbam, res, se)
        if block_cfgs is None:
            block_cfgs = [(64,True,False,True,True),
                        (128,True,False,True,True),
                        (256,True,True, True,True),
                        (320,True,False,True,True)]
            
        # width scaling
        wm = float(width_mult)
        scaled_cfgs = []
        for out_ch, do_pool, cbam, res, se in block_cfgs:
            scaled_cfgs.append((_scale_ch(out_ch, wm), do_pool, cbam, res, se))
        block_cfgs = scaled_cfgs

        # stem: also scale a bit
        stem_out = _scale_ch(64, wm)

        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, stem_out//2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(stem_out//2), nn.GELU(),
            nn.Conv2d(stem_out//2, stem_out, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(stem_out), nn.GELU(),
        )
        # stages
        ch = stem_out
        layers, ch = [], stem_out
        for out_ch, do_pool, cbam, res, se in block_cfgs:
            layers.append(ConvAttnBlock(ch, out_ch,
                                       use_cbam=cbam,
                                       use_res=res,
                                       use_se=se))
            
            if do_pool:
                # stride-2 conv instead of MaxPool for more capacity, otherwise it takes for ever
                layers.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1, bias=False))
                layers.append(nn.BatchNorm2d(out_ch))
                layers.append(nn.GELU())
            ch = out_ch
        self.model = nn.Sequential(*layers)
        # global pooling head (Avg or GeM), if you run shrooms.ipynb it will throw an error cause
        #it doesnt support gem and i havent gone back to change the function call
        self.global_pool = GeM() if global_pool.lower() == "gem" else nn.AdaptiveAvgPool2d(1)
        self.feat_dim = ch
        # classifier
        mlp = []
        in_feat = ch
        for u in mlp_units:
            mlp += [nn.Linear(in_feat, u),
                    nn.BatchNorm1d(u),
                    nn.GELU(),
                    nn.Dropout(0.4)]
            in_feat = u
        mlp.append(nn.Linear(in_feat, num_classes))
        self.classifier = nn.Sequential(*mlp)
        # kaiming init works well with GELU in practice, at least from what ive seen lol
        self.apply(self._init_weights)
        self.name = 'Attentive CNN'
    
    @staticmethod
    def _init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if getattr(m, 'bias', None) is not None: nn.init.zeros_(m.bias)
        if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
    def forward(self, x):
        x = self.stem(x)
        x = self.model(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1).contiguous()   
        return self.classifier(x)

class AuxSpeciesHead(nn.Module):
    r"""
    Lightweight linear classifier used as an auxiliary head during contrastive pretraining.
    """
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
    def forward(self, f):
        # contiguous useful for amp for speed, dont use on MPS 
        # cause that breaks everything and I have no clue why
        return self.fc(f.contiguous())