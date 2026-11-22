"""
模块 3：论文核心分类与域泛化模型库
包含：
1. BaselineCNN: 4层轻量2D-CNN (对齐项目书与论文1基线)
2. DualViewDGResNetCBAM: 本论文主力创新模型
   - 支持单视图(Log-Mel) / 双视图(Log-Mel + CWT 小波尺度图)自适应门控融合
   - CBAM 通道+空间注意力模块
   - MixStyle 跨设备频谱特征风格随机化 (抗听诊器捷径学习)
   - GRL 梯度反转域对抗分支 (Domain-Adversarial Learning)
   - MC-Dropout 推理期不确定性估计 + Grad-CAM 特征图钩子
3. CRNNCBAM: 2D-CNN + BiGRU + Attention Pooling 对比模型 (对齐论文2)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


# ---------------- 基础组件：CBAM / MixStyle / GRL ----------------
class CBAM(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        mid = max(4, channels // reduction)
        self.mlp = nn.Sequential(
            nn.Linear(channels, mid, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels, bias=False),
        )
        self.spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        # Channel Attention
        b, c, _, _ = x.shape
        avg_p = F.adaptive_avg_pool2d(x, 1).view(b, c)
        max_p = F.adaptive_max_pool2d(x, 1).view(b, c)
        ca = torch.sigmoid(self.mlp(avg_p) + self.mlp(max_p)).view(b, c, 1, 1)
        x = x * ca
        # Spatial Attention
        avg_s = torch.mean(x, dim=1, keepdim=True)
        max_s, _ = torch.max(x, dim=1, keepdim=True)
        sa = torch.sigmoid(self.spatial(torch.cat([avg_s, max_s], dim=1)))
        return x * sa


class MixStyle(nn.Module):
    """跨听诊器域特征统计量风格混合 (仅在训练期激活，破坏模型对特定设备底噪的捷径依赖)"""
    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6):
        super().__init__()
        self.p = p
        self.alpha = alpha
        self.eps = eps

    def forward(self, x):
        if not self.training or torch.rand(1).item() > self.p or x.size(0) < 2:
            return x
        b = x.size(0)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True, unbiased=False)
        sig = (var + self.eps).sqrt()
        x_normed = (x - mu) / sig

        lmda = torch.distributions.Beta(self.alpha, self.alpha).sample((b, 1, 1, 1)).to(x.device)
        perm = torch.randperm(b, device=x.device)
        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu * lmda + mu2 * (1 - lmda)
        sig_mix = sig * lmda + sig2 * (1 - lmda)
        return x_normed * sig_mix + mu_mix


class _GradReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None


def grad_reverse(x, lambd: float = 1.0):
    return _GradReverseFn.apply(x, lambd)


# ---------------- 模型 1：4层基线 2D-CNN ----------------
class BaselineCNN(nn.Module):
    def __init__(self, num_classes: int = 2, num_domains: int = 10, dropout: float = 0.3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
        )
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x_mel, x_cwt=None, grl_lambda: float = 0.0):
        fmap = self.features(x_mel)
        emb = F.adaptive_avg_pool2d(fmap, 1).flatten(1)
        logits = self.classifier(self.drop(emb))
        return {"logits": logits, "domain_logits": None, "embedding": emb, "fmap": fmap}


# ---------------- 模型 2：主力创新模型 DualViewDGResNetCBAM ----------------
def _build_resnet18_stem_and_layers(use_cbam: bool = True, use_mixstyle: bool = True, mixstyle_p: float = 0.5):
    base = resnet18(weights=None)
    conv1 = nn.Conv2d(1, 64, kernel_size=5, stride=2, padding=2, bias=False)
    bn1 = base.bn1
    relu = base.relu
    maxpool = base.maxpool
    layer1 = base.layer1
    layer2 = base.layer2
    layer3 = base.layer3
    layer4 = base.layer4
    cbam3 = CBAM(256) if use_cbam else nn.Identity()
    cbam4 = CBAM(512) if use_cbam else nn.Identity()
    ms1 = MixStyle(p=mixstyle_p) if use_mixstyle else nn.Identity()
    ms2 = MixStyle(p=mixstyle_p) if use_mixstyle else nn.Identity()
    return nn.ModuleDict({
        "conv1": conv1, "bn1": bn1, "relu": relu, "maxpool": maxpool,
        "layer1": layer1, "ms1": ms1,
        "layer2": layer2, "ms2": ms2,
        "layer3": layer3, "cbam3": cbam3,
        "layer4": layer4, "cbam4": cbam4,
    })


class ResNetEncoder(nn.Module):
    def __init__(self, use_cbam: bool = True, use_mixstyle: bool = True, mixstyle_p: float = 0.5):
        super().__init__()
        self.m = _build_resnet18_stem_and_layers(use_cbam, use_mixstyle, mixstyle_p)

    def forward(self, x):
        x = self.m["maxpool"](self.m["relu"](self.m["bn1"](self.m["conv1"](x))))
        x = self.m["ms1"](self.m["layer1"](x))
        x = self.m["ms2"](self.m["layer2"](x))
        x = self.m["cbam3"](self.m["layer3"](x))
        fmap = self.m["cbam4"](self.m["layer4"](x))
        emb = F.adaptive_avg_pool2d(fmap, 1).flatten(1)
        return emb, fmap


class DualViewDGResNetCBAM(nn.Module):
    """
    核心论文创新模型：
    - 可灵活开关 use_cwt / use_cbam / use_mixstyle / use_grl 用于完整消融实验 (Table 4)
    - 支持 enable_mc_dropout() 推理期多次前向采样计算预测方差/熵不确定度
    """
    def __init__(
        self,
        num_classes: int = 2,
        num_domains: int = 12,
        use_cwt: bool = True,
        use_cbam: bool = True,
        use_mixstyle: bool = True,
        use_grl: bool = True,
        dropout: float = 0.35,
    ):
        super().__init__()
        self.use_cwt = use_cwt
        self.use_grl = use_grl

        self.enc_mel = ResNetEncoder(use_cbam=use_cbam, use_mixstyle=use_mixstyle)
        if use_cwt:
            self.enc_cwt = ResNetEncoder(use_cbam=use_cbam, use_mixstyle=use_mixstyle)
            # 自适应门控双视图融合
            self.gate = nn.Sequential(
                nn.Linear(1024, 256),
                nn.ReLU(inplace=True),
                nn.Linear(256, 2),
                nn.Softmax(dim=-1),
            )
            self.proj = nn.Sequential(nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True))
        else:
            self.proj = nn.Sequential(nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(inplace=True))

        self.mc_drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(256, num_classes)

        if use_grl:
            self.domain_disc = nn.Sequential(
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(128, num_domains),
            )

    def forward(self, x_mel, x_cwt=None, grl_lambda: float = 0.2):
        emb_mel, fmap_mel = self.enc_mel(x_mel)
        if self.use_cwt and x_cwt is not None:
            emb_cwt, _ = self.enc_cwt(x_cwt)
            w = self.gate(torch.cat([emb_mel, emb_cwt], dim=1))  # [B, 2]
            fused = w[:, 0:1] * emb_mel + w[:, 1:2] * emb_cwt
        else:
            fused = emb_mel

        emb = self.proj(fused)
        logits = self.classifier(self.mc_drop(emb))

        domain_logits = None
        if self.use_grl and grl_lambda > 0:
            rev_emb = grad_reverse(emb, grl_lambda)
            domain_logits = self.domain_disc(rev_emb)

        return {"logits": logits, "domain_logits": domain_logits, "embedding": emb, "fmap": fmap_mel}


# ---------------- 模型 3：CRNN-CBAM (2D-CNN + BiGRU + Attention Pooling) ----------------
class CRNNCBAM(nn.Module):
    def __init__(self, num_classes: int = 2, num_domains: int = 12, dropout: float = 0.35):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), CBAM(64), nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), CBAM(128), nn.MaxPool2d((2, 1)),
        )
        self.gru = nn.GRU(128, 128, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.attn = nn.Linear(256, 1)
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x_mel, x_cwt=None, grl_lambda: float = 0.0):
        fmap = self.cnn(x_mel)  # [B, 128, F', T']
        seq = fmap.mean(dim=2).transpose(1, 2)  # [B, T', 128]
        out, _ = self.gru(seq)  # [B, T', 256]
        weights = torch.softmax(self.attn(out), dim=1)
        emb = torch.sum(out * weights, dim=1)
        logits = self.classifier(self.drop(emb))
        return {"logits": logits, "domain_logits": None, "embedding": emb, "fmap": fmap}


def build_model(model_name: str, num_classes: int = 2, num_domains: int = 12, **kwargs):
    if model_name == "baseline_cnn":
        return BaselineCNN(num_classes=num_classes, num_domains=num_domains)
    elif model_name == "crnn_cbam":
        return CRNNCBAM(num_classes=num_classes, num_domains=num_domains)
    elif model_name == "dg_resnet_cbam":
        return DualViewDGResNetCBAM(num_classes=num_classes, num_domains=num_domains, **kwargs)
    else:
        raise ValueError(f"Unknown model: {model_name}")
