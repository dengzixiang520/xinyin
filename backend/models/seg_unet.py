"""
模块 2：五层一维 U-Net (1D U-Net) 心音四状态分割模型 + 生理周期状态转移 Viterbi 后处理解码
输入: [B, 4, 512] (Hilbert / 同态 / 小波 / PSD 四通道 50Hz 包络)
输出: [B, 4, 512] 逐帧四状态 logits (0=S1, 1=收缩期, 2=S2, 3=舒张期)
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 9):
        super().__init__()
        pad = k // 2
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size=k, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_ch, out_ch, kernel_size=k, padding=pad, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class PCGUNet1D(nn.Module):
    def __init__(self, in_ch: int = 4, num_classes: int = 4, base_ch: int = 16):
        super().__init__()
        c1, c2, c3, c4, c5 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8, base_ch * 16
        self.enc1 = ConvBlock1D(in_ch, c1)
        self.enc2 = ConvBlock1D(c1, c2)
        self.enc3 = ConvBlock1D(c2, c3)
        self.enc4 = ConvBlock1D(c3, c4)
        self.bottleneck = ConvBlock1D(c4, c5)

        self.pool = nn.MaxPool1d(2)

        self.up4 = nn.ConvTranspose1d(c5, c4, kernel_size=2, stride=2)
        self.dec4 = ConvBlock1D(c4 * 2, c4)
        self.up3 = nn.ConvTranspose1d(c4, c3, kernel_size=2, stride=2)
        self.dec3 = ConvBlock1D(c3 * 2, c3)
        self.up2 = nn.ConvTranspose1d(c3, c2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock1D(c2 * 2, c2)
        self.up1 = nn.ConvTranspose1d(c2, c1, kernel_size=2, stride=2)
        self.dec1 = ConvBlock1D(c1 * 2, c1)

        self.head = nn.Conv1d(c1, num_classes, kernel_size=1)

    @staticmethod
    def _match_and_cat(up_feat, skip_feat):
        if up_feat.shape[-1] != skip_feat.shape[-1]:
            up_feat = F.interpolate(up_feat, size=skip_feat.shape[-1], mode="linear", align_corners=False)
        return torch.cat([up_feat, skip_feat], dim=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.bottleneck(self.pool(e4))

        d4 = self.dec4(self._match_and_cat(self.up4(b), e4))
        d3 = self.dec3(self._match_and_cat(self.up3(d4), e3))
        d2 = self.dec2(self._match_and_cat(self.up2(d3), e2))
        d1 = self.dec1(self._match_and_cat(self.up1(d2), e1))
        return self.head(d1)


def viterbi_cardiac_decode(probs: np.ndarray) -> np.ndarray:
    """
    生理循环约束 Viterbi 后处理解码：
    强制只允许保持当前状态 (s -> s) 或按生理顺序推进 (S1->收缩期->S2->舒张期->S1)，
    彻底消除碎片化与生理不可能的跳变 (如 S1 直接跳 S2)。
    probs: [4, T] 逐帧概率 (经过 softmax)
    """
    log_p = np.log(np.clip(probs, 1e-7, 1.0))  # [4, T]
    T = log_p.shape[1]
    # 转移矩阵 (只允许自转移或顺时针下一状态)
    trans = np.full((4, 4), -1e9, dtype=np.float64)
    for s in range(4):
        trans[s, s] = np.log(0.92)
        trans[s, (s + 1) % 4] = np.log(0.08)

    dp = np.zeros((4, T), dtype=np.float64)
    ptr = np.zeros((4, T), dtype=np.int64)
    dp[:, 0] = log_p[:, 0]

    for t in range(1, T):
        for s in range(4):
            scores = dp[:, t - 1] + trans[:, s]
            best_prev = int(np.argmax(scores))
            dp[s, t] = scores[best_prev] + log_p[s, t]
            ptr[s, t] = best_prev

    path = np.zeros(T, dtype=np.int64)
    path[-1] = int(np.argmax(dp[:, -1]))
    for t in range(T - 2, -1, -1):
        path[t] = ptr[path[t + 1], t + 1]
    return path


def extract_event_centers(seq: np.ndarray, state_id: int, env_sr: int = 50):
    """从逐帧序列中提取指定状态(如 S1=0, S2=2)的连续区间中心时间戳(秒)"""
    centers = []
    in_seg = False
    start = 0
    for i, v in enumerate(seq):
        if v == state_id and not in_seg:
            in_seg = True
            start = i
        elif v != state_id and in_seg:
            in_seg = False
            if i - start >= 2:  # 至少 40ms
                centers.append(((start + i - 1) / 2.0) / env_sr)
    if in_seg and (len(seq) - start >= 2):
        centers.append(((start + len(seq) - 1) / 2.0) / env_sr)
    return centers


def compute_event_f1(pred_seq: np.ndarray, gt_seq: np.ndarray, tol_sec: float = 0.06, env_sr: int = 50):
    """事件级 S1/S2 边界命中率评估 (±60ms 容忍度，避免逐点准确率被舒张期虚高)"""
    tp = fp = fn = 0
    valid_mask = gt_seq >= 0
    if not np.any(valid_mask):
        return 0.0, 0.0, 0.0
    for st in (0, 2):  # S1 和 S2
        p_centers = extract_event_centers(pred_seq, st, env_sr)
        g_centers = extract_event_centers(gt_seq, st, env_sr)
        matched_g = set()
        for pc in p_centers:
            best_j = None
            best_d = tol_sec
            for j, gc in enumerate(g_centers):
                if j in matched_g:
                    continue
                d = abs(pc - gc)
                if d <= best_d:
                    best_d = d
                    best_j = j
            if best_j is not None:
                matched_g.add(best_j)
                tp += 1
            else:
                fp += 1
        fn += len(g_centers) - len(matched_g)
    prec = tp / (tp + fp + 1e-8)
    rec = tp / (tp + fn + 1e-8)
    f1 = 2 * prec * rec / (prec + rec + 1e-8)
    return prec, rec, f1
