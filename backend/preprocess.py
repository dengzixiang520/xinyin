"""
心音信号预处理、四通道包络、双时频视图 (Log-Mel + CWT)、六维 SQA 质量特征与域随机化增强模块。
"""
import numpy as np
import scipy.signal as sig
import librosa
import pywt

# numpy>=2.0 已移除 _trapz(改名为 np.trapezoid); 此处兼容 1.x / 2.x
_trapz = getattr(np, "trapz", None) or np.trapezoid


def load_and_preprocess_audio(wav_path: str, target_sr: int = 2000, lowcut: float = 25.0, highcut: float = 400.0):
    """读取音频 -> 单声道 -> 重采样 -> Hampel去尖峰 -> 二阶Butterworth带通 -> 归一化"""
    y, sr = librosa.load(wav_path, sr=target_sr, mono=True)
    if y is None or len(y) == 0:
        return np.zeros(target_sr * 5, dtype=np.float32), target_sr
    y = np.nan_to_num(y.astype(np.float32))
    raw_copy = y.copy()

    # 1. Hampel 去尖峰 (滑动中值 + 3σ，抑制听诊器摩擦脉冲)
    y = hampel_despike(y, window_size=int(target_sr * 0.05) | 1, n_sigmas=3.5)

    # 2. 二阶 Butterworth 带通滤波 (25 - 400 Hz)
    nyq = 0.5 * target_sr
    high = min(highcut, nyq * 0.95)
    b, a = sig.butter(2, [lowcut / nyq, high / nyq], btype="band")
    if len(y) > 15:
        y = sig.filtfilt(b, a, y).astype(np.float32)

    # 3. 稳健幅值归一化 (99.5分位归一化，避免单个残余毛刺压扁整体波形)
    scale = np.percentile(np.abs(y), 99.5) + 1e-8
    y = np.clip(y / scale, -1.5, 1.5).astype(np.float32)
    return y, raw_copy


def hampel_despike(x: np.ndarray, window_size: int = 101, n_sigmas: float = 3.5) -> np.ndarray:
    """快速向量化近似 Hampel 去尖峰"""
    if len(x) < window_size:
        return x
    med = sig.medfilt(x, kernel_size=min(window_size, 51))
    diff = np.abs(x - med)
    mad = np.median(diff) + 1e-8
    thresh = n_sigmas * 1.4826 * mad
    out = x.copy()
    mask = diff > thresh
    out[mask] = med[mask]
    return out


def compute_4ch_envelopes(y: np.ndarray, sr: int = 2000, env_sr: int = 50) -> np.ndarray:
    """
    计算论文 3 (Applied Sciences 2021) 的四通道心音包络特征，降采样到 50Hz，供 1D U-Net 四状态分割使用：
    Ch0: Hilbert 包络
    Ch1: 同态包络 (Homomorphic)
    Ch2: 小波包络 (db6 D3-D5 细节能量)
    Ch3: 局部 PSD 频带能量包络 (40-120Hz)
    返回 shape: [4, T_50hz]
    """
    n_out = max(1, int(round(len(y) * env_sr / sr)))

    # 1. Hilbert 包络
    analytic = sig.hilbert(y)
    env_hilb = np.abs(analytic)

    # 2. 同态包络 (对数幅值低通滤波再指数还原)
    nyq = 0.5 * sr
    b_lp, a_lp = sig.butter(2, min(8.0 / nyq, 0.9), btype="low")
    log_env = np.log(env_hilb + 1e-6)
    env_homo = np.exp(sig.filtfilt(b_lp, a_lp, log_env)) if len(y) > 15 else env_hilb

    # 3. 小波多尺度包络 (db6)
    max_lvl = min(5, pywt.dwt_max_level(len(y), pywt.Wavelet("db6").dec_len))
    if max_lvl >= 3:
        coeffs = pywt.wavedec(y, "db6", level=max_lvl)
        # 仅保留中低频心音细节层
        for i in range(len(coeffs)):
            if i == 0 or i > 3:
                coeffs[i] = np.zeros_like(coeffs[i])
        y_wav = pywt.waverec(coeffs, "db6")[: len(y)]
        env_wav = np.abs(sig.hilbert(y_wav))
    else:
        env_wav = env_hilb

    # 4. PSD 局部频带能量包络
    f, t_stft, Zxx = sig.stft(y, fs=sr, nperseg=min(256, len(y)), noverlap=min(192, len(y) // 2))
    band_mask = (f >= 40) & (f <= 150)
    psd_1d = np.mean(np.abs(Zxx[band_mask, :]) ** 2, axis=0) if np.any(band_mask) else np.mean(np.abs(Zxx) ** 2, axis=0)
    env_psd = np.interp(np.linspace(0, 1, n_out), np.linspace(0, 1, len(psd_1d)), psd_1d)

    # 统一降采样至 50Hz 并做 z-score 标准化
    envs = []
    for e in [env_hilb, env_homo, env_wav]:
        e_ds = sig.resample(e, n_out)
        e_ds = (e_ds - np.mean(e_ds)) / (np.std(e_ds) + 1e-6)
        envs.append(e_ds)
    env_psd = (env_psd - np.mean(env_psd)) / (np.std(env_psd) + 1e-6)
    envs.append(env_psd)

    return np.stack(envs, axis=0).astype(np.float32)


def extract_log_mel(y: np.ndarray, sr: int = 2000, n_mels: int = 64, n_fft: int = 256, hop_length: int = 64) -> np.ndarray:
    """提取标准 Log-Mel 时频图 [n_mels, T] 并做实例级标准化"""
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=20.0, fmax=min(800.0, sr / 2.0)
    )
    log_mel = librosa.power_to_db(mel + 1e-8, ref=np.max)
    log_mel = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-6)
    return log_mel.astype(np.float32)


def extract_cwt_scalogram(y: np.ndarray, sr: int = 2000, n_scales: int = 64, target_time_frames: int = 157) -> np.ndarray:
    """提取 Morlet 连续小波尺度图 (CWT Scalogram) [n_scales, target_time_frames]"""
    # 降采样到 1000Hz 加速 CWT (已覆盖 20-500Hz 核心心音频带)
    ds_factor = max(1, sr // 1000)
    y_ds = y[::ds_factor]
    fs = sr / ds_factor
    freqs = np.geomspace(25.0, min(450.0, fs * 0.48), n_scales)
    fc = pywt.central_frequency("morl")
    scales = fc * fs / freqs
    coef, _ = pywt.cwt(y_ds, scales, "morl", sampling_period=1.0 / fs)
    mag = np.log1p(np.abs(coef))
    # 沿时间轴线性插值对齐到与 Log-Mel 相同的帧数
    x_old = np.linspace(0, 1, mag.shape[1])
    x_new = np.linspace(0, 1, target_time_frames)
    out = np.empty((n_scales, target_time_frames), dtype=np.float32)
    for i in range(n_scales):
        out[i] = np.interp(x_new, x_old, mag[i])
    out = (out - np.mean(out)) / (np.std(out) + 1e-6)
    return out


def extract_sqa_6d_features(raw_y: np.ndarray, sr: int = 2000) -> dict:
    """
    对应项目书第 6 章的六维可解释采集质量特征提取：
    1. duration_sec: 录音时长
    2. rms_energy: 均方根能量
    3. silence_ratio: 静音帧比例
    4. clipping_ratio: 峰值削波比例
    5. snr_proxy_db: 信号主频带(25-200Hz)与高频底噪带(500-950Hz)能量比 (dB)
    6. stability_cv: 分段能量的变异系数 (越低越稳定)
    """
    n = len(raw_y)
    dur = n / float(sr)
    if n < sr // 2:
        return {
            "duration_sec": dur, "rms_energy": 0.0, "silence_ratio": 1.0,
            "clipping_ratio": 0.0, "snr_proxy_db": -20.0, "stability_cv": 5.0
        }

    rms = float(np.sqrt(np.mean(raw_y ** 2) + 1e-12))
    peak = float(np.max(np.abs(raw_y)) + 1e-12)
    clipping_ratio = float(np.mean(np.abs(raw_y) >= 0.98 * max(peak, 0.95)))

    # 帧级能量算静音比例与稳定性
    frame_len = int(sr * 0.2)
    n_frames = max(1, n // frame_len)
    frames = raw_y[: n_frames * frame_len].reshape(n_frames, frame_len)
    f_rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    silence_ratio = float(np.mean(f_rms < 0.05 * (np.median(f_rms) + 1e-6)))
    stability_cv = float(np.std(f_rms) / (np.mean(f_rms) + 1e-8))

    # 频带信噪比估计 (心音核心带 25-250Hz vs 高频干扰带 500-950Hz)
    freqs, psd = sig.welch(raw_y, fs=sr, nperseg=min(1024, n))
    sig_power = float(_trapz(psd[(freqs >= 25) & (freqs <= 250)], freqs[(freqs >= 25) & (freqs <= 250)]) + 1e-12)
    noise_mask = (freqs >= 500) & (freqs <= min(950, sr * 0.48))
    noise_power = float(_trapz(psd[noise_mask], freqs[noise_mask]) + 1e-12) if np.any(noise_mask) else 1e-6
    snr_db = float(10.0 * np.log10(sig_power / noise_power))

    return {
        "duration_sec": round(dur, 3),
        "rms_energy": round(rms, 6),
        "silence_ratio": round(silence_ratio, 4),
        "clipping_ratio": round(clipping_ratio, 5),
        "snr_proxy_db": round(snr_db, 3),
        "stability_cv": round(stability_cv, 4),
    }


def apply_stethoscope_freq_aug(y: np.ndarray, sr: int = 2000) -> np.ndarray:
    """域随机化增强：随机模拟不同听诊器(压电/MEMS/电容)的频响倾斜与带通偏移"""
    fft = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(len(y), 1.0 / sr)
    # 生成平滑随机频响增益曲线 (±6 dB)
    slope = np.random.uniform(-0.8, 0.8)
    center = np.random.uniform(80.0, 260.0)
    bw = np.random.uniform(60.0, 200.0)
    gain_db = slope * ((freqs - 150.0) / 300.0) * 4.0 + np.random.uniform(-3.0, 3.0) * np.exp(-0.5 * ((freqs - center) / bw) ** 2)
    gain = 10.0 ** (gain_db / 20.0)
    y_aug = np.fft.irfft(fft * gain, n=len(y)).astype(np.float32)
    return y_aug


def apply_lung_noise_aug(y: np.ndarray, lung_y: np.ndarray, snr_db_range=(5.0, 18.0)) -> np.ndarray:
    """生理级肺音干扰增强：叠加来自 HLS-CMDS 的真实肺音源轨"""
    if len(lung_y) < len(y):
        reps = int(np.ceil(len(y) / len(lung_y)))
        lung_y = np.tile(lung_y, reps)
    start = np.random.randint(0, max(1, len(lung_y) - len(y)))
    seg_lung = lung_y[start : start + len(y)]
    p_sig = np.mean(y ** 2) + 1e-10
    p_noise = np.mean(seg_lung ** 2) + 1e-10
    snr_db = np.random.uniform(*snr_db_range)
    scale = np.sqrt(p_sig / (p_noise * (10.0 ** (snr_db / 10.0))))
    return (y + scale * seg_lung).astype(np.float32)


def apply_spec_augment(spec: np.ndarray, freq_mask_param: int = 8, time_mask_param: int = 16) -> np.ndarray:
    """SpecAugment 时频随机遮挡"""
    out = spec.copy()
    n_freq, n_time = out.shape
    f = np.random.randint(0, min(freq_mask_param, n_freq // 4) + 1)
    if f > 0:
        f0 = np.random.randint(0, n_freq - f)
        out[f0 : f0 + f, :] = 0.0
    t = np.random.randint(0, min(time_mask_param, n_time // 4) + 1)
    if t > 0:
        t0 = np.random.randint(0, n_time - t)
        out[:, t0 : t0 + t] = 0.0
    return out
