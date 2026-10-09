"""Audio feature extraction for CryWatch — pure numpy/scipy, zero heavy deps.

Pipeline: wav -> mono 16kHz -> fixed-length clip -> log-mel spectrogram ->
segmented mean+std pooling -> fixed feature vector.

Designed so the whole front-end runs on a microcontroller-class CPU; the
only learned part is a tiny MLP (see model.py), which is what gets
deployed to the UGen300's Hailo-10H NPU.
"""
import numpy as np
from scipy.io import wavfile
from scipy import signal as scipy_signal

TARGET_SR = 8000       # dataset native rate; cry fundamentals live < 4 kHz
CLIP_SECONDS = 7.0     # dataset clips run ~7 s; use the whole clip
N_MELS = 40
N_FFT = 200          # 25 ms @ 8 kHz
HOP = 80             # 10 ms @ 8 kHz
N_SEGMENTS = 6       # temporal pooling segments


def load_wav(path, target_sr=TARGET_SR):
    sr, data = wavfile.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    data = data.astype(np.float64)
    # normalize int ranges to [-1, 1]
    if data.dtype == np.float64 and np.abs(data).max() > 1.5:
        data = data / 32768.0
    if sr != target_sr:
        n_out = int(len(data) * target_sr / sr)
        data = scipy_signal.resample(data, n_out)
    return data


def fix_length(data, seconds=CLIP_SECONDS, sr=TARGET_SR):
    want = int(seconds * sr)
    if len(data) > want:
        # center crop
        start = (len(data) - want) // 2
        return data[start:start + want]
    if len(data) < want:
        return np.pad(data, (0, want - len(data)))
    return data


def _mel_filterbank(sr, n_fft, n_mels):
    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(mel):
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)

    fmin, fmax = 0.0, sr / 2.0
    mel_points = np.linspace(hz_to_mel(fmin), hz_to_mel(fmax), n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(1, n_mels + 1):
        f_m1, f_m, f_p1 = bins[m - 1], bins[m], bins[m + 1]
        if f_m > f_m1:
            fb[m - 1, f_m1:f_m] = (np.arange(f_m1, f_m) - f_m1) / max(f_m - f_m1, 1)
        if f_p1 > f_m:
            fb[m - 1, f_m:f_p1] = (f_p1 - np.arange(f_m, f_p1)) / max(f_p1 - f_m, 1)
    return fb


_MEL_FB = None


def log_mel_spectrogram(data, sr=TARGET_SR, n_mels=N_MELS,
                        n_fft=N_FFT, hop=HOP):
    global _MEL_FB
    if _MEL_FB is None or _MEL_FB.shape[0] != n_mels:
        _MEL_FB = _mel_filterbank(sr, n_fft, n_mels)
    _, _, Z = scipy_signal.stft(data, fs=sr, window='hann',
                                nperseg=n_fft, noverlap=n_fft - hop,
                                nfft=n_fft, boundary='zeros', padded=True)
    power = np.abs(Z) ** 2
    mel = _MEL_FB @ power
    return np.log(mel + 1e-10)


def extract_features(path):
    """Fixed-length feature vector for one wav file."""
    data = fix_length(load_wav(path))
    mel = log_mel_spectrogram(data)          # (n_mels, n_frames)
    # per-band standardization (uses clip stats only — no dataset leakage)
    mel = (mel - mel.mean()) / (mel.std() + 1e-8)
    segs = np.array_split(mel, N_SEGMENTS, axis=1)
    feats = []
    for s in segs:
        feats.append(s.mean(axis=1))
        feats.append(s.std(axis=1))
    return np.concatenate(feats)              # (n_mels * n_segments * 2,)


def feature_dim():
    return N_MELS * N_SEGMENTS * 2
