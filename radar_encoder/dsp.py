# -*- coding: utf-8 -*-
"""Fixed DSP for M4Human raw I/Q -> range-Doppler-angle tensor.

Validated on p2 action 10:
  - direct IFFT is dominated by near-range leakage (peak at 0 m)
  - subtracting the clip mean (static clutter removal) moves the peak to ~2.17 m,
    matching the RPC cluster at 2.3-3.4 m
  - pairs[:,0] = 16 antenna pairs, pairs[:,1] = 20 chirps
"""

import json

import numpy as np

N_PAIR = 16
N_CHIRP = 20
N_FREQ = 151
RANGE_BIN_TRUE = 3e8 / (2 * 1.6e9)  # 9.375 cm true range resolution


def load_pairs(config_path):
    cfg = json.load(open(config_path))
    return np.asarray(cfg["pairs"], dtype=np.int64)


def iq_to_rda(iq, pairs, n_range=256, subtract_bg=True, eps=1e-6):
    """Convert a clip of raw I/Q into a range-Doppler-angle tensor.

    Args:
        iq: complex64 [T, 320, 151]  (T frames, 16 pairs x 20 chirps, 151 samples)
        pairs: int array [320, 2] from radar_config.json (col0=pair, col1=chirp)
        n_range: FFT size for the range axis (151 -> 256 zero-padded)
        subtract_bg: remove the clip mean (static clutter / direct coupling)

    Returns:
        float32 [T, 2, n_range, 20, 16] with channels (log-magnitude, phase/pi)
    """
    iq = np.asarray(iq, dtype=np.complex64)
    t, rows, nfreq = iq.shape
    assert rows == pairs.shape[0] and nfreq == N_FREQ, (iq.shape, pairs.shape)

    p0, p1 = pairs[:, 0], pairs[:, 1]
    pmin, cmin = int(p0.min()), int(p1.min())
    n_pair = int(p0.max()) - pmin + 1
    n_chirp = int(p1.max()) - cmin + 1

    if subtract_bg:
        iq = iq - iq.mean(axis=0, keepdims=True)

    cube = np.zeros((t, n_chirp, n_pair, nfreq), dtype=np.complex64)
    for r in range(rows):
        cube[:, p1[r] - cmin, p0[r] - pmin, :] = iq[:, r, :]

    win = np.hanning(nfreq).astype(np.float32)
    cube *= win

    rc = np.fft.ifft(cube, n=n_range, axis=-1).astype(np.complex64)
    rd = np.fft.fftshift(np.fft.fft(rc, axis=1), axes=1)
    rda = np.fft.fftshift(np.fft.fft(rd, axis=2), axes=2)

    # [T, chirp, pair, range] -> [T, range, chirp, pair]
    rda = np.transpose(rda, (0, 3, 1, 2))

    mag = np.log1p(np.abs(rda)).astype(np.float32)
    mag = (mag - mag.mean()) / (mag.std() + eps)
    phase = (np.angle(rda) / np.pi).astype(np.float32)
    return np.stack([mag, phase], axis=1)


def range_profile(iq, pairs, n_range=256, subtract_bg=True):
    """Linear range profile (sum of |IFFT| over chirps and pairs), used for validation."""
    iq = np.asarray(iq, dtype=np.complex64)
    t, rows, nfreq = iq.shape
    p0, p1 = pairs[:, 0], pairs[:, 1]
    pmin, cmin = int(p0.min()), int(p1.min())
    n_pair = int(p0.max()) - pmin + 1
    n_chirp = int(p1.max()) - cmin + 1
    if subtract_bg:
        iq = iq - iq.mean(axis=0, keepdims=True)
    cube = np.zeros((t, n_chirp, n_pair, nfreq), dtype=np.complex64)
    for r in range(rows):
        cube[:, p1[r] - cmin, p0[r] - pmin, :] = iq[:, r, :]
    cube *= np.hanning(nfreq).astype(np.float32)
    rc = np.fft.ifft(cube, n=n_range, axis=-1)
    return np.abs(rc).sum(axis=(0, 1, 2))


def range_peak_m(rda, n_range=256):
    """Return the strongest range bin in metres (excluding near/far edge bins)."""
    prof = np.abs(rda[:, 0]).sum(axis=(0, 2, 3))  # over T, chirp, pair
    prof[:4] = 0.0
    prof[-4:] = 0.0
    peak = int(np.argmax(prof))
    return peak * (151.0 / n_range) * RANGE_BIN_TRUE, peak
