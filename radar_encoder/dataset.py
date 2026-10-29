# -*- coding: utf-8 -*-
"""Dataset utilities for M4Human int16 I/Q LMDBs.

Keys are "[subject, action, frame]". A clip is T consecutive stored frames
(T=16 by default, stride=8), converted to a range-Doppler-angle tensor by dsp.py.
"""

import lmdb
import numpy as np

try:
    from .dsp import iq_to_rda
except ImportError:  # run as a plain script
    from dsp import iq_to_rda

HDR = None
import struct
HDR = struct.Struct("<IIIIf")


def parse_key(key):
    s, a, f = key.decode().strip("[]").split(",")
    return int(s), int(a), int(f)


def key_str(subject, action, frame):
    return f"[{subject}, {action}, {frame}]"


def decode_iq(value):
    magic, h, w, c, scale = HDR.unpack_from(value, 0)
    data = np.frombuffer(value, dtype="<i2", offset=HDR.size).reshape(c, h, w)
    i = data[0].astype(np.float32) * scale
    q = data[1].astype(np.float32) * scale
    return (i + 1j * q).astype(np.complex64)


def build_clips(frames, t_frames=16, stride=8):
    """frames: sorted list of frame ids -> list of clips (lists of ids)."""
    frames = sorted(frames)
    if len(frames) < t_frames:
        return []
    return [frames[i:i + t_frames] for i in range(0, len(frames) - t_frames + 1, stride)]


class IQClipDataset:
    def __init__(self, lmdb_path, subject, pairs, t_frames=16, stride=8, actions=None):
        self.lmdb_path = str(lmdb_path)
        self.subject = subject
        self.pairs = pairs
        self.t_frames = t_frames
        self.stride = stride
        self.env = lmdb.open(self.lmdb_path, subdir=False, readonly=True,
                             lock=False, readahead=False)
        self.frames_by_action = self._scan()
        self.clips = []
        for action, frames in sorted(self.frames_by_action.items()):
            if actions is not None and action not in actions:
                continue
            for clip in build_clips(frames, t_frames, stride):
                self.clips.append((action, clip))

    def _scan(self):
        prefix = f"[{self.subject}, ".encode()
        out = {}
        with self.env.begin() as txn:
            cur = txn.cursor()
            cur.set_range(prefix)
            for k, _ in cur:
                if not k.startswith(prefix):
                    break
                s, a, f = parse_key(k)
                out.setdefault(a, []).append(f)
        return {a: sorted(v) for a, v in out.items()}

    def __len__(self):
        return len(self.clips)

    def _read(self, action, frames):
        prefix = f"[{self.subject}, {action}, ".encode()
        keys = [key_str(self.subject, action, f).encode() for f in frames]
        with self.env.begin() as txn:
            vals = [txn.get(k) for k in keys]
        if any(v is None for v in vals):
            return None
        return np.stack([decode_iq(v) for v in vals])

    def __getitem__(self, i):
        action, frames = self.clips[i]
        iq = self._read(action, frames)
        if iq is None:
            return None
        return {
            "x": iq_to_rda(iq, self.pairs),
            "subject": self.subject,
            "action": action,
            "frames": frames,
            "center": frames[len(frames) // 2],
        }

    def load_clip_at(self, action, center):
        """Load the T stored frames nearest to a center frame, sorted by frame id."""
        frames = self.frames_by_action.get(action, [])
        if not frames:
            return None
        near = sorted(frames, key=lambda f: abs(f - center))[:self.t_frames]
        near = sorted(near)
        iq = self._read(action, near)
        if iq is None:
            return None
        return iq_to_rda(iq, self.pairs)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["env"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.env = lmdb.open(self.lmdb_path, subdir=False, readonly=True,
                             lock=False, readahead=False)

    def close(self):
        self.env.close()


def collate(batch):
    batch = [b for b in batch if b is not None]
    x = np.stack([b["x"] for b in batch])
    return {
        "x": x,
        "subject": [b["subject"] for b in batch],
        "action": [b["action"] for b in batch],
        "frames": [b["frames"] for b in batch],
        "center": [b["center"] for b in batch],
    }
