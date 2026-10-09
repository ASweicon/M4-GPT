# -*- coding: utf-8 -*-
"""M4Human metadata: paths, physical limits, action names, split configuration."""
import os

# ---- LMDB paths (相对本项目根目录, Mac/服务器通用) ----
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RPC = os.path.join(_ROOT, "radar_pc.lmdb")
IND = os.path.join(_ROOT, "indicator.lmdb")

# ---- physical limits (metres), radar frame: x horizontal, y forward(depth), z up ----
LIM = {
    "x": (-3.0, 2.5),
    "y": (0.0, 6.0),
    "z": (0.0, 3.0),
}

# ---- protocol groups ----
INPLACE_ACTIONS = list(range(1, 31))    # P1, 30 actions
SIT_ACTIONS = list(range(31, 36))       # P2, 5 actions
NIP_ACTIONS = list(range(36, 51))       # P3, 15 actions

# ---- split settings (official config.yaml) ----
S2_TEST = [19, 5, 8, 10]
S2_VAL = [17]
S3_TEST = [2, 11, 23, 28, 29, 33, 38, 43, 47, 50]
S3_VAL = [24, 10, 32]
S1_RATIO = (0.75, 0.05, 0.20)

# ---- action id -> name (TODO = not shown in paper table) ----
ACTION_NAMES = {
    1: "Chest expanding horizontally",
    2: "Chest expanding vertically",
    3: "Left side twist",
    4: "Right side twist",
    5: "Raising left arm",
    6: "Raising right arm",
    7: "Waving left arm",
    8: "Waving right arm",
    9: "Picking up things",
    10: "Throwing toward left side",
    11: "Throwing toward right side",
    12: "Kicking toward left direction using right leg",
    13: "Action 13 (TODO name)",
    14: "Bowing forward",
    15: "Stretching and relaxing in free form",
    16: "Mark time",
    17: "Left upper limb extension",
    18: "Right upper limb extension",
    19: "Left front lunge",
    20: "Right front lunge",
    21: "Both upper limbs extension",
    22: "Squat",
    23: "Left side lunge",
    24: "Action 24 (TODO name)",
    25: "Action 25 (TODO name)",
    26: "Right limbs extension",
    27: "Jumping up",
    28: "Tai Chi",
    29: "High knees",
    30: "Neck rotations while standing with both legs",
    31: "Slowly stand up and sit down from a chair",
    32: "Sit and left leg kick/extension",
    33: "Sit and right leg kick/extension",
    34: "Sit and raise left dumbbell (arm curls)",
    35: "Sit and raise right dumbbell (arm curls)",
    36: "Walk in straight line (fast)",
    37: "Walk in straight line (slow)",
    38: "Action 38 (TODO name)",
    39: "Walk in curves (slow)",
    40: "Non-in-place Lunge",
    41: "Front kick walk",
    42: "Sideways walking",
    43: "Badminton",
    44: "Table tennis (ping pong)",
    45: "Baseball",
    46: "Volleyball",
    47: "Free-place Boxing",
    48: "Walk in straight line and pick up item",
    49: "Action 49 (TODO name)",
    50: "Action 50 (TODO name)",
}


def coarse_group(action):
    if action in INPLACE_ACTIONS:
        return "in-place"
    if action in SIT_ACTIONS:
        return "sit-in-place"
    return "non-in-place"
