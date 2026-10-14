import io
import json
import math
import queue
import shutil
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    import cv2
except Exception:
    cv2 = None

try:
    import mediapipe as mp
except Exception:
    mp = None

try:
    import trimesh
except Exception:
    trimesh = None


DATA_DIR = Path("handsulpt_data")


# =============================================================================
# CONFIGURATION
# Every semantic threshold lives here. The comment on each line explains it.
# =============================================================================

DEFAULT_CONFIG = {
    "palm_width_min": 0.05,                 # minimum normalized palm width before a hand is rejected
    "pinch_enter": 0.35,                    # pinch ratio below this starts a pinch
    "pinch_exit": 0.50,                     # pinch ratio above this ends a pinch, giving hysteresis
    "open_palm_max_pinch": 0.70,            # pinch ratio above this is treated as an open palm
    "arming_hold_ms": 1000,                 # open palm must be held this long to arm risky actions
    "refractory_ms": 350,                   # ignore duplicate gesture events inside this window
    "rotation_gain": 2.2,                   # radians of model rotation per normalized palm-width movement
    "deadzone_palm_widths": 0.04,           # ignore small palm movements before rotation starts
    "max_rotation_rad_per_s": 6.0,          # maximum allowed angular speed for safety
    "one_euro_min_cutoff": 1.0,             # One Euro filter minimum cutoff frequency in Hz
    "one_euro_beta": 0.02,                  # One Euro filter speed coefficient
    "one_euro_d_cutoff": 1.0,               # One Euro filter derivative cutoff frequency in Hz
    "scale_min": 0.3,                       # minimum model scale
    "scale_max": 3.0,                       # maximum model scale
    "scale_deadzone_ratio": 0.02,           # ignore small relative two-hand distance changes
    "scale_smoothing_tau": 0.12,            # exponential smoothing time constant for scale in seconds
    "roll_gain": 1.5,                       # radians of roll per radian of hand-line twist
    "roll_deadzone_rad": 0.03,              # ignore small twist changes before roll starts
    "tracking_loss_lock_ms": 400,           # freeze controls after hands vanish for this long
    "task_success_angle_deg": 12.0,         # trial succeeds when final angle error is below this
    "task_timeout_s": 20.0,                 # maximum seconds allowed for one trial
    "trials_per_method": 20,                # number of trials per method in evaluation
    "users_simulated": 5,                   # number of synthetic users in evaluation
    "demo_speed": 1.0,                      # simulation playback speed multiplier
    "target_fps": 30,                       # desired processing/display frame rate
    "epsilon": 1e-6,                        # numerical floor for divisions and norm checks
    "two_hand_require_pinch": 1.0,          # require both hands pinched for two-hand scale/roll, 1=yes 0=no
    "demo_cycle_seconds": 26.0,             # length of the automatic demo gesture cycle
}

CONFIG_COMMENTS = {
    "palm_width_min": "minimum normalized palm width before a hand is rejected",
    "pinch_enter": "pinch ratio below this starts a pinch",
    "pinch_exit": "pinch ratio above this ends a pinch, giving hysteresis",
    "open_palm_max_pinch": "pinch ratio above this is treated as an open palm",
    "arming_hold_ms": "open palm must be held this long to arm risky actions",
    "refractory_ms": "ignore duplicate gesture events inside this window",
    "rotation_gain": "radians of model rotation per normalized palm-width movement",
    "deadzone_palm_widths": "ignore small palm movements before rotation starts",
    "max_rotation_rad_per_s": "maximum allowed angular speed for safety",
    "one_euro_min_cutoff": "One Euro filter minimum cutoff frequency in Hz",
    "one_euro_beta": "One Euro filter speed coefficient",
    "one_euro_d_cutoff": "One Euro filter derivative cutoff frequency in Hz",
    "scale_min": "minimum model scale",
    "scale_max": "maximum model scale",
    "scale_deadzone_ratio": "ignore small relative two-hand distance changes",
    "scale_smoothing_tau": "exponential smoothing time constant for scale in seconds",
    "roll_gain": "radians of roll per radian of hand-line twist",
    "roll_deadzone_rad": "ignore small twist changes before roll starts",
    "tracking_loss_lock_ms": "freeze controls after hands vanish for this long",
    "task_success_angle_deg": "trial succeeds when final angle error is below this",
    "task_timeout_s": "maximum seconds allowed for one trial",
    "trials_per_method": "number of trials per method in evaluation",
    "users_simulated": "number of synthetic users in evaluation",
    "demo_speed": "simulation playback speed multiplier",
    "target_fps": "desired processing/display frame rate",
    "epsilon": "numerical floor for divisions and norm checks",
    "two_hand_require_pinch": "require both hands pinched for two-hand scale/roll, 1=yes 0=no",
    "demo_cycle_seconds": "length of the automatic demo gesture cycle",
}


def ss_setdefault(key: str, value: Any) -> None:
    """Safely initialize Streamlit session state without relying on dict.setdefault behavior."""
    if key not in st.session_state:
        st.session_state[key] = value


def mapping_payload(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Return a JSON-friendly mapping file containing comments and values."""
    return {"_comments": CONFIG_COMMENTS, "values": cfg}


def sanitize_config(values: Dict[str, Any]) -> Dict[str, Any]:
    """Copy defaults, apply valid numeric overrides, and repair dangerous threshold combinations."""
    cfg = DEFAULT_CONFIG.copy()
    eps = float(DEFAULT_CONFIG["epsilon"])

    for k, v in values.items():
        if k in cfg and isinstance(v, (int, float)) and not isinstance(v, bool):
            cfg[k] = float(v)

    positive_keys = [
        "palm_width_min",
        "pinch_enter",
        "pinch_exit",
        "open_palm_max_pinch",
        "arming_hold_ms",
        "refractory_ms",
        "rotation_gain",
        "deadzone_palm_widths",
        "max_rotation_rad_per_s",
        "one_euro_min_cutoff",
        "one_euro_beta",
        "one_euro_d_cutoff",
        "scale_min",
        "scale_max",
        "scale_deadzone_ratio",
        "scale_smoothing_tau",
        "roll_gain",
        "roll_deadzone_rad",
        "tracking_loss_lock_ms",
        "task_success_angle_deg",
        "task_timeout_s",
        "trials_per_method",
        "users_simulated",
        "demo_speed",
        "target_fps",
        "epsilon",
        "demo_cycle_seconds",
    ]

    for k in positive_keys:
        if float(cfg[k]) <= 0.0:
            cfg[k] = float(DEFAULT_CONFIG[k])

    if float(cfg["pinch_enter"]) >= float(cfg["pinch_exit"]):
        cfg["pinch_enter"] = float(DEFAULT_CONFIG["pinch_enter"])
        cfg["pinch_exit"] = float(DEFAULT_CONFIG["pinch_exit"])

    if float(cfg["scale_min"]) >= float(cfg["scale_max"]):
        cfg["scale_min"] = float(DEFAULT_CONFIG["scale_min"])
        cfg["scale_max"] = float(DEFAULT_CONFIG["scale_max"])

    if float(cfg["open_palm_max_pinch"]) <= float(cfg["pinch_exit"]):
        cfg["open_palm_max_pinch"] = float(DEFAULT_CONFIG["open_palm_max_pinch"])

    cfg["two_hand_require_pinch"] = 1.0 if float(cfg.get("two_hand_require_pinch", 1.0)) > 0.5 else 0.0
    cfg["epsilon"] = max(eps, float(cfg["epsilon"]))
    cfg["target_fps"] = float(np.clip(float(cfg["target_fps"]), 5.0, 60.0))
    cfg["demo_speed"] = float(np.clip(float(cfg["demo_speed"]), 0.1, 4.0))
    cfg["trials_per_method"] = float(max(1, int(float(cfg["trials_per_method"]))))
    cfg["users_simulated"] = float(max(1, int(float(cfg["users_simulated"]))))

    return cfg


# =============================================================================
# QUATERNION AND GEOMETRY HELPERS
# =============================================================================

def q_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    n = float(np.linalg.norm(q))
    if n < DEFAULT_CONFIG["epsilon"]:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / n


def q_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton quaternion multiplication, order [w, x, y, z]."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def q_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    n = float(np.linalg.norm(axis))
    eps = DEFAULT_CONFIG["epsilon"]
    if n < eps or not np.isfinite(angle):
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    axis = axis / n
    half = 0.5 * float(angle)
    s = math.sin(half)
    return q_normalize(np.array([math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s], dtype=float))


def q_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q_normalize(np.asarray(q, dtype=float))
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def q_from_euler_deg(rx: float, ry: float, rz: float) -> np.ndarray:
    qx = q_from_axis_angle(np.array([1.0, 0.0, 0.0]), math.radians(rx))
    qy = q_from_axis_angle(np.array([0.0, 1.0, 0.0]), math.radians(ry))
    qz = q_from_axis_angle(np.array([0.0, 0.0, 1.0]), math.radians(rz))
    return q_normalize(q_mul(qz, q_mul(qy, qx)))


def angle_between_quats(q1: np.ndarray, q2: np.ndarray) -> float:
    q1 = q_normalize(np.asarray(q1, dtype=float))
    q2 = q_normalize(np.asarray(q2, dtype=float))
    dot = float(abs(np.dot(q1, q2)))
    dot = min(1.0, max(0.0, dot))
    return float(math.degrees(2.0 * math.acos(dot)))


def random_quaternion(rng: Optional[np.random.Generator] = None) -> np.ndarray:
    if rng is None:
        rng = np.random.default_rng()
    u1, u2, u3 = rng.random(3)
    w = math.sqrt(max(0.0, 1.0 - u1)) * math.sin(2.0 * math.pi * u2)
    x = math.sqrt(max(0.0, 1.0 - u1)) * math.cos(2.0 * math.pi * u2)
    y = math.sqrt(max(0.0, u1)) * math.sin(2.0 * math.pi * u3)
    z = math.sqrt(max(0.0, u1)) * math.cos(2.0 * math.pi * u3)
    return q_normalize(np.array([w, x, y, z], dtype=float))


def wrap_angle(a: float) -> float:
    return float((float(a) + math.pi) % (2.0 * math.pi) - math.pi)


def apply_deadzone_vec(vec: np.ndarray, deadzone: float) -> np.ndarray:
    vec = np.asarray(vec, dtype=float)
    if vec.size == 0 or not np.all(np.isfinite(vec)):
        return np.zeros_like(vec, dtype=float)
    n = float(np.linalg.norm(vec))
    if n <= float(deadzone) or n < DEFAULT_CONFIG["epsilon"]:
        return np.zeros_like(vec, dtype=float)
    return vec * ((n - float(deadzone)) / n)


# =============================================================================
# ONE EURO FILTER
# =============================================================================

class OneEuro:
    """Scalar One Euro filter for noisy positional signals."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.0, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev: Optional[float] = None
        self.dx_prev = 0.0
        self.t_prev: Optional[float] = None
        self.x_filt_prev: Optional[float] = None

    def alpha(self, cutoff: float, dt: float) -> float:
        if dt <= 0.0:
            return 1.0
        cutoff = max(DEFAULT_CONFIG["epsilon"], float(cutoff))
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: float, t: float) -> float:
        x = float(x)
        t = float(t)
        if not np.isfinite(x):
            return float(self.x_filt_prev if self.x_filt_prev is not None else 0.0)

        if self.t_prev is None or self.x_prev is None or self.x_filt_prev is None:
            self.t_prev = t
            self.x_prev = x
            self.x_filt_prev = x
            self.dx_prev = 0.0
            return x

        dt = t - self.t_prev
        if dt <= DEFAULT_CONFIG["epsilon"]:
            return float(self.x_filt_prev)

        dx = (x - self.x_prev) / dt
        a_d = self.alpha(self.d_cutoff, dt)
        dx_filt = self.dx_prev + a_d * (dx - self.dx_prev)

        cutoff = self.min_cutoff + self.beta * abs(dx_filt)
        a = self.alpha(cutoff, dt)
        x_filt = self.x_filt_prev + a * (x - self.x_filt_prev)

        self.t_prev = t
        self.x_prev = x
        self.dx_prev = dx_filt
        self.x_filt_prev = x_filt
        return float(x_filt)


class OneEuroVector:
    """Vector wrapper around scalar OneEuro filters for landmark arrays."""

    def __init__(self, shape: Tuple[int, ...], min_cutoff: float, beta: float, d_cutoff: float):
        self.shape = tuple(int(s) for s in shape)
        count = int(np.prod(self.shape))
        self.filters = [OneEuro(min_cutoff=min_cutoff, beta=beta, d_cutoff=d_cutoff) for _ in range(count)]

    def filter(self, arr: np.ndarray, t: float) -> np.ndarray:
        a = np.asarray(arr, dtype=float).reshape(-1)
        out = np.empty_like(a, dtype=float)
        for i, v in enumerate(a):
            out[i] = self.filters[i].filter(float(v), float(t))
        return out.reshape(self.shape)


# =============================================================================
# GESTURES, HYSTERESIS, HOLD TIMERS, EVENT BUS
# =============================================================================

@dataclass
class GestureEvent:
    type: str
    hand: str
    t_ms: int
    confidence: float
    params: Dict[str, Any] = field(default_factory=dict)


class EventBus:
    """Small in-memory event bus with refractory suppression and recent-event history."""

    def __init__(self, refractory_ms: float, maxlen: int = 1000):
        self.refractory_ms = float(refractory_ms)
        self.last: Dict[Tuple[str, str], float] = {}
        self.events: deque = deque(maxlen=int(maxlen))

    def emit(
        self,
        type_: str,
        hand: str,
        t_ms: int,
        confidence: float = 1.0,
        params: Optional[Dict[str, Any]] = None,
    ) -> Optional[GestureEvent]:
        key = (type_, hand)
        last = self.last.get(key, -1e18)
        if float(t_ms) - last < self.refractory_ms:
            return None
        self.last[key] = float(t_ms)
        ev = GestureEvent(type=type_, hand=hand, t_ms=int(t_ms), confidence=float(confidence), params=params or {})
        self.events.appendleft(ev)
        return ev


class PinchDetector:
    """Hysteretic pinch detector."""

    def __init__(self, enter: float, exit_: float):
        self.enter = float(enter)
        self.exit = float(exit_)
        self.active = False

    def update(self, ratio: float) -> bool:
        if not np.isfinite(ratio):
            self.active = False
            return False
        if not self.active and ratio < self.enter:
            self.active = True
        elif self.active and ratio > self.exit:
            self.active = False
        return self.active


class HoldTimer:
    """Trigger once after a condition has been continuously true for hold_ms."""

    def __init__(self, hold_ms: float):
        self.hold_ms = float(hold_ms)
        self.start_ms: Optional[float] = None
        self.triggered = False

    def update(self, condition: bool, t_ms: int) -> bool:
        if not condition:
            self.start_ms = None
            self.triggered = False
            return False
        if self.start_ms is None:
            self.start_ms = float(t_ms)
        if (not self.triggered) and (float(t_ms) - self.start_ms >= self.hold_ms):
            self.triggered = True
            return True
        return False


# =============================================================================
# HAND METRICS AND PURE MAPPING FUNCTIONS
# =============================================================================

def compute_palm_metrics(landmarks: np.ndarray, eps: float) -> Optional[Dict[str, Any]]:
    """Compute palm width, palm center, palm normal, and pinch ratio from 21 landmarks."""
    lms = np.asarray(landmarks, dtype=float)
    if lms.shape != (21, 3) or not np.all(np.isfinite(lms)):
        return None

    palm_width = float(np.linalg.norm(lms[5] - lms[17]))
    if palm_width < eps:
        return None

    palm_center = np.mean(lms[[0, 5, 9, 13, 17]], axis=0)
    v1 = lms[5] - lms[0]
    v2 = lms[17] - lms[0]
    normal = np.cross(v1, v2)
    nn = float(np.linalg.norm(normal))
    if nn < eps:
        normal = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        normal = normal / nn

    pinch_ratio = float(np.linalg.norm(lms[4] - lms[8]) / max(palm_width, eps))

    return {
        "palm_width": palm_width,
        "palm_center": palm_center.astype(float),
        "palm_normal": normal.astype(float),
        "pinch_ratio": pinch_ratio,
    }


def rotation_delta_quat(delta_xy: np.ndarray, cfg: Dict[str, Any], dt: float) -> np.ndarray:
    """Convert deadzoned palm displacement into a clamped incremental rotation quaternion."""
    eps = float(cfg["epsilon"])
    d = apply_deadzone_vec(np.asarray(delta_xy, dtype=float), float(cfg["deadzone_palm_widths"]))
    angle_y = float(d[0] * cfg["rotation_gain"])
    angle_x = float(-d[1] * cfg["rotation_gain"])
    mag = math.hypot(angle_x, angle_y)
    max_angle = float(cfg["max_rotation_rad_per_s"] * max(dt, eps))
    if mag > max_angle and mag > eps:
        factor = max_angle / mag
        angle_x *= factor
        angle_y *= factor

    qy = q_from_axis_angle(np.array([0.0, 1.0, 0.0]), angle_y)
    qx = q_from_axis_angle(np.array([1.0, 0.0, 0.0]), angle_x)
    return q_normalize(q_mul(qy, qx))


def scale_update(d_norm: float, d0: Optional[float], cfg: Dict[str, Any], current_scale: float, dt: float) -> float:
    """Two-hand scale mapping with clamp, deadzone, and exponential smoothing."""
    eps = float(cfg["epsilon"])
    if d0 is None or float(d0) <= eps or not np.isfinite(d_norm):
        return float(current_scale)

    target = float(np.clip(float(d_norm) / float(d0), float(cfg["scale_min"]), float(cfg["scale_max"])))
    if abs(target - 1.0) <= float(cfg["scale_deadzone_ratio"]):
        target = float(current_scale)

    tau = max(eps, float(cfg.get("scale_smoothing_tau", 0.12)))
    alpha = 1.0 - math.exp(-max(dt, eps) / tau)
    return float(current_scale + alpha * (target - current_scale))


def roll_delta_quat(angle: float, prev_angle: Optional[float], cfg: Dict[str, Any], dt: float) -> np.ndarray:
    """Two-hand roll mapping from change in hand-line angle."""
    ident = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    if prev_angle is None or not np.isfinite(angle) or not np.isfinite(prev_angle):
        return ident

    da = wrap_angle(float(angle) - float(prev_angle))
    dz = float(cfg["roll_deadzone_rad"])
    if abs(da) <= dz:
        return ident

    eff = math.copysign(abs(da) - dz, da) * float(cfg["roll_gain"])
    max_angle = float(cfg["max_rotation_rad_per_s"] * max(dt, float(cfg["epsilon"])))
    if abs(eff) > max_angle:
        eff = math.copysign(max_angle, eff)

    return q_from_axis_angle(np.array([0.0, 0.0, 1.0]), eff)


def tracking_lock_update(
    valid_hands: bool,
    locked: bool,
    lost_since: Optional[float],
    t_ms: int,
    lock_ms: float,
) -> Tuple[bool, Optional[float], Optional[str]]:
    """Return updated lock state, lost timestamp, and optional lock/unlock event name."""
    event = None
    if not valid_hands:
        if lost_since is None:
            lost_since = float(t_ms)
        if (not locked) and (float(t_ms) - float(lost_since) >= float(lock_ms)):
            locked = True
            event = "tracking_lost"
    else:
        if locked:
            locked = False
            event = "tracking_resumed"
        lost_since = None
    return bool(locked), lost_since, event


# =============================================================================
# SYNTHETIC DEMO HAND GENERATOR
# =============================================================================

DEMO_RNG = np.random.default_rng(11)

TEMPLATE = np.array(
    [
        [0.00, -0.10, 0.0],  # 0 wrist
        [-0.05, -0.06, 0.0], # 1 thumb CMC
        [-0.08, -0.02, 0.0], # 2 thumb MCP
        [-0.09, 0.02, 0.0],  # 3 thumb IP
        [-0.07, 0.06, 0.0],  # 4 thumb tip
        [-0.04, 0.02, 0.0],  # 5 index MCP
        [-0.04, 0.07, 0.0],  # 6 index PIP
        [-0.04, 0.10, 0.0],  # 7 index DIP
        [-0.04, 0.13, 0.0],  # 8 index tip
        [0.00, 0.03, 0.0],   # 9 middle MCP
        [0.00, 0.08, 0.0],   # 10 middle PIP
        [0.00, 0.11, 0.0],   # 11 middle DIP
        [0.00, 0.14, 0.0],   # 12 middle tip
        [0.04, 0.02, 0.0],   # 13 ring MCP
        [0.04, 0.07, 0.0],   # 14 ring PIP
        [0.04, 0.10, 0.0],   # 15 ring DIP
        [0.04, 0.13, 0.0],   # 16 ring tip
        [0.07, 0.00, 0.0],   # 17 pinky MCP
        [0.08, 0.04, 0.0],   # 18 pinky PIP
        [0.08, 0.06, 0.0],   # 19 pinky DIP
        [0.08, 0.08, 0.0],   # 20 pinky tip
    ],
    dtype=np.float32,
)


def demo_phase_label(t: float) -> str:
    """Human-readable label for the current automatic demo phase."""
    cycle = float(DEFAULT_CONFIG.get("demo_cycle_seconds", 26.0))
    phase = float(t) % cycle
    if 20.0 <= phase < 22.0:
        return "Tracking-loss lock demonstration"
    if phase < 2.0:
        return "Arming: open palms held"
    if phase < 10.0:
        return "One-hand pinch rotation"
    if phase < 20.0:
        return "Two-hand scale + roll"
    return "Resume: open palms"


def make_demo_landmarks(center: Tuple[float, float], pinch: float, mirror: bool, t: float, hand_id: str) -> np.ndarray:
    """Create a plausible 21-point hand landmark array for the local demo."""
    pts = TEMPLATE.copy()
    if mirror:
        pts[:, 0] *= -1.0

    pinch = float(np.clip(pinch, 0.0, 1.0))
    thumb0 = pts[4].copy()
    index0 = pts[8].copy()

    pts[4] = (1.0 - pinch) * thumb0 + pinch * index0
    pts[8] = (1.0 - 0.25 * pinch) * index0 + 0.25 * pinch * thumb0

    spread = (1.0 - pinch) * 0.01
    pts[[8, 12, 16, 20], 1] += spread

    seed = 0.0 if hand_id == "Right" else 1.7
    breath = 1.0 + 0.015 * math.sin(t * 1.7 + seed)
    pts = pts * breath

    pts[:, 0] += float(center[0])
    pts[:, 1] += float(center[1])
    pts[:, 2] = 0.004 * math.sin(t * 2.1 + seed)

    pts += DEMO_RNG.normal(0.0, 0.0012, pts.shape).astype(np.float32)

    pts[:, 0] = np.clip(pts[:, 0], 0.03, 0.97)
    pts[:, 1] = np.clip(pts[:, 1], 0.03, 0.97)
    return pts.astype(np.float32)


def simulate_demo_hands(t: float) -> Tuple[Dict[str, np.ndarray], str]:
    """Generate synthetic hands for an immediately runnable demo."""
    cycle = float(DEFAULT_CONFIG.get("demo_cycle_seconds", 26.0))
    phase = float(t) % cycle

    # Deliberate tracking-loss window to demonstrate locking.
    if 20.0 <= phase < 22.0:
        return {}, demo_phase_label(t)

    if phase < 2.0:
        # Arming: both hands open.
        r_center = (0.38 + 0.01 * math.sin(t), 0.55 + 0.01 * math.cos(t))
        l_center = (0.62 + 0.01 * math.cos(t), 0.55 + 0.01 * math.sin(t))
        r_pinch = 0.05
        l_pinch = 0.05
    elif phase < 10.0:
        # One-hand rotation: right hand pinches and moves in a circle.
        ang = 0.8 * (phase - 2.0)
        r_center = (0.45 + 0.10 * math.sin(ang), 0.55 + 0.08 * math.cos(ang))
        l_center = (0.65, 0.55)
        r_pinch = 0.95
        l_pinch = 0.10
    elif phase < 20.0:
        # Two-hand scale and roll: hands move apart/together and twist.
        u = phase - 10.0
        sep = 0.22 + 0.10 * math.sin(0.55 * u)
        twist = 0.45 * math.sin(0.35 * u)
        mid_x = 0.50
        mid_y = 0.55 + 0.03 * math.sin(0.25 * u)
        r_center = (mid_x - 0.5 * sep, mid_y - 0.5 * sep * twist)
        l_center = (mid_x + 0.5 * sep, mid_y + 0.5 * sep * twist)
        r_pinch = 0.85
        l_pinch = 0.85
    else:
        # Resume open palms after tracking loss.
        r_center = (0.40 + 0.01 * math.sin(t), 0.55)
        l_center = (0.60 + 0.01 * math.cos(t), 0.55)
        r_pinch = 0.05
        l_pinch = 0.05

    hands: Dict[str, np.ndarray] = {}
    hands["Right"] = make_demo_landmarks(r_center, r_pinch, mirror=False, t=t, hand_id="Right")
    hands["Left"] = make_demo_landmarks(l_center, l_pinch, mirror=True, t=t, hand_id="Left")
    return hands, demo_phase_label(t)


# =============================================================================
# FRAME SOURCE THREAD
# Capture/tracking runs outside the Streamlit rendering loop and keeps only newest frame.
# =============================================================================

class FrameSource:
    """Threaded frame producer for demo or optional local webcam input."""

    def __init__(self, mode: str, camera_index: int = 0, fps: int = 30, demo_speed: float = 1.0):
        self.mode = mode
        self.actual_mode = mode
        self.camera_index = int(camera_index)
        self.fps = max(5, min(60, int(fps)))
        self.demo_speed = float(demo_speed)
        self.q: queue.Queue = queue.Queue(maxsize=1)
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.t0 = time.time()
        self.cap = None
        self.mp_hands = None

    def start(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        self._release()

    def _release(self) -> None:
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        try:
            if self.mp_hands is not None:
                self.mp_hands.close()
        except Exception:
            pass
        self.cap = None
        self.mp_hands = None

    def _put(self, frame: Dict[str, Any]) -> None:
        try:
            self.q.get_nowait()
        except queue.Empty:
            pass
        try:
            self.q.put_nowait(frame)
        except Exception:
            pass

    def get_newest(self) -> Optional[Dict[str, Any]]:
        try:
            return self.q.get_nowait()
        except queue.Empty:
            return None

    def _init_camera(self) -> None:
        if cv2 is None or mp is None:
            self.actual_mode = "demo"
            return
        try:
            self.cap = cv2.VideoCapture(self.camera_index)
            if not self.cap.isOpened():
                self.cap.release()
                self.cap = None
                self.actual_mode = "demo"
                return
            self.mp_hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.actual_mode = "camera"
        except Exception:
            self._release()
            self.actual_mode = "demo"

    def _run(self) -> None:
        if self.mode == "camera":
            self._init_camera()

        while not self.stop_event.is_set():
            loop_t0 = time.perf_counter()

            if self.actual_mode == "camera":
                frame = self._camera_frame()
            else:
                frame = self._demo_frame()

            self._put(frame)

            period = 1.0 / float(self.fps)
            elapsed = time.perf_counter() - loop_t0
            sleep_s = max(0.0, period - elapsed)
            time.sleep(sleep_s)

        self._release()

    def _demo_frame(self) -> Dict[str, Any]:
        wall_t = time.time() - self.t0
        sim_t = wall_t * self.demo_speed
        hands, phase = simulate_demo_hands(sim_t)
        tracking_ms = float(np.clip(DEMO_RNG.normal(4.2, 1.1), 1.0, 20.0))
        return {
            "t_ms": int(wall_t * 1000.0),
            "hands": hands,
            "source": "demo",
            "capture_ms": 0.0,
            "tracking_ms": tracking_ms,
            "phase": phase,
        }

    def _camera_frame(self) -> Dict[str, Any]:
        wall_t = time.time() - self.t0
        if self.cap is None or self.mp_hands is None:
            return self._demo_frame()

        cap_t0 = time.perf_counter()
        ok, bgr = self.cap.read()
        capture_ms = (time.perf_counter() - cap_t0) * 1000.0

        if not ok:
            self.actual_mode = "demo"
            return self._demo_frame()

        try:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        except Exception:
            self.actual_mode = "demo"
            return self._demo_frame()

        tr_t0 = time.perf_counter()
        try:
            results = self.mp_hands.process(rgb)
        except Exception:
            self.actual_mode = "demo"
            return self._demo_frame()
        tracking_ms = (time.perf_counter() - tr_t0) * 1000.0

        hands: Dict[str, np.ndarray] = {}
        if results.multi_hand_landmarks:
            for i, hand_lms in enumerate(results.multi_hand_landmarks):
                arr = np.array([[lm.x, lm.y, lm.z] for lm in hand_lms.landmark], dtype=np.float32)
                label = "Right"
                try:
                    if results.multi_handedness and i < len(results.multi_handedness):
                        label = results.multi_handedness[i].classification[0].label
                except Exception:
                    label = "Right" if i == 0 else "Left"
                if label in hands:
                    label = f"{label}_{i}"
                hands[label] = arr

        return {
            "t_ms": int(wall_t * 1000.0),
            "hands": hands,
            "source": "camera",
            "capture_ms": float(capture_ms),
            "tracking_ms": float(tracking_ms),
            "phase": "Local webcam mode",
        }


# =============================================================================
# 3D MODEL LOADING AND RENDERING
# =============================================================================

def normalize_vertices(vertices: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Center and scale vertices while preserving face-index validity."""
    v = np.asarray(vertices, dtype=float)
    if v.ndim != 2 or v.shape[1] != 3:
        return np.zeros((0, 3), dtype=float)

    v = np.nan_to_num(v, nan=0.0, posinf=0.0, neginf=0.0)
    if len(v) == 0:
        return np.array([[0.0, 0.0, 0.0]], dtype=float)

    center = np.mean(v, axis=0)
    v = v - center
    max_r = float(np.max(np.linalg.norm(v, axis=1))) if len(v) else 1.0
    if max_r > eps:
        v = v / max_r
    return v


def default_model() -> Dict[str, Any]:
    """Create an immediate local demo model so the app never starts blank."""
    if trimesh is not None:
        try:
            mesh = trimesh.creation.icosphere(subdivisions=2, radius=1.0)
            verts = normalize_vertices(np.asarray(mesh.vertices, dtype=float))
            faces = np.asarray(mesh.faces, dtype=int)
            return {"name": "Built-in demo icosphere", "vertices": verts, "faces": faces}
        except Exception:
            pass

    verts = np.array(
        [
            [-1.0, -1.0, -1.0],
            [1.0, -1.0, -1.0],
            [1.0, 1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [1.0, 1.0, 1.0],
            [-1.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    faces = np.array(
        [
            [0, 1, 2], [0, 2, 3],
            [4, 6, 5], [4, 7, 6],
            [0, 4, 5], [0, 5, 1],
            [1, 5, 6], [1, 6, 2],
            [2, 6, 7], [2, 7, 3],
            [3, 7, 4], [3, 4, 0],
        ],
        dtype=int,
    )
    return {"name": "Built-in demo cube", "vertices": normalize_vertices(verts), "faces": faces}


def load_model_from_file(file_obj: Any) -> Dict[str, Any]:
    """Load a local glTF/GLB file into normalized vertices and faces."""
    if trimesh is None:
        raise RuntimeError("trimesh is not installed. Run: pip install -r requirements.txt")

    name = getattr(file_obj, "name", "model.glb")
    ext = Path(name).suffix.lower().lstrip(".")
    data = file_obj.getvalue()

    mesh = trimesh.load(io.BytesIO(data), file_type=ext if ext else None, force="mesh")

    if isinstance(mesh, trimesh.Scene):
        geoms = list(mesh.geometry.values())
        if not geoms:
            raise RuntimeError("No geometry found in glTF file.")
        mesh = trimesh.util.concatenate(geoms)

    verts = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)

    if len(faces) > 20000:
        try:
            hull = mesh.convex_hull
            verts = np.asarray(hull.vertices, dtype=float)
            faces = np.asarray(hull.faces, dtype=int)
        except Exception:
            pass

    verts = normalize_vertices(verts)
    return {"name": name, "vertices": verts, "faces": faces}


def transform_vertices(model: Dict[str, Any], quat: np.ndarray, scale: float) -> np.ndarray:
    """Apply quaternion rotation and uniform scale to model vertices."""
    verts = np.asarray(model.get("vertices", np.zeros((0, 3))), dtype=float)
    if verts.size == 0:
        return verts
    if not np.isfinite(scale):
        scale = 1.0
    r = q_to_matrix(quat)
    return (float(scale) * (r @ verts.T)).T


def make_figure(
    model: Dict[str, Any],
    quat: np.ndarray,
    scale: float,
    target_quat: Optional[np.ndarray] = None,
) -> go.Figure:
    """Build a Plotly 3D figure, optionally showing a transparent target ghost."""
    verts = transform_vertices(model, quat, scale)
    faces = np.asarray(model.get("faces", np.zeros((0, 3), dtype=int)), dtype=int)

    fig = go.Figure()

    if len(verts) > 0 and len(faces) > 0:
        fig.add_trace(
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color="#7ec8ff",
                opacity=0.96,
                flatshading=True,
                lighting=dict(ambient=0.35, diffuse=0.65, specular=0.18, roughness=0.55),
                lightposition=dict(x=100, y=100, z=120),
                name=model.get("name", "model"),
            )
        )
    elif len(verts) > 0:
        fig.add_trace(
            go.Scatter3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                mode="markers",
                marker=dict(size=2, color="#7ec8ff"),
                name=model.get("name", "points"),
            )
        )

    if target_quat is not None and len(verts) > 0 and len(faces) > 0:
        tverts = transform_vertices(model, target_quat, 1.0)
        fig.add_trace(
            go.Mesh3d(
                x=tverts[:, 0],
                y=tverts[:, 1],
                z=tverts[:, 2],
                i=faces[:, 0],
                j=faces[:, 1],
                k=faces[:, 2],
                color="#ffb703",
                opacity=0.16,
                flatshading=True,
                name="target orientation",
            )
        )

    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[-2, 2], showbackground=False, showgrid=False, zeroline=False, ticks=""),
            yaxis=dict(range=[-2, 2], showbackground=False, showgrid=False, zeroline=False, ticks=""),
            zaxis=dict(range=[-2, 2], showbackground=False, showgrid=False, zeroline=False, ticks=""),
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.2)),
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        height=520,
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font_color="white",
        showlegend=False,
    )
    return fig


# =============================================================================
# TRIALS, BENCHMARKS, TESTS
# =============================================================================

def generate_demo_trials(cfg: Dict[str, Any], seed: int = 3) -> list:
    """Create a few visible demo trials so Task lab is not empty on first run."""
    rng = np.random.default_rng(seed)
    rows = []
    for i, method in enumerate(["Gesture", "Mouse", "Gesture", "Mouse"], start=1):
        base_t = 6.4 if method == "Gesture" else 5.0
        base_e = 9.2 if method == "Gesture" else 7.1
        t = max(1.0, float(rng.normal(base_t, 1.1)))
        e = max(0.3, float(rng.normal(base_e, 2.0)))
        rows.append(
            {
                "trial_id": f"demo-{i}",
                "method": method,
                "user": "demo",
                "time_s": round(t, 3),
                "error_deg": round(e, 3),
                "success": bool(e <= float(cfg["task_success_angle_deg"])),
                "timed_out": False,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
        )
    return rows


def generate_benchmark(cfg: Dict[str, Any], seed: int = 17) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate synthetic 5-user, 20-trial-per-method evaluation data."""
    rng = np.random.default_rng(seed)
    rows = []
    trials = max(1, int(cfg["trials_per_method"]))
    users = max(1, int(cfg["users_simulated"]))

    for method in ["Gesture", "Mouse"]:
        for user in range(1, users + 1):
            skill = float(rng.normal(0.0, 0.14))
            for trial in range(1, trials + 1):
                learning = max(0.55, 1.0 - 0.018 * (trial - 1))
                if method == "Gesture":
                    base_t, base_e = 7.0, 10.0
                else:
                    base_t, base_e = 5.2, 7.2

                time_s = max(1.0, float(base_t * (1.0 + skill) * learning + rng.normal(0.0, 0.9)))
                error_deg = max(0.2, float(base_e * (1.0 + 0.45 * skill) * learning + rng.normal(0.0, 1.7)))
                success = bool(error_deg <= float(cfg["task_success_angle_deg"]))

                rows.append(
                    {
                        "method": method,
                        "user": f"user_{user}",
                        "trial": trial,
                        "time_s": round(time_s, 3),
                        "error_deg": round(error_deg, 3),
                        "success": success,
                    }
                )

    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("method")
        .agg(
            mean_time_s=("time_s", "mean"),
            std_time_s=("time_s", "std"),
            mean_error_deg=("error_deg", "mean"),
            std_error_deg=("error_deg", "std"),
            success_rate=("success", "mean"),
            trials=("trial", "count"),
        )
        .reset_index()
    )
    for col in ["mean_time_s", "std_time_s", "mean_error_deg", "std_error_deg", "success_rate"]:
        summary[col] = summary[col].round(3)
    return detail, summary


def run_self_tests() -> pd.DataFrame:
    """Pure-function tests for math, gestures, filtering, locking, mapping, and config safety."""
    cfg = sanitize_config(DEFAULT_CONFIG.copy())
    eps = float(cfg["epsilon"])
    rows = []

    def check(name: str, cond: bool, detail: str = "") -> None:
        rows.append({"Test": name, "Result": "PASS" if bool(cond) else "FAIL", "Detail": str(detail)})

    ident = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    qz = q_from_axis_angle(np.array([0.0, 0.0, 1.0]), math.pi / 2.0)
    qx = q_from_axis_angle(np.array([1.0, 0.0, 0.0]), math.pi / 2.0)

    check("Quaternion identity multiply", np.allclose(q_mul(ident, qz), qz, atol=1e-6))
    check("Quaternion normalize", np.isclose(np.linalg.norm(q_normalize(np.array([2.0, 0.0, 0.0, 0.0]))), 1.0))
    check("Quaternion angle same", np.isclose(angle_between_quats(ident, ident), 0.0, atol=1e-5))
    check("Quaternion 90 degrees", np.isclose(angle_between_quats(ident, qz), 90.0, atol=1e-4))
    check("Quaternion non-commutative", not np.allclose(q_mul(qz, qx), q_mul(qx, qz), atol=1e-3))

    m = compute_palm_metrics(TEMPLATE, eps)
    check("Palm metrics exist", m is not None)
    if m is not None:
        check("Palm width positive", m["palm_width"] > eps)
        check("Palm center shape", m["palm_center"].shape == (3,))
        check("Palm normal unit", np.isclose(np.linalg.norm(m["palm_normal"]), 1.0, atol=1e-5))
        check("Pinch ratio finite", np.isfinite(m["pinch_ratio"]))

    pdet = PinchDetector(0.35, 0.50)
    check("Pinch enter", pdet.update(0.34) is True)
    check("Pinch hysteresis hold", pdet.update(0.45) is True)
    check("Pinch exit", pdet.update(0.51) is False)

    ht = HoldTimer(100)
    check("Arm not immediate", ht.update(True, 0) is False)
    check("Arm pending", ht.update(True, 50) is False)
    check("Arm after hold", ht.update(True, 100) is True)
    check("Arm reset", ht.update(False, 150) is False and ht.update(True, 160) is False)

    bus = EventBus(100)
    e1 = bus.emit("pinch_start", "Right", 0)
    e2 = bus.emit("pinch_start", "Right", 50)
    e3 = bus.emit("pinch_start", "Right", 150)
    check("Event first emits", e1 is not None)
    check("Refractory blocks duplicate", e2 is None)
    check("Refractory expires", e3 is not None)

    for typ in ["pinch_start", "pinch_end", "arm_ready", "tracking_lost", "tracking_resumed", "two_hand_start", "two_hand_end"]:
        b2 = EventBus(0)
        check(f"Gesture type emits: {typ}", b2.emit(typ, "Right", 0) is not None)

    d = apply_deadzone_vec(np.array([0.10, 0.0]), 0.04)
    check("Deadzone subtracts magnitude", np.allclose(d, [0.06, 0.0], atol=1e-8))
    check("Deadzone zero inside", np.allclose(apply_deadzone_vec(np.array([0.02, 0.0]), 0.04), [0.0, 0.0]))

    qr = rotation_delta_quat(np.array([10.0, 0.0]), cfg, 0.03)
    max_deg = math.degrees(float(cfg["max_rotation_rad_per_s"] * 0.03))
    check("Rotation speed clamp", angle_between_quats(ident, qr) <= max_deg + 1e-3)

    s_max = scale_update(10.0, 1.0, cfg, 1.0, 1.0)
    s_min = scale_update(0.01, 1.0, cfg, 1.0, 1.0)
    s_dz = scale_update(1.005, 1.0, cfg, 1.0, 1.0)
    check("Scale clamp max", s_max <= float(cfg["scale_max"]) + 1e-6)
    check("Scale clamp min", s_min >= float(cfg["scale_min"]) - 1e-6)
    check("Scale deadzone", np.isclose(s_dz, 1.0, atol=1e-6))

    qroll = roll_delta_quat(1.0, 0.0, cfg, 0.03)
    check("Roll speed clamp", angle_between_quats(ident, qroll) <= max_deg + 1e-3)
    check("Roll deadzone", np.allclose(roll_delta_quat(0.02, 0.0, cfg, 0.03), ident, atol=1e-6))

    locked, lost, ev = tracking_lock_update(False, False, None, 0, 400)
    check("Tracking loss not immediate", locked is False and ev is None)
    locked, lost, ev = tracking_lock_update(False, False, 0, 500, 400)
    check("Tracking loss locks", locked is True and ev == "tracking_lost")
    locked, lost, ev = tracking_lock_update(True, True, 500, 600, 400)
    check("Tracking resume unlocks", locked is False and ev == "tracking_resumed")

    t = np.linspace(0.0, 1.0, 100)
    clean = np.sin(2.0 * np.pi * t)
    noisy = clean + np.random.default_rng(1).normal(0.0, 0.2, size=clean.shape)
    filt = OneEuro(min_cutoff=1.0, beta=0.02, d_cutoff=1.0)
    y = np.array([filt.filter(float(noisy[i]), float(t[i])) for i in range(len(t))])
    raw_rmse = float(np.sqrt(np.mean((noisy - clean) ** 2)))
    filt_rmse = float(np.sqrt(np.mean((y - clean) ** 2)))
    check("One Euro reduces noise", filt_rmse < raw_rmse, f"raw={raw_rmse:.3f}, filtered={filt_rmse:.3f}")

    check("Demo phase label exists", isinstance(demo_phase_label(0.5), str) and len(demo_phase_label(0.5)) > 0)

    bad_cfg = sanitize_config({"rotation_gain": 999, "scale_min": 5, "scale_max": 2, "pinch_enter": 0.8, "pinch_exit": 0.2})
    check("Sanitize scale range", float(bad_cfg["scale_min"]) < float(bad_cfg["scale_max"]))
    check("Sanitize pinch hysteresis", float(bad_cfg["pinch_enter"]) < float(bad_cfg["pinch_exit"]))

    return pd.DataFrame(rows)


# =============================================================================
# STREAMLIT SESSION STATE AND PROCESSING
# =============================================================================

def init_session() -> None:
    """Initialize all persistent app state once per Streamlit session."""
    ss_setdefault("config", sanitize_config(DEFAULT_CONFIG.copy()))
    ss_setdefault("quat", np.array([1.0, 0.0, 0.0, 0.0], dtype=float))
    ss_setdefault("scale", 1.0)
    ss_setdefault("filters", {})
    ss_setdefault("pinch", {})
    ss_setdefault("arming", {})
    ss_setdefault("last_palm_center", {})
    ss_setdefault("prev_pinch", {})
    ss_setdefault("two_hand", {"active": False, "d0": None, "prev_angle": None})
    ss_setdefault("lost_since", None)
    ss_setdefault("locked", False)
    ss_setdefault("armed", False)
    ss_setdefault("last_frame_t_s", None)
    ss_setdefault("frame_times", deque(maxlen=60))
    ss_setdefault("metrics", {"fps": 0.0, "latency_ms": 0.0, "frames": 0, "source": "demo"})
    ss_setdefault("event_bus", EventBus(sanitize_config(DEFAULT_CONFIG.copy())["refractory_ms"], maxlen=1000))
    ss_setdefault("trials", generate_demo_trials(sanitize_config(DEFAULT_CONFIG.copy())))
    ss_setdefault("model", default_model())
    ss_setdefault("loaded_model_name", "built-in demo model")
    ss_setdefault("recording", False)
    ss_setdefault("record_buffer", [])
    ss_setdefault("run_simulation", True)
    ss_setdefault("task_method", "Gesture")
    ss_setdefault("mouse_euler", [0.0, 0.0, 0.0])
    ss_setdefault("last_snapshot", {"hands": {}, "t_ms": 0, "dt": 0.0, "proc_ms": 0.0, "phase": ""})

    if "test_results" not in st.session_state:
        st.session_state.test_results = run_self_tests()

    if "benchmark_detail" not in st.session_state or "benchmark_summary" not in st.session_state:
        detail, summary = generate_benchmark(sanitize_config(DEFAULT_CONFIG.copy()), seed=17)
        st.session_state.benchmark_detail = detail
        st.session_state.benchmark_summary = summary


def reset_controls() -> None:
    """Reset pose, scale, arming, locks, filters, and mouse slider state."""
    st.session_state.quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    st.session_state.scale = 1.0
    st.session_state.armed = False
    st.session_state.locked = False
    st.session_state.lost_since = None
    st.session_state.last_frame_t_s = None
    st.session_state.filters = {}
    st.session_state.pinch = {}
    st.session_state.arming = {}
    st.session_state.last_palm_center = {}
    st.session_state.prev_pinch = {}
    st.session_state.two_hand = {"active": False, "d0": None, "prev_angle": None}
    st.session_state.event_bus = EventBus(st.session_state.config["refractory_ms"], maxlen=1000)
    st.session_state.mouse_euler = [0.0, 0.0, 0.0]
    for k in ["mouse_rx", "mouse_ry", "mouse_rz"]:
        if k in st.session_state:
            st.session_state[k] = 0.0


def start_trial(method: str) -> None:
    """Begin a new orientation-matching trial."""
    rng = np.random.default_rng(int(time.time() * 1000.0) % 1000000)
    st.session_state.trial = {
        "id": uuid.uuid4().hex[:8],
        "method": method,
        "target_q": random_quaternion(rng),
        "start_time": time.time(),
        "logged": False,
    }
    st.session_state.quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    st.session_state.scale = 1.0
    if method.lower().startswith("mouse"):
        st.session_state.mouse_euler = [0.0, 0.0, 0.0]
        for k in ["mouse_rx", "mouse_ry", "mouse_rz"]:
            if k in st.session_state:
                st.session_state[k] = 0.0


def log_trial(error_deg: float, timed_out: bool) -> None:
    """Append the current trial result to the task log."""
    trial = st.session_state.get("trial")
    if not trial or trial.get("logged"):
        return
    cfg = st.session_state.config
    elapsed = time.time() - float(trial["start_time"])
    row = {
        "trial_id": trial["id"],
        "method": trial["method"],
        "user": "current",
        "time_s": round(float(elapsed), 3),
        "error_deg": round(float(error_deg), 3),
        "success": bool((not timed_out) and float(error_deg) <= float(cfg["task_success_angle_deg"])),
        "timed_out": bool(timed_out),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    st.session_state.trials.append(row)
    trial["logged"] = True


def process_frame(frame: Dict[str, Any]) -> Dict[str, Any]:
    """Filter landmarks, detect gestures, update pose, and record metrics for one frame."""
    cfg = st.session_state.config
    eps = float(cfg["epsilon"])
    t_ms = int(frame.get("t_ms", 0))
    t_s = t_ms / 1000.0

    last = st.session_state.get("last_frame_t_s")
    if last is None:
        dt = 1.0 / max(1.0, float(cfg["target_fps"]))
    else:
        dt = max(eps, min(0.1, t_s - float(last)))
    st.session_state.last_frame_t_s = t_s

    proc_t0 = time.perf_counter()
    raw_hands = frame.get("hands", {})
    filtered: Dict[str, Dict[str, Any]] = {}

    for hid, lms in raw_hands.items():
        arr = np.asarray(lms, dtype=float)
        if arr.shape != (21, 3):
            continue

        if hid not in st.session_state.filters:
            st.session_state.filters[hid] = OneEuroVector(
                arr.shape,
                min_cutoff=float(cfg["one_euro_min_cutoff"]),
                beta=float(cfg["one_euro_beta"]),
                d_cutoff=float(cfg["one_euro_d_cutoff"]),
            )

        sm = st.session_state.filters[hid].filter(arr, t_s)
        m = compute_palm_metrics(sm, eps)
        if m is None or float(m["palm_width"]) < float(cfg["palm_width_min"]):
            continue

        if hid not in st.session_state.pinch:
            st.session_state.pinch[hid] = PinchDetector(float(cfg["pinch_enter"]), float(cfg["pinch_exit"]))
        pinch = st.session_state.pinch[hid].update(float(m["pinch_ratio"]))

        open_palm = bool(float(m["pinch_ratio"]) > float(cfg["open_palm_max_pinch"]))

        if hid not in st.session_state.arming:
            st.session_state.arming[hid] = HoldTimer(float(cfg["arming_hold_ms"]))
        armed_now = st.session_state.arming[hid].update(open_palm, t_ms)

        if armed_now:
            st.session_state.armed = True
            st.session_state.event_bus.emit("arm_ready", hid, t_ms, 0.85, {"hand": hid})

        filtered[hid] = {
            "landmarks": sm,
            "palm_width": float(m["palm_width"]),
            "palm_center": np.asarray(m["palm_center"], dtype=float),
            "palm_normal": np.asarray(m["palm_normal"], dtype=float),
            "pinch_ratio": float(m["pinch_ratio"]),
            "pinch": bool(pinch),
            "open": bool(open_palm),
        }

    bus = st.session_state.event_bus
    bus.refractory_ms = float(cfg["refractory_ms"])

    for hid, h in filtered.items():
        prev = bool(st.session_state.prev_pinch.get(hid, False))
        if h["pinch"] and not prev:
            bus.emit("pinch_start", hid, t_ms, 0.90, {"pinch_ratio": float(h["pinch_ratio"])})
        elif (not h["pinch"]) and prev:
            bus.emit("pinch_end", hid, t_ms, 0.90, {"pinch_ratio": float(h["pinch_ratio"])})
        st.session_state.prev_pinch[hid] = h["pinch"]

    for store in ["prev_pinch", "last_palm_center", "pinch", "arming"]:
        for hid in list(st.session_state[store].keys()):
            if hid not in filtered:
                del st.session_state[store][hid]

    valid = bool(filtered)
    locked, lost_since, lock_event = tracking_lock_update(
        valid,
        st.session_state.locked,
        st.session_state.lost_since,
        t_ms,
        float(cfg["tracking_loss_lock_ms"]),
    )
    st.session_state.locked = locked
    st.session_state.lost_since = lost_since

    if lock_event == "tracking_lost":
        bus.emit("tracking_lost", "global", t_ms, 1.0, {})
        st.session_state.armed = False
        st.session_state.two_hand = {"active": False, "d0": None, "prev_angle": None}
        st.session_state.pinch = {}
        st.session_state.arming = {}
    elif lock_event == "tracking_resumed":
        bus.emit("tracking_resumed", "global", t_ms, 1.0, {})
        for hid, h in filtered.items():
            st.session_state.last_palm_center[hid] = np.asarray(h["palm_center"][:2], dtype=float).copy()
        st.session_state.two_hand = {"active": False, "d0": None, "prev_angle": None}
        st.session_state.pinch = {}
        st.session_state.arming = {}

    if valid and (not locked) and st.session_state.armed:
        # One-hand rotation while pinching.
        for hid, h in filtered.items():
            center = np.asarray(h["palm_center"][:2], dtype=float)
            prev = st.session_state.last_palm_center.get(hid)
            if prev is None:
                prev = center.copy()

            if h["pinch"]:
                dq = rotation_delta_quat(center - prev, cfg, dt)
                st.session_state.quat = q_normalize(q_mul(dq, st.session_state.quat))

            st.session_state.last_palm_center[hid] = center.copy()

        # Two-hand scale and roll.
        hids = sorted(filtered.keys())[:2]
        require_pinch = float(cfg.get("two_hand_require_pinch", 1.0)) > 0.5
        two_active = len(filtered) >= 2
        if two_active and require_pinch:
            two_active = bool(filtered[hids[0]]["pinch"] and filtered[hids[1]]["pinch"])

        if two_active:
            c0 = np.asarray(filtered[hids[0]]["palm_center"][:2], dtype=float)
            c1 = np.asarray(filtered[hids[1]]["palm_center"][:2], dtype=float)
            avg_pw = 0.5 * (float(filtered[hids[0]]["palm_width"]) + float(filtered[hids[1]]["palm_width"]))
            d_norm = float(np.linalg.norm(c1 - c0) / max(avg_pw, eps))

            th = st.session_state.two_hand
            if not th.get("active"):
                th = {
                    "active": True,
                    "d0": d_norm,
                    "prev_angle": math.atan2(float(c1[1] - c0[1]), float(c1[0] - c0[0])),
                }
                bus.emit("two_hand_start", "global", t_ms, 0.90, {"distance_norm": float(d_norm)})
            else:
                st.session_state.scale = scale_update(d_norm, th.get("d0"), cfg, float(st.session_state.scale), dt)

                angle = math.atan2(float(c1[1] - c0[1]), float(c1[0] - c0[0]))
                dq = roll_delta_quat(angle, th.get("prev_angle"), cfg, dt)
                st.session_state.quat = q_normalize(q_mul(dq, st.session_state.quat))
                th["prev_angle"] = angle

            st.session_state.two_hand = th
        else:
            if st.session_state.two_hand.get("active"):
                st.session_state.two_hand = {"active": False, "d0": None, "prev_angle": None}
                bus.emit("two_hand_end", "global", t_ms, 0.90, {})
    else:
        if valid and (not locked):
            for hid, h in filtered.items():
                st.session_state.last_palm_center[hid] = np.asarray(h["palm_center"][:2], dtype=float).copy()

        if st.session_state.two_hand.get("active"):
            st.session_state.two_hand = {"active": False, "d0": None, "prev_angle": None}
            bus.emit("two_hand_end", "global", t_ms, 0.90, {})

    proc_ms = (time.perf_counter() - proc_t0) * 1000.0

    times = st.session_state.frame_times
    now = time.perf_counter()
    times.append(now)
    fps = 0.0
    if len(times) >= 2:
        span = times[-1] - times[0]
        if span > eps:
            fps = float((len(times) - 1) / span)

    latency = float(frame.get("capture_ms", 0.0)) + float(frame.get("tracking_ms", 0.0)) + float(proc_ms)
    st.session_state.metrics = {
        "fps": fps,
        "latency_ms": latency,
        "frames": int(st.session_state.metrics.get("frames", 0)) + 1,
        "source": frame.get("source", "demo"),
    }

    if st.session_state.get("recording", False):
        buf = st.session_state.record_buffer
        buf.append(
            {
                "t_ms": t_ms,
                "source": frame.get("source", "demo"),
                "phase": frame.get("phase", ""),
                "fps": round(fps, 3),
                "latency_ms": round(latency, 3),
                "armed": bool(st.session_state.armed),
                "locked": bool(st.session_state.locked),
                "scale": round(float(st.session_state.scale), 5),
                "quat_w": round(float(st.session_state.quat[0]), 6),
                "quat_x": round(float(st.session_state.quat[1]), 6),
                "quat_y": round(float(st.session_state.quat[2]), 6),
                "quat_z": round(float(st.session_state.quat[3]), 6),
                "hands": int(len(filtered)),
            }
        )
        if len(buf) > 5000:
            buf.pop(0)

    return {
        "hands": filtered,
        "t_ms": t_ms,
        "dt": dt,
        "proc_ms": proc_ms,
        "phase": frame.get("phase", ""),
    }


# =============================================================================
# UI HELPERS
# =============================================================================

def privacy_banner(mode: str, recording: bool) -> str:
    """Plain-language privacy and capture status line."""
    if mode == "camera":
        cam = "🔴 Webcam is ON. Processing is local-only. Please tell people in view that the camera is on."
    elif mode == "demo":
        cam = "🟢 Automatic demo mode. Webcam is OFF."
    else:
        cam = "⚪ Simulation is stopped."
    rec = "⏺ Recording local session log." if recording else "⏹ Not recording."
    return f"{cam} {rec} No video, audio, or landmark data is uploaded."


def events_dataframe(limit: int = 20) -> pd.DataFrame:
    """Convert recent GestureEvent objects to a table."""
    events = list(st.session_state.event_bus.events)[:limit]
    cols = ["type", "hand", "t_ms", "confidence", "params"]
    if not events:
        return pd.DataFrame(columns=cols)
    rows = []
    for e in events:
        rows.append(
            {
                "type": e.type,
                "hand": e.hand,
                "t_ms": e.t_ms,
                "confidence": round(float(e.confidence), 3),
                "params": json.dumps(e.params, default=str),
            }
        )
    return pd.DataFrame(rows, columns=cols)


def metrics_dataframe(snapshot: Dict[str, Any]) -> pd.DataFrame:
    """Compact live metrics table."""
    m = st.session_state.metrics
    q = np.round(st.session_state.quat, 4).tolist()
    return pd.DataFrame(
        [
            {
                "FPS": round(float(m.get("fps", 0.0)), 1),
                "Latency (ms)": round(float(m.get("latency_ms", 0.0)), 1),
                "Processed frames": int(m.get("frames", 0)),
                "Source": m.get("source", "demo"),
                "Phase": snapshot.get("phase", ""),
                "Scale (x)": round(float(st.session_state.scale), 3),
                "Quaternion [w,x,y,z]": q,
                "Armed": bool(st.session_state.armed),
                "Locked": bool(st.session_state.locked),
            }
        ]
    )


def feedback_markdown(snapshot: Dict[str, Any]) -> str:
    """User-facing live feedback text."""
    lines = []

    if snapshot.get("phase"):
        lines.append(f"**Demo/control phase:** {snapshot['phase']}")

    lines.append(
        f"**Armed:** {'✅' if st.session_state.armed else '❌'} | "
        f"**Locked:** {'🔒' if st.session_state.locked else '🔓'} | "
        f"**Scale:** {float(st.session_state.scale):.3f}×"
    )

    hands = snapshot.get("hands", {})
    if hands:
        for hid, h in hands.items():
            pc = h["palm_center"]
            pn = h["palm_normal"]
            lines.append(
                f"- **{hid}**: pinch={'🤏' if h['pinch'] else '✋'}, "
                f"ratio={h['pinch_ratio']:.3f}, "
                f"palm=({pc[0]:.3f}, {pc[1]:.3f}, {pc[2]:.3f}), "
                f"normal=({pn[0]:.2f}, {pn[1]:.2f}, {pn[2]:.2f})"
            )
    else:
        lines.append("- No valid hands. In demo mode this happens during the tracking-loss interval.")

    ev = events_dataframe(5)
    if not ev.empty:
        lines.append("")
        lines.append("**Recent gestures**")
        lines.append(ev[["type", "hand", "t_ms", "confidence"]].to_string(index=False))

    return "\n\n".join(lines)


def render_live(snapshot: Dict[str, Any], areas: Dict[str, Any]) -> None:
    """Render the main live 3D control view."""
    fig = make_figure(st.session_state.model, st.session_state.quat, st.session_state.scale)
    areas["view"].plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False},
        key="plot_live_view",
    )
    areas["feedback"].markdown(feedback_markdown(snapshot))
    areas["metrics"].dataframe(metrics_dataframe(snapshot), use_container_width=True)


def render_debug(snapshot: Dict[str, Any], areas: Dict[str, Any]) -> None:
    """Render landmark/debug tables."""
    if snapshot.get("phase"):
        areas["hands"].markdown(f"**Demo/control phase:** {snapshot['phase']}")

    hands = snapshot.get("hands", {})
    if hands:
        rows = []
        for hid, h in hands.items():
            rows.append(
                {
                    "hand": hid,
                    "palm_x": round(float(h["palm_center"][0]), 4),
                    "palm_y": round(float(h["palm_center"][1]), 4),
                    "palm_z": round(float(h["palm_center"][2]), 4),
                    "normal_x": round(float(h["palm_normal"][0]), 3),
                    "normal_y": round(float(h["palm_normal"][1]), 3),
                    "normal_z": round(float(h["palm_normal"][2]), 3),
                    "palm_width": round(float(h["palm_width"]), 4),
                    "pinch_ratio": round(float(h["pinch_ratio"]), 4),
                    "pinch": bool(h["pinch"]),
                    "open": bool(h["open"]),
                }
            )
        areas["hands"].dataframe(pd.DataFrame(rows), use_container_width=True)
    else:
        areas["hands"].info("No valid hands. In demo mode this happens during the tracking-loss interval.")

    areas["events"].dataframe(events_dataframe(30), use_container_width=True)
    areas["metrics"].dataframe(metrics_dataframe(snapshot), use_container_width=True)


def render_task_gesture(snapshot: Dict[str, Any], areas: Dict[str, Any]) -> None:
    """Render gesture-based orientation task."""
    trial = st.session_state.get("trial")
    target_q = trial["target_q"] if trial and (not trial.get("logged")) else None

    fig = make_figure(st.session_state.model, st.session_state.quat, st.session_state.scale, target_quat=target_q)
    areas["view"].plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False},
        key="plot_task_view",
    )

    cfg = st.session_state.config

    if trial and not trial.get("logged"):
        err = angle_between_quats(st.session_state.quat, trial["target_q"])
        elapsed = time.time() - float(trial["start_time"])
        remaining = max(0.0, float(cfg["task_timeout_s"]) - elapsed)
        areas["task"].markdown(
            f"**Trial {trial['id']} — Gesture**\n\n"
            f"- Target ghost shown in orange.\n"
            f"- Error: {err:.2f}°\n"
            f"- Elapsed: {elapsed:.2f} s\n"
            f"- Remaining: {remaining:.2f} s\n"
            f"- Success if ≤ {float(cfg['task_success_angle_deg']):.1f}°"
        )

        if err <= float(cfg["task_success_angle_deg"]):
            log_trial(err, False)
            areas["task"].success("Trial logged: success.")
        elif elapsed >= float(cfg["task_timeout_s"]):
            log_trial(err, True)
            areas["task"].warning("Trial logged: timeout.")
    else:
        areas["task"].markdown("No active gesture trial. Press **Start gesture trial** above.")

    df = pd.DataFrame(st.session_state.trials)
    if not df.empty:
        areas["log"].dataframe(df.tail(20), use_container_width=True)
    else:
        areas["log"].info("No trials logged yet.")


def render_mouse_task() -> None:
    """Render mouse/slider comparison task."""
    cfg = st.session_state.config

    if st.button("Start mouse trial", key="btn_start_mouse_trial"):
        start_trial("Mouse")

    if "mouse_euler" not in st.session_state:
        st.session_state.mouse_euler = [0.0, 0.0, 0.0]

    rx = st.slider("Rotate X (pitch) °", -180.0, 180.0, float(st.session_state.mouse_euler[0]), key="mouse_rx")
    ry = st.slider("Rotate Y (yaw) °", -180.0, 180.0, float(st.session_state.mouse_euler[1]), key="mouse_ry")
    rz = st.slider("Rotate Z (roll) °", -180.0, 180.0, float(st.session_state.mouse_euler[2]), key="mouse_rz")

    st.session_state.mouse_euler = [rx, ry, rz]
    st.session_state.quat = q_from_euler_deg(rx, ry, rz)

    trial = st.session_state.get("trial")
    target_q = None
    if trial and (not trial.get("logged")) and str(trial.get("method", "")).lower().startswith("mouse"):
        target_q = trial["target_q"]

    if trial and (not trial.get("logged")) and str(trial.get("method", "")).lower().startswith("mouse"):
        err = angle_between_quats(st.session_state.quat, trial["target_q"])
        elapsed = time.time() - float(trial["start_time"])
        st.metric("Error to target", f"{err:.2f}°")
        st.metric("Elapsed", f"{elapsed:.2f} s")
        if st.button("Log mouse attempt", key="btn_log_mouse_attempt"):
            log_trial(err, elapsed >= float(cfg["task_timeout_s"]))
    elif trial and trial.get("logged"):
        st.info("Mouse trial logged.")
    else:
        st.info("Start a mouse trial to log an attempt.")

    st.plotly_chart(
        make_figure(st.session_state.model, st.session_state.quat, st.session_state.scale, target_quat=target_q),
        use_container_width=True,
        config={"displayModeBar": False},
        key="plot_mouse_view",
    )

    df = pd.DataFrame(st.session_state.trials)
    if not df.empty:
        st.dataframe(df.tail(20), use_container_width=True)


def render_evaluation() -> None:
    """Render simulated evaluation summary."""
    st.subheader("Evaluation")
    st.caption("Synthetic demo data: 5 users × 20 trials per method. Replace with real user logs for publication.")

    if st.button("Regenerate simulated evaluation", key="btn_regen_eval"):
        detail, summary = generate_benchmark(st.session_state.config, seed=int(time.time() * 1000.0) % 1000000)
        st.session_state.benchmark_detail = detail
        st.session_state.benchmark_summary = summary

    st.markdown("**Summary**")
    st.dataframe(st.session_state.benchmark_summary, use_container_width=True)

    st.markdown("**Trial detail, first 100 rows**")
    st.dataframe(st.session_state.benchmark_detail.head(100), use_container_width=True)

    st.download_button(
        "Download benchmark CSV",
        st.session_state.benchmark_detail.to_csv(index=False).encode("utf-8"),
        file_name="benchmark.csv",
        mime="text/csv",
        key="dl_benchmark",
    )


def render_self_tests_page() -> None:
    """Run and display pure-function tests."""
    st.subheader("Self-tests")
    if st.button("Run tests now", key="btn_run_tests"):
        st.session_state.test_results = run_self_tests()

    df = st.session_state.test_results
    passes = int((df["Result"] == "PASS").sum())
    total = int(len(df))
    st.metric("Passed", f"{passes}/{total}")
    st.dataframe(df, use_container_width=True)


def save_recording() -> list:
    """Save local CSV logs only when the user explicitly presses Save."""
    paths = []
    DATA_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    buf = st.session_state.get("record_buffer", [])
    if buf:
        p = DATA_DIR / f"session_{stamp}.csv"
        pd.DataFrame(buf).to_csv(p, index=False)
        paths.append(str(p))

    ev = events_dataframe(10000)
    if not ev.empty:
        p = DATA_DIR / f"events_{stamp}.csv"
        ev.to_csv(p, index=False)
        paths.append(str(p))

    tr = pd.DataFrame(st.session_state.get("trials", []))
    if not tr.empty:
        p = DATA_DIR / f"trials_{stamp}.csv"
        tr.to_csv(p, index=False)
        paths.append(str(p))

    st.session_state.record_buffer = []
    return paths


def delete_all_data() -> None:
    """Stop capture, delete local data folder, clear session, and restart."""
    source = st.session_state.get("frame_source")
    if source is not None:
        source.stop()
    shutil.rmtree(DATA_DIR, ignore_errors=True)
    st.session_state.clear()
    st.rerun()


def render_data_page() -> None:
    """Render privacy, logs, downloads, mapping export, and license table."""
    st.subheader("Data, privacy, and licenses")

    source = st.session_state.get("frame_source")
    mode = source.actual_mode if source is not None else "stopped"
    st.markdown(privacy_banner(mode, bool(st.session_state.get("recording", False))))

    if st.button("Delete all local data and reset session", key="btn_delete_all_data"):
        delete_all_data()

    st.markdown("**Gesture event log**")
    ev = events_dataframe(100)
    st.dataframe(ev, use_container_width=True)
    st.download_button(
        "Download events CSV",
        ev.to_csv(index=False).encode("utf-8"),
        file_name="gesture_events.csv",
        mime="text/csv",
        key="dl_events",
    )

    st.markdown("**Task log**")
    tr = pd.DataFrame(st.session_state.trials)
    st.dataframe(tr, use_container_width=True)
    st.download_button(
        "Download task log CSV",
        tr.to_csv(index=False).encode("utf-8"),
        file_name="task_log.csv",
        mime="text/csv",
        key="dl_task_log",
    )

    st.download_button(
        "Download mapping.json",
        json.dumps(mapping_payload(st.session_state.config), indent=2).encode("utf-8"),
        file_name="mapping.json",
        mime="application/json",
        key="dl_mapping_json",
    )

    st.markdown("**Library and asset licenses, no versions shown**")
    license_rows = [
        {"Item": "Streamlit", "License": "Apache-2.0"},
        {"Item": "NumPy", "License": "BSD-3-Clause"},
        {"Item": "pandas", "License": "BSD-3-Clause"},
        {"Item": "Plotly", "License": "MIT"},
        {"Item": "MediaPipe", "License": "Apache-2.0"},
        {"Item": "OpenCV", "License": "Apache-2.0"},
        {"Item": "trimesh", "License": "MIT"},
        {"Item": "Pillow", "License": "MIT-CMU / historical PIL license"},
        {"Item": "Built-in demo icosphere/cube", "License": "Generated locally by this app"},
    ]
    st.dataframe(pd.DataFrame(license_rows), use_container_width=True)


# =============================================================================
# MAIN APP
# =============================================================================

st.set_page_config(page_title="HandSculpt", page_icon="✋", layout="wide", initial_sidebar_state="expanded")

init_session()

cfg = st.session_state.config

st.sidebar.title("✋ HandSculpt")
st.sidebar.caption("Local-only 3D model control prototype. No API key. No upload.")

page = st.sidebar.selectbox(
    "View",
    ["Live 3D", "Hand debug", "Task lab", "Evaluation", "Self-tests", "Data & privacy"],
    key="page_select",
)

input_mode = st.sidebar.radio(
    "Input source",
    ["Automatic demo (recommended)", "Local webcam (optional)"],
    key="input_source_radio",
)

camera_index = 0
if "webcam" in input_mode.lower():
    camera_index = int(
        st.sidebar.number_input(
            "Camera index",
            min_value=0,
            max_value=10,
            value=0,
            step=1,
            key="camera_index_number",
        )
    )

run_sim = st.sidebar.checkbox(
    "Run live simulation",
    value=bool(st.session_state.get("run_simulation", True)),
    key="run_sim_checkbox",
)
st.session_state.run_simulation = bool(run_sim)

fps = max(5, min(60, int(cfg.get("target_fps", 30))))
demo_speed = float(cfg.get("demo_speed", 1.0))
desired_mode = "demo" if input_mode.lower().startswith("automatic") else "camera"

if st.session_state.run_simulation:
    old = st.session_state.get("frame_source")
    if (
        old is None
        or old.mode != desired_mode
        or old.camera_index != camera_index
        or old.fps != fps
        or abs(old.demo_speed - demo_speed) > 1e-6
    ):
        if old is not None:
            old.stop()
        st.session_state.frame_source = FrameSource(desired_mode, camera_index, fps, demo_speed)
        st.session_state.frame_source.start()
    source = st.session_state.frame_source
else:
    old = st.session_state.get("frame_source")
    if old is not None:
        old.stop()
    st.session_state.frame_source = None
    source = None

if st.sidebar.button("Reset controls", key="btn_reset_controls"):
    reset_controls()

if st.sidebar.button("Arm now", key="btn_arm_now"):
    st.session_state.armed = True

if st.sidebar.button("Disarm", key="btn_disarm"):
    st.session_state.armed = False

uploaded = st.sidebar.file_uploader("Upload glTF/GLB", type=["glb", "gltf"], key="model_uploader")
if uploaded is not None and st.session_state.get("loaded_model_name") != uploaded.name:
    try:
        st.session_state.model = load_model_from_file(uploaded)
        st.session_state.loaded_model_name = uploaded.name
        st.sidebar.success(f"Loaded {uploaded.name}")
    except Exception as exc:
        st.sidebar.error(f"Model load failed: {exc}")

rec1, rec2 = st.sidebar.columns(2)
if rec1.button("⏺ Record", key="btn_record"):
    st.session_state.recording = True
if rec2.button("⏹ Save", key="btn_save"):
    saved = save_recording()
    st.session_state.recording = False
    if saved:
        st.sidebar.success("Saved locally: " + ", ".join(saved))
    else:
        st.sidebar.info("Nothing to save.")

with st.sidebar.expander("mapping.json"):
    mapping_text = st.text_area(
        "Edit values",
        value=json.dumps(mapping_payload(cfg), indent=2),
        height=360,
        key="mapping_text_area",
    )
    c1, c2 = st.columns(2)
    if c1.button("Apply", key="btn_apply_mapping"):
        try:
            payload = json.loads(mapping_text)
            values = payload.get("values", payload)
            st.session_state.config = sanitize_config(values)
            st.session_state.event_bus.refractory_ms = st.session_state.config["refractory_ms"]
            st.success("Mapping applied.")
        except Exception as exc:
            st.error(f"Invalid mapping: {exc}")
    if c2.button("Reset", key="btn_reset_mapping"):
        st.session_state.config = sanitize_config(DEFAULT_CONFIG.copy())
        st.rerun()

# Refresh local config reference after possible sidebar edits.
cfg = st.session_state.config

status = st.empty()
source_mode = source.actual_mode if source is not None else "stopped"
status.markdown(privacy_banner(source_mode, bool(st.session_state.get("recording", False))))

areas = None
task_method = st.session_state.get("task_method", "Gesture")

if page == "Live 3D":
    st.subheader("Live 3D model")
    left, right = st.columns([2.2, 1.0])
    areas = {"view": left.empty(), "feedback": right.empty(), "metrics": st.empty()}

elif page == "Hand debug":
    st.subheader("Hand landmarks and gesture debug")
    areas = {"hands": st.empty(), "events": st.empty(), "metrics": st.empty()}

elif page == "Task lab":
    st.subheader("Task lab")
    task_method = st.radio(
        "Trial input method",
        ["Gesture", "Mouse (slider proxy)"],
        index=0 if task_method == "Gesture" else 1,
        key="task_method_radio",
    )
    st.session_state.task_method = task_method

    if task_method == "Gesture":
        if st.button("Start gesture trial", key="btn_start_gesture_trial"):
            start_trial("Gesture")
        left, right = st.columns([2.2, 1.0])
        areas = {"view": left.empty(), "task": right.empty(), "log": st.empty()}
    else:
        render_mouse_task()
        areas = None

elif page == "Evaluation":
    render_evaluation()
    areas = None

elif page == "Self-tests":
    render_self_tests_page()
    areas = None

else:
    render_data_page()
    areas = None

loop_active = source is not None and bool(st.session_state.get("run_simulation", True)) and areas is not None

default_snapshot = {"hands": {}, "t_ms": 0, "dt": 0.0, "proc_ms": 0.0, "phase": ""}

if loop_active:
    try:
        frame = source.get_newest()
        if frame is not None:
            snapshot = process_frame(frame)
        else:
            snapshot = st.session_state.get("last_snapshot", default_snapshot)

        st.session_state.last_snapshot = snapshot

        source_mode = source.actual_mode if source is not None else "stopped"
        status.markdown(privacy_banner(source_mode, bool(st.session_state.get("recording", False))))

        if page == "Live 3D":
            render_live(snapshot, areas)
        elif page == "Hand debug":
            render_debug(snapshot, areas)
        elif page == "Task lab" and task_method == "Gesture":
            render_task_gesture(snapshot, areas)

        period = 1.0 / max(1.0, float(cfg.get("target_fps", 30)))
        time.sleep(period)
        st.rerun()

    except Exception as exc:
        st.error(f"Live loop error: {exc}")

else:
    snapshot = st.session_state.get("last_snapshot", default_snapshot)
    source_mode = source.actual_mode if source is not None else "stopped"
    status.markdown(privacy_banner(source_mode, bool(st.session_state.get("recording", False))))

    if page == "Live 3D" and areas is not None:
        render_live(snapshot, areas)
    elif page == "Hand debug" and areas is not None:
        render_debug(snapshot, areas)
    elif page == "Task lab" and task_method == "Gesture" and areas is not None:
        render_task_gesture(snapshot, areas)