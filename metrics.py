# metrics.py
# Football player tracking metrics – M4 friendly, numpy only
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple

def smooth_1d(x: np.ndarray, win: int = 5) -> np.ndarray:
    if win <= 1 or len(x) < win: return x
    k = np.ones(win)/win
    return np.convolve(x, k, mode="same")

class TrackMetrics:
    """
    Accumulates per-track center positions and computes:
      - speed (mph)
      - acceleration
      - distance_ran
      - displacement_from_snap
    All in field units (yards if px_per_yard > 0, else pixels).
    """
    def __init__(self, fps: float = 30.0, px_per_yard: float = 0.0):
        self.fps = fps
        self.px_per_yard = px_per_yard if px_per_yard > 0 else None  # None = px mode
        self.units = "yd" if self.px_per_yard else "px"
        # track_id -> list[(frame_idx, cx, cy)]
        self.history: Dict[int, List[Tuple[int,float,float]]] = defaultdict(list)

    def update(self, frame_idx: int, tracks: List[Dict]):
        """tracks: [{'id': int, 'x1':…, 'y1':…, 'x2':…, 'y2':…}, …]"""
        for t in tracks:
            tid = int(t["id"])
            cx = (t["x1"]+t["x2"])*0.5
            cy = (t["y1"]+t["y2"])*0.5
            self.history[tid].append((frame_idx, cx, cy))

    def _series(self, tid: int):
        h = self.history.get(tid, [])
        if len(h) < 2: return None, None, None
        h = sorted(h)
        frames = np.array([f for f,_,_ in h], dtype=float)
        xs = np.array([x for _,x,_ in h], dtype=float)
        ys = np.array([y for _,_,y in h], dtype=float)
        return frames, xs, ys

    def speed_series(self, tid: int, smooth_win: int = 5):
        frames, xs, ys = self._series(tid)
        if frames is None: return np.array([]), np.array([])
        # px / frame
        dx = np.diff(xs); dy = np.diff(ys)
        dist_px = np.sqrt(dx*dx + dy*dy)
        dt_frames = np.diff(frames); dt_frames[dt_frames==0] = 1
        v_px_per_frame = dist_px / dt_frames
        # convert to units/sec then mph if yards
        v = v_px_per_frame * self.fps
        if self.px_per_yard:
            v_yd_s = v / self.px_per_yard
            v_mph = v_yd_s * 3600.0 / 1760.0
            v_out = v_mph
        else:
            v_out = v  # px/s
        t_mid = (frames[1:] + frames[:-1]) * 0.5
        if smooth_win > 1:
            v_out = smooth_1d(v_out, smooth_win)
        return t_mid, v_out

    def accel_series(self, tid: int, delta_frames: int = 50, smooth_win: int = 5):
        """Acceleration as Δv over delta_frames window, in (mph/s) or (px/s²)."""
        t, v = self.speed_series(tid, smooth_win=1)
        if len(v) < 2: return np.array([]), np.array([])
        # v is in mph or px/s, t is in frames
        # convert t to seconds
        t_sec = t / self.fps
        # rolling difference over delta_frames
        df = max(1, int(delta_frames))
        if len(v) <= df:
            return np.array([]), np.array([])
        dv = v[df:] - v[:-df]
        dt = t_sec[df:] - t_sec[:-df]
        dt[dt == 0] = 1e-3
        a = dv / dt
        t_a = t_sec[df:]
        if smooth_win > 1:
            a = smooth_1d(a, smooth_win)
        return t_a * self.fps, a  # return time in frames for easy sync

    def distance_ran(self, tid: int) -> float:
        frames, xs, ys = self._series(tid)
        if frames is None: return 0.0
        d_px = np.sum(np.sqrt(np.diff(xs)**2 + np.diff(ys)**2))
        if self.px_per_yard: return d_px / self.px_per_yard
        return d_px

    def displacement_from_snap(self, tid: int, at_frame: int = None) -> float:
        h = self.history.get(tid, [])
        if len(h) < 1: return 0.0
        h = sorted(h)
        x0, y0 = h[0][1], h[0][2]
        if at_frame is None:
            x1, y1 = h[-1][1], h[-1][2]
        else:
            # nearest <= at_frame
            candidates = [p for p in h if p[0] <= at_frame]
            if not candidates: return 0.0
            x1, y1 = candidates[-1][1], candidates[-1][2]
        d_px = np.hypot(x1-x0, y1-y0)
        if self.px_per_yard: return d_px / self.px_per_yard
        return d_px

    def summary(self, tid: int, start_frame: int = None, end_frame: int = None, accel_delta: int = 50):
        """Return dict with avg_speed, max_speed, max_accel over optional frame window."""
        t_v, v = self.speed_series(tid)
        if len(v):
            mask = np.ones(len(t_v), dtype=bool)
            if start_frame is not None: mask &= t_v >= start_frame
            if end_frame is not None: mask &= t_v <= end_frame
            v_sel = v[mask] if mask.any() else v
            avg_speed = float(np.mean(v_sel)) if len(v_sel) else 0.0
            max_speed = float(np.max(v_sel)) if len(v_sel) else 0.0
        else:
            avg_speed = max_speed = 0.0

        t_a, a = self.accel_series(tid, delta_frames=accel_delta)
        if len(a):
            mask = np.ones(len(t_a), dtype=bool)
            if start_frame is not None: mask &= t_a >= start_frame
            if end_frame is not None: mask &= t_a <= end_frame
            a_sel = a[mask] if mask.any() else a
            max_accel = float(np.max(np.abs(a_sel))) if len(a_sel) else 0.0
        else:
            max_accel = 0.0

        dist = self.distance_ran(tid)
        disp = self.displacement_from_snap(tid, end_frame)
        return {
            "avg_speed": avg_speed,
            "max_speed": max_speed,
            "max_accel": max_accel,
            "distance_ran": dist,
            "displacement": disp,
            "units_speed": "mph" if self.px_per_yard else "px/s",
            "units_dist": self.units,
        }

    def player_distance(self, tid_a: int, tid_b: int, frame_idx: int = None) -> float:
        """Euclidean distance between two players at given frame (nearest)."""
        ha = self.history.get(tid_a, [])
        hb = self.history.get(tid_b, [])
        if not ha or not hb: return 0.0
        def pos_at(hist, f):
            if f is None: return hist[-1][1], hist[-1][2]
            cand = [p for p in hist if p[0] <= f]
            if not cand: cand = [hist[0]]
            return cand[-1][1], cand[-1][2]
        xa, ya = pos_at(sorted(ha), frame_idx)
        xb, yb = pos_at(sorted(hb), frame_idx)
        d_px = np.hypot(xa-xb, ya-yb)
        if self.px_per_yard: return d_px / self.px_per_yard
        return d_px
