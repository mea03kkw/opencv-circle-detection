
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Circle Inspector (ALL-IN-ONE) — Stable Tuned Version (Option A)

Purpose
-------
A contour-first OpenCV pipeline tuned for printed outline circles on paper
(radius ≈ 40–60 px at 720p). Uses robust, tolerant thresholds and a slightly
more generous edge pipeline so GOOD circles do not flip to BAD under typical
lighting/tilt, while still catching broken/dashed/jagged/elliptical shapes.

Controls
--------
q = Quit
s = Save current annotated frame
c = Print per-circle metrics to console

Notes
-----
- Classification is driven by contour features (NOT Hough), so even broken or
  jagged circles can be labeled correctly.
- Paper gating is kept conservative; you can disable it by setting
  PAPER_ONLY=False below if you want to test with hands/phones in frame.

Dependencies
------------
  pip install opencv-python numpy
  (pyyaml is optional only if you plan to load a YAML config)
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# ========================== Stable Tuned Defaults ============================
# Camera
FRAME_WIDTH  = 1280
FRAME_HEIGHT = 720

# Candidate radius band (pixels)
MIN_RADIUS = 28      # tolerant low end so smaller printed circles still pass
MAX_RADIUS = 110     # tolerant high end (adjust if you work much closer/farther)

# Edge detection & morphology (generous for thin printed rings)
CANNY_LOW   = 40     # was 60
CANNY_HIGH  = 120    # was 140
DILATE_ITER = 2      # was 1 — helps connect faint edges
ERODE_ITER  = 0
CLOSE_ITER  = 1      # morphology close to bridge micro-gaps

# Classification thresholds (tolerant)
PER_RATIO_MIN_GOOD = 0.72   # arcLength/(2πr) for smooth/mostly-complete
PER_RATIO_BAD      = 0.50   # dashed/broken if below this
AXIS_RATIO_MAX     = 1.15   # ellipse reject (major/minor)
STD_NORM_GOOD      = 0.08   # std(distance_to_center)/r for smooth edge
STD_NORM_BAD       = 0.14   # jagged if above this

# Paper gating (ignore dark / non-paper interiors)
PAPER_ONLY  = True
PAPER_V_MIN = 120     # HSV V brightness inside the circle; lower = more tolerant

# Output
OUTPUT_DIR = 'output'

# ============================================================================

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


class CircleInspector:
    def __init__(self, camera_id: int = 0, config_path: Optional[str] = None) -> None:
        # (Optional) parse a YAML config — but we default to stable tuned values
        self.cfg = self._load_config(config_path)

        # Camera
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  int(self.cfg.get('frame_width',  FRAME_WIDTH)))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(self.cfg.get('frame_height', FRAME_HEIGHT)))

        # Radius band
        self.min_radius = int(self.cfg.get('min_radius', MIN_RADIUS))
        self.max_radius = int(self.cfg.get('max_radius', MAX_RADIUS))

        # Edge & morphology
        self.canny1      = int(self.cfg.get('canny_low',   CANNY_LOW))
        self.canny2      = int(self.cfg.get('canny_high',  CANNY_HIGH))
        self.dilate_iter = int(self.cfg.get('dilate_iter', DILATE_ITER))
        self.erode_iter  = int(self.cfg.get('erode_iter',  ERODE_ITER))
        self.close_iter  = int(self.cfg.get('close_iter',  CLOSE_ITER))

        # Classification thresholds
        self.per_ratio_min_good = float(self.cfg.get('per_ratio_min_good', PER_RATIO_MIN_GOOD))
        self.per_ratio_bad      = float(self.cfg.get('per_ratio_bad',      PER_RATIO_BAD))
        self.axis_ratio_max     = float(self.cfg.get('axis_ratio_max',     AXIS_RATIO_MAX))
        self.std_norm_good      = float(self.cfg.get('std_norm_good',      STD_NORM_GOOD))
        self.std_norm_bad       = float(self.cfg.get('std_norm_bad',       STD_NORM_BAD))

        # Paper gating
        self.paper_only  = bool(self.cfg.get('paper_only',  PAPER_ONLY))
        self.paper_v_min = int(self.cfg.get('paper_v_min', PAPER_V_MIN))

        # HUD
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.col_good = (0, 255, 0)
        self.col_bad  = (0, 0, 255)
        self.col_info = (255, 255, 255)

        # Output
        self.output_dir = Path(self.cfg.get('output_dir', OUTPUT_DIR))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Stats
        self.fps = 0.0
        self._t0 = time.time()
        self._frames = 0

        print("CircleInspector (Stable Tuned) initialized. Press 'q' to quit.")

    # ---------------------------- Config (optional) -------------------------
    def _load_config(self, path: Optional[str]) -> Dict:
        if not path:
            return {}
        try:
            import yaml
            with open(path, 'r', encoding='utf-8') as f:
                cfg = yaml.safe_load(f) or {}
                print(f"Loaded config: {path}")
                return cfg
        except Exception as e:
            print(f"[WARN] Could not load config '{path}': {e}")
            return {}

    # --------------------------- Preprocess --------------------------------
    def preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g = clahe.apply(gray)
        g = cv2.GaussianBlur(g, (5, 5), 0)

        edges = cv2.Canny(g, self.canny1, self.canny2)
        if self.dilate_iter > 0:
            edges = cv2.dilate(edges, None, iterations=self.dilate_iter)
        if self.erode_iter > 0:
            edges = cv2.erode(edges, None, iterations=self.erode_iter)
        if self.close_iter > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=self.close_iter)

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        return g, edges, hsv

    # ----------------------------- Detect ----------------------------------
    def detect_contours(self, edges: np.ndarray) -> List[np.ndarray]:
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        return contours

    # ----------------------------- Features --------------------------------
    def _features_from_contour(self, cnt: np.ndarray, hsv: np.ndarray, gray: np.ndarray) -> Optional[Dict]:
        if len(cnt) < 10:
            return None
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        cx, cy, r = int(cx), int(cy), float(r)
        if r < self.min_radius or r > self.max_radius:
            return None

        per = float(cv2.arcLength(cnt, True))
        ideal = 2.0 * np.pi * r
        per_ratio = clamp(per / ideal, 0.0, 2.0)

        pts = cnt.reshape(-1, 2)
        d = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
        std_norm = float(np.std(d) / max(r, 1e-6))

        axis_ratio = 1.0
        if len(cnt) >= 5:
            try:
                (_, axes, _) = cv2.fitEllipse(cnt)
                major, minor = float(max(axes)), float(min(axes))
                if minor > 1e-6:
                    axis_ratio = major / minor
            except Exception:
                axis_ratio = 1.0

        # Paper gating (HSV V inside 0.6r disk)
        H, W = gray.shape[:2]
        mask = np.zeros((H, W), dtype=np.uint8)
        cv2.circle(mask, (cx, cy), int(0.6 * r), 255, -1)
        roi = mask > 0
        mean_v = float(np.mean(hsv[..., 2][roi])) if np.any(roi) else 0.0

        return {
            'center': (cx, cy),
            'radius': int(round(r)),
            'per_ratio': per_ratio,
            'std_norm': std_norm,
            'axis_ratio': axis_ratio,
            'mean_v': mean_v,
            'contour': cnt,
        }

    # ----------------------------- Classify --------------------------------
    def _classify(self, f: Dict) -> Optional[str]:
        # Paper-only mask: ignore detections inside dark/non-paper interiors (hands/phones)
        if self.paper_only and f['mean_v'] < self.paper_v_min:
            return None

        # BAD rules
        if f['per_ratio'] < self.per_ratio_bad:
            return 'BAD - incomplete'
        if f['axis_ratio'] > self.axis_ratio_max:
            return 'BAD - ellipse'
        if f['std_norm'] > self.std_norm_bad:
            return 'BAD - jagged'

        # GOOD rule (smooth & sufficiently complete)
        if f['per_ratio'] >= self.per_ratio_min_good and f['std_norm'] <= self.std_norm_good:
            return 'GOOD'

        return 'BAD - uncertain'

    # ------------------------- Deduplicate circles --------------------------
    def _dedupe(self, items: List[Dict]) -> List[Dict]:
        items = sorted(items, key=lambda x: (-x['per_ratio'], x['std_norm']))
        kept: List[Dict] = []
        for it in items:
            drop = False
            for k in kept:
                dx = it['center'][0] - k['center'][0]
                dy = it['center'][1] - k['center'][1]
                if (dx * dx + dy * dy) ** 0.5 < 0.5 * min(it['radius'], k['radius']):
                    drop = True
                    break
            if not drop:
                kept.append(it)
        return kept

    # ------------------------------ Process --------------------------------
    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        gray, edges, hsv = self.preprocess(frame)
        contours = self.detect_contours(edges)

        feats: List[Dict] = []
        for cnt in contours:
            f = self._features_from_contour(cnt, hsv, gray)
            if f is None:
                continue
            label = self._classify(f)
            if label is None:
                continue
            f['label'] = label
            feats.append(f)

        feats = self._dedupe(feats)

        # Draw
        out = frame.copy()
        for f in feats:
            color = self.col_good if f['label'].startswith('GOOD') else self.col_bad
            cv2.circle(out, f['center'], f['radius'], color, 3)
            cv2.circle(out, f['center'], 3, (0, 255, 255), -1)
            text = 'GOOD' if f['label'].startswith('GOOD') else 'BAD'
            (tw, th), _ = cv2.getTextSize(text, self.font, 0.9, 2)
            tx = f['center'][0] - tw // 2
            ty = f['center'][1] - f['radius'] - 8
            cv2.rectangle(out, (tx - 4, ty - th - 4), (tx + tw + 4, ty + 4), (255, 255, 255), -1)
            cv2.putText(out, text, (tx, ty), self.font, 0.9, (0, 120, 0) if text == 'GOOD' else (0, 0, 160), 2)

        # HUD
        self._frames += 1
        dt = time.time() - self._t0
        if dt >= 1.0:
            self.fps = self._frames / dt
            self._frames = 0
            self._t0 = time.time()

        good = sum(1 for f in feats if f['label'].startswith('GOOD'))
        bad  = sum(1 for f in feats if not f['label'].startswith('GOOD'))
        cv2.putText(out, f"FPS: {self.fps:.1f}", (10, 30), self.font, 0.7, self.col_info, 2)
        cv2.putText(out, f"Circles: {len(feats)}", (10, 60), self.font, 0.7, self.col_info, 2)
        cv2.putText(out, f"GOOD: {good}", (10, 90), self.font, 0.7, (0, 255, 0), 2)
        cv2.putText(out, f"BAD: {bad}",   (10, 120), self.font, 0.7, (0, 0, 255), 2)
        cv2.putText(out, "Method: OpenCV Traditional (Contour-first)", (10, 150), self.font, 0.6, (0, 255, 255), 1)
        cv2.putText(out, "Press q=Quit, s=Save, c=Details", (10, out.shape[0] - 16), self.font, 0.55, (200, 200, 200), 1)

        return out, feats

    # ------------------------------ Save -----------------------------------
    def save(self, frame: np.ndarray, feats: List[Dict], filename: Optional[str] = None) -> str:
        if filename is None:
            filename = f"circle_{time.strftime('%Y%m%d_%H%M%S')}.png"
        path = self.output_dir / filename
        cv2.imwrite(str(path), frame)
        print(f"Saved {path}")
        return str(path)

    # ------------------------------- Loop ----------------------------------
    def run(self) -> None:
        print("=== Circle Detection (Stable Tuned) Started ===")
        while True:
            ok, frame = self.cap.read()
            if not ok:
                print("[ERR] Failed to read camera frame.")
                break
            vis, feats = self.process_frame(frame)
            cv2.imshow('Circle Inspector', vis)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            if k == ord('s'):
                self.save(vis, feats)
            if k == ord('c'):
                print(f"\n=== Details ({len(feats)}) ===")
                for i, f in enumerate(feats, 1):
                    print(f"{i:02d}. {f['label']} center={f['center']} r={f['radius']} per_ratio={f['per_ratio']:.2f} std_norm={f['std_norm']:.3f} axis={f['axis_ratio']:.3f} V={f['mean_v']:.1f}")
        self.cap.release()
        cv2.destroyAllWindows()
        print("Stopped.")


# ================================ CLI ======================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Circle Inspector (Stable Tuned)')
    p.add_argument('--camera', type=int, default=0, help='Camera device ID (default 0)')
    p.add_argument('--config', type=str, default=None, help='Optional YAML config path')
    return p.parse_args()


def main() -> None:
    args = parse_args()
    insp = CircleInspector(camera_id=args.camera, config_path=args.config)
    insp.run()


if __name__ == '__main__':
    main()
