"""
╔══════════════════════════════════════════════════════════════╗
║   LANE DETECTION IN REAR VIEW USING HOUGH'S TRANSFORM       ║
║   --------------------------------------------------------   ║
║   Source  : Video File (road video)                         ║
║   Pipeline: Grayscale → GaussianBlur → Canny →              ║
║             ROI Mask → HoughLinesP → Lane Overlay            ║
║   Output  : Full GUI Dashboard (Pygame)                      ║
╚══════════════════════════════════════════════════════════════╝

Run:
    pip install opencv-python numpy pygame
    python lane_detection_hough.py

Controls:
    Q / ESC     → Quit
    SPACE       → Pause / Resume video
    S           → Save screenshot
    R           → Reset parameters to default
    ↑ / ↓       → Canny threshold high  (+/- 10)
    ← / →       → Canny threshold low   (+/- 10)
    T           → Toggle ROI overlay
    F           → Flip frame (mirror video)
"""

import cv2
import numpy as np
import pygame
import math
import sys
import time
import os
from dataclasses import dataclass
from typing import Optional, Tuple


# ══════════════════════════════════════════════
#  ▶  VIDEO FILE PATH  — change if needed
# ══════════════════════════════════════════════
VIDEO_PATH = "14635827_3840_2160_30fps.mp4"


# ══════════════════════════════════════════════
#  LAYOUT CONSTANTS
# ══════════════════════════════════════════════
SCREEN_W  = 1400
SCREEN_H  = 820
SIDEBAR_W = 320

FEED_W = (SCREEN_W - SIDEBAR_W - 30) // 2
FEED_H = int(FEED_W * 9 / 16)

PROC_W = (SCREEN_W - SIDEBAR_W - 40) // 4
PROC_H = int(PROC_W * 9 / 16)

FPS_TARGET = 30

# ══════════════════════════════════════════════
#  COLOUR PALETTE
# ══════════════════════════════════════════════
BG        = (10,  12,  20)
PANEL     = (18,  22,  35)
PANEL2    = (24,  30,  46)
ACCENT    = (0,   200, 255)
ACCENT2   = (255, 180, 0)
GREEN     = (50,  230, 110)
RED       = (255, 70,  60)
WHITE     = (230, 235, 245)
GREY      = (100, 110, 130)
DARK_GREY = (40,  46,  62)
LANE_L    = (0,   255, 120)
LANE_R    = (255, 200, 0)


# ══════════════════════════════════════════════
#  HOUGH PARAMETERS
# ══════════════════════════════════════════════
@dataclass
class HoughParams:
    canny_low:      int   = 50
    canny_high:     int   = 150
    blur_ksize:     int   = 5
    hough_thresh:   int   = 20
    min_line_len:   int   = 40
    max_line_gap:   int   = 20
    roi_top_ratio:  float = 0.45
    roi_side_ratio: float = 0.08

    def reset(self):
        self.__init__()


# ══════════════════════════════════════════════
#  PIPELINE RESULT
# ══════════════════════════════════════════════
@dataclass
class PipelineResult:
    original:     np.ndarray
    flipped:      np.ndarray
    gray:         np.ndarray
    blurred:      np.ndarray
    canny:        np.ndarray
    roi_mask:     np.ndarray
    masked_canny: np.ndarray
    hough_raw:    np.ndarray
    overlay:      np.ndarray
    left_line:    Optional[Tuple[int,int,int,int]] = None
    right_line:   Optional[Tuple[int,int,int,int]] = None
    num_lines:    int   = 0
    vanishing_x:  Optional[float] = None
    lane_width_px:Optional[float] = None
    left_angle:   Optional[float] = None
    right_angle:  Optional[float] = None
    fps:          float = 0.0


# ══════════════════════════════════════════════
#  PIPELINE FUNCTIONS
# ══════════════════════════════════════════════

def build_roi(h, w, params):
    top_y   = int(h * params.roi_top_ratio)
    side_in = int(w * params.roi_side_ratio)
    pts = np.array([
        [side_in,       h],
        [w - side_in,   h],
        [int(w * 0.62), top_y],
        [int(w * 0.38), top_y],
    ], dtype=np.int32)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def average_lines(lines, h, w):
    left_pts, right_pts = [], []
    if lines is None:
        return None, None
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 == x2:
            continue
        slope = (y2 - y1) / (x2 - x1)
        if abs(slope) < 0.3:
            continue
        if slope < 0:
            left_pts.extend([(x1, y1), (x2, y2)])
        else:
            right_pts.extend([(x1, y1), (x2, y2)])

    def fit_line(pts):
        if len(pts) < 2:
            return None
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        try:
            m, b = np.polyfit(xs, ys, 1)
        except np.linalg.LinAlgError:
            return None
        y_bot = h
        y_top = int(h * 0.45)
        x_bot = int((y_bot - b) / m) if abs(m) > 1e-6 else xs[0]
        x_top = int((y_top - b) / m) if abs(m) > 1e-6 else xs[0]
        return (x_bot, y_bot, x_top, y_top)

    return fit_line(left_pts), fit_line(right_pts)


def line_angle(x1, y1, x2, y2):
    dx, dy = x2 - x1, y2 - y1
    if dy == 0:
        return 90.0
    return math.degrees(math.atan2(abs(dx), abs(dy)))


def run_pipeline(frame, params, flip=False):
    original = frame.copy()
    flipped  = cv2.flip(frame, 1) if flip else frame.copy()
    h, w     = flipped.shape[:2]

    gray    = cv2.cvtColor(flipped, cv2.COLOR_BGR2GRAY)
    ksize   = params.blur_ksize | 1
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), 0)
    canny   = cv2.Canny(blurred, params.canny_low, params.canny_high)

    roi_mask = build_roi(h, w, params)
    masked   = cv2.bitwise_and(canny, roi_mask)

    lines = cv2.HoughLinesP(
        masked,
        rho=1, theta=np.pi/180,
        threshold=params.hough_thresh,
        minLineLength=params.min_line_len,
        maxLineGap=params.max_line_gap,
    )
    num_lines = len(lines) if lines is not None else 0

    hough_raw = np.zeros((h, w, 3), dtype=np.uint8)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(hough_raw, (x1,y1), (x2,y2), (0,255,0), 1)
    cv2.putText(hough_raw, f"{num_lines} lines",
                (8, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,200,0), 1)

    left_line, right_line = average_lines(lines, h, w)

    overlay = flipped.copy()
    if left_line and right_line:
        pts = np.array([
            [left_line[0],  left_line[1]],
            [right_line[0], right_line[1]],
            [right_line[2], right_line[3]],
            [left_line[2],  left_line[3]],
        ], dtype=np.int32)
        fill = overlay.copy()
        cv2.fillPoly(fill, [pts], (0, 180, 255))
        overlay = cv2.addWeighted(overlay, 0.75, fill, 0.25, 0)

    if left_line:
        cv2.line(overlay, (left_line[0],left_line[1]),
                 (left_line[2],left_line[3]), (0,255,120), 4)
    if right_line:
        cv2.line(overlay, (right_line[0],right_line[1]),
                 (right_line[2],right_line[3]), (255,200,0), 4)

    roi_pts = np.array([
        [int(w*0.08), h], [int(w*0.92), h],
        [int(w*0.62), int(h*params.roi_top_ratio)],
        [int(w*0.38), int(h*params.roi_top_ratio)],
    ], dtype=np.int32)
    cv2.polylines(overlay, [roi_pts], True, (80,80,200), 1)

    vanishing_x = lane_width_px = left_angle_v = right_angle_v = None

    if left_line and right_line:
        def seg2line(x1,y1,x2,y2):
            a=y2-y1; b=x1-x2; c=a*x1+b*y1
            return a,b,c
        a1,b1,c1 = seg2line(*left_line)
        a2,b2,c2 = seg2line(*right_line)
        det = a1*b2 - a2*b1
        if abs(det) > 1e-6:
            vx = (c1*b2 - c2*b1) / det
            vy = (a1*c2 - a2*c1) / det
            vanishing_x = vx
            cv2.circle(overlay, (int(vx),int(vy)), 10, (255,80,80), -1)
            cv2.putText(overlay, "VP", (int(vx)+12,int(vy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,80,80), 1)
        lane_width_px = abs(right_line[0] - left_line[0])

    if left_line:  left_angle_v  = line_angle(*left_line)
    if right_line: right_angle_v = line_angle(*right_line)

    return PipelineResult(
        original=original, flipped=flipped,
        gray=cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR),
        blurred=cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR),
        canny=cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR),
        roi_mask=cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR),
        masked_canny=cv2.cvtColor(masked, cv2.COLOR_GRAY2BGR),
        hough_raw=hough_raw, overlay=overlay,
        left_line=left_line, right_line=right_line,
        num_lines=num_lines, vanishing_x=vanishing_x,
        lane_width_px=lane_width_px,
        left_angle=left_angle_v, right_angle=right_angle_v,
    )


# ══════════════════════════════════════════════
#  PYGAME HELPERS
# ══════════════════════════════════════════════

def np_to_surf(arr, w, h):
    rgb     = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    return pygame.surfarray.make_surface(resized.swapaxes(0,1))


class FontCache:
    def __init__(self):
        self._c = {}
    def get(self, size, bold=False):
        k = (size, bold)
        if k not in self._c:
            self._c[k] = pygame.font.SysFont("monospace", size, bold=bold)
        return self._c[k]

FC = None


def txt(surf, text, x, y, color=WHITE, size=13, bold=False, anchor="topleft"):
    font = FC.get(size, bold)
    s = font.render(str(text), True, color)
    r = s.get_rect()
    setattr(r, anchor, (x, y))
    surf.blit(s, r)


def panel(surf, rect, color=PANEL, border=ACCENT, radius=8):
    s = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
    s.fill((*color, 255))
    pygame.draw.rect(s, border, s.get_rect(), 1, border_radius=radius)
    surf.blit(s, rect.topleft)


def label_panel(surf, rect, title, color=ACCENT):
    txt(surf, title, rect.x+8, rect.y+5, color, 11, bold=True)


def bar(surf, x, y, w, h, val, vmin, vmax, col=ACCENT, bg=DARK_GREY):
    ratio = max(0, min(1, (val-vmin)/max(vmax-vmin,1)))
    pygame.draw.rect(surf, bg,  (x, y, w, h), border_radius=3)
    pygame.draw.rect(surf, col, (x, y, int(w*ratio), h), border_radius=3)


def indicator(surf, x, y, r, on, col_on=GREEN):
    pygame.draw.circle(surf, col_on if on else DARK_GREY, (x,y), r)
    pygame.draw.circle(surf, WHITE, (x,y), r, 1)


# ══════════════════════════════════════════════
#  DASHBOARD
# ══════════════════════════════════════════════

class Dashboard:
    def __init__(self, screen):
        self.screen  = screen
        self.history = []
        self.max_hist= 80

    def _add_history(self, result):
        self.history.append({
            "fps": result.fps, "lines": result.num_lines,
            "left": result.left_line is not None,
            "right": result.right_line is not None,
        })
        if len(self.history) > self.max_hist:
            self.history.pop(0)

    def render(self, result, params, flip, show_roi,
               frame_count, paused=False, video_pos=0.0):
        self._add_history(result)
        scr = self.screen
        scr.fill(BG)

        # Header
        pygame.draw.rect(scr, PANEL2, (0, 0, SCREEN_W, 38))
        pygame.draw.line(scr, ACCENT, (0,38), (SCREEN_W,38), 1)
        title = "⏸ PAUSED  —  LANE DETECTION  (HOUGH TRANSFORM)" if paused else \
                "LANE DETECTION IN REAR VIEW  ─  HOUGH TRANSFORM PIPELINE"
        txt(scr, title, SCREEN_W//2, 10,
            ACCENT2 if paused else ACCENT, 14, bold=True, anchor="midtop")
        txt(scr, f"Frame #{frame_count:05d}  |  {video_pos:.1f}s",
            SCREEN_W-12, 10, GREY, 11, anchor="topright")

        SX = SCREEN_W - SIDEBAR_W + 5
        SW = SIDEBAR_W - 10
        TOP_Y  = 46
        PROC_Y = TOP_Y + FEED_H + 10

        # --- Main feeds ---
        raw_surf = np_to_surf(result.flipped, FEED_W, FEED_H)
        pygame.draw.rect(scr, ACCENT, (9, TOP_Y-1, FEED_W+2, FEED_H+2), 1)
        scr.blit(raw_surf, (10, TOP_Y))
        txt(scr, "● VIDEO INPUT (flipped)" if flip else "● VIDEO INPUT",
            16, TOP_Y+6, ACCENT, 11, bold=True)

        MID_X = 10 + FEED_W + 10
        ov_surf = np_to_surf(result.overlay, FEED_W, FEED_H)
        pygame.draw.rect(scr, ACCENT2, (MID_X-1, TOP_Y-1, FEED_W+2, FEED_H+2), 1)
        scr.blit(ov_surf, (MID_X, TOP_Y))
        txt(scr, "● LANE DETECTION OUTPUT", MID_X+6, TOP_Y+6, ACCENT2, 11, bold=True)

        for i, (lbl, det, col) in enumerate([
            ("LEFT",  result.left_line  is not None, LANE_L),
            ("RIGHT", result.right_line is not None, LANE_R),
        ]):
            bx = MID_X + FEED_W - 110 + i*58
            pygame.draw.rect(scr, col if det else DARK_GREY,
                             (bx, TOP_Y+6, 52, 18), border_radius=4)
            txt(scr, lbl, bx+26, TOP_Y+8, BG if det else GREY, 10,
                bold=True, anchor="midtop")

        # --- Pipeline panels ---
        proc_imgs = [
            (result.gray,         "① GRAYSCALE",   GREY),
            (result.canny,        "② CANNY EDGES", (255,120,60)),
            (result.masked_canny, "③ ROI MASKED",  (120,80,255)),
            (result.hough_raw,    "④ HOUGH LINES", GREEN),
        ]
        for i, (img, lbl, col) in enumerate(proc_imgs):
            px = 10 + i*(PROC_W+9)
            ps = np_to_surf(img, PROC_W, PROC_H)
            pygame.draw.rect(scr, col, (px-1, PROC_Y-1, PROC_W+2, PROC_H+2), 1)
            scr.blit(ps, (px, PROC_Y))
            txt(scr, lbl, px+5, PROC_Y+4, col, 10, bold=True)

        # --- Sidebar ---
        sy = TOP_Y

        # Detection stats
        r = pygame.Rect(SX, sy, SW, 155)
        panel(scr, r)
        label_panel(scr, r, "DETECTION STATS", ACCENT)
        cy = sy + 22
        for lbl, val, col in [
            ("Lines detected", f"{result.num_lines}",
             GREEN if result.num_lines>0 else RED),
            ("Left lane",  "✓ YES" if result.left_line  else "✗ NO",
             GREEN if result.left_line  else RED),
            ("Right lane", "✓ YES" if result.right_line else "✗ NO",
             GREEN if result.right_line else RED),
            ("Vanishing X", f"{result.vanishing_x:.1f}px"
             if result.vanishing_x else "—", WHITE),
            ("Lane width", f"{result.lane_width_px:.0f}px"
             if result.lane_width_px else "—", WHITE),
            ("Left angle", f"{result.left_angle:.1f}°"
             if result.left_angle else "—", LANE_L),
            ("Right angle", f"{result.right_angle:.1f}°"
             if result.right_angle else "—", LANE_R),
        ]:
            txt(scr, lbl, SX+10, cy, GREY, 11)
            txt(scr, val, SX+SW-8, cy, col, 12, bold=True, anchor="topright")
            cy += 18

        # Hough params
        sy2 = sy + 162
        r2 = pygame.Rect(SX, sy2, SW, 210)
        panel(scr, r2, border=ACCENT2)
        label_panel(scr, r2, "HOUGH PARAMETERS  (keyboard)", ACCENT2)
        cy2 = sy2 + 22
        for plbl, val, vmin, vmax, col in [
            ("Canny Low  [← →]", params.canny_low,    0,  255, ACCENT2),
            ("Canny High [↑ ↓]", params.canny_high,   0,  255, ACCENT2),
            ("Blur kernel",       params.blur_ksize,   1,   21, GREY),
            ("Hough thresh",      params.hough_thresh, 5,   80, GREY),
            ("Min line len",      params.min_line_len, 10, 100, GREY),
            ("Max line gap",      params.max_line_gap, 5,   60, GREY),
        ]:
            txt(scr, plbl, SX+10, cy2, GREY, 10)
            txt(scr, str(val), SX+SW-8, cy2, col, 11, bold=True, anchor="topright")
            cy2 += 13
            bar(scr, SX+10, cy2, SW-20, 6, val, vmin, vmax, col)
            cy2 += 14

        # Performance
        sy3 = sy2 + 218
        r3 = pygame.Rect(SX, sy3, SW, 80)
        panel(scr, r3, border=GREEN)
        label_panel(scr, r3, "PERFORMANCE", GREEN)
        txt(scr, f"FPS: {result.fps:.1f}", SX+10, sy3+22, GREEN, 20, bold=True)
        if len(self.history) > 2:
            pts = [(SX+10+int(i*(SW-20)/self.max_hist),
                    sy3+68-int(h["fps"]/60*30))
                   for i,h in enumerate(self.history)]
            if len(pts) > 1:
                pygame.draw.lines(scr, GREEN, False, pts, 1)
        txt(scr, "LINES", SX+SW//2, sy3+22, ACCENT, 11)
        if len(self.history) > 2:
            lpts = [(SX+SW//2+int(i*(SW//2-15)/self.max_hist),
                     sy3+68-int(h["lines"]/60*30))
                    for i,h in enumerate(self.history)]
            if len(lpts) > 1:
                pygame.draw.lines(scr, ACCENT, False, lpts, 1)

        # Confidence
        sy4 = sy3 + 88
        r4 = pygame.Rect(SX, sy4, SW, 60)
        panel(scr, r4)
        label_panel(scr, r4, "DETECTION CONFIDENCE", ACCENT)
        conf = ((1 if result.left_line else 0) +
                (1 if result.right_line else 0) +
                (1 if result.vanishing_x else 0)) / 3.0
        cc = GREEN if conf > 0.6 else ACCENT2 if conf > 0.3 else RED
        bar(scr, SX+10, sy4+30, SW-20, 14, conf, 0, 1, cc)
        txt(scr, f"{int(conf*100)}%", SX+SW//2, sy4+30, cc, 13,
            bold=True, anchor="midtop")

        # Controls
        sy5 = sy4 + 68
        r5 = pygame.Rect(SX, sy5, SW, 165)
        panel(scr, r5, border=GREY)
        label_panel(scr, r5, "KEYBOARD CONTROLS", GREY)
        for i, (key, desc) in enumerate([
            ("Q/ESC",  "Quit"),
            ("SPACE",  "Pause / Resume"),
            ("S",      "Screenshot"),
            ("R",      "Reset params"),
            ("↑ / ↓", "Canny high ±10"),
            ("← / →", "Canny low  ±10"),
            ("F",      "Flip frame"),
            ("T",      "Toggle ROI"),
        ]):
            ky = sy5 + 22 + i*18
            pygame.draw.rect(scr, DARK_GREY, (SX+8, ky, 46, 14), border_radius=3)
            txt(scr, key, SX+31, ky, ACCENT2, 10, bold=True, anchor="midtop")
            txt(scr, desc, SX+60, ky, GREY, 10)

        # Toggle indicators
        sy6 = sy5 + 173
        r6 = pygame.Rect(SX, sy6, SW, 44)
        panel(scr, r6, border=DARK_GREY)
        for i, (lbl, state, col) in enumerate([
            ("FLIP",   flip,   GREEN),
            ("ROI",    show_roi, GREEN),
            ("PAUSED", paused, ACCENT2),
        ]):
            ix = SX + 16 + i*90
            indicator(scr, ix, sy6+22, 6, state, col)
            txt(scr, lbl, ix+10, sy6+16, GREY, 10)

        # Footer
        fy = SCREEN_H - 22
        pygame.draw.line(scr, DARK_GREY, (0,fy), (SCREEN_W,fy), 1)
        txt(scr, "Lane Detection in Rear View using Hough's Transform  │  "
            "Grayscale → GaussianBlur → Canny → ROI Mask → HoughLinesP → Overlay",
            SCREEN_W//2, fy+3, GREY, 10, anchor="midtop")

        pygame.display.flip()


# ══════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════

def main():
    global FC

    pygame.init()
    FC = FontCache()

    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Lane Detection — Hough Transform Dashboard")
    clock = pygame.time.Clock()

    # Open video file
    if not os.path.exists(VIDEO_PATH):
        print(f"[ERROR] Video file not found: '{VIDEO_PATH}'")
        print("  Place your video in the same folder as this script.")
        print(f"  Then set:  VIDEO_PATH = '<your_filename>'")
        sys.exit(1)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {VIDEO_PATH}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps    = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[INFO] Loaded : {VIDEO_PATH}")
    print(f"[INFO] Frames : {total_frames}  |  FPS: {video_fps:.1f}")

    params      = HoughParams()
    dashboard   = Dashboard(screen)
    flip        = False
    show_roi    = True
    paused      = False
    frame_count = 0
    screenshots = 0
    t_prev      = time.time()
    fps_smooth  = 30.0
    last_frame  = None

    while True:
        # Read frame
        if not paused:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("[INFO] Video ended — looping.")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
                if not ok:
                    break
            last_frame = frame
        else:
            frame = last_frame
            if frame is None:
                clock.tick(FPS_TARGET)
                continue

        result = run_pipeline(frame, params, flip)

        t_now      = time.time()
        dt         = max(t_now - t_prev, 1e-6)
        t_prev     = t_now
        if not paused:
            fps_smooth = 0.9*fps_smooth + 0.1*(1.0/dt)
        result.fps  = fps_smooth
        frame_count += 1

        video_pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                cap.release(); pygame.quit(); sys.exit()

            if event.type == pygame.KEYDOWN:
                k = event.key
                if k in (pygame.K_q, pygame.K_ESCAPE):
                    cap.release(); pygame.quit(); sys.exit()
                elif k == pygame.K_SPACE:
                    paused = not paused
                elif k == pygame.K_r:
                    params.reset()
                elif k == pygame.K_f:
                    flip = not flip
                elif k == pygame.K_t:
                    show_roi = not show_roi
                elif k == pygame.K_UP:
                    params.canny_high = min(255, params.canny_high + 10)
                elif k == pygame.K_DOWN:
                    params.canny_high = max(params.canny_low+5, params.canny_high-10)
                elif k == pygame.K_RIGHT:
                    params.canny_low = min(params.canny_high-5, params.canny_low+10)
                elif k == pygame.K_LEFT:
                    params.canny_low = max(0, params.canny_low-10)
                elif k == pygame.K_s:
                    fname = f"screenshot_{screenshots:03d}.png"
                    pygame.image.save(screen, fname)
                    screenshots += 1
                    print(f"[Screenshot saved] {fname}")

        dashboard.render(result, params, flip, show_roi,
                         frame_count, paused, video_pos)
        clock.tick(FPS_TARGET)

    cap.release()
    pygame.quit()


if __name__ == "__main__":
    main()