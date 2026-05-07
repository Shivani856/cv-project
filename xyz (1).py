"""
╔══════════════════════════════════════════════════════════════════════╗
║   REVERSE PARKING ASSIST + VIDEO LANE DETECTION  — Merged MVP       ║
║   ─────────────────────────────────────────────────────────────────  ║
║   Left panel  : Parking simulation (Ackermann model + trajectory)   ║
║   Right panel : Video file OR live camera + Hough lane detection     ║
║   Bottom      : Full pipeline thumbnails + sidebar stats             ║
╠══════════════════════════════════════════════════════════════════════╣
║  Install:  pip install opencv-python numpy pygame                    ║
║                                                                      ║
║  ▶  Set VIDEO_PATH below to your downloaded video file              ║
║     (or leave blank / press C to use a live camera instead)         ║
╠══════════════════════════════════════════════════════════════════════╣
║  Controls                                                            ║
║   ↑ / ↓        Forward / Reverse                                     ║
║   ← / →        Steer simulation                                      ║
║   Space         Brake / Pause video                                  ║
║   0             Centre steer                                         ║
║   1-3           Steer preset 15/30/45° Left                         ║
║   4-6           Steer preset 15/30/45° Right                        ║
║   C             Toggle camera (overrides video)                      ║
║   G             Toggle trajectory guides                             ║
║   F             Flip video frame                                     ║
║   T             Toggle ROI overlay                                   ║
║   S             Save screenshot                                      ║
║   R             Reset simulation + Hough params                      ║
║   Q / ESC       Quit                                                 ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import cv2
import numpy as np
import pygame
import sys
import math
import time
import os
from dataclasses import dataclass
from typing import Optional, Tuple

# ══════════════════════════════════════════════════════
#  ▶  SET YOUR VIDEO PATH HERE
#     Leave as "" to run in synthetic/camera-only mode
# ══════════════════════════════════════════════════════
VIDEO_PATH = "14635827_3840_2160_30fps.mp4"   # ← change me

# ─────────────────────────────────────────────────────
# WINDOW / LAYOUT CONSTANTS
# ─────────────────────────────────────────────────────
WIN_W, WIN_H    = 1400, 820
SIM_W, SIM_H    = 640, 400      # left panel  – simulation
CAM_W, CAM_H    = 700, 400      # right panel – video / camera
SIDEBAR_W       = 320
PROC_W          = (WIN_W - SIDEBAR_W - 40) // 4
PROC_H          = int(PROC_W * 9 / 16)
PAD             = 10
FPS_TARGET      = 30

# Road geometry (simulation coords)
ROAD_L, ROAD_R  = 30, 610
ROAD_T, ROAD_B  = 20, 390
ROAD_W          = ROAD_R - ROAD_L
LANE_W          = ROAD_W // 3
DIV1            = ROAD_L + LANE_W
DIV2            = ROAD_L + LANE_W * 2
LANE_CX         = [ROAD_L + LANE_W * 0.5,
                   ROAD_L + LANE_W * 1.5,
                   ROAD_L + LANE_W * 2.5]

# Car dimensions
CAR_W, CAR_H    = 32, 56
MAX_STEER       = 45.0

# Physics
STEER_RATE   = 0.8
STEER_RETURN = 0.94
ACCEL        = 0.03
MAX_SPD      = 1.2
FRICTION     = 0.96
BRAKE        = 0.90
TURN_RATE    = 0.030

# ─────────────────────────────────────────────────────
# COLOUR PALETTE  (shared by both panels)
# ─────────────────────────────────────────────────────
# Simulation
C_BG            = (10,  12,  20)
C_SIDEWALK      = (180, 175, 165)
C_ROAD          = (90,  90,  85)
C_ROAD_EDGE     = (230, 220, 195)
C_DIV_DASH      = (200, 195, 180)
C_LANE_LABEL    = (160, 155, 145)
C_CAR_BODY      = (50,  110, 200)
C_CAR_ROOF      = (30,  80,  160)
C_CAR_WIN       = (180, 220, 255)
C_HEADLIGHT     = (255, 230, 80)
C_TAILLIGHT     = (220, 50,  50)
C_REVERSE_LIGHT = (240, 240, 240)
C_TRAJ_FWD      = (30,  170, 80)
C_TRAJ_REV      = (255, 160, 0)
C_TRAJ_WARN     = (220, 50,  50)
C_RAIL_L        = (30,  140, 255)
C_RAIL_R        = (30,  200, 100)
C_DIST_30       = (220, 50,  50)
C_DIST_60       = (220, 170, 20)
C_DIST_100      = (30,  180, 80)

# Dashboard / UI
BG          = C_BG
PANEL       = (18,  22,  35)
PANEL2      = (24,  30,  46)
ACCENT      = (0,   200, 255)
ACCENT2     = (255, 180, 0)
GREEN       = (50,  230, 110)
RED         = (255, 70,  60)
WHITE       = (230, 235, 245)
GREY        = (100, 110, 130)
DARK_GREY   = (40,  46,  62)
LANE_L_COL  = (0,   255, 120)
LANE_R_COL  = (255, 200, 0)

C_TEXT      = WHITE
C_TEXT_DIM  = GREY
C_HUD_BG    = PANEL


# ══════════════════════════════════════════════════════
#  HOUGH PARAMETERS  (video/camera pipeline)
# ══════════════════════════════════════════════════════
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


# ══════════════════════════════════════════════════════
#  PIPELINE RESULT  (video/camera pipeline)
# ══════════════════════════════════════════════════════
@dataclass
class PipelineResult:
    original:      np.ndarray
    flipped:       np.ndarray
    gray:          np.ndarray
    blurred:       np.ndarray
    canny:         np.ndarray
    roi_mask:      np.ndarray
    masked_canny:  np.ndarray
    hough_raw:     np.ndarray
    overlay:       np.ndarray
    left_line:     Optional[Tuple[int,int,int,int]] = None
    right_line:    Optional[Tuple[int,int,int,int]] = None
    num_lines:     int   = 0
    vanishing_x:   Optional[float] = None
    lane_width_px: Optional[float] = None
    left_angle:    Optional[float] = None
    right_angle:   Optional[float] = None
    fps:           float = 0.0
    steer_deg:     float = 0.0    # steer suggestion in degrees


# ══════════════════════════════════════════════════════
#  VIDEO / CAMERA PIPELINE FUNCTIONS
# ══════════════════════════════════════════════════════

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

    # ROI outline
    roi_pts = np.array([
        [int(w*params.roi_side_ratio), h],
        [int(w*(1-params.roi_side_ratio)), h],
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

    # Steer suggestion from video lanes
    steer_deg = 0.0
    mid_x = w / 2
    if left_line and right_line:
        lane_cx = (left_line[0] + right_line[0]) / 2
        offset  = (lane_cx - mid_x) / max(mid_x, 1)
        steer_deg = offset * MAX_STEER
    elif left_line:
        steer_deg = min(15.0, (mid_x - left_line[0]) / max(mid_x,1) * MAX_STEER)
    elif right_line:
        steer_deg = max(-15.0, (right_line[0] - mid_x) / max(mid_x,1) * -MAX_STEER)

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
        steer_deg=steer_deg,
    )


# ══════════════════════════════════════════════════════
#  SIMULATION CLASSES
# ══════════════════════════════════════════════════════

class Car:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x      = float(LANE_CX[1])
        self.y      = float((ROAD_T + ROAD_B) // 2)
        self.angle  = 0.0
        self.speed  = 0.0
        self.steer  = 0.0

    def update(self, keys, steer_target):
        if keys[pygame.K_UP]:
            self.speed = min(self.speed + ACCEL, MAX_SPD)
        elif keys[pygame.K_DOWN]:
            self.speed = max(self.speed - ACCEL, -MAX_SPD)
        elif keys[pygame.K_SPACE]:
            self.speed *= BRAKE
        else:
            self.speed *= FRICTION

        if steer_target is not None:
            diff = steer_target - self.steer
            self.steer += diff * 0.08
            if abs(diff) < 0.1:
                self.steer = steer_target
        else:
            if keys[pygame.K_LEFT]:
                self.steer = max(self.steer - STEER_RATE, -MAX_STEER)
            elif keys[pygame.K_RIGHT]:
                self.steer = min(self.steer + STEER_RATE,  MAX_STEER)
            else:
                self.steer *= STEER_RETURN

        rad = math.radians(self.angle)
        self.x     += math.sin(rad) * self.speed
        self.y     -= math.cos(rad) * self.speed
        self.angle += self.steer * self.speed * TURN_RATE
        self.x = max(ROAD_L + CAR_W/2 + 2, min(ROAD_R - CAR_W/2 - 2, self.x))
        self.y = max(ROAD_T + CAR_H/2 + 2, min(ROAD_B - CAR_H/2 - 2, self.y))

    def get_corners(self):
        rad = math.radians(self.angle)
        cos, sin = math.cos(rad), math.sin(rad)
        corners = [(-CAR_W/2, -CAR_H/2), (CAR_W/2, -CAR_H/2),
                   (CAR_W/2,  CAR_H/2),  (-CAR_W/2, CAR_H/2)]
        return [(self.x + lx*cos - ly*sin, self.y + lx*sin + ly*cos)
                for lx, ly in corners]

    def rear_point(self):
        rad = math.radians(self.angle)
        return (self.x - math.sin(rad)*CAR_H/2,
                self.y + math.cos(rad)*CAR_H/2)

    def current_lane(self):
        if self.x < DIV1: return 1
        if self.x < DIV2: return 2
        return 3

    def lane_bounds(self):
        l = self.current_lane()
        return ROAD_L + (l-1)*LANE_W, ROAD_L + l*LANE_W


def get_trajectory(car, steps=50, step_dist=8):
    pts = []
    steer_rad = math.radians(car.steer)
    wb = CAR_H * 0.65
    px, py, pang = car.x, car.y, math.radians(car.angle)
    for _ in range(steps + 1):
        pts.append((px, py, pang))
        if abs(car.steer) < 0.3:
            px -= math.sin(pang) * step_dist
            py += math.cos(pang) * step_dist
        else:
            R    = wb / math.tan(abs(steer_rad))
            sign = 1 if car.steer > 0 else -1
            dang = (step_dist / R) * sign
            pang -= dang
            px   -= math.sin(pang) * step_dist
            py   += math.cos(pang) * step_dist
    return pts


# ══════════════════════════════════════════════════════
#  PYGAME HELPERS
# ══════════════════════════════════════════════════════

def np_to_surf(arr, w, h):
    rgb     = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_LINEAR)
    return pygame.surfarray.make_surface(resized.swapaxes(0, 1))


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


def bar_widget(surf, x, y, w, h, val, vmin, vmax, col=ACCENT, bg=DARK_GREY):
    ratio = max(0, min(1, (val-vmin)/max(vmax-vmin, 1)))
    pygame.draw.rect(surf, bg,  (x, y, w, h), border_radius=3)
    pygame.draw.rect(surf, col, (x, y, int(w*ratio), h), border_radius=3)


def indicator(surf, x, y, r, on, col_on=GREEN):
    pygame.draw.circle(surf, col_on if on else DARK_GREY, (x, y), r)
    pygame.draw.circle(surf, WHITE, (x, y), r, 1)


# ── Simulation drawing helpers ──────────────────────

def lerp_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))


def draw_dashed_line(surf, color, p1, p2, dash=18, gap=12, width=2):
    dx, dy = p2[0]-p1[0], p2[1]-p1[1]
    length = math.hypot(dx, dy)
    if length == 0: return
    ux, uy = dx/length, dy/length
    pos = 0; drawing = True
    while pos < length:
        seg = min(dash if drawing else gap, length - pos)
        if drawing:
            sx, sy = p1[0]+ux*pos, p1[1]+uy*pos
            ex, ey = p1[0]+ux*(pos+seg), p1[1]+uy*(pos+seg)
            pygame.draw.line(surf, color, (int(sx),int(sy)), (int(ex),int(ey)), width)
        pos += seg; drawing = not drawing


def draw_car(surf, car):
    cx, cy, ang = car.x, car.y, car.angle
    rad = math.radians(ang)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    def rot(lx, ly):
        return (cx + lx*cos_a - ly*sin_a, cy + lx*sin_a + ly*cos_a)
    hw, hh = CAR_W/2, CAR_H/2
    # Shadow
    pts_s = [rot(hw+2,hh+2), rot(-hw-2,hh+2), rot(-hw-2,-hh-2), rot(hw+2,-hh-2)]
    shadow_surf = pygame.Surface((SIM_W, SIM_H), pygame.SRCALPHA)
    pygame.draw.polygon(shadow_surf, (0,0,0,40), [(int(x),int(y)) for x,y in pts_s])
    surf.blit(shadow_surf, (0,0))
    # Body
    pts = [rot(hw,-hh), rot(-hw,-hh), rot(-hw,hh), rot(hw,hh)]
    pygame.draw.polygon(surf, C_CAR_BODY, [(int(x),int(y)) for x,y in pts])
    # Roof
    roof_pts = [rot(hw-5,-hh+13), rot(-hw+5,-hh+13), rot(-hw+5,hh-13), rot(hw-5,hh-13)]
    pygame.draw.polygon(surf, C_CAR_ROOF, [(int(x),int(y)) for x,y in roof_pts])
    # Windshields
    fw_pts = [rot(hw-6,-hh+4), rot(-hw+6,-hh+4), rot(-hw+6,-hh+14), rot(hw-6,-hh+14)]
    pygame.draw.polygon(surf, C_CAR_WIN, [(int(x),int(y)) for x,y in fw_pts])
    rw_pts = [rot(hw-6,hh-14), rot(-hw+6,hh-14), rot(-hw+6,hh-4), rot(hw-6,hh-4)]
    pygame.draw.polygon(surf, C_CAR_WIN, [(int(x),int(y)) for x,y in rw_pts])
    # Lights
    is_reversing = car.speed < -0.05
    hl_col = C_HEADLIGHT
    tl_col = C_REVERSE_LIGHT if is_reversing else C_TAILLIGHT
    for side in [-1, 1]:
        hp = rot(side*(hw-4), -hh+3)
        tp = rot(side*(hw-4),  hh-3)
        pygame.draw.circle(surf, hl_col, (int(hp[0]),int(hp[1])), 3)
        pygame.draw.circle(surf, tl_col, (int(tp[0]),int(tp[1])), 3)


def draw_simulation(surf, car, warning, world_y, show_guides, result: PipelineResult):
    surf.fill((70, 70, 65))
    # Sidewalk
    pygame.draw.rect(surf, C_SIDEWALK, (0, 0, SIM_W, SIM_H))
    # Road
    pygame.draw.rect(surf, C_ROAD, (ROAD_L, ROAD_T, ROAD_W, ROAD_B - ROAD_T))
    # Road edges
    pygame.draw.line(surf, C_ROAD_EDGE, (ROAD_L, ROAD_T), (ROAD_L, ROAD_B), 3)
    pygame.draw.line(surf, C_ROAD_EDGE, (ROAD_R, ROAD_T), (ROAD_R, ROAD_B), 3)
    # Lane dividers
    dash_offset = int(world_y) % 30
    for x in [DIV1, DIV2]:
        y = ROAD_T - dash_offset
        while y < ROAD_B:
            y1 = max(ROAD_T, y)
            y2 = min(ROAD_B, y + 18)
            if y1 < y2:
                pygame.draw.line(surf, C_DIV_DASH, (x, y1), (x, y2), 1)
            y += 30

    # Trajectory guides
    if show_guides:
        traj = get_trajectory(car, steps=55, step_dist=7)
        for i in range(1, len(traj)):
            px, py, _ = traj[i-1]
            cx2, cy2, _ = traj[i]
            t = i / len(traj)
            if car.speed >= 0:
                col = lerp_color(C_TRAJ_FWD, C_TRAJ_WARN, t)
            else:
                col = lerp_color(C_TRAJ_REV, C_TRAJ_WARN, t)
            if ROAD_T < py < ROAD_B and ROAD_L < px < ROAD_R:
                pygame.draw.line(surf, col, (int(px),int(py)), (int(cx2),int(cy2)), 2)

        # Rail lines from rear axle
        rear = car.rear_point()
        for dist, col in [(30, C_DIST_30), (60, C_DIST_60), (100, C_DIST_100)]:
            rad = math.radians(car.angle)
            ep = (rear[0] + math.sin(rad)*dist, rear[1] - math.cos(rad)*dist)
            pygame.draw.line(surf, col, (int(rear[0]),int(rear[1])), (int(ep[0]),int(ep[1])), 1)

        # Steer suggestion arrow from video pipeline
        if abs(result.steer_deg) > 2:
            arr_x = int(car.x)
            arr_y = int(car.y)
            sug_ang = car.angle + result.steer_deg
            rad2 = math.radians(sug_ang)
            tip = (arr_x + math.sin(rad2)*22, arr_y - math.cos(rad2)*22)
            left_a  = (arr_x + math.sin(rad2-2.4)*11, arr_y - math.cos(rad2-2.4)*11)
            right_a = (arr_x + math.sin(rad2+2.4)*11, arr_y - math.cos(rad2+2.4)*11)
            pygame.draw.polygon(surf, ACCENT2, [
                (int(tip[0]),int(tip[1])),
                (int(left_a[0]),int(left_a[1])),
                (int(right_a[0]),int(right_a[1]))
            ])

    draw_car(surf, car)

    # Warning flash
    if warning:
        flash = pygame.Surface((SIM_W, SIM_H), pygame.SRCALPHA)
        flash.fill((220, 50, 50, 30))
        surf.blit(flash, (0, 0))
        font_w = pygame.font.SysFont("Arial", 14, bold=True)
        w_s = font_w.render("⚠ WARNING", True, (220, 50, 50))
        surf.blit(w_s, (SIM_W//2 - w_s.get_width()//2, 8))

    # Lane label
    font_s = pygame.font.SysFont("Arial", 11)
    surf.blit(font_s.render(f"Lane {car.current_lane()}", True, C_LANE_LABEL), (5, 5))


# ══════════════════════════════════════════════════════
#  DASHBOARD SIDEBAR
# ══════════════════════════════════════════════════════

class Dashboard:
    def __init__(self, screen):
        self.screen  = screen
        self.history = []
        self.max_hist = 80

    def _add_history(self, result):
        self.history.append({"fps": result.fps, "lines": result.num_lines,
                              "left": result.left_line is not None,
                              "right": result.right_line is not None})
        if len(self.history) > self.max_hist:
            self.history.pop(0)

    def render(self, result: PipelineResult, params: HoughParams,
               car: Car, flip, show_roi, frame_count, paused, video_pos,
               cam_active, show_guides, steer_target, warning):
        self._add_history(result)
        scr = self.screen

        # ── Header ──────────────────────────────────────
        pygame.draw.rect(scr, PANEL2, (0, 0, WIN_W, 38))
        pygame.draw.line(scr, ACCENT, (0,38), (WIN_W,38), 1)
        src_label = "CAMERA" if cam_active else ("VIDEO" if result.num_lines >= 0 else "SYNTH")
        title = f"⏸ PAUSED  —  REVERSE PARK + LANE DETECT  [{src_label}]" if paused else \
                f"REVERSE PARKING ASSIST  ─  HOUGH LANE DETECTION  [{src_label}]"
        txt(scr, title, WIN_W//2, 10, ACCENT2 if paused else ACCENT, 13, bold=True, anchor="midtop")
        txt(scr, f"Frame #{frame_count:05d}  |  {video_pos:.1f}s",
            WIN_W-12, 10, GREY, 11, anchor="topright")

        SX = WIN_W - SIDEBAR_W + 5
        SW = SIDEBAR_W - 10

        # ── Simulation panel (left) ──────────────────────
        sim_surf = pygame.Surface((SIM_W, SIM_H))
        draw_simulation(sim_surf, car, warning, 0.0, show_guides, result)
        scr.blit(sim_surf, (PAD, 46))
        pygame.draw.rect(scr, ACCENT, (PAD-1, 45, SIM_W+2, SIM_H+2), 1)
        txt(scr, "● SIMULATION — PARKING ASSIST", PAD+6, 50, ACCENT, 11, bold=True)

        # ── Video/camera panel (right) ───────────────────
        MID_X = PAD + SIM_W + PAD
        ov_surf = np_to_surf(result.overlay, CAM_W, CAM_H)
        scr.blit(ov_surf, (MID_X, 46))
        pygame.draw.rect(scr, ACCENT2, (MID_X-1, 45, CAM_W+2, CAM_H+2), 1)
        txt(scr, "● LANE DETECTION OUTPUT", MID_X+6, 50, ACCENT2, 11, bold=True)

        for i, (lbl, det, col) in enumerate([
            ("LEFT",  result.left_line  is not None, LANE_L_COL),
            ("RIGHT", result.right_line is not None, LANE_R_COL),
        ]):
            bx = MID_X + CAM_W - 120 + i*62
            pygame.draw.rect(scr, col if det else DARK_GREY,
                             (bx, 50, 56, 18), border_radius=4)
            txt(scr, lbl, bx+28, 52, BG if det else GREY, 10, bold=True, anchor="midtop")

        # ── Pipeline thumbnails ──────────────────────────
        PROC_Y = 46 + SIM_H + PAD
        proc_imgs = [
            (result.gray,         "① GRAYSCALE",   GREY),
            (result.canny,        "② CANNY EDGES", (255,120,60)),
            (result.masked_canny, "③ ROI MASKED",  (120,80,255)),
            (result.hough_raw,    "④ HOUGH LINES", GREEN),
        ]
        for i, (img, lbl, col) in enumerate(proc_imgs):
            px = PAD + i*(PROC_W+9)
            ps = np_to_surf(img, PROC_W, PROC_H)
            pygame.draw.rect(scr, col, (px-1, PROC_Y-1, PROC_W+2, PROC_H+2), 1)
            scr.blit(ps, (px, PROC_Y))
            txt(scr, lbl, px+5, PROC_Y+4, col, 10, bold=True)

        # ── Sidebar ─────────────────────────────────────
        sy = 46

        # Detection stats
        r = pygame.Rect(SX, sy, SW, 155)
        panel(scr, r)
        label_panel(scr, r, "DETECTION STATS", ACCENT)
        cy = sy + 22
        for lbl, val, col in [
            ("Lines detected", f"{result.num_lines}",
             GREEN if result.num_lines > 0 else RED),
            ("Left lane",  "✓ YES" if result.left_line  else "✗ NO",
             GREEN if result.left_line  else RED),
            ("Right lane", "✓ YES" if result.right_line else "✗ NO",
             GREEN if result.right_line else RED),
            ("Vanishing X", f"{result.vanishing_x:.1f}px"
             if result.vanishing_x else "—", WHITE),
            ("Lane width", f"{result.lane_width_px:.0f}px"
             if result.lane_width_px else "—", WHITE),
            ("Steer suggest", f"{result.steer_deg:+.1f}°", ACCENT2),
        ]:
            txt(scr, lbl, SX+10, cy, GREY, 11)
            txt(scr, val, SX+SW-8, cy, col, 12, bold=True, anchor="topright")
            cy += 18

        # Sim stats
        sy_sim = sy + 163
        r_sim = pygame.Rect(SX, sy_sim, SW, 100)
        panel(scr, r_sim, border=ACCENT2)
        label_panel(scr, r_sim, "SIMULATION STATS", ACCENT2)
        cy_s = sy_sim + 22
        for lbl, val, col in [
            ("Speed",   f"{abs(car.speed)*50:.1f} km/h", GREEN),
            ("Steer",   f"{car.steer:+.1f}°",           ACCENT2),
            ("Lane",    f"{car.current_lane()}",         WHITE),
            ("Warning", "YES" if warning else "NO",      RED if warning else GREEN),
        ]:
            txt(scr, lbl, SX+10, cy_s, GREY, 11)
            txt(scr, val, SX+SW-8, cy_s, col, 12, bold=True, anchor="topright")
            cy_s += 18

        # Hough params
        sy2 = sy_sim + 108
        r2 = pygame.Rect(SX, sy2, SW, 165)
        panel(scr, r2, border=ACCENT2)
        label_panel(scr, r2, "HOUGH PARAMS  (↑↓←→ to adjust)", ACCENT2)
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
            bar_widget(scr, SX+10, cy2, SW-20, 6, val, vmin, vmax, col)
            cy2 += 14

        # Performance
        sy3 = sy2 + 173
        r3 = pygame.Rect(SX, sy3, SW, 80)
        panel(scr, r3, border=GREEN)
        label_panel(scr, r3, "PERFORMANCE", GREEN)
        txt(scr, f"FPS: {result.fps:.1f}", SX+10, sy3+22, GREEN, 20, bold=True)
        if len(self.history) > 2:
            h_pts = [(SX+10+int(i*(SW-20)/self.max_hist),
                      sy3+68-int(h_["fps"]/60*30))
                     for i, h_ in enumerate(self.history)]
            if len(h_pts) > 1:
                pygame.draw.lines(scr, GREEN, False, h_pts, 1)

        # Confidence
        sy4 = sy3 + 88
        r4 = pygame.Rect(SX, sy4, SW, 60)
        panel(scr, r4)
        label_panel(scr, r4, "DETECTION CONFIDENCE", ACCENT)
        conf = ((1 if result.left_line else 0) +
                (1 if result.right_line else 0) +
                (1 if result.vanishing_x else 0)) / 3.0
        cc = GREEN if conf > 0.6 else ACCENT2 if conf > 0.3 else RED
        bar_widget(scr, SX+10, sy4+30, SW-20, 14, conf, 0, 1, cc)
        txt(scr, f"{int(conf*100)}%", SX+SW//2, sy4+30, cc, 13, bold=True, anchor="midtop")

        # Controls
        sy5 = sy4 + 68
        r5 = pygame.Rect(SX, sy5, SW, 195)
        panel(scr, r5, border=GREY)
        label_panel(scr, r5, "KEYBOARD CONTROLS", GREY)
        for i, (key, desc) in enumerate([
            ("Q/ESC",  "Quit"),
            ("SPACE",  "Brake / Pause video"),
            ("↑↓",    "Drive forward/back"),
            ("←→",    "Steer / Canny low"),
            ("↑↓ vid","Canny high ±10"),
            ("0",      "Centre steer"),
            ("1-3",    "Preset steer L"),
            ("4-6",    "Preset steer R"),
            ("C",      "Camera toggle"),
            ("F/T/S/G","Flip/ROI/Shot/Guides"),
            ("R",      "Reset all"),
        ]):
            ky = sy5 + 22 + i*16
            pygame.draw.rect(scr, DARK_GREY, (SX+8, ky, 46, 13), border_radius=3)
            txt(scr, key, SX+31, ky, ACCENT2, 9, bold=True, anchor="midtop")
            txt(scr, desc, SX+60, ky, GREY, 9)

        # Toggle indicators
        sy6 = sy5 + 203
        r6 = pygame.Rect(SX, sy6, SW, 44)
        panel(scr, r6, border=DARK_GREY)
        for i, (lbl, state, col) in enumerate([
            ("FLIP",   flip,       GREEN),
            ("ROI",    show_roi,   GREEN),
            ("CAM",    cam_active, ACCENT),
            ("GUIDES", show_guides,ACCENT2),
        ]):
            ix = SX + 10 + i*70
            indicator(scr, ix, sy6+22, 6, state, col)
            txt(scr, lbl, ix+10, sy6+16, GREY, 9)

        # Footer
        fy = WIN_H - 20
        pygame.draw.line(scr, DARK_GREY, (0, fy), (WIN_W, fy), 1)
        txt(scr, "Reverse Parking Assist  +  Lane Detection in Rear View using Hough Transform",
            WIN_W//2, fy+2, GREY, 10, anchor="midtop")

        pygame.display.flip()


# ══════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════

def main():
    global FC

    pygame.init()
    FC = FontCache()
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    pygame.display.set_caption("Reverse Parking Assist + Hough Lane Detection")
    clock = pygame.time.Clock()

    # ── Video source ─────────────────────────────────
    use_video = VIDEO_PATH and os.path.exists(VIDEO_PATH)
    cap = None
    cam_active = False
    total_frames = 0
    video_fps_src = 30.0

    if use_video:
        cap = cv2.VideoCapture(VIDEO_PATH)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {VIDEO_PATH}")
            use_video = False
            cap = None
        else:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_fps_src = cap.get(cv2.CAP_PROP_FPS) or 30.0
            print(f"[INFO] Video loaded: {VIDEO_PATH}  ({total_frames} frames @ {video_fps_src:.1f} fps)")
    else:
        if VIDEO_PATH:
            print(f"[WARN] Video not found: '{VIDEO_PATH}'  — running in synthetic mode.")
        else:
            print("[INFO] No VIDEO_PATH set — running in synthetic mode (press C for camera).")

    def try_open_camera():
        nonlocal cap, cam_active, use_video
        for idx in [0, 1, 2]:
            c = cv2.VideoCapture(idx)
            if c.isOpened():
                c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                if cap: cap.release()
                cap = c
                cam_active = True
                use_video  = False
                print(f"✅ Camera opened at index {idx}")
                return
        print("❌ No camera found")

    def close_camera():
        nonlocal cap, cam_active
        if cap and cam_active:
            cap.release()
            cap = None
        cam_active = False

    # ── Simulation state ─────────────────────────────
    car          = Car()
    steer_target = None
    show_guides  = True
    warning      = False

    # ── Pipeline / dashboard state ───────────────────
    params     = HoughParams()
    dashboard  = Dashboard(screen)
    flip       = False
    show_roi   = True
    paused     = False
    frame_count = 0
    screenshots = 0
    t_prev      = time.time()
    fps_smooth  = 30.0
    last_frame  = None
    video_pos   = 0.0

    # Blank pipeline result (synthetic frame fallback)
    def make_synthetic_frame(car_ref):
        syn = np.zeros((270, 480, 3), dtype=np.uint8)
        syn[:] = (50, 50, 48)
        vp_y = int(270 * 0.4)
        steer_off = int((car_ref.steer / MAX_STEER) * 480 * 0.15)
        for li in range(4):
            frac = li / 3
            xn = int(480*0.02 + frac*480*0.96 + steer_off*0.5)
            xf = int(480*0.30 + frac*480*0.40 + steer_off*0.1)
            is_edge = (li == 0 or li == 3)
            color = (210, 205, 190) if is_edge else (140, 135, 125)
            thick = 3 if is_edge else 1
            cv2.line(syn, (xn, 270), (xf, vp_y), color, thick)
        return syn

    result = run_pipeline(make_synthetic_frame(car), params, flip)

    # ── Main loop ─────────────────────────────────────
    running = True
    while running:
        frame_count += 1
        screen.fill(BG)

        # Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                k = event.key
                # Quit
                if k in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                # Pause (video) / Brake (sim) both on SPACE — handled in physics too
                elif k == pygame.K_SPACE:
                    if use_video or cam_active:
                        paused = not paused
                # Screenshot
                elif k == pygame.K_s:
                    fname = f"screenshot_{screenshots:03d}.png"
                    pygame.image.save(screen, fname)
                    screenshots += 1
                    print(f"[Screenshot] {fname}")
                # Reset sim + params
                elif k == pygame.K_r:
                    car.reset(); steer_target = None; params.reset()
                # Flip video
                elif k == pygame.K_f:
                    flip = not flip
                # ROI toggle
                elif k == pygame.K_t:
                    show_roi = not show_roi
                # Guides toggle
                elif k == pygame.K_g:
                    show_guides = not show_guides
                # Camera toggle
                elif k == pygame.K_c:
                    if cam_active:
                        close_camera()
                    else:
                        try_open_camera()
                # Steer presets
                elif k == pygame.K_0:   steer_target =   0.0
                elif k == pygame.K_1:   steer_target = -15.0
                elif k == pygame.K_2:   steer_target = -30.0
                elif k == pygame.K_3:   steer_target = -45.0
                elif k == pygame.K_4:   steer_target =  15.0
                elif k == pygame.K_5:   steer_target =  30.0
                elif k == pygame.K_6:   steer_target =  45.0
                # Canny adjustments
                elif k == pygame.K_UP:
                    params.canny_high = min(255, params.canny_high + 10)
                elif k == pygame.K_DOWN:
                    params.canny_high = max(params.canny_low+5, params.canny_high-10)
                elif k == pygame.K_RIGHT:
                    params.canny_low = min(params.canny_high-5, params.canny_low+10)
                elif k == pygame.K_LEFT:
                    params.canny_low = max(0, params.canny_low-10)
                # Clear steer preset on manual steer keys
                # (arrow keys also adjust Canny — no conflict, handled above)

        # ── Physics update ─────────────────────────────
        keys = pygame.key.get_pressed()
        car.update(keys, steer_target)

        # ── Video / camera frame ───────────────────────
        if not paused:
            if use_video and cap:
                ok, frame = cap.read()
                if not ok or frame is None:
                    print("[INFO] Video ended — looping.")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ok, frame = cap.read()
                    if not ok: frame = None
                last_frame = frame
            elif cam_active and cap and frame_count % 2 == 0:
                ok, frame = cap.read()
                if ok:
                    last_frame = frame
                else:
                    close_camera()

        frame_to_process = last_frame if last_frame is not None else make_synthetic_frame(car)

        # ── Run Hough pipeline ─────────────────────────
        if frame_count % 2 == 0 or result is None:
            result = run_pipeline(frame_to_process, params, flip)

        # FPS
        t_now = time.time()
        dt = max(t_now - t_prev, 1e-6)
        t_prev = t_now
        if not paused:
            fps_smooth = 0.9*fps_smooth + 0.1*(1.0/dt)
        result.fps = fps_smooth

        if use_video and cap:
            video_pos = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # ── Warning check ──────────────────────────────
        lb, rb = car.lane_bounds()
        lane   = car.current_lane()
        corners = car.get_corners()
        v = any(cx < ROAD_L or cx > ROAD_R for cx, cy in corners)
        if car.x - CAR_W/2 - 3 < lb and lane > 1: v = True
        if car.x + CAR_W/2 + 3 > rb and lane < 3: v = True
        if abs(result.steer_deg) > 32:             v = True
        warning = v

        # ── Render ─────────────────────────────────────
        dashboard.render(
            result, params, car, flip, show_roi,
            frame_count, paused, video_pos,
            cam_active, show_guides, steer_target, warning,
        )

        clock.tick(FPS_TARGET)

    # Cleanup
    if cap:
        cap.release()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()