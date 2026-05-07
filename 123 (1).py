"""
╔══════════════════════════════════════════════════════════════════╗
║     REVERSE PARKING ASSIST — Python MVP                         ║
║     Hough Transform + Real Camera + Simulation + Trajectory     ║
╚══════════════════════════════════════════════════════════════════╝

Libraries:
  pip install opencv-python numpy pygame

Controls:
  Arrow Up/Down   → Forward / Reverse
  Arrow Left/Right→ Steer (±45°)
  Space           → Brake
  1 / 2 / 3       → Steer preset  15° / 30° / 45° Left
  4 / 5 / 6       → Steer preset  15° / 30° / 45° Right
  0               → Centre steer
  C               → Toggle camera on/off
  G               → Toggle trajectory guides
  R               → Reset simulation
  Q / Esc         → Quit
"""

import cv2
import numpy as np
import pygame
import sys
import math
import time

# ─────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────
# Window layout
WIN_W, WIN_H    = 1280, 760
SIM_W, SIM_H    = 620, 420     # left panel: simulation
CAM_W, CAM_H    = 620, 420     # right panel: camera + Hough overlay
PIPE_W, PIPE_H  = 300, 180     # each pipeline stage thumbnail
PAD             = 10

# Road geometry (simulation canvas coords)
ROAD_L, ROAD_R  = 30, 590
ROAD_T, ROAD_B  = 20, 400
ROAD_W          = ROAD_R - ROAD_L        # 560
LANE_W          = ROAD_W // 3            # ~186
DIV1            = ROAD_L + LANE_W
DIV2            = ROAD_L + LANE_W * 2
LANE_CX         = [ROAD_L + LANE_W * 0.5,
                   ROAD_L + LANE_W * 1.5,
                   ROAD_L + LANE_W * 2.5]

# Car
CAR_W, CAR_H    = 32, 56
MAX_STEER       = 45.0

# Physics — slow & realistic
STEER_RATE      = 0.8     # deg/frame
STEER_RETURN    = 0.94
ACCEL           = 0.03
MAX_SPD         = 1.2
FRICTION        = 0.96
BRAKE           = 0.90
TURN_RATE       = 0.030   # angle change per steer-unit per speed-unit

# Colors (R, G, B)
C_BG            = (40,  40,  38)
C_SIDEWALK      = (180, 175, 165)
C_ROAD          = (90,  90,  85)
C_LANE_LIGHT    = (100, 100, 95)
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
C_HOUGH_LINE    = (0,   255, 120)
C_LEFT_LANE     = (30,  140, 255)
C_RIGHT_LANE    = (255, 80,  80)
C_CENTRE_LANE   = (255, 220, 0)
C_WARNING_FLASH = (220, 50,  50)
C_PANEL_BG      = (22,  22,  20)
C_TEXT          = (200, 195, 185)
C_TEXT_DIM      = (120, 115, 110)
C_HUD_BG        = (30,  30,  28)

# ─────────────────────────────────────────────────────────────────
# SIMULATION STATE
# ─────────────────────────────────────────────────────────────────
class Car:
    def __init__(self):
        self.reset()

    def reset(self):
        self.x      = float(LANE_CX[1])
        self.y      = float((ROAD_T + ROAD_B) // 2)
        self.angle  = 0.0   # degrees, 0 = up
        self.speed  = 0.0
        self.steer  = 0.0

    def update(self, keys, steer_target):
        # Speed
        if keys[pygame.K_UP]:
            self.speed = min(self.speed + ACCEL, MAX_SPD)
        elif keys[pygame.K_DOWN]:
            self.speed = max(self.speed - ACCEL, -MAX_SPD)
        elif keys[pygame.K_SPACE]:
            self.speed *= BRAKE
        else:
            self.speed *= FRICTION

        # Steer
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

        # Physics
        rad = math.radians(self.angle)
        self.x     += math.sin(rad) * self.speed
        self.y     -= math.cos(rad) * self.speed
        self.angle += self.steer * self.speed * TURN_RATE

        # Clamp inside road
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


# ─────────────────────────────────────────────────────────────────
# TRAJECTORY PREDICTION (Ackermann Bicycle Model)
# ─────────────────────────────────────────────────────────────────
def get_trajectory(car, steps=50, step_dist=8):
    pts = []
    steer_rad = math.radians(car.steer)
    wb = CAR_H * 0.65   # wheelbase approximation

    px, py, pang = car.x, car.y, math.radians(car.angle)

    for _ in range(steps + 1):
        pts.append((px, py, pang))
        if abs(car.steer) < 0.3:
            # Straight
            px -= math.sin(pang) * step_dist
            py += math.cos(pang) * step_dist
        else:
            # Ackermann arc: R = wheelbase / tan(steer)
            R    = wb / math.tan(abs(steer_rad))
            sign = 1 if car.steer > 0 else -1
            dang = (step_dist / R) * sign
            pang -= dang
            px   -= math.sin(pang) * step_dist
            py   += math.cos(pang) * step_dist
    return pts


# ─────────────────────────────────────────────────────────────────
# HOUGH TRANSFORM PIPELINE (Pure NumPy — cv2 shortcuts)
# ─────────────────────────────────────────────────────────────────
class HoughPipeline:
    """
    Runs the full CV pipeline on a camera frame (or synthetic image):
      1. Grayscale conversion
      2. Gaussian blur
      3. Canny edge detection (Sobel + NMS + double threshold)
      4. ROI trapezoid mask
      5. Hough Transform accumulator
      6. Peak extraction + NMS
      7. Left/Right lane classification
      8. Steer suggestion
    Returns debug images for each stage + detected lane data.
    """

    def __init__(self, w, h):
        self.w, self.h = w, h
        # Pre-compute Gaussian kernel (5x5, sigma=1.4)
        k = np.array([1,4,6,4,1], dtype=np.float32)
        self.gauss_k = np.outer(k, k) / 256.0
        # Sobel kernels
        self.sobel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
        self.sobel_y = np.array([[-1,-2,-1],[0,0,0],[1,2,1]],  dtype=np.float32)

        # Hough params
        self.theta_n   = 180
        self.thetas    = np.linspace(0, np.pi, self.theta_n, endpoint=False)
        self.cos_t     = np.cos(self.thetas)
        self.sin_t     = np.sin(self.thetas)
        self.rho_max   = int(np.ceil(np.sqrt(w**2 + h**2)))

        # Results
        self.left_lane  = None   # {'slope': float, 'x_bot': float}
        self.right_lane = None
        self.hough_lines = []    # list of (theta, rho)
        self.steer_suggestion = 0.0
        self.line_count = 0

        # Debug images (H x W x 3 uint8)
        self.dbg_gray  = np.zeros((h, w, 3), dtype=np.uint8)
        self.dbg_edge  = np.zeros((h, w, 3), dtype=np.uint8)
        self.dbg_roi   = np.zeros((h, w, 3), dtype=np.uint8)
        self.dbg_hough = np.zeros((h, w, 3), dtype=np.uint8)

    def run(self, frame_bgr):
        """Process one frame through the full pipeline."""
        # Resize to pipeline resolution
        img = cv2.resize(frame_bgr, (self.w, self.h))

        # ── Step 1: Grayscale ─────────────────────────
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        self.dbg_gray = cv2.cvtColor(gray.astype(np.uint8), cv2.COLOR_GRAY2BGR)

        # ── Step 2: Gaussian Blur ─────────────────────
        from scipy.ndimage import convolve
        blurred = cv2.filter2D(gray, -1, self.gauss_k)

        # ── Step 3: Canny Edges ───────────────────────
        gx  = cv2.filter2D(blurred, cv2.CV_32F, self.sobel_x)
        gy  = cv2.filter2D(blurred, cv2.CV_32F, self.sobel_y)
        mag = np.sqrt(gx**2 + gy**2)
        ang = np.arctan2(gy, gx)

        # Non-maximum suppression
        mag_nms = np.zeros_like(mag)
        angle_q = (np.degrees(ang) % 180)
        h, w = mag.shape
        for y in range(1, h-1):
            for x in range(1, w-1):
                m = mag[y, x]
                a = angle_q[y, x]
                if a < 22.5 or a >= 157.5:
                    n1, n2 = mag[y, x+1], mag[y, x-1]
                elif a < 67.5:
                    n1, n2 = mag[y-1, x+1], mag[y+1, x-1]
                elif a < 112.5:
                    n1, n2 = mag[y-1, x], mag[y+1, x]
                else:
                    n1, n2 = mag[y-1, x-1], mag[y+1, x+1]
                mag_nms[y, x] = m if (m >= n1 and m >= n2) else 0

        # Double threshold
        low, high = 30, 80
        edges = np.zeros_like(mag_nms, dtype=np.uint8)
        edges[mag_nms > high]  = 255
        edges[(mag_nms > low) & (mag_nms <= high)] = 128
        self.dbg_edge = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # ── Step 4: ROI Mask ──────────────────────────
        roi_mask = np.zeros((self.h, self.w), dtype=np.uint8)
        roi_top_y = int(self.h * 0.40)
        roi_top_l = int(self.w * 0.30)
        roi_top_r = int(self.w * 0.70)
        # Fill trapezoid
        poly = np.array([[0, self.h],
                          [roi_top_l, roi_top_y],
                          [roi_top_r, roi_top_y],
                          [self.w,   self.h]], dtype=np.int32)
        cv2.fillPoly(roi_mask, [poly], 255)
        roi_edges = cv2.bitwise_and(edges, edges, mask=roi_mask)

        dbg_roi = cv2.cvtColor(roi_edges, cv2.COLOR_GRAY2BGR)
        cv2.polylines(dbg_roi, [poly], True, (30, 180, 255), 1)
        self.dbg_roi = dbg_roi

        # ── Step 5: Hough Transform ───────────────────
        edge_pts = np.argwhere(roi_edges > 0)   # (y, x)
        self.dbg_hough = np.zeros((self.h, self.w, 3), dtype=np.uint8)

        if len(edge_pts) < 10:
            self.left_lane = self.right_lane = None
            self.line_count = 0
            self.steer_suggestion = 0.0
            return

        ys = edge_pts[:, 0].astype(np.float32)
        xs = edge_pts[:, 1].astype(np.float32)

        # Vectorized accumulator
        acc = np.zeros((self.theta_n, self.rho_max * 2), dtype=np.int32)
        for ti in range(self.theta_n):
            rhos = (xs * self.cos_t[ti] + ys * self.sin_t[ti]).astype(np.int32) + self.rho_max
            valid = (rhos >= 0) & (rhos < self.rho_max * 2)
            np.add.at(acc[ti], rhos[valid], 1)

        # ── Step 6: Peak extraction + NMS ────────────
        threshold = max(10, len(edge_pts) // 25)
        peaks = np.argwhere(acc > threshold)    # (theta_idx, rho_idx)
        if len(peaks) == 0:
            self.left_lane = self.right_lane = None
            self.line_count = 0
            self.steer_suggestion = 0.0
            return

        # Sort by votes descending
        votes = acc[peaks[:, 0], peaks[:, 1]]
        order = np.argsort(-votes)
        peaks = peaks[order]

        kept = []
        for ti_idx, ri_idx in peaks:
            theta = self.thetas[ti_idx]
            rho   = ri_idx - self.rho_max
            # NMS: suppress if too close to an already-kept line
            too_close = False
            for k in kept:
                if abs(rho - k[1]) < 15 and abs(theta - k[0]) < 0.25:
                    too_close = True
                    break
            if not too_close:
                kept.append((theta, rho))
            if len(kept) >= 20:
                break

        self.hough_lines = kept
        self.line_count  = len(kept)

        # Draw raw Hough lines on debug canvas (faint green)
        for theta, rho in kept:
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            if abs(sin_t) < 0.01:
                continue
            x0 = int(rho * cos_t)
            y0 = int(rho * sin_t)
            x1 = int(x0 + self.w * (-sin_t))
            y1 = int(y0 + self.h * cos_t)
            x2 = int(x0 - self.w * (-sin_t))
            y2 = int(y0 - self.h * cos_t)
            cv2.line(self.dbg_hough, (x1, y1), (x2, y2), (0, 200, 80), 1)

        # ── Step 7: Classify Left / Right lanes ───────
        left_lines, right_lines = [], []
        mid_x = self.w / 2

        for theta, rho in kept:
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            if abs(sin_t) < 0.01:
                continue   # near-vertical → skip
            # slope in image space: dx/dy
            slope = cos_t / sin_t
            # x at bottom of image
            x_bot = (rho - (self.h - 1) * sin_t) / (cos_t if abs(cos_t) > 0.001 else 0.001)

            if slope < -0.15 and x_bot < mid_x:
                left_lines.append({'slope': slope, 'x_bot': x_bot})
            elif slope > 0.15 and x_bot > mid_x:
                right_lines.append({'slope': slope, 'x_bot': x_bot})

        def avg_lane(lines):
            if not lines:
                return None
            s = sum(l['slope'] for l in lines) / len(lines)
            x = sum(l['x_bot'] for l in lines) / len(lines)
            return {'slope': s, 'x_bot': x}

        self.left_lane  = avg_lane(left_lines)
        self.right_lane = avg_lane(right_lines)

        # Draw averaged left (blue) and right (red) on debug
        def draw_avg(lane, color):
            if lane is None:
                return
            s, xb = lane['slope'], lane['x_bot']
            y2 = self.h
            x2 = int(xb)
            y1 = int(self.h * 0.40)
            x1 = int(x2 - s * (y2 - y1))
            cv2.line(self.dbg_hough, (x1, y1), (x2, y2), color, 2)

        draw_avg(self.left_lane,  (255, 100,  30))   # BGR: blue
        draw_avg(self.right_lane, (80,  80,  220))   # BGR: red

        # Centre line
        if self.left_lane and self.right_lane:
            cx = int((self.left_lane['x_bot'] + self.right_lane['x_bot']) / 2)
            cv2.line(self.dbg_hough, (cx, int(self.h*0.5)), (cx, self.h), (0, 220, 255), 2)

        # Label
        cv2.putText(self.dbg_hough,
                    f"{self.line_count} lines | L:{len(left_lines)} R:{len(right_lines)}",
                    (4, self.h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 120), 1)

        # ── Step 8: Steer suggestion ──────────────────
        if self.left_lane and self.right_lane:
            lane_cx = (self.left_lane['x_bot'] + self.right_lane['x_bot']) / 2
            offset  = (lane_cx - mid_x) / mid_x      # -1..1
            self.steer_suggestion = offset * MAX_STEER
        elif self.left_lane:
            self.steer_suggestion = min(15.0, (mid_x - self.left_lane['x_bot']) / mid_x * MAX_STEER)
        elif self.right_lane:
            self.steer_suggestion = max(-15.0, (self.right_lane['x_bot'] - mid_x) / mid_x * -MAX_STEER)
        else:
            self.steer_suggestion = 0.0


# ─────────────────────────────────────────────────────────────────
# DRAWING HELPERS
# ─────────────────────────────────────────────────────────────────
def lerp_color(c1, c2, t):
    return tuple(int(c1[i] + (c2[i] - c1[i]) * t) for i in range(3))

def draw_dashed_line(surf, color, p1, p2, dash=18, gap=12, width=2):
    dx, dy = p2[0]-p1[0], p2[1]-p1[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return
    ux, uy = dx/length, dy/length
    pos = 0
    drawing = True
    while pos < length:
        seg = min(dash if drawing else gap, length - pos)
        if drawing:
            sx, sy = p1[0]+ux*pos, p1[1]+uy*pos
            ex, ey = p1[0]+ux*(pos+seg), p1[1]+uy*(pos+seg)
            pygame.draw.line(surf, color, (int(sx), int(sy)), (int(ex), int(ey)), width)
        pos += seg
        drawing = not drawing

def draw_arrow(surf, color, pos, angle_deg, size=14):
    """Draw a direction arrow at pos."""
    rad = math.radians(angle_deg)
    tip = (pos[0] + math.sin(rad)*size, pos[1] - math.cos(rad)*size)
    left  = (pos[0] + math.sin(rad-2.4)*size*0.5, pos[1] - math.cos(rad-2.4)*size*0.5)
    right = (pos[0] + math.sin(rad+2.4)*size*0.5, pos[1] - math.cos(rad+2.4)*size*0.5)
    pygame.draw.polygon(surf, color, [tip, left, right])


def draw_car(surf, car):
    """Draw car as a rotated rectangle with lights and windows."""
    cx, cy, ang = car.x, car.y, car.angle
    rad = math.radians(ang)
    cos_a, sin_a = math.cos(rad), math.sin(rad)

    def rot(lx, ly):
        return (cx + lx*cos_a - ly*sin_a,
                cy + lx*sin_a + ly*cos_a)

    hw, hh = CAR_W/2, CAR_H/2

    # Shadow
    pts_s = [rot(hw+2, hh+2), rot(-hw-2, hh+2), rot(-hw-2, -hh-2), rot(hw+2, -hh-2)]
    shadow_surf = pygame.Surface((SIM_W, SIM_H), pygame.SRCALPHA)
    pygame.draw.polygon(shadow_surf, (0,0,0,40), [(int(x),int(y)) for x,y in pts_s])
    surf.blit(shadow_surf, (0,0))

    # Body
    pts = [rot(hw, -hh), rot(-hw, -hh), rot(-hw, hh), rot(hw, hh)]
    pygame.draw.polygon(surf, C_CAR_BODY, [(int(x),int(y)) for x,y in pts])

    # Roof (inner rectangle)
    roof_pts = [rot(hw-5, -hh+13), rot(-hw+5, -hh+13),
                rot(-hw+5,  hh-13), rot(hw-5,  hh-13)]
    pygame.draw.polygon(surf, C_CAR_ROOF, [(int(x),int(y)) for x,y in roof_pts])

    # Front windshield
    fw_pts = [rot(hw-6, -hh+4), rot(-hw+6, -hh+4),
              rot(-hw+6, -hh+14), rot(hw-6,  -hh+14)]
    pygame.draw.polygon(surf, C_CAR_WIN, [(int(x),int(y)) for x,y in fw_pts])

    # Rear windshield
    rw_pts = [rot(hw-6, hh-14), rot(-hw+6, hh-14),
              rot(-hw+6, hh-4),  rot(hw-6,  hh-4)]
    pygame.draw.polygon(surf, C_CAR_WIN, [(int(x),int(y)) for x,y in rw_pts])

    # Headlights (front)
    for sx in [-hw+2, hw-8]:
        hl = [rot(sx, -hh), rot(sx+6, -hh), rot(sx+6, -hh+4), rot(sx, -hh+4)]
        pygame.draw.polygon(surf, C_HEADLIGHT, [(int(x),int(y)) for x,y in hl])

    # Tail lights (rear) — brighter when reversing
    tl_color = (255, 50, 50) if car.speed < -0.05 else C_TAILLIGHT
    for sx in [-hw+2, hw-8]:
        tl = [rot(sx, hh-4), rot(sx+6, hh-4), rot(sx+6, hh), rot(sx, hh)]
        pygame.draw.polygon(surf, tl_color, [(int(x),int(y)) for x,y in tl])

    # Reverse white light
    if car.speed < -0.1:
        rev_pts = [rot(-5, hh-4), rot(5, hh-4), rot(5, hh), rot(-5, hh)]
        pygame.draw.polygon(surf, C_REVERSE_LIGHT, [(int(x),int(y)) for x,y in rev_pts])

    # Outline
    pygame.draw.polygon(surf, (20,20,20), [(int(x),int(y)) for x,y in pts], 1)


def draw_trajectory(surf, car, warning, offset_x=0, offset_y=0):
    """Draw Ackermann reverse trajectory with guide rails and distance markers."""
    pts = get_trajectory(car)
    is_reversing = car.speed < -0.05

    if warning:
        traj_color = C_TRAJ_WARN
    elif is_reversing:
        traj_color = C_TRAJ_REV
    else:
        traj_color = C_TRAJ_FWD

    # Centre dashed path
    for i in range(len(pts) - 1):
        frac = i / (len(pts) - 1)
        alpha = int((1 - frac * 0.65) * 220)
        col   = (*traj_color, alpha)
        # pygame doesn't support per-segment alpha on surfaces easily
        # so we fade the color instead
        faded = lerp_color(traj_color, C_ROAD, frac * 0.6)
        x1, y1 = pts[i][0]+offset_x,   pts[i][1]+offset_y
        x2, y2 = pts[i+1][0]+offset_x, pts[i+1][1]+offset_y
        draw_dashed_line(surf, faded, (x1,y1), (x2,y2), dash=6, gap=5, width=2)

    # Left and right edge rails
    def draw_rail(side, color):
        rail_pts = []
        for px, py, pa in pts:
            rx = px + math.cos(pa) * side * (CAR_W/2 + 4) + offset_x
            ry = py + math.sin(pa) * side * (CAR_W/2 + 4) + offset_y
            rail_pts.append((int(rx), int(ry)))
        if len(rail_pts) > 1:
            # Draw as dashed polyline
            for i in range(len(rail_pts)-1):
                if i % 3 != 2:  # simple dash pattern
                    pygame.draw.line(surf, color, rail_pts[i], rail_pts[i+1], 1)

    draw_rail(-1, C_RAIL_L if not warning else C_TRAJ_WARN)
    draw_rail( 1, C_RAIL_R if not warning else C_TRAJ_WARN)

    # Distance markers at 30 / 60 / 100 cm
    markers = [(30, C_DIST_30, "30cm"), (60, C_DIST_60, "60cm"), (100, C_DIST_100, "100cm")]
    cum = 0
    mi  = 0
    for i in range(1, len(pts)):
        if mi >= len(markers):
            break
        dx = pts[i][0] - pts[i-1][0]
        dy = pts[i][1] - pts[i-1][1]
        cum += math.hypot(dx, dy)
        target, color, label = markers[mi]
        if cum >= target:
            px, py, pa = pts[i]
            perp_x, perp_y = math.cos(pa), math.sin(pa)
            hw = CAR_W/2 + 4
            lx = int(px - perp_x*hw + offset_x)
            ly = int(py - perp_y*hw + offset_y)
            rx = int(px + perp_x*hw + offset_x)
            ry = int(py + perp_y*hw + offset_y)
            pygame.draw.line(surf, color, (lx,ly), (rx,ry), 2)
            # Label
            font_s = pygame.font.SysFont("Arial", 10)
            lbl = font_s.render(label, True, color)
            surf.blit(lbl, (rx + 4, ry - 6))
            mi += 1

    # Steer angle arc at rear axle
    if abs(car.steer) > 0.5:
        rad    = math.radians(car.angle)
        rear_x = car.x - math.sin(rad)*CAR_H/2 + offset_x
        rear_y = car.y + math.cos(rad)*CAR_H/2 + offset_y
        arc_r  = 26
        base   = rad + math.pi/2
        sweep  = (car.steer / MAX_STEER) * (math.pi/2)
        arc_color = C_TRAJ_WARN if warning else C_CENTRE_LANE
        # Draw arc as line segments
        steps = 20
        for i in range(steps):
            t0 = base + sweep * i/steps
            t1 = base + sweep * (i+1)/steps
            x0 = rear_x + math.cos(t0)*arc_r
            y0 = rear_y + math.sin(t0)*arc_r
            x1 = rear_x + math.cos(t1)*arc_r
            y1 = rear_y + math.sin(t1)*arc_r
            pygame.draw.line(surf, arc_color, (int(x0),int(y0)), (int(x1),int(y1)), 2)
        mid_ang = base + sweep/2
        font_s  = pygame.font.SysFont("Arial", 10, bold=True)
        deg_lbl = font_s.render(f"{abs(car.steer):.0f}°", True, arc_color)
        lx = int(rear_x + math.cos(mid_ang)*(arc_r+14))
        ly = int(rear_y + math.sin(mid_ang)*(arc_r+14))
        surf.blit(deg_lbl, (lx - 10, ly - 6))


def draw_simulation(surf, car, warning, world_y, show_guides, hough):
    """Draw the full 3-lane road simulation."""
    surf.fill(C_SIDEWALK)   # sidewalk background

    # Road surface
    pygame.draw.rect(surf, C_ROAD, (ROAD_L, ROAD_T, ROAD_W, ROAD_B - ROAD_T))

    # Lane tint (every other lane slightly lighter)
    for i in [0, 2]:
        lx = ROAD_L + i * LANE_W
        tint = pygame.Surface((LANE_W, ROAD_B - ROAD_T), pygame.SRCALPHA)
        tint.fill((255, 255, 255, 18))
        surf.blit(tint, (lx, ROAD_T))

    # Scrolling lane dividers
    dash, gap = 28, 16
    total = dash + gap
    offset = world_y % total
    for dx in [DIV1, DIV2]:
        y = ROAD_T - offset
        while y < ROAD_B + total:
            y1c = max(ROAD_T, int(y))
            y2c = min(ROAD_B, int(y + dash))
            if y1c < y2c:
                pygame.draw.line(surf, C_DIV_DASH, (dx, y1c), (dx, y2c), 2)
            y += total

    # Road edges (solid white lines)
    pygame.draw.line(surf, C_ROAD_EDGE, (ROAD_L, ROAD_T), (ROAD_L, ROAD_B), 3)
    pygame.draw.line(surf, C_ROAD_EDGE, (ROAD_R, ROAD_T), (ROAD_R, ROAD_B), 3)
    pygame.draw.line(surf, C_DIV_DASH,  (ROAD_L, ROAD_T), (ROAD_R, ROAD_T), 2)
    pygame.draw.line(surf, C_DIV_DASH,  (ROAD_L, ROAD_B), (ROAD_R, ROAD_B), 2)

    # Lane labels
    font_s = pygame.font.SysFont("Arial", 11)
    for i, label in enumerate(["Lane 1", "Lane 2", "Lane 3"]):
        lbl = font_s.render(label, True, C_LANE_LABEL)
        surf.blit(lbl, (int(LANE_CX[i]) - lbl.get_width()//2, ROAD_T + 6))

    # Warning: flash current lane boundary
    if warning and int(time.time()*5) % 2 == 0:
        lb, rb = car.lane_bounds()
        s = pygame.Surface((rb-lb, ROAD_B-ROAD_T), pygame.SRCALPHA)
        s.fill((220, 50, 50, 40))
        surf.blit(s, (lb, ROAD_T))
        pygame.draw.rect(surf, C_WARNING_FLASH, (lb, ROAD_T, rb-lb, ROAD_B-ROAD_T), 3)

    # Hough suggestion label on simulation
    if show_guides and abs(hough.steer_suggestion) > 2:
        dir_str = "▶" if hough.steer_suggestion > 0 else "◀"
        font_h  = pygame.font.SysFont("Arial", 11, bold=True)
        lbl     = font_h.render(f"Hough: {dir_str} {abs(hough.steer_suggestion):.0f}°", True, C_CENTRE_LANE)
        surf.blit(lbl, (SIM_W//2 - lbl.get_width()//2, 8))

    # Trajectory
    if show_guides:
        draw_trajectory(surf, car, warning)

    # Car
    draw_car(surf, car)


def draw_camera_view(surf, cam_frame_bgr, car, hough, warning):
    """Draw camera frame with Hough overlays."""
    if cam_frame_bgr is not None:
        # Convert BGR → RGB and blit
        frame_rgb = cv2.cvtColor(cam_frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = cv2.resize(frame_rgb, (CAM_W, CAM_H))
        pg_surf   = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
        surf.blit(pg_surf, (0, 0))
    else:
        # Synthetic rear-view
        surf.fill((25, 25, 22))
        vp_x, vp_y = CAM_W//2, int(CAM_H*0.4)
        r_near, r_far = CAM_W*0.96, CAM_W*0.08
        steer_off = (car.steer / MAX_STEER) * CAM_W * 0.18

        # Road trapezoid
        road_pts = [
            (int((CAM_W-r_near)/2), CAM_H),
            (int((CAM_W-r_far)/2 + steer_off*0.2), vp_y),
            (int((CAM_W+r_far)/2 + steer_off*0.2), vp_y),
            (int((CAM_W+r_near)/2), CAM_H),
        ]
        pygame.draw.polygon(surf, (55, 55, 50), road_pts)

        # Lane lines
        for li in range(4):
            frac = li / 3
            xn = int((CAM_W-r_near)/2 + frac*r_near + steer_off*0.5)
            xf = int((CAM_W-r_far)/2  + frac*r_far  + steer_off*0.1)
            is_edge = (li == 0 or li == 3)
            color = (220, 215, 200) if is_edge else (160, 155, 145)
            width = 3 if is_edge else 1
            if is_edge:
                pygame.draw.line(surf, color, (xn, CAM_H), (xf, vp_y), width)
            else:
                draw_dashed_line(surf, color, (xn, CAM_H), (xf, vp_y), dash=20, gap=14, width=width)

        # Car hood silhouette
        hood_pts = [(0, CAM_H), (0, int(CAM_H*0.82)),
                    (int(CAM_W*0.15), int(CAM_H*0.72)),
                    (int(CAM_W*0.35), int(CAM_H*0.78)),
                    (int(CAM_W*0.65), int(CAM_H*0.78)),
                    (int(CAM_W*0.85), int(CAM_H*0.72)),
                    (CAM_W, int(CAM_H*0.82)), (CAM_W, CAM_H)]
        pygame.draw.polygon(surf, (15, 15, 14), hood_pts)

    # ── Overlay detected Hough lanes ──────────────
    src_w, src_h = hough.w, hough.h

    def draw_lane_line(lane, color, label):
        if lane is None:
            return
        s, xb = lane['slope'], lane['x_bot']
        y2 = src_h
        y1 = int(src_h * 0.40)
        x2 = int(xb)
        x1 = int(x2 - s * (y2 - y1))
        # Scale to camera canvas
        cx1 = int(x1 * CAM_W / src_w)
        cy1 = int(y1 * CAM_H / src_h)
        cx2 = int(x2 * CAM_W / src_w)
        cy2 = int(y2 * CAM_H / src_h)
        pygame.draw.line(surf, color, (cx1, cy1), (cx2, cy2), 3)
        font_s = pygame.font.SysFont("Arial", 10, bold=True)
        lbl = font_s.render(label, True, color)
        surf.blit(lbl, (cx2 - 20, cy2 - 18))

    draw_lane_line(hough.left_lane,  C_LEFT_LANE,  "LEFT")
    draw_lane_line(hough.right_lane, C_RIGHT_LANE, "RIGHT")

    # Centre line
    if hough.left_lane and hough.right_lane:
        cx = int((hough.left_lane['x_bot'] + hough.right_lane['x_bot'])/2 * CAM_W/src_w)
        draw_dashed_line(surf, C_CENTRE_LANE,
                         (cx, int(CAM_H*0.4)), (cx, CAM_H), dash=10, gap=7, width=2)
        font_s = pygame.font.SysFont("Arial", 10, bold=True)
        lbl = font_s.render("CENTRE", True, C_CENTRE_LANE)
        surf.blit(lbl, (cx - lbl.get_width()//2, int(CAM_H*0.42)))

    # ROI trapezoid outline
    roi_pts = [(0, CAM_H), (int(CAM_W*0.3), int(CAM_H*0.4)),
               (int(CAM_W*0.7), int(CAM_H*0.4)), (CAM_W, CAM_H)]
    pygame.draw.lines(surf, (30, 180, 255, 60), False, roi_pts, 1)

    # Trajectory overlay on camera
    draw_cam_trajectory(surf, car, warning)

    # Hough steer suggestion
    if abs(hough.steer_suggestion) > 2:
        dir_str = "▶ RIGHT" if hough.steer_suggestion > 0 else "◀ LEFT"
        font_h  = pygame.font.SysFont("Arial", 11, bold=True)
        txt     = f"Hough suggests: {dir_str} {abs(hough.steer_suggestion):.0f}°"
        lbl     = font_h.render(txt, True, C_CENTRE_LANE)
        surf.blit(lbl, (CAM_W//2 - lbl.get_width()//2, 8))

    # Warning overlay
    if warning and int(time.time()*5) % 2 == 0:
        warn_s = pygame.Surface((CAM_W, CAM_H), pygame.SRCALPHA)
        warn_s.fill((220, 50, 50, 35))
        surf.blit(warn_s, (0, 0))
        pygame.draw.rect(surf, C_WARNING_FLASH, (0, 0, CAM_W, CAM_H), 4)

    # Label
    font_s = pygame.font.SysFont("Arial", 10)
    lbl    = font_s.render("LIVE CAM + HOUGH" if cam_frame_bgr is not None else "SIM CAM + HOUGH",
                           True, (160, 155, 145))
    surf.blit(lbl, (6, CAM_H - 16))


def draw_cam_trajectory(surf, car, warning):
    """Project simulation trajectory onto the camera perspective view."""
    pts      = get_trajectory(car)
    rev      = car.speed < -0.05
    vp_y     = CAM_H * 0.4

    def sim_to_cam(sx, sy):
        rad     = math.radians(car.angle)
        dx, dy  = sx - car.x, sy - car.y
        behind  = dx*(-math.sin(rad)) + dy*math.cos(rad)
        lateral = dx*math.cos(rad)    + dy*math.sin(rad)
        if behind < 0:
            return None
        t    = min(behind / 220, 1)
        cam_y = CAM_H - (CAM_H - vp_y) * 0.85 * t
        cam_x = CAM_W/2 + lateral * 1.8 * (1 - t*0.7) + (car.steer/MAX_STEER)*CAM_W*0.12*(1-t)
        return int(cam_x), int(cam_y), t

    if warning:
        traj_col = C_TRAJ_WARN
    elif rev:
        traj_col = C_TRAJ_REV
    else:
        traj_col = C_TRAJ_FWD

    prev = None
    for i, (px, py, pa) in enumerate(pts):
        cp = sim_to_cam(px, py)
        if cp is None:
            prev = None
            continue
        if prev:
            frac  = i / (len(pts)-1)
            color = lerp_color(traj_col, (40, 40, 38), frac * 0.6)
            lw    = max(1, int((1 - prev[2]) * 4))
            if i % 3 != 2:
                pygame.draw.line(surf, color, (prev[0], prev[1]), (cp[0], cp[1]), lw)
        prev = cp

    # Edge rails
    for side, color in [(-1, C_RAIL_L if not warning else C_TRAJ_WARN),
                         (1,  C_RAIL_R if not warning else C_TRAJ_WARN)]:
        rail = []
        for px, py, pa in pts:
            ex = px + math.cos(pa) * side * (CAR_W/2 + 3)
            ey = py + math.sin(pa) * side * (CAR_W/2 + 3)
            cp = sim_to_cam(ex, ey)
            if cp:
                rail.append((cp[0], cp[1]))
        for i in range(len(rail)-1):
            if i % 3 != 2:
                pygame.draw.line(surf, color, rail[i], rail[i+1], 1)


def draw_pipeline_panel(surf, hough, x_off, y_off):
    """Draw 4 Hough pipeline stage thumbnails."""
    PW2, PH2 = 290, 165
    stages = [
        (hough.dbg_gray,  "① Grayscale"),
        (hough.dbg_edge,  "② Canny Edges"),
        (hough.dbg_roi,   "③ ROI Mask"),
        (hough.dbg_hough, "④ Hough Lines"),
    ]
    font_s = pygame.font.SysFont("Arial", 10, bold=True)
    for i, (img_bgr, label) in enumerate(stages):
        col = i % 2
        row = i // 2
        px  = x_off + col * (PW2 + PAD)
        py  = y_off + row * (PH2 + 18)

        # Convert cv2 BGR → pygame surface
        img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_small = cv2.resize(img_rgb, (PW2, PH2))
        pg_s      = pygame.surfarray.make_surface(img_small.swapaxes(0, 1))
        surf.blit(pg_s, (px, py + 14))
        pygame.draw.rect(surf, (60, 60, 58), (px, py+14, PW2, PH2), 1)

        # Label
        lbl = font_s.render(label, True, C_TEXT)
        surf.blit(lbl, (px, py))


def draw_hud(surf, car, hough, warning, cam_active, show_guides, steer_target, font_m, font_s):
    """Draw HUD panel at bottom."""
    hud_y   = SIM_H + 10
    hud_rects = []

    cards = [
        ("Speed",         f"{abs(car.speed):.2f}"),
        ("Steer",         f"{car.steer:.0f}°"),
        ("Lane",          str(car.current_lane())),
        ("Hough Lines",   str(hough.line_count)),
        ("Left Lane",     f"{hough.left_lane['x_bot']:.0f}px" if hough.left_lane  else "--"),
        ("Right Lane",    f"{hough.right_lane['x_bot']:.0f}px" if hough.right_lane else "--"),
        ("H.Steer Sug.",  f"{hough.steer_suggestion:+.0f}°"),
        ("Status",        "⚠ WARN" if warning else "Safe"),
        ("Camera",        "ON" if cam_active else "OFF"),
    ]

    card_w = (WIN_W - 2*PAD) // len(cards)
    for i, (label, value) in enumerate(cards):
        cx = PAD + i * card_w
        if warning and label == "Status":
            bg = (60, 15, 15)
            vc = C_WARNING_FLASH
        elif label == "Camera" and cam_active:
            bg = (15, 45, 15)
            vc = C_TRAJ_FWD
        else:
            bg = C_HUD_BG
            vc = C_TEXT

        pygame.draw.rect(surf, bg, (cx, hud_y, card_w-4, 50), border_radius=6)
        lbl_s  = font_s.render(label, True, C_TEXT_DIM)
        val_s  = font_m.render(value, True, vc)
        surf.blit(lbl_s, (cx + (card_w-4-lbl_s.get_width())//2, hud_y + 6))
        surf.blit(val_s, (cx + (card_w-4-val_s.get_width())//2, hud_y + 22))

    # Steer bar
    bar_y = hud_y + 58
    bar_w = WIN_W - 2*PAD
    pygame.draw.rect(surf, C_HUD_BG, (PAD, bar_y, bar_w, 10), border_radius=5)
    pct = car.steer / MAX_STEER   # -1..1
    if pct >= 0:
        fill_x = PAD + bar_w//2
        fill_w = int(pct * bar_w/2)
    else:
        fill_x = PAD + int((0.5+pct/2)*bar_w)
        fill_w = int(-pct * bar_w/2)
    bar_color = (200, 130, 20) if abs(pct) > 0.75 else (30, 120, 220)
    if fill_w > 0:
        pygame.draw.rect(surf, bar_color, (fill_x, bar_y, fill_w, 10), border_radius=5)
    # Centre tick
    pygame.draw.line(surf, C_TEXT, (PAD+bar_w//2, bar_y-2), (PAD+bar_w//2, bar_y+12), 2)
    font_s2 = pygame.font.SysFont("Arial", 9)
    surf.blit(font_s2.render("◀ 45°", True, C_TEXT_DIM), (PAD, bar_y+12))
    r_lbl = font_s2.render("45° ▶", True, C_TEXT_DIM)
    surf.blit(r_lbl, (WIN_W - PAD - r_lbl.get_width(), bar_y+12))

    # Controls hint
    hint_y = bar_y + 28
    hint = ("↑↓ Drive  |  ←→ Steer  |  Space Brake  |  0 Centre  |  1-3 Left 15/30/45°  |  "
            "4-6 Right 15/30/45°  |  C Camera  |  G Guides  |  R Reset  |  Q Quit")
    hint_s = font_s.render(hint, True, C_TEXT_DIM)
    surf.blit(hint_s, (WIN_W//2 - hint_s.get_width()//2, hint_y))


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    pygame.init()
    pygame.display.set_caption("Reverse Parking Assist — Hough Transform + Live Camera")
    screen = pygame.display.set_mode((WIN_W, WIN_H))
    clock  = pygame.time.Clock()

    # Fonts
    font_m = pygame.font.SysFont("Arial", 15, bold=True)
    font_s = pygame.font.SysFont("Arial", 11)
    font_h = pygame.font.SysFont("Arial", 13, bold=True)

    # Surfaces
    sim_surf = pygame.Surface((SIM_W, SIM_H))
    cam_surf = pygame.Surface((CAM_W, CAM_H))

    # Sim state
    car        = Car()
    world_y    = 0.0
    show_guides = True
    steer_target = None
    warning    = False

    # Camera
    cam_active = False
    cap        = None
    cam_frame  = None     # latest BGR frame

    # Hough pipeline (processes 320x180 frames)
    hough = HoughPipeline(320, 180)

    frame_count = 0

    def try_open_camera():
        nonlocal cap, cam_active, cam_frame
        for idx in [0, 1, 2]:
            c = cv2.VideoCapture(idx)
            if c.isOpened():
                c.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
                c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap = c
                cam_active = True
                print(f"✅ Camera opened at index {idx}")
                return True
        print("❌ No camera found")
        cam_active = False
        return False

    def close_camera():
        nonlocal cap, cam_active, cam_frame
        if cap:
            cap.release()
            cap = None
        cam_active = False
        cam_frame  = None

    # ── Main Loop ────────────────────────────────
    running = True
    while running:
        frame_count += 1

        # ── Events ───────────────────────────────
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                # Steer presets
                if event.key == pygame.K_0:
                    steer_target = 0.0
                elif event.key == pygame.K_1:
                    steer_target = -15.0
                elif event.key == pygame.K_2:
                    steer_target = -30.0
                elif event.key == pygame.K_3:
                    steer_target = -45.0
                elif event.key == pygame.K_4:
                    steer_target = 15.0
                elif event.key == pygame.K_5:
                    steer_target = 30.0
                elif event.key == pygame.K_6:
                    steer_target = 45.0
                # Toggle camera
                elif event.key == pygame.K_c:
                    if cam_active:
                        close_camera()
                    else:
                        try_open_camera()
                # Toggle guides
                elif event.key == pygame.K_g:
                    show_guides = not show_guides
                # Reset
                elif event.key == pygame.K_r:
                    car.reset()
                    world_y = 0.0
                    steer_target = None
                # Quit
                elif event.key in [pygame.K_q, pygame.K_ESCAPE]:
                    running = False
                # Manual steer clears preset
                elif event.key in [pygame.K_LEFT, pygame.K_RIGHT]:
                    steer_target = None

        # ── Physics update ────────────────────────
        keys = pygame.key.get_pressed()
        car.update(keys, steer_target)
        rad     = math.radians(car.angle)
        world_y += car.speed * math.cos(rad)

        # ── Lane / Warning check ──────────────────
        lb, rb = car.lane_bounds()
        lane   = car.current_lane()
        corners = car.get_corners()
        v = any(cx < ROAD_L or cx > ROAD_R for cx, cy in corners)
        if car.x - CAR_W/2 - 3 < lb and lane > 1: v = True
        if car.x + CAR_W/2 + 3 > rb and lane < 3: v = True
        if abs(hough.steer_suggestion) > 32:        v = True
        warning = v

        # ── Grab camera frame ─────────────────────
        if cam_active and cap and frame_count % 2 == 0:
            ret, frame = cap.read()
            if ret:
                # Flip if front-facing camera
                cam_frame = frame
            else:
                close_camera()

        # ── Hough pipeline (every 3rd frame) ─────
        if frame_count % 3 == 0:
            if cam_frame is not None:
                hough.run(cam_frame)
            else:
                # Synthesise road image for Hough
                syn = np.zeros((180, 320, 3), dtype=np.uint8)
                syn[:] = (50, 50, 48)  # road color
                vp_y = int(180 * 0.4)
                steer_off = int((car.steer / MAX_STEER) * 320 * 0.15)
                for li in range(4):
                    frac = li / 3
                    xn = int(320*0.02 + frac*320*0.96 + steer_off*0.5)
                    xf = int(320*0.30 + frac*320*0.40 + steer_off*0.1)
                    is_edge = (li == 0 or li == 3)
                    color = (210, 205, 190) if is_edge else (140, 135, 125)
                    thick = 3 if is_edge else 1
                    cv2.line(syn, (xn, 180), (xf, vp_y), color, thick)
                hough.run(syn)

        # ── Draw ──────────────────────────────────
        screen.fill(C_BG)

        # Left panel: Simulation
        draw_simulation(sim_surf, car, warning, world_y, show_guides, hough)
        screen.blit(sim_surf, (PAD, PAD))
        lbl = font_h.render("Simulation — Reverse Parking Assist", True, C_TEXT)
        screen.blit(lbl, (PAD, PAD - 2))   # above panel (tiny overlap is fine)

        # Right panel: Camera + Hough overlay
        draw_camera_view(cam_surf,
                         cam_frame if cam_active else None,
                         car, hough, warning)
        screen.blit(cam_surf, (PAD + SIM_W + PAD, PAD))
        lbl2 = font_h.render("Camera + Hough Lane Detection", True, C_TEXT)
        screen.blit(lbl2, (PAD + SIM_W + PAD, PAD - 2))

        # Bottom: Pipeline thumbnails
        pipe_y = PAD + SIM_H + PAD
        draw_pipeline_panel(screen, hough, PAD, pipe_y)

        # HUD
        hud_base = pipe_y + (PIPE_H + 18)*2 + PAD
        draw_hud(screen, car, hough, warning,
                 cam_active, show_guides, steer_target, font_m, font_s)

        # Panel borders
        pygame.draw.rect(screen, (60,60,58), (PAD, PAD, SIM_W, SIM_H), 1)
        pygame.draw.rect(screen, (60,60,58), (PAD+SIM_W+PAD, PAD, CAM_W, CAM_H), 1)

        pygame.display.flip()
        clock.tick(30)

    # Cleanup
    close_camera()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()