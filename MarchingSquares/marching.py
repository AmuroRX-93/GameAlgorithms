"""
Marching Squares — 2D contour extraction.

Given a scalar field sampled on a grid, the algorithm walks each cell ("square")
and emits the line segments that make up the iso-contour at level T:

  1. Sample the field at the four corners.
  2. Threshold each corner against T -> 4 bits -> case index 0..15.
  3. Look up the case in a 16-entry table to know which edges of the square
     the contour crosses.
  4. Place each crossing point either at the edge midpoint (fast, blocky look)
     or by linear interpolation between the two corner values (smooth).

The scalar field in this demo is the sum of inverse-square contributions from
N "metaballs". Move the mouse to drag balls, scroll to resize them, and use
the sliders to tweak grid resolution and the iso threshold in real time.
"""

import math
import random
import sys

import numpy as np
import pygame

# ---------------------------------------------------------------------------
# Layout / colors
# ---------------------------------------------------------------------------

WIDTH, HEIGHT = 1280, 800
HUD_W = 280
SIM_W = WIDTH - HUD_W
SIM_H = HEIGHT
FPS = 60

BG = (14, 18, 28)
PANEL_BG = (22, 26, 40)
PANEL_BORDER = (60, 70, 95)
TEXT = (220, 226, 240)
TEXT_DIM = (140, 150, 175)
ACCENT = (245, 210, 80)

INSIDE_FILL = (60, 110, 175)        # Filled region inside the iso-surface.
INSIDE_FILL_GLOW = (90, 165, 235)
CONTOUR = (220, 240, 255)           # Iso-line color.
GRID_DOT_INSIDE = (110, 200, 250)
GRID_DOT_OUTSIDE = (60, 70, 95)
BALL_OUTLINE = (245, 210, 80)
BALL_OUTLINE_DRAG = (255, 255, 255)

SLIDER_TRACK = (50, 58, 80)
SLIDER_FILL = (130, 200, 240)
SLIDER_KNOB = (220, 240, 255)


# ---------------------------------------------------------------------------
# Marching Squares core
# ---------------------------------------------------------------------------
#
# Corner indexing for each cell:
#
#       (0)----top-----(1)
#        |              |
#       left           right
#        |              |
#       (3)---bottom---(2)
#
# Bit value 1 means "corner is INSIDE" (field >= threshold).
# Case index = bit3*8 + bit2*4 + bit1*2 + bit0  with bits ordered (0,1,2,3).
#
# Each case emits 0, 1, or 2 line segments. Each segment is described by the
# two edges it connects, named with a single letter:
#   T = top edge midpoint
#   R = right edge midpoint
#   B = bottom edge midpoint
#   L = left edge midpoint
#
# Cases 5 and 10 are "saddle" cases: two diagonally-opposite corners are
# inside. We resolve them using the average of the four corners (the standard
# "asymptotic decider" approximation) so the lines don't cross weirdly.

# Edge -> the two corner indices that bound it (used for linear interp).
EDGE_CORNERS = {
    "T": (0, 1),
    "R": (1, 2),
    "B": (3, 2),
    "L": (0, 3),
}

# For each case (excluding saddles 5, 10), the line segment(s) to draw, as
# pairs of edge letters.
#
# Naming reminder: corners are indexed 0=TL, 1=TR, 2=BR, 3=BL. Edges between
# corners: T (0-1), R (1-2), B (3-2), L (0-3). For each case, the contour
# crosses the edges that bound a transition between an "inside" and an
# "outside" corner.
CASE_SEGMENTS = {
    0:  [],
    1:  [("L", "T")],   # only corner 0 inside  -> wraps top-left
    2:  [("T", "R")],   # only corner 1 inside  -> wraps top-right
    3:  [("L", "R")],   # 0,1 inside (top half)
    4:  [("R", "B")],   # only corner 2 inside  -> wraps bottom-right
    6:  [("T", "B")],   # 1,2 inside (right half)
    7:  [("L", "B")],   # 0,1,2 inside (3 outside) -> small bottom-left cut
    8:  [("B", "L")],   # only corner 3 inside  -> wraps bottom-left
    9:  [("T", "B")],   # 0,3 inside (left half)
    11: [("R", "B")],   # 0,1,3 inside (2 outside) -> small bottom-right cut
    12: [("L", "R")],   # 2,3 inside (bottom half)
    13: [("T", "R")],   # 0,2,3 inside (1 outside) -> small top-right cut
    14: [("L", "T")],   # 1,2,3 inside (0 outside) -> small top-left cut
    15: [],
}

# Saddles depend on the sign of the cell-center value. With center INSIDE
# (avg >= threshold) the two regions connect through the middle ("touching"
# topology). With center OUTSIDE they are separated.
# Case 5: corners 0 and 2 inside, 1 and 3 outside.
# Case 10: corners 1 and 3 inside, 0 and 2 outside.
SADDLE_5_CONNECTED  = [("T", "R"), ("B", "L")]   # center inside; wraps OUT corners 1 and 3
SADDLE_5_SEPARATED  = [("L", "T"), ("R", "B")]   # center outside; isolates IN corners 0 and 2
SADDLE_10_CONNECTED = [("L", "T"), ("R", "B")]   # center inside; wraps OUT corners 0 and 2
SADDLE_10_SEPARATED = [("T", "R"), ("B", "L")]   # center outside; isolates IN corners 1 and 3


def case_index(c0, c1, c2, c3, threshold):
    """Pack the four corner values' threshold tests into 0..15."""
    idx = 0
    if c0 >= threshold: idx |= 1
    if c1 >= threshold: idx |= 2
    if c2 >= threshold: idx |= 4
    if c3 >= threshold: idx |= 8
    return idx


def edge_point(edge, corner_xy, corner_val, threshold, interpolate):
    """Compute the (x, y) where this edge crosses the iso-contour.

    `corner_xy[i]` is the (x, y) of corner i.
    `corner_val[i]` is the field value at corner i.
    With `interpolate=False` we just use the edge midpoint, giving a chunky
    pixel-art look; with `interpolate=True` we linearly interpolate using the
    corner values for a smooth contour.
    """
    a, b = EDGE_CORNERS[edge]
    ax, ay = corner_xy[a]
    bx, by = corner_xy[b]
    if not interpolate:
        return (0.5 * (ax + bx), 0.5 * (ay + by))
    va = corner_val[a]
    vb = corner_val[b]
    denom = vb - va
    if abs(denom) < 1e-12:
        t = 0.5
    else:
        t = (threshold - va) / denom
        t = max(0.0, min(1.0, t))
    return (ax + t * (bx - ax), ay + t * (by - ay))


# ---------------------------------------------------------------------------
# Scalar field (metaballs)
# ---------------------------------------------------------------------------
#
# The contribution of one ball at (cx, cy) with radius r to a point (x, y) is
#   value = r^2 / max(eps, (x - cx)^2 + (y - cy)^2)
# This is the classic metaball kernel: 1 right at the center, falling off
# smoothly toward 0 outside. The threshold T then carves a smooth blob.


def compute_field(balls, sample_x, sample_y, out=None):
    """Vectorized field evaluation on a grid.

    `sample_x` shape (NX,), `sample_y` shape (NY,). Returns a 2D array of
    shape (NY, NX).
    """
    NX = sample_x.shape[0]
    NY = sample_y.shape[0]
    if out is None or out.shape != (NY, NX):
        out = np.zeros((NY, NX), dtype=np.float32)
    else:
        out.fill(0.0)
    if not balls:
        return out
    # Broadcast: shape (NY, NX) for dx, dy, then accumulate.
    eps = np.float32(1.0)
    for cx, cy, r in balls:
        dx = sample_x[None, :] - np.float32(cx)
        dy = sample_y[:, None] - np.float32(cy)
        d2 = dx * dx + dy * dy + eps
        out += np.float32(r * r) / d2
    return out


# ---------------------------------------------------------------------------
# Marching squares extraction
# ---------------------------------------------------------------------------


def extract_contours(field, sample_x, sample_y, threshold, interpolate=True,
                     emit_polygons=True):
    """Walk every cell and return a list of line segments and inside-fill
    polygons.

    Returns:
      segments: list of ((x1, y1), (x2, y2)) — the iso-contour line set.
      polygons: list of [(x, y), ...] polygons giving the filled region for
                each cell that is partially or fully inside the iso-surface.
                If ``emit_polygons=False`` this list is always empty (faster
                when you fill with a rasterized mask instead).

    Performance: most of the work (case classification + edge-crossing
    coordinates) is vectorized in numpy. Only cells with a non-trivial case
    (idx in 1..14) enter the Python loop.
    """
    NY, NX = field.shape
    if NY < 2 or NX < 2:
        return [], []

    # ----- Vectorized case classification --------------------------------
    inside = field >= threshold  # bool grid (NY, NX)
    c0 = inside[:-1, :-1]
    c1 = inside[:-1, 1:]
    c2 = inside[1:,  1:]
    c3 = inside[1:,  :-1]
    idx_grid = (c0.astype(np.uint8)
                | (c1.astype(np.uint8) << 1)
                | (c2.astype(np.uint8) << 2)
                | (c3.astype(np.uint8) << 3))

    # ----- Precompute edge crossings per cell ----------------------------
    # Corner values (NY-1, NX-1).
    v0 = field[:-1, :-1]
    v1 = field[:-1, 1:]
    v2 = field[1:,  1:]
    v3 = field[1:,  :-1]

    # Corner XY (broadcasted).
    x0 = sample_x[:-1][None, :]
    x1 = sample_x[1:][None, :]
    y0 = sample_y[:-1][:, None]
    y1 = sample_y[1:][:, None]

    if interpolate:
        eps = np.float32(1e-12)
        # T edge: corners 0 (x0,y0) -> 1 (x1,y0). Y is constant.
        denom_T = (v1 - v0)
        t_T = np.where(np.abs(denom_T) < eps, np.float32(0.5),
                       (np.float32(threshold) - v0) / np.where(np.abs(denom_T) < eps, eps, denom_T))
        t_T = np.clip(t_T, 0.0, 1.0)
        T_x = x0 + t_T * (x1 - x0)
        T_y = np.broadcast_to(y0, T_x.shape).astype(np.float32, copy=False)

        # R edge: 1 (x1,y0) -> 2 (x1,y1).
        denom_R = (v2 - v1)
        t_R = np.where(np.abs(denom_R) < eps, np.float32(0.5),
                       (np.float32(threshold) - v1) / np.where(np.abs(denom_R) < eps, eps, denom_R))
        t_R = np.clip(t_R, 0.0, 1.0)
        R_x = np.broadcast_to(x1, t_R.shape).astype(np.float32, copy=False)
        R_y = y0 + t_R * (y1 - y0)

        # B edge: 3 (x0,y1) -> 2 (x1,y1). Y constant.
        denom_B = (v2 - v3)
        t_B = np.where(np.abs(denom_B) < eps, np.float32(0.5),
                       (np.float32(threshold) - v3) / np.where(np.abs(denom_B) < eps, eps, denom_B))
        t_B = np.clip(t_B, 0.0, 1.0)
        B_x = x0 + t_B * (x1 - x0)
        B_y = np.broadcast_to(y1, B_x.shape).astype(np.float32, copy=False)

        # L edge: 0 (x0,y0) -> 3 (x0,y1).
        denom_L = (v3 - v0)
        t_L = np.where(np.abs(denom_L) < eps, np.float32(0.5),
                       (np.float32(threshold) - v0) / np.where(np.abs(denom_L) < eps, eps, denom_L))
        t_L = np.clip(t_L, 0.0, 1.0)
        L_x = np.broadcast_to(x0, t_L.shape).astype(np.float32, copy=False)
        L_y = y0 + t_L * (y1 - y0)
    else:
        # Midpoints — same shape as the cell grid.
        mx = 0.5 * (x0 + x1)
        my = 0.5 * (y0 + y1)
        ny1 = idx_grid.shape[0]; nx1 = idx_grid.shape[1]
        T_x = np.broadcast_to(mx, (ny1, nx1)).astype(np.float32, copy=False)
        T_y = np.broadcast_to(y0, (ny1, nx1)).astype(np.float32, copy=False)
        R_x = np.broadcast_to(x1, (ny1, nx1)).astype(np.float32, copy=False)
        R_y = np.broadcast_to(my, (ny1, nx1)).astype(np.float32, copy=False)
        B_x = np.broadcast_to(mx, (ny1, nx1)).astype(np.float32, copy=False)
        B_y = np.broadcast_to(y1, (ny1, nx1)).astype(np.float32, copy=False)
        L_x = np.broadcast_to(x0, (ny1, nx1)).astype(np.float32, copy=False)
        L_y = np.broadcast_to(my, (ny1, nx1)).astype(np.float32, copy=False)

    # Saddle resolution: center value = avg of corners.
    center_inside = (v0 + v1 + v2 + v3) * 0.25 >= threshold

    # ----- Walk only non-trivial cells ----------------------------------
    nontrivial = (idx_grid != 0) & (idx_grid != 15)
    js, is_ = np.nonzero(nontrivial)

    segments = []
    polygons = []

    # Local refs for speed inside the hot loop.
    case_segs_table = CASE_SEGMENTS
    saddle5c = SADDLE_5_CONNECTED; saddle5s = SADDLE_5_SEPARATED
    saddle10c = SADDLE_10_CONNECTED; saddle10s = SADDLE_10_SEPARATED

    # Lookups for edge-letter -> (x_grid, y_grid).
    edge_grids = {
        "T": (T_x, T_y),
        "R": (R_x, R_y),
        "B": (B_x, B_y),
        "L": (L_x, L_y),
    }

    for j, i in zip(js.tolist(), is_.tolist()):
        idx = int(idx_grid[j, i])
        if idx == 5:
            segs = saddle5c if center_inside[j, i] else saddle5s
        elif idx == 10:
            segs = saddle10c if center_inside[j, i] else saddle10s
        else:
            segs = case_segs_table[idx]

        # Resolve edge points by table lookup into the precomputed grids.
        edge_pts = {}
        for e1, e2 in segs:
            if e1 not in edge_pts:
                gx, gy = edge_grids[e1]
                edge_pts[e1] = (float(gx[j, i]), float(gy[j, i]))
            if e2 not in edge_pts:
                gx, gy = edge_grids[e2]
                edge_pts[e2] = (float(gx[j, i]), float(gy[j, i]))
            segments.append((edge_pts[e1], edge_pts[e2]))

        cx0 = float(sample_x[i]);     cy0 = float(sample_y[j])
        cx1 = float(sample_x[i + 1]); cy1 = float(sample_y[j + 1])
        if emit_polygons:
            corner_xy = ((cx0, cy0), (cx1, cy0), (cx1, cy1), (cx0, cy1))
            poly = _cell_inside_polygon(idx, corner_xy, edge_pts, segs)
            if poly:
                polygons.append(poly)

    # Cells that are entirely inside (case 15) get a full square fill.
    if emit_polygons:
        full = (idx_grid == 15)
        if full.any():
            full_js, full_is = np.nonzero(full)
            for j, i in zip(full_js.tolist(), full_is.tolist()):
                cx0 = float(sample_x[i]);     cy0 = float(sample_y[j])
                cx1 = float(sample_x[i + 1]); cy1 = float(sample_y[j + 1])
                polygons.append([(cx0, cy0), (cx1, cy0), (cx1, cy1), (cx0, cy1)])

    return segments, polygons


# Walking order for cell corners and the edges between them.
# (corner_idx, outgoing_edge_letter)
CORNER_WALK = [(0, "T"), (1, "R"), (2, "B"), (3, "L")]


def _cell_inside_polygon(idx, corner_xy, edge_pts, segs):
    """For a cell with the given case, return the polygon that is INSIDE the
    iso-surface (so we can fill it solid). Returns [] if nothing inside.

    Algorithm: walk corners 0->1->2->3 in order. Whenever the corner is
    "inside" we add it to the polygon. Whenever we cross an edge that's part
    of the contour, we add the crossing point. This produces a clockwise
    polygon for the inside region.
    """
    if idx == 0:
        return []
    if idx == 15:
        return [corner_xy[0], corner_xy[1], corner_xy[2], corner_xy[3]]

    inside = [(idx >> b) & 1 for b in range(4)]
    # Edges that bound the contour, as a set for quick lookup of crossings.
    contour_edges = set()
    for e1, e2 in segs:
        contour_edges.add(e1)
        contour_edges.add(e2)

    poly = []
    for k in range(4):
        c_now = k
        c_next = (k + 1) % 4
        edge = CORNER_WALK[k][1]   # edge from corner k to corner k+1
        if inside[c_now]:
            poly.append(corner_xy[c_now])
        if inside[c_now] != inside[c_next]:
            # Crossing on this edge.
            if edge in edge_pts:
                poly.append(edge_pts[edge])
    return poly


# ---------------------------------------------------------------------------
# Metaball entities
# ---------------------------------------------------------------------------


class Ball:
    __slots__ = ("x", "y", "vx", "vy", "r")

    def __init__(self, x, y, r=80.0):
        self.x = float(x)
        self.y = float(y)
        self.r = float(r)
        # Slow drift so the scene moves on its own.
        ang = random.uniform(0, math.tau)
        spd = random.uniform(20.0, 60.0)
        self.vx = math.cos(ang) * spd
        self.vy = math.sin(ang) * spd

    def update(self, dt, bounds):
        x0, y0, x1, y1 = bounds
        self.x += self.vx * dt
        self.y += self.vy * dt
        # Bounce inside the simulation rectangle.
        if self.x < x0 + self.r * 0.3:
            self.x = x0 + self.r * 0.3
            self.vx = abs(self.vx)
        elif self.x > x1 - self.r * 0.3:
            self.x = x1 - self.r * 0.3
            self.vx = -abs(self.vx)
        if self.y < y0 + self.r * 0.3:
            self.y = y0 + self.r * 0.3
            self.vy = abs(self.vy)
        elif self.y > y1 - self.r * 0.3:
            self.y = y1 - self.r * 0.3
            self.vy = -abs(self.vy)

    def hit(self, mx, my):
        dx = mx - self.x
        dy = my - self.y
        return dx * dx + dy * dy <= max(20.0, self.r * 0.4) ** 2


# ---------------------------------------------------------------------------
# Slider UI
# ---------------------------------------------------------------------------


class Slider:
    HEIGHT = 14
    LABEL_GAP = 18

    def __init__(self, x, y, w, label, lo, hi, step, getter, setter, fmt=None):
        self.label = label
        self.lo = lo
        self.hi = hi
        self.step = step
        self.getter = getter
        self.setter = setter
        self.fmt = fmt
        self.rect = pygame.Rect(x, y + Slider.LABEL_GAP, w, Slider.HEIGHT)

    def value_to_x(self, v):
        t = (v - self.lo) / (self.hi - self.lo)
        return int(self.rect.x + t * self.rect.w)

    def x_to_value(self, x):
        t = max(0.0, min(1.0, (x - self.rect.x) / self.rect.w))
        v = self.lo + t * (self.hi - self.lo)
        return round(v / self.step) * self.step

    def draw(self, surface, font):
        v = self.getter()
        if self.fmt is not None:
            label = f"{self.label}: {self.fmt(v)}"
        elif v == int(v):
            label = f"{self.label}: {int(v)}"
        else:
            label = f"{self.label}: {round(v, 2)}"
        surface.blit(font.render(label, True, TEXT),
                     (self.rect.x, self.rect.y - Slider.LABEL_GAP + 1))
        pygame.draw.rect(surface, SLIDER_TRACK, self.rect, border_radius=4)
        knob_x = self.value_to_x(v)
        fill = pygame.Rect(self.rect.x, self.rect.y, knob_x - self.rect.x, self.rect.h)
        pygame.draw.rect(surface, SLIDER_FILL, fill, border_radius=4)
        pygame.draw.circle(surface, SLIDER_KNOB, (knob_x, self.rect.centery), 7)

    def hit(self, mx, my):
        return (self.rect.x - 4 <= mx <= self.rect.right + 4
                and self.rect.y - 6 <= my <= self.rect.bottom + 6)

    def update_from_mouse(self, mx):
        v = self.x_to_value(mx)
        if self.step >= 1:
            v = int(v)
        self.setter(v)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main():
    pygame.init()
    pygame.display.set_caption("Marching Squares")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Menlo,Consolas,monospace", 13)
    small = pygame.font.SysFont("Menlo,Consolas,monospace", 11)
    title_font = pygame.font.SysFont("Menlo,Consolas,monospace", 16, bold=True)

    state = {
        "cell_size": 18,        # Pixels between samples (smaller = smoother).
        "threshold": 1.6,       # Iso-level.
        "interpolate": 1,       # Linear interpolation along edges.
        "show_grid": 0,         # Show sample dots.
        "show_balls": 1,        # Show ball outlines.
        "animate": 1,           # Whether balls drift.
    }

    balls = [
        Ball(SIM_W * 0.35, SIM_H * 0.45, r=120),
        Ball(SIM_W * 0.55, SIM_H * 0.45, r=110),
        Ball(SIM_W * 0.45, SIM_H * 0.65, r=90),
        Ball(SIM_W * 0.65, SIM_H * 0.35, r=70),
    ]

    pad = 16
    sx = SIM_W + pad
    sw = HUD_W - 2 * pad
    sliders = [
        Slider(sx, 90,  sw, "Cell size (px)", 4, 48, 1,
               lambda: state["cell_size"], lambda v: state.update(cell_size=int(v))),
        Slider(sx, 132, sw, "Iso threshold",   0.2, 5.0, 0.05,
               lambda: state["threshold"], lambda v: state.update(threshold=float(v)),
               fmt=lambda v: f"{v:.2f}"),
        Slider(sx, 174, sw, "Linear interp",   0, 1, 1,
               lambda: state["interpolate"], lambda v: state.update(interpolate=int(v))),
        Slider(sx, 216, sw, "Animate",         0, 1, 1,
               lambda: state["animate"], lambda v: state.update(animate=int(v))),
        Slider(sx, 258, sw, "Show sample grid", 0, 1, 1,
               lambda: state["show_grid"], lambda v: state.update(show_grid=int(v))),
        Slider(sx, 300, sw, "Show balls",      0, 1, 1,
               lambda: state["show_balls"], lambda v: state.update(show_balls=int(v))),
    ]

    active_slider = None
    dragging_ball = None     # The Ball currently held with the left mouse.
    drag_offset = (0, 0)
    field_buf = None

    sim_bounds = (0, 0, SIM_W, SIM_H)

    # Pre-compute what we can each frame; the sample grid only needs rebuilding
    # when cell_size changes.
    cached_cs = None
    sample_x = sample_y = None
    # Reusable small surface for rasterized inside-fill (one pixel per cell).
    fill_lo = None  # pygame.Surface sized (NX-1, NY-1)
    fill_lo_arr = None  # 3D uint8 view we update via surfarray

    while True:
        dt_ms = clock.tick(FPS)
        dt = min(0.05, dt_ms / 1000.0)

        # ---- Events ------------------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit(0)
                elif event.key == pygame.K_SPACE:
                    state["animate"] = 0 if state["animate"] else 1
                elif event.key == pygame.K_g:
                    state["show_grid"] = 0 if state["show_grid"] else 1
                elif event.key == pygame.K_i:
                    state["interpolate"] = 0 if state["interpolate"] else 1
                elif event.key == pygame.K_c:
                    balls.clear()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                if mx > SIM_W:
                    for s in sliders:
                        if s.hit(mx, my):
                            active_slider = s
                            s.update_from_mouse(mx)
                            break
                else:
                    if event.button == 1:
                        # Pick or create.
                        picked = None
                        for b in reversed(balls):
                            if b.hit(mx, my):
                                picked = b
                                break
                        if picked is None:
                            new = Ball(mx, my, r=90)
                            balls.append(new)
                            picked = new
                        dragging_ball = picked
                        drag_offset = (picked.x - mx, picked.y - my)
                    elif event.button == 3:
                        # Right click: delete a ball under the cursor.
                        for b in list(balls):
                            if b.hit(mx, my):
                                balls.remove(b)
                                break
                    elif event.button == 4:  # scroll up: enlarge nearest
                        target = _nearest_ball(balls, mx, my)
                        if target:
                            target.r = min(300.0, target.r * 1.12)
                    elif event.button == 5:  # scroll down: shrink nearest
                        target = _nearest_ball(balls, mx, my)
                        if target:
                            target.r = max(20.0, target.r / 1.12)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    if dragging_ball is not None:
                        # Give the ball a fresh random drift on release so it
                        # doesn't sit motionless after being held.
                        ang = random.uniform(0, math.tau)
                        spd = random.uniform(20.0, 60.0)
                        dragging_ball.vx = math.cos(ang) * spd
                        dragging_ball.vy = math.sin(ang) * spd
                    dragging_ball = None
                active_slider = None
            elif event.type == pygame.MOUSEMOTION:
                if active_slider:
                    active_slider.update_from_mouse(event.pos[0])
                if dragging_ball:
                    mx, my = event.pos
                    dragging_ball.x = mx + drag_offset[0]
                    dragging_ball.y = my + drag_offset[1]
                    # Stop drifting while held.
                    dragging_ball.vx = 0.0
                    dragging_ball.vy = 0.0

        # ---- Update ------------------------------------------------------
        if state["animate"]:
            for b in balls:
                if b is dragging_ball:
                    continue
                b.update(dt, sim_bounds)

        # Re-build sample grid if cell size changed.
        cs = state["cell_size"]
        if cs != cached_cs:
            cached_cs = cs
            sample_x = np.arange(0, SIM_W + cs, cs, dtype=np.float32)
            sample_y = np.arange(0, SIM_H + cs, cs, dtype=np.float32)
            field_buf = None  # Force reallocation below.
            # Surface for the rasterized fill, one pixel per cell.
            fill_lo = pygame.Surface((sample_x.shape[0] - 1,
                                       sample_y.shape[0] - 1))
            fill_lo_arr = pygame.surfarray.pixels3d(fill_lo)

        ball_tuples = [(b.x, b.y, b.r) for b in balls]
        field_buf = compute_field(ball_tuples, sample_x, sample_y, out=field_buf)
        threshold = state["threshold"]
        interp = bool(state["interpolate"])
        n_cells = (field_buf.shape[0] - 1) * (field_buf.shape[1] - 1)
        emit_polys = (n_cells <= 5000)
        segments, polygons = extract_contours(field_buf, sample_x, sample_y,
                                              threshold, interpolate=interp,
                                              emit_polygons=emit_polys)

        # ---- Render ------------------------------------------------------
        screen.fill(BG)

        # Pick a fill strategy: small grids -> draw per-cell polygons (smooth
        # edges from linear interpolation). Large grids -> rasterize the
        # binary inside-mask once and upscale (constant-time draw, blocky
        # edges that the contour line then hides).
        if emit_polys:
            for poly in polygons:
                if len(poly) >= 3:
                    pygame.draw.polygon(screen, INSIDE_FILL, poly)
        else:
            inside_mask = (field_buf[:-1, :-1] >= threshold)
            fill_lo_arr[:] = BG
            fill_lo_arr[inside_mask.T] = INSIDE_FILL
            scaled = pygame.transform.scale(fill_lo, (SIM_W, SIM_H))
            screen.blit(scaled, (0, 0))

        # Iso-contour line set.
        for p1, p2 in segments:
            pygame.draw.line(screen, CONTOUR, p1, p2, 2)

        # Sample grid dots.
        if state["show_grid"]:
            NY = field_buf.shape[0]
            NX = field_buf.shape[1]
            for j in range(NY):
                yy = float(sample_y[j])
                if yy < 0 or yy > SIM_H:
                    continue
                for i in range(NX):
                    xx = float(sample_x[i])
                    if xx < 0 or xx > SIM_W:
                        continue
                    val = field_buf[j, i]
                    color = GRID_DOT_INSIDE if val >= threshold else GRID_DOT_OUTSIDE
                    pygame.draw.circle(screen, color, (int(xx), int(yy)), 1 if cs > 10 else 1)

        # Ball outlines (visualize source positions).
        if state["show_balls"]:
            for b in balls:
                col = BALL_OUTLINE_DRAG if b is dragging_ball else BALL_OUTLINE
                pygame.draw.circle(screen, col, (int(b.x), int(b.y)), 5, 0)
                pygame.draw.circle(screen, col, (int(b.x), int(b.y)),
                                   int(max(8, b.r * 0.3)), 1)

        # Sim panel border.
        pygame.draw.rect(screen, PANEL_BORDER, (0, 0, SIM_W, SIM_H), 1)

        # ---- HUD --------------------------------------------------------
        pygame.draw.rect(screen, PANEL_BG, (SIM_W, 0, HUD_W, HEIGHT))
        pygame.draw.line(screen, PANEL_BORDER, (SIM_W, 0), (SIM_W, HEIGHT), 1)
        screen.blit(title_font.render("Marching Squares", True, ACCENT),
                    (SIM_W + 16, 14))

        screen.blit(small.render(f"FPS: {clock.get_fps():.0f}", True, TEXT_DIM),
                    (SIM_W + 16, 38))
        screen.blit(small.render(f"Balls: {len(balls)}    Cells: "
                                  f"{(field_buf.shape[1]-1)*(field_buf.shape[0]-1)}",
                                  True, TEXT_DIM), (SIM_W + 16, 54))
        screen.blit(small.render(f"Segments: {len(segments)}",
                                  True, TEXT_DIM), (SIM_W + 16, 70))

        for s in sliders:
            s.draw(screen, font)

        help_lines = [
            "Left click empty:  add ball",
            "Left click + drag: move ball",
            "Right click ball:  delete",
            "Wheel on ball:     resize",
            "Space: animate on/off",
            "G: sample grid    I: interp",
            "C: clear all balls",
            "Esc: quit",
            "",
            "Threshold defines the iso-",
            "surface; cell size controls",
            "smoothness vs. cost.",
        ]
        y = HEIGHT - 16 * len(help_lines) - 10
        for line in help_lines:
            screen.blit(small.render(line, True, TEXT_DIM), (SIM_W + 16, y))
            y += 16

        pygame.display.flip()


def _nearest_ball(balls, mx, my):
    best = None
    best_d2 = float("inf")
    for b in balls:
        dx = b.x - mx
        dy = b.y - my
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            best = b
    return best


if __name__ == "__main__":
    main()
