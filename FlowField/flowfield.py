"""
Flow-field pathfinding for many agents.

When N agents all want to reach the same goal on a grid, running A* N times
is wasteful. Instead we run ONE reverse Dijkstra from the goal: that gives
us the shortest-path cost to the goal at every walkable cell. Then for each
cell we look at its 8 neighbors and pick the one with the lowest cost — the
"flow direction" for that cell. Every agent then steers toward whatever
flow direction is at its current cell. Done.

Pipeline:
  1. Cost field: per-cell traversal cost (1 for grass, 4 for sand, ∞ wall)
  2. Integration field: Dijkstra from goal -> cumulative cost grid
  3. Flow field: for each cell, the direction (dx, dy) toward the lowest-
     cost neighbor (8-connected with sqrt(2) diagonal cost)
  4. Agents: steering forces using the flow field + light separation force
     so they don't pile up on top of each other

The whole field is recomputed only when:
  - the goal moves,
  - the cost field changes (you painted a wall or sand).
Agents query the field every frame, but the query is just an array lookup.
"""

import math
import sys
from collections import deque
import heapq

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

CELL = 20
COLS = SIM_W // CELL
ROWS = SIM_H // CELL

BG = (16, 20, 30)
PANEL_BG = (22, 26, 40)
PANEL_BORDER = (60, 70, 95)
TEXT = (220, 226, 240)
TEXT_DIM = (140, 150, 175)
ACCENT = (245, 210, 80)

GRID_LINE = (28, 34, 50)
WALL = (40, 50, 70)
SAND = (110, 95, 55)         # higher cost, but passable
GRASS = (40, 60, 75)
GOAL = (240, 230, 80)
AGENT = (130, 200, 240)
AGENT_GLOW = (200, 230, 255)
ARROW = (110, 130, 170)
HEATMAP_LO = (40, 60, 90)
HEATMAP_HI = (255, 110, 110)

SLIDER_TRACK = (50, 58, 80)
SLIDER_FILL = (130, 200, 240)
SLIDER_KNOB = (220, 240, 255)


# ---------------------------------------------------------------------------
# Field types
# ---------------------------------------------------------------------------
# Per-cell terrain cost. We use plain numpy arrays (ROWS, COLS).
WALL_COST = 1e9
GRASS_COST = 1.0
SAND_COST = 4.0


# 8-connected neighbor offsets and their step costs.
# Diagonal moves cost sqrt(2). We pre-multiply by the destination cell's
# cost so going INTO sand costs 4 * step.
NEIGHBORS = [
    (-1, -1, math.sqrt(2)),
    ( 0, -1, 1.0),
    ( 1, -1, math.sqrt(2)),
    (-1,  0, 1.0),
    ( 1,  0, 1.0),
    (-1,  1, math.sqrt(2)),
    ( 0,  1, 1.0),
    ( 1,  1, math.sqrt(2)),
]


def integrate_field(cost, goal):
    """Reverse Dijkstra from `goal`, returning the cumulative cost grid.

    Cells unreachable or behind walls keep cost = +inf.
    """
    rows, cols = cost.shape
    integ = np.full((rows, cols), np.inf, dtype=np.float64)
    gx, gy = goal
    if not (0 <= gx < cols and 0 <= gy < rows):
        return integ
    if cost[gy, gx] >= WALL_COST:
        return integ  # goal is inside a wall; nobody can reach it

    integ[gy, gx] = 0.0
    pq = [(0.0, gx, gy)]
    while pq:
        c, x, y = heapq.heappop(pq)
        if c > integ[y, x]:
            continue
        for dx, dy, step in NEIGHBORS:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < cols and 0 <= ny < rows):
                continue
            cell_cost = cost[ny, nx]
            if cell_cost >= WALL_COST:
                continue
            # Disallow corner-cutting through diagonally-adjacent walls.
            if dx != 0 and dy != 0:
                if cost[y, x + dx] >= WALL_COST and cost[y + dy, x] >= WALL_COST:
                    continue
            new_c = c + step * cell_cost
            if new_c < integ[ny, nx]:
                integ[ny, nx] = new_c
                heapq.heappush(pq, (new_c, nx, ny))
    return integ


def flow_field(integ):
    """Per-cell direction vector (dx, dy) pointing to the lowest-cost
    neighbor. Returns two float arrays of shape (rows, cols)."""
    rows, cols = integ.shape
    fx = np.zeros((rows, cols), dtype=np.float32)
    fy = np.zeros((rows, cols), dtype=np.float32)
    # For each of the 8 directions, build a shifted copy of integ and pick
    # the minimum. We track which direction won via argmin over a stacked
    # tensor.
    INF = np.float32(1e18)
    stacked = np.full((8, rows, cols), INF, dtype=np.float32)
    dirs = []
    for k, (dx, dy, _) in enumerate(NEIGHBORS):
        # Slice source/destination ranges so we don't go out of bounds.
        src_x0 = max(0, -dx);  src_x1 = cols - max(0, dx)
        src_y0 = max(0, -dy);  src_y1 = rows - max(0, dy)
        dst_x0 = max(0,  dx);  dst_x1 = cols - max(0, -dx)
        dst_y0 = max(0,  dy);  dst_y1 = rows - max(0, -dy)
        stacked[k, src_y0:src_y1, src_x0:src_x1] = \
            integ[dst_y0:dst_y1, dst_x0:dst_x1]
        # Normalize the direction vector here so flow magnitude is 1.
        length = math.hypot(dx, dy)
        dirs.append((dx / length, dy / length))

    best = np.argmin(stacked, axis=0)
    # Where this cell's own integration cost is +inf, leave the flow as 0.
    valid = np.isfinite(integ)
    for k, (dxn, dyn) in enumerate(dirs):
        m = (best == k) & valid
        fx[m] = dxn
        fy[m] = dyn
    return fx, fy


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------


class Agents:
    """Pile of agents stored in numpy SoA. They steer by sampling the flow
    field at their current cell, plus a light separation force so they
    don't merge into one giant blob."""

    def __init__(self, capacity=2000):
        self.cap = capacity
        self.pos = np.zeros((0, 2), dtype=np.float32)
        self.vel = np.zeros((0, 2), dtype=np.float32)

    @property
    def n(self):
        return self.pos.shape[0]

    def add(self, x, y):
        self.pos = np.vstack([self.pos, [[x, y]]]).astype(np.float32)
        self.vel = np.vstack([self.vel, [[0.0, 0.0]]]).astype(np.float32)

    def add_many(self, xs, ys):
        new_pos = np.column_stack([xs, ys]).astype(np.float32)
        self.pos = np.vstack([self.pos, new_pos]) if self.n else new_pos
        new_vel = np.zeros_like(new_pos)
        self.vel = np.vstack([self.vel, new_vel]) if self.vel.shape[0] else new_vel

    def update(self, fx, fy, cost, dt, max_speed, separation, agent_radius):
        if self.n == 0:
            return
        rows, cols = fx.shape

        # Sample the flow field at each agent's current cell.
        cx = np.clip((self.pos[:, 0] / CELL).astype(np.int32), 0, cols - 1)
        cy = np.clip((self.pos[:, 1] / CELL).astype(np.int32), 0, rows - 1)
        steer_x = fx[cy, cx]
        steer_y = fy[cy, cx]

        # Light separation: for each agent, push away from neighbors that
        # are closer than `agent_radius * 2`. We do an N*N broadcast which
        # is fine up to ~1000 agents on this hardware.
        if self.n > 1 and separation > 0:
            dx = self.pos[:, 0:1] - self.pos[:, 0:1].T
            dy = self.pos[:, 1:2] - self.pos[:, 1:2].T
            d2 = dx * dx + dy * dy
            r = float(agent_radius * 2.0)
            mask = (d2 > 0) & (d2 < r * r)
            inv = np.where(mask, 1.0 / np.sqrt(np.where(d2 > 0, d2, 1.0)), 0.0)
            # Each pair contributes a unit-length push from j toward i.
            push_x = (dx * inv).sum(axis=1)
            push_y = (dy * inv).sum(axis=1)
            steer_x += push_x * separation
            steer_y += push_y * separation

        # Integrate with mild inertia for a less jittery look.
        target_vx = steer_x * max_speed
        target_vy = steer_y * max_speed
        blend = 1.0 - math.exp(-8.0 * dt)
        self.vel[:, 0] += (target_vx - self.vel[:, 0]) * blend
        self.vel[:, 1] += (target_vy - self.vel[:, 1]) * blend

        # Agents stuck in walls (cost == inf) just stop instead of jittering.
        in_wall = (cost[cy, cx] >= WALL_COST)
        if in_wall.any():
            self.vel[in_wall] *= 0.5

        # Integrate position.
        self.pos[:, 0] += self.vel[:, 0] * dt
        self.pos[:, 1] += self.vel[:, 1] * dt

        # Hard wall collision: if agent crosses into a wall cell, project it
        # back along the velocity direction to the cell boundary. Cheap and
        # works because cells are axis-aligned.
        cx2 = np.clip((self.pos[:, 0] / CELL).astype(np.int32), 0, cols - 1)
        cy2 = np.clip((self.pos[:, 1] / CELL).astype(np.int32), 0, rows - 1)
        in_wall_now = (cost[cy2, cx2] >= WALL_COST)
        if in_wall_now.any():
            # Snap back to the cell boundary the agent came from. We push
            # the agent away from the wall by `agent_radius`.
            self.pos[in_wall_now, 0] = (cx[in_wall_now] + 0.5) * CELL
            self.pos[in_wall_now, 1] = (cy[in_wall_now] + 0.5) * CELL
            self.vel[in_wall_now] = 0.0

        # Clamp to play area.
        np.clip(self.pos[:, 0], 1, SIM_W - 1, out=self.pos[:, 0])
        np.clip(self.pos[:, 1], 1, SIM_H - 1, out=self.pos[:, 1])


# ---------------------------------------------------------------------------
# Slider UI
# ---------------------------------------------------------------------------


class Slider:
    HEIGHT = 14
    LABEL_GAP = 18

    def __init__(self, x, y, w, label, lo, hi, step, getter, setter, fmt=None):
        self.label = label
        self.lo = lo; self.hi = hi; self.step = step
        self.getter = getter; self.setter = setter
        self.fmt = fmt
        self.rect = pygame.Rect(x, y + Slider.LABEL_GAP, w, Slider.HEIGHT)

    def draw(self, surface, font):
        v = self.getter()
        if self.fmt is not None: label = f"{self.label}: {self.fmt(v)}"
        elif v == int(v):        label = f"{self.label}: {int(v)}"
        else:                    label = f"{self.label}: {round(v, 2)}"
        surface.blit(font.render(label, True, TEXT),
                     (self.rect.x, self.rect.y - Slider.LABEL_GAP + 1))
        pygame.draw.rect(surface, SLIDER_TRACK, self.rect, border_radius=4)
        t = (v - self.lo) / (self.hi - self.lo)
        knob_x = int(self.rect.x + t * self.rect.w)
        fill = pygame.Rect(self.rect.x, self.rect.y, knob_x - self.rect.x, self.rect.h)
        pygame.draw.rect(surface, SLIDER_FILL, fill, border_radius=4)
        pygame.draw.circle(surface, SLIDER_KNOB, (knob_x, self.rect.centery), 7)

    def hit(self, mx, my):
        return (self.rect.x - 4 <= mx <= self.rect.right + 4
                and self.rect.y - 6 <= my <= self.rect.bottom + 6)

    def update_from_mouse(self, mx):
        t = max(0.0, min(1.0, (mx - self.rect.x) / self.rect.w))
        v = self.lo + t * (self.hi - self.lo)
        v = round(v / self.step) * self.step
        if self.step >= 1: v = int(v)
        self.setter(v)


# ---------------------------------------------------------------------------
# Initial map
# ---------------------------------------------------------------------------


def make_default_cost():
    cost = np.full((ROWS, COLS), GRASS_COST, dtype=np.float32)
    # A few walls forming a maze-like obstacle.
    for x in range(8, 28):
        cost[10, x] = WALL_COST
    for y in range(10, 24):
        cost[y, 28] = WALL_COST
    for x in range(20, 38):
        cost[24, x] = WALL_COST
    for y in range(6, 16):
        cost[y, 40] = WALL_COST
    # A patch of sand (slow terrain).
    for y in range(20, 32):
        for x in range(35, 50):
            if cost[y, x] != WALL_COST:
                cost[y, x] = SAND_COST
    return cost


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------


def render_terrain(surface, cost):
    """Pre-render terrain background to a surface (cached, since cost rarely
    changes mid-frame)."""
    surface.fill(GRASS)
    for y in range(ROWS):
        for x in range(COLS):
            c = cost[y, x]
            if c >= WALL_COST:
                pygame.draw.rect(surface, WALL,
                                 (x * CELL, y * CELL, CELL, CELL))
            elif c >= SAND_COST - 0.01:
                pygame.draw.rect(surface, SAND,
                                 (x * CELL, y * CELL, CELL, CELL))
    # Subtle grid lines.
    for x in range(0, SIM_W, CELL):
        pygame.draw.line(surface, GRID_LINE, (x, 0), (x, SIM_H), 1)
    for y in range(0, SIM_H, CELL):
        pygame.draw.line(surface, GRID_LINE, (0, y), (SIM_W, y), 1)


def render_heatmap(surface, integ):
    """Faint overlay coloring cells by their integration cost."""
    finite = np.isfinite(integ)
    if not finite.any():
        return
    lo = 0.0
    hi = float(integ[finite].max())
    if hi <= 0:
        return
    norm = np.zeros_like(integ, dtype=np.float32)
    norm[finite] = (integ[finite] - lo) / (hi - lo)
    overlay = pygame.Surface((COLS, ROWS), pygame.SRCALPHA)
    arr = pygame.surfarray.pixels3d(overlay)
    alpha = pygame.surfarray.pixels_alpha(overlay)
    # Numpy color blend: low (blue-ish) to high (red-ish).
    lo_col = np.array(HEATMAP_LO, dtype=np.float32)
    hi_col = np.array(HEATMAP_HI, dtype=np.float32)
    color = lo_col[None, None, :] * (1 - norm[..., None]) + hi_col[None, None, :] * norm[..., None]
    # surfarray pixels3d is (W, H, 3) — note the transpose.
    arr[...] = color.transpose(1, 0, 2).astype(np.uint8)
    a = np.where(finite, 90, 0).astype(np.uint8).T
    alpha[...] = a
    del arr, alpha
    scaled = pygame.transform.scale(overlay, (SIM_W, SIM_H))
    surface.blit(scaled, (0, 0))


def render_arrows(surface, fx, fy, integ):
    """Tiny arrow per cell pointing along the flow."""
    rows, cols = fx.shape
    for y in range(rows):
        for x in range(cols):
            if not math.isfinite(integ[y, x]):
                continue
            dx = float(fx[y, x]); dy = float(fy[y, x])
            if dx == 0 and dy == 0:
                continue
            cxp = x * CELL + CELL // 2
            cyp = y * CELL + CELL // 2
            ex = cxp + int(dx * (CELL * 0.4))
            ey = cyp + int(dy * (CELL * 0.4))
            pygame.draw.line(surface, ARROW, (cxp, cyp), (ex, ey), 1)
            pygame.draw.circle(surface, ARROW, (ex, ey), 1)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    pygame.init()
    pygame.display.set_caption("Flow-Field Pathfinding")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Menlo,Consolas,monospace", 13)
    small = pygame.font.SysFont("Menlo,Consolas,monospace", 11)
    title_font = pygame.font.SysFont("Menlo,Consolas,monospace", 16, bold=True)

    state = {
        "max_speed":   140.0,
        "separation":  60.0,
        "agent_count": 250,
        "show_heatmap": 1,
        "show_arrows":  1,
        "paint_mode":   1,    # 1=wall, 2=sand, 3=grass(erase)
    }

    cost = make_default_cost()
    goal = (COLS - 4, ROWS // 2)
    integ = integrate_field(cost, goal)
    fx, fy = flow_field(integ)

    agents = Agents()
    # Spawn initial agents on the left side.
    rng = np.random.default_rng(1)
    xs = rng.uniform(20, 6 * CELL, state["agent_count"])
    ys = rng.uniform(20, SIM_H - 20, state["agent_count"])
    agents.add_many(xs, ys)

    # Cached terrain surface.
    terrain_surf = pygame.Surface((SIM_W, SIM_H))
    render_terrain(terrain_surf, cost)

    # Sliders.
    pad = 16
    sx = SIM_W + pad
    sw = HUD_W - 2 * pad
    sliders = [
        Slider(sx,  90, sw, "Max speed (px/s)", 30, 400, 5,
               lambda: state["max_speed"], lambda v: state.update(max_speed=float(v))),
        Slider(sx, 132, sw, "Separation",       0, 200, 5,
               lambda: state["separation"], lambda v: state.update(separation=float(v))),
        Slider(sx, 174, sw, "Agent count",     0, 1500, 25,
               lambda: state["agent_count"], lambda v: state.update(agent_count=int(v))),
        Slider(sx, 216, sw, "Show heatmap",    0, 1, 1,
               lambda: state["show_heatmap"], lambda v: state.update(show_heatmap=int(v))),
        Slider(sx, 258, sw, "Show flow arrows", 0, 1, 1,
               lambda: state["show_arrows"], lambda v: state.update(show_arrows=int(v))),
    ]

    active_slider = None
    last_count = state["agent_count"]
    paint_active = False
    field_dirty = False

    def repaint_field():
        nonlocal integ, fx, fy
        integ = integrate_field(cost, goal)
        fx, fy = flow_field(integ)
        render_terrain(terrain_surf, cost)

    def cell_at(mx, my):
        return (mx // CELL, my // CELL)

    def paint_at(mx, my):
        nonlocal field_dirty
        cx, cy = cell_at(mx, my)
        if not (0 <= cx < COLS and 0 <= cy < ROWS):
            return
        if state["paint_mode"] == 1:
            cost[cy, cx] = WALL_COST
        elif state["paint_mode"] == 2:
            cost[cy, cx] = SAND_COST
        else:
            cost[cy, cx] = GRASS_COST
        field_dirty = True

    while True:
        dt_ms = clock.tick(FPS)
        dt = min(1.0 / 30.0, dt_ms / 1000.0)

        # ---- Events --------------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit(0)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit(0)
                elif event.key == pygame.K_1:
                    state["paint_mode"] = 1  # wall
                elif event.key == pygame.K_2:
                    state["paint_mode"] = 2  # sand
                elif event.key == pygame.K_3:
                    state["paint_mode"] = 3  # erase
                elif event.key == pygame.K_h:
                    state["show_heatmap"] = 0 if state["show_heatmap"] else 1
                elif event.key == pygame.K_a:
                    state["show_arrows"] = 0 if state["show_arrows"] else 1
                elif event.key == pygame.K_r:
                    cost[...] = GRASS_COST
                    field_dirty = True
                elif event.key == pygame.K_c:
                    agents.pos = np.zeros((0, 2), dtype=np.float32)
                    agents.vel = np.zeros((0, 2), dtype=np.float32)
                    state["agent_count"] = 0
                    last_count = 0
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
                        # Left click in sim sets the goal.
                        cx, cy = cell_at(mx, my)
                        if 0 <= cx < COLS and 0 <= cy < ROWS and cost[cy, cx] < WALL_COST:
                            goal = (int(cx), int(cy))
                            field_dirty = True
                    elif event.button == 3:
                        paint_active = True
                        paint_at(mx, my)
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 3:
                    paint_active = False
                active_slider = None
            elif event.type == pygame.MOUSEMOTION:
                if active_slider:
                    active_slider.update_from_mouse(event.pos[0])
                if paint_active and event.pos[0] < SIM_W:
                    paint_at(*event.pos)

        # If agent count slider moved, add or trim agents.
        if state["agent_count"] != last_count:
            target = state["agent_count"]
            if target > agents.n:
                more = target - agents.n
                xs = rng.uniform(20, 6 * CELL, more)
                ys = rng.uniform(20, SIM_H - 20, more)
                agents.add_many(xs, ys)
            else:
                keep = target
                agents.pos = agents.pos[:keep]
                agents.vel = agents.vel[:keep]
            last_count = state["agent_count"]

        # ---- Recompute field if dirty ---------------------------------
        if field_dirty:
            repaint_field()
            field_dirty = False

        # ---- Step agents ---------------------------------------------
        agents.update(fx, fy, cost, dt,
                      state["max_speed"], state["separation"] / 100.0,
                      agent_radius=CELL * 0.45)

        # ---- Render ---------------------------------------------------
        screen.blit(terrain_surf, (0, 0))
        if state["show_heatmap"]:
            render_heatmap(screen, integ)
        if state["show_arrows"]:
            render_arrows(screen, fx, fy, integ)

        # Goal.
        gx, gy = goal
        rect = pygame.Rect(gx * CELL + 2, gy * CELL + 2, CELL - 4, CELL - 4)
        pygame.draw.rect(screen, GOAL, rect, border_radius=4)
        pygame.draw.rect(screen, (255, 255, 255), rect, 1, border_radius=4)

        # Agents.
        if agents.n > 0:
            for i in range(agents.n):
                px = int(agents.pos[i, 0])
                py = int(agents.pos[i, 1])
                pygame.draw.circle(screen, AGENT, (px, py), 4)

        pygame.draw.rect(screen, PANEL_BORDER, (0, 0, SIM_W, SIM_H), 1)

        # ---- HUD -----------------------------------------------------
        pygame.draw.rect(screen, PANEL_BG, (SIM_W, 0, HUD_W, HEIGHT))
        pygame.draw.line(screen, PANEL_BORDER, (SIM_W, 0), (SIM_W, HEIGHT), 1)
        screen.blit(title_font.render("Flow-Field Pathfinding", True, ACCENT),
                    (SIM_W + 16, 14))

        mode_label = {1: "WALL", 2: "SAND", 3: "ERASE"}[state["paint_mode"]]
        screen.blit(small.render(f"FPS: {clock.get_fps():.0f}", True, TEXT_DIM),
                    (SIM_W + 16, 38))
        screen.blit(small.render(f"Agents: {agents.n}    Paint: {mode_label}",
                                  True, TEXT_DIM), (SIM_W + 16, 54))
        screen.blit(small.render(f"Goal: ({goal[0]}, {goal[1]})",
                                  True, TEXT_DIM), (SIM_W + 16, 70))

        for s in sliders:
            s.draw(screen, font)

        help_lines = [
            "Left click: set goal",
            "Right click + drag:",
            "  paint terrain",
            "1 / 2 / 3: paint mode",
            "  (wall / sand / erase)",
            "H: heatmap   A: arrows",
            "R: clear all walls",
            "C: clear all agents",
            "Esc: quit",
            "",
            "Heatmap = distance to",
            "goal. Arrows = local",
            "flow direction.",
        ]
        y = HEIGHT - 16 * len(help_lines) - 10
        for line in help_lines:
            screen.blit(small.render(line, True, TEXT_DIM), (SIM_W + 16, y))
            y += 16

        pygame.display.flip()


if __name__ == "__main__":
    main()
