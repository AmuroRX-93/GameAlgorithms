"""
Verlet integration with distance constraints.

This is the algorithm Thomas Jakobsen described in his classic Hitman:
Codename 47 paper. The trick is to skip storing per-particle velocity:

    new_pos = pos + (pos - prev_pos) * damping + acceleration * dt^2
    prev_pos = pos
    pos = new_pos

The expression `(pos - prev_pos)` IS the velocity (implicitly), so motion
just falls out of the integration step. Constraints are then enforced by
relaxation: for each rigid edge of length L between particles A and B, we
compute the current distance D, the error (D - L), and shove A and B
toward each other (or apart) by half the error each. Doing this for every
constraint several times per frame causes the whole system to converge
toward a globally valid configuration. The more iterations, the stiffer
the system feels.

Demo: a hanging cloth pinned at the top, plus a few free rigid bodies
(squares built from 4 particles and 6 constraints). You can drag any
node, or right-click-drag to slice constraints apart like cutting cloth.
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

CLOTH_LINE = (130, 200, 240)
CLOTH_PINNED = (245, 210, 80)
CLOTH_DRAG = (255, 255, 255)
RIGID_LINE = (245, 170, 90)
RIGID_FILL = (90, 60, 35)
SLICE_TRAIL = (240, 110, 110)
GROUND = (40, 50, 70)

SLIDER_TRACK = (50, 58, 80)
SLIDER_FILL = (130, 200, 240)
SLIDER_KNOB = (220, 240, 255)


# ---------------------------------------------------------------------------
# Particle pool (Structure of Arrays so the integrator stays vectorized)
# ---------------------------------------------------------------------------


class World:
    """All particles live in one set of numpy arrays.

    Constraints reference particles by index. Removing particles isn't
    supported — instead we mark constraints as inactive when they're cut.
    This keeps the array-based integrator simple.
    """

    def __init__(self):
        # Particle state: position, previous position, inverse mass, pinned flag.
        self.pos = np.zeros((0, 2), dtype=np.float64)
        self.prev = np.zeros((0, 2), dtype=np.float64)
        self.inv_mass = np.zeros((0,), dtype=np.float64)
        self.pinned = np.zeros((0,), dtype=bool)

        # Constraints: parallel arrays for two endpoints, rest length, alive flag.
        self.c_a = []
        self.c_b = []
        self.c_rest = []
        self.c_alive = []
        # Optional grouping for pretty rendering — each constraint belongs to a "body".
        self.c_body = []

        # Cached numpy views of the constraint arrays (rebuilt lazily).
        self._cached = False
        self._a = None
        self._b = None
        self._rest = None
        self._alive_mask = None

    # ---- Particle creation ---------------------------------------------

    def add_particle(self, x, y, mass=1.0, pinned=False):
        idx = self.pos.shape[0]
        self.pos = np.vstack([self.pos, [[x, y]]])
        self.prev = np.vstack([self.prev, [[x, y]]])
        inv = 0.0 if pinned else 1.0 / mass
        self.inv_mass = np.append(self.inv_mass, inv)
        self.pinned = np.append(self.pinned, pinned)
        return idx

    def add_constraint(self, a, b, rest=None, body=0):
        if rest is None:
            dx = self.pos[a, 0] - self.pos[b, 0]
            dy = self.pos[a, 1] - self.pos[b, 1]
            rest = math.hypot(dx, dy)
        self.c_a.append(a)
        self.c_b.append(b)
        self.c_rest.append(rest)
        self.c_alive.append(True)
        self.c_body.append(body)
        self._cached = False

    def cut_constraint(self, k):
        if 0 <= k < len(self.c_alive) and self.c_alive[k]:
            self.c_alive[k] = False
            self._cached = False

    def _rebuild_cache(self):
        self._a = np.asarray(self.c_a, dtype=np.int64)
        self._b = np.asarray(self.c_b, dtype=np.int64)
        self._rest = np.asarray(self.c_rest, dtype=np.float64)
        self._alive_mask = np.asarray(self.c_alive, dtype=bool)
        self._cached = True

    # ---- Integration ---------------------------------------------------

    def integrate(self, gravity, wind, damping, dt):
        """Verlet step:  new = pos + (pos - prev) * damping + a * dt^2"""
        if self.pos.shape[0] == 0:
            return
        # acceleration from gravity + wind, applied uniformly per unit mass
        ax = wind
        ay = gravity
        free = ~self.pinned
        # velocity (implicit) = pos - prev
        vel = (self.pos - self.prev) * damping
        new_pos = self.pos.copy()
        new_pos[free, 0] = self.pos[free, 0] + vel[free, 0] + ax * dt * dt
        new_pos[free, 1] = self.pos[free, 1] + vel[free, 1] + ay * dt * dt
        self.prev = self.pos.copy()
        self.pos = new_pos

    # ---- Constraint relaxation ----------------------------------------
    #
    # Distance constraint: for each edge (a, b) with rest length L, push
    # endpoints along the edge so that |pos[a] - pos[b]| == L.
    # We process all edges once, then repeat `iterations` times. More
    # iterations -> stiffer rope/cloth.

    def satisfy_constraints(self, iterations):
        if self.pos.shape[0] == 0 or not self.c_a:
            return
        if not self._cached:
            self._rebuild_cache()
        a = self._a
        b = self._b
        rest = self._rest
        alive = self._alive_mask
        if not alive.any():
            return
        # Pre-extract live constraints; we still need an in-place loop because
        # particle updates depend on prior updates (Gauss-Seidel relaxation).
        a_live = a[alive]
        b_live = b[alive]
        rest_live = rest[alive]
        pos = self.pos
        inv_mass = self.inv_mass
        pinned = self.pinned

        # Gauss-Seidel relaxation: update each constraint sequentially so the
        # next constraint sees the updated positions. This converges far
        # faster and more stably than the vectorized Jacobi version (which
        # can overshoot and let cloth balloon out under gravity).
        #
        # We keep the hot loop in pure Python ints/floats by pre-converting
        # the index arrays. Direct ndarray element access (`pos[i, 0]`) is
        # the bottleneck; using a 1D flat view halves the call overhead.
        n_constraints = a_live.shape[0]
        a_py = a_live.tolist()
        b_py = b_live.tolist()
        rest_py = rest_live.tolist()
        # Flat views: pos_flat[2*i + 0] = pos[i, 0], pos_flat[2*i + 1] = pos[i, 1].
        pos_flat = pos.reshape(-1)
        inv_mass_py = inv_mass.tolist()
        sqrt = math.sqrt
        for _ in range(iterations):
            for k in range(n_constraints):
                ai = a_py[k]
                bi = b_py[k]
                rest_k = rest_py[k]
                ai2 = ai + ai  # ai * 2
                bi2 = bi + bi
                ax = pos_flat[ai2]; ay = pos_flat[ai2 + 1]
                bx = pos_flat[bi2]; by = pos_flat[bi2 + 1]
                dx = bx - ax; dy = by - ay
                d2 = dx * dx + dy * dy
                if d2 < 1e-12:
                    continue
                d = sqrt(d2)
                err = (d - rest_k) / d
                ima = inv_mass_py[ai]
                imb = inv_mass_py[bi]
                inv_sum = ima + imb
                if inv_sum < 1e-9:
                    continue
                # Per-particle correction = (delta * err) * (im / inv_sum).
                wa = ima / inv_sum
                wb = imb / inv_sum
                cx = dx * err
                cy = dy * err
                if ima > 0.0:
                    pos_flat[ai2]     = ax + cx * wa
                    pos_flat[ai2 + 1] = ay + cy * wa
                if imb > 0.0:
                    pos_flat[bi2]     = bx - cx * wb
                    pos_flat[bi2 + 1] = by - cy * wb

    # ---- World collisions ---------------------------------------------

    def collide_bounds(self, x0, y0, x1, y1):
        """Clamp particles to the simulation rectangle. The "ground" gets a
        bit of friction so things settle instead of sliding forever."""
        if self.pos.shape[0] == 0:
            return
        pos = self.pos
        prev = self.prev
        # Left wall.
        m = pos[:, 0] < x0
        pos[m, 0] = x0
        # Right wall.
        m = pos[:, 0] > x1
        pos[m, 0] = x1
        # Ceiling.
        m = pos[:, 1] < y0
        pos[m, 1] = y0
        # Ground (with friction: blend prev x toward pos x).
        m = pos[:, 1] > y1
        if m.any():
            pos[m, 1] = y1
            # Friction: drag horizontal velocity toward zero.
            prev[m, 0] = pos[m, 0] - (pos[m, 0] - prev[m, 0]) * 0.5


# ---------------------------------------------------------------------------
# Scene builders
# ---------------------------------------------------------------------------


def build_cloth(world, x, y, cols, rows, spacing, body_id):
    """A grid of particles connected by horizontal & vertical springs.

    The first row is pinned, plus a small gap in the middle to make tearing
    look more dramatic when you slice the top.
    """
    indices = [[0] * cols for _ in range(rows)]
    for j in range(rows):
        for i in range(cols):
            px = x + i * spacing
            py = y + j * spacing
            pin = (j == 0) and (i % max(1, cols // 6) == 0 or i == 0 or i == cols - 1)
            indices[j][i] = world.add_particle(px, py, mass=1.0, pinned=pin)
    # Edges.
    for j in range(rows):
        for i in range(cols):
            if i + 1 < cols:
                world.add_constraint(indices[j][i], indices[j][i + 1], body=body_id)
            if j + 1 < rows:
                world.add_constraint(indices[j][i], indices[j + 1][i], body=body_id)
    return indices


def build_rigid_box(world, cx, cy, size, body_id):
    """A square built from 4 corner particles + 4 sides + 2 diagonals.

    Diagonals make it a proper rigid body: without them the square would
    collapse into a parallelogram immediately.
    """
    s = size * 0.5
    p0 = world.add_particle(cx - s, cy - s)
    p1 = world.add_particle(cx + s, cy - s)
    p2 = world.add_particle(cx + s, cy + s)
    p3 = world.add_particle(cx - s, cy + s)
    sides = [(p0, p1), (p1, p2), (p2, p3), (p3, p0)]
    diags = [(p0, p2), (p1, p3)]
    for a, b in sides + diags:
        world.add_constraint(a, b, body=body_id)
    return (p0, p1, p2, p3)


def build_rope(world, x0, y0, x1, y1, segments, body_id, pin_first=True):
    """A rope of N particles connecting (x0,y0) to (x1,y1)."""
    indices = []
    for k in range(segments + 1):
        t = k / segments
        px = x0 + (x1 - x0) * t
        py = y0 + (y1 - y0) * t
        pinned = pin_first and k == 0
        indices.append(world.add_particle(px, py, pinned=pinned))
    for k in range(segments):
        world.add_constraint(indices[k], indices[k + 1], body=body_id)
    return indices


# ---------------------------------------------------------------------------
# Slice tool: cut any constraint that crosses a recent mouse path
# ---------------------------------------------------------------------------


def segments_intersect(p1, p2, p3, p4):
    """True if segment p1-p2 intersects p3-p4 (proper or touching)."""
    def ccw(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    d1 = ccw(p3, p4, p1)
    d2 = ccw(p3, p4, p2)
    d3 = ccw(p1, p2, p3)
    d4 = ccw(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


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
# Scene + main loop
# ---------------------------------------------------------------------------


def reset_scene(state):
    """Build the world from scratch using the current state values."""
    w = World()
    body_id = 1
    cols = state["cloth_cols"]
    rows = state["cloth_rows"]
    spacing = 18
    cloth_w = (cols - 1) * spacing
    cloth_x = (SIM_W - cloth_w) // 2
    cloth_y = 60
    cloth = build_cloth(w, cloth_x, cloth_y, cols, rows, spacing, body_id)
    body_id += 1

    # A short rope on the left and a couple of rigid boxes that will fall.
    build_rope(w, 80, 60, 80, 320, 18, body_id, pin_first=True)
    body_id += 1
    build_rigid_box(w, 220, 90, 60, body_id); body_id += 1
    build_rigid_box(w, 800, 70, 70, body_id); body_id += 1
    build_rigid_box(w, 880, 60, 50, body_id); body_id += 1
    return w


def main():
    pygame.init()
    pygame.display.set_caption("Verlet Cloth & Rigid Bodies")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Menlo,Consolas,monospace", 13)
    small = pygame.font.SysFont("Menlo,Consolas,monospace", 11)
    title_font = pygame.font.SysFont("Menlo,Consolas,monospace", 16, bold=True)

    state = {
        "gravity": 900.0,    # px / s^2
        "wind":     0.0,
        "damping":  0.992,   # multiplier on (pos - prev) each step
        "iters":    6,       # constraint relaxation iterations
        "cloth_cols": 36,
        "cloth_rows": 22,
        "show_nodes": 0,
        "paused": 0,
    }

    world = reset_scene(state)

    # Sliders.
    pad = 16
    sx = SIM_W + pad
    sw = HUD_W - 2 * pad
    sliders = [
        Slider(sx, 90,  sw, "Gravity", 0, 2500, 25,
               lambda: state["gravity"], lambda v: state.update(gravity=float(v))),
        Slider(sx, 132, sw, "Wind",    -800, 800, 10,
               lambda: state["wind"], lambda v: state.update(wind=float(v))),
        Slider(sx, 174, sw, "Damping", 0.90, 1.00, 0.001,
               lambda: state["damping"], lambda v: state.update(damping=float(v)),
               fmt=lambda v: f"{v:.3f}"),
        Slider(sx, 216, sw, "Iterations", 1, 30, 1,
               lambda: state["iters"], lambda v: state.update(iters=int(v))),
        Slider(sx, 258, sw, "Cloth cols", 8, 60, 1,
               lambda: state["cloth_cols"], lambda v: state.update(cloth_cols=int(v))),
        Slider(sx, 300, sw, "Cloth rows", 4, 40, 1,
               lambda: state["cloth_rows"], lambda v: state.update(cloth_rows=int(v))),
        Slider(sx, 342, sw, "Show nodes", 0, 1, 1,
               lambda: state["show_nodes"], lambda v: state.update(show_nodes=int(v))),
    ]

    active_slider = None
    drag_idx = None             # Index of particle being dragged.
    drag_offset = (0.0, 0.0)
    slice_active = False
    last_slice_pos = None
    last_dims = (state["cloth_cols"], state["cloth_rows"])

    sim_bounds = (0, 0, SIM_W, SIM_H - 4)

    # Visual: keep a short trail of slice-tool positions for a swoosh.
    slice_trail = []

    while True:
        dt_ms = clock.tick(FPS)
        dt = min(1.0 / 30.0, dt_ms / 1000.0)

        # ---- Events ----------------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit(0)
                elif event.key == pygame.K_SPACE:
                    state["paused"] = 0 if state["paused"] else 1
                elif event.key == pygame.K_r:
                    world = reset_scene(state)
                    drag_idx = None
                elif event.key == pygame.K_b:
                    # Drop a new box at the cursor.
                    mx, my = pygame.mouse.get_pos()
                    if mx < SIM_W:
                        body_id = max(world.c_body) + 1 if world.c_body else 1
                        build_rigid_box(world, mx, my, random.uniform(40, 80), body_id)
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
                        # Pick the nearest particle within a radius.
                        if world.pos.shape[0]:
                            d2 = (world.pos[:, 0] - mx) ** 2 + (world.pos[:, 1] - my) ** 2
                            k = int(np.argmin(d2))
                            if d2[k] < 30 ** 2:
                                drag_idx = k
                                drag_offset = (world.pos[k, 0] - mx, world.pos[k, 1] - my)
                    elif event.button == 3:
                        slice_active = True
                        last_slice_pos = (mx, my)
                        slice_trail = [(mx, my)]
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    drag_idx = None
                elif event.button == 3:
                    slice_active = False
                    last_slice_pos = None
                active_slider = None
            elif event.type == pygame.MOUSEMOTION:
                if active_slider:
                    active_slider.update_from_mouse(event.pos[0])
                if drag_idx is not None:
                    mx, my = event.pos
                    nx = mx + drag_offset[0]
                    ny = my + drag_offset[1]
                    world.pos[drag_idx, 0] = nx
                    world.pos[drag_idx, 1] = ny
                    # Reset prev so we don't fling the particle on release.
                    world.prev[drag_idx, 0] = nx
                    world.prev[drag_idx, 1] = ny
                if slice_active:
                    mx, my = event.pos
                    if last_slice_pos is not None:
                        # Cut any alive constraint whose endpoints' segment
                        # crosses the mouse-move segment.
                        p1 = last_slice_pos
                        p2 = (mx, my)
                        for k, alive in enumerate(world.c_alive):
                            if not alive:
                                continue
                            a = world.c_a[k]
                            b = world.c_b[k]
                            pa = (world.pos[a, 0], world.pos[a, 1])
                            pb = (world.pos[b, 0], world.pos[b, 1])
                            if segments_intersect(p1, p2, pa, pb):
                                world.cut_constraint(k)
                    last_slice_pos = (mx, my)
                    slice_trail.append((mx, my))
                    if len(slice_trail) > 24:
                        slice_trail.pop(0)

        # If cloth dims changed via sliders, rebuild.
        if (state["cloth_cols"], state["cloth_rows"]) != last_dims:
            last_dims = (state["cloth_cols"], state["cloth_rows"])
            world = reset_scene(state)
            drag_idx = None

        # ---- Update ----------------------------------------------------
        if not state["paused"]:
            # Single full-dt step is fine with Gauss-Seidel relaxation; we
            # used to substep here but the new solver converges quickly
            # enough that two substeps were just doing twice the work for
            # the same visual quality.
            world.integrate(state["gravity"], state["wind"],
                            state["damping"], dt)
            world.satisfy_constraints(state["iters"])
            world.collide_bounds(*sim_bounds)

        # Decay slice trail.
        if not slice_active and slice_trail:
            slice_trail = slice_trail[-12:]
            if len(slice_trail) > 0:
                slice_trail = slice_trail[1:] if random.random() < 0.5 else slice_trail

        # ---- Render ----------------------------------------------------
        screen.fill(BG)
        # Ground line.
        pygame.draw.line(screen, GROUND, (0, SIM_H - 3), (SIM_W, SIM_H - 3), 2)

        # Constraints (cloth = blue, rigid = orange).
        for k, alive in enumerate(world.c_alive):
            if not alive:
                continue
            a = world.c_a[k]
            b = world.c_b[k]
            ax, ay = world.pos[a]
            bx, by = world.pos[b]
            body = world.c_body[k]
            color = CLOTH_LINE if body == 1 else RIGID_LINE
            pygame.draw.line(screen, color, (ax, ay), (bx, by), 1)

        # Particles.
        if state["show_nodes"]:
            for k in range(world.pos.shape[0]):
                px, py = world.pos[k]
                if world.pinned[k]:
                    pygame.draw.circle(screen, CLOTH_PINNED, (int(px), int(py)), 3)
                else:
                    pygame.draw.circle(screen, CLOTH_LINE, (int(px), int(py)), 2)

        # Highlight dragged particle.
        if drag_idx is not None:
            px, py = world.pos[drag_idx]
            pygame.draw.circle(screen, CLOTH_DRAG, (int(px), int(py)), 6, 2)

        # Slice trail (a fading red line).
        if len(slice_trail) >= 2:
            for i in range(1, len(slice_trail)):
                p1 = slice_trail[i - 1]
                p2 = slice_trail[i]
                alpha = int(60 + 195 * (i / len(slice_trail)))
                col = (min(255, SLICE_TRAIL[0]), min(255, SLICE_TRAIL[1]),
                       min(255, SLICE_TRAIL[2]))
                pygame.draw.line(screen, col, p1, p2, 2)

        # Border.
        pygame.draw.rect(screen, PANEL_BORDER, (0, 0, SIM_W, SIM_H), 1)

        # ---- HUD ------------------------------------------------------
        pygame.draw.rect(screen, PANEL_BG, (SIM_W, 0, HUD_W, HEIGHT))
        pygame.draw.line(screen, PANEL_BORDER, (SIM_W, 0), (SIM_W, HEIGHT), 1)
        screen.blit(title_font.render("Verlet Physics", True, ACCENT),
                    (SIM_W + 16, 14))

        n_alive = sum(1 for a in world.c_alive if a)
        screen.blit(small.render(f"FPS: {clock.get_fps():.0f}    "
                                  f"{'PAUSED' if state['paused'] else 'RUNNING'}",
                                  True, TEXT_DIM), (SIM_W + 16, 38))
        screen.blit(small.render(f"Particles: {world.pos.shape[0]}    "
                                  f"Constraints: {n_alive}/{len(world.c_alive)}",
                                  True, TEXT_DIM), (SIM_W + 16, 54))

        for s in sliders:
            s.draw(screen, font)

        help_lines = [
            "Left click + drag:",
            "  pull a particle",
            "Right click + drag:",
            "  slice constraints",
            "B: drop a box at cursor",
            "Space: pause",
            "R: reset scene",
            "Esc: quit",
            "",
            "More iterations =",
            "stiffer cloth/rope.",
            "Damping < 1 bleeds",
            "energy slowly.",
        ]
        y = HEIGHT - 16 * len(help_lines) - 10
        for line in help_lines:
            screen.blit(small.render(line, True, TEXT_DIM), (SIM_W + 16, y))
            y += 16

        pygame.display.flip()


if __name__ == "__main__":
    main()
