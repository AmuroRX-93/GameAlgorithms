"""
Boids: Fish School Simulation with Sharks.

Implements Craig Reynolds' three classic rules:
  1. Separation - steer to avoid crowding nearby flockmates.
  2. Alignment  - steer toward the average heading of nearby flockmates.
  3. Cohesion   - steer toward the average position of nearby flockmates.

Plus extras: shark predators (with avoidance), wall avoidance, mouse cursor
attraction/repulsion, and a tweakable parameter panel.

Performance: position/velocity stored as (N, 2) numpy arrays. Neighbor lookup
uses a uniform spatial hash grid -> O(N) average per frame instead of O(N^2).
A uniform grid is plenty fast up to ~2000 fish on a laptop CPU.
"""

import math
import random
import sys

import numpy as np
import pygame

WIDTH, HEIGHT = 1280, 800
HUD_W = 280                 # Right-side parameter panel width.
SIM_W = WIDTH - HUD_W
SIM_H = HEIGHT
FPS = 60

# Colors.
BG = (8, 14, 30)
DEEP = (4, 8, 18)
FISH_BODY = (130, 200, 240)
FISH_TIP = (220, 240, 255)
SHARK_BODY = (220, 80, 80)
SHARK_TIP = (255, 230, 230)
PANEL_BG = (22, 26, 40)
PANEL_BORDER = (60, 70, 95)
TEXT = (220, 226, 240)
TEXT_DIM = (140, 150, 175)
ACCENT = (245, 210, 80)
SLIDER_TRACK = (50, 58, 80)
SLIDER_FILL = (130, 200, 240)
SLIDER_KNOB = (220, 240, 255)


# ---------------------------------------------------------------------------
# Tunable parameters (mutable; bound to the slider panel).
# ---------------------------------------------------------------------------


class Params:
    """All tweakable simulation parameters with their UI ranges."""

    def __init__(self):
        # (attr_name, label, min, max, default, step)
        self.specs = [
            ("num_fish",        "Fish count",        50,   2000,  200,  10),
            ("num_sharks",      "Shark count",       0,    8,     1,    1),
            ("perception",      "Perception radius", 20,   120,   55,   1),
            ("separation_dist", "Separation dist",   8,    50,    20,   1),
            ("max_speed",       "Fish max speed",    50,   400,   180,  5),
            ("shark_speed",     "Shark max speed",   50,   400,   140,  5),
            ("shark_radius",    "Shark fear radius", 40,   300,   140,  5),
            ("w_separation",    "Separation weight", 0,    400,   180,  5),
            ("w_alignment",     "Alignment weight",  0,    400,   90,   5),
            ("w_cohesion",      "Cohesion weight",   0,    400,   60,   5),
            ("w_shark_flee",    "Shark flee weight", 0,    1000,  400,  10),
            ("w_wall",          "Wall avoid weight", 0,    400,   180,  5),
        ]
        for attr, _label, _lo, _hi, default, _step in self.specs:
            setattr(self, attr, default)

        # Mouse interaction: 0 = off, 1 = attract, -1 = repulse.
        self.mouse_mode = 0
        self.mouse_force = 600.0
        self.show_neighbors = False  # Highlight neighborhood circles.


# ---------------------------------------------------------------------------
# Spatial hash grid for O(N) neighbor lookup.
# ---------------------------------------------------------------------------


def build_grid(positions, cell_size):
    """Return dict {(cell_x, cell_y): np.array of indices into positions}.

    `positions` is shape (N, 2). cell_size should match the perception radius.
    """
    cells = (positions / cell_size).astype(np.int32)
    grid = {}
    for i in range(positions.shape[0]):
        key = (int(cells[i, 0]), int(cells[i, 1]))
        grid.setdefault(key, []).append(i)
    return grid


def neighbor_indices(grid, cell_x, cell_y):
    """Yield index lists from the 3x3 cell block around (cell_x, cell_y)."""
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            bucket = grid.get((cell_x + dx, cell_y + dy))
            if bucket:
                yield bucket


# ---------------------------------------------------------------------------
# Boids simulation core.
# ---------------------------------------------------------------------------


def init_swarm(n, w, h, max_speed, rng):
    pos = rng.random((n, 2)).astype(np.float32) * np.array([w, h], dtype=np.float32)
    angles = rng.random(n).astype(np.float32) * 2 * math.pi
    vel = np.stack([np.cos(angles), np.sin(angles)], axis=1) * (max_speed * 0.5)
    return pos, vel.astype(np.float32)


def limit(vec, max_mag):
    """Clamp the magnitude of each row in `vec` to `max_mag` (in place safe).

    Works on shape (2,) or (N, 2). Rows with zero magnitude pass through.
    """
    if vec.ndim == 1:
        m = math.hypot(vec[0], vec[1])
        if m > max_mag and m > 1e-6:
            return vec * (max_mag / m)
        return vec
    mag = np.linalg.norm(vec, axis=1)
    over = mag > max_mag
    if not over.any():
        return vec
    scale = np.ones_like(mag)
    scale[over] = max_mag / mag[over]
    return vec * scale[:, None]


def _step_dense(pos, vel, params, perception_sq, sep_sq, max_speed):
    """N x N dense pairwise approach. Best for small N (~< 400)."""
    diff = pos[None, :, :] - pos[:, None, :]
    d2 = (diff * diff).sum(axis=2)
    np.fill_diagonal(d2, np.inf)

    in_perc = d2 < perception_sq
    too_close = d2 < sep_sq

    perc_count = in_perc.sum(axis=1, keepdims=True)
    safe = np.maximum(perc_count, 1)
    pos_sum = in_perc.astype(np.float32) @ pos
    coh_force = pos_sum / safe - pos
    coh_force[perc_count[:, 0] == 0] = 0

    vel_sum = in_perc.astype(np.float32) @ vel
    align_force = vel_sum / safe - vel
    align_force[perc_count[:, 0] == 0] = 0

    inv_d = np.zeros_like(d2)
    mask = too_close & (d2 > 1e-6)
    inv_d[mask] = 1.0 / np.sqrt(d2[mask])
    sep_force = (-diff * inv_d[:, :, None]).sum(axis=1)
    return sep_force, align_force, coh_force


def _step_hashed(pos, vel, params, perception, perception_sq, sep_sq):
    """Spatial-hash approach. Best for large N."""
    n = pos.shape[0]
    cells = (pos / perception).astype(np.int32)
    cell_key = cells[:, 0] * 1_000_003 + cells[:, 1]
    sort_idx = np.argsort(cell_key, kind="stable")
    cell_key_sorted = cell_key[sort_idx]
    diffs = np.diff(cell_key_sorted)
    bucket_starts = np.concatenate([[0], np.where(diffs != 0)[0] + 1, [n]])

    cells_sorted = cells[sort_idx]
    grid = {}
    for k in range(len(bucket_starts) - 1):
        s, e = bucket_starts[k], bucket_starts[k + 1]
        key = (int(cells_sorted[s, 0]), int(cells_sorted[s, 1]))
        grid[key] = sort_idx[s:e]

    sep_force = np.zeros_like(pos)
    align_force = np.zeros_like(pos)
    coh_force = np.zeros_like(pos)

    for (cx, cy), idx_self in grid.items():
        cand = [grid[(cx + dx, cy + dy)] for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                if (cx + dx, cy + dy) in grid]
        if not cand:
            continue
        cand_idx = np.concatenate(cand)
        d = pos[cand_idx][None, :, :] - pos[idx_self][:, None, :]
        d2 = (d * d).sum(axis=2)

        in_perc = (d2 < perception_sq) & (d2 > 1e-6)
        too_close = (d2 < sep_sq) & (d2 > 1e-6)

        perc_count = in_perc.sum(axis=1)
        safe = np.maximum(perc_count, 1)[:, None]

        pos_sum = (in_perc[:, :, None] * pos[cand_idx][None, :, :]).sum(axis=1)
        coh_force[idx_self] = pos_sum / safe - pos[idx_self]
        coh_force[idx_self[perc_count == 0]] = 0

        vel_sum = (in_perc[:, :, None] * vel[cand_idx][None, :, :]).sum(axis=1)
        align_force[idx_self] = vel_sum / safe - vel[idx_self]
        align_force[idx_self[perc_count == 0]] = 0

        inv_d = np.zeros_like(d2)
        m = too_close
        inv_d[m] = 1.0 / np.sqrt(d2[m])
        sep_force[idx_self] = (-d * inv_d[:, :, None]).sum(axis=1)

    return sep_force, align_force, coh_force


def step_simulation(pos, vel, sharks_pos, sharks_vel, params, mouse_pos, dt):
    """Advance one simulation step. Auto-picks dense vs hashed based on N."""
    n = pos.shape[0]
    if n == 0:
        return pos, vel, sharks_pos, sharks_vel

    perception = float(params.perception)
    sep_dist = float(params.separation_dist)
    max_speed = float(params.max_speed)
    perception_sq = perception * perception
    sep_sq = sep_dist * sep_dist

    if n < 500:
        sep_force, align_force, coh_force = _step_dense(
            pos, vel, params, perception_sq, sep_sq, max_speed
        )
    else:
        sep_force, align_force, coh_force = _step_hashed(
            pos, vel, params, perception, perception_sq, sep_sq
        )

    # ---- Wall avoidance.
    margin = 60.0
    wall = np.zeros_like(pos)
    left = pos[:, 0] < margin
    right = pos[:, 0] > SIM_W - margin
    top = pos[:, 1] < margin
    bot = pos[:, 1] > SIM_H - margin
    wall[left, 0] += (margin - pos[left, 0]) / margin
    wall[right, 0] -= (pos[right, 0] - (SIM_W - margin)) / margin
    wall[top, 1] += (margin - pos[top, 1]) / margin
    wall[bot, 1] -= (pos[bot, 1] - (SIM_H - margin)) / margin

    # ---- Shark fear.
    flee = np.zeros_like(pos)
    if sharks_pos.shape[0] > 0:
        sr2 = params.shark_radius ** 2
        sd = pos[:, None, :] - sharks_pos[None, :, :]
        sd2 = (sd * sd).sum(axis=2)
        scared = sd2 < sr2
        sd_norm = np.sqrt(np.maximum(sd2, 1e-6))
        weight = np.where(scared, 1 - sd_norm / params.shark_radius, 0.0)
        flee = (sd / sd_norm[:, :, None] * weight[:, :, None]).sum(axis=1)

    # ---- Mouse.
    mouse = np.zeros_like(pos)
    if params.mouse_mode != 0 and mouse_pos is not None:
        mx, my = mouse_pos
        # Vector pointing from the mouse to each fish. Multiplying by mode = +1
        # pushes fish AWAY (repel), mode = -1 pulls them TOWARD (attract).
        away = pos - np.array([mx, my], dtype=np.float32)
        dist2 = (away * away).sum(axis=1) + 1e-6
        in_range = dist2 < (200 * 200)
        if in_range.any():
            inv = 1.0 / np.sqrt(dist2[in_range])[:, None]
            mouse[in_range] = away[in_range] * inv * params.mouse_force * params.mouse_mode

    accel = (
        limit(sep_force, max_speed) * params.w_separation
        + limit(align_force, max_speed) * params.w_alignment
        + limit(coh_force, max_speed) * params.w_cohesion
        + flee * params.w_shark_flee
        + wall * params.w_wall
        + mouse
    ) / 100.0

    new_vel = vel + accel * dt
    new_vel = limit(new_vel, max_speed)
    new_pos = pos + new_vel * dt

    new_pos[:, 0] %= SIM_W
    new_pos[:, 1] %= SIM_H

    # ---- Sharks.
    if sharks_pos.shape[0] > 0:
        d = new_pos[None, :, :] - sharks_pos[:, None, :]
        d2 = (d * d).sum(axis=2)
        target_idx = np.argmin(d2, axis=1)
        target = new_pos[target_idx]

        accel_s = (target - sharks_pos) * 0.05
        sw = np.zeros_like(sharks_pos)
        sw[sharks_pos[:, 0] < margin, 0] += 1
        sw[sharks_pos[:, 0] > SIM_W - margin, 0] -= 1
        sw[sharks_pos[:, 1] < margin, 1] += 1
        sw[sharks_pos[:, 1] > SIM_H - margin, 1] -= 1
        accel_s += sw * 80.0

        sharks_vel = limit(sharks_vel + accel_s * dt, params.shark_speed)
        sharks_pos = sharks_pos + sharks_vel * dt
        sharks_pos[:, 0] = np.clip(sharks_pos[:, 0], 0, SIM_W - 1)
        sharks_pos[:, 1] = np.clip(sharks_pos[:, 1], 0, SIM_H - 1)

    return new_pos.astype(np.float32), new_vel.astype(np.float32), sharks_pos, sharks_vel


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def draw_fish(surface, pos, vel, body_color, tip_color, size=5.0):
    """Render fish as small triangles oriented along velocity."""
    n = pos.shape[0]
    if n == 0:
        return
    speeds = np.linalg.norm(vel, axis=1) + 1e-6
    dirx = vel[:, 0] / speeds
    diry = vel[:, 1] / speeds
    tip = pos + np.stack([dirx, diry], axis=1) * size * 1.6
    bl = pos + np.stack([-diry, dirx], axis=1) * size * 0.7 - np.stack([dirx, diry], axis=1) * size * 0.6
    br = pos - np.stack([-diry, dirx], axis=1) * size * 0.7 - np.stack([dirx, diry], axis=1) * size * 0.6
    # pygame.draw.polygon needs python float/int tuples, not numpy scalars.
    tip_l = tip.tolist()
    bl_l = bl.tolist()
    br_l = br.tolist()
    for i in range(n):
        pygame.draw.polygon(
            surface, body_color,
            [tip_l[i], bl_l[i], br_l[i]],
        )


def draw_shark(surface, pos, vel, size=14.0):
    """Larger triangle, red, with a tail line."""
    if pos.shape[0] == 0:
        return
    speeds = np.linalg.norm(vel, axis=1) + 1e-6
    dirx = vel[:, 0] / speeds
    diry = vel[:, 1] / speeds
    pos_l = pos.tolist()
    for i in range(pos.shape[0]):
        dx, dy = float(dirx[i]), float(diry[i])
        cx, cy = pos_l[i]
        tip = (cx + dx * size * 1.8, cy + dy * size * 1.8)
        bl = (cx + (-dy) * size * 0.9 - dx * size * 0.7,
              cy + dx * size * 0.9 - dy * size * 0.7)
        br = (cx - (-dy) * size * 0.9 - dx * size * 0.7,
              cy - dx * size * 0.9 - dy * size * 0.7)
        tail = (cx - dx * size * 1.4, cy - dy * size * 1.4)
        pygame.draw.polygon(surface, SHARK_BODY, [tip, bl, br])
        pygame.draw.line(surface, SHARK_TIP, (cx, cy), tail, 2)


# ---------------------------------------------------------------------------
# Slider UI
# ---------------------------------------------------------------------------


class Slider:
    HEIGHT = 14
    LABEL_GAP = 18

    def __init__(self, x, y, w, attr, label, lo, hi, step):
        self.attr = attr
        self.label = label
        self.lo = lo
        self.hi = hi
        self.step = step
        self.rect = pygame.Rect(x, y + Slider.LABEL_GAP, w, Slider.HEIGHT)
        self.dragging = False

    def value_to_x(self, value):
        t = (value - self.lo) / (self.hi - self.lo)
        return int(self.rect.x + t * self.rect.w)

    def x_to_value(self, x):
        t = (x - self.rect.x) / self.rect.w
        t = max(0.0, min(1.0, t))
        v = self.lo + t * (self.hi - self.lo)
        return round(v / self.step) * self.step

    def draw(self, surface, font, params):
        v = getattr(params, self.attr)
        # Label above the bar.
        label = f"{self.label}: {int(v) if isinstance(v, int) or v == int(v) else round(v, 1)}"
        surface.blit(font.render(label, True, TEXT), (self.rect.x, self.rect.y - Slider.LABEL_GAP + 1))

        # Track + filled portion.
        pygame.draw.rect(surface, SLIDER_TRACK, self.rect, border_radius=4)
        knob_x = self.value_to_x(v)
        fill = pygame.Rect(self.rect.x, self.rect.y, knob_x - self.rect.x, self.rect.h)
        pygame.draw.rect(surface, SLIDER_FILL, fill, border_radius=4)
        pygame.draw.circle(surface, SLIDER_KNOB, (knob_x, self.rect.centery), 7)

    def hit(self, mx, my):
        return self.rect.x - 4 <= mx <= self.rect.right + 4 and self.rect.y - 6 <= my <= self.rect.bottom + 6

    def update_from_mouse(self, mx, params):
        v = self.x_to_value(mx)
        # Cast to int when the underlying default was an int (count fields).
        cur = getattr(params, self.attr)
        if isinstance(cur, int) or self.step >= 1:
            v = int(v)
        setattr(params, self.attr, v)


def build_sliders(params):
    sliders = []
    pad = 16
    x = SIM_W + pad
    y = 90
    width = HUD_W - 2 * pad
    spacing = 42
    for attr, label, lo, hi, _default, step in params.specs:
        sliders.append(Slider(x, y, width, attr, label, lo, hi, step))
        y += spacing
    return sliders


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main():
    pygame.init()
    pygame.display.set_caption("Boids: Fish School + Sharks")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Menlo,Consolas,monospace", 13)
    small = pygame.font.SysFont("Menlo,Consolas,monospace", 11)
    title_font = pygame.font.SysFont("Menlo,Consolas,monospace", 16, bold=True)

    rng = np.random.default_rng()
    params = Params()
    sliders = build_sliders(params)

    pos, vel = init_swarm(params.num_fish, SIM_W, SIM_H, params.max_speed, rng)
    sharks_pos, sharks_vel = init_swarm(params.num_sharks, SIM_W, SIM_H, params.shark_speed, rng)

    paused = False
    active_slider = None

    while True:
        dt = clock.tick(FPS) / 1000.0
        dt = min(dt, 1 / 30)  # Cap to avoid tunneling on hitches.

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit(0)
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_r:
                    pos, vel = init_swarm(params.num_fish, SIM_W, SIM_H, params.max_speed, rng)
                    sharks_pos, sharks_vel = init_swarm(params.num_sharks, SIM_W, SIM_H, params.shark_speed, rng)
                elif event.key == pygame.K_n:
                    params.show_neighbors = not params.show_neighbors
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                if mx > SIM_W:
                    for s in sliders:
                        if s.hit(mx, my):
                            active_slider = s
                            s.update_from_mouse(mx, params)
                            break
                else:
                    if event.button == 1:
                        # Left click: attract fish toward cursor.
                        params.mouse_mode = -1
                    elif event.button == 3:
                        # Right click: repel.
                        params.mouse_mode = 1
            elif event.type == pygame.MOUSEBUTTONUP:
                active_slider = None
                params.mouse_mode = 0
            elif event.type == pygame.MOUSEMOTION:
                if active_slider:
                    active_slider.update_from_mouse(event.pos[0], params)

        # If user changed counts via sliders, resize swarms.
        if pos.shape[0] != params.num_fish:
            if params.num_fish > pos.shape[0]:
                add = params.num_fish - pos.shape[0]
                npos, nvel = init_swarm(add, SIM_W, SIM_H, params.max_speed, rng)
                pos = np.vstack([pos, npos]).astype(np.float32)
                vel = np.vstack([vel, nvel]).astype(np.float32)
            else:
                pos = pos[:params.num_fish]
                vel = vel[:params.num_fish]
        if sharks_pos.shape[0] != params.num_sharks:
            if params.num_sharks > sharks_pos.shape[0]:
                add = params.num_sharks - sharks_pos.shape[0]
                spos, svel = init_swarm(add, SIM_W, SIM_H, params.shark_speed, rng)
                sharks_pos = np.vstack([sharks_pos, spos]).astype(np.float32)
                sharks_vel = np.vstack([sharks_vel, svel]).astype(np.float32)
            else:
                sharks_pos = sharks_pos[:params.num_sharks]
                sharks_vel = sharks_vel[:params.num_sharks]

        if not paused:
            mouse_pos = pygame.mouse.get_pos() if params.mouse_mode != 0 else None
            if mouse_pos and mouse_pos[0] >= SIM_W:
                mouse_pos = None
            pos, vel, sharks_pos, sharks_vel = step_simulation(
                pos, vel, sharks_pos, sharks_vel, params, mouse_pos, dt
            )

        # ---- Render scene.
        screen.fill(DEEP)
        # Gradient background hint.
        for i in range(0, SIM_H, 4):
            t = i / SIM_H
            c = (int(8 + 8 * t), int(14 + 16 * t), int(30 + 30 * t))
            pygame.draw.rect(screen, c, (0, i, SIM_W, 4))

        draw_fish(screen, pos, vel, FISH_BODY, FISH_TIP, size=4.0)
        draw_shark(screen, sharks_pos, sharks_vel, size=12.0)

        # Mouse interaction range indicator while holding a button on the water.
        if params.mouse_mode != 0:
            mx, my = pygame.mouse.get_pos()
            if mx < SIM_W:
                ring = (130, 220, 130) if params.mouse_mode == -1 else (240, 110, 110)
                pygame.draw.circle(screen, ring, (mx, my), 200, width=2)

        # ---- Right panel.
        pygame.draw.rect(screen, PANEL_BG, (SIM_W, 0, HUD_W, HEIGHT))
        pygame.draw.line(screen, PANEL_BORDER, (SIM_W, 0), (SIM_W, HEIGHT), 1)
        screen.blit(title_font.render("Boids Parameters", True, ACCENT), (SIM_W + 16, 14))
        screen.blit(small.render(f"FPS: {clock.get_fps():.0f}   N: {pos.shape[0]}   Sharks: {sharks_pos.shape[0]}",
                                  True, TEXT_DIM), (SIM_W + 16, 38))
        screen.blit(small.render(f"{'PAUSED' if paused else 'RUNNING'}   (Space pause, R reset)",
                                  True, TEXT_DIM), (SIM_W + 16, 54))

        for s in sliders:
            s.draw(screen, font, params)

        # Bottom help.
        help_lines = [
            "Left click on water:  attract",
            "Right click on water: repel",
            "Space: pause   R: reset",
            "Esc: quit",
        ]
        y = HEIGHT - 16 * len(help_lines) - 10
        for line in help_lines:
            screen.blit(small.render(line, True, TEXT_DIM), (SIM_W + 16, y))
            y += 16

        pygame.display.flip()


if __name__ == "__main__":
    main()
