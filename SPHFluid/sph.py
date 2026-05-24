"""
2D Smoothed Particle Hydrodynamics (SPH) fluid with a position-based
separation pass for stability.

Each particle carries a position, velocity, and density. Every step:

  1. Find every particle's neighbors within the kernel radius h using a
     spatial hash grid (cell size = h, look at the 3x3 cell block).
  2. Density at particle i: sum_j m * W_poly6(|r_i - r_j|, h).
  3. Pressure from a clamped linear EOS: p = max(0, k * (rho - rho0)).
     Negative pressures are zeroed out so neighbor particles can never
     pull each other together — only push apart. This is what stops the
     classic SPH failure mode where surface particles get sucked back
     into the bulk and the whole fluid flattens against the floor.
  4. Forces: pressure gradient (spiky kernel), viscosity (viscosity
     kernel laplacian), gravity, mouse drag.
  5. Symplectic Euler integration.
  6. Position-based separation pass: for any pair of particles closer
     than h*0.55, project them apart so they're exactly that far. This
     is one PBD-style constraint iteration that catches the cases
     where the pressure force alone wasn't strong enough to prevent
     interpenetration.
  7. Tank collision (clamp to bounds, reflect velocity).

The naive O(N^2) neighbor search would melt at >500 particles. We use a
uniform spatial hash grid with cell size = h so each particle only
checks the 9 cells around it (~O(N)).

Mouse drag pulls fluid toward the cursor (left click) or pushes it away
(right click). Spacebar pauses; R resets.
"""

import sys
import math
import numpy as np
import pygame

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
WIDTH, HEIGHT = 1280, 800
HUD_W = 280
SIM_W = WIDTH - HUD_W
SIM_H = HEIGHT
FPS = 60

# World coordinates use the same units as screen pixels; this keeps things
# simple for visualization. h is the kernel "smoothing length"; particles
# only interact with neighbors closer than h.
H_DEFAULT = 16.0
PARTICLE_MASS = 1.0
# REST_DENSITY is set after the kernels are defined, by calibrating against
# a hex-packed reference patch at the same spacing spawn_block uses.
REST_DENSITY = 1.0  # placeholder; replaced below

# Colors
BG = (10, 14, 22)
PANEL_BG = (22, 26, 40)
PANEL_BORDER = (60, 70, 95)
TEXT = (220, 226, 240)
TEXT_DIM = (140, 150, 175)
ACCENT = (245, 210, 80)
TANK = (50, 60, 85)
TANK_LINE = (90, 110, 150)
SLIDER_TRACK = (50, 58, 80)
SLIDER_FILL = (130, 200, 240)
SLIDER_KNOB = (220, 240, 255)


# ---------------------------------------------------------------------------
# SPH kernels (Müller et al. 2003)
# ---------------------------------------------------------------------------
# poly6:    used for density.
# spiky:    used for the pressure gradient (its gradient is well-behaved
#           at small distances, unlike poly6 which goes flat near r=0).
# viscosity-laplacian: used for the viscosity force.
# All kernels are zero outside r > h.
#
# The constants below are the standard 2D normalizations that make the
# kernel integrate to 1 over a disk of radius h.


def poly6(r2, h):
    """W(r) for r2 = |r|^2. Returns 0 where r2 > h^2."""
    h2 = h * h
    coef = 4.0 / (math.pi * h ** 8)
    diff = h2 - r2
    out = np.where(r2 < h2, coef * diff * diff * diff, 0.0)
    return out


def spiky_grad_coef(r, h):
    """Magnitude of grad W_spiky / r. The full gradient is this scalar
    times the displacement vector (r_i - r_j). Returns 0 where r >= h.

    grad W_spiky = -30 / (pi h^5) * (h - r)^2 * r_hat
    so we return -30 / (pi h^5) * (h - r)^2 / r, callers multiply by dx, dy.
    """
    coef = -30.0 / (math.pi * h ** 5)
    out = np.where((r > 0) & (r < h),
                   coef * (h - r) ** 2 / np.maximum(r, 1e-9),
                   0.0)
    return out


def visc_lap_coef(r, h):
    """Laplacian of the viscosity kernel: 40 / (pi h^5) * (h - r)."""
    coef = 40.0 / (math.pi * h ** 5)
    return np.where((r >= 0) & (r < h), coef * (h - r), 0.0)


def _calibrate_rest_density():
    """Compute the density that a particle in the middle of a square-packed
    block (at the same spacing spawn_block uses) would have. We use this
    as the rest density so the equation of state is happy at rest."""
    h = H_DEFAULT
    spacing = h * 0.85
    pts = []
    for ry in range(-5, 6):
        for cx in range(-5, 6):
            pts.append((cx * spacing, ry * spacing))
    pts = np.array(pts, dtype=np.float32)
    r2 = (pts[:, 0] ** 2 + pts[:, 1] ** 2).astype(np.float32)
    return float(PARTICLE_MASS * poly6(r2, h).sum())


REST_DENSITY = _calibrate_rest_density()


# ---------------------------------------------------------------------------
# Spatial hash grid
# ---------------------------------------------------------------------------


class Grid:
    """Uniform spatial hash with cell size = h. Stores particle indices
    per cell so neighbor queries scan only 9 cells."""

    def __init__(self, w, h, cell):
        self.cell = cell
        self.cols = max(1, int(math.ceil(w / cell)))
        self.rows = max(1, int(math.ceil(h / cell)))

    def rebuild(self, pos):
        """Bucket every particle by cell. Returns:
          cell_start : (rows*cols + 1,) int32 — CSR-style offsets
          cell_items : (N,)              int32 — particle indices
          cell_idx   : (N,)              int32 — each particle's cell id
        """
        N = pos.shape[0]
        cell = self.cell
        cx = np.clip((pos[:, 0] / cell).astype(np.int32), 0, self.cols - 1)
        cy = np.clip((pos[:, 1] / cell).astype(np.int32), 0, self.rows - 1)
        cell_idx = cy * self.cols + cx
        n_cells = self.rows * self.cols
        counts = np.bincount(cell_idx, minlength=n_cells)
        cell_start = np.zeros(n_cells + 1, dtype=np.int32)
        np.cumsum(counts, out=cell_start[1:])
        # Sort particle indices by cell so cell_items[cell_start[c]:cell_start[c+1]]
        # are the particles in cell c.
        order = np.argsort(cell_idx, kind="stable").astype(np.int32)
        cell_items = order
        return cell_start, cell_items, cell_idx

    def neighbors_of_cell(self, c):
        """Yield up to 9 neighboring cell ids around c (including itself).
        Returns a python list of ints."""
        cy, cx = divmod(c, self.cols)
        out = []
        for dy in (-1, 0, 1):
            ny = cy + dy
            if ny < 0 or ny >= self.rows:
                continue
            for dx in (-1, 0, 1):
                nx = cx + dx
                if nx < 0 or nx >= self.cols:
                    continue
                out.append(ny * self.cols + nx)
        return out


# ---------------------------------------------------------------------------
# Fluid
# ---------------------------------------------------------------------------


class Fluid:
    def __init__(self, n=900, h=H_DEFAULT, tank=None):
        self.h = h
        self.mass = PARTICLE_MASS
        self.rest_density = REST_DENSITY
        self.stiffness = 200.0      # gas constant k for the linear EOS
        self.viscosity = 0.4
        self.gravity = 700.0
        self.tank = tank or pygame.Rect(40, 80, SIM_W - 80, SIM_H - 120)
        self.pos = np.zeros((0, 2), dtype=np.float32)
        self.vel = np.zeros((0, 2), dtype=np.float32)
        self.density = np.zeros(0, dtype=np.float32)
        self.pressure = np.zeros(0, dtype=np.float32)
        self.grid = Grid(SIM_W, SIM_H, h)
        self.spawn_block(n)

    def spawn_block(self, n):
        """Pack n particles into a block in the lower half of the tank.
        Spacing is slightly less than h so each particle starts with
        ~3-5 neighbors and a density close to rest_density."""
        h = self.h
        spacing = h * 0.85
        # Choose a column count that produces a roughly square block.
        cols = max(1, int(math.sqrt(n)))
        rows = (n + cols - 1) // cols
        # Center the block horizontally and place it in the lower 60% of
        # the tank so it has room to fall and slosh.
        block_w = cols * spacing
        block_h = rows * spacing
        x0 = self.tank.left + (self.tank.width - block_w) * 0.5
        y0 = self.tank.bottom - block_h - 20
        idx = 0
        pts = []
        for r in range(rows):
            for c in range(cols):
                if idx >= n:
                    break
                jx = (np.random.rand() - 0.5) * spacing * 0.05
                jy = (np.random.rand() - 0.5) * spacing * 0.05
                pts.append((x0 + c * spacing + jx, y0 + r * spacing + jy))
                idx += 1
        pts = np.array(pts, dtype=np.float32)
        self.pos = pts
        self.vel = np.zeros_like(pts)
        self.density = np.zeros(pts.shape[0], dtype=np.float32)
        self.pressure = np.zeros(pts.shape[0], dtype=np.float32)

    def reset(self, n):
        self.spawn_block(n)

    # -----------------------------------------------------------------
    # Core SPH step
    # -----------------------------------------------------------------

    def step(self, dt, mouse_pos=None, mouse_force=0.0, mouse_radius=120.0):
        """Advance the simulation by dt seconds.

        Spatial-hash neighbor finding: bucket particles by cells of size h,
        then for each non-empty cell c, compute pair displacements between
        c and each of its 9 cell neighbors (including itself, with j>i to
        avoid double-counting). The cartesian product per cell-pair is
        done with a single broadcast, so the Python loop is over cells,
        not particles. With h=16 and a typical fluid puddle, that's a few
        hundred cells, each with ~1-5 particles.
        """
        N = self.pos.shape[0]
        if N == 0:
            return

        h = self.h
        h2 = h * h
        if self.grid.cell != h:
            self.grid = Grid(SIM_W, SIM_H, h)
        cell_start, cell_items, cell_idx = self.grid.rebuild(self.pos)
        cols = self.grid.cols
        rows = self.grid.rows

        # ---- Build pair index arrays via cell-vs-cell broadcasts ---
        ii_chunks = []
        jj_chunks = []
        # We iterate over non-empty cells only.
        non_empty = np.nonzero(np.diff(cell_start))[0]
        for c in non_empty:
            a = cell_start[c]; b = cell_start[c + 1]
            here = cell_items[a:b]   # particle indices in this cell
            cy, cx = divmod(int(c), cols)
            for dy_ in (-1, 0, 1):
                ny = cy + dy_
                if ny < 0 or ny >= rows:
                    continue
                for dx_ in (-1, 0, 1):
                    nx = cx + dx_
                    if nx < 0 or nx >= cols:
                        continue
                    nc = ny * cols + nx
                    if nc < c:
                        continue   # processed from the lower-id cell
                    a2 = cell_start[nc]; b2 = cell_start[nc + 1]
                    if b2 == a2:
                        continue
                    there = cell_items[a2:b2]
                    if nc == c:
                        # Within the same cell: take upper-triangle pairs.
                        if here.size < 2:
                            continue
                        I, J = np.meshgrid(here, here, indexing="ij")
                        keep = I < J
                        ii_chunks.append(I[keep])
                        jj_chunks.append(J[keep])
                    else:
                        I, J = np.meshgrid(here, there, indexing="ij")
                        ii_chunks.append(I.ravel())
                        jj_chunks.append(J.ravel())

        if ii_chunks:
            ii = np.concatenate(ii_chunks).astype(np.int32)
            jj = np.concatenate(jj_chunks).astype(np.int32)
        else:
            ii = np.zeros(0, dtype=np.int32)
            jj = np.zeros(0, dtype=np.int32)

        # Filter to actual h-radius neighbors (cell-block is just a coarse cull).
        if ii.size:
            dx = self.pos[jj, 0] - self.pos[ii, 0]
            dy = self.pos[jj, 1] - self.pos[ii, 1]
            r2 = dx * dx + dy * dy
            mask = r2 < h2
            if mask.size and not mask.all():
                ii = ii[mask]; jj = jj[mask]
                dx = dx[mask]; dy = dy[mask]; r2 = r2[mask]
            r = np.sqrt(r2)
        else:
            dx = dy = r = r2 = np.zeros(0, dtype=np.float32)

        # ---- Density -----------------------------------------------
        # density_i = m * poly6(0) + sum over neighbors j of m * poly6(r_ij^2).
        # Each pair (i, j) contributes to BOTH i and j.
        self_w = float(poly6(np.float32(0.0), h))
        self.density.fill(self.mass * self_w)
        if ii.size:
            w = (self.mass * poly6(r2.astype(np.float32), h)).astype(np.float32)
            np.add.at(self.density, ii, w)
            np.add.at(self.density, jj, w)

        # ---- Pressure (linear, only positive) ---------------------
        # We only push when the density is ABOVE rest. SPH normally has
        # negative pressure pulling particles together when density is
        # low, but that's what tears boundary particles into the bulk
        # and causes them to clump along the floor. Clamping the
        # pressure at zero turns the EOS into a one-sided "no
        # interpenetration" force, which is what we actually want for
        # game-like fluid.
        self.pressure = np.maximum(self.stiffness * (self.density - self.rest_density), 0.0).astype(np.float32)

        # ---- Acceleration accumulators -----------------------------
        ax = np.zeros(N, dtype=np.float32)
        ay = np.zeros(N, dtype=np.float32)

        if ii.size:
            inv_rho = 1.0 / np.maximum(self.density, 1e-6)
            # +30/(pi h^5) * (h-r)^2 / r — multiplied by dx (j - i) gives
            # gradW pointing from i toward j. With a NEGATIVE sign in
            # mag_*, the pressure force pushes i AWAY from j.
            eps = 1e-5
            gcoef = ((30.0 / (math.pi * h ** 5)) * (h - r) ** 2 / np.maximum(r, eps)).astype(np.float32)
            p_avg = (0.5 * (self.pressure[ii] + self.pressure[jj])).astype(np.float32)
            mag_i = -self.mass * p_avg * inv_rho[jj] * gcoef
            np.add.at(ax, ii, mag_i * dx)
            np.add.at(ay, ii, mag_i * dy)
            mag_j = -self.mass * p_avg * inv_rho[ii] * gcoef
            np.add.at(ax, jj, -mag_j * dx)
            np.add.at(ay, jj, -mag_j * dy)

            # XSPH-style viscosity (averages neighbor velocities into i's
            # velocity directly later, but here we use the standard SPH
            # viscosity force for consistency).
            lap = ((40.0 / (math.pi * h ** 5)) * (h - r)).astype(np.float32)
            dvx = self.vel[jj, 0] - self.vel[ii, 0]
            dvy = self.vel[jj, 1] - self.vel[ii, 1]
            v_mag_i = self.viscosity * self.mass * inv_rho[jj] * lap
            np.add.at(ax, ii, v_mag_i * dvx)
            np.add.at(ay, ii, v_mag_i * dvy)
            v_mag_j = self.viscosity * self.mass * inv_rho[ii] * lap
            np.add.at(ax, jj, -v_mag_j * dvx)
            np.add.at(ay, jj, -v_mag_j * dvy)

        # Gravity
        ay += self.gravity

        # Mouse interaction
        if mouse_pos is not None and mouse_force != 0.0:
            mxp, myp = mouse_pos
            ddx = mxp - self.pos[:, 0]
            ddy = myp - self.pos[:, 1]
            d2m = ddx * ddx + ddy * ddy
            inside = d2m < (mouse_radius * mouse_radius)
            if inside.any():
                d = np.sqrt(np.maximum(d2m, 1e-6))
                fall = np.maximum(0.0, 1.0 - d / mouse_radius)
                ux = ddx / np.maximum(d, 1e-6)
                uy = ddy / np.maximum(d, 1e-6)
                ax[inside] += (mouse_force * fall[inside] * ux[inside]).astype(np.float32)
                ay[inside] += (mouse_force * fall[inside] * uy[inside]).astype(np.float32)

        # ---- Integrate ---------------------------------------------
        self.vel[:, 0] += ax * dt
        self.vel[:, 1] += ay * dt
        speed = np.sqrt(self.vel[:, 0] ** 2 + self.vel[:, 1] ** 2)
        max_v = 600.0
        too_fast = speed > max_v
        if too_fast.any():
            scale = max_v / speed[too_fast]
            self.vel[too_fast, 0] *= scale
            self.vel[too_fast, 1] *= scale
        self.pos[:, 0] += self.vel[:, 0] * dt
        self.pos[:, 1] += self.vel[:, 1] * dt

        # ---- Position-based separation pass ------------------------
        # The pressure force above struggles to keep particles separated
        # under heavy gravitational loading; if neighbors get too close,
        # the pressure gradient blows up but the time step doesn't catch
        # the correction in time. We run several Jacobi-style position
        # projections that force every pair closer than `min_sep` apart
        # to be exactly `min_sep` apart, splitting the correction by
        # mass. This is essentially a coarse PBF density-constraint
        # iteration operating directly on the spacing instead of the
        # kernel density, and it's what stops the fluid from collapsing
        # into a single line on the floor.
        if ii.size:
            min_sep = h * 0.55
            col0 = self.pos[:, 0]
            col1 = self.pos[:, 1]
            for _ in range(3):
                dxp = self.pos[jj, 0] - self.pos[ii, 0]
                dyp = self.pos[jj, 1] - self.pos[ii, 1]
                rp = np.sqrt(np.maximum(dxp * dxp + dyp * dyp, 1e-9))
                penetrating = rp < min_sep
                if not penetrating.any():
                    break
                ip = ii[penetrating]; jp = jj[penetrating]
                dxp = dxp[penetrating]; dyp = dyp[penetrating]
                rp = rp[penetrating]
                overlap = (min_sep - rp) * 0.5
                ux = dxp / rp
                uy = dyp / rp
                cx_ = -ux * overlap
                cy_ = -uy * overlap
                np.add.at(col0, ip, cx_)
                np.add.at(col1, ip, cy_)
                np.add.at(col0, jp, -cx_)
                np.add.at(col1, jp, -cy_)

        # ---- Tank collision ----------------------------------------
        damp = 0.3
        left = self.tank.left + 1.0
        right = self.tank.right - 1.0
        top = self.tank.top + 1.0
        bot = self.tank.bottom - 1.0
        m = self.pos[:, 0] < left
        self.pos[m, 0] = left;  self.vel[m, 0] *= -damp
        m = self.pos[:, 0] > right
        self.pos[m, 0] = right; self.vel[m, 0] *= -damp
        m = self.pos[:, 1] < top
        self.pos[m, 1] = top;   self.vel[m, 1] *= -damp
        m = self.pos[:, 1] > bot
        self.pos[m, 1] = bot;   self.vel[m, 1] *= -damp


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
        else:                    label = f"{self.label}: {round(v, 3)}"
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
# Main
# ---------------------------------------------------------------------------


def velocity_color(speed):
    """Map per-particle speed to a blue->cyan->white color."""
    t = np.clip(speed / 400.0, 0.0, 1.0)
    # blue (40, 90, 200) -> cyan (90, 220, 255) -> white
    r = 40 + (255 - 40) * t * t
    g = 90 + (240 - 90) * t
    b = 200 + (255 - 200) * t
    out = np.stack([r, g, b], axis=1).astype(np.uint8)
    return out


def main():
    pygame.init()
    pygame.display.set_caption("SPH Fluid")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Menlo,Consolas,monospace", 13)
    small = pygame.font.SysFont("Menlo,Consolas,monospace", 11)
    title_font = pygame.font.SysFont("Menlo,Consolas,monospace", 16, bold=True)

    fluid = Fluid(n=900)

    state = {
        "n":         900,
        "gravity":   700.0,
        "stiffness": 200.0,
        "viscosity": 0.4,
        "mouse_force": 1500.0,
        "running":   True,
    }

    pad = 16
    sx = SIM_W + pad
    sw = HUD_W - 2 * pad

    sliders = [
        Slider(sx,  90, sw, "Particles",   200, 2000, 50,
               lambda: state["n"],
               lambda v: state.update(n=int(v))),
        Slider(sx, 132, sw, "Gravity",     0, 2000, 25,
               lambda: state["gravity"],
               lambda v: (state.update(gravity=float(v)),
                          setattr(fluid, "gravity", float(v)))),
        Slider(sx, 174, sw, "Stiffness",   50, 1000, 10,
               lambda: state["stiffness"],
               lambda v: (state.update(stiffness=float(v)),
                          setattr(fluid, "stiffness", float(v)))),
        Slider(sx, 216, sw, "Viscosity",   0.0, 2.0, 0.02,
               lambda: state["viscosity"],
               lambda v: (state.update(viscosity=float(v)),
                          setattr(fluid, "viscosity", float(v)))),
        Slider(sx, 258, sw, "Mouse force", 0, 5000, 50,
               lambda: state["mouse_force"],
               lambda v: state.update(mouse_force=float(v))),
    ]

    active_slider = None
    last_n = state["n"]
    mouse_mode = 0   # -1 attract, +1 repel, 0 off
    # SPH stiffness is moderate so a single fixed step at 1/120 is stable.
    fixed_dt = 1.0 / 120.0
    SUBSTEPS = 2

    while True:
        clock.tick(FPS)
        mx, my = pygame.mouse.get_pos()

        # ---- Events ------------------------------------------------
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit(0)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit(0)
                elif event.key == pygame.K_SPACE:
                    state["running"] = not state["running"]
                elif event.key == pygame.K_r:
                    fluid.reset(state["n"])
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mxd, myd = event.pos
                if mxd > SIM_W:
                    for s in sliders:
                        if s.hit(mxd, myd):
                            active_slider = s
                            s.update_from_mouse(mxd)
                            break
                else:
                    if event.button == 1:
                        mouse_mode = -1   # attract
                    elif event.button == 3:
                        mouse_mode = +1   # repel
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button in (1, 3):
                    mouse_mode = 0
                active_slider = None
            elif event.type == pygame.MOUSEMOTION:
                if active_slider:
                    active_slider.update_from_mouse(event.pos[0])

        # Particle-count slider changed -> respawn.
        if state["n"] != last_n:
            fluid.reset(state["n"])
            last_n = state["n"]

        # ---- Step --------------------------------------------------
        if state["running"]:
            mforce = mouse_mode * state["mouse_force"]
            mpos = (mx, my) if mx < SIM_W and mouse_mode != 0 else None
            for _ in range(SUBSTEPS):
                fluid.step(fixed_dt, mouse_pos=mpos, mouse_force=mforce,
                           mouse_radius=140.0)

        # ---- Render -------------------------------------------------
        screen.fill(BG)
        # Tank
        pygame.draw.rect(screen, TANK, fluid.tank, border_radius=6)
        pygame.draw.rect(screen, TANK_LINE, fluid.tank, 2, border_radius=6)

        # Particles colored by speed
        speed = np.sqrt(fluid.vel[:, 0] ** 2 + fluid.vel[:, 1] ** 2)
        colors = velocity_color(speed)
        radius = max(2, int(fluid.h * 0.30))
        # Drawing individual circles is the bottleneck at high N. We
        # batch by precomputing positions as ints.
        ix = fluid.pos[:, 0].astype(np.int32)
        iy = fluid.pos[:, 1].astype(np.int32)
        for i in range(fluid.pos.shape[0]):
            pygame.draw.circle(screen, colors[i], (int(ix[i]), int(iy[i])), radius)

        # Mouse cursor indicator
        if mx < SIM_W and mouse_mode != 0:
            col = (130, 230, 255) if mouse_mode < 0 else (255, 150, 130)
            pygame.draw.circle(screen, col, (mx, my), 140, 1)

        # ---- HUD ---------------------------------------------------
        pygame.draw.rect(screen, PANEL_BG, (SIM_W, 0, HUD_W, HEIGHT))
        pygame.draw.line(screen, PANEL_BORDER, (SIM_W, 0), (SIM_W, HEIGHT), 1)
        screen.blit(title_font.render("SPH Fluid", True, ACCENT),
                    (SIM_W + 16, 14))
        screen.blit(small.render(f"FPS: {clock.get_fps():.0f}    "
                                  f"{'RUN' if state['running'] else 'PAUSE'}",
                                  True, TEXT_DIM), (SIM_W + 16, 40))
        screen.blit(small.render(f"particles: {fluid.pos.shape[0]}",
                                  True, TEXT_DIM), (SIM_W + 16, 56))

        for s in sliders:
            s.draw(screen, font)

        help_lines = [
            "Left click + drag: pull",
            "Right click + drag: push",
            "Space: pause / resume",
            "R: reset",
            "Esc: quit",
            "",
            "Stiffness raises pressure;",
            "high values = bouncier but",
            "less stable. Viscosity adds",
            "internal friction (honey vs",
            "water).",
        ]
        y = HEIGHT - 16 * len(help_lines) - 10
        for line in help_lines:
            screen.blit(small.render(line, True, TEXT_DIM), (SIM_W + 16, y))
            y += 16

        pygame.display.flip()


if __name__ == "__main__":
    main()
