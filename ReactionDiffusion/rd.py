"""
Gray-Scott reaction-diffusion.

Two simulated chemicals U and V live on a grid. They diffuse, U feeds in
from outside, V kills off, and the reaction U + 2V -> 3V converts U into
more V. Two scalar parameters (feed rate F and kill rate k) decide what
kind of pattern emerges:

    dU/dt = Du * laplacian(U) - U*V^2 + F*(1 - U)
    dV/dt = Dv * laplacian(V) + U*V^2 - (F + k)*V

Tiny changes in F and k produce wildly different families of patterns:
spots that divide like cells, labyrinthine mazes, propagating worms,
soliton waves, coral-like growth, and so on. This is the canonical
"Turing pattern" demo — biology gets a lot of its skin patterns from
roughly this same math.

Implementation:
- Laplacian computed with the standard 9-point stencil, fully vectorized
  via four shifted views of V (and U). Boundaries are toroidal (wrap)
  which keeps things simple and removes edge artifacts.
- The grid runs at lower resolution (GRID_W x GRID_H) and gets blitted
  scaled up. We render by mapping V to a colormap into a uint8 RGB
  buffer, then pygame.surfarray it onto a small surface and scale.
- Multiple substeps per frame so the simulation visibly evolves at a
  comfortable rate even with small dt.
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

# Sim grid resolution. Smaller = faster but blockier.
GRID_W = 320
GRID_H = 200
PIX_W = SIM_W / GRID_W
PIX_H = SIM_H / GRID_H

# Colors.
BG = (12, 14, 22)
PANEL_BG = (22, 26, 40)
PANEL_BORDER = (60, 70, 95)
TEXT = (220, 226, 240)
TEXT_DIM = (140, 150, 175)
ACCENT = (245, 210, 80)
SLIDER_TRACK = (50, 58, 80)
SLIDER_FILL = (130, 200, 240)
SLIDER_KNOB = (220, 240, 255)
BTN_BG = (40, 48, 70)
BTN_BG_HOT = (60, 72, 100)
BTN_BORDER = (90, 100, 130)


# ---------------------------------------------------------------------------
# Gray-Scott core
# ---------------------------------------------------------------------------


class GrayScott:
    """Gray-Scott reaction-diffusion on a (rows, cols) grid with toroidal
    boundaries. State is two float32 grids, U and V. We update with a
    simple forward-Euler step and a 9-point laplacian stencil — that's
    the standard recipe and it's stable for dt <= ~1.0 with these
    diffusion rates."""

    def __init__(self, w, h, Du=0.16, Dv=0.08, F=0.060, k=0.062, dt=1.0):
        self.w = w
        self.h = h
        self.Du = Du
        self.Dv = Dv
        self.F = F
        self.k = k
        self.dt = dt
        self.U = np.ones((h, w), dtype=np.float32)
        self.V = np.zeros((h, w), dtype=np.float32)
        self.seed_blob()

    def seed_blob(self):
        """Drop a small square of V into the center to kick things off.
        Without a seed the equilibrium U=1, V=0 is stable forever."""
        self.U[...] = 1.0
        self.V[...] = 0.0
        cy, cx = self.h // 2, self.w // 2
        r = 8
        self.U[cy - r:cy + r, cx - r:cx + r] = 0.5
        self.V[cy - r:cy + r, cx - r:cx + r] = 0.25
        # Sprinkle a little noise so symmetry breaks.
        rng = np.random.default_rng(0)
        noise = rng.uniform(-0.01, 0.01, size=self.V.shape).astype(np.float32)
        self.V += noise * 0.5
        np.clip(self.V, 0.0, 1.0, out=self.V)

    def randomize(self):
        rng = np.random.default_rng()
        self.U[...] = 1.0
        self.V[...] = 0.0
        for _ in range(40):
            cy = rng.integers(8, self.h - 8)
            cx = rng.integers(8, self.w - 8)
            r = int(rng.integers(3, 8))
            self.U[cy - r:cy + r, cx - r:cx + r] = 0.5
            self.V[cy - r:cy + r, cx - r:cx + r] = 0.25

    def clear(self):
        self.U[...] = 1.0
        self.V[...] = 0.0

    def step(self, n=1):
        """Advance the simulation by `n` substeps."""
        U = self.U
        V = self.V
        Du = self.Du
        Dv = self.Dv
        F = self.F
        k = self.k
        dt = self.dt
        for _ in range(n):
            # 9-point laplacian with wrap-around boundaries. The 9-point
            # variant is noticeably less anisotropic than the 5-point one,
            # which matters for Gray-Scott because the patterns are very
            # sensitive to grid axis alignment.
            #
            # Weights:  0.05  0.20  0.05
            #           0.20 -1.00  0.20
            #           0.05  0.20  0.05
            #
            # np.roll is O(NM) but vectorized, so it's plenty fast.
            U_up    = np.roll(U, -1, axis=0)
            U_dn    = np.roll(U,  1, axis=0)
            U_lf    = np.roll(U, -1, axis=1)
            U_rt    = np.roll(U,  1, axis=1)
            U_uplf  = np.roll(U_up, -1, axis=1)
            U_uprt  = np.roll(U_up,  1, axis=1)
            U_dnlf  = np.roll(U_dn, -1, axis=1)
            U_dnrt  = np.roll(U_dn,  1, axis=1)
            lapU = (0.20 * (U_up + U_dn + U_lf + U_rt)
                    + 0.05 * (U_uplf + U_uprt + U_dnlf + U_dnrt)
                    - U)

            V_up    = np.roll(V, -1, axis=0)
            V_dn    = np.roll(V,  1, axis=0)
            V_lf    = np.roll(V, -1, axis=1)
            V_rt    = np.roll(V,  1, axis=1)
            V_uplf  = np.roll(V_up, -1, axis=1)
            V_uprt  = np.roll(V_up,  1, axis=1)
            V_dnlf  = np.roll(V_dn, -1, axis=1)
            V_dnrt  = np.roll(V_dn,  1, axis=1)
            lapV = (0.20 * (V_up + V_dn + V_lf + V_rt)
                    + 0.05 * (V_uplf + V_uprt + V_dnlf + V_dnrt)
                    - V)

            uvv = U * V * V
            U += (Du * lapU - uvv + F * (1.0 - U)) * dt
            V += (Dv * lapV + uvv - (F + k) * V) * dt
            np.clip(U, 0.0, 1.0, out=U)
            np.clip(V, 0.0, 1.0, out=V)

    def paint(self, cx, cy, radius, amount=0.5):
        """Inject V into a circular region, like dripping reagent."""
        x0 = max(0, cx - radius); x1 = min(self.w, cx + radius + 1)
        y0 = max(0, cy - radius); y1 = min(self.h, cy + radius + 1)
        if x0 >= x1 or y0 >= y1:
            return
        ys = np.arange(y0, y1)[:, None]
        xs = np.arange(x0, x1)[None, :]
        d2 = (xs - cx) ** 2 + (ys - cy) ** 2
        mask = d2 <= radius * radius
        falloff = np.maximum(0.0, 1.0 - np.sqrt(d2) / radius).astype(np.float32)
        self.V[y0:y1, x0:x1] += (mask * falloff * amount).astype(np.float32)
        self.U[y0:y1, x0:x1] -= (mask * falloff * amount * 0.5).astype(np.float32)
        np.clip(self.U, 0.0, 1.0, out=self.U)
        np.clip(self.V, 0.0, 1.0, out=self.V)


# ---------------------------------------------------------------------------
# Colormap
# ---------------------------------------------------------------------------


def build_colormap():
    """Build a 256-entry RGB lookup table that goes from deep purple
    through teal to warm yellow. We'll index it with V (scaled to 0..255)."""
    stops = [
        (0.00, (10,  10,  35)),
        (0.20, (35,  20,  85)),
        (0.40, (40,  90, 140)),
        (0.55, (40, 170, 170)),
        (0.70, (200, 220, 110)),
        (0.85, (255, 180,  70)),
        (1.00, (255, 240, 200)),
    ]
    t = np.linspace(0.0, 1.0, 256)
    cmap = np.zeros((256, 3), dtype=np.float32)
    for i in range(len(stops) - 1):
        t0, c0 = stops[i]
        t1, c1 = stops[i + 1]
        m = (t >= t0) & (t <= t1)
        a = (t[m] - t0) / max(1e-9, (t1 - t0))
        cmap[m, 0] = c0[0] + (c1[0] - c0[0]) * a
        cmap[m, 1] = c0[1] + (c1[1] - c0[1]) * a
        cmap[m, 2] = c0[2] + (c1[2] - c0[2]) * a
    return np.clip(cmap, 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Presets — well-known (F, k) sweet spots that give visually distinct
# pattern families. Du/Dv/dt are kept fixed.
# ---------------------------------------------------------------------------
PRESETS = [
    ("Mitosis",   0.0367, 0.0649),  # cells split and divide
    ("Coral",     0.0545, 0.0620),  # coral-like growth
    ("Spots",     0.0294, 0.0570),  # stationary spots
    ("Maze",      0.0290, 0.0570),  # labyrinthine stripes
    ("Solitons",  0.0300, 0.0620),  # propagating wave packets
    ("Worms",     0.0780, 0.0610),  # wriggling stripes
    ("Holes",     0.0390, 0.0580),  # negative spots
    ("Chaos",     0.0260, 0.0510),  # noisy, never settles
]


# ---------------------------------------------------------------------------
# UI: sliders and buttons
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
        else:                    label = f"{self.label}: {round(v, 4)}"
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


class Button:
    def __init__(self, x, y, w, h, label, onclick):
        self.rect = pygame.Rect(x, y, w, h)
        self.label = label
        self.onclick = onclick
        self.hot = False

    def draw(self, surface, font):
        bg = BTN_BG_HOT if self.hot else BTN_BG
        pygame.draw.rect(surface, bg, self.rect, border_radius=4)
        pygame.draw.rect(surface, BTN_BORDER, self.rect, 1, border_radius=4)
        txt = font.render(self.label, True, TEXT)
        surface.blit(txt, txt.get_rect(center=self.rect.center))

    def hit(self, mx, my):
        return self.rect.collidepoint(mx, my)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    pygame.init()
    pygame.display.set_caption("Gray-Scott Reaction-Diffusion")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Menlo,Consolas,monospace", 13)
    small = pygame.font.SysFont("Menlo,Consolas,monospace", 11)
    title_font = pygame.font.SysFont("Menlo,Consolas,monospace", 16, bold=True)

    sim = GrayScott(GRID_W, GRID_H, F=0.0367, k=0.0649)  # default = Mitosis
    cmap = build_colormap()

    state = {
        "F": sim.F,
        "k": sim.k,
        "Du": sim.Du,
        "Dv": sim.Dv,
        "substeps": 12,
        "brush": 8,
        "running": True,
    }

    # Persistent surface for the simulation (we surfarray into this then
    # scale up to fill SIM_W x SIM_H).
    sim_surf = pygame.Surface((GRID_W, GRID_H))

    # Sliders.
    pad = 16
    sx = SIM_W + pad
    sw = HUD_W - 2 * pad
    sliders = [
        Slider(sx,  90, sw, "Feed F",   0.0, 0.10, 0.0001,
               lambda: state["F"],
               lambda v: (state.update(F=float(v)), setattr(sim, "F", float(v)))),
        Slider(sx, 132, sw, "Kill k",   0.0, 0.10, 0.0001,
               lambda: state["k"],
               lambda v: (state.update(k=float(v)), setattr(sim, "k", float(v)))),
        Slider(sx, 174, sw, "Du",       0.0, 0.30, 0.005,
               lambda: state["Du"],
               lambda v: (state.update(Du=float(v)), setattr(sim, "Du", float(v)))),
        Slider(sx, 216, sw, "Dv",       0.0, 0.30, 0.005,
               lambda: state["Dv"],
               lambda v: (state.update(Dv=float(v)), setattr(sim, "Dv", float(v)))),
        Slider(sx, 258, sw, "Substeps", 1, 30, 1,
               lambda: state["substeps"],
               lambda v: state.update(substeps=int(v))),
        Slider(sx, 300, sw, "Brush",    1, 30, 1,
               lambda: state["brush"],
               lambda v: state.update(brush=int(v))),
    ]

    # Preset buttons (two columns).
    btns = []
    btn_y = 350
    for i, (name, F, k) in enumerate(PRESETS):
        col = i % 2
        row = i // 2
        bw = (sw - 8) // 2
        bx = sx + col * (bw + 8)
        by = btn_y + row * 30
        def make_cb(F=F, k=k):
            def cb():
                sim.F = F; sim.k = k
                state["F"] = F; state["k"] = k
                sim.seed_blob()
            return cb
        btns.append(Button(bx, by, bw, 24, name, make_cb()))

    # Action buttons at the bottom.
    action_y = btn_y + ((len(PRESETS) + 1) // 2) * 30 + 12
    bw = (sw - 8) // 2
    btns.append(Button(sx,           action_y, bw, 24, "Random",
                       lambda: sim.randomize()))
    btns.append(Button(sx + bw + 8,  action_y, bw, 24, "Clear",
                       lambda: sim.clear()))
    btns.append(Button(sx,           action_y + 30, bw, 24, "Pause",
                       lambda: state.update(running=not state["running"])))
    btns.append(Button(sx + bw + 8,  action_y + 30, bw, 24, "Re-seed",
                       lambda: sim.seed_blob()))

    active_slider = None
    painting = False

    def screen_to_grid(mx, my):
        gx = int(mx / PIX_W)
        gy = int(my / PIX_H)
        return (max(0, min(GRID_W - 1, gx)),
                max(0, min(GRID_H - 1, gy)))

    while True:
        dt_ms = clock.tick(FPS)

        # ---- Events -------------------------------------------------
        mx, my = pygame.mouse.get_pos()
        for b in btns:
            b.hot = b.hit(mx, my)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit(0)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit(0)
                elif event.key == pygame.K_SPACE:
                    state["running"] = not state["running"]
                elif event.key == pygame.K_r:
                    sim.randomize()
                elif event.key == pygame.K_c:
                    sim.clear()
                elif event.key == pygame.K_s:
                    sim.seed_blob()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                if mx > SIM_W:
                    handled = False
                    for s in sliders:
                        if s.hit(mx, my):
                            active_slider = s
                            s.update_from_mouse(mx)
                            handled = True
                            break
                    if not handled:
                        for b in btns:
                            if b.hit(mx, my):
                                b.onclick()
                                break
                else:
                    if event.button == 1:
                        painting = True
                        gx, gy = screen_to_grid(mx, my)
                        sim.paint(gx, gy, state["brush"])
            elif event.type == pygame.MOUSEBUTTONUP:
                active_slider = None
                if event.button == 1:
                    painting = False
            elif event.type == pygame.MOUSEMOTION:
                if active_slider:
                    active_slider.update_from_mouse(event.pos[0])
                if painting and event.pos[0] < SIM_W:
                    gx, gy = screen_to_grid(*event.pos)
                    sim.paint(gx, gy, state["brush"])

        # ---- Step --------------------------------------------------
        if state["running"]:
            sim.step(state["substeps"])

        # ---- Render -------------------------------------------------
        # Map V into a uint8 index, look up colors, blit scaled.
        v_norm = np.clip(sim.V * 255.0, 0, 255).astype(np.uint8)
        rgb = cmap[v_norm]                          # (H, W, 3)
        # surfarray expects (W, H, 3) so transpose.
        pygame.surfarray.blit_array(sim_surf, rgb.transpose(1, 0, 2))
        scaled = pygame.transform.scale(sim_surf, (SIM_W, SIM_H))
        screen.blit(scaled, (0, 0))

        # Brush indicator.
        if mx < SIM_W:
            r = max(2, int(state["brush"] * (PIX_W + PIX_H) * 0.5))
            pygame.draw.circle(screen, (255, 255, 255), (mx, my), r, 1)

        pygame.draw.rect(screen, PANEL_BORDER, (0, 0, SIM_W, SIM_H), 1)

        # ---- HUD ----------------------------------------------------
        pygame.draw.rect(screen, PANEL_BG, (SIM_W, 0, HUD_W, HEIGHT))
        pygame.draw.line(screen, PANEL_BORDER, (SIM_W, 0), (SIM_W, HEIGHT), 1)
        screen.blit(title_font.render("Gray-Scott RD", True, ACCENT),
                    (SIM_W + 16, 14))
        screen.blit(small.render(f"FPS: {clock.get_fps():.0f}    "
                                  f"{'RUN' if state['running'] else 'PAUSE'}",
                                  True, TEXT_DIM), (SIM_W + 16, 40))
        screen.blit(small.render(f"grid {GRID_W}x{GRID_H}",
                                  True, TEXT_DIM), (SIM_W + 16, 56))
        screen.blit(small.render(f"F={state['F']:.4f}  k={state['k']:.4f}",
                                  True, TEXT_DIM), (SIM_W + 16, 72))

        for s in sliders:
            s.draw(screen, font)

        # Section label for presets.
        screen.blit(small.render("PRESETS", True, TEXT_DIM),
                    (SIM_W + 16, btn_y - 14))
        for b in btns:
            b.draw(screen, small)

        help_lines = [
            "Left click + drag: paint V",
            "Space: pause/resume",
            "S: re-seed center",
            "R: randomize",
            "C: clear",
            "Esc: quit",
            "",
            "Tiny F/k changes flip",
            "the pattern family. Try",
            "the presets, then nudge",
            "F & k by 0.001 each.",
        ]
        y = HEIGHT - 16 * len(help_lines) - 10
        for line in help_lines:
            screen.blit(small.render(line, True, TEXT_DIM), (SIM_W + 16, y))
            y += 16

        pygame.display.flip()


if __name__ == "__main__":
    main()
