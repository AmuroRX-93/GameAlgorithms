# Verlet Physics

Cloth + rigid bodies via Verlet integration and Jakobsen-style distance
constraint relaxation.

## Run

```bash
cd Verlet
pip install -r requirements.txt
python verlet.py
```

## How it works

Verlet integration replaces the usual `(pos, vel)` pair with `(pos, prev)`.
Each step:

```
new_pos  = pos + (pos - prev) * damping + acceleration * dt^2
prev_pos = pos
pos      = new_pos
```

The implicit velocity is just `pos - prev`. That makes constraint solving
trivial: when you snap a particle's `pos` to satisfy a constraint, its
velocity automatically follows.

A **distance constraint** between particles A and B with rest length `L`:

```
delta  = posB - posA
dist   = |delta|
diff   = (dist - L) / dist
correction = delta * diff / (1/mA + 1/mB)
posA += correction * (1/mA)
posB -= correction * (1/mB)
```

Looping over every constraint several times per frame (Jakobsen
relaxation) makes the system converge to a globally consistent state.
The more **iterations**, the stiffer the cloth/rope feels.

Cloth is just a grid of particles connected by horizontal/vertical
distance constraints. A rigid box is 4 particles + 4 sides + 2
diagonals — the diagonals are what keep it from collapsing into a
parallelogram.

## Controls

- **Left click + drag** — grab a particle and pull (drags pinned cloth
  out of shape, swings ropes around)
- **Right click + drag** — slice through cloth/rope, cutting any
  constraint your cursor crosses
- **B** — drop a fresh rigid box where the cursor is
- **Space** — pause
- **R** — reset the scene
- **Esc** — quit

Sliders on the right adjust gravity, wind, damping, the number of
relaxation iterations per step, and the cloth resolution. Nodes
visualization toggles particle dots on top of the springs.

## Notes

- **Substepping** (2 sub-ticks per frame) keeps things stable when the
  cloth is being yanked hard.
- The integrator is fully numpy-vectorized; the constraint solver runs
  one vectorized "Jacobi" pass per iteration instead of strict
  Gauss-Seidel. This is fast and looks fine for cloth; if you need
  stiffer behavior, raise iterations.
- The ground line at the bottom has friction so settled bodies don't
  slide forever.
