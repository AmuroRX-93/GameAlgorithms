# SPH Fluid

2D smoothed-particle hydrodynamics with a position-based separation
pass to keep things stable. Particles flow, slosh, splash off walls,
and react to mouse pulls and pushes.

## Run

```bash
cd SPHFluid
pip install -r requirements.txt
python sph.py
```

## What's happening per step

1. **Spatial-hash neighbor search** — every particle is bucketed into a
   uniform grid with cell size `h` (the kernel radius), so each
   particle only checks the 9 cells around it. The Python loop is over
   non-empty cells, not particles, and inside each cell pair we use
   `np.meshgrid` so the cartesian product is one numpy op.
2. **Density** — `ρ_i = Σ_j m · W_poly6(|r_i - r_j|², h)`. The poly6
   kernel is from Müller et al. 2003.
3. **Pressure** — clamped linear EOS, `p_i = max(0, k · (ρ_i − ρ₀))`.
   Negative pressures are zeroed out so neighbors can never *pull* each
   other together — only push apart. This is the single most important
   tweak for keeping a game-style fluid stable: standard SPH lets
   surface particles get sucked into the bulk, which is what causes
   the entire pile to flatten into a single line on the floor.
4. **Forces** — pressure gradient via the spiky kernel, viscosity via
   the viscosity-kernel laplacian, then gravity, then the optional
   mouse force.
5. **Symplectic Euler** — velocity then position.
6. **Separation projection** — three Jacobi-style passes that find any
   pair of particles closer than `h·0.55` and project them apart so
   they're exactly that far. This catches the cases where the
   pressure gradient alone wouldn't be strong enough at the chosen dt
   to prevent interpenetration. It's effectively a coarse PBF density
   constraint operating directly on spacing.
7. **Tank collision** — clamp positions to the tank rectangle and
   reflect the velocity component normal to the wall with light
   damping.

The rest density is auto-calibrated at startup: we drop a 11×11 patch
of virtual particles at the same spacing as `spawn_block` uses, sum
their poly6 contributions, and use that as `ρ₀`. Without this step
the EOS would be wrong by orders of magnitude.

## Controls

- **Left click + drag** in the simulation — pull fluid toward the cursor
- **Right click + drag** — push fluid away from the cursor
- **Space** — pause / resume
- **R** — reset (re-spawn the block)
- **Esc** — quit

Sliders on the right:

- **Particles** — 200 to 2000
- **Gravity** — 0 to 2000 px/s²
- **Stiffness** — pressure constant `k`. Higher = more incompressible
  but at some point dt isn't small enough and it explodes.
- **Viscosity** — internal friction. Low = water, high = honey.
- **Mouse force** — strength of the cursor pull/push.

## Performance

At default settings (900 particles, h=16, 2 substeps per frame, dt=1/120):

- Steady-state cost: ~7 ms/frame on this hardware (60 fps with budget
  to spare).
- 1500 particles: ~10 ms/frame.
- The neighbor-pair construction is the bottleneck; everything after
  it is one big numpy expression.

## Notes

- The separation pass relies on `np.add.at` over a 1D view of one
  position column. Note that `np.add.at(self.pos[:, 0], ...)` can land
  on a temporary copy if the column view isn't contiguous — we take a
  named local for the column so the buffered scatter writes back to
  the real storage.
- Velocity is clamped to a max of 600 px/s. Without that, an aggressive
  mouse drag could shove a particle a full kernel-width per frame and
  overshoot all the constraints.
