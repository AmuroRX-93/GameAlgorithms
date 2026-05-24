# Gray-Scott Reaction-Diffusion

Two virtual chemicals U and V live on a grid. They diffuse, U gets fed
in from outside, V decays away, and the reaction `U + 2V -> 3V` keeps
turning U into more V. The whole story is just two coupled PDEs:

```
dU/dt = Du * laplacian(U) - U*V^2 + F*(1 - U)
dV/dt = Dv * laplacian(V) + U*V^2 - (F + k)*V
```

What makes this magical: tiny tweaks to the two scalars `F` (feed rate)
and `k` (kill rate) flip the simulation between **completely different
families of patterns** — coral growth, dividing cells, mazes,
solitons, worms, spots, holes. This is the classical "Turing pattern"
machinery that biology seems to use for animal skin patterns.

## Run

```bash
cd ReactionDiffusion
pip install -r requirements.txt
python rd.py
```

## Controls

- **Left click + drag** in the simulation — paint V (drips reagent in)
- **Space** — pause / resume
- **S** — re-seed a blob in the center
- **R** — randomize seeds
- **C** — clear (resets to U=1, V=0; nothing happens until you paint
  or re-seed)
- **Esc** — quit

The right-side panel has:

- Sliders for `F`, `k`, `Du`, `Dv`, substeps-per-frame, and brush size
- Eight preset buttons that jump to known-good `(F, k)` points: Mitosis,
  Coral, Spots, Maze, Solitons, Worms, Holes, Chaos
- Random / Clear / Pause / Re-seed action buttons

## How it works

- **Grid**: 320 × 200 float32 cells, blitted scaled-up to the window.
  Boundaries are toroidal (wrap-around) so we don't get edge artifacts.
- **Laplacian**: 9-point stencil with weights
  `0.05 0.20 0.05 / 0.20 -1.0 0.20 / 0.05 0.20 0.05`. This is much less
  axis-anisotropic than the 5-point stencil, which matters because RD
  patterns are very sensitive to grid alignment.
- **Time stepping**: forward Euler, dt = 1.0, with `substeps` updates
  per rendered frame so the patterns evolve at a watchable speed.
  Stable because Du and Dv are small.
- **Vectorization**: every laplacian and reaction term is one numpy
  expression over the whole grid. ~5.5 ms per frame at 12 substeps,
  leaving plenty of headroom inside the 16.7 ms budget.
- **Rendering**: V is mapped through a 256-entry RGB colormap (purple
  → teal → yellow), written into a small `pygame.Surface` via
  `surfarray`, then `pygame.transform.scale`'d up to the simulation
  pane.

## Tips

- Hit a preset, wait a few seconds, then nudge `F` or `k` by ±0.001.
  You'll see one pattern family melt into another.
- "Solitons" is fun: paint a few small drops, watch them propagate as
  self-sustaining wave packets that bounce and split.
- "Mitosis" really does look like cell division — every spot grows,
  pinches, and splits into two.
