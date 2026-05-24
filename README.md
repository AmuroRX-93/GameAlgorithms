# Game Algorithms

A collection of 2D game-development algorithms implemented from scratch
in Python with `pygame` and `numpy`. Each project is self-contained,
visualizes the algorithm in real time, and exposes its parameters as
sliders so you can play with them.

**Play in your browser:** <https://amurorx-93.github.io/GameAlgorithms/>
(currently SoloPong; more demos to come.)

Run any of them locally:

```bash
cd <project>
pip install -r requirements.txt
python <main>.py
```

## Projects

### Simulation / Physics
- **[SPHFluid](SPHFluid/)** — 2D Smoothed-Particle Hydrodynamics. ~1500
  particles slosh in a tank with mouse pull/push.
- **[Verlet](Verlet/)** — Verlet integration with distance constraints.
  Cloth, soft bodies, ropes — Gauss-Seidel solver for stability.
- **[ReactionDiffusion](ReactionDiffusion/)** — Gray-Scott Turing
  patterns. Cells divide, mazes grow, solitons propagate; one slider
  changes everything.
- **[Boids](Boids/)** — Reynolds' three rules (separation, alignment,
  cohesion) with a predator and obstacles. Spatial-hash O(N) neighbor
  search.

### Pathfinding / AI
- **[Pathfinding](Pathfinding/)** — Side-by-side BFS, Dijkstra, Greedy,
  A\* visualization on the same maze.
- **[FlowField](FlowField/)** — Reverse-Dijkstra flow-field navigation
  for hundreds of agents sharing a goal (Supreme Commander style).

### Procedural Generation / Geometry
- **[WaveFunctionCollapse](WaveFunctionCollapse/)** — Tile-based WFC
  with AC-3 propagation; tile auto-rotation, persistent rendering.
- **[MarchingSquares](MarchingSquares/)** — Vectorized Marching Squares
  on metaball fields; smooth contours plus filled isosurfaces.

### Rendering
- **[2DVisibility](2DVisibility/)** — Visibility polygon + soft shadow
  raycasting with multiple colored lights.

### Games
- **[SoloPong](SoloPong/)** — Single-player Pong with bricks, multi-ball,
  and CRT-style scanlines. Atari-style 5-zone paddle reflection.
  Includes an AI opponent that never misses and two AI demo modes.
  [Play in browser](https://amurorx-93.github.io/GameAlgorithms/SoloPong/).

## Stack

- Python 3.10+
- [pygame](https://www.pygame.org/) for windowing, input, and 2D
  rendering
- [numpy](https://numpy.org/) for the math-heavy bits (vectorized
  kernels, distance grids, mask buffers)

Each subfolder has its own README explaining the algorithm and
controls.
