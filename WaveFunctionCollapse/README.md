# Wave Function Collapse

A live, tile-based WFC generator. Watch cells in superposition shrink as
constraints propagate from each collapse.

## Run

```bash
cd WaveFunctionCollapse
pip install -r requirements.txt
python wfc.py
```

## How it works

Each grid cell starts in superposition (could be any tile). Every step:

1. **Observe** — find the cell with minimum entropy (fewest options left).
2. **Collapse** — pick one tile from its options, weighted by frequency.
3. **Propagate** — for each neighbor, drop tiles whose touching socket no
   longer has a compatible partner. If a neighbor's wave shrinks, propagate
   from it too.

If a cell ever has zero options, that's a contradiction. The demo restarts
with a fresh seed (toggleable).

The shipped tileset is a "circuit" theme with 6 base shapes (blank,
straight, corner, T, cross, endpoint) auto-rotated to 16 unique tiles.
Sockets are integers (0 = empty, 1 = wire); tiles fit if shared edge codes
match.

## Controls

- **Space** — pause / resume
- **N** — single step
- **Enter** — solve to completion
- **R** — new random seed
- **Esc** — quit
- Sliders on the right adjust grid size, simulation speed, and auto-restart.

The yellow outline marks the cell most recently collapsed; darker cells
have fewer remaining options.
