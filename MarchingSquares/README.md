# Marching Squares

Real-time iso-contour extraction over a metaball scalar field.

## Run

```bash
cd MarchingSquares
pip install -r requirements.txt
python marching.py
```

## How it works

1. **Sample a scalar field on a grid.** Each "metaball" contributes
   `r² / (distance² + ε)` to every grid point, summed across all balls.
2. **For each cell** (the square between four neighboring sample points),
   threshold the corner values to get a 4-bit "case index" 0..15.
3. **Look up the case** in a 16-entry table. Each case tells you which edges
   of the cell the contour crosses (e.g. case 1 connects the left edge to
   the bottom edge).
4. **Place each crossing.** Either at the edge midpoint (chunky) or by
   linear interpolation between the two corner values (smooth).
5. **Saddle cases** (5 and 10) — diagonally opposite corners both inside —
   are resolved using the average of the four corners (the "asymptotic
   decider"). This keeps the contour from self-intersecting.
6. The same case info is reused to build a filled polygon for the inside
   region of each cell, giving you a solid blob.

## Controls

- Left click empty space — add a ball
- Left click + drag a ball — move it
- Right click a ball — delete it
- Wheel over a ball — resize it
- **Space** — toggle automatic drift
- **G** — sample-grid dots
- **I** — linear interpolation on/off (compare to chunky midpoint mode)
- **C** — clear all balls
- **Esc** — quit
- Sliders on the right tweak cell size, iso threshold, and overlays.

Lower the cell size for a smoother contour at the cost of more cells; raise
the threshold to make the blobs shrink.
