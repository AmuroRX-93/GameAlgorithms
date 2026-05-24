# Flow-Field Pathfinding

When hundreds of agents share a goal, running A\* per agent is wasteful.
Instead we run **one** reverse Dijkstra from the goal, build a flow field
out of the resulting distance grid, and let every agent steer by sampling
the field. This is the trick *Supreme Commander*, *Planetary Annihilation*,
and *StarCraft II*'s influence map use.

## Run

```bash
cd FlowField
pip install -r requirements.txt
python flowfield.py
```

## Pipeline

1. **Cost field** — per-cell traversal cost. Walls are infinite, sand is
   slow (cost 4), grass is fast (cost 1).
2. **Integration field** — Dijkstra from the goal cell outward. Each
   cell's value is the cheapest path-cost to the goal. Diagonal moves
   cost √2 and corner-cutting through walls is forbidden.
3. **Flow field** — for each cell, look at its 8 neighbors and pick the
   one with the lowest integration cost. The unit vector pointing toward
   that neighbor is the cell's "flow direction". Vectorized via
   `np.argmin` over a stacked tensor of shifted views.
4. **Agents** — sample the flow at their current cell, steer toward it
   with mild inertia, and apply a small mutual-repulsion force so they
   don't pile up on the goal.

The field is recomputed only when the goal moves or you paint terrain;
agents query it for free every frame.

## Controls

- **Left click** in the simulation — set a new goal
- **Right click + drag** — paint terrain in the current mode
- **1** wall, **2** sand, **3** erase (back to grass)
- **H** toggle the distance heatmap (low = blue, high = red)
- **A** toggle the per-cell flow arrows
- **R** clear all walls
- **C** delete all agents
- Sliders on the right: max speed, separation force, agent count

## Notes

- The N\*N separation force is fine up to ~1500 agents on this grid; if
  you wanted to scale further you'd swap it for a spatial hash like in
  the Boids project.
- The cost-field is a `(rows, cols)` numpy array, so painting a wall is
  just `cost[y, x] = WALL_COST` followed by a single re-integration.
- Diagonal corner-cutting is disabled: an agent can't squeeze through
  the corner where two walls meet diagonally.
