"""
Pathfinding Algorithms Comparison
Visualizes BFS, Dijkstra, Greedy Best-First, and A* on the same grid.

Each algorithm runs in lock-step (one frontier-pop per "tick"), so you can
literally watch them race. The visualization shows:
  - light blue cells: explored (closed set)
  - bright yellow:    current frontier (open set)
  - white path:       final shortest path (when found)
  - dark blue cells:  walls
  - green:            start    red: goal

Controls:
  - Left click & drag:   paint walls
  - Right click & drag:  erase walls
  - S then click:        place START
  - G then click:        place GOAL
  - SPACE:               run / pause
  - N:                   step one frame
  - R:                   reset (keep map)
  - C:                   clear map
  - M:                   randomize maze
  - 1/2/3/4:             toggle BFS / Dijkstra / Greedy / A*
  - +/-:                 speed up / slow down
  - ESC:                 quit
"""

import heapq
import math
import random
import sys
import time
from collections import deque

import pygame

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CELL = 16                    # Pixel size of each grid cell.
COLS, ROWS = 34, 24          # Grid is 34 x 24 cells per panel.
PANEL_W = COLS * CELL
PANEL_H = ROWS * CELL
PANELS_X = 2                 # 2x2 layout of algorithm panels.
PANELS_Y = 2
GAP = 16                     # Pixel gap between panels.
HUD_H = 50                   # Top header height.

WIDTH = PANELS_X * PANEL_W + (PANELS_X + 1) * GAP
HEIGHT = HUD_H + PANELS_Y * PANEL_H + (PANELS_Y + 1) * GAP
FPS = 60

# Movement: 8-directional (with sqrt(2) cost for diagonals).
NEIGHBORS_8 = [
    (-1, -1, math.sqrt(2)), (0, -1, 1.0), (1, -1, math.sqrt(2)),
    (-1,  0, 1.0),                         (1,  0, 1.0),
    (-1,  1, math.sqrt(2)), (0,  1, 1.0), (1,  1, math.sqrt(2)),
]

# Color palette.
BG = (16, 18, 28)
PANEL_BG = (28, 32, 46)
GRID_LINE = (40, 46, 64)
WALL = (50, 60, 92)
EMPTY = (70, 82, 110)
EXPLORED = (88, 130, 200)
FRONTIER = (245, 210, 80)
PATH = (240, 245, 255)
START = (90, 220, 130)
GOAL = (235, 90, 90)
TEXT = (220, 224, 240)
TEXT_DIM = (140, 150, 180)
HEADER = (245, 210, 80)


# ---------------------------------------------------------------------------
# Algorithms
# ---------------------------------------------------------------------------
#
# Each algorithm is implemented as a small generator-like state machine so the
# main loop can step them in lock-step. Common state:
#   came_from[node] = parent node on the discovered path
#   cost_so_far[node] = best known g-cost from start (Dijkstra/A* only)
#   open / closed sets
#   `done`, `path` once the goal is reached
#
# `step()` pops one node from the frontier and expands its neighbors.
# `step()` returns True when the algorithm is finished (success or failure).


def octile(a, b):
    """Octile distance heuristic for 8-connected grids (admissible)."""
    dx = abs(a[0] - b[0])
    dy = abs(a[1] - b[1])
    return (dx + dy) + (math.sqrt(2) - 2) * min(dx, dy)


class Search:
    """Base class: shared state + path reconstruction."""

    name = "Base"
    color = (200, 200, 200)

    def __init__(self, grid, start, goal):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.came_from = {start: None}
        self.closed = set()
        self.expansions = 0  # How many cells we've fully expanded.
        self.done = False
        self.success = False
        self.path = []
        self.elapsed_us = 0  # Cumulative microseconds spent stepping.

    def neighbors(self, node):
        x, y = node
        for dx, dy, w in NEIGHBORS_8:
            nx, ny = x + dx, y + dy
            if 0 <= nx < COLS and 0 <= ny < ROWS and not self.grid[ny][nx]:
                # Disallow corner-cutting through diagonal walls.
                if dx and dy:
                    if self.grid[y][nx] or self.grid[ny][x]:
                        continue
                yield (nx, ny), w

    def reconstruct(self, end):
        node = end
        out = []
        while node is not None:
            out.append(node)
            node = self.came_from[node]
        out.reverse()
        return out

    def step(self):
        """Return True if finished. Override in subclasses."""
        raise NotImplementedError

    def frontier_set(self):
        """Cells currently in the open set, for visualization."""
        return set()


class BFS(Search):
    name = "BFS"
    color = (90, 200, 240)

    def __init__(self, grid, start, goal):
        super().__init__(grid, start, goal)
        self.open = deque([start])
        self.in_open = {start}

    def frontier_set(self):
        return self.in_open

    def step(self):
        if self.done:
            return True
        if not self.open:
            self.done = True
            return True
        t0 = time.perf_counter_ns()
        node = self.open.popleft()
        self.in_open.discard(node)
        if node in self.closed:
            self.elapsed_us += (time.perf_counter_ns() - t0) // 1000
            return False
        self.closed.add(node)
        self.expansions += 1
        if node == self.goal:
            self.path = self.reconstruct(node)
            self.success = True
            self.done = True
            self.elapsed_us += (time.perf_counter_ns() - t0) // 1000
            return True
        for nxt, _w in self.neighbors(node):
            if nxt not in self.came_from:
                self.came_from[nxt] = node
                self.open.append(nxt)
                self.in_open.add(nxt)
        self.elapsed_us += (time.perf_counter_ns() - t0) // 1000
        return False


class Dijkstra(Search):
    name = "Dijkstra"
    color = (200, 140, 240)

    def __init__(self, grid, start, goal):
        super().__init__(grid, start, goal)
        self.cost_so_far = {start: 0.0}
        self.pq = [(0.0, 0, start)]
        self._counter = 1
        self.in_open = {start}

    def frontier_set(self):
        return self.in_open

    def step(self):
        if self.done:
            return True
        if not self.pq:
            self.done = True
            return True
        t0 = time.perf_counter_ns()
        cost, _, node = heapq.heappop(self.pq)
        self.in_open.discard(node)
        if node in self.closed:
            self.elapsed_us += (time.perf_counter_ns() - t0) // 1000
            return False
        self.closed.add(node)
        self.expansions += 1
        if node == self.goal:
            self.path = self.reconstruct(node)
            self.success = True
            self.done = True
            self.elapsed_us += (time.perf_counter_ns() - t0) // 1000
            return True
        for nxt, w in self.neighbors(node):
            new_cost = cost + w
            if new_cost < self.cost_so_far.get(nxt, math.inf):
                self.cost_so_far[nxt] = new_cost
                self.came_from[nxt] = node
                heapq.heappush(self.pq, (new_cost, self._counter, nxt))
                self._counter += 1
                self.in_open.add(nxt)
        self.elapsed_us += (time.perf_counter_ns() - t0) // 1000
        return False


class GreedyBestFirst(Search):
    name = "Greedy"
    color = (240, 170, 90)

    def __init__(self, grid, start, goal):
        super().__init__(grid, start, goal)
        self.pq = [(octile(start, goal), 0, start)]
        self._counter = 1
        self.in_open = {start}

    def frontier_set(self):
        return self.in_open

    def step(self):
        if self.done:
            return True
        if not self.pq:
            self.done = True
            return True
        t0 = time.perf_counter_ns()
        _h, _, node = heapq.heappop(self.pq)
        self.in_open.discard(node)
        if node in self.closed:
            self.elapsed_us += (time.perf_counter_ns() - t0) // 1000
            return False
        self.closed.add(node)
        self.expansions += 1
        if node == self.goal:
            self.path = self.reconstruct(node)
            self.success = True
            self.done = True
            self.elapsed_us += (time.perf_counter_ns() - t0) // 1000
            return True
        for nxt, _w in self.neighbors(node):
            if nxt not in self.came_from:
                self.came_from[nxt] = node
                heapq.heappush(self.pq, (octile(nxt, self.goal), self._counter, nxt))
                self._counter += 1
                self.in_open.add(nxt)
        self.elapsed_us += (time.perf_counter_ns() - t0) // 1000
        return False


class AStar(Search):
    name = "A*"
    color = (130, 230, 150)

    def __init__(self, grid, start, goal):
        super().__init__(grid, start, goal)
        self.cost_so_far = {start: 0.0}
        self.pq = [(octile(start, goal), 0, start)]
        self._counter = 1
        self.in_open = {start}

    def frontier_set(self):
        return self.in_open

    def step(self):
        if self.done:
            return True
        if not self.pq:
            self.done = True
            return True
        t0 = time.perf_counter_ns()
        _f, _, node = heapq.heappop(self.pq)
        self.in_open.discard(node)
        if node in self.closed:
            self.elapsed_us += (time.perf_counter_ns() - t0) // 1000
            return False
        self.closed.add(node)
        self.expansions += 1
        if node == self.goal:
            self.path = self.reconstruct(node)
            self.success = True
            self.done = True
            self.elapsed_us += (time.perf_counter_ns() - t0) // 1000
            return True
        g = self.cost_so_far[node]
        for nxt, w in self.neighbors(node):
            new_cost = g + w
            if new_cost < self.cost_so_far.get(nxt, math.inf):
                self.cost_so_far[nxt] = new_cost
                self.came_from[nxt] = node
                f = new_cost + octile(nxt, self.goal)
                heapq.heappush(self.pq, (f, self._counter, nxt))
                self._counter += 1
                self.in_open.add(nxt)
        self.elapsed_us += (time.perf_counter_ns() - t0) // 1000
        return False


# ---------------------------------------------------------------------------
# Map generation & utilities
# ---------------------------------------------------------------------------


def empty_grid():
    return [[False] * COLS for _ in range(ROWS)]


def random_maze(rng=None, density=0.28):
    """Random walls with a guaranteed-clear border around start/goal."""
    rng = rng or random.Random()
    grid = empty_grid()
    for y in range(ROWS):
        for x in range(COLS):
            grid[y][x] = rng.random() < density

    # Carve a few long corridors so the maze isn't pure noise.
    for _ in range(rng.randint(3, 6)):
        if rng.random() < 0.5:
            y = rng.randint(2, ROWS - 3)
            x0 = rng.randint(0, COLS // 2)
            x1 = rng.randint(COLS // 2, COLS - 1)
            for x in range(x0, x1 + 1):
                grid[y][x] = False
        else:
            x = rng.randint(2, COLS - 3)
            y0 = rng.randint(0, ROWS // 2)
            y1 = rng.randint(ROWS // 2, ROWS - 1)
            for y in range(y0, y1 + 1):
                grid[y][x] = False
    return grid


def clear_around(grid, cell, radius=1):
    cx, cy = cell
    for y in range(max(0, cy - radius), min(ROWS, cy + radius + 1)):
        for x in range(max(0, cx - radius), min(COLS, cx + radius + 1)):
            grid[y][x] = False


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def panel_origin(idx):
    """Return (px, py) screen coords of the top-left of panel `idx`."""
    col = idx % PANELS_X
    row = idx // PANELS_X
    px = GAP + col * (PANEL_W + GAP)
    py = HUD_H + GAP + row * (PANEL_H + GAP)
    return px, py


def cell_rect(panel_xy, cx, cy):
    px, py = panel_xy
    return pygame.Rect(px + cx * CELL, py + cy * CELL, CELL, CELL)


def draw_panel(surface, font, small_font, idx, search):
    px, py = panel_origin(idx)
    panel_rect = pygame.Rect(px, py, PANEL_W, PANEL_H)
    pygame.draw.rect(surface, PANEL_BG, panel_rect)

    # Cells.
    grid = search.grid
    closed = search.closed
    frontier = search.frontier_set()
    path_set = set(search.path)
    start = search.start
    goal = search.goal

    for y in range(ROWS):
        for x in range(COLS):
            r = cell_rect((px, py), x, y)
            if grid[y][x]:
                pygame.draw.rect(surface, WALL, r)
                continue
            if (x, y) in path_set:
                pygame.draw.rect(surface, PATH, r)
            elif (x, y) in frontier:
                pygame.draw.rect(surface, FRONTIER, r)
            elif (x, y) in closed:
                pygame.draw.rect(surface, EXPLORED, r)
            else:
                pygame.draw.rect(surface, EMPTY, r)

    # Start / goal markers (drawn on top).
    sr = cell_rect((px, py), *start)
    gr = cell_rect((px, py), *goal)
    pygame.draw.rect(surface, START, sr)
    pygame.draw.rect(surface, GOAL, gr)

    # Light grid lines.
    for i in range(COLS + 1):
        x = px + i * CELL
        pygame.draw.line(surface, GRID_LINE, (x, py), (x, py + PANEL_H))
    for j in range(ROWS + 1):
        y = py + j * CELL
        pygame.draw.line(surface, GRID_LINE, (px, y), (px + PANEL_W, y))

    # Per-panel header strip (algorithm name + stats).
    title = f"{search.name}"
    status = "running" if not search.done else ("found" if search.success else "no path")
    plen = sum(
        math.hypot(search.path[i + 1][0] - search.path[i][0],
                   search.path[i + 1][1] - search.path[i][1])
        for i in range(len(search.path) - 1)
    )
    stats = f"expanded={search.expansions}  len={plen:.1f}  t={search.elapsed_us / 1000:.1f}ms  ({status})"

    title_surf = font.render(title, True, search.color)
    stats_surf = small_font.render(stats, True, TEXT_DIM)
    surface.blit(title_surf, (px + 6, py + 4))
    surface.blit(stats_surf, (px + 6, py + 4 + title_surf.get_height()))

    # Panel border.
    pygame.draw.rect(surface, search.color, panel_rect, width=2)


def draw_header(surface, font, small_font, paused, speed, mode, enabled):
    pygame.draw.rect(surface, BG, (0, 0, WIDTH, HUD_H))
    title = font.render("Pathfinding Comparison", True, HEADER)
    surface.blit(title, (GAP, 8))

    state = "PAUSED" if paused else "RUNNING"
    info = f"{state}  |  speed: {speed} steps/frame  |  click mode: {mode}"
    info_surf = small_font.render(info, True, TEXT)
    surface.blit(info_surf, (GAP, 8 + title.get_height()))

    flags = " ".join(
        f"[{'X' if enabled[i] else ' '}] {n}" for i, n in enumerate(["BFS", "Dijkstra", "Greedy", "A*"])
    )
    flag_surf = small_font.render(flags, True, TEXT_DIM)
    surface.blit(flag_surf, (WIDTH - flag_surf.get_width() - GAP, 8))

    keys = "SPACE run | N step | R reset | C clear | M maze | S/G start/goal | 1-4 toggle | +/- speed"
    keys_surf = small_font.render(keys, True, TEXT_DIM)
    surface.blit(keys_surf, (WIDTH - keys_surf.get_width() - GAP, 8 + title.get_height()))


# ---------------------------------------------------------------------------
# Hit-testing: which panel + cell is the mouse over?
# ---------------------------------------------------------------------------


def mouse_to_cell(mx, my):
    """Return ((panel_idx, cx, cy)) or None."""
    for idx in range(PANELS_X * PANELS_Y):
        px, py = panel_origin(idx)
        if px <= mx < px + PANEL_W and py <= my < py + PANEL_H:
            cx = (mx - px) // CELL
            cy = (my - py) // CELL
            if 0 <= cx < COLS and 0 <= cy < ROWS:
                return idx, cx, cy
    return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

ALGO_CLASSES = [BFS, Dijkstra, GreedyBestFirst, AStar]


def make_searches(grid, start, goal):
    """Build a fresh search instance per enabled algorithm.

    Each search gets its own copy of the grid so wall edits don't bleed across
    panels mid-run.
    """
    return [cls([row[:] for row in grid], start, goal) for cls in ALGO_CLASSES]


def main():
    pygame.init()
    pygame.display.set_caption("Pathfinding Algorithms Comparison")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Menlo,Consolas,monospace", 16, bold=True)
    small_font = pygame.font.SysFont("Menlo,Consolas,monospace", 12)

    grid = random_maze(random.Random(7), density=0.25)
    start = (2, ROWS // 2)
    goal = (COLS - 3, ROWS // 2)
    clear_around(grid, start, 1)
    clear_around(grid, goal, 1)

    enabled = [True, True, True, True]
    searches = make_searches(grid, start, goal)
    paused = True
    speed = 4                  # Steps per algorithm per frame.
    click_mode = "wall"        # "wall" / "start" / "goal"
    painting = None            # "add" / "erase" / None while dragging

    def reset_searches():
        nonlocal searches
        searches = make_searches(grid, start, goal)

    while True:
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
                elif event.key == pygame.K_n:
                    # Single-step (also pauses).
                    paused = True
                    for i, s in enumerate(searches):
                        if enabled[i] and not s.done:
                            s.step()
                elif event.key == pygame.K_r:
                    reset_searches()
                    paused = True
                elif event.key == pygame.K_c:
                    grid = empty_grid()
                    reset_searches()
                    paused = True
                elif event.key == pygame.K_m:
                    grid = random_maze(random.Random(), density=0.27)
                    clear_around(grid, start, 1)
                    clear_around(grid, goal, 1)
                    reset_searches()
                    paused = True
                elif event.key == pygame.K_s:
                    click_mode = "start"
                elif event.key == pygame.K_g:
                    click_mode = "goal"
                elif event.key == pygame.K_w:
                    click_mode = "wall"
                elif event.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                    idx = event.key - pygame.K_1
                    enabled[idx] = not enabled[idx]
                elif event.key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
                    speed = min(200, speed * 2)
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    speed = max(1, speed // 2)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                hit = mouse_to_cell(*event.pos)
                if hit:
                    _idx, cx, cy = hit
                    if click_mode == "start" and not grid[cy][cx]:
                        start = (cx, cy)
                        reset_searches()
                        paused = True
                        click_mode = "wall"
                    elif click_mode == "goal" and not grid[cy][cx]:
                        goal = (cx, cy)
                        reset_searches()
                        paused = True
                        click_mode = "wall"
                    else:
                        if event.button == 1:
                            painting = "add"
                        elif event.button == 3:
                            painting = "erase"
                        if painting and (cx, cy) != start and (cx, cy) != goal:
                            grid[cy][cx] = (painting == "add")
                            reset_searches()

            elif event.type == pygame.MOUSEBUTTONUP:
                painting = None

            elif event.type == pygame.MOUSEMOTION and painting:
                hit = mouse_to_cell(*event.pos)
                if hit:
                    _idx, cx, cy = hit
                    if (cx, cy) != start and (cx, cy) != goal:
                        new_val = (painting == "add")
                        if grid[cy][cx] != new_val:
                            grid[cy][cx] = new_val
                            reset_searches()

        # Step the algorithms.
        if not paused:
            for _ in range(speed):
                any_active = False
                for i, s in enumerate(searches):
                    if enabled[i] and not s.done:
                        s.step()
                        any_active = True
                if not any_active:
                    paused = True
                    break

        # Draw.
        screen.fill(BG)
        draw_header(screen, font, small_font, paused, speed, click_mode, enabled)
        for i, s in enumerate(searches):
            draw_panel(screen, font, small_font, i, s)

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
