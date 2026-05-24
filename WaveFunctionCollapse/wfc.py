"""
Wave Function Collapse — 2D tile-based generator.

Implementation notes:
  - Each cell holds a `wave`: the set of tile indices still possible there.
  - Tiles describe their 4 edge "sockets" (top, right, bottom, left). Two
    tiles can be adjacent if and only if the touching sockets are equal.
  - Each step:
      1. Pick the cell with minimum entropy (smallest wave > 1).
      2. Collapse it to one tile (weighted random).
      3. Propagate: any neighbor whose options no longer have a compatible
         partner gets its wave shrunk; if it shrank, propagate further.
  - Contradictions (empty wave) are recovered by full restart with a fresh
    random seed. For well-designed tilesets this is rare on small grids.

This file ships with a hand-built "circuit" tileset drawn procedurally so the
demo doesn't depend on external image assets.
"""

import math
import random
import sys
import time

import pygame

WIDTH, HEIGHT = 1280, 800
HUD_W = 280
GRID_W = WIDTH - HUD_W
GRID_H = HEIGHT
FPS = 60

# Colors.
BG = (16, 18, 28)
PANEL_BG = (22, 26, 40)
PANEL_BORDER = (60, 70, 95)
TEXT = (220, 226, 240)
TEXT_DIM = (140, 150, 175)
ACCENT = (245, 210, 80)
SLIDER_TRACK = (50, 58, 80)
SLIDER_FILL = (130, 200, 240)
SLIDER_KNOB = (220, 240, 255)

# Tile rendering colors (the "circuit" theme).
TILE_BG = (28, 38, 60)
WIRE = (130, 200, 240)
WIRE_GLOW = (200, 230, 255)
NODE = (245, 210, 80)
SUPERPOS = (52, 62, 88)        # Cell still in superposition.
LOWEST_ENTROPY = (245, 210, 80)
COLLAPSED_FLASH = (255, 255, 255)


# ---------------------------------------------------------------------------
# Tileset: hand-defined "circuit" pieces.
#
# Each base tile has:
#   sockets: (top, right, bottom, left) - opaque tags. Two tiles match if
#            socket[i] of one == socket[(i+2) % 4] of the neighbor.
#   draw:    (callable) renders the tile content into a CELL x CELL surface.
#
# We then auto-generate rotations: each base tile yields up to 4 rotated
# variants (deduplicated by socket+drawing).
# ---------------------------------------------------------------------------

# Socket vocabulary: edges are categorized as one of these symbols.
EMPTY = "empty"
WIRE_S = "wire"
PIPE = "pipe"


class TileSpec:
    """A base tile description plus its rotations."""

    def __init__(self, name, sockets, weight, draw_fn):
        self.name = name
        self.sockets = sockets
        self.weight = weight
        self.draw_fn = draw_fn  # called with (surface, cell_size, rot) where rot in 0..3


def draw_blank(surf, sz, rot):
    pygame.draw.rect(surf, TILE_BG, (0, 0, sz, sz))


def draw_wire_straight(surf, sz, rot):
    """Horizontal line of wire (rotated 90deg becomes vertical)."""
    pygame.draw.rect(surf, TILE_BG, (0, 0, sz, sz))
    cx = sz // 2
    if rot % 2 == 0:
        pygame.draw.rect(surf, WIRE, (0, cx - 2, sz, 4))
        pygame.draw.rect(surf, WIRE_GLOW, (0, cx - 1, sz, 2))
    else:
        pygame.draw.rect(surf, WIRE, (cx - 2, 0, 4, sz))
        pygame.draw.rect(surf, WIRE_GLOW, (cx - 1, 0, 2, sz))


def draw_wire_corner(surf, sz, rot):
    """Right-angle wire connecting two adjacent edges. rot 0 = top + right."""
    pygame.draw.rect(surf, TILE_BG, (0, 0, sz, sz))
    cx = sz // 2
    # Endpoints around the perimeter, ordered top, right, bottom, left.
    points = [(cx, 0), (sz, cx), (cx, sz), (0, cx)]
    a = points[rot]
    b = points[(rot + 1) % 4]
    # Draw an L-shape via the center.
    pygame.draw.line(surf, WIRE, a, (cx, cx), 4)
    pygame.draw.line(surf, WIRE, (cx, cx), b, 4)
    pygame.draw.line(surf, WIRE_GLOW, a, (cx, cx), 2)
    pygame.draw.line(surf, WIRE_GLOW, (cx, cx), b, 2)


def draw_wire_t(surf, sz, rot):
    """T-junction. rot 0 = wire on top + right + left (no bottom)."""
    pygame.draw.rect(surf, TILE_BG, (0, 0, sz, sz))
    cx = sz // 2
    points = [(cx, 0), (sz, cx), (cx, sz), (0, cx)]
    # Three arms, missing the side opposite to rot+2.
    arms = [(rot + 0) % 4, (rot + 1) % 4, (rot + 3) % 4]
    for k in arms:
        pygame.draw.line(surf, WIRE, points[k], (cx, cx), 4)
        pygame.draw.line(surf, WIRE_GLOW, points[k], (cx, cx), 2)
    pygame.draw.circle(surf, NODE, (cx, cx), 4)


def draw_wire_cross(surf, sz, rot):
    pygame.draw.rect(surf, TILE_BG, (0, 0, sz, sz))
    cx = sz // 2
    pygame.draw.rect(surf, WIRE, (0, cx - 2, sz, 4))
    pygame.draw.rect(surf, WIRE, (cx - 2, 0, 4, sz))
    pygame.draw.rect(surf, WIRE_GLOW, (0, cx - 1, sz, 2))
    pygame.draw.rect(surf, WIRE_GLOW, (cx - 1, 0, 2, sz))
    pygame.draw.circle(surf, NODE, (cx, cx), 5)


def draw_wire_dead_end(surf, sz, rot):
    """A wire that comes from one edge and stops at a node."""
    pygame.draw.rect(surf, TILE_BG, (0, 0, sz, sz))
    cx = sz // 2
    points = [(cx, 0), (sz, cx), (cx, sz), (0, cx)]
    a = points[rot]
    pygame.draw.line(surf, WIRE, a, (cx, cx), 4)
    pygame.draw.line(surf, WIRE_GLOW, a, (cx, cx), 2)
    pygame.draw.circle(surf, NODE, (cx, cx), 5)


def rotate_sockets(sockets, rot):
    """Rotate the (top, right, bottom, left) tuple by `rot` quarters CW."""
    return tuple(sockets[(i - rot) % 4] for i in range(4))


def build_tileset():
    """Return a flat list of (sockets, draw_fn, weight, rot) variants.

    Each base tile is expanded to its unique rotations; duplicates with the
    same sockets and same `(name, rot)` rendering are filtered by sockets only
    to avoid functionally-identical entries.
    """
    bases = [
        TileSpec("blank",   (EMPTY, EMPTY, EMPTY, EMPTY), 8.0,  draw_blank),
        TileSpec("straight", (WIRE_S, EMPTY, WIRE_S, EMPTY), 3.0, draw_wire_straight),
        TileSpec("corner",  (WIRE_S, WIRE_S, EMPTY, EMPTY), 3.0, draw_wire_corner),
        TileSpec("tee",     (WIRE_S, WIRE_S, EMPTY, WIRE_S), 1.5, draw_wire_t),
        TileSpec("cross",   (WIRE_S, WIRE_S, WIRE_S, WIRE_S), 0.6, draw_wire_cross),
        TileSpec("dead",    (WIRE_S, EMPTY, EMPTY, EMPTY), 0.7, draw_wire_dead_end),
    ]

    variants = []
    seen_sockets = set()
    for base in bases:
        for rot in range(4):
            rs = rotate_sockets(base.sockets, rot)
            key = (base.name, rs)
            # Symmetric tiles produce duplicate rotations; skip them.
            if (base.name, rs) in seen_sockets:
                continue
            seen_sockets.add((base.name, rs))
            variants.append({
                "name": f"{base.name}-{rot}",
                "sockets": rs,
                "weight": base.weight,
                "draw_fn": base.draw_fn,
                "rot": rot,
            })
    return variants


# ---------------------------------------------------------------------------
# Tile definitions
# ---------------------------------------------------------------------------
#
# Sockets are simple integer codes:
#   0 = empty edge (no wire crossing this side)
#   1 = wire endpoint at this side (a wire reaches the middle of this edge)
#
# Each tile is described by:
#   sockets:  (top, right, bottom, left)  with codes above
#   draw_fn:  function(surface, rect) drawing the tile graphics
#
# We programmatically rotate the base tiles so we don't need to define each
# rotation by hand.


def _draw_blank(surf, rect):
    pygame.draw.rect(surf, TILE_BG, rect)


def _draw_straight(surf, rect):
    """Horizontal wire."""
    pygame.draw.rect(surf, TILE_BG, rect)
    cx = rect.centerx
    cy = rect.centery
    pygame.draw.line(surf, WIRE, (rect.left, cy), (rect.right, cy), 4)


def _draw_corner(surf, rect):
    """Wire turning from right edge to bottom edge (an "L" rotated 0)."""
    pygame.draw.rect(surf, TILE_BG, rect)
    cx, cy = rect.centerx, rect.centery
    pygame.draw.line(surf, WIRE, (cx, cy), (rect.right, cy), 4)
    pygame.draw.line(surf, WIRE, (cx, cy), (cx, rect.bottom), 4)
    pygame.draw.circle(surf, WIRE, (cx, cy), 3)


def _draw_t(surf, rect):
    """T-junction: connects right, bottom, left (top is empty)."""
    pygame.draw.rect(surf, TILE_BG, rect)
    cx, cy = rect.centerx, rect.centery
    pygame.draw.line(surf, WIRE, (rect.left, cy), (rect.right, cy), 4)
    pygame.draw.line(surf, WIRE, (cx, cy), (cx, rect.bottom), 4)
    pygame.draw.circle(surf, NODE, (cx, cy), 4)


def _draw_cross(surf, rect):
    """All four sides connected."""
    pygame.draw.rect(surf, TILE_BG, rect)
    cx, cy = rect.centerx, rect.centery
    pygame.draw.line(surf, WIRE, (rect.left, cy), (rect.right, cy), 4)
    pygame.draw.line(surf, WIRE, (cx, rect.top), (cx, rect.bottom), 4)
    pygame.draw.circle(surf, NODE, (cx, cy), 4)


def _draw_endpoint(surf, rect):
    """Dead-end: wire goes from right edge to a node in the center."""
    pygame.draw.rect(surf, TILE_BG, rect)
    cx, cy = rect.centerx, rect.centery
    pygame.draw.line(surf, WIRE, (cx, cy), (rect.right, cy), 4)
    pygame.draw.circle(surf, NODE, (cx, cy), 5)


def rotate_sockets(sockets, k):
    """Rotate (top, right, bottom, left) clockwise by k * 90 degrees."""
    s = list(sockets)
    for _ in range(k % 4):
        # CW rotation: top <- left, right <- top, bottom <- right, left <- bottom.
        s = [s[3], s[0], s[1], s[2]]
    return tuple(s)


def make_rotated_draw(base_draw, k):
    """Wrap a draw function so it renders rotated by k * 90 degrees CW."""
    def draw(surf, rect):
        # Render the tile to a small temp surface, rotate, then blit.
        size = rect.width
        tmp = pygame.Surface((size, size))
        base_draw(tmp, pygame.Rect(0, 0, size, size))
        if k:
            tmp = pygame.transform.rotate(tmp, -90 * k)  # negative = CW
        surf.blit(tmp, rect.topleft)
    return draw


def build_tileset():
    """Build the full rotated tileset.

    Returns (tiles, weights) where:
      tiles[i]   = (sockets, draw_fn)
      weights[i] = relative frequency for weighted random collapse.
    """
    base = [
        # (sockets,        draw_fn,       weight, n_rotations)
        ((0, 0, 0, 0),     _draw_blank,    8.0,   1),
        ((0, 1, 0, 1),     _draw_straight, 3.0,   2),
        ((0, 1, 1, 0),     _draw_corner,   2.5,   4),
        ((0, 1, 1, 1),     _draw_t,        1.5,   4),
        ((1, 1, 1, 1),     _draw_cross,    0.4,   1),
        ((0, 1, 0, 0),     _draw_endpoint, 0.6,   4),
    ]
    tiles = []
    weights = []
    for sockets, draw_fn, w, nrot in base:
        seen_sockets = set()
        for k in range(nrot):
            sk = rotate_sockets(sockets, k)
            if sk in seen_sockets:
                continue  # Skip duplicates (e.g. straight pipe has 2 unique rots).
            seen_sockets.add(sk)
            tiles.append((sk, make_rotated_draw(draw_fn, k)))
            weights.append(w)
    return tiles, weights


def compatibility_table(tiles):
    """Precompute, for each (tile, direction), the set of compatible neighbors.

    Directions: 0=top, 1=right, 2=bottom, 3=left.
    Compatibility: tile A's edge in direction D == tile B's edge in opposite D.
    """
    n = len(tiles)
    opposite = {0: 2, 1: 3, 2: 0, 3: 1}
    table = [[set() for _ in range(4)] for _ in range(n)]
    for i, (sa, _) in enumerate(tiles):
        for j, (sb, _) in enumerate(tiles):
            for d in range(4):
                if sa[d] == sb[opposite[d]]:
                    table[i][d].add(j)
    return table


# ---------------------------------------------------------------------------
# WFC core algorithm
# ---------------------------------------------------------------------------
#
# Direction vectors must match socket order (top, right, bottom, left).
DIRS = [(0, -1), (1, 0), (0, 1), (-1, 0)]


class WFC:
    """Stateful WFC solver. Call step() to advance one collapse + propagation."""

    def __init__(self, cols, rows, tiles, weights, table, seed=None):
        self.cols = cols
        self.rows = rows
        self.tiles = tiles
        self.weights = weights
        self.table = table
        self.n_tiles = len(tiles)
        self.rng = random.Random(seed)
        self.collapsed_count = 0
        self.contradictions = 0
        self.last_collapse_pos = None
        self.done = False
        self.failed = False
        self._init_wave()

    def _init_wave(self):
        # `wave[y][x]` is a set of tile indices still possible for that cell.
        full = set(range(self.n_tiles))
        self.wave = [[set(full) for _ in range(self.cols)] for _ in range(self.rows)]
        self.collapsed_count = 0
        self.last_collapse_pos = None
        self.done = False
        self.failed = False
        # Cells that changed since the last time the renderer pulled dirties.
        # Renderer reads + clears this set so we only repaint what's needed.
        self.dirty_cells = {(x, y) for y in range(self.rows) for x in range(self.cols)}
        # Set of (x, y) positions still in superposition; speeds up min-entropy
        # search from O(rows*cols) to O(remaining).
        self.uncollapsed = {(x, y) for y in range(self.rows) for x in range(self.cols)}

    # ---- Entropy ---------------------------------------------------------

    def _entropy(self, x, y):
        w = self.wave[y][x]
        n = len(w)
        if n <= 1:
            return math.inf
        # Shannon entropy weighted by tile weights, with a small random tie-break.
        total = 0.0
        for t in w:
            total += self.weights[t]
        h = 0.0
        inv = 1.0 / total
        for t in w:
            p = self.weights[t] * inv
            h -= p * math.log(p)
        # Tiny noise to break ties so the algorithm doesn't always pick same cell.
        return h + self.rng.random() * 1e-3

    def _find_min_entropy(self):
        best = None
        best_h = math.inf
        for pos in self.uncollapsed:
            x, y = pos
            h = self._entropy(x, y)
            if h < best_h:
                best_h = h
                best = pos
        return best

    # ---- Collapse + propagation -----------------------------------------

    def _collapse(self, x, y):
        options = list(self.wave[y][x])
        if not options:
            return False
        weights = [self.weights[t] for t in options]
        chosen = self.rng.choices(options, weights=weights, k=1)[0]
        self.wave[y][x] = {chosen}
        self.last_collapse_pos = (x, y)
        self.collapsed_count += 1
        self.uncollapsed.discard((x, y))
        self.dirty_cells.add((x, y))
        return True

    def _propagate(self, start):
        """Iteratively shrink neighbor waves to maintain consistency.

        For each updated cell, look at every neighbor: a neighbor's tile T is
        still valid only if at least one tile in this cell allows T on the
        shared edge (i.e. T is in the union of compatibility[allowed][dir]).

        We use a directed-arc queue: an entry ``(x, y, d)`` means "the wave at
        (x, y) just changed, so the neighbor in direction ``d`` may need to
        shrink". This is essentially AC-3 over the tile-adjacency constraint.
        """
        # Seed the queue with all four arcs out of the start cell.
        sx, sy = start
        queue = [(sx, sy, d) for d in range(4)]
        in_queue = {(sx, sy, d) for d in range(4)}
        table = self.table
        while queue:
            x, y, d = queue.pop()
            in_queue.discard((x, y, d))
            current = self.wave[y][x]
            if not current:
                self.failed = True
                return False
            dx, dy = DIRS[d]
            nx, ny = x + dx, y + dy
            if not (0 <= nx < self.cols and 0 <= ny < self.rows):
                continue
            neighbor = self.wave[ny][nx]
            # Allowed tiles in neighbor = union of compatibility[t][d] for t in current.
            allowed = set()
            for t in current:
                allowed |= table[t][d]
            new_neighbor = neighbor & allowed
            if new_neighbor == neighbor:
                continue
            if not new_neighbor:
                self.failed = True
                return False
            self.wave[ny][nx] = new_neighbor
            self.dirty_cells.add((nx, ny))
            if len(new_neighbor) == 1:
                self.uncollapsed.discard((nx, ny))
            # The neighbor changed; enqueue all four of its outgoing arcs
            # (including the one back to (x,y), which is harmless and helps
            # in rare asymmetric tilesets).
            for d2 in range(4):
                key = (nx, ny, d2)
                if key not in in_queue:
                    in_queue.add(key)
                    queue.append(key)
        return True

    # ---- Public step ----------------------------------------------------

    def step(self):
        """Advance one observe + propagate iteration.

        Returns True if work was done; False once the wave is fully collapsed
        or has hit a contradiction (in which case `failed` is True).
        """
        if self.done or self.failed:
            return False
        cell = self._find_min_entropy()
        if cell is None:
            self.done = True
            return False
        x, y = cell
        if not self._collapse(x, y):
            self.failed = True
            return False
        if not self._propagate((x, y)):
            self.contradictions += 1
            return False
        return True

    def restart(self, new_seed=None):
        if new_seed is not None:
            self.rng = random.Random(new_seed)
        self._init_wave()


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_wave(surface, wfc, cell_size, origin):
    """Legacy reference renderer (unused in main loop).

    Kept for documentation/testing. The main loop uses ``TileRenderer`` which
    pre-renders tiles and only repaints dirty cells.
    """
    ox, oy = origin
    for y in range(wfc.rows):
        for x in range(wfc.cols):
            options = wfc.wave[y][x]
            rect = pygame.Rect(ox + x * cell_size, oy + y * cell_size, cell_size, cell_size)
            if len(options) == 1:
                tile_idx = next(iter(options))
                _sockets, draw_fn = wfc.tiles[tile_idx]
                draw_fn(surface, rect)
            else:
                n = len(options)
                t = 1.0 - n / wfc.n_tiles
                shade = int(34 + 60 * t)
                pygame.draw.rect(surface, (shade, shade + 4, shade + 12), rect)
                if cell_size >= 26 and n > 0:
                    color = (90, 110, 150)
                    pygame.draw.rect(surface, color, rect.inflate(-cell_size + 4, -cell_size + 4))


class TileRenderer:
    """Pre-renders tile and superposition surfaces for one cell size.

    Holds:
      - `tile_surfs[i]`: rendered surface for collapsed tile i.
      - `super_surfs[bucket]`: surfaces for "still in superposition" cells,
        indexed by a small number of brightness buckets (more options ->
        darker). Bucketing avoids creating one surface per option-count.
      - `grid_surface`: persistent surface containing the entire grid; the
        renderer only repaints the cells listed in `wfc.dirty_cells`.
    """

    NUM_SHADE_BUCKETS = 6

    def __init__(self, tiles, cell_size, cols, rows, n_tiles):
        self.cell_size = cell_size
        self.cols = cols
        self.rows = rows
        self.n_tiles = n_tiles
        self._build_tile_cache(tiles)
        self._build_shade_cache()
        # The full grid surface; rendered cells are blitted in here, then the
        # whole thing is blitted to the screen each frame (a single fast op).
        self.grid_surface = pygame.Surface((cell_size * cols, cell_size * rows))
        self.grid_surface.fill(PANEL_BG)
        # First frame should paint everything.
        self.first_frame = True

    def _build_tile_cache(self, tiles):
        self.tile_surfs = []
        for sockets, draw_fn in tiles:
            surf = pygame.Surface((self.cell_size, self.cell_size))
            draw_fn(surf, pygame.Rect(0, 0, self.cell_size, self.cell_size))
            self.tile_surfs.append(surf.convert())

    def _build_shade_cache(self):
        # Bucket index 0 = max options (darkest), NUM_SHADE_BUCKETS-1 = fewest.
        self.super_surfs = []
        for b in range(self.NUM_SHADE_BUCKETS):
            t = b / max(1, self.NUM_SHADE_BUCKETS - 1)
            shade = int(34 + 60 * t)
            color = (shade, shade + 4, shade + 12)
            surf = pygame.Surface((self.cell_size, self.cell_size))
            surf.fill(color)
            if self.cell_size >= 26:
                inner = pygame.Rect(2, 2, self.cell_size - 4, self.cell_size - 4)
                pygame.draw.rect(surf, (90, 110, 150), inner)
            self.super_surfs.append(surf.convert())

    def _shade_bucket(self, n_options):
        # n_options in [2 .. n_tiles]; map fewer options -> higher bucket.
        t = 1.0 - (n_options - 1) / max(1, self.n_tiles - 1)
        b = int(t * (self.NUM_SHADE_BUCKETS - 1) + 0.5)
        return max(0, min(self.NUM_SHADE_BUCKETS - 1, b))

    def render(self, wfc):
        """Repaint only dirty cells; the caller blits `grid_surface` to screen."""
        cs = self.cell_size
        gs = self.grid_surface
        if self.first_frame:
            # Initial: paint everything as the maximum-options bucket.
            full_surf = self.super_surfs[0]
            for y in range(wfc.rows):
                for x in range(wfc.cols):
                    gs.blit(full_surf, (x * cs, y * cs))
            self.first_frame = False
            wfc.dirty_cells.clear()
            return

        if not wfc.dirty_cells:
            return
        for (x, y) in wfc.dirty_cells:
            options = wfc.wave[y][x]
            n = len(options)
            if n == 1:
                tile_idx = next(iter(options))
                gs.blit(self.tile_surfs[tile_idx], (x * cs, y * cs))
            elif n == 0:
                # Contradiction; show as bright red to make it visible.
                pygame.draw.rect(gs, (180, 40, 50),
                                 (x * cs, y * cs, cs, cs))
            else:
                gs.blit(self.super_surfs[self._shade_bucket(n)], (x * cs, y * cs))
        wfc.dirty_cells.clear()


def render_overlays(surface, wfc, cell_size, origin):
    ox, oy = origin
    if wfc.last_collapse_pos:
        x, y = wfc.last_collapse_pos
        rect = pygame.Rect(ox + x * cell_size, oy + y * cell_size, cell_size, cell_size)
        pygame.draw.rect(surface, ACCENT, rect, width=2)


# ---------------------------------------------------------------------------
# Sliders
# ---------------------------------------------------------------------------


class Slider:
    HEIGHT = 14
    LABEL_GAP = 18

    def __init__(self, x, y, w, label, lo, hi, step, getter, setter):
        self.label = label
        self.lo = lo
        self.hi = hi
        self.step = step
        self.getter = getter
        self.setter = setter
        self.rect = pygame.Rect(x, y + Slider.LABEL_GAP, w, Slider.HEIGHT)

    def value_to_x(self, v):
        t = (v - self.lo) / (self.hi - self.lo)
        return int(self.rect.x + t * self.rect.w)

    def x_to_value(self, x):
        t = max(0.0, min(1.0, (x - self.rect.x) / self.rect.w))
        v = self.lo + t * (self.hi - self.lo)
        return round(v / self.step) * self.step

    def draw(self, surface, font):
        v = self.getter()
        label = f"{self.label}: {int(v) if v == int(v) else round(v, 1)}"
        surface.blit(font.render(label, True, TEXT), (self.rect.x, self.rect.y - Slider.LABEL_GAP + 1))
        pygame.draw.rect(surface, SLIDER_TRACK, self.rect, border_radius=4)
        knob_x = self.value_to_x(v)
        fill = pygame.Rect(self.rect.x, self.rect.y, knob_x - self.rect.x, self.rect.h)
        pygame.draw.rect(surface, SLIDER_FILL, fill, border_radius=4)
        pygame.draw.circle(surface, SLIDER_KNOB, (knob_x, self.rect.centery), 7)

    def hit(self, mx, my):
        return self.rect.x - 4 <= mx <= self.rect.right + 4 and self.rect.y - 6 <= my <= self.rect.bottom + 6

    def update_from_mouse(self, mx):
        v = self.x_to_value(mx)
        if self.step >= 1:
            v = int(v)
        self.setter(v)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main():
    pygame.init()
    pygame.display.set_caption("Wave Function Collapse")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Menlo,Consolas,monospace", 13)
    small = pygame.font.SysFont("Menlo,Consolas,monospace", 11)
    title_font = pygame.font.SysFont("Menlo,Consolas,monospace", 16, bold=True)

    tiles, weights = build_tileset()
    table = compatibility_table(tiles)

    # Tunables (closure-captured by the sliders).
    state = {
        "cols": 28,
        "rows": 18,
        "speed": 30,        # Steps per frame.
        "auto_restart": 1,  # Whether to restart on contradiction.
        "auto_regen": 1,    # Whether to start a new run automatically when done.
    }

    def make_wfc():
        return WFC(state["cols"], state["rows"], tiles, weights, table,
                   seed=random.randint(0, 999_999))

    wfc = make_wfc()

    # Compute cell size to fit the grid into the left panel.
    def cell_metrics():
        cs = min((GRID_W - 32) // wfc.cols, (GRID_H - 32) // wfc.rows)
        cs = max(8, cs)
        gw = cs * wfc.cols
        gh = cs * wfc.rows
        ox = (GRID_W - gw) // 2
        oy = (GRID_H - gh) // 2
        return cs, (ox, oy)

    # Sliders.
    pad = 16
    sx = GRID_W + pad
    sw = HUD_W - 2 * pad
    sliders = [
        Slider(sx, 90,  sw, "Cols",  6, 60, 1,
               lambda: state["cols"], lambda v: state.update(cols=int(v))),
        Slider(sx, 132, sw, "Rows",  6, 50, 1,
               lambda: state["rows"], lambda v: state.update(rows=int(v))),
        Slider(sx, 174, sw, "Speed (steps/frame)", 1, 500, 1,
               lambda: state["speed"], lambda v: state.update(speed=int(v))),
        Slider(sx, 216, sw, "Auto-restart on fail", 0, 1, 1,
               lambda: state["auto_restart"], lambda v: state.update(auto_restart=int(v))),
        Slider(sx, 258, sw, "Auto-regenerate", 0, 1, 1,
               lambda: state["auto_regen"], lambda v: state.update(auto_regen=int(v))),
    ]

    paused = False
    active_slider = None
    last_grid_dims = (state["cols"], state["rows"])
    done_pause_frames = 0  # Counts frames that wfc has been in `done` state.

    cs, origin = None, None
    tile_renderer = None

    def rebuild_renderer():
        nonlocal cs, origin, tile_renderer
        cs2 = min((GRID_W - 32) // wfc.cols, (GRID_H - 32) // wfc.rows)
        cs2 = max(8, cs2)
        gw = cs2 * wfc.cols
        gh = cs2 * wfc.rows
        ox = (GRID_W - gw) // 2
        oy = (GRID_H - gh) // 2
        cs = cs2
        origin = (ox, oy)
        tile_renderer = TileRenderer(tiles, cs, wfc.cols, wfc.rows, wfc.n_tiles)

    rebuild_renderer()

    while True:
        dt_ms = clock.tick(FPS)

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
                elif event.key == pygame.K_r:
                    wfc = make_wfc()
                    rebuild_renderer()
                elif event.key == pygame.K_n:
                    # Single step.
                    paused = True
                    if not wfc.done and not wfc.failed:
                        wfc.step()
                elif event.key == pygame.K_RETURN:
                    # Solve to completion immediately (with restarts).
                    paused = True
                    t0 = time.perf_counter()
                    while not wfc.done and time.perf_counter() - t0 < 2.0:
                        if not wfc.step():
                            if wfc.failed and state["auto_restart"]:
                                wfc.restart(random.randint(0, 999_999))
                            else:
                                break
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = event.pos
                if mx > GRID_W:
                    for s in sliders:
                        if s.hit(mx, my):
                            active_slider = s
                            s.update_from_mouse(mx)
                            break
            elif event.type == pygame.MOUSEBUTTONUP:
                active_slider = None
            elif event.type == pygame.MOUSEMOTION:
                if active_slider:
                    active_slider.update_from_mouse(event.pos[0])

        # If grid dims changed via sliders, rebuild WFC.
        if (state["cols"], state["rows"]) != last_grid_dims:
            last_grid_dims = (state["cols"], state["rows"])
            wfc = make_wfc()
            rebuild_renderer()

        # Step the WFC.
        if not paused:
            for _ in range(state["speed"]):
                if wfc.done:
                    break
                if wfc.failed:
                    if state["auto_restart"]:
                        wfc.restart(random.randint(0, 999_999))
                    else:
                        break
                else:
                    wfc.step()

        # Auto-regenerate after the wave is done so the demo keeps producing
        # new patterns. We let it sit for a moment so you can actually see
        # the finished result.
        if wfc.done:
            done_pause_frames += 1
            if state["auto_regen"] and done_pause_frames > 60:  # ~1s @60fps
                wfc = make_wfc()
                rebuild_renderer()
                done_pause_frames = 0
        else:
            done_pause_frames = 0

        # Render.
        screen.fill(BG)
        # Grid background border.
        gw = cs * wfc.cols
        gh = cs * wfc.rows
        ox, oy = origin
        pygame.draw.rect(screen, PANEL_BG, (ox - 2, oy - 2, gw + 4, gh + 4))

        # Repaint dirty cells into the persistent grid surface, then blit it.
        tile_renderer.render(wfc)
        screen.blit(tile_renderer.grid_surface, origin)
        render_overlays(screen, wfc, cs, origin)

        # Right side panel.
        pygame.draw.rect(screen, PANEL_BG, (GRID_W, 0, HUD_W, HEIGHT))
        pygame.draw.line(screen, PANEL_BORDER, (GRID_W, 0), (GRID_W, HEIGHT), 1)
        screen.blit(title_font.render("Wave Function Collapse", True, ACCENT), (GRID_W + 16, 14))

        progress = wfc.collapsed_count
        total = wfc.cols * wfc.rows
        pct = 100 * progress / max(1, total)
        status = "DONE" if wfc.done else ("FAILED" if wfc.failed else ("PAUSED" if paused else "RUNNING"))
        screen.blit(small.render(f"FPS: {clock.get_fps():.0f}   {status}",
                                  True, TEXT_DIM), (GRID_W + 16, 38))
        screen.blit(small.render(f"Tiles: {wfc.n_tiles}   "
                                  f"Collapsed: {progress}/{total} ({pct:.0f}%)",
                                  True, TEXT_DIM), (GRID_W + 16, 54))
        screen.blit(small.render(f"Restarts after fail: {wfc.contradictions}",
                                  True, TEXT_DIM), (GRID_W + 16, 70))

        for s in sliders:
            s.draw(screen, font)

        # Help text at the bottom.
        help_lines = [
            "Space: pause/run",
            "N: single step",
            "Enter: solve to end",
            "R: new seed (restart)",
            "Esc: quit",
            "",
            "Yellow box: cell just",
            "collapsed this step.",
            "Darker cells = fewer",
            "options remaining.",
            "Auto-regen restarts",
            "1s after each finish.",
        ]
        y = HEIGHT - 16 * len(help_lines) - 10
        for line in help_lines:
            screen.blit(small.render(line, True, TEXT_DIM), (GRID_W + 16, y))
            y += 16

        pygame.display.flip()


if __name__ == "__main__":
    main()
