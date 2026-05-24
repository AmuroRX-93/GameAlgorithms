"""
2D Visibility / Lighting Simulation
Inspired by Red Blob Games' visibility algorithm.

Algorithm (O(N log N)):
  1. Collect all wall segment endpoints.
  2. For each endpoint, cast 3 rays from the light source: one directly at the
     endpoint, and two slightly offset (+/- epsilon) to handle corners.
  3. For each ray, find the closest wall intersection. That hit point is a
     vertex of the visibility polygon.
  4. Sort the hit points by angle around the source and draw the polygon.

Controls:
  - Move the mouse to move the light source.
  - Left click to toggle "AI grenade view" mode (shows danger zones).
  - Press R to randomize the map.
  - Press SPACE to add/remove a second light (compare two viewpoints).
  - Press ESC or close the window to quit.
"""

import math
import random
import sys

import numpy as np
import pygame

WIDTH, HEIGHT = 1024, 720
FPS = 60

# Lighting renders at half resolution (large speedup, imperceptible blur).
LIGHT_W, LIGHT_H = WIDTH // 2, HEIGHT // 2

# Pre-built coordinate grids for the per-pixel falloff renderer (low-res).
_PX_X = np.arange(LIGHT_W, dtype=np.float32)[:, None] * 2.0
_PX_Y = np.arange(LIGHT_H, dtype=np.float32)[None, :] * 2.0

# Color palette tuned to look like the screenshot (deep blue room, warm light).
COLOR_BG = (10, 14, 28)
COLOR_WALL = (24, 32, 60)
COLOR_WALL_EDGE = (60, 80, 130)
COLOR_FLOOR_DARK = (18, 26, 50)
COLOR_LIGHT = (255, 220, 150)
COLOR_LIGHT_CORE = (255, 245, 210)
COLOR_DANGER = (210, 110, 60)
COLOR_SAFE = (40, 60, 110)
COLOR_TEXT = (220, 220, 230)

EPSILON = 1e-4
ANGLE_NUDGE = 1e-5


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def segments_from_rect(x, y, w, h):
    """Return the four edges of an axis-aligned rectangle as line segments."""
    p1 = (x, y)
    p2 = (x + w, y)
    p3 = (x + w, y + h)
    p4 = (x, y + h)
    return [(p1, p2), (p2, p3), (p3, p4), (p4, p1)]


def ray_segment_intersection(rx, ry, rdx, rdy, ax, ay, bx, by):
    """Intersect ray (origin r, direction d) with segment AB.

    Returns (t, point) where t is the distance along the ray, or None if there
    is no forward intersection. Uses the standard parametric form solved with
    Cramer's rule.
    """
    sdx = bx - ax
    sdy = by - ay

    denom = rdx * sdy - rdy * sdx
    if abs(denom) < EPSILON:
        return None  # Parallel.

    # T1 = distance along ray, T2 = position along segment in [0, 1].
    t1 = ((ax - rx) * sdy - (ay - ry) * sdx) / denom
    t2 = ((ax - rx) * rdy - (ay - ry) * rdx) / denom

    if t1 < 0 or t2 < 0 or t2 > 1:
        return None
    return t1, (rx + rdx * t1, ry + rdy * t1)


def point_in_obstacle(point, obstacles, margin=0):
    """Return True if `point` is inside any obstacle rectangle (with margin)."""
    px, py = point
    for x, y, w, h in obstacles:
        if x - margin <= px <= x + w + margin and y - margin <= py <= y + h + margin:
            return True
    return False


def clamp_point_outside_obstacles(point, obstacles, margin=2):
    """If `point` is inside an obstacle, push it to the nearest outside edge.

    Walls are axis-aligned rectangles, so the closest exit is whichever of the
    four edges has the smallest perpendicular distance from the point.
    """
    px, py = point
    for x, y, w, h in obstacles:
        left, right = x - margin, x + w + margin
        top, bottom = y - margin, y + h + margin
        if left <= px <= right and top <= py <= bottom:
            d_left = px - left
            d_right = right - px
            d_top = py - top
            d_bottom = bottom - py
            d_min = min(d_left, d_right, d_top, d_bottom)
            if d_min == d_left:
                px = left
            elif d_min == d_right:
                px = right
            elif d_min == d_top:
                py = top
            else:
                py = bottom
    return (px, py)


def prepare_segments(segments):
    """Pre-pack segments into numpy arrays for vectorized intersection.

    Returns (ax, ay, sdx, sdy) where each is shape (N,) float32: segment
    start xy and segment direction xy = end - start.
    """
    if not segments:
        empty = np.zeros(0, dtype=np.float32)
        return empty, empty, empty, empty
    arr = np.asarray(segments, dtype=np.float32)  # (N, 2, 2)
    ax = arr[:, 0, 0]
    ay = arr[:, 0, 1]
    sdx = arr[:, 1, 0] - ax
    sdy = arr[:, 1, 1] - ay
    return ax, ay, sdx, sdy


def compute_visibility_polygon(source, segments_packed, max_dist):
    """Vectorized visibility polygon.

    `segments_packed` is the tuple returned by prepare_segments(). For each
    candidate angle we cast a ray and find the closest segment intersection
    using numpy broadcasting (one vector op for all N segments at once).
    """
    ax, ay, sdx, sdy = segments_packed
    sx, sy = float(source[0]), float(source[1])
    n_seg = ax.shape[0]
    if n_seg == 0:
        return []

    # Build candidate angles: every segment endpoint, +/- nudge.
    end_x = ax + sdx
    end_y = ay + sdy
    px = np.concatenate([ax, end_x])
    py = np.concatenate([ay, end_y])
    base_angles = np.arctan2(py - sy, px - sx)
    angles = np.concatenate([
        base_angles - ANGLE_NUDGE,
        base_angles,
        base_angles + ANGLE_NUDGE,
    ])  # shape (M,) where M = 6 * n_seg

    rdx = np.cos(angles).astype(np.float32)
    rdy = np.sin(angles).astype(np.float32)

    # Solve ray-segment intersection for every (angle, segment) pair.
    # Shapes: rays (M, 1), segments (1, N) -> result (M, N).
    rdx_ = rdx[:, None]
    rdy_ = rdy[:, None]
    ax_ = ax[None, :]
    ay_ = ay[None, :]
    sdx_ = sdx[None, :]
    sdy_ = sdy[None, :]

    denom = rdx_ * sdy_ - rdy_ * sdx_
    safe = np.abs(denom) > EPSILON
    denom_safe = np.where(safe, denom, 1.0)

    # T1 = ray param; T2 = segment param.
    t1 = ((ax_ - sx) * sdy_ - (ay_ - sy) * sdx_) / denom_safe
    t2 = ((ax_ - sx) * rdy_ - (ay_ - sy) * rdx_) / denom_safe

    valid = safe & (t1 >= 0) & (t2 >= 0) & (t2 <= 1)
    t1_masked = np.where(valid, t1, np.inf)
    t_min = t1_masked.min(axis=1)  # (M,)

    # Cap to max_dist so rays that miss everything still terminate cleanly.
    t_min = np.minimum(t_min, max_dist)
    hit_x = sx + rdx * t_min
    hit_y = sy + rdy * t_min

    # Sort by angle and emit polygon.
    order = np.argsort(angles)
    return list(zip(hit_x[order].tolist(), hit_y[order].tolist()))


# ---------------------------------------------------------------------------
# Map generation
# ---------------------------------------------------------------------------


def build_map(seed=None):
    """Generate a fully random barrier layout.

    The map is composed of three kinds of obstacles:
      - thin random walls (horizontal or vertical segments of varied length)
      - solid pillar/box blocks
      - a screen border so rays always terminate

    Placement uses rejection sampling against an inflated bounding box so
    nothing overlaps and there's always navigable space between barriers.
    `obstacles` is the render list, `segments` is fed to the raycaster.
    """
    rng = random.Random(seed)

    border = [
        (0, 0, WIDTH, 8),
        (0, HEIGHT - 8, WIDTH, 8),
        (0, 0, 8, HEIGHT),
        (WIDTH - 8, 0, 8, HEIGHT),
    ]

    obstacles = []
    placed_rects = []  # Inflated rects used for collision checks.

    inset = 60          # Keep a free margin around the screen edges.
    spacing = 24        # Min gap between barriers (inflation per side).
    wall_thickness = 14
    max_attempts_per_item = 25

    def try_place(x, y, w, h):
        rect = pygame.Rect(x - spacing, y - spacing, w + 2 * spacing, h + 2 * spacing)
        if rect.left < 0 or rect.top < 0 or rect.right > WIDTH or rect.bottom > HEIGHT:
            return False
        for other in placed_rects:
            if rect.colliderect(other):
                return False
        placed_rects.append(rect)
        obstacles.append((x, y, w, h))
        return True

    # 1. Long random walls (the dominant features that create rooms/corridors).
    wall_count = rng.randint(10, 14)
    for _ in range(wall_count):
        for _try in range(max_attempts_per_item):
            horizontal = rng.random() < 0.5
            if horizontal:
                w = rng.randint(120, 320)
                h = wall_thickness
            else:
                w = wall_thickness
                h = rng.randint(120, 320)
            x = rng.randint(inset, WIDTH - inset - w)
            y = rng.randint(inset, HEIGHT - inset - h)
            if try_place(x, y, w, h):
                break

    # 2. Smaller pillar/box obstacles to break up open space.
    box_count = rng.randint(6, 10)
    for _ in range(box_count):
        for _try in range(max_attempts_per_item):
            size = rng.randint(28, 70)
            w = size
            h = size if rng.random() < 0.6 else rng.randint(28, 70)
            x = rng.randint(inset, WIDTH - inset - w)
            y = rng.randint(inset, HEIGHT - inset - h)
            if try_place(x, y, w, h):
                break

    segments = []
    for rect in border + obstacles:
        segments.extend(segments_from_rect(*rect))

    return segments, obstacles, border


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def draw_radial_light(surface, source, polygon, radius, color, intensity=1.0, halflife_ratio=0.35):
    """Render a single radial light with per-pixel inverse-square falloff.

    Computes the brightness field at half resolution (LIGHT_W x LIGHT_H) for
    speed; `apply_lighting` upscales to full resolution before modulation.
    Returns a (LIGHT_W, LIGHT_H) float32 array in [0, 1].
    """
    if radius <= 0:
        return np.zeros((LIGHT_W, LIGHT_H), dtype=np.float32)

    sx, sy = source
    halflife = max(1.0, radius * halflife_ratio)

    dx = _PX_X - sx
    dy = _PX_Y - sy
    dist2 = dx * dx + dy * dy

    base = 1.0 / (1.0 + dist2 / (halflife * halflife))
    window = np.clip(1.0 - dist2 / (radius * radius), 0.0, 1.0) ** 2
    brightness = base * window * intensity

    # Polygon visibility mask, also at half resolution.
    mask_surf = pygame.Surface((LIGHT_W, LIGHT_H))
    mask_surf.fill((0, 0, 0))
    if len(polygon) >= 3:
        scaled_poly = [(p[0] * 0.5, p[1] * 0.5) for p in polygon]
        pygame.draw.polygon(mask_surf, (255, 255, 255), scaled_poly)
    poly_mask = pygame.surfarray.pixels_red(mask_surf).astype(np.float32) / 255.0
    brightness = brightness * poly_mask

    return np.clip(brightness, 0.0, 1.0).astype(np.float32)


def apply_lighting(surface, lights, ambient=0.08, tint=(255, 220, 150)):
    """Combine multiple light contributions and modulate the scene in place.

    Implementation: build two low-res surfaces — a `mult` surface holding the
    brightness as gray (used with BLEND_MULT to darken unlit pixels) and an
    `add` surface holding the colored bloom (used with BLEND_ADD). Both are
    blitted once at full resolution; pygame's C blits are far faster than
    numpy float math on uint8 arrays.
    """
    if not lights:
        dark = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        dark.fill((0, 0, 0, int(255 * (1 - ambient))))
        surface.blit(dark, (0, 0))
        return

    total = np.full((LIGHT_W, LIGHT_H), ambient, dtype=np.float32)
    color_r = np.zeros((LIGHT_W, LIGHT_H), dtype=np.float32)
    color_g = np.zeros((LIGHT_W, LIGHT_H), dtype=np.float32)
    color_b = np.zeros((LIGHT_W, LIGHT_H), dtype=np.float32)

    for brightness, light_tint in lights:
        np.maximum(total, brightness + ambient, out=total)
        color_r += brightness * light_tint[0]
        color_g += brightness * light_tint[1]
        color_b += brightness * light_tint[2]

    np.clip(total, 0.0, 1.0, out=total)

    # Multiplicative darkness layer: gray = brightness * 255.
    mult_low = pygame.Surface((LIGHT_W, LIGHT_H))
    rgb = pygame.surfarray.pixels3d(mult_low)
    gray = (total * 255.0).astype(np.uint8)
    rgb[..., 0] = gray
    rgb[..., 1] = gray
    rgb[..., 2] = gray
    del rgb

    # Additive colored bloom layer (warm halo near bright pixels).
    add_low = pygame.Surface((LIGHT_W, LIGHT_H))
    rgb = pygame.surfarray.pixels3d(add_low)
    rgb[..., 0] = np.clip(color_r * 0.45, 0, 255).astype(np.uint8)
    rgb[..., 1] = np.clip(color_g * 0.45, 0, 255).astype(np.uint8)
    rgb[..., 2] = np.clip(color_b * 0.45, 0, 255).astype(np.uint8)
    del rgb

    # Upscale to full resolution and apply via pygame's native blend modes.
    mult_full = pygame.transform.scale(mult_low, (WIDTH, HEIGHT))
    add_full = pygame.transform.scale(add_low, (WIDTH, HEIGHT))
    surface.blit(mult_full, (0, 0), special_flags=pygame.BLEND_RGB_MULT)
    surface.blit(add_full, (0, 0), special_flags=pygame.BLEND_RGB_ADD)


def draw_danger_with_falloff(surface, source, polygon, max_range, danger_color, safe_color):
    """AI grenade view with per-pixel distance falloff.

    Visible pixels closer to the AI are red-hot (high hit chance); pixels near
    the throw range fade out. Outside the visibility polygon (or beyond
    `max_range`) is treated as safe cover.
    """
    cover = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    cover.fill((*safe_color, 110))
    surface.blit(cover, (0, 0))

    if len(polygon) < 3 or max_range <= 0:
        return

    sx, sy = source
    halflife = max_range * 0.45

    dx = _PX_X - sx
    dy = _PX_Y - sy
    dist2 = dx * dx + dy * dy

    base = 1.0 / (1.0 + dist2 / (halflife * halflife))
    window = np.clip(1.0 - dist2 / (max_range * max_range), 0.0, 1.0) ** 2
    falloff = base * window
    alpha = np.clip(falloff * 230.0, 0, 230).astype(np.uint8)

    danger = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    rgb = pygame.surfarray.pixels3d(danger)
    a = pygame.surfarray.pixels_alpha(danger)
    rgb[..., 0] = danger_color[0]
    rgb[..., 1] = danger_color[1]
    rgb[..., 2] = danger_color[2]
    a[...] = alpha
    del rgb, a

    mask = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
    pygame.draw.polygon(mask, (255, 255, 255, 255), polygon)
    danger.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)

    surface.blit(danger, (0, 0))


def draw_walls(surface, obstacles, border):
    for rect in border:
        pygame.draw.rect(surface, COLOR_WALL, rect)
    for rect in obstacles:
        pygame.draw.rect(surface, COLOR_WALL, rect)
        pygame.draw.rect(surface, COLOR_WALL_EDGE, rect, width=1)


def draw_hud(surface, font, mode_text, fps):
    lines = [
        "2D Visibility Simulation",
        f"Mode: {mode_text}",
        f"FPS: {fps:.0f}",
        "Move mouse to move light",
        "Wheel: adjust radius / range",
        "Up/Down (or +/-): light intensity",
        "LMB: AI view  |  RMB: place 2nd light",
        "SPACE: toggle 2nd light",
        "R: randomize map  |  ESC: quit",
    ]
    for i, line in enumerate(lines):
        text = font.render(line, True, COLOR_TEXT)
        surface.blit(text, (16, 14 + i * 18))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main():
    pygame.init()
    pygame.display.set_caption("2D Visibility / Lighting Simulation")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Menlo,Consolas,monospace", 14)

    seed = random.randint(0, 9999)
    segments, obstacles, border = build_map(seed)
    segments_packed = prepare_segments(segments)
    max_dist = math.hypot(WIDTH, HEIGHT)

    danger_mode = False
    second_light = False
    second_pos = (WIDTH * 0.75, HEIGHT * 0.5)

    light_radius = 380.0       # Main light maximum reach (pixels).
    light_intensity = 1.0      # Brightness multiplier.
    grenade_range = 320.0      # AI throw range for danger view.
    radius_step = 20.0
    intensity_step = 0.1

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit(0)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit(0)
                elif event.key == pygame.K_r:
                    seed = random.randint(0, 9999)
                    segments, obstacles, border = build_map(seed)
                    segments_packed = prepare_segments(segments)
                elif event.key == pygame.K_SPACE:
                    second_light = not second_light
                elif event.key in (pygame.K_UP, pygame.K_EQUALS, pygame.K_PLUS):
                    light_intensity = min(3.0, light_intensity + intensity_step)
                elif event.key in (pygame.K_DOWN, pygame.K_MINUS):
                    light_intensity = max(0.1, light_intensity - intensity_step)
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    danger_mode = not danger_mode
                elif event.button == 3:
                    second_pos = clamp_point_outside_obstacles(
                        pygame.mouse.get_pos(), obstacles + border
                    )
            elif event.type == pygame.MOUSEWHEEL:
                # Scroll wheel adjusts light reach (or grenade range in AI view).
                if danger_mode:
                    grenade_range = max(60.0, min(900.0, grenade_range + event.y * radius_step))
                else:
                    light_radius = max(60.0, min(1200.0, light_radius + event.y * radius_step))

        mouse = clamp_point_outside_obstacles(
            pygame.mouse.get_pos(), obstacles + border
        )

        # Base scene drawn at full brightness; lighting will modulate it.
        # Floor uses a slightly brighter base so dark areas read as "shadow",
        # not "no scene". Walls are drawn before lighting so they get shaded too.
        screen.fill((48, 56, 88))
        draw_walls(screen, obstacles, border)

        poly = compute_visibility_polygon(mouse, segments_packed, max_dist)

        if danger_mode:
            # Danger view doesn't need ambient lighting modulation; just darken
            # the base scene then overlay the danger gradient.
            dark = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            dark.fill((0, 0, 0, 170))
            screen.blit(dark, (0, 0))
            draw_danger_with_falloff(
                screen, mouse, poly, grenade_range, COLOR_DANGER, COLOR_SAFE
            )
        else:
            lights = []
            b1 = draw_radial_light(
                screen, mouse, poly, light_radius, COLOR_LIGHT, intensity=light_intensity
            )
            lights.append((b1, COLOR_LIGHT))

            if second_light:
                poly2 = compute_visibility_polygon(second_pos, segments_packed, max_dist)
                b2 = draw_radial_light(
                    screen, second_pos, poly2, light_radius * 0.85, (140, 200, 255),
                    intensity=light_intensity * 0.9,
                )
                lights.append((b2, (140, 200, 255)))

            apply_lighting(screen, lights, ambient=0.06)

        # Light source markers (drawn on top, full brightness).
        pygame.draw.circle(screen, COLOR_LIGHT_CORE, mouse, 6)
        pygame.draw.circle(screen, (0, 0, 0), mouse, 6, width=1)
        if second_light and not danger_mode:
            pygame.draw.circle(screen, (180, 220, 255), second_pos, 6)
            pygame.draw.circle(screen, (0, 0, 0), second_pos, 6, width=1)

        if danger_mode:
            mode_text = f"AI grenade view  (range={grenade_range:.0f}px)"
        else:
            mode_text = f"Light  (radius={light_radius:.0f}px, intensity={light_intensity:.1f}x)"
        draw_hud(screen, font, mode_text, clock.get_fps())

        pygame.display.flip()
        clock.tick(FPS)


if __name__ == "__main__":
    main()
