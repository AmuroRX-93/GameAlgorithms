"""
Single-player Pong, in the spirit of Atari's 1972 original but with the
opponent replaced by the top wall. The ball bounces off the left, right,
and top walls forever; only the bottom is lethal. The paddle is split
into 5 zones — the further from center the ball hits, the steeper the
bounce angle, exactly like the Atari arcade cabinet.

Twists added on top of plain Pong:
  - Bricks: a wall of breakable bricks sits in the upper third. Smashing
    a brick scores points and frees the ball back downward.
  - Multi-ball: a new ball joins the field every 100 points, up to a cap.
    Lose them all and it's game over.
  - Speed-up: each paddle hit speeds the ball up a touch, capped so it
    stays catchable.
  - CRT post: scanlines + a soft additive glow over the play field.

Controls:
  - Left / Right arrows or A / D — move paddle
  - Mouse — paddle follows cursor X (more precise)
  - Space — launch the ball at the start of a life / unpause
  - R — restart the game after game over
  - Esc — quit
"""

import math
import random
import sys

import pygame

# ---------------------------------------------------------------------------
# Layout / constants
# ---------------------------------------------------------------------------
WIDTH, HEIGHT = 960, 720
FPS = 60

# Play field bounds: a small inset from the window so the CRT bezel reads.
FIELD_PAD = 24
FIELD = pygame.Rect(FIELD_PAD, FIELD_PAD, WIDTH - 2 * FIELD_PAD, HEIGHT - 2 * FIELD_PAD)

PADDLE_W = 110
PADDLE_H = 14
PADDLE_Y = FIELD.bottom - 38
PADDLE_SPEED = 720.0     # px/s for keyboard control
PADDLE_ZONES = 5         # 5-segment angle table (Atari style)

BALL_R = 7
BALL_SPEED_START = 380.0
BALL_SPEED_MAX = 760.0
BALL_SPEEDUP = 1.04      # multiplier per paddle hit

BRICK_ROWS = 5
BRICK_COLS = 12
BRICK_GAP = 4
BRICK_TOP = FIELD.top + 80
BRICK_H = 22

# Score thresholds where a new ball spawns. Cap the active balls so the
# screen doesn't turn into a snowstorm.
EXTRA_BALL_EVERY = 100
MAX_BALLS = 5

# Colors. Slightly oversaturated to play nicely with the additive glow.
BG          = (8, 10, 16)
BEZEL       = (28, 32, 50)
BEZEL_LINE  = (60, 70, 95)
FIELD_BG    = (10, 14, 22)
FIELD_LINE  = (40, 50, 75)
PADDLE_COL  = (180, 230, 255)
PADDLE_GLOW = (90, 160, 220)
BALL_COL    = (255, 240, 200)
TEXT        = (220, 226, 240)
TEXT_DIM    = (140, 150, 175)
ACCENT      = (245, 210, 80)
GAMEOVER    = (255, 110, 110)

# Per-row brick colors (rainbow-ish, like Breakout).
BRICK_PALETTE = [
    (235,  90,  90),
    (245, 170,  80),
    (245, 220,  90),
    (130, 220, 130),
    (110, 180, 240),
]


# ---------------------------------------------------------------------------
# Entities
# ---------------------------------------------------------------------------


class Paddle:
    """The player's paddle. Lives at a fixed y; x clamps to the field."""

    def __init__(self):
        self.x = FIELD.centerx - PADDLE_W * 0.5
        self.target_x = self.x   # used by mouse smoothing
        self.use_mouse = False

    @property
    def rect(self):
        return pygame.Rect(int(self.x), PADDLE_Y, PADDLE_W, PADDLE_H)

    def update(self, dt, keys, mouse_pos):
        # Mouse takes priority once it moves; switch back to keys when
        # the user presses arrows/A/D.
        kb_dir = 0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            kb_dir -= 1
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            kb_dir += 1
        if kb_dir != 0:
            self.use_mouse = False
            self.x += kb_dir * PADDLE_SPEED * dt
        elif self.use_mouse:
            # Snap toward the mouse with a tiny ease so it isn't jittery.
            target = mouse_pos[0] - PADDLE_W * 0.5
            self.x += (target - self.x) * min(1.0, dt * 18)
        # Clamp within the field.
        self.x = max(FIELD.left, min(FIELD.right - PADDLE_W, self.x))

    def notice_mouse_move(self, mouse_pos):
        # Called when the mouse actually moves; we switch into mouse mode.
        self.use_mouse = True


class Ball:
    """A round bouncing ball. Stores position as floats so collisions
    don't suffer from integer rounding."""

    __slots__ = ("x", "y", "vx", "vy", "speed", "alive", "trail")

    def __init__(self, x, y, vx, vy, speed):
        self.x = x; self.y = y
        self.vx = vx; self.vy = vy
        self.speed = speed
        self.alive = True
        # Recent positions for a short motion trail.
        self.trail = []

    def step(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.trail.append((self.x, self.y))
        if len(self.trail) > 8:
            self.trail.pop(0)


class Brick:
    __slots__ = ("rect", "color", "alive", "points")

    def __init__(self, rect, color, points):
        self.rect = rect
        self.color = color
        self.alive = True
        self.points = points


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_bricks():
    """Lay out a grid of bricks. Returns a list of Brick."""
    inner_w = FIELD.width - (BRICK_COLS + 1) * BRICK_GAP
    bw = inner_w // BRICK_COLS
    bricks = []
    for row in range(BRICK_ROWS):
        for col in range(BRICK_COLS):
            x = FIELD.left + BRICK_GAP + col * (bw + BRICK_GAP)
            y = BRICK_TOP + row * (BRICK_H + BRICK_GAP)
            color = BRICK_PALETTE[row % len(BRICK_PALETTE)]
            # Higher rows are worth more.
            points = (BRICK_ROWS - row) * 10
            bricks.append(Brick(pygame.Rect(x, y, bw, BRICK_H), color, points))
    return bricks


def reflect_off_paddle(ball, paddle):
    """Atari-style 5-zone paddle bounce. The closer to a paddle edge the
    ball lands, the steeper the new vertical angle. Always reflects
    upward."""
    rel = (ball.x - paddle.x) / PADDLE_W      # 0..1 across the paddle
    rel = max(0.0, min(1.0, rel))
    # Bucket into zones: leftmost = -2, ..., rightmost = +2.
    zone = int(rel * PADDLE_ZONES)
    if zone >= PADDLE_ZONES:
        zone = PADDLE_ZONES - 1
    # Map zone to an angle from straight-up (0) toward the side. Edge
    # zones get ~60deg, center zone ~10deg.
    half = (PADDLE_ZONES - 1) * 0.5
    t = (zone - half) / half       # -1..+1
    angle = math.radians(60.0) * t
    speed = min(BALL_SPEED_MAX, ball.speed * BALL_SPEEDUP)
    ball.speed = speed
    ball.vx = math.sin(angle) * speed
    ball.vy = -math.cos(angle) * speed


def aabb_circle_collision(rect, x, y, r):
    """Return (hit, normal_x, normal_y, push_dx, push_dy). Standard
    closest-point-on-rect test. Normal is the outward direction from the
    rect to the ball; push_d* is how far we need to nudge the ball to
    leave the rect again."""
    cx = max(rect.left, min(x, rect.right))
    cy = max(rect.top, min(y, rect.bottom))
    dx = x - cx
    dy = y - cy
    d2 = dx * dx + dy * dy
    if d2 >= r * r:
        return False, 0, 0, 0, 0
    if d2 < 1e-9:
        # Center inside the rect: pick the smallest-overlap side.
        left   = abs(x - rect.left)
        right  = abs(rect.right - x)
        top    = abs(y - rect.top)
        bottom = abs(rect.bottom - y)
        m = min(left, right, top, bottom)
        if m == left:    return True, -1, 0, -(left + r), 0
        if m == right:   return True,  1, 0,  (right + r), 0
        if m == top:     return True, 0, -1, 0, -(top + r)
        return True, 0, 1, 0, (bottom + r)
    d = math.sqrt(d2)
    nx = dx / d
    ny = dy / d
    push = r - d
    return True, nx, ny, nx * push, ny * push


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------


class Game:
    def __init__(self):
        self.paddle = Paddle()
        self.bricks = make_bricks()
        self.balls = []
        self.score = 0
        self.next_extra_at = EXTRA_BALL_EVERY
        self.lives = 3
        self.state = "ready"      # ready | playing | gameover | paused
        self.flash_timer = 0.0    # brief paddle flash on hit
        self.shake = 0.0          # screen shake on brick destruction

    # ----- Ball spawning -----------------------------------------------

    def attach_ball_to_paddle(self):
        """Spawn a single ball stuck to the top of the paddle, ready to
        be launched with Space."""
        cx = self.paddle.x + PADDLE_W * 0.5
        cy = PADDLE_Y - BALL_R - 1
        self.balls = [Ball(cx, cy, 0.0, 0.0, BALL_SPEED_START)]

    def launch_pinned_ball(self):
        for b in self.balls:
            if b.vx == 0 and b.vy == 0:
                # Random small horizontal kick, always upward.
                ang = math.radians(random.uniform(-30, 30))
                b.vx =  math.sin(ang) * b.speed
                b.vy = -math.cos(ang) * b.speed

    def spawn_extra_ball(self):
        """Add a ball mid-flight from a random existing ball, kicked to
        a slightly different angle. Caps at MAX_BALLS."""
        if len(self.balls) >= MAX_BALLS or not self.balls:
            return
        src = random.choice(self.balls)
        ang = math.atan2(src.vy, src.vx) + random.choice([-0.6, 0.6])
        speed = src.speed
        nb = Ball(src.x, src.y,
                  math.cos(ang) * speed, math.sin(ang) * speed, speed)
        self.balls.append(nb)

    # ----- Game flow ----------------------------------------------------

    def reset(self):
        self.paddle = Paddle()
        self.bricks = make_bricks()
        self.score = 0
        self.next_extra_at = EXTRA_BALL_EVERY
        self.lives = 3
        self.state = "ready"
        self.attach_ball_to_paddle()

    def lose_ball(self, b):
        b.alive = False
        if all(not bb.alive for bb in self.balls):
            self.lives -= 1
            if self.lives <= 0:
                self.state = "gameover"
            else:
                self.state = "ready"
                self.attach_ball_to_paddle()

    def add_score(self, points):
        self.score += points
        if self.score >= self.next_extra_at:
            self.next_extra_at += EXTRA_BALL_EVERY
            self.spawn_extra_ball()

    # ----- Update -------------------------------------------------------

    def update(self, dt, keys, mouse_pos):
        self.flash_timer = max(0.0, self.flash_timer - dt)
        self.shake = max(0.0, self.shake - dt * 8)

        self.paddle.update(dt, keys, mouse_pos)

        if self.state in ("ready", "gameover", "paused"):
            # Even when paused, keep the pinned ball glued to the paddle
            # so it doesn't drift.
            if self.state == "ready":
                for b in self.balls:
                    if b.vx == 0 and b.vy == 0:
                        b.x = self.paddle.x + PADDLE_W * 0.5
                        b.y = PADDLE_Y - BALL_R - 1
            return

        # Step each ball with sub-stepping so a fast ball can't tunnel
        # through a brick or the paddle. Up to 4 substeps per frame.
        for b in self.balls:
            if not b.alive:
                continue
            steps = max(1, int(math.ceil(b.speed * dt / (BALL_R * 1.4))))
            steps = min(steps, 4)
            sdt = dt / steps
            for _ in range(steps):
                b.step(sdt)
                self._handle_collisions(b)

        # Drop dead balls.
        self.balls = [b for b in self.balls if b.alive]

        # If we've cleared every brick, spawn a new layer (endless mode).
        if not any(br.alive for br in self.bricks):
            self.bricks = make_bricks()

    def _handle_collisions(self, b):
        # Walls: left, right, top.
        if b.x - BALL_R < FIELD.left:
            b.x = FIELD.left + BALL_R
            b.vx = abs(b.vx)
        if b.x + BALL_R > FIELD.right:
            b.x = FIELD.right - BALL_R
            b.vx = -abs(b.vx)
        if b.y - BALL_R < FIELD.top:
            b.y = FIELD.top + BALL_R
            b.vy = abs(b.vy)

        # Bottom: lose this ball.
        if b.y - BALL_R > FIELD.bottom:
            self.lose_ball(b)
            return

        # Paddle.
        prect = self.paddle.rect
        hit, nx, ny, pdx, pdy = aabb_circle_collision(prect, b.x, b.y, BALL_R)
        if hit and b.vy > 0:
            # Push ball back out, then use the Atari-style angle table.
            b.x += pdx; b.y += pdy
            reflect_off_paddle(b, self.paddle)
            self.flash_timer = 0.12

        # Bricks. We break the first brick we hit per substep, which is
        # plenty for the speeds involved.
        for br in self.bricks:
            if not br.alive:
                continue
            hit, nx, ny, pdx, pdy = aabb_circle_collision(br.rect, b.x, b.y, BALL_R)
            if not hit:
                continue
            br.alive = False
            self.add_score(br.points)
            self.shake = 0.20
            # Push out and reflect along the collision normal.
            b.x += pdx; b.y += pdy
            # Reflect velocity along the normal vector.
            vdotn = b.vx * nx + b.vy * ny
            b.vx -= 2 * vdotn * nx
            b.vy -= 2 * vdotn * ny
            # Slight speed up so the field doesn't get boring.
            new_speed = min(BALL_SPEED_MAX, b.speed * 1.01)
            scale = new_speed / max(1e-6, math.hypot(b.vx, b.vy))
            b.vx *= scale; b.vy *= scale
            b.speed = new_speed
            break


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def draw_field(screen):
    pygame.draw.rect(screen, FIELD_BG, FIELD)
    # Center dashed line — Pong's signature midline. Vertical here since
    # there's no opponent above; we draw a thin horizontal band for
    # decoration instead.
    dash_h = 14
    dash_gap = 10
    cx = FIELD.centerx
    y = FIELD.top + 8
    while y < FIELD.bottom - 8:
        pygame.draw.line(screen, FIELD_LINE,
                         (cx, y), (cx, min(FIELD.bottom - 8, y + dash_h)), 3)
        y += dash_h + dash_gap
    pygame.draw.rect(screen, FIELD_LINE, FIELD, 2)


def draw_paddle(screen, paddle, flash):
    rect = paddle.rect
    glow = PADDLE_GLOW if flash <= 0 else (240, 250, 255)
    inflate = 8
    glow_rect = rect.inflate(inflate, inflate)
    glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
    pygame.draw.rect(glow_surf, (*glow, 80),
                     glow_surf.get_rect(), border_radius=6)
    screen.blit(glow_surf, glow_rect.topleft, special_flags=pygame.BLEND_ADD)
    pygame.draw.rect(screen, PADDLE_COL, rect, border_radius=4)


def draw_ball(screen, b):
    # Trail.
    for i, (tx, ty) in enumerate(b.trail):
        a = int(60 * (i + 1) / len(b.trail))
        s = pygame.Surface((BALL_R * 4, BALL_R * 4), pygame.SRCALPHA)
        pygame.draw.circle(s, (*BALL_COL, a), (BALL_R * 2, BALL_R * 2),
                           int(BALL_R * 0.9))
        screen.blit(s, (tx - BALL_R * 2, ty - BALL_R * 2),
                    special_flags=pygame.BLEND_ADD)
    pygame.draw.circle(screen, BALL_COL, (int(b.x), int(b.y)), BALL_R)
    # Tiny inner highlight.
    pygame.draw.circle(screen, (255, 255, 240),
                       (int(b.x - 2), int(b.y - 2)), max(1, BALL_R - 4))


def draw_bricks(screen, bricks):
    for br in bricks:
        if not br.alive:
            continue
        pygame.draw.rect(screen, br.color, br.rect, border_radius=3)
        # Top highlight stripe for that classic Breakout shading.
        hl = pygame.Rect(br.rect.x + 2, br.rect.y + 2,
                         br.rect.width - 4, 3)
        pygame.draw.rect(screen, (255, 255, 255, 60), hl, border_radius=2)


def draw_hud(screen, font_big, font_small, game):
    # Score (top center).
    txt = font_big.render(f"{game.score:>04d}", True, ACCENT)
    screen.blit(txt, txt.get_rect(midtop=(WIDTH // 2, 6)))

    # Lives (left).
    label = font_small.render("LIVES", True, TEXT_DIM)
    screen.blit(label, (FIELD.left + 2, 8))
    for i in range(game.lives):
        pygame.draw.rect(screen, PADDLE_COL,
                         (FIELD.left + 2 + i * 18, 26, 14, 4))

    # Active balls (right).
    label = font_small.render("BALLS", True, TEXT_DIM)
    screen.blit(label, (FIELD.right - 60, 8))
    for i, _ in enumerate(game.balls):
        pygame.draw.circle(screen, BALL_COL,
                           (FIELD.right - 6 - i * 14, 28), 4)


def draw_overlay_text(screen, font_big, font_small, game):
    if game.state == "ready":
        big = font_big.render("PRESS  SPACE", True, TEXT)
        screen.blit(big, big.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 12)))
        small = font_small.render("Move with mouse / arrows.  Esc to quit.",
                                  True, TEXT_DIM)
        screen.blit(small, small.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 22)))
    elif game.state == "gameover":
        big = font_big.render("GAME  OVER", True, GAMEOVER)
        screen.blit(big, big.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 12)))
        small = font_small.render(f"Final score {game.score}.  R to restart.",
                                  True, TEXT_DIM)
        screen.blit(small, small.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 22)))
    elif game.state == "paused":
        big = font_big.render("PAUSED", True, ACCENT)
        screen.blit(big, big.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 12)))


def draw_scanlines(screen):
    """Cheap CRT scanline overlay: every other pixel row at low alpha."""
    overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    for y in range(0, HEIGHT, 2):
        pygame.draw.line(overlay, (0, 0, 0, 60), (0, y), (WIDTH, y))
    screen.blit(overlay, (0, 0))


def draw_glow(screen):
    """Soft additive glow: scale screen down then up. Cheap bloom."""
    small = pygame.transform.smoothscale(screen, (WIDTH // 4, HEIGHT // 4))
    blurred = pygame.transform.smoothscale(small, (WIDTH, HEIGHT))
    blurred.set_alpha(70)
    screen.blit(blurred, (0, 0), special_flags=pygame.BLEND_ADD)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    pygame.init()
    pygame.display.set_caption("Solo Pong")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font_big = pygame.font.SysFont("Menlo,Consolas,monospace", 36, bold=True)
    font_med = pygame.font.SysFont("Menlo,Consolas,monospace", 18, bold=True)
    font_small = pygame.font.SysFont("Menlo,Consolas,monospace", 12)

    pygame.mouse.set_visible(False)

    game = Game()
    game.attach_ball_to_paddle()

    while True:
        dt_ms = clock.tick(FPS)
        dt = min(1.0 / 30.0, dt_ms / 1000.0)
        keys = pygame.key.get_pressed()
        mouse_pos = pygame.mouse.get_pos()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit(0)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit(0)
                elif event.key == pygame.K_SPACE:
                    if game.state == "ready":
                        game.state = "playing"
                        game.launch_pinned_ball()
                    elif game.state == "playing":
                        game.state = "paused"
                    elif game.state == "paused":
                        game.state = "playing"
                elif event.key == pygame.K_r:
                    if game.state == "gameover":
                        game.reset()
            elif event.type == pygame.MOUSEMOTION:
                game.paddle.notice_mouse_move(mouse_pos)

        game.update(dt, keys, mouse_pos)

        # ---- Render ------------------------------------------------
        screen.fill(BEZEL)
        # Bezel inner outline.
        pygame.draw.rect(screen, BEZEL_LINE,
                         FIELD.inflate(8, 8), 2, border_radius=4)

        # Optional screen shake.
        ox = oy = 0
        if game.shake > 0:
            ox = random.randint(-3, 3)
            oy = random.randint(-3, 3)

        # Draw the field & contents to a temp surface so we can offset it.
        layer = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        draw_field(layer)
        draw_bricks(layer, game.bricks)
        draw_paddle(layer, game.paddle, game.flash_timer)
        for b in game.balls:
            if b.alive:
                draw_ball(layer, b)
        screen.blit(layer, (ox, oy))

        draw_hud(screen, font_med, font_small, game)
        draw_overlay_text(screen, font_big, font_small, game)

        draw_glow(screen)
        draw_scanlines(screen)

        pygame.display.flip()


if __name__ == "__main__":
    main()
