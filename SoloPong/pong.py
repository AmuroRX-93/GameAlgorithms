"""
Pong, two flavors in one binary.

Mode 1 - Solo (Atari-style with bricks):
  Ball bounces off the left, right, and top walls forever; only the
  bottom is lethal. Paddle is split into 5 zones (Atari 1972 angle
  table). Breakable bricks sit in the upper third; clear them all and
  a fresh wall spawns. Multi-ball every 100 points (capped).

Mode 2 - vs AI (survival):
  Classic Pong layout but rotated 90deg, so paddles are horizontal
  bars at the top (AI) and bottom (player). The AI never misses -
  it predicts where the ball will arrive (including wall reflections)
  and slides to meet it. Score grows with survival time and with each
  successful return; lose when the ball passes your paddle.

Controls:
  - Menu: 1 = Solo, 2 = vs AI
  - Left / Right or A / D - move paddle
  - Mouse - paddle follows cursor X
  - Space - launch / pause / resume
  - R - restart after game over
  - M - back to menu
  - Esc - quit
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

FIELD_PAD = 24
FIELD = pygame.Rect(FIELD_PAD, FIELD_PAD, WIDTH - 2 * FIELD_PAD, HEIGHT - 2 * FIELD_PAD)

PADDLE_W = 110
PADDLE_H = 14
PADDLE_Y = FIELD.bottom - 38       # solo / player paddle y
AI_PADDLE_Y = FIELD.top + 24       # AI paddle y in vs-AI mode
PADDLE_SPEED = 720.0
PADDLE_ZONES = 5

# AI tuning. The AI is intentionally fast enough to always reach its
# target; what makes the game playable is that it adds a small return
# wobble so the ball trajectory keeps changing.
AI_PADDLE_SPEED = 1100.0
AI_RETURN_JITTER = 0.18            # 0..1, fraction of paddle half used as offset

BALL_R = 7
BALL_SPEED_START = 380.0
BALL_SPEED_MAX = 760.0
BALL_SPEEDUP = 1.04

BRICK_ROWS = 5
BRICK_COLS = 12
BRICK_GAP = 4
BRICK_TOP = FIELD.top + 80         # solo mode brick start
BRICK_H = 22

# vs-AI mode brick band sits in the middle of the field.
AI_BRICK_ROWS = 3
AI_BRICK_TOP = FIELD.centery - (AI_BRICK_ROWS * (BRICK_H + BRICK_GAP)) // 2

EXTRA_BALL_EVERY = 100
MAX_BALLS = 5

# vs-AI scoring: per-second survival reward + per-return reward.
AI_SURVIVAL_PER_SEC = 5
AI_RETURN_BONUS = 10
AI_PLAYER_HIT_BONUS = 5

BG          = (8, 10, 16)
BEZEL       = (28, 32, 50)
BEZEL_LINE  = (60, 70, 95)
FIELD_BG    = (10, 14, 22)
FIELD_LINE  = (40, 50, 75)
PADDLE_COL  = (180, 230, 255)
PADDLE_GLOW = (90, 160, 220)
AI_PADDLE_COL  = (255, 170, 170)
AI_PADDLE_GLOW = (220, 90, 90)
BALL_COL    = (255, 240, 200)
TEXT        = (220, 226, 240)
TEXT_DIM    = (140, 150, 175)
ACCENT      = (245, 210, 80)
GAMEOVER    = (255, 110, 110)

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
    """Horizontal paddle. y is fixed; x clamps to the field."""

    def __init__(self, y):
        self.x = FIELD.centerx - PADDLE_W * 0.5
        self.y = y
        self.use_mouse = False

    @property
    def rect(self):
        return pygame.Rect(int(self.x), int(self.y), PADDLE_W, PADDLE_H)

    def update(self, dt, keys, mouse_pos):
        kb_dir = 0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            kb_dir -= 1
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            kb_dir += 1
        if kb_dir != 0:
            self.use_mouse = False
            self.x += kb_dir * PADDLE_SPEED * dt
        elif self.use_mouse:
            target = mouse_pos[0] - PADDLE_W * 0.5
            self.x += (target - self.x) * min(1.0, dt * 18)
        self.x = max(FIELD.left, min(FIELD.right - PADDLE_W, self.x))

    def notice_mouse_move(self, mouse_pos):
        self.use_mouse = True


class AIPaddle:
    """Top paddle controlled by a perfect predictor. Never misses."""

    def __init__(self, y):
        self.x = FIELD.centerx - PADDLE_W * 0.5
        self.y = y
        self.target_x = self.x

    @property
    def rect(self):
        return pygame.Rect(int(self.x), int(self.y), PADDLE_W, PADDLE_H)

    def think(self, balls):
        """Pick the most-threatening ball heading up, predict its x at
        self.y including left/right wall reflections, and aim our center
        at that x (with a tiny deliberate offset so returns vary)."""
        threat = None
        best_t = float("inf")
        for b in balls:
            if not b.alive or b.vy >= 0:
                continue
            # Time until the ball center reaches our paddle plane.
            t = (self.y + PADDLE_H - b.y) / b.vy
            if t < 0:
                continue
            if t < best_t:
                best_t = t
                threat = b
        if threat is None:
            # No incoming ball: glide back to center to look ready.
            self.target_x = FIELD.centerx - PADDLE_W * 0.5
            return

        # Predict landing x via mirror-reflection of the field width.
        x_pred = threat.x + threat.vx * best_t
        span = FIELD.width
        # Map x_pred into [FIELD.left, FIELD.right] by repeated reflection.
        rel = x_pred - FIELD.left
        period = 2 * span
        rel = rel % period
        if rel < 0:
            rel += period
        if rel > span:
            rel = period - rel
        x_land = FIELD.left + rel

        # Add deterministic-but-varying offset so reflections aren't
        # straight back. We bias toward the side opposite to the ball's
        # vx so the rally drifts.
        offset = AI_RETURN_JITTER * (PADDLE_W * 0.5)
        x_land -= math.copysign(offset, threat.vx)

        self.target_x = x_land - PADDLE_W * 0.5

    def update(self, dt, balls):
        self.think(balls)
        # Move toward target_x at capped speed.
        dx = self.target_x - self.x
        max_step = AI_PADDLE_SPEED * dt
        if dx > max_step:
            dx = max_step
        elif dx < -max_step:
            dx = -max_step
        self.x += dx
        self.x = max(FIELD.left, min(FIELD.right - PADDLE_W, self.x))


class Ball:
    __slots__ = ("x", "y", "vx", "vy", "speed", "alive", "trail")

    def __init__(self, x, y, vx, vy, speed):
        self.x = x; self.y = y
        self.vx = vx; self.vy = vy
        self.speed = speed
        self.alive = True
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


def make_bricks(rows=BRICK_ROWS, top=BRICK_TOP):
    inner_w = FIELD.width - (BRICK_COLS + 1) * BRICK_GAP
    bw = inner_w // BRICK_COLS
    bricks = []
    for row in range(rows):
        for col in range(BRICK_COLS):
            x = FIELD.left + BRICK_GAP + col * (bw + BRICK_GAP)
            y = top + row * (BRICK_H + BRICK_GAP)
            color = BRICK_PALETTE[row % len(BRICK_PALETTE)]
            points = (rows - row) * 10
            bricks.append(Brick(pygame.Rect(x, y, bw, BRICK_H), color, points))
    return bricks


def reflect_off_paddle(ball, paddle, downward=False):
    """Atari 5-zone reflection. downward=True flips the bounce direction
    (used for the AI paddle which reflects the ball back toward the
    player)."""
    rel = (ball.x - paddle.x) / PADDLE_W
    rel = max(0.0, min(1.0, rel))
    zone = int(rel * PADDLE_ZONES)
    if zone >= PADDLE_ZONES:
        zone = PADDLE_ZONES - 1
    half = (PADDLE_ZONES - 1) * 0.5
    t = (zone - half) / half
    angle = math.radians(60.0) * t
    speed = min(BALL_SPEED_MAX, ball.speed * BALL_SPEEDUP)
    ball.speed = speed
    ball.vx = math.sin(angle) * speed
    ball.vy = (math.cos(angle) if downward else -math.cos(angle)) * speed


def aabb_circle_collision(rect, x, y, r):
    cx = max(rect.left, min(x, rect.right))
    cy = max(rect.top, min(y, rect.bottom))
    dx = x - cx
    dy = y - cy
    d2 = dx * dx + dy * dy
    if d2 >= r * r:
        return False, 0, 0, 0, 0
    if d2 < 1e-9:
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
    """One Game instance can run either gameplay mode. mode is "solo"
    or "ai"; we branch on it for layout, scoring, and update rules."""

    def __init__(self, mode="solo"):
        self.mode = mode
        self.paddle = Paddle(PADDLE_Y)
        self.ai = AIPaddle(AI_PADDLE_Y) if mode == "ai" else None
        if mode == "ai":
            self.bricks = make_bricks(rows=AI_BRICK_ROWS, top=AI_BRICK_TOP)
        else:
            self.bricks = make_bricks()
        self.balls = []
        self.score = 0
        self.next_extra_at = EXTRA_BALL_EVERY
        self.lives = 3 if mode == "solo" else 1
        self.state = "ready"
        self.flash_timer = 0.0       # player paddle flash
        self.ai_flash_timer = 0.0
        self.shake = 0.0
        self.elapsed = 0.0           # seconds survived (vs-AI mode)
        self.score_accum = 0.0       # fractional survival score carry

    # ----- Ball spawning -----------------------------------------------

    def attach_ball_to_paddle(self):
        cx = self.paddle.x + PADDLE_W * 0.5
        cy = self.paddle.y - BALL_R - 1
        self.balls = [Ball(cx, cy, 0.0, 0.0, BALL_SPEED_START)]

    def launch_pinned_ball(self):
        for b in self.balls:
            if b.vx == 0 and b.vy == 0:
                ang = math.radians(random.uniform(-30, 30))
                b.vx =  math.sin(ang) * b.speed
                b.vy = -math.cos(ang) * b.speed

    def spawn_extra_ball(self):
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
        self.__init__(mode=self.mode)
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
        self.score += int(points)
        if self.mode == "solo" and self.score >= self.next_extra_at:
            self.next_extra_at += EXTRA_BALL_EVERY
            self.spawn_extra_ball()

    # ----- Update -------------------------------------------------------

    def update(self, dt, keys, mouse_pos):
        self.flash_timer = max(0.0, self.flash_timer - dt)
        self.ai_flash_timer = max(0.0, self.ai_flash_timer - dt)
        self.shake = max(0.0, self.shake - dt * 8)

        self.paddle.update(dt, keys, mouse_pos)
        if self.ai is not None:
            self.ai.update(dt, self.balls)

        if self.state in ("ready", "gameover", "paused"):
            if self.state == "ready":
                for b in self.balls:
                    if b.vx == 0 and b.vy == 0:
                        b.x = self.paddle.x + PADDLE_W * 0.5
                        b.y = self.paddle.y - BALL_R - 1
            return

        # Survival score in vs-AI mode.
        if self.mode == "ai":
            self.elapsed += dt
            self.score_accum += AI_SURVIVAL_PER_SEC * dt
            whole = int(self.score_accum)
            if whole > 0:
                self.score_accum -= whole
                self.add_score(whole)

        # Sub-stepped ball updates so fast balls don't tunnel.
        for b in self.balls:
            if not b.alive:
                continue
            steps = max(1, int(math.ceil(b.speed * dt / (BALL_R * 1.4))))
            steps = min(steps, 4)
            sdt = dt / steps
            for _ in range(steps):
                b.step(sdt)
                self._handle_collisions(b)

        self.balls = [b for b in self.balls if b.alive]

        # Brick wall regenerates when cleared.
        if not any(br.alive for br in self.bricks):
            if self.mode == "ai":
                self.bricks = make_bricks(rows=AI_BRICK_ROWS, top=AI_BRICK_TOP)
            else:
                self.bricks = make_bricks()

    def _handle_collisions(self, b):
        # Side walls always reflect.
        if b.x - BALL_R < FIELD.left:
            b.x = FIELD.left + BALL_R
            b.vx = abs(b.vx)
        if b.x + BALL_R > FIELD.right:
            b.x = FIELD.right - BALL_R
            b.vx = -abs(b.vx)

        if self.mode == "solo":
            # Top wall reflects; bottom is lethal.
            if b.y - BALL_R < FIELD.top:
                b.y = FIELD.top + BALL_R
                b.vy = abs(b.vy)
            if b.y - BALL_R > FIELD.bottom:
                self.lose_ball(b)
                return
        else:
            # vs AI: top is lethal for AI (but AI never misses, so this
            # is mostly cosmetic - we still bounce off the top wall as a
            # safety net in case prediction fails). Bottom is lethal for
            # the player.
            if b.y - BALL_R < FIELD.top:
                b.y = FIELD.top + BALL_R
                b.vy = abs(b.vy)
            if b.y - BALL_R > FIELD.bottom:
                self.lose_ball(b)
                return

        # Player paddle.
        prect = self.paddle.rect
        hit, nx, ny, pdx, pdy = aabb_circle_collision(prect, b.x, b.y, BALL_R)
        if hit and b.vy > 0:
            b.x += pdx; b.y += pdy
            reflect_off_paddle(b, self.paddle, downward=False)
            self.flash_timer = 0.12
            if self.mode == "ai":
                self.add_score(AI_PLAYER_HIT_BONUS)

        # AI paddle (vs-AI only).
        if self.ai is not None:
            arect = self.ai.rect
            hit, nx, ny, pdx, pdy = aabb_circle_collision(arect, b.x, b.y, BALL_R)
            if hit and b.vy < 0:
                b.x += pdx; b.y += pdy
                reflect_off_paddle(b, self.ai, downward=True)
                self.ai_flash_timer = 0.12
                self.add_score(AI_RETURN_BONUS)

        # Bricks. First brick hit per substep wins.
        for br in self.bricks:
            if not br.alive:
                continue
            hit, nx, ny, pdx, pdy = aabb_circle_collision(br.rect, b.x, b.y, BALL_R)
            if not hit:
                continue
            br.alive = False
            self.add_score(br.points)
            self.shake = 0.20
            b.x += pdx; b.y += pdy
            vdotn = b.vx * nx + b.vy * ny
            b.vx -= 2 * vdotn * nx
            b.vy -= 2 * vdotn * ny
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
    dash_h = 14
    dash_gap = 10
    cx = FIELD.centerx
    y = FIELD.top + 8
    while y < FIELD.bottom - 8:
        pygame.draw.line(screen, FIELD_LINE,
                         (cx, y), (cx, min(FIELD.bottom - 8, y + dash_h)), 3)
        y += dash_h + dash_gap
    pygame.draw.rect(screen, FIELD_LINE, FIELD, 2)


def draw_paddle(screen, paddle, flash, color=PADDLE_COL, glow_color=PADDLE_GLOW):
    rect = paddle.rect
    glow = glow_color if flash <= 0 else (240, 250, 255)
    inflate = 8
    glow_rect = rect.inflate(inflate, inflate)
    glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
    pygame.draw.rect(glow_surf, (*glow, 80),
                     glow_surf.get_rect(), border_radius=6)
    screen.blit(glow_surf, glow_rect.topleft, special_flags=pygame.BLEND_ADD)
    pygame.draw.rect(screen, color, rect, border_radius=4)


def draw_ball(screen, b):
    for i, (tx, ty) in enumerate(b.trail):
        a = int(60 * (i + 1) / len(b.trail))
        s = pygame.Surface((BALL_R * 4, BALL_R * 4), pygame.SRCALPHA)
        pygame.draw.circle(s, (*BALL_COL, a), (BALL_R * 2, BALL_R * 2),
                           int(BALL_R * 0.9))
        screen.blit(s, (tx - BALL_R * 2, ty - BALL_R * 2),
                    special_flags=pygame.BLEND_ADD)
    pygame.draw.circle(screen, BALL_COL, (int(b.x), int(b.y)), BALL_R)
    pygame.draw.circle(screen, (255, 255, 240),
                       (int(b.x - 2), int(b.y - 2)), max(1, BALL_R - 4))


def draw_bricks(screen, bricks):
    for br in bricks:
        if not br.alive:
            continue
        pygame.draw.rect(screen, br.color, br.rect, border_radius=3)
        hl = pygame.Rect(br.rect.x + 2, br.rect.y + 2,
                         br.rect.width - 4, 3)
        pygame.draw.rect(screen, (255, 255, 255, 60), hl, border_radius=2)


def draw_hud(screen, font_med, font_small, game):
    txt = font_med.render(f"{game.score:>05d}", True, ACCENT)
    screen.blit(txt, txt.get_rect(midtop=(WIDTH // 2, 6)))

    if game.mode == "solo":
        label = font_small.render("LIVES", True, TEXT_DIM)
        screen.blit(label, (FIELD.left + 2, 8))
        for i in range(game.lives):
            pygame.draw.rect(screen, PADDLE_COL,
                             (FIELD.left + 2 + i * 18, 26, 14, 4))
        label = font_small.render("BALLS", True, TEXT_DIM)
        screen.blit(label, (FIELD.right - 60, 8))
        for i, _ in enumerate(game.balls):
            pygame.draw.circle(screen, BALL_COL,
                               (FIELD.right - 6 - i * 14, 28), 4)
    else:
        label = font_small.render("TIME", True, TEXT_DIM)
        screen.blit(label, (FIELD.left + 2, 8))
        t = font_small.render(f"{game.elapsed:6.1f}s", True, TEXT)
        screen.blit(t, (FIELD.left + 2, 22))
        label = font_small.render("MODE", True, TEXT_DIM)
        screen.blit(label, (FIELD.right - 60, 8))
        m = font_small.render("vs AI", True, AI_PADDLE_COL)
        screen.blit(m, (FIELD.right - 60, 22))


def draw_overlay_text(screen, font_big, font_small, game):
    if game.state == "ready":
        big = font_big.render("PRESS  SPACE", True, TEXT)
        screen.blit(big, big.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 12)))
        hint = "Move with mouse / arrows.  M = menu, Esc = quit."
        small = font_small.render(hint, True, TEXT_DIM)
        screen.blit(small, small.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 22)))
    elif game.state == "gameover":
        big = font_big.render("GAME  OVER", True, GAMEOVER)
        screen.blit(big, big.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 12)))
        if game.mode == "ai":
            line = f"Survived {game.elapsed:.1f}s.  Score {game.score}.  R restart, M menu."
        else:
            line = f"Final score {game.score}.  R restart, M menu."
        small = font_small.render(line, True, TEXT_DIM)
        screen.blit(small, small.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 22)))
    elif game.state == "paused":
        big = font_big.render("PAUSED", True, ACCENT)
        screen.blit(big, big.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 12)))


def draw_menu(screen, font_big, font_med, font_small):
    title = font_big.render("PONG", True, ACCENT)
    screen.blit(title, title.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 120)))

    sub = font_med.render("Choose a mode", True, TEXT_DIM)
    screen.blit(sub, sub.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 60)))

    opt1 = font_med.render("[1]  SOLO  -  bricks, multi-ball, 3 lives",
                           True, TEXT)
    screen.blit(opt1, opt1.get_rect(center=(WIDTH // 2, HEIGHT // 2)))

    opt2 = font_med.render("[2]  vs AI  -  survive a perfect opponent",
                           True, AI_PADDLE_COL)
    screen.blit(opt2, opt2.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 36)))

    foot = font_small.render("Esc to quit.", True, TEXT_DIM)
    screen.blit(foot, foot.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 100)))


def draw_scanlines(screen):
    overlay = pygame.Surface(screen.get_size(), pygame.SRCALPHA)
    for y in range(0, HEIGHT, 2):
        pygame.draw.line(overlay, (0, 0, 0, 60), (0, y), (WIDTH, y))
    screen.blit(overlay, (0, 0))


def draw_glow(screen):
    small = pygame.transform.smoothscale(screen, (WIDTH // 4, HEIGHT // 4))
    blurred = pygame.transform.smoothscale(small, (WIDTH, HEIGHT))
    blurred.set_alpha(70)
    screen.blit(blurred, (0, 0), special_flags=pygame.BLEND_ADD)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    pygame.init()
    pygame.display.set_caption("Pong")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font_big = pygame.font.SysFont("Menlo,Consolas,monospace", 36, bold=True)
    font_med = pygame.font.SysFont("Menlo,Consolas,monospace", 18, bold=True)
    font_small = pygame.font.SysFont("Menlo,Consolas,monospace", 12)

    pygame.mouse.set_visible(False)

    # Top-level scene: "menu" or "play".
    scene = "menu"
    game = None

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
                if scene == "menu":
                    if event.key == pygame.K_1:
                        game = Game(mode="solo")
                        game.attach_ball_to_paddle()
                        scene = "play"
                    elif event.key == pygame.K_2:
                        game = Game(mode="ai")
                        game.attach_ball_to_paddle()
                        scene = "play"
                else:
                    if event.key == pygame.K_SPACE:
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
                    elif event.key == pygame.K_m:
                        scene = "menu"
                        game = None
            elif event.type == pygame.MOUSEMOTION:
                if scene == "play" and game is not None:
                    game.paddle.notice_mouse_move(mouse_pos)

        # ---- Render -----------------------------------------------------
        screen.fill(BEZEL)
        pygame.draw.rect(screen, BEZEL_LINE,
                         FIELD.inflate(8, 8), 2, border_radius=4)

        if scene == "menu":
            layer = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            draw_field(layer)
            screen.blit(layer, (0, 0))
            draw_menu(screen, font_big, font_med, font_small)
        else:
            game.update(dt, keys, mouse_pos)

            ox = oy = 0
            if game.shake > 0:
                ox = random.randint(-3, 3)
                oy = random.randint(-3, 3)

            layer = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            draw_field(layer)
            draw_bricks(layer, game.bricks)
            draw_paddle(layer, game.paddle, game.flash_timer,
                        color=PADDLE_COL, glow_color=PADDLE_GLOW)
            if game.ai is not None:
                draw_paddle(layer, game.ai, game.ai_flash_timer,
                            color=AI_PADDLE_COL, glow_color=AI_PADDLE_GLOW)
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
