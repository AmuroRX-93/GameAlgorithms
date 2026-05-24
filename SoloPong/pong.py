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

# AI tuning. The AI is fast enough to always reach its target. The
# real challenge for the player comes from where the AI aims: it
# enumerates the 5 reflection zones, simulates each one, and picks
# the return that lands farthest from the player's current x. See
# AIPaddle.think.
AI_PADDLE_SPEED = 1200.0
# Small reflection-angle jitter (radians) added on AI bounces. Continuous
# noise here breaks the otherwise-finite state space of the rally and
# stops AI vs AI demos from collapsing into a 2- or 4-cycle.
AI_ANGLE_JITTER = math.radians(8.0)

BALL_R = 7
BALL_SPEED_START = 380.0
BALL_SPEED_MAX = 760.0
BALL_SPEEDUP = 1.04

BRICK_ROWS = 5
BRICK_COLS = 12
BRICK_GAP = 4
BRICK_TOP = FIELD.top + 80         # solo mode brick start
BRICK_H = 22

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

    def auto_update(self, dt, target_center_x, max_speed=PADDLE_SPEED):
        """Demo / AI-driven move: glide toward target_center_x at the
        given speed cap. Used in demo modes."""
        target = target_center_x - PADDLE_W * 0.5
        diff = target - self.x
        step = max_speed * dt
        if diff > step:
            diff = step
        elif diff < -step:
            diff = -step
        self.x += diff
        self.x = max(FIELD.left, min(FIELD.right - PADDLE_W, self.x))

    def notice_mouse_move(self, mouse_pos):
        self.use_mouse = True


class AIPaddle:
    """Top paddle controlled by a perfect predictor. Never misses.

    Aiming strategy: instead of meeting the ball at paddle center
    (which makes the 5-zone reflector send it back near-vertical), the
    AI offsets itself so the ball lands in an edge zone of the paddle,
    producing a steep bounce. It picks the side away from the player
    so the player has to chase across the field."""

    def __init__(self, y, player=None, bottom=False):
        self.x = FIELD.centerx - PADDLE_W * 0.5
        self.y = y
        self.target_x = self.x
        self.player = player    # opposing paddle (for aim-away logic)
        self.bottom = bottom    # True for the bottom paddle in demo mode
        # Cached aim decision per rally. We resample only when the ball
        # we're tracking changes or its vy direction flips, so the
        # paddle doesn't twitch every frame chasing fresh random choices.
        self._aim_key = None
        self._aim_zone = None

    @property
    def rect(self):
        return pygame.Rect(int(self.x), int(self.y), PADDLE_W, PADDLE_H)

    def think(self, balls, opponent_y=None, bricks=None):
        """For each incoming ball, plan a return that maximizes the
        distance between the ball's eventual x at the opposing paddle
        plane and the opposing paddle's current x. We enumerate the 5
        reflection zones, simulate the resulting trajectory (including
        side-wall reflections via mirror folding), and pick the zone
        with the farthest landing offset from the opponent.

        For the top AI (default, bottom=False) incoming balls have
        vy < 0 and the contact happens at self.y + PADDLE_H. For the
        bottom AI (bottom=True) it's the mirror: incoming have vy > 0
        and contact happens at self.y.

        If `bricks` is provided and we have no opposing paddle (Solo
        demo), the score also includes whether the trajectory will
        actually hit a brick, biasing the AI toward productive
        offensive shots when only a few bricks are left."""
        if self.bottom:
            contact_y = self.y
            incoming_sign = +1   # vy > 0 means heading toward us
        else:
            contact_y = self.y + PADDLE_H
            incoming_sign = -1   # vy < 0 means heading toward us

        threat = None
        best_t = float("inf")
        for b in balls:
            if not b.alive:
                continue
            if incoming_sign * b.vy <= 0:
                continue
            t = (contact_y - b.y) / b.vy
            if t < 0:
                continue
            if t < best_t:
                best_t = t
                threat = b
        if threat is None:
            self.target_x = FIELD.centerx - PADDLE_W * 0.5
            return

        x_contact = _fold_into_field(threat.x + threat.vx * best_t)

        # Opponent's current center, defaults to field center when no
        # opponent is set (e.g. solo demo: just keep ball in play).
        if self.player is not None:
            player_cx = self.player.x + PADDLE_W * 0.5
        else:
            player_cx = FIELD.centerx

        out_speed = min(BALL_SPEED_MAX, threat.speed * BALL_SPEEDUP)
        if opponent_y is not None:
            target_y = opponent_y
        elif self.player is not None:
            target_y = self.player.y
        else:
            # Solo demo: aim for top wall plane (just keep angles wild).
            target_y = FIELD.top
        dy = abs(target_y - contact_y)
        half = (PADDLE_ZONES - 1) * 0.5

        # Score every zone by how far it puts the ball from the
        # opponent's current x. Then weighted-sample one. Using random
        # sampling (instead of strict argmax) breaks the deterministic
        # symmetry that produces dead-loop rallies in AI vs AI demos,
        # and keeps the human-vs-AI mode from feeling robotic.
        dists = []
        zone_vels = []
        for zone in range(PADDLE_ZONES):
            t_zone = (zone - half) / half
            angle = math.radians(60.0) * t_zone
            vx_out = math.sin(angle) * out_speed
            vy_mag = math.cos(angle) * out_speed
            travel_t = dy / max(vy_mag, 1e-3)
            x_arrive = _fold_into_field(x_contact + vx_out * travel_t)
            dists.append(abs(x_arrive - player_cx))
            # Outgoing vy is "away from us": top AI bounces down (+),
            # bottom AI bounces up (-).
            vy_out = vy_mag if self.bottom else -vy_mag
            zone_vels.append((vx_out, vy_out))

        # Solo-demo brick-targeting bonus: when there's no opposing
        # paddle and bricks remain, prefer zones whose trajectory
        # actually hits a brick. Bias scales with sparsity, so in a
        # full layout the AI still shoots loosely, and as the field
        # empties it works the leftovers.
        use_brick_aim = (
            self.player is None and bricks
            and self.bottom  # only the demo's bottom paddle
        )

        # Resample only when this is a new rally (different ball or
        # the ball's vy flipped sign), so the paddle commits to one
        # aim and doesn't twitch every frame.
        key = (id(threat), threat.vy > 0)
        if key != self._aim_key:
            brick_bonus = [0.0] * PADDLE_ZONES
            if use_brick_aim:
                alive_bricks = [br for br in bricks if br.alive]
                if alive_bricks:
                    # Sparsity factor: 0.3 when full, 1.0 when nearly
                    # empty. Even in a packed field we still bias
                    # toward zones that will actually clip a brick
                    # (in Solo demo there's no opponent, so the
                    # "distance from opponent" term is meaningless).
                    # The bias just gets stronger as bricks thin out
                    # and missing them becomes costly.
                    n = len(alive_bricks)
                    sparsity = 0.3 + 0.7 * max(0.0, min(1.0, (50 - n) / 46.0))
                    for zi, (vx_out, vy_out) in enumerate(zone_vels):
                        hit_t = _raycast_first_brick_hit(
                            x_contact, contact_y, vx_out, vy_out,
                            alive_bricks,
                        )
                        if hit_t is not None:
                            # Earlier hits get a bigger bonus.
                            brick_bonus[zi] = sparsity * (1.5 + 1.0 / max(hit_t, 0.05))

            max_d = max(dists) or 1.0
            # Two-tier randomness to defeat dead-loops:
            #   30% of the time pick a fully random zone, ignoring
            #   distance, so the trajectory takes a real detour.
            #   The remaining 70% sample from a softmax-ish weighting
            #   that prefers the farthest zones but keeps every zone
            #   reachable. The combination guarantees the rally never
            #   settles into a 2- or 4-cycle the way a strict argmax
            #   does.
            #
            # In Solo demo with sparse bricks, brick_bonus dominates
            # so the AI actively works the remaining targets instead
            # of bouncing randomly.
            rand_chance = 0.05 if use_brick_aim and any(brick_bonus) else 0.30
            if random.random() < rand_chance:
                chosen = random.randrange(PADDLE_ZONES)
            else:
                weights = [
                    (d / max_d) ** 2 + 0.18 + brick_bonus[i]
                    for i, d in enumerate(dists)
                ]
                total = sum(weights)
                r = random.random() * total
                acc = 0.0
                chosen = 0
                for i, w in enumerate(weights):
                    acc += w
                    if r <= acc:
                        chosen = i
                        break
            self._aim_key = key
            self._aim_zone = chosen

        zone_center_rel = (self._aim_zone + 0.5) / PADDLE_ZONES
        self.target_x = x_contact - zone_center_rel * PADDLE_W

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


def _fold_into_field(x):
    """Mirror-fold an arbitrary x coordinate into [FIELD.left, FIELD.right],
    so a ball traveling in a straight line through the side walls ends
    up at the same x as if it had reflected off them."""
    span = FIELD.width
    period = 2.0 * span
    rel = (x - FIELD.left) % period
    if rel < 0:
        rel += period
    if rel > span:
        rel = period - rel
    return FIELD.left + rel


def _raycast_first_brick_hit(x0, y0, vx, vy, alive_bricks, max_t=2.5):
    """Simulate a free-flying ball from (x0, y0) with velocity (vx, vy),
    bouncing off the side walls and the top wall, until it hits any
    alive brick or `max_t` seconds elapse. Returns the elapsed time on
    impact, or None if no brick is hit. Used by the AI Solo demo to
    pick reflection zones whose trajectory actually clears bricks."""
    if vy == 0:
        return None
    x, y = x0, y0
    t = 0.0
    dt = 1.0 / 120.0  # finer than render dt; we're doing maybe ~300 steps total
    while t < max_t:
        x += vx * dt
        y += vy * dt
        t += dt
        # Reflect off side walls.
        if x < FIELD.left:
            x = 2 * FIELD.left - x
            vx = -vx
        elif x > FIELD.right:
            x = 2 * FIELD.right - x
            vx = -vx
        # Stop when leaving the playable area at top/bottom.
        if y < FIELD.top or y > FIELD.bottom:
            return None
        for br in alive_bricks:
            if br.rect.collidepoint(x, y):
                return t
    return None


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


def reflect_off_paddle(ball, paddle, downward=False, angle_jitter=0.0):
    """Atari 5-zone reflection. downward=True flips the bounce direction
    (used for the AI paddle which reflects the ball back toward the
    player). angle_jitter (radians) adds a small uniform noise to the
    reflection angle on AI bounces, which keeps two perfect AIs from
    collapsing into a finite-state cycle in the demo modes.

    The center zone gives angle=0 -> a perfectly vertical bounce, which
    in Solo mode can trap the ball in a vertical column between paddle
    and wall/brick forever. We always nudge that case off-vertical by a
    small random amount."""
    rel = (ball.x - paddle.x) / PADDLE_W
    rel = max(0.0, min(1.0, rel))
    zone = int(rel * PADDLE_ZONES)
    if zone >= PADDLE_ZONES:
        zone = PADDLE_ZONES - 1
    half = (PADDLE_ZONES - 1) * 0.5
    t = (zone - half) / half
    angle = math.radians(60.0) * t
    if angle_jitter:
        angle += random.uniform(-angle_jitter, angle_jitter)
    # Anti-vertical guard: any time the bounce would leave the ball
    # within ~3 degrees of straight up/down, nudge it 4-10 degrees off
    # to one side. Prevents the "90deg trap" the user reported.
    if abs(angle) < math.radians(3.0):
        sign = 1.0 if random.random() < 0.5 else -1.0
        angle = sign * math.radians(random.uniform(4.0, 10.0))
    # Clamp to slightly less than 90deg so vy stays well non-zero.
    angle = max(-math.radians(75.0), min(math.radians(75.0), angle))
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
    or "ai"; we branch on it for layout, scoring, and update rules.
    demo=True replaces human input with an automated player so the
    game becomes a self-running showcase."""

    def __init__(self, mode="solo", demo=False):
        self.mode = mode
        self.demo = demo
        self.paddle = Paddle(PADDLE_Y)
        # In vs-AI gameplay the AI sits up top. In ai-vs-ai demo we
        # also need a TOP AI; in solo demo there's no AI opponent.
        self.ai = AIPaddle(AI_PADDLE_Y, player=self.paddle) if mode == "ai" else None
        # Demo controller for the bottom paddle: a separate AIPaddle
        # whose y is PADDLE_Y. It plans the same way the top AI does
        # (predicts arrival, picks a zone). For solo demo the player
        # AI doesn't need an opponent to aim at, but it still benefits
        # from prediction; we'll point its "player" reference at None
        # so it just centers its target on the predicted arrival.
        self.player_ai = None
        if demo:
            opponent = self.ai          # may be None (solo demo)
            self.player_ai = AIPaddle(PADDLE_Y, player=opponent)
            # The bottom AI bounces the ball UP, so we need to flip its
            # contact-vs-arrival math. We mark it as 'bottom' so think()
            # treats vy < 0 (away) and vy > 0 (incoming) inverted.
            self.player_ai.bottom = True
        if mode == "ai":
            self.bricks = []
        else:
            self.bricks = make_bricks()
        self.balls = []
        self.score = 0
        self.next_extra_at = EXTRA_BALL_EVERY
        self.lives = 3 if mode == "solo" else 1
        self.state = "ready"
        self.flash_timer = 0.0
        self.ai_flash_timer = 0.0
        self.shake = 0.0
        self.elapsed = 0.0
        self.score_accum = 0.0

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
        self.__init__(mode=self.mode, demo=self.demo)
        self.attach_ball_to_paddle()

    def lose_ball(self, b):
        b.alive = False
        if all(not bb.alive for bb in self.balls):
            self.lives -= 1
            if self.lives <= 0:
                if self.demo:
                    # Demo never ends; just respawn.
                    self.lives = 3 if self.mode == "solo" else 1
                    self.score = 0
                    self.elapsed = 0.0
                    self.score_accum = 0.0
                    self.state = "ready"
                    if self.mode == "solo":
                        self.bricks = make_bricks()
                    self.attach_ball_to_paddle()
                else:
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

        if self.demo and self.player_ai is not None:
            # Demo: bottom paddle is AI-driven. Use the same predictor
            # but ignore keyboard / mouse. think() computes a target_x
            # in screen coordinates; we then move the actual paddle
            # toward it at the AI speed cap.
            self.player_ai.think(self.balls, bricks=self.bricks)
            self.paddle.auto_update(dt, self.player_ai.target_x + PADDLE_W * 0.5,
                                    max_speed=AI_PADDLE_SPEED)
        else:
            self.paddle.update(dt, keys, mouse_pos)
        if self.ai is not None:
            self.ai.update(dt, self.balls)

        if self.state in ("ready", "gameover", "paused"):
            if self.state == "ready":
                for b in self.balls:
                    if b.vx == 0 and b.vy == 0:
                        b.x = self.paddle.x + PADDLE_W * 0.5
                        b.y = self.paddle.y - BALL_R - 1
                # Demo: auto-launch the pinned ball.
                if self.demo:
                    self.state = "playing"
                    self.launch_pinned_ball()
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

        # Brick wall regenerates when cleared (solo mode only; vs-AI
        # has no bricks).
        if self.mode == "solo" and not any(br.alive for br in self.bricks):
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

        # Player paddle. In demo mode the bottom paddle is AI-driven,
        # so apply the same reflection-angle jitter.
        prect = self.paddle.rect
        hit, nx, ny, pdx, pdy = aabb_circle_collision(prect, b.x, b.y, BALL_R)
        if hit and b.vy > 0:
            b.x += pdx; b.y += pdy
            jitter = AI_ANGLE_JITTER if self.demo else 0.0
            reflect_off_paddle(b, self.paddle, downward=False,
                               angle_jitter=jitter)
            self.flash_timer = 0.12
            if self.mode == "ai":
                self.add_score(AI_PLAYER_HIT_BONUS)

        # AI paddle (vs-AI only). Always uses jitter.
        if self.ai is not None:
            arect = self.ai.rect
            hit, nx, ny, pdx, pdy = aabb_circle_collision(arect, b.x, b.y, BALL_R)
            if hit and b.vy < 0:
                b.x += pdx; b.y += pdy
                reflect_off_paddle(b, self.ai, downward=True,
                                   angle_jitter=AI_ANGLE_JITTER)
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
    screen.blit(title, title.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 160)))

    sub = font_med.render("Choose a mode", True, TEXT_DIM)
    screen.blit(sub, sub.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 100)))

    opt1 = font_med.render("[1]  SOLO  -  bricks, multi-ball, 3 lives",
                           True, TEXT)
    screen.blit(opt1, opt1.get_rect(center=(WIDTH // 2, HEIGHT // 2 - 40)))

    opt2 = font_med.render("[2]  vs AI  -  survive a perfect opponent",
                           True, AI_PADDLE_COL)
    screen.blit(opt2, opt2.get_rect(center=(WIDTH // 2, HEIGHT // 2)))

    demo_label = font_med.render("Demos", True, TEXT_DIM)
    screen.blit(demo_label, demo_label.get_rect(
        center=(WIDTH // 2, HEIGHT // 2 + 50)))

    opt3 = font_med.render("[3]  AI  vs  AI  -  watch them rally forever",
                           True, ACCENT)
    screen.blit(opt3, opt3.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 86)))

    opt4 = font_med.render("[4]  AI  Solo  -  watch a perfect bricks run",
                           True, ACCENT)
    screen.blit(opt4, opt4.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 118)))

    foot = font_small.render("Esc to quit.", True, TEXT_DIM)
    screen.blit(foot, foot.get_rect(center=(WIDTH // 2, HEIGHT // 2 + 170)))


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


def _menu_button_rect():
    """Bounding box of the 'MENU' UI button drawn inside the play field
    (placed below the HUD row so it doesn't overlap LIVES/BALLS/TIME)."""
    return pygame.Rect(FIELD.left + 8, FIELD.top + 44, 76, 24)


def draw_menu_button(screen, font_small, hover):
    """A small clickable MENU button shown during gameplay."""
    rect = _menu_button_rect()
    border = ACCENT if hover else BEZEL_LINE
    surf = pygame.Surface(rect.size, pygame.SRCALPHA)
    pygame.draw.rect(surf, (28, 32, 50, 200), surf.get_rect(),
                     border_radius=4)
    pygame.draw.rect(surf, border, surf.get_rect(), 1, border_radius=4)
    screen.blit(surf, rect.topleft)
    label = font_small.render("MENU", True, TEXT if hover else TEXT_DIM)
    screen.blit(label, label.get_rect(center=rect.center))


def main():
    pygame.init()
    pygame.display.set_caption("Pong")
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    font_big = pygame.font.SysFont("Menlo,Consolas,monospace", 36, bold=True)
    font_med = pygame.font.SysFont("Menlo,Consolas,monospace", 18, bold=True)
    font_small = pygame.font.SysFont("Menlo,Consolas,monospace", 12)

    pygame.mouse.set_visible(True)

    # Top-level scene: "menu" or "play".
    scene = "menu"
    game = None

    def start(mode, demo=False):
        g = Game(mode=mode, demo=demo)
        g.attach_ball_to_paddle()
        return g

    while True:
        dt_ms = clock.tick(FPS)
        dt = min(1.0 / 30.0, dt_ms / 1000.0)
        keys = pygame.key.get_pressed()
        mouse_pos = pygame.mouse.get_pos()

        # Reusable hover test for the MENU button.
        menu_btn_rect = _menu_button_rect()
        menu_btn_hover = (scene == "play"
                          and menu_btn_rect.collidepoint(mouse_pos))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit(0)
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit(); sys.exit(0)
                if scene == "menu":
                    if event.key == pygame.K_1:
                        game = start("solo")
                        scene = "play"
                    elif event.key == pygame.K_2:
                        game = start("ai")
                        scene = "play"
                    elif event.key == pygame.K_3:
                        # AI vs AI demo.
                        game = start("ai", demo=True)
                        scene = "play"
                    elif event.key == pygame.K_4:
                        # AI plays solo demo.
                        game = start("solo", demo=True)
                        scene = "play"
                else:
                    if event.key == pygame.K_SPACE and not game.demo:
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
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if scene == "play" and menu_btn_rect.collidepoint(event.pos):
                    scene = "menu"
                    game = None
            elif event.type == pygame.MOUSEMOTION:
                if scene == "play" and game is not None and not game.demo:
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
            # In demos we color the bottom paddle red too, to signal it
            # isn't player-controlled.
            if game.demo:
                bot_col, bot_glow = AI_PADDLE_COL, AI_PADDLE_GLOW
            else:
                bot_col, bot_glow = PADDLE_COL, PADDLE_GLOW
            draw_paddle(layer, game.paddle, game.flash_timer,
                        color=bot_col, glow_color=bot_glow)
            if game.ai is not None:
                draw_paddle(layer, game.ai, game.ai_flash_timer,
                            color=AI_PADDLE_COL, glow_color=AI_PADDLE_GLOW)
            for b in game.balls:
                if b.alive:
                    draw_ball(layer, b)
            screen.blit(layer, (ox, oy))

            draw_hud(screen, font_med, font_small, game)
            draw_overlay_text(screen, font_big, font_small, game)
            draw_menu_button(screen, font_small, menu_btn_hover)

        draw_glow(screen)
        draw_scanlines(screen)

        pygame.display.flip()


if __name__ == "__main__":
    main()
