"""Four-player Battle Pong: human vs three AIs, one paddle on each
edge of the field. Each player has 3 lives; lose one when a ball
crosses your edge. At 0 lives the paddle disappears and that edge
becomes a reflecting wall, so the remaining players keep playing.

This module is self-contained: it reuses constants and the Ball class
from pong.py but defines its own BattlePaddle / BattleAI / BattleGame
because the geometry (paddles on all four sides, including vertical
ones on the left/right) is different enough that bolting it onto the
existing Game would have made every collision branch messy.

Public API used by pong.py:
    SIDES = ("top", "bottom", "left", "right")
    BattleGame(player_side: str)
        .update(dt, keys, mouse_pos)
        .draw(screen, fonts)
        .state in {"ready", "playing", "paused", "gameover"}
        .winner -> "you" | "ai" | None
"""
import math
import random

import pygame

from pong import (
    WIDTH, HEIGHT, FIELD, FIELD_PAD,
    PADDLE_W, PADDLE_H, PADDLE_SPEED, PADDLE_ZONES,
    AI_PADDLE_SPEED, AI_ANGLE_JITTER,
    BALL_R, BALL_SPEED_START, BALL_SPEED_MAX, BALL_SPEEDUP,
    Ball,
    BG, FIELD_BG, FIELD_LINE, BALL_COL, TEXT, TEXT_DIM,
    ACCENT, GAMEOVER, PADDLE_COL, PADDLE_GLOW,
    aabb_circle_collision,
    draw_field, draw_ball, draw_scanlines,
    _menu_button_rect, draw_menu_button,
)

SIDES = ("top", "bottom", "left", "right")

# Distinct color per side so the player can track who's who. Player
# inherits the side color, but we also tag their paddle visually with
# a brighter glow in draw_battle_paddle.
SIDE_COLORS = {
    "top":    ((255, 170, 170), (220,  90,  90)),   # red
    "bottom": ((180, 230, 255), ( 90, 160, 220)),   # blue
    "left":   ((180, 255, 200), ( 90, 200, 130)),   # green
    "right":  ((255, 220, 140), (230, 180,  60)),   # yellow
}
SIDE_LABELS = {
    "top":    "TOP",
    "bottom": "BOTTOM",
    "left":   "LEFT",
    "right":  "RIGHT",
}

# How far the paddle is inset from the field edge.
PADDLE_INSET = 30
# How far the score / lives label sits from the field edge.
LABEL_INSET = 8

BATTLE_LIVES = 3

# Per-rally chance that an AI commits to a "wrong" target and lets the
# ball through. With four paddles (one human, three AIs) on the field,
# perfect AIs would simply rally between themselves forever once the
# human paddle dies. A small fumble rate keeps games finite and gives
# AIs a chance to lose too.
AI_MISS_RATE = 0.12


def _other_sides(side):
    return tuple(s for s in SIDES if s != side)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _paddle_size(side):
    """(w, h) for a paddle on the given side. Top/bottom paddles are
    horizontal; left/right are vertical (rotated)."""
    if side in ("top", "bottom"):
        return PADDLE_W, PADDLE_H
    return PADDLE_H, PADDLE_W


def _paddle_axis(side):
    """Return (axis, range_lo, range_hi) where axis is "x" if the
    paddle slides horizontally, "y" if vertically. range_lo/hi are
    the inclusive clamping range for the paddle's top-left corner."""
    if side in ("top", "bottom"):
        w, _ = _paddle_size(side)
        return "x", FIELD.left, FIELD.right - w
    _, h = _paddle_size(side)
    return "y", FIELD.top, FIELD.bottom - h


def _paddle_initial_pos(side):
    """Top-left corner placement when a paddle is created."""
    w, h = _paddle_size(side)
    if side == "top":
        return (FIELD.centerx - w / 2, FIELD.top + PADDLE_INSET)
    if side == "bottom":
        return (FIELD.centerx - w / 2, FIELD.bottom - PADDLE_INSET - h)
    if side == "left":
        return (FIELD.left + PADDLE_INSET,         FIELD.centery - h / 2)
    return (FIELD.right - PADDLE_INSET - w,        FIELD.centery - h / 2)


def _contact_axis(side):
    """The fixed coordinate of the paddle's contact plane: a y for
    top/bottom paddles, an x for left/right paddles. Used by the AI
    to compute time-to-arrival."""
    if side == "top":
        x, y = _paddle_initial_pos(side)
        return ("y", y + PADDLE_H)        # ball arrives at paddle's bottom edge
    if side == "bottom":
        x, y = _paddle_initial_pos(side)
        return ("y", y)                   # arrives at paddle's top edge
    if side == "left":
        x, y = _paddle_initial_pos(side)
        return ("x", x + PADDLE_H)        # arrives at paddle's right edge
    x, y = _paddle_initial_pos(side)
    return ("x", x)                       # arrives at paddle's left edge


def _slide_coord_for_paddle(side, paddle_x, paddle_y):
    """Return the coordinate the paddle moves along (corner top-left)."""
    return paddle_x if side in ("top", "bottom") else paddle_y


def _slide_extent(side):
    """The paddle's length along its movement axis."""
    w, h = _paddle_size(side)
    return w if side in ("top", "bottom") else h


# ---------------------------------------------------------------------------
# Paddle
# ---------------------------------------------------------------------------


class BattlePaddle:
    """One paddle in battle mode. Knows which side it's on, has a
    sliding axis ("x" or "y"), and tracks lives. When lives reaches 0
    the paddle is marked dead -- it stops being drawn and stops being
    a collision target, so balls just pass through into the wall
    behind it (which then reflects them; see BattleGame._handle_walls)."""

    def __init__(self, side, controller="human"):
        self.side = side
        self.controller = controller   # "human" or "ai"
        self.alive = True
        self.lives = BATTLE_LIVES
        x, y = _paddle_initial_pos(side)
        self.x, self.y = x, y
        self.use_mouse = False
        self.flash_timer = 0.0

    @property
    def rect(self):
        w, h = _paddle_size(self.side)
        return pygame.Rect(int(self.x), int(self.y), w, h)

    @property
    def axis(self):
        return "x" if self.side in ("top", "bottom") else "y"

    def slide_pos(self):
        return self.x if self.axis == "x" else self.y

    def set_slide(self, val):
        _, lo, hi = _paddle_axis(self.side)
        val = max(lo, min(hi, val))
        if self.axis == "x":
            self.x = val
        else:
            self.y = val

    # ----- Movement -----------------------------------------------------

    def update_human(self, dt, keys, mouse_pos):
        # The mapping is: along the paddle's axis, "decrease" key is
        # left/up and "increase" key is right/down. We accept arrows
        # and WASD interchangeably. Mouse follows the cursor's
        # corresponding coordinate.
        if self.axis == "x":
            kb = 0
            if keys[pygame.K_LEFT]  or keys[pygame.K_a]: kb -= 1
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]: kb += 1
            mouse_target = mouse_pos[0] - _slide_extent(self.side) / 2
        else:
            kb = 0
            if keys[pygame.K_UP]    or keys[pygame.K_w]: kb -= 1
            if keys[pygame.K_DOWN]  or keys[pygame.K_s]: kb += 1
            mouse_target = mouse_pos[1] - _slide_extent(self.side) / 2

        if kb != 0:
            self.use_mouse = False
            self.set_slide(self.slide_pos() + kb * PADDLE_SPEED * dt)
        elif self.use_mouse:
            cur = self.slide_pos()
            self.set_slide(cur + (mouse_target - cur) * min(1.0, dt * 18))

    def update_ai(self, dt, target_slide, max_speed=AI_PADDLE_SPEED):
        cur = self.slide_pos()
        diff = target_slide - cur
        step = max_speed * dt
        if diff > step:  diff = step
        elif diff < -step: diff = -step
        self.set_slide(cur + diff)

    def notice_mouse_move(self):
        self.use_mouse = True


# ---------------------------------------------------------------------------
# Reflection
# ---------------------------------------------------------------------------


def _reflect(ball, paddle, angle_jitter=0.0):
    """5-zone Atari reflection generalized to all four sides. We work
    in the paddle's local frame: "u" is the sliding axis (where on
    the paddle the ball hit) and "n" is the inward-pointing normal
    (the direction the ball flies away). Then we map back to (vx, vy).

    Same anti-vertical guard as pong.reflect_off_paddle: any bounce
    that would send the ball within ~3 deg of straight along the
    normal gets nudged 4-10 deg sideways."""
    side = paddle.side
    rect = paddle.rect

    if side in ("top", "bottom"):
        # u is along x, n along +/- y.
        rel = (ball.x - rect.left) / rect.width
        u_axis_len = rect.width
        n_sign = 1 if side == "top" else -1   # bounce away from edge
    else:
        rel = (ball.y - rect.top) / rect.height
        u_axis_len = rect.height
        n_sign = 1 if side == "left" else -1

    rel = max(0.0, min(1.0, rel))
    zone = int(rel * PADDLE_ZONES)
    if zone >= PADDLE_ZONES:
        zone = PADDLE_ZONES - 1
    half = (PADDLE_ZONES - 1) * 0.5
    t = (zone - half) / half
    angle = math.radians(60.0) * t
    if angle_jitter:
        angle += random.uniform(-angle_jitter, angle_jitter)
    if abs(angle) < math.radians(3.0):
        sign = 1.0 if random.random() < 0.5 else -1.0
        angle = sign * math.radians(random.uniform(4.0, 10.0))
    angle = max(-math.radians(75.0), min(math.radians(75.0), angle))

    speed = min(BALL_SPEED_MAX, ball.speed * BALL_SPEEDUP)
    ball.speed = speed
    # In the local frame: vu = sin(angle)*speed (along the paddle),
    # vn = cos(angle)*speed * n_sign (into the field).
    vu = math.sin(angle) * speed
    vn = math.cos(angle) * speed * n_sign
    if side in ("top", "bottom"):
        ball.vx = vu
        ball.vy = vn
    else:
        ball.vy = vu
        ball.vx = vn


# ---------------------------------------------------------------------------
# AI: predict where the ball will arrive at this paddle's contact plane
# (with mirror-folding off the two perpendicular walls), and slide so
# the ball lands in our paddle. We don't aim aggressively here -- with
# 4 paddles around the field, just being a perfect blocker is already
# very hard for the human to score against.
# ---------------------------------------------------------------------------


def _fold_into_range(v, lo, hi):
    """Mirror-fold v into [lo, hi] so a straight line through the two
    walls at lo/hi maps to its reflected coordinate. Same trick as
    pong._fold_into_field, generalized to either axis."""
    span = hi - lo
    if span <= 0:
        return v
    period = 2.0 * span
    rel = (v - lo) % period
    if rel < 0:
        rel += period
    if rel > span:
        rel = period - rel
    return lo + rel


def _predict_arrival(side, ball):
    """Predict (arrival_coord_along_paddle_axis, time_until_arrival)
    for a ball heading toward this side's contact plane. Returns
    (None, inf) if the ball isn't heading toward this side.

    Side-walls perpendicular to the contact plane are treated as
    perfect reflectors via mirror folding; this matches the actual
    in-game collision response so the prediction stays accurate as
    long as the ball doesn't bounce off another paddle on the way."""
    plane_axis, plane_val = _contact_axis(side)

    if plane_axis == "y":
        # Ball must be moving toward this y plane.
        if side == "top"    and ball.vy >= 0: return None, float("inf")
        if side == "bottom" and ball.vy <= 0: return None, float("inf")
        if ball.vy == 0:
            return None, float("inf")
        t = (plane_val - ball.y) / ball.vy
        if t < 0:
            return None, float("inf")
        x_arrive = _fold_into_range(
            ball.x + ball.vx * t, FIELD.left, FIELD.right)
        return x_arrive, t
    else:
        if side == "left"  and ball.vx >= 0: return None, float("inf")
        if side == "right" and ball.vx <= 0: return None, float("inf")
        if ball.vx == 0:
            return None, float("inf")
        t = (plane_val - ball.x) / ball.vx
        if t < 0:
            return None, float("inf")
        y_arrive = _fold_into_range(
            ball.y + ball.vy * t, FIELD.top, FIELD.bottom)
        return y_arrive, t


def _enemy_paddle(paddle, paddles, player_side):
    """Pick which other paddle this AI is "hunting" right now.

    Strategy: while the human is alive, every AI gangs up on the
    human (makes the human-vs-3-AIs fantasy actually feel like 1v3).
    Once the human dies, switch to a stable pairing: each AI locks
    onto the survivor whose side faces it across the field. That
    keeps the AI-vs-AI endgame as targeted duels, so somebody loses
    their lives quickly instead of three perfect-aim paddles forming
    a stable triangle. Returns None if there's no valid target."""
    candidates = [
        p for p in paddles.values()
        if p.alive and p.side != paddle.side
    ]
    if not candidates:
        return None
    human = paddles.get(player_side)
    if human is not None and human.alive and human.side != paddle.side:
        return human

    # All-AI phase: prefer the paddle on the opposite side.
    opposite = {"top": "bottom", "bottom": "top",
                "left": "right", "right": "left"}
    opp = paddles.get(opposite[paddle.side])
    if opp is not None and opp.alive:
        return opp
    # Fallback: surviving opponent with most lives.
    side_priority = {s: i for i, s in enumerate(SIDES)}
    candidates.sort(
        key=lambda p: (-p.lives, side_priority[p.side]))
    return candidates[0]


def _zone_outgoing(paddle, contact_along, zone, out_speed):
    """For a paddle on `paddle.side`, given the contact point along
    the paddle's slide axis (a coordinate, not a relative) and a zone
    index 0..PADDLE_ZONES-1, return the outgoing (vx, vy) of the ball
    after a clean reflect (no jitter, no anti-vertical guard -- this
    is the AI's *plan*, not a real physics step).

    contact_along is x for top/bottom paddles, y for left/right."""
    half = (PADDLE_ZONES - 1) * 0.5
    t = (zone - half) / half
    angle = math.radians(60.0) * t
    side = paddle.side
    if side in ("top", "bottom"):
        n_sign = 1 if side == "top" else -1     # vy after bounce
        vx = math.sin(angle) * out_speed
        vy = math.cos(angle) * out_speed * n_sign
    else:
        n_sign = 1 if side == "left" else -1    # vx after bounce
        vy = math.sin(angle) * out_speed
        vx = math.cos(angle) * out_speed * n_sign
    return vx, vy


def _arrive_offset_from_enemy(paddle, contact_along, zone, out_speed,
                               enemy):
    """Simulate the bounced trajectory and return how far it lands
    from `enemy`'s current paddle center (in enemy's slide-axis units).
    Larger == harder for enemy to reach. Wall reflections on the two
    perpendicular walls are handled by mirror-folding.

    Returns (offset, hit_enemy_plane: bool). hit_enemy_plane is False
    when the bounced ball never reaches the enemy's contact plane
    (e.g. the geometry doesn't intersect that side at all)."""
    vx, vy = _zone_outgoing(paddle, contact_along, zone, out_speed)

    # Where does the ball start? At the paddle's contact plane, at
    # contact_along on the slide axis.
    paddle_axis, paddle_plane = _contact_axis(paddle.side)
    if paddle_axis == "y":
        x0, y0 = contact_along, paddle_plane
    else:
        x0, y0 = paddle_plane, contact_along

    enemy_axis, enemy_plane = _contact_axis(enemy.side)
    if enemy_axis == "y":
        # Need vy != 0 in the right direction to ever reach the plane.
        if vy == 0 or (enemy_plane - y0) * vy < 0:
            return 0.0, False
        t = (enemy_plane - y0) / vy
        if t <= 0:
            return 0.0, False
        x_at = _fold_into_range(x0 + vx * t, FIELD.left, FIELD.right)
        enemy_cx = enemy.x + _slide_extent(enemy.side) * 0.5
        return abs(x_at - enemy_cx), True
    else:
        if vx == 0 or (enemy_plane - x0) * vx < 0:
            return 0.0, False
        t = (enemy_plane - x0) / vx
        if t <= 0:
            return 0.0, False
        y_at = _fold_into_range(y0 + vy * t, FIELD.top, FIELD.bottom)
        enemy_cy = enemy.y + _slide_extent(enemy.side) * 0.5
        return abs(y_at - enemy_cy), True


def _ai_target_slide(paddle, balls, paddles=None, player_side=None):
    """Where (in slide-axis coords) should this AI paddle's top-left
    corner be?

    Two-step plan:
      1. Predict where the soonest-incoming ball will arrive at our
         contact plane.
      2. If we have an enemy locked, evaluate the 5 reflection zones
         and pick the one whose post-bounce trajectory lands farthest
         from the enemy paddle. Place ourselves so the ball hits *that
         zone* of our paddle, not just our center.

    With three perfect AIs in play, AI-vs-AI rallies would never end.
    To keep the game finite (and to give the human a fighting chance
    in the survival race), each AI has a small per-rally chance to
    "fumble" -- it commits to a target offset that misses the ball by
    enough to drop it. The fumble is decided once per ball
    direction-flip, so a single mistake doesn't spam every frame; it's
    a clean miss, not a jitter."""
    best_t = float("inf")
    best_arrive = None
    best_ball = None
    for b in balls:
        if not b.alive:
            continue
        arrive, t = _predict_arrival(paddle.side, b)
        if arrive is None:
            continue
        if t < best_t:
            best_t = t
            best_arrive = arrive
            best_ball = b

    if best_arrive is None:
        paddle._fumble_key = None
        _, lo, hi = _paddle_axis(paddle.side)
        return (lo + hi) * 0.5

    extent = _slide_extent(paddle.side)

    # Pick a zone to aim for. Without an enemy reference (shouldn't
    # happen in normal battle, but kept for robustness) we fall back
    # to dead-center placement.
    enemy = (_enemy_paddle(paddle, paddles, player_side)
             if paddles is not None else None)

    # Cache the choice per rally so the paddle moves smoothly to its
    # target instead of re-evaluating zones every frame and twitching.
    key = id(best_ball)
    if getattr(paddle, "_aim_key", None) != key:
        paddle._aim_key = key
        # Fumble decision is also per-rally so a miss is decisive.
        if random.random() < AI_MISS_RATE:
            paddle._fumble_offset = extent * random.uniform(0.9, 1.4)
            if random.random() < 0.5:
                paddle._fumble_offset = -paddle._fumble_offset
        else:
            paddle._fumble_offset = 0.0

        if enemy is None:
            paddle._aim_zone = (PADDLE_ZONES - 1) // 2
        else:
            # Find the threat ball's speed for outgoing speedup math.
            out_speed = min(BALL_SPEED_MAX, best_ball.speed * BALL_SPEEDUP)
            scores = []
            any_hit = False
            for z in range(PADDLE_ZONES):
                offset, hit = _arrive_offset_from_enemy(
                    paddle, best_arrive, z, out_speed, enemy)
                if hit:
                    any_hit = True
                scores.append(offset if hit else -1.0)
            if any_hit:
                # Weighted random over zones, biased heavily toward
                # the farthest. Pure argmax would make every AI
                # paddle pick the same edge zone every rally and
                # rallies would loop; the bias + randomness breaks
                # the symmetry.
                max_s = max(scores) or 1.0
                weights = [(max(0.0, s) / max_s) ** 2 + 0.18 for s in scores]
                total = sum(weights)
                r = random.random() * total
                acc = 0.0
                chosen = 0
                for i, w in enumerate(weights):
                    acc += w
                    if r <= acc:
                        chosen = i
                        break
                paddle._aim_zone = chosen
            else:
                # No zone reaches the enemy plane (rare; unusual
                # geometry like the enemy is on the same axis).
                paddle._aim_zone = (PADDLE_ZONES - 1) // 2

    # Place ourselves so contact happens in the chosen zone.
    # The ball arrives at coord `best_arrive` along our slide axis;
    # we want that coord to land inside zone `_aim_zone` of our
    # paddle. Zone center as a fraction of paddle length:
    zone_center_rel = (paddle._aim_zone + 0.5) / PADDLE_ZONES
    base = best_arrive - zone_center_rel * extent
    return base + paddle._fumble_offset


# ---------------------------------------------------------------------------
# Game state
# ---------------------------------------------------------------------------


class BattleGame:
    """1 vs 3: human picks one of the four sides, the other three are
    perfect-prediction AIs. Each player has 3 lives. Lose one when a
    ball crosses your side's contact plane (i.e. it gets past your
    paddle, or your paddle is dead and the ball passes the line where
    your paddle used to be -- but in the dead case the side is a wall,
    see _handle_walls). When a paddle hits 0 lives it dies; that side
    becomes a permanent reflecting wall.

    Game ends when one player remains alive. If that player is the
    human, they win; otherwise the AIs win and we still report which
    side won so we can show its color in the GAME OVER overlay."""

    mode = "battle"

    def __init__(self, player_side):
        if player_side not in SIDES:
            raise ValueError(f"player_side must be one of {SIDES}")
        self.player_side = player_side
        self.paddles = {
            side: BattlePaddle(
                side,
                controller="human" if side == player_side else "ai",
            )
            for side in SIDES
        }
        # One ball, spawned at center pinned until SPACE.
        self.balls = [
            Ball(FIELD.centerx, FIELD.centery, 0.0, 0.0, BALL_SPEED_START)
        ]
        self.state = "ready"
        self.winner_side = None
        self.shake = 0.0
        self.elapsed = 0.0

    # ----- Helpers ------------------------------------------------------

    def alive_paddles(self):
        return [p for p in self.paddles.values() if p.alive]

    def attach_ball(self):
        """Pin the ball at field center, ready for SPACE."""
        for b in self.balls:
            b.x = FIELD.centerx
            b.y = FIELD.centery
            b.vx = 0.0
            b.vy = 0.0
            b.speed = BALL_SPEED_START
            b.alive = True
            b.trail.clear()

    def launch_ball(self):
        """Send the pinned ball off in a random direction. Avoid
        near-axis-aligned launches so the first volley doesn't go
        straight to one paddle."""
        for b in self.balls:
            if b.vx == 0 and b.vy == 0:
                # Pick a random angle, then reject any that's within
                # 15 deg of an axis (those produce boring opening volleys).
                while True:
                    ang = random.uniform(0, 2 * math.pi)
                    snap = ang % (math.pi / 2)
                    if math.radians(15) < snap < math.radians(75):
                        break
                b.vx = math.cos(ang) * b.speed
                b.vy = math.sin(ang) * b.speed

    # ----- Update -------------------------------------------------------

    def update(self, dt, keys, mouse_pos):
        self.shake = max(0.0, self.shake - dt * 8)
        for p in self.paddles.values():
            p.flash_timer = max(0.0, p.flash_timer - dt)

        # Move paddles. Human input only goes to the player's paddle;
        # the rest are AI.
        for p in self.paddles.values():
            if not p.alive:
                continue
            if p.controller == "human":
                p.update_human(dt, keys, mouse_pos)
            else:
                target = _ai_target_slide(
                    p, self.balls,
                    paddles=self.paddles,
                    player_side=self.player_side)
                p.update_ai(dt, target)

        if self.state in ("ready", "paused", "gameover"):
            return

        self.elapsed += dt

        # Sub-stepped ball motion to avoid tunneling at high speed.
        for b in self.balls:
            if not b.alive:
                continue
            steps = max(1, int(math.ceil(b.speed * dt / (BALL_R * 1.4))))
            steps = min(steps, 6)
            sdt = dt / steps
            for _ in range(steps):
                b.step(sdt)
                self._handle_collisions(b)
                if not b.alive:
                    break

        # Ball lost -> figure out which side, deduct a life, respawn.
        if all(not b.alive for b in self.balls):
            # _handle_collisions already credited the losing side and
            # set state appropriately; if the game isn't over, respawn
            # immediately and keep playing -- battle pong is fast and
            # asking the player to press SPACE between every life loss
            # would kill the pace.
            if self.state == "playing":
                self.attach_ball()
                self.launch_ball()

    # ----- Collisions ---------------------------------------------------

    def _handle_collisions(self, b):
        # 1. Edges / walls. For each side: if that side's paddle is
        #    dead OR the side has no paddle (shouldn't happen in
        #    battle, kept symmetric), the edge reflects the ball.
        #    Otherwise the edge is the "death plane" -- if the ball
        #    crosses it (past the paddle), the owner of that side
        #    loses a life and the ball is removed.
        #
        # We define each side's death plane as just past the field
        # edge (i.e. one ball radius beyond), so a ball that gets past
        # the paddle still has to travel a bit before counting as a
        # miss -- gives the rendering a visible "ball goes off-screen"
        # feeling instead of vanishing the instant it grazes the edge.
        for side in SIDES:
            paddle = self.paddles[side]
            if paddle.alive:
                continue   # alive paddles are handled in step 3
            # Side has no paddle: it's a wall.
            if side == "top" and b.y - BALL_R < FIELD.top:
                b.y = FIELD.top + BALL_R
                b.vy = abs(b.vy)
            elif side == "bottom" and b.y + BALL_R > FIELD.bottom:
                b.y = FIELD.bottom - BALL_R
                b.vy = -abs(b.vy)
            elif side == "left" and b.x - BALL_R < FIELD.left:
                b.x = FIELD.left + BALL_R
                b.vx = abs(b.vx)
            elif side == "right" and b.x + BALL_R > FIELD.right:
                b.x = FIELD.right - BALL_R
                b.vx = -abs(b.vx)

        # 2. Death-plane checks for sides whose paddle is still alive.
        for side in SIDES:
            paddle = self.paddles[side]
            if not paddle.alive:
                continue
            crossed = False
            if side == "top"    and b.y < FIELD.top - BALL_R:    crossed = True
            if side == "bottom" and b.y > FIELD.bottom + BALL_R: crossed = True
            if side == "left"   and b.x < FIELD.left - BALL_R:   crossed = True
            if side == "right"  and b.x > FIELD.right + BALL_R:  crossed = True
            if crossed:
                self._lose_life(side)
                b.alive = False
                return

        # 3. Paddle collisions (only alive paddles can be hit).
        for side in SIDES:
            paddle = self.paddles[side]
            if not paddle.alive:
                continue
            hit, nx, ny, pdx, pdy = aabb_circle_collision(
                paddle.rect, b.x, b.y, BALL_R)
            if not hit:
                continue
            # Only count as a return if the ball was approaching the
            # paddle. Without this an AI paddle's quick swipe could
            # accidentally "pull" a receding ball back.
            approaching = (
                (side == "top"    and b.vy < 0) or
                (side == "bottom" and b.vy > 0) or
                (side == "left"   and b.vx < 0) or
                (side == "right"  and b.vx > 0)
            )
            if not approaching:
                continue
            b.x += pdx; b.y += pdy
            jitter = AI_ANGLE_JITTER if paddle.controller == "ai" else 0.0
            _reflect(b, paddle, angle_jitter=jitter)
            paddle.flash_timer = 0.12
            return  # one paddle hit per substep

    def _lose_life(self, side):
        paddle = self.paddles[side]
        paddle.lives -= 1
        self.shake = 0.30
        if paddle.lives <= 0:
            paddle.alive = False
        # Check for game end: a single survivor wins.
        alive = self.alive_paddles()
        if len(alive) <= 1:
            self.state = "gameover"
            self.winner_side = alive[0].side if alive else None

    # ----- Public flow control -----------------------------------------

    @property
    def winner(self):
        if self.winner_side is None:
            return None
        return "you" if self.winner_side == self.player_side else "ai"

    def reset(self):
        self.__init__(self.player_side)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _draw_battle_paddle(screen, paddle, is_player):
    """Render a single paddle. Dead paddles are skipped. Live ones use
    their side's color; the player's paddle gets a brighter glow so it
    stands out from the three AIs on screen."""
    if not paddle.alive:
        return
    color, glow_color = SIDE_COLORS[paddle.side]
    if paddle.flash_timer > 0:
        glow = (240, 250, 255)
    else:
        glow = glow_color
    rect = paddle.rect
    inflate = 10 if is_player else 8
    glow_rect = rect.inflate(inflate, inflate)
    glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
    pygame.draw.rect(glow_surf, (*glow, 110 if is_player else 80),
                     glow_surf.get_rect(), border_radius=6)
    screen.blit(glow_surf, glow_rect.topleft, special_flags=pygame.BLEND_ADD)
    pygame.draw.rect(screen, color, rect, border_radius=4)


def _draw_lives_ring(screen, font_small, paddle, is_player):
    """Show "OWNER  ###" on the side of the screen for each player's
    remaining lives. We anchor each label just outside its side's edge
    of the field so the player can see at a glance who's losing."""
    color, _ = SIDE_COLORS[paddle.side]
    if not paddle.alive:
        color = (90, 95, 110)

    label = SIDE_LABELS[paddle.side]
    if is_player:
        label = f"{label} (YOU)"
    text = f"{label}  {paddle.lives}/{BATTLE_LIVES}"
    surf = font_small.render(text, True, color)
    rect = surf.get_rect()

    # Place the label outside the field, hugging its side. The labels
    # share the screen's outer frame; we don't rotate anything (small
    # text reads fine even on the side edges).
    if paddle.side == "top":
        rect.midbottom = (FIELD.centerx, FIELD.top - 4)
    elif paddle.side == "bottom":
        rect.midtop = (FIELD.centerx, FIELD.bottom + 4)
    elif paddle.side == "left":
        rect.topleft = (FIELD.left - 4, FIELD.top - 18)
    else:
        rect.topright = (FIELD.right + 4, FIELD.top - 18)
    screen.blit(surf, rect)


def draw_battle(screen, fonts, game, mouse_pos):
    """Full-frame rendering for BattleGame. fonts is the (font_big,
    font_med, font_small) tuple shared with main.py."""
    font_big, font_med, font_small = fonts
    screen.fill(BG)

    # Subtle screen shake on big events (life loss). Apply by drawing
    # the field offset; we just translate everything by a small jitter.
    ox = oy = 0
    if game.shake > 0:
        amp = game.shake * 6
        ox = random.uniform(-amp, amp)
        oy = random.uniform(-amp, amp)

    # We can't easily translate everything, so we just draw normally
    # at (ox, oy) using a sub-surface trick: paint to a temp surface
    # then blit shifted. For simplicity, skip the shake when amp is
    # very small; otherwise use a temp surface.
    if game.shake > 0.05:
        tmp = pygame.Surface((WIDTH, HEIGHT))
        tmp.fill(BG)
        target = tmp
    else:
        target = screen

    draw_field(target)

    for side in SIDES:
        _draw_battle_paddle(
            target, game.paddles[side],
            is_player=(side == game.player_side))

    for b in game.balls:
        if b.alive:
            draw_ball(target, b)

    if target is not screen:
        screen.blit(target, (int(ox), int(oy)))

    # HUD: per-side lives.
    for side in SIDES:
        _draw_lives_ring(screen, font_small, game.paddles[side],
                         is_player=(side == game.player_side))

    # Big top-center label: "1 v 1 v 1 v 1" + elapsed time.
    title = font_small.render("1 v 1 v 1 v 1", True, TEXT_DIM)
    screen.blit(title, title.get_rect(midtop=(WIDTH // 2, 4)))
    if game.state == "playing":
        t = font_med.render(f"{game.elapsed:5.1f}s", True, ACCENT)
        screen.blit(t, t.get_rect(midtop=(WIDTH // 2, 18)))

    # Overlays.
    if game.state == "ready":
        big = font_big.render("PRESS  SPACE", True, TEXT)
        screen.blit(big, big.get_rect(
            center=(WIDTH // 2, HEIGHT // 2 - 12)))
        sub = font_small.render(
            "You play the highlighted side. Three AIs fight for the rest.",
            True, TEXT_DIM)
        screen.blit(sub, sub.get_rect(
            center=(WIDTH // 2, HEIGHT // 2 + 22)))
    elif game.state == "paused":
        big = font_big.render("PAUSED", True, ACCENT)
        screen.blit(big, big.get_rect(
            center=(WIDTH // 2, HEIGHT // 2 - 12)))
    elif game.state == "gameover":
        if game.winner == "you":
            big = font_big.render("YOU  WIN", True, ACCENT)
        elif game.winner_side is not None:
            color, _ = SIDE_COLORS[game.winner_side]
            big = font_big.render(
                f"{SIDE_LABELS[game.winner_side]}  WINS", True, color)
        else:
            big = font_big.render("DRAW", True, GAMEOVER)
        screen.blit(big, big.get_rect(
            center=(WIDTH // 2, HEIGHT // 2 - 12)))
        line = f"Survived {game.elapsed:.1f}s.  R restart, M menu."
        small = font_small.render(line, True, TEXT_DIM)
        screen.blit(small, small.get_rect(
            center=(WIDTH // 2, HEIGHT // 2 + 22)))

    # Reuse pong's shared MENU button so the user always has a way out.
    btn_rect = _menu_button_rect()
    btn_hover = btn_rect.collidepoint(mouse_pos)
    draw_menu_button(screen, font_small, btn_hover)

    draw_scanlines(screen)


# ---------------------------------------------------------------------------
# Side-picker screen
# ---------------------------------------------------------------------------


def draw_side_picker(screen, fonts, hovered):
    """Mini-screen shown after the user picks BATTLE: lets them pick
    which side they want to play. Returns nothing; main.py drives the
    pygame event loop and hands selections back to us."""
    font_big, font_med, font_small = fonts
    screen.fill(BG)

    title = font_big.render("CHOOSE  YOUR  SIDE", True, ACCENT)
    screen.blit(title, title.get_rect(center=(WIDTH // 2, 110)))

    sub = font_small.render(
        "Click a paddle (or press T / B / L / R).  Esc cancels.",
        True, TEXT_DIM)
    screen.blit(sub, sub.get_rect(center=(WIDTH // 2, 150)))

    # Mock-up of the field with the four candidate paddles.
    pygame.draw.rect(screen, FIELD_BG, FIELD)
    pygame.draw.rect(screen, FIELD_LINE, FIELD, 2)

    for side in SIDES:
        x, y = _paddle_initial_pos(side)
        w, h = _paddle_size(side)
        rect = pygame.Rect(int(x), int(y), w, h)
        color, glow = SIDE_COLORS[side]
        is_hover = (side == hovered)
        if is_hover:
            inflate = 12
            glow_rect = rect.inflate(inflate, inflate)
            glow_surf = pygame.Surface(glow_rect.size, pygame.SRCALPHA)
            pygame.draw.rect(
                glow_surf, (*glow, 120),
                glow_surf.get_rect(), border_radius=8)
            screen.blit(glow_surf, glow_rect.topleft,
                        special_flags=pygame.BLEND_ADD)
        pygame.draw.rect(screen, color, rect, border_radius=4)

        # Label.
        label = SIDE_LABELS[side]
        surf = font_small.render(label, True, TEXT if is_hover else TEXT_DIM)
        if side == "top":
            screen.blit(surf, surf.get_rect(midtop=(rect.centerx, rect.bottom + 6)))
        elif side == "bottom":
            screen.blit(surf, surf.get_rect(midbottom=(rect.centerx, rect.top - 6)))
        elif side == "left":
            screen.blit(surf, surf.get_rect(midleft=(rect.right + 6, rect.centery)))
        else:
            screen.blit(surf, surf.get_rect(midright=(rect.left - 6, rect.centery)))

    draw_scanlines(screen)


def hovered_side(mouse_pos):
    """Return which side's mock paddle the mouse is over, or None."""
    mx, my = mouse_pos
    for side in SIDES:
        x, y = _paddle_initial_pos(side)
        w, h = _paddle_size(side)
        # Inflate the click target a bit so it's easier to hit.
        rect = pygame.Rect(int(x) - 8, int(y) - 8, w + 16, h + 16)
        if rect.collidepoint(mx, my):
            return side
    return None




