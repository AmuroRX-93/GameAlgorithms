# Pong

Two flavors in one binary, picked from the start menu.

## Modes

### 1. Solo (Atari + Breakout)
Single-player Pong in the spirit of Atari's 1972 cabinet — the
opponent is replaced by the top wall, only the bottom is lethal.
Plus a layer of breakable bricks in the upper third, multi-ball,
and CRT scanlines + bloom for that arcade-monitor look.

### 2. vs AI (survival)
Classic Pong layout. The AI sits at the top and **never misses** —
it predicts the ball's landing point including wall reflections,
slides to meet it, and adds a small return offset so the rally
keeps shifting. Your goal is to **stay alive as long as possible**:
score grows with every second you survive, every ball you return,
and every brick you smash in the middle band. One miss = game over.

## Run

```bash
cd SoloPong
pip install -r requirements.txt
python pong.py
```

## Controls

- **1 / 2** — pick a mode at the menu
- **Mouse** — paddle follows the cursor
- **Left / Right** or **A / D** — keyboard control
- **Space** — launch ball / pause / resume
- **R** — restart after game over
- **M** — back to mode menu
- **Esc** — quit

## What's faithful to the original

- **5-zone paddle.** The paddle is split into 5 horizontal segments;
  each returns the ball at a different angle. Edge zones bounce
  ~60° off vertical, center zone ~10°. This is the trick that gives
  real angular control — exactly what Atari did in 1972.
- **Speed-up on every paddle hit** (4 % per hit, capped). Rallies get
  faster the longer they last.
- **Tight collision.** Ball is a circle, paddles and bricks are
  AABBs, collision uses closest-point-on-rect so corners deflect
  correctly.

## What's added

- **Two modes**, picked from the menu (`1` solo, `2` vs AI).
- **Perfect AI opponent.** In vs-AI mode the top paddle simulates an
  unbeatable ghost: it predicts where the ball will arrive at its
  y-plane (mirroring x against the side walls so reflections are
  baked in), then glides toward that target at a high cap. To keep
  the game interesting, returns are biased away from straight-back
  by `AI_RETURN_JITTER`, so the rally drifts every bounce.
- **Survival scoring.** vs-AI mode awards 5 points/sec just for
  staying alive, plus 10 per AI return and 5 per player return.
  Bricks pay normally.
- **Bricks.** Solo mode has a full 5×12 wall up top; vs-AI mode has
  a 3×12 band in the middle that both players can smash through.
- **Multi-ball** (solo only). Every 100 points spawns a new ball
  mid-flight, up to 5 active.
- **Sub-stepping.** Fast balls advance in up to 4 sub-steps per
  frame so a hot ball can never tunnel through a brick or paddle.
- **CRT post.** Soft additive bloom (downscale + upscale) and 1-px
  scanlines.
- **Lives** (solo only). 3 to start. vs-AI is one strike out.
- **Screen shake** on brick destruction, paddle flash on hit.

## Tuning

All knobs live as constants near the top of `pong.py`:
`PADDLE_W`, `PADDLE_SPEED`, `AI_PADDLE_SPEED`, `AI_RETURN_JITTER`,
`AI_SURVIVAL_PER_SEC`, `AI_RETURN_BONUS`, `BALL_SPEED_START`,
`BALL_SPEED_MAX`, `BALL_SPEEDUP`, `BRICK_ROWS`, `BRICK_COLS`,
`AI_BRICK_ROWS`, `EXTRA_BALL_EVERY`, `MAX_BALLS`. The 5-zone
reflection lives in `reflect_off_paddle`; the AI's prediction lives
in `AIPaddle.think`.
