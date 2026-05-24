# Solo Pong

Single-player Pong in the spirit of Atari's 1972 cabinet, but with the
opponent replaced by the top wall. The ball bounces off three walls
forever; only the bottom is lethal. Plus a layer of breakable bricks
in the upper third, multi-ball, and a CRT scanline + bloom finish so
it looks like an arcade monitor.

## Run

```bash
cd SoloPong
pip install -r requirements.txt
python pong.py
```

## Controls

- **Mouse** — paddle follows the cursor (precise)
- **Left / Right** or **A / D** — keyboard control (paddle slides at a
  fixed speed)
- **Space** — launch the ball at the start of a life; toggles pause
  during play
- **R** — restart after game over
- **Esc** — quit

## What's faithful to the original

- 5-zone paddle. The paddle is split into 5 horizontal segments and
  each one returns the ball at a different angle. Edge zones bounce
  the ball back at ~60° from vertical, the center zone at ~10°. This
  is the trick that gives the player real angular control — exactly
  what Atari did in 1972.
- Speed-up on every paddle hit (4 % per hit, capped). Rallies get
  faster the longer they last.
- Tight, deterministic physics: ball is a circle, paddle and bricks
  are AABBs, collision uses the standard closest-point-on-rect test
  so the ball deflects correctly off corners.

## What's added

- **Bricks**. A grid of 5 × 12 bricks sits in the upper third. Higher
  rows score more points. Smashing them all spawns a fresh wall
  (endless mode).
- **Multi-ball**. Every 100 points spawns a new ball mid-flight (up to
  5 simultaneously). The new ball forks off an existing one with a
  slight angle change.
- **Sub-stepping**. Fast balls advance in up to 4 sub-steps per frame
  so a hot ball can never tunnel through a brick or the paddle.
- **CRT post**. Soft additive bloom (downscale + upscale) plus 1-px
  scanlines for that arcade-monitor look.
- **Lives**. You start with 3. Lose all your active balls, lose a life;
  out of lives, game over.
- **Screen shake** on brick destruction, paddle flash on hit.

## Tuning

All knobs live as constants near the top of `pong.py`:
`PADDLE_W`, `PADDLE_SPEED`, `BALL_SPEED_START`, `BALL_SPEED_MAX`,
`BALL_SPEEDUP`, `BRICK_ROWS`, `BRICK_COLS`, `EXTRA_BALL_EVERY`,
`MAX_BALLS`. The 5-zone reflection lives in `reflect_off_paddle`.
