"""Entry point used by pygbag to package the game for the browser.

pygbag packages whatever directory contains a `main.py` and starts it
with `asyncio.run(main())`. We just delegate to pong.main so the
desktop entry point (`python pong.py`) and the browser entry point
share exactly the same code path.
"""
import asyncio

from pong import main


if __name__ == "__main__":
    asyncio.run(main())
