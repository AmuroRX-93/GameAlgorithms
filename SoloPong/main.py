"""Entry point used by pygbag to package the game for the browser.

pygbag scans this file statically for `async def main` and a top-level
`asyncio.run(main())` call -- if it doesn't see them in main.py itself,
the WebAssembly runtime starts but the game loop is never executed and
the canvas stays at 1x1 px (the symptom: solid gray page in the browser).

To keep that scan happy without duplicating the game code, we re-export
pong's main coroutine under a private name and wrap it.
"""
import asyncio

from pong import main as _pong_main


async def main():
    await _pong_main()


asyncio.run(main())
