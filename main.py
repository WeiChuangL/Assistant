import asyncio
import sys

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


async def run_cli_mode():
    from src.vector_store.db import init_db
    from src.cli.app import run_cli

    print("Initializing database...")
    await init_db()
    print("Database ready.\n")
    await run_cli()


async def _run_web():
    import uvicorn
    config = uvicorn.Config("src.api.server:app", host="0.0.0.0", port=8000, reload=False)
    server = uvicorn.Server(config)
    await server.serve()

def run_web_mode():
    if sys.platform == "win32":
        import selectors
        loop_factory = lambda: asyncio.SelectorEventLoop(selectors.SelectSelector())
        asyncio.run(_run_web(), loop_factory=loop_factory)
    else:
        asyncio.run(_run_web())


if __name__ == "__main__":
    if "--web" in sys.argv:
        print("Starting web server at http://localhost:8000")
        run_web_mode()
    else:
        asyncio.run(run_cli_mode())
