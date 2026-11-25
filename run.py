import asyncio
import sys
import uvicorn

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

    print("=" * 60)
    print("Starting ShapeBytesAEO Intelligence Engine on http://localhost:8000")
    print("=" * 60)
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False)
