import asyncio
from pathlib import Path
from playwright.async_api import async_playwright
from app.config import settings

SESSION_DIR = Path(__file__).resolve().parent / "data" / "browser_session"

async def main():
    print("=" * 70)
    print("HERMES BROWSER AGENT - 1-TIME SESSION SETUP")
    print("=" * 70)
    print(f"Session data will be saved permanently in: {SESSION_DIR}\n")
    print("A visible Chrome window is opening now:")
    print("1. Log in to your ChatGPT account (https://chatgpt.com)")
    print("2. Log in to your Google Gemini account (https://gemini.google.com)\n")
    print("When you are logged in, press ENTER in this terminal to save and exit.")
    print("=" * 70)

    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    
    async with async_playwright() as p:
        context = await p.chromium.launch_persistent_context(
            user_data_dir=str(SESSION_DIR),
            headless=False,
            viewport={"width": 1280, "height": 800},
            args=["--disable-blink-features=AutomationControlled", "--start-maximized"]
        )
        
        # Open Tab 1: ChatGPT
        page1 = await context.new_page()
        await page1.goto("https://chatgpt.com")

        # Open Tab 2: Gemini
        page2 = await context.new_page()
        await page2.goto("https://gemini.google.com/app")

        # Wait for user confirmation in terminal
        await asyncio.to_thread(input, "\n👉 Once logged into ChatGPT & Google, press [ENTER] here to save session: ")
        
        await context.close()
        print("\n✅ Session successfully saved! Hermes can now run browser audits automatically.")

if __name__ == "__main__":
    asyncio.run(main())
