
import asyncio
from playwright.async_api import async_playwright
async def main():
    async with async_playwright() as p:
        browser=await p.chromium.launch(headless=True, executable_path="/usr/bin/chromium", args=["--no-sandbox"])
        page=await browser.new_page(viewport={"width":1440,"height":950})
        await page.goto("http://127.0.0.1:5055",wait_until="networkidle")
        await page.screenshot(path="/mnt/data/AI_Resume_Assignment5/docs/figures/ui_dashboard.png",full_page=True)
        await browser.close()
asyncio.run(main())
