import asyncio
from playwright.async_api import async_playwright, expect

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()

        try:
            await page.goto("http://127.0.0.1:5000")

            # Fill in the search form
            await page.fill("#first-name", "Albert")
            await page.fill("#last-name", "Einstein")

            # Click the search button
            await page.click("#search-btn")

            # Wait for the results section to be visible
            results_section = page.locator("#results-section")
            await expect(results_section).to_be_visible(timeout=60000)

            # Wait for the summary section to be populated
            summary_total_pubs = page.locator("#summary-total-pubs")
            await expect(summary_total_pubs).not_to_have_text("0", timeout=10000)

            # Take a screenshot of the summary section
            summary_section = page.locator("#results-summary-section")
            await summary_section.screenshot(path="jules-scratch/verification/verification.png")

        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(main())
