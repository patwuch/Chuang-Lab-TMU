import json
import csv
import time
from playwright.sync_api import sync_playwright, TimeoutError

SEARCH_URL = "https://promedmail.org/search"
COOKIES_FILE = "/home/patwuch/projects/promedscrape/cookies.json"
OUTPUT_FILE = "promed_filtered.csv"
HEADERS = ["id", "date", "title", "region", "disease", "host", "location", "source"]
LOAD_MORE_SELECTOR = "button:has-text('Load more')"
MAX_LOAD_ATTEMPTS = 5
DISEASES = ["Chikungunya", "Zika virus", "Dengue"]

def normalize_cookies(cookies):
    """Ensure cookies are compatible with Playwright."""
    for cookie in cookies:
        s = cookie.get("sameSite")
        if isinstance(s, str):
            s = s.capitalize()
            if s not in ["Strict", "Lax", "None"]:
                s = "Lax"
            cookie["sameSite"] = s
        else:
            cookie["sameSite"] = "Lax"
    return cookies

with sync_playwright() as p:
    # Headless browser for server/CLI
    browser = p.chromium.launch(headless=True)
    context = browser.new_context()

    # Load cookies
    with open(COOKIES_FILE, "r") as f:
        cookies = json.load(f)
    cookies = normalize_cookies(cookies)
    context.add_cookies(cookies)

    page = context.new_page()
    page.goto(SEARCH_URL)

    # ==============================
    # Sidebar: scroll, show more, checkboxes
    # ==============================
    try:
        # Wait for sidebar container
        sidebar_selector = "div.relative.h-full.flex.flex-col.overflow-y-auto"
        page.wait_for_selector(sidebar_selector, timeout=10000)

        # Scroll sidebar to bottom
        page.eval_on_selector(sidebar_selector, "(el) => el.scrollTop = el.scrollHeight")
        time.sleep(0.5)

        # Click "Show more" if exists
        show_more = page.locator('button:has-text("Show more")')
        if show_more.count() > 0:
            show_more.scroll_into_view_if_needed()
            show_more.click()
            time.sleep(0.5)

        # Click disease checkboxes
        for disease in DISEASES:
            checkbox = page.locator(f'button[id="{disease}"]')
            checkbox.scroll_into_view_if_needed()
            checkbox.click()
            time.sleep(0.2)
            state = checkbox.get_attribute("data-state")
            print(f"{disease} checked:", state)

        print("✅ Sidebar filters applied successfully.")
    except TimeoutError:
        print("❌ Error during sidebar filter setup. Sidebar may not have loaded.")

    # ==============================
    # Scraping loop
    # ==============================
    seen_ids = set()
    load_attempts = 0
    total_saved = 0

    while True:
        rows = page.query_selector_all("tr.border-b")
        batch = []

        for row in rows:
            cells = row.query_selector_all("td")
            if len(cells) != 8:
                continue
            cell_texts = [cell.inner_text().strip() for cell in cells]
            entry_id = cell_texts[0]
            if entry_id not in seen_ids:
                seen_ids.add(entry_id)
                batch.append({
                    "id": entry_id,
                    "date": cell_texts[1],
                    "title": cell_texts[2],
                    "region": cell_texts[3],
                    "disease": cell_texts[4],
                    "host": cell_texts[5],
                    "location": cell_texts[6],
                    "source": cell_texts[7],
                })

        if batch:
            with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=HEADERS)
                if total_saved == 0:
                    writer.writeheader()
                writer.writerows(batch)
            total_saved += len(batch)
            print(f"💾 Saved {len(batch)} new rows (Total: {total_saved})")

        # Click "Load more" if available
        try:
            load_more = page.locator(LOAD_MORE_SELECTOR).first
            if load_more.is_visible():
                print("Clicking 'Load more'...")
                load_more.scroll_into_view_if_needed()
                load_more.click()
                page.wait_for_function(
                    f"() => document.querySelectorAll('tr.border-b').length > {len(seen_ids)}",
                    timeout=15000
                )
                load_attempts = 0
                time.sleep(0.5)
            else:
                print("No more 'Load more' button. Finished scraping.")
                break
        except TimeoutError:
            load_attempts += 1
            print(f"Load attempt failed ({load_attempts}/{MAX_LOAD_ATTEMPTS})")
            if load_attempts >= MAX_LOAD_ATTEMPTS:
                print("Max load attempts reached. Stopping.")
                break
            time.sleep(3)

    print(f"✅ Finished scraping. Total rows saved: {total_saved}")
    browser.close()
