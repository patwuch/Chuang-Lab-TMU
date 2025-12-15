import pandas as pd
import time
import json
import csv
from playwright.sync_api import sync_playwright

# ==============================
# Load cookies
# ==============================
try:
    with open("/home/patwuch/projects/promedscrape/cookies.json", "r") as f:
        cookies = json.load(f)
except FileNotFoundError:
    print("Error: cookies.json not found. Please ensure the file exists and is accessible.")
    exit()

# Normalize SameSite
for cookie in cookies:
    s = cookie.get("sameSite")
    if isinstance(s, str):
        s = s.capitalize()
        if s not in ["Strict", "Lax", "None"]:
            s = "Lax"
        cookie["sameSite"] = s
    else:
        # if missing or None, set default
        cookie["sameSite"] = "Lax"

SEARCH_URL = "https://www.promedmail.org/search"
MAX_LOAD_ATTEMPTS = 5
LOAD_MORE_SELECTOR = "button:has-text('Load more')"

OUTPUT_FILE = "promed_results.csv"
HEADERS = ["id", "date", "title", "region", "disease", "host", "location", "source"]

# ==============================
# Initialize CSV with headers
# ==============================
with open(OUTPUT_FILE, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=HEADERS)
    writer.writeheader()

# ==============================
# Main scraping logic
# ==============================
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    context = browser.new_context()
    context.add_cookies(cookies)
    page = context.new_page()
    page.goto(SEARCH_URL)
    # ===================================
    print("🔍 Dumping all <button> elements for debugging...")
    buttons = page.query_selector_all("button")
    for i, btn in enumerate(buttons[:30]):  # limit output
        text = btn.inner_text().strip()
        cls = btn.get_attribute("class")
        print(f"{i:02d}: text='{text}' | class='{cls}'")
    print(f"Total buttons found: {len(buttons)}")

        # Step 1: Handle initial modal or consent
    try:
        print("🧩 Checking for modal...")
        continue_btn = page.locator("button", has_text="Continue")
        if continue_btn.is_visible():
            print("👉 Clicking 'Continue' button to proceed...")
            continue_btn.click()
            page.wait_for_timeout(1500)
        else:
            print("✅ No modal detected.")
    except Exception as e:
        print(f"⚠️ Could not close modal: {e}")

    # 🕵️ Inspect the environment after clicking "Continue"
    print("🌐 Current URL:", page.url)
    print("📄 Page title:", page.title())
    print("🧩 Number of iframes:", len(page.frames))
    for i, frame in enumerate(page.frames):
        print(f"Frame {i}: URL={frame.url}")

    # Optional: dump outerHTML snippet for debugging
    html_snippet = page.content()[:2000]
    print("\n=== HTML snippet ===\n", html_snippet)

    # Step 2: Wait for main content to load
    page.wait_for_load_state("networkidle")

    # Step 3: Dump buttons again to confirm sidebar toggle appears
    print("🔍 Dumping buttons again after modal closed...")
    buttons = page.query_selector_all("button")
    for i, btn in enumerate(buttons[:30]):
        text = btn.inner_text().strip()
        cls = btn.get_attribute("class")
        print(f"{i:02d}: text='{text}' | class='{cls}'")
    print(f"Total buttons found: {len(buttons)}")


    # ===================================
    # Open sidebar and select disease filters (fixed version)
    # ===================================
    try:
        print("⏳ Waiting for page to fully load...")
        page.wait_for_load_state("networkidle", timeout=30000)

        # STEP 1: Try to find and click the sidebar toggle
        print("➡️ Searching for sidebar toggle button...")
        toggle_locators = [
            'button[data-sentry-source-file="sidebar-toggle.tsx"]',
            'button:has(svg.lucide-chevron-left)',   # fallback: icon-based button
            'button:has(svg[class*="chevron"])',     # broader fallback
        ]

        sidebar_toggle = None
        for selector in toggle_locators:
            loc = page.locator(selector)
            if loc.count() > 0:
                sidebar_toggle = loc.first
                break

        if sidebar_toggle:
            sidebar_toggle.scroll_into_view_if_needed()
            sidebar_toggle.wait_for(state="visible", timeout=15000)
            print("✅ Sidebar toggle located, clicking...")
            sidebar_toggle.click()
            page.wait_for_timeout(2000)
        else:
            raise Exception("Sidebar toggle button not found by any selector.")

        # STEP 2: Expand the "Diseases" accordion
        print("⏳ Waiting for 'Diseases' accordion trigger...")
        disease_trigger = page.locator('button:has-text("Diseases")')
        disease_trigger.wait_for(state="visible", timeout=15000)
        expanded = disease_trigger.get_attribute("aria-expanded")
        if expanded == "false":
            print("➡️ Expanding 'Diseases' accordion...")
            disease_trigger.click()
            page.wait_for_function(
                """() => {
                    const btn = [...document.querySelectorAll('button')].find(b => b.textContent.includes('Diseases'));
                    return btn && btn.getAttribute('aria-expanded') === 'true';
                }""",
                timeout=10000
            )
        print("✅ 'Diseases' accordion ready.")

        # STEP 3: Scroll sidebar container to reveal checkboxes
        sidebar_container = page.locator('div.relative.h-full.flex.flex-col.overflow-y-auto')
        sidebar_container.evaluate("el => el.scrollTop = el.scrollHeight")
        page.wait_for_timeout(500)

        # STEP 4: Click "Show more" if present
        show_more = page.locator('button:has-text("Show more")')
        if show_more.count() > 0 and show_more.first.is_visible():
            print("➡️ Clicking 'Show more' to reveal all diseases...")
            show_more.first.evaluate("btn => btn.click()")
            page.wait_for_timeout(500)

        # ==============================
        # STEP 5: Click target disease checkboxes reliably
        # ==============================
        target_diseases = ["Chikungunya", "Zika virus", "Dengue"]

        for disease in target_diseases:
            # Locate by ID first (works even with spaces)
            checkbox = page.locator(f'button[role="checkbox"][id="{disease}"]')

            # Fallback: locate by text if ID fails
            if checkbox.count() == 0:
                checkbox = page.locator(f'button[role="checkbox"]:has-text("{disease}")')

            if checkbox.count() > 0:
                checkbox = checkbox.first
                # Scroll into view in the scrollable sidebar
                checkbox.scroll_into_view_if_needed()
                aria_checked = checkbox.get_attribute("aria-checked")
                if aria_checked != "true":
                    print(f"🩺 Clicking '{disease}' checkbox...")
                    # Click via JS to avoid pointer interception
                    checkbox.evaluate("btn => btn.click()")
                    page.wait_for_timeout(400)
                else:
                    print(f"✅ '{disease}' already checked.")
            else:
                print(f"⚠️ Checkbox not found: {disease}")


        print("🎯 Sidebar filters applied successfully.\n")
        page.wait_for_timeout(1000)

    except Exception as e:
        print(f"❌ Error during sidebar filter setup: {e}")





    seen_ids = set()
    load_attempts = 0
    total_saved = 0

    while True:
        # ==============================
        # Scrape visible rows (new ones only)
        # ==============================
        rows = page.query_selector_all("tr.border-b")
        batch = []

        for row in rows:
            cells = row.query_selector_all("td")
            if len(cells) != 8:
                continue  # skip header/footer rows

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

        # ==============================
        # Write batch to CSV immediately
        # ==============================
        if batch:
            with open(OUTPUT_FILE, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=HEADERS)
                writer.writerows(batch)
            total_saved += len(batch)
            print(f"💾 Saved {len(batch)} new rows (Total: {total_saved})")

        # ==============================
        # Try clicking "Load more"
        # ==============================
        try:
            load_more = page.wait_for_selector(LOAD_MORE_SELECTOR, state="visible", timeout=5000)
            print("Clicking 'Load more'...")
            load_more.click()

            # Wait for more rows to appear
            page.wait_for_function(
                f"() => document.querySelectorAll('tr.border-b').length > {len(seen_ids)}",
                timeout=15000
            )

            load_attempts = 0
            time.sleep(0.5)  # let UI settle

        except Exception as e:
            if page.query_selector(LOAD_MORE_SELECTOR) is None:
                print("No more 'Load more' button. Finished scraping.")
                break

            load_attempts += 1
            print(f"Click attempt failed ({load_attempts}/{MAX_LOAD_ATTEMPTS}). Error: {e}")
            if load_attempts >= MAX_LOAD_ATTEMPTS:
                print("Max load attempts reached. Stopping.")
                break
            time.sleep(3)

    print(f"✅ Finished scraping. Total rows saved: {total_saved}")
    browser.close()
