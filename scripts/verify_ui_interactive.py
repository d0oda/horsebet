import os
import sys
from playwright.sync_api import sync_playwright

def run_verification():
    artifact_dir = "/Users/ryfei.wang/.gemini/antigravity-ide/brain/d4e25817-b1b3-4a4c-b94b-d105106518a5"
    os.makedirs(artifact_dir, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1600, "height": 1100})
        page = context.new_page()

        print("1. Navigating to http://localhost:8000/...")
        page.goto("http://localhost:8000/", wait_until="networkidle")

        # Capture initial daily view
        page.wait_for_selector(".daily-returns-card", timeout=15000)
        daily_shot = os.path.join(artifact_dir, "01_daily_view.png")
        page.screenshot(path=daily_shot, full_page=False)
        print(f"   Captured initial view -> {daily_shot}")

        # 2. Click '📈 All-Time' button in quick nav
        print("2. Clicking '📈 All-Time' button in sidebar...")
        page.click("#btn-quick-alltime")
        page.wait_for_selector(".all-time-returns-card", timeout=15000)
        
        all_time_card = page.locator(".all-time-returns-card")
        header_text = all_time_card.locator(".returns-title").inner_text()
        print(f"   All-Time Card Title: {header_text}")
        assert "All-Time Performance" in header_text
        assert "January 2026" in header_text

        # Verify 4 Strategy cards
        strat_items = all_time_card.locator(".strategy-return-item")
        strat_count = strat_items.count()
        print(f"   Found {strat_count} strategy cards")
        assert strat_count == 4

        # Verify Milestones
        milestone_items = all_time_card.locator(".milestone-card")
        m_count = milestone_items.count()
        print(f"   Found {m_count} milestone cards")
        assert m_count == 4

        # Verify Monthly breakdown table
        table = all_time_card.locator(".monthly-timeline-table")
        rows = table.locator("tbody tr")
        row_count = rows.count()
        print(f"   Found {row_count} monthly breakdown rows")
        assert row_count == 8

        alltime_shot = os.path.join(artifact_dir, "02_all_time_view.png")
        page.screenshot(path=alltime_shot, full_page=False)
        print(f"   Captured All-Time view -> {alltime_shot}")

        # 3. Drill down into March 2026
        print("3. Clicking 'View Month →' on March 2026 row...")
        march_row = table.locator("tr").filter(has_text="March 2026")
        march_row.locator(".timeline-view-btn").click()
        page.wait_for_selector(".monthly-returns-card", timeout=15000)

        monthly_card = page.locator(".monthly-returns-card")
        m_title = monthly_card.locator(".returns-title").inner_text()
        print(f"   Monthly Card Title: {m_title}")
        assert "March 2026" in m_title

        march_shot = os.path.join(artifact_dir, "03_march_2026_monthly_view.png")
        page.screenshot(path=march_shot, full_page=False)
        print(f"   Captured Monthly March view -> {march_shot}")

        # 4. Click '🌐 All-Time' toggle to switch back
        print("4. Clicking '🌐 All-Time' view toggle in monthly card...")
        monthly_card.locator(".view-toggle-btn").filter(has_text="All-Time").click()
        page.wait_for_selector(".all-time-returns-card", timeout=15000)
        print("   Successfully returned to All-Time view!")

        # 5. Click '📅 Daily' view toggle to switch to daily view
        print("5. Clicking '📅 Daily' view toggle in all-time card...")
        page.locator(".view-toggle-btn").filter(has_text="Daily").click()
        page.wait_for_selector(".daily-returns-card:not(.monthly-returns-card):not(.all-time-returns-card)", timeout=15000)
        print("   Successfully returned to Daily view!")

        browser.close()
        print("\n✨ ALL UI AND WORKFLOW ASSERTIONS PASSED PERFECTLY!")

if __name__ == "__main__":
    run_verification()
