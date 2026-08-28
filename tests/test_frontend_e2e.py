"""
UmaEdge — Comprehensive Frontend End-to-End Test Suite.

Tests:
1. Header & Initial Page Load
2. Calendar Navigation & State
3. Race List & Summary Grid
4. Filter Toolbar (All, Value Bets, Venues)
5. Race Card Accordion & Lazy-Loading
6. Executive Staking Portfolio & Dynamic Budget Switcher
7. Strategy Mode Router (AI Auto, Balanced, Pure Win, Dual Dutching)
8. Fair Odds Pricing Table & Analytical Verdicts
9. Detailed Stats Drawer (L3F, Prob Bars, Finish Badges)
10. Single Race Prediction Refresh
11. Batch Predict All Races Button & Flow
12. Empty / Unscraped Date Handling & Fetch Prompt
13. Toast Notification System & Dismissal
14. Mobile Responsive Layout & Viewport (375x667)
15. Zero Uncaught Console Errors
"""

import re
import time
import pytest
from playwright.sync_api import sync_playwright, Page, expect

BASE_URL = "http://localhost:3000"
API_URL = "http://localhost:8000"


@pytest.fixture(scope="module")
def browser_context():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        yield context
        browser.close()


@pytest.fixture
def page(browser_context):
    page = browser_context.new_page()
    # Capture console messages
    page.console_logs = []
    page.on("console", lambda msg: page.console_logs.append(f"[{msg.type}] {msg.text}"))
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle")
    yield page
    page.close()


def test_header_and_initial_load(page: Page):
    """Test 1: Header renders correctly with logo, status badge, and model version."""
    logo = page.locator(".logo")
    expect(logo).to_have_text("UmaEdge")

    status_badge = page.locator("#header-status")
    expect(status_badge).to_be_visible()
    time.sleep(0.5)
    text = status_badge.text_content()
    assert "Live" in text or "Offline" in text or "Loading" in text


def test_calendar_rendering_and_navigation(page: Page):
    """Test 2: Calendar widget month navigation and day clicking."""
    month_label = page.locator("#cal-month-label")
    expect(month_label).to_be_visible()
    initial_month = month_label.text_content()

    # Click prev month
    page.locator("#cal-prev").click()
    time.sleep(0.2)
    prev_month = month_label.text_content()
    assert prev_month != initial_month

    # Click next month
    page.locator("#cal-next").click()
    time.sleep(0.2)
    expect(month_label).to_have_text(initial_month)

    # Click a day from recent manifest list if present
    manifest_rows = page.locator(".manifest-row")
    if manifest_rows.count() > 0:
        first_manifest_row = manifest_rows.first
        first_manifest_row.click()
        page.wait_for_selector(".summary-grid", timeout=5000)
        
        # Verify header date updated
        header_date = page.locator("#header-date")
        expect(header_date).not_to_have_text("Select a date")


def test_race_list_and_summary_grid(page: Page):
    """Test 3: Select 2026-08-23 and verify summary stats and race cards."""
    page.evaluate("selectDate('2026-08-23')")
    page.wait_for_selector(".summary-grid", timeout=5000)

    # Check stat cards
    stat_cards = page.locator(".stat-card")
    expect(stat_cards.first).to_be_visible()
    races_stat = stat_cards.filter(has_text="RACES").locator(".stat-value")
    expect(races_stat).to_have_text("36")

    # Verify 36 race cards are rendered
    race_cards = page.locator(".race-card")
    expect(race_cards).to_have_count(36)


def test_filter_toolbar(page: Page):
    """Test 4: Interactive toolbar filtering (All, Value Bets, Venues)."""
    page.evaluate("selectDate('2026-08-23')")
    page.wait_for_selector(".raceday-toolbar", timeout=5000)

    # 1. Click Value Bets filter
    bets_chip = page.locator("#chip-filter-bets")
    bets_chip.click()
    time.sleep(0.3)
    expect(bets_chip).to_have_class(re.compile(r"\bactive\b"))

    filtered_cards = page.locator(".race-card")
    bets_count_str = bets_chip.locator(".chip-count").text_content()
    expected_bets_count = int(bets_count_str) if bets_count_str.isdigit() else 0

    if expected_bets_count > 0:
        expect(filtered_cards).to_have_count(expected_bets_count)
        for i in range(filtered_cards.count()):
            expect(filtered_cards.nth(i)).to_have_class(re.compile(r"\bhas-bet\b"))
    else:
        expect(page.locator(".empty-filter-state")).to_be_visible()

    # 2. Click a Venue filter if present
    venue_chips = page.locator(".filter-chip[data-filter]")
    if venue_chips.count() > 0:
        first_venue = venue_chips.first
        first_venue.click()
        time.sleep(0.3)
        expect(first_venue).to_have_class(re.compile(r"\bactive\b"))
        expect(page.locator(".race-card")).to_have_count(12)  # JRA holds 12 races per venue

    # 3. Restore All Races
    page.locator("#chip-filter-all").click()
    time.sleep(0.3)
    expect(page.locator(".race-card")).to_have_count(36)


def test_race_card_accordion_and_betting_analysis(page: Page):
    """Test 5: Expand race card and test betting analysis, budget switcher, and strategy modes."""
    page.evaluate("selectDate('2026-08-23')")
    page.wait_for_selector(".race-card", timeout=5000)

    # Open the first race card
    first_card = page.locator(".race-card").first
    header = first_card.locator(".race-header")
    header.click()
    time.sleep(0.8)

    expect(first_card).to_have_class(re.compile(r"\bopen\b"))

    # Verify Staking Card is loaded
    staking_card = first_card.locator(".staking-card")
    expect(staking_card).to_be_visible()

    # Test Budget switcher
    budget_chips = staking_card.locator(".budget-chip")
    expect(budget_chips).to_have_count(5)  # 500, 1000, 2500, 5000, 10000

    # Click ¥5,000 chip
    chip_5k = budget_chips.filter(has_text="¥5,000")
    chip_5k.click()
    time.sleep(0.5)
    expect(chip_5k).to_have_class(re.compile(r"\bactive\b"))

    # Test Strategy Mode Buttons
    strategy_btns = staking_card.locator(".strategy-mode-btn")
    expect(strategy_btns).to_have_count(4)

    # Click Pure Win mode
    btn_pure_win = strategy_btns.filter(has_text="Pure Win")
    btn_pure_win.click()
    time.sleep(0.5)
    expect(btn_pure_win).to_have_class(re.compile(r"\bactive\b"))

    # Click Dual Dutching mode
    btn_dutching = strategy_btns.filter(has_text="Dual Dutching")
    btn_dutching.click()
    time.sleep(0.5)
    expect(btn_dutching).to_have_class(re.compile(r"\bactive\b"))

    # Verify Pricing Table
    pricing_card = first_card.locator(".pricing-card")
    expect(pricing_card).to_be_visible()
    pricing_rows = pricing_card.locator("tbody tr")
    assert pricing_rows.count() > 0


def test_detailed_stats_drawer(page: Page):
    """Test 6: Toggle collapsible full entries drawer and verify metrics."""
    page.evaluate("selectDate('2026-08-23')")
    page.wait_for_selector(".race-card", timeout=5000)

    first_card = page.locator(".race-card").first
    if not "open" in (first_card.get_attribute("class") or ""):
        first_card.locator(".race-header").click()
        time.sleep(0.8)

    drawer_btn = first_card.locator(".stats-drawer-btn")
    expect(drawer_btn).to_be_visible()

    # Open drawer
    drawer_btn.click()
    time.sleep(0.3)
    drawer_content = first_card.locator(".stats-drawer-content")
    expect(drawer_content).to_have_class(re.compile(r"\bopen\b"))

    # Check entries table rows
    entry_rows = drawer_content.locator("tbody tr")
    assert entry_rows.count() >= 5

    # Check probability bars and finish positions
    first_entry = entry_rows.first
    expect(first_entry.locator(".prob-bar-fill")).to_be_visible()
    expect(first_entry.locator(".pos-num")).to_be_visible()


def test_single_race_prediction_refresh(page: Page):
    """Test 7: Single race predict/repredict button interaction."""
    page.evaluate("selectDate('2026-08-23')")
    page.wait_for_selector(".race-card", timeout=5000)

    predict_btn = page.locator(".predict-btn").first
    expect(predict_btn).to_be_visible()

    # Click predict/repredict
    predict_btn.click()
    time.sleep(0.2)

    # Verify button shows loading or done
    assert "Predicting" in predict_btn.text_content() or "Done" in predict_btn.text_content() or "↺" in predict_btn.text_content()


def test_batch_predict_button_interaction(page: Page):
    """Test 8: Batch predict button rendering and click handler."""
    page.evaluate("selectDate('2026-08-23')")
    page.wait_for_selector("#btn-predict-all", timeout=5000)

    batch_btn = page.locator("#btn-predict-all")
    expect(batch_btn).to_be_visible()
    assert "Predict All" in batch_btn.text_content() or "Repredict All" in batch_btn.text_content()


def test_refetch_races_button_interaction(page: Page):
    """Test 8b: Refetch races button rendering in section actions."""
    page.evaluate("selectDate('2026-08-23')")
    page.wait_for_selector("#btn-refetch-races", timeout=5000)

    refetch_btn = page.locator("#btn-refetch-races")
    expect(refetch_btn).to_be_visible()
    assert "Refetch Races" in refetch_btn.text_content()


def test_provisional_refetch_button_interaction(page: Page):
    """Test 8c: Provisional entries banner has refetch button for provisional cards (<24 races)."""
    page.evaluate("selectDate('2026-02-08')")
    page.wait_for_selector("#btn-provisional-refetch", timeout=5000)

    provisional_btn = page.locator("#btn-provisional-refetch")
    expect(provisional_btn).to_be_visible()
    assert "Refetch Full Card" in provisional_btn.text_content()


def test_quick_navigation_shortcuts(page: Page):
    """Test 8d: Quick date navigation shortcut buttons (Upcoming, Today, Latest)."""
    expect(page.locator("#btn-quick-upcoming")).to_be_visible()
    expect(page.locator("#btn-quick-today")).to_be_visible()
    expect(page.locator("#btn-quick-latest")).to_be_visible()

    # Click Upcoming
    page.locator("#btn-quick-upcoming").click()
    time.sleep(0.5)
    header_date = page.locator("#header-date").text_content()
    assert header_date != "Select a date"


def test_sync_results_button(page: Page):
    """Test 8e: Past race days display the Sync Results button."""
    page.evaluate("selectDate('2026-08-23')")
    page.wait_for_selector("#btn-sync-results", timeout=5000)
    sync_btn = page.locator("#btn-sync-results")
    expect(sync_btn).to_be_visible()
    assert "Sync Results" in sync_btn.text_content()


def test_surface_filters(page: Page):
    """Test 8f: Turf and Dirt surface filters filter race cards properly."""
    page.evaluate("selectDate('2026-08-23')")
    page.wait_for_selector("#chip-filter-turf", timeout=5000)

    turf_chip = page.locator("#chip-filter-turf")
    dirt_chip = page.locator("#chip-filter-dirt")
    expect(turf_chip).to_be_visible()
    expect(dirt_chip).to_be_visible()

    # Click Turf filter
    turf_chip.click()
    time.sleep(0.3)
    expect(turf_chip).to_have_class(re.compile(r"\bactive\b"))
    cards = page.locator(".race-card")
    for i in range(cards.count()):
        expect(cards.nth(i).locator(".tag-turf")).to_be_visible()

    # Click Dirt filter
    dirt_chip.click()
    time.sleep(0.3)
    expect(dirt_chip).to_have_class(re.compile(r"\bactive\b"))
    for i in range(cards.count()):
        expect(cards.nth(i).locator(".tag-dirt")).to_be_visible()


def test_custom_budget_input(page: Page):
    """Test 8g: Custom budget numeric input dynamically updates tickets in staking card."""
    page.evaluate("selectDate('2026-08-23')")
    page.wait_for_selector(".race-card", timeout=5000)

    first_card = page.locator(".race-card").first
    if "open" not in (first_card.get_attribute("class") or ""):
        first_card.locator(".race-header").click()
        time.sleep(0.8)

    budget_input = first_card.locator(".custom-budget-input")
    expect(budget_input).to_be_visible()
    budget_input.fill("3000")
    budget_input.press("Enter")
    time.sleep(0.6)
    # Value should persist
    expect(budget_input).to_have_value("3000")


def test_single_race_rescrape_button(page: Page):
    """Test 8h: Single race rescrape button exists in race actions."""
    page.evaluate("selectDate('2026-08-23')")
    page.wait_for_selector(".race-card", timeout=5000)

    first_card = page.locator(".race-card").first
    if "open" not in (first_card.get_attribute("class") or ""):
        first_card.locator(".race-header").click()
        time.sleep(0.8)

    rescrape_btn = first_card.locator(".race-rescrape-btn")
    expect(rescrape_btn).to_be_visible()
    assert "Rescrape" in rescrape_btn.text_content()


def test_empty_date_handling(page: Page):
    """Test 9: Selecting an unscraped/empty date shows graceful prompt."""
    page.evaluate("selectDate('2028-01-01')")
    time.sleep(0.8)

    fetch_state = page.locator(".fetch-state")
    expect(fetch_state).to_be_visible()
    expect(page.locator("#btn-fetch-2028-01-01")).to_be_visible()


def test_toast_notification_system(page: Page):
    """Test 10: Toast notifications create, render, and auto-dismiss correctly."""
    page.evaluate("showToast('🔥 High-Value Bet Found on Race 11!', 'success', 2000)")
    
    toast = page.locator(".toast-item.toast-success")
    expect(toast).to_be_visible()
    expect(toast).to_contain_text("High-Value Bet Found on Race 11")

    # Close button test
    close_btn = toast.locator(".toast-close")
    close_btn.click()
    time.sleep(0.5)
    expect(toast).not_to_be_visible()


def test_mobile_responsive_layout(page: Page):
    """Test 11: Mobile viewport (375x667) renders properly with adapted layout."""
    page.set_viewport_size({"width": 375, "height": 667})
    page.evaluate("selectDate('2026-08-23')")
    page.wait_for_selector(".summary-grid", timeout=5000)

    # In mobile, manifest summary should be hidden, summary grid should be 2 columns
    manifest_summary = page.locator("#manifest-summary")
    expect(manifest_summary).not_to_be_visible()

    # Race cards should still be clickable and readable
    race_cards = page.locator(".race-card")
    expect(race_cards).to_have_count(36)


def test_no_console_errors(page: Page):
    """Test 12: Verify zero critical uncaught JavaScript errors in console during interactions."""
    page.evaluate("selectDate('2026-08-23')")
    time.sleep(1.0)

    error_logs = [log for log in page.console_logs if log.startswith("[error]")]
    critical_errors = [e for e in error_logs if "favicon" not in e.lower()]
    assert len(critical_errors) == 0, f"Critical console errors detected: {critical_errors}"
