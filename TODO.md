# UmaEdge — Data Backfill Status

> Updated: 2026-03-04
> DB: 8,920 races | 38,872 horses | 121,563 results

---

## ✅ Completed

| Backfill | Result |
|----------|--------|
| 2021 race coverage | 3,456 races / 106 dates scraped via JRA EN ✅ |
| Sire names | All filled (only 1 ghost horse remaining — no valid ID) |
| Race class | All 8,920 races have class assigned |
| Jockey win rates | 627 / 628 computed (1 has no finished results) |
| Trainer win rates | 734 / 743 computed (9 are orphaned/no results) |
| Running style | All filled from corner positions ✅ |

---

## 🔴 After Each Year Scrape — Re-run These (no scraping, instant)

Each year scrape adds new horses, results, and entries. Re-run these after each scrape finishes:

- [x] **Sire names** — already filled by JRA scraper inline
- [x] **Race class** — already filled by JRA scraper inline
- [x] **Jockey/trainer win rates** — recomputed with 2021 data
- [x] **Running style** — backfill in progress now
  ```
  python -m scraper.backfill_running_style
  ```

---

## 🟡 Still Needed — Scraping Required

### Race Coverage Gaps

| Year | Races | Dates | Status |
|------|-------|-------|--------|
| 2019 | 2,084 | 91 | ✅ Good |
| 2020 | 304 | 23 | 🔄 Scraping now |
| 2021 | 3,456 | 106 | ✅ Complete |
| 2022 | 550 | 43 | 🔄 Scraping now |
| 2023 | 550 | 38 | 🔄 Scraping now |
| 2024 | 681 | 44 | 🔄 Scraping now |
| 2025 | 1,224 | 76 | � Scraping now |
| 2026 | 72 | 2 | Current season |

Years 2020, 2022–2025 are scraping now in background:
```
for year in 2020 2022 2023 2024 2025; do
  python -m scraper.backfill_jra --year $year --workers 4
done
```

After scrapes finish, re-run:
```
python -m scraper.backfill_running_style
```
Then recompute jockey/trainer win rates via SQL.

---

## ⚪ Nice to Have

- [ ] Per-venue ROI breakdown charts
- [ ] Odds range slider on results page
