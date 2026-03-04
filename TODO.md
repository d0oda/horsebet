# UmaEdge — Project Status

> Updated: 2026-03-04
> DB: 8,920 races | 38,872 horses | 123,220 entries | 122,161 results

---

## ✅ Completed

### Data Backfills

| Backfill | Result |
|----------|--------|
| 2021 race coverage | 3,456 races / 106 dates scraped via JRA EN ✅ |
| Sire names | All filled (only 1 ghost horse remaining — no valid ID) |
| Race class | All 8,920 races have class assigned |
| Jockey win rates | 627 / 628 computed (1 has no finished results) |
| Trainer win rates | 734 / 743 computed (9 are orphaned/no results) |
| Running style | All filled from corner positions ✅ |
| Horse body weight | **99.8%** (123,005/123,220) — 215 missing are scratched/DNF ✅ |
| Horse ID dedup | 37,494 jra_* IDs resolved; 33,694 duplicates merged ✅ |

### Sprint 8 — Domain-Specific Features (18 new columns)

| Feature | Method | Keys |
|---------|--------|------|
| Speed Figures | `_speed_figure_features()` | `speed_figure_last`, `speed_figure_best`, `speed_figure_avg3` |
| Jockey-Trainer Combo | `_jockey_trainer_combo_features()` | `jt_combo_runs`, `jt_combo_win_pct`, `jt_combo_place_pct` |
| Beaten Lengths | `_horse_rolling_features()` | `beaten_lengths_avg3`, `beaten_lengths_best`, `class_adjusted_margin` |
| Fitness Curve | `_horse_rolling_features()` | `is_fresh`, `is_rested`, `is_stale` |
| Field Quality | `_field_quality_features()` | `field_avg_career_win_pct`, `horse_vs_field_quality` |
| Weight vs Field | `_build()` | `weight_vs_field_avg`, `weight_per_kg_body` |
| Age × Class | `_build()` | `age_x_class`, `is_improving_3yo` |

Tests: **238/238 pass**

---

## 🔴 After Each Year Scrape — Re-run These (no scraping, instant)

- [x] **Sire names** — filled by JRA scraper inline
- [x] **Race class** — filled by JRA scraper inline
- [x] **Jockey/trainer win rates** — recomputed with 2021 data
- [x] **Running style** — `python -m scraper.backfill_running_style`
- [x] **Horse weights** — `python -m scraper.backfill_weights_netkeiba --workers 3`

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
| 2025 | 1,224 | 76 | 🔄 Scraping now |
| 2026 | 72 | 2 | Current season |

---

## 🔵 Next Steps

- [ ] Retrain model with Sprint 8 features: `python -m models.train`
- [ ] Run `python -m models.verify_features` to check population rates
- [ ] Clean orphaned jra_* horse records (8,206 remaining with no entries)

## ⚪ Nice to Have

- [ ] Per-venue ROI breakdown charts
- [ ] Odds range slider on results page
