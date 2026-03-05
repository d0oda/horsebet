# UmaEdge — Project Status

> Updated: 2026-03-05
> DB: 8,962 races | 31,835 horses | 122,164 entries | 122,164 results

---

## ✅ Completed

### Data Backfills

| Backfill | Result |
|----------|--------|
| 2021 race coverage | 3,456 races / 106 dates scraped via JRA EN ✅ |
| Sire names | All filled (only 1 ghost horse remaining — no valid ID) |
| Race class | All 8,962 races have class assigned |
| Jockey win rates | 627 / 628 computed (1 has no finished results) |
| Trainer win rates | 734 / 743 computed (9 are orphaned/no results) |
| Horse body weight | **100%** — 1,056 scratched entries deleted ✅ |
| Horse ID dedup | 37,494 jra_* IDs resolved; 33,694 duplicates merged ✅ |
| Broodmare sire | 30,962/39,171 actual runners filled (17,036 updated) ✅ |
| Orphaned horses | 8,163 deleted (870 kept as broodmare sire refs) ✅ |

### Sectional Times & Speed Data (2026-03-04/05)

| Column | Before | After | Method |
|--------|--------|-------|--------|
| `last_3f_secs` | 63.4% | **100%** ✅ | Re-scraped 3,245 races from netkeiba (`backfill_sectionals.py`) |
| `first_3f_secs` | 60.4% | **100%** ✅ | Derived `time_secs - last_3f_secs` via SQL migration |
| `corner_positions` | 63.4% | **100%** ✅ | Filled alongside last_3f backfill + metadata backfill |
| `running_style` | 98.4% | **100%** ✅ | Derived from corner_positions (`backfill_running_style.py`) + re-scrape |
| `margin` | 97.3% | **100%** ✅ | Set to 0 for all winners (finish_pos=1) via SQL migration |
| `weather` | 98.1% | **100%** ✅ | Re-scraped from netkeiba (`backfill_race_metadata.py`) |
| `going` | 95.5% | **100%** ✅ | Re-scraped + sibling race inference; fixed parser for `稍`→`稍重` and `不`→`不良` |
| `horse_weight` | 99.8% | **100%** ✅ | Deleted 1,056 scratched/withdrawn entries |
| `odds_win` | 99.6% | **100%** ✅ | Deleted 1,056 scratched/withdrawn entries |

### Feature Engineering Fix (2026-03-04)

- `avg_first_3f` / `best_first_3f` now fall back to `time_secs - last_3f_secs` when `first_3f_secs` is missing
- Test added: `test_first_3f_fallback_from_time_minus_last3f` — all 44 tests pass

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
- [x] **Sectional times** — `python -m scraper.backfill_sectionals --workers 3`
- [x] **Race metadata** — `python -m scraper.backfill_race_metadata --workers 2`

---

## ✅ All Data Columns — 100% Coverage

Every column in races, entries, and results is fully populated. No gaps remain.

## 🔵 Next Steps

- [x] ~~Retrain model with Sprint 8 features~~ — `python -m models.train`
- [x] ~~Run `verify_features`~~ — All 16 groups ✅ OK except Odds Movement (0% — no time-series odds data)
- [x] ~~Debug remaining going gaps~~ — Fixed: netkeiba abbreviates `不良`→`不` and `稍重`→`稍`. Now 100%.
- [x] ~~Clean orphaned jra_* horse records~~ — 8,163 deleted (870 kept as broodmare sire refs)
- [x] ~~Fix margin parser crash~~ — `_parse_margin_to_lengths()` handles all JRA formats + dead heats (同着/DH)
- [x] ~~Drop dead features: `odds_slope`, `odds_late_money`, `odds_vol`~~ — removed from features.py + verify_features.py

## ⚪ Nice to Have

- [ ] Per-venue ROI breakdown charts
- [ ] Odds range slider on results page
