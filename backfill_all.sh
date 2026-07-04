#!/bin/bash
for year in 2018 2019 2020 2022 2023 2024 2025; do
    echo "======================================"
    echo " Starting backfill for year $year"
    echo "======================================"
    .venv/bin/python -m scraper.batch_scrape_concurrent --year $year --workers 3
done
echo "ALL YEARS COMPLETED!"
