# Database Backfill Pipeline

To completely backfill your database for a historical year, you should first navigate to this folder and then run the scripts in order from your terminal:

source .venv/bin/activate

```
cd work/backfill
```

### 1. Scrape the Year
```
./1_scrape_year.sh 2024
```
*This finds all race dates for the year, splits the work into 4 concurrent threads (months), and scrapes all the pre-race data and results.*

### 2. Fix Missing Results
```
./2_fix_missing_results.sh
```
*Run this after step 1 finishes. Because Netkeiba sometimes moves older results to an archive server, the first script might miss finish positions. This script scans your DB and automatically patches any broken races using the legacy archive.*

### 3. Backfill U-Index
```
./3_backfill_u_index.sh 2024
```
*Run this last. This scrapes the proprietary U-Index speed figures from Umanity and links them to the horses in your database.*


### 4. How to pause and resume a script

To pause a command that is currently running in your terminal, press:

Ctrl + Z

This will suspend (pause) the process.

When you get back and want to resume it, you have two options:

Type fg and press Enter to resume it in the foreground (where you left off).
Type bg and press Enter to resume it in the background, allowing you to use the terminal for other things while it finishes.



# 1. Activate your virtual environment
source .venv/bin/activate

# 2. Run the Sire backfill (this will find all horses missing a sire and scrape them)
python -m scraper.backfill_sires --workers 4

# 3. Run the Broodmare Sire backfill (maternal grandfathers)
python -m scraper.backfill_broodmare_sire --workers 4
