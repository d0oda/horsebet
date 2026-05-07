import requests
from bs4 import BeautifulSoup
import re

url = "https://umanity.jp/racedata/race_8_1.php?code=2026030706020501" 
headers = {"User-Agent": "Mozilla/5.0"}

resp = requests.get(url, headers=headers)
soup = BeautifulSoup(resp.content, 'html.parser')

print("Fetching odds...")
for row in soup.select("table.entry_table tr"):
    horse_links = row.select("a[href*='horse_top.php']")
    if not horse_links: continue
    
    cells = row.find_all('td')
    if len(cells) > 10:
        pp = cells[1].get_text(strip=True)
        horse = horse_links[0].get_text(strip=True)
        # Often odds are towards the right, maybe cell 13/14
        odds_candidates = [c.get_text(strip=True) for c in cells]
        print(f"[{pp}] {horse}: {odds_candidates}")
