import requests
import re
from bs4 import BeautifulSoup
url = "https://race.netkeiba.com/top/race_list.html?kaisai_date=20260307"
headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
resp = requests.get(url, headers=headers)
resp.encoding = "EUC-JP"
soup = BeautifulSoup(resp.text, "html.parser")

links = soup.find_all("a", href=re.compile(r"/race/\d{12}"))
print(f"Direct links containing /race/: {len(links)}")
for l in links[:3]: print(l.get("href"))

js_ids = re.findall(r'race_id["\']?\s*[:=]\s*["\']?(\d{12})', str(soup))
print(f"JS ids: {len(js_ids)}")
if js_ids: print(js_ids[:3])

b_links = soup.find_all("a", href=re.compile(r"shutuba.html\?race_id=\d{12}"))
print(f"shutuba links: {len(b_links)}")
for l in b_links[:3]: print(l.get("href"))
print("Done")
