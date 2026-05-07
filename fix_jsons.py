import json
from pathlib import Path
for d in ['2026-04-04', '2026-04-05', '2026-04-11', '2026-04-12', '2026-04-18', '2026-04-19', '2026-04-25', '2026-04-26']:
    path = Path(f"results/predictions_{d}.json")
    if not path.exists(): continue
    with open(path) as f:
        data = json.load(f)
    data["date"] = d
    data["model"] = "retrain_20260507_1645"
    data["ev_threshold"] = 0.20
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
