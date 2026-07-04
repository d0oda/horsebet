import logging
import time
import pandas as pd
from models.features import FeatureBuilder

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("rebuild_all")

def main():
    log.info("Starting Full Database Feature Rebuild (2014-2026)...")
    t0 = time.time()
    
    fb = FeatureBuilder()
    df = fb.build_features_all()
    
    if df.empty:
        log.error("Feature building returned an empty DataFrame!")
        return

    min_date = df["date"].min()
    max_date = df["date"].max()
    rows = len(df)
    cols = len(df.columns)
    
    log.info(f"✅ Built {rows} feature vectors with {cols} columns.")
    log.info(f"Date range: {min_date} -> {max_date}")
    
    out_path = "data/features.parquet"
    log.info(f"Saving to {out_path}...")
    df.to_parquet(out_path, index=False)
    
    t_total = time.time() - t0
    log.info(f"🎉 Done in {t_total:.1f}s!")

if __name__ == "__main__":
    main()
