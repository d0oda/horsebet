import logging
import pandas as pd
from models.features import FeatureBuilder
from models.ensemble import HybridEnsemble
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_hybrid")

def main():
    log.info("Loading features...")
    fb = FeatureBuilder()
    df = fb.build_features_all()
    
    if df.empty:
        log.error("No training data")
        return
        
    hybrid = HybridEnsemble(calibration_method="none")
    log.info("Training hybrid model...")
    # Use the same data cutoff as previous models or train on all
    hybrid.train(df)
    
    version = "2026_hybrid_latest"
    hybrid.save(version=version)
    log.info(f"Hybrid model saved as {version}")

if __name__ == "__main__":
    main()
