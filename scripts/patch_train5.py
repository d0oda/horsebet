import sys

with open("models/train.py", "r") as f:
    content = f.read()

train_build_old = """    else:
        log.info("No cache found. Building features from scratch...")
        fb = FeatureBuilder()
        df = fb.build_features_all()

    if df.empty:"""

train_build_new = """    else:
        log.info("No cache found. Building features from scratch...")
        fb = FeatureBuilder()
        df = fb.build_features_all()
        if not df.empty:
            log.info("Saving features to cache data/features.parquet...")
            import os
            os.makedirs("data", exist_ok=True)
            if "date" in df.columns:
                df["date"] = df["date"].astype(str)
            df.to_parquet("data/features.parquet", index=False)

    if df.empty:"""

content = content.replace(train_build_old, train_build_new)

with open("models/train.py", "w") as f:
    f.write(content)

print("Applied feature cache save patch to models/train.py")
