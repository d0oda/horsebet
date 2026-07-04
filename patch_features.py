import sys

with open("models/features.py", "r") as f:
    lines = f.readlines()

build_idx = -1
for i, line in enumerate(lines):
    if line.startswith("    def _build("):
        build_idx = i
        break

loop_start = -1
for i in range(build_idx, len(lines)):
    if "for i, (idx, row) in enumerate(race_df.iterrows()):" in lines[i]:
        loop_start = i
        break

loop_end = -1
for i in range(loop_start, len(lines)):
    if "df = pd.DataFrame(feature_rows)" in lines[i]:
        loop_end = i
        break

loop_body = lines[loop_start+1:loop_end]
# Remove the logging logic from loop_body since it won't work correctly in chunks
new_loop_body = []
skip = False
for line in loop_body:
    if "if i > 0 and i % 1000 == 0:" in line:
        skip = True
        continue
    if skip:
        if line.strip() == "" or line.startswith("            features = {"):
            skip = False
        else:
            continue
    new_loop_body.append(line)

process_chunk_code = [
    "    def _process_chunk(self, chunk_df: pd.DataFrame, race_df: pd.DataFrame, history_df: pd.DataFrame, winner_times_dict: dict, second_times_dict: dict) -> list:\n",
    "        feature_rows = []\n",
    "        for idx, row in chunk_df.iterrows():\n"
]

# Adjust indentation of loop_body (remove 4 spaces because we removed enumerate)
for line in new_loop_body:
    if line.startswith("            "):
        process_chunk_code.append("    " + line[4:])
    else:
        process_chunk_code.append(line)

replacement_build_code = [
    "        total = len(race_df)\n",
    "        log.info(f'Building features for {total} entries...')\n",
    "        t_loop = time.time()\n",
    "        \n",
    "        race_ids = race_df['race_id'].unique()\n",
    "        import multiprocessing\n",
    "        from joblib import Parallel, delayed\n",
    "        n_jobs = multiprocessing.cpu_count()\n",
    "        race_id_chunks = np.array_split(race_ids, min(n_jobs * 2, len(race_ids)))\n",
    "        chunk_dfs = [race_df[race_df['race_id'].isin(chunk)] for chunk in race_id_chunks]\n",
    "        \n",
    "        log.info(f'Starting joblib pool with {n_jobs} workers across {len(chunk_dfs)} chunks...')\n",
    "        \n",
    "        results = Parallel(n_jobs=n_jobs)(\n",
    "            delayed(self._process_chunk)(chunk_df, race_df, history_df, winner_times_dict, second_times_dict)\n",
    "            for chunk_df in chunk_dfs\n",
    "        )\n",
    "        \n",
    "        feature_rows = []\n",
    "        for res in results:\n",
    "            feature_rows.extend(res)\n",
    "        \n",
    "        elapsed = time.time() - t_loop\n",
    "        rate = total / elapsed if elapsed > 0 else 0\n",
    "        log.info(f'Completed {total} entries | {rate:.0f} entries/s | Total Time {elapsed:.0f}s')\n",
    "        \n",
    "        df = pd.DataFrame(feature_rows)\n"
]

# Insert process_chunk before _build
lines.insert(build_idx, "".join(process_chunk_code) + "\n")

# Find the new indices
for i, line in enumerate(lines):
    if "total = len(race_df)" in line and "Building features for" in lines[i+1]:
        start_rep = i
        break

for i in range(start_rep, len(lines)):
    if "df = pd.DataFrame(feature_rows)" in lines[i]:
        end_rep = i + 1
        break

new_lines = lines[:start_rep] + replacement_build_code + lines[end_rep:]

with open("models/features.py", "w") as f:
    f.writelines(new_lines)

print("features.py successfully patched.")
