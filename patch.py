import re

with open("models/features.py", "r") as f:
    code = f.read()

# Replace basic `group[group["date"] < race_date]`
code = re.sub(
    r'(?P<indent>[ \t]+)(?P<var>\w+) = group\[group\["date"\] < race_date\]',
    r'\g<indent>idx = group["date"].searchsorted(race_date, side="left")\n\g<indent>\g<var> = group.iloc[:idx].iloc[::-1]',
    code
)

# Replace the course_id & date filter (lines 684 and 725 approx)
code = re.sub(
    r'(?P<indent>[ \t]+)(?P<var>\w+) = group\[\s*\(group\["course_id"\] == course_id\)\s*&\s*\(group\["date"\] < race_date\)\s*\]',
    r'\g<indent>idx = group["date"].searchsorted(race_date, side="left")\n\g<indent>\g<var> = group.iloc[:idx].iloc[::-1]\n\g<indent>\g<var> = \g<var>[\g<var>["course_id"] == course_id]',
    code
)

# Remove all `hist.sort_values("date", ascending=False)` since we already reversed it!
code = re.sub(
    r'\n[ \t]+(?P<var>\w+) = (?P=var)\.sort_values\("date", ascending=False\)',
    '',
    code
)

# Finally, sort history_df by date in `_build`
code = code.replace(
    '        history_df = self._load_horse_history()\n\n        # Pre-index history',
    '        history_df = self._load_horse_history()\n        history_df = history_df.sort_values("date", ascending=True)\n\n        # Pre-index history'
)

with open("models/features.py", "w") as f:
    f.write(code)
print("Done patching features.py")
