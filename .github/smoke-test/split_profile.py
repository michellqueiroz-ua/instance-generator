"""Split a reqreate_profile.json into a compact JSON plus the cProfile tables.

The cProfile text reports are thousands of lines. Inlining them in the issue
body along with the JSON would blow past GitHub's size limit, so lift them out
into their own files and leave the JSON small enough to read at a glance.
"""

import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "reqreate_profile.json"

with open(path, encoding="utf-8") as fh:
    data = json.load(fh)

notes = data.get("notes", {})
cumulative = notes.pop("cprofile_top_cumulative", "")
own_time = notes.pop("cprofile_top_own_time", "")

with open("cprofile_cumulative.txt", "w", encoding="utf-8") as fh:
    fh.write(cumulative)
with open("cprofile_own_time.txt", "w", encoding="utf-8") as fh:
    fh.write(own_time)

print(json.dumps(data, indent=2))
