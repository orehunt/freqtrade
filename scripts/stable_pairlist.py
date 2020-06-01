"""
Test pairlist `n_calls` times, then remove pairs which are not present
`max_strikes` times
"""
from freqtrade.configuration import TimeRange
from pathlib import Path
from user_data.modules.helper import read_json_file
from glob import glob
import os

n_calls = 60
max_strikes = 3
stable_pairs = []
strikes = {}

list(map(os.remove, glob("/tmp/tp*")))
# for i in range(n_calls):
#     with open(f"/tmp/tp{i}", 'w') as fl:
#         subprocess.run(f"tp.sh", cwd="/freqtrade", stdout=fl)

for f in glob("/tmp/tp*"):
    pairs = read_json_file(f)
    if not stable_pairs:
        stable_pairs = pairs
    else:
        for n, p in enumerate(stable_pairs):
            if p not in pairs:
                strikes[p] = 0 if p not in strikes else strikes[p] + 1
                if strikes[p] > max_strikes:
                    del stable_pairs[n]
print(stable_pairs)
