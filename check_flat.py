"""Report flat (zero-amplitude) epochs and channels in the prepared HBN cache.

Reads only the cache, so it is fast and touches no BDF.

    python check_flat.py --config configs/hbn_light.json
"""
import argparse, json
from pathlib import Path
import numpy as np

EPS = 1e-8
ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/hbn_light.json")
ap.add_argument("--write-exclude", default="",
                help="Optional path to write the list of affected subjects.")
a = ap.parse_args()

cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
root = Path(cfg["cache"])
meta = json.loads((root / "manifest.json").read_text())

bad_all, bad_some, rows = [], [], []
for rec in meta["records"]:
    with np.load(root / rec["cache"], allow_pickle=False) as z:
        x = z["x"]                                  # [epochs, channels, samples]
    rms = np.sqrt(np.mean(x.astype(np.float64) ** 2, axis=-1))   # [epochs, channels]
    valid = rms > EPS
    dead_epochs = int(np.sum(valid.sum(1) == 0))    # epochs with NO usable channel
    dead_ch = int(np.sum(valid.sum(0) == 0))        # channels flat in every epoch
    names = [cfg["channels"][j] for j in np.flatnonzero(valid.sum(0) == 0)]
    rows.append((rec["subject"], dead_epochs, dead_ch, float(np.median(rms)), names))
    if dead_epochs:
        bad_all.append(rec["subject"])
    elif dead_ch:
        bad_some.append(rec["subject"])

print(f"records: {len(rows)}")
print(f"subjects with >=1 fully flat epoch (these break the run): {len(bad_all)}")
for s in bad_all:
    d = next(r for r in rows if r[0] == s)
    print(f"   {s}  dead_epochs={d[1]}/12  dead_channels={d[2]}/{len(cfg['channels'])}  median_rms={d[3]:.4g}")
print(f"subjects with a channel flat throughout (run survives): {len(bad_some)}")
from collections import Counter
which = Counter(n for r in rows for n in r[4])
if which:
    print("  dead channels, by electrode (subjects affected):")
    for n, c in which.most_common():
        print(f"    {n}: {c}/{len(rows)}")

med = np.array([r[3] for r in rows])
print(f"\nmedian epoch RMS across subjects: min {med.min():.4g}  "
      f"p05 {np.quantile(med,.05):.4g}  median {np.median(med):.4g}  max {med.max():.4g}")

if a.write_exclude and bad_all:
    Path(a.write_exclude).write_text("\n".join(bad_all) + "\n", encoding="utf-8")
    print(f"\nwrote {len(bad_all)} subject ids to {a.write_exclude}")
