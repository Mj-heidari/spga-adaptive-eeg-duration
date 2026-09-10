"""Build the reviewed HBN manifest and label table for Repo2.

Reads a release's participants.tsv, keeps subjects whose RestingState
recording is marked available and long enough, and writes:

  local/manifest.csv  one RestingState recording per subject
  local/labels.csv    subject, binary label, group, label_source, task

Labels come from participants.tsv only. Nothing is inferred from a file
name, and no diagnosis is invented: this release ships age, sex and the
CBCL bifactor scores, and no ADHD diagnosis, so `--target` selects among
what is actually there.

  --target age        age >= --age-cut years  (primary endpoint)
  --target attention  CBCL attention above the cohort median (negative control)
  --target sex        male = 1                (secondary control)

Run from the Repo2 root:

    python build_hbn_cohort.py --config configs/hbn_light.json --target age
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def rows_tsv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def write_csv(path, rows, fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def num(row, key):
    try:
        v = float(row[key])
    except (KeyError, TypeError, ValueError):
        return None
    return None if v != v else v


def duration_seconds(path):
    import mne
    raw = mne.io.read_raw_bdf(path, preload=False, verbose="ERROR")
    try:
        return raw.n_times / float(raw.info["sfreq"])
    finally:
        raw.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/hbn_light.json")
    ap.add_argument("--target", choices=["age", "attention", "sex"], default="age")
    ap.add_argument("--age-cut", type=float, default=10.0)
    ap.add_argument("--start", type=float, default=20.0,
                    help="Seconds skipped at the start of the recording.")
    ap.add_argument("--seconds", type=float, default=48.0,
                    help="Retained duration; must be 4 * max_epochs_hbn.")
    ap.add_argument("--max-subjects", type=int, default=0, help="0 = no cap.")
    ap.add_argument("--exclude-file", default="",
                    help="Text file with one subject id per line to exclude "
                         "(e.g. local/flat_subjects.txt from check_flat.py).")
    a = ap.parse_args()

    cfg = json.loads(Path(a.config).read_text(encoding="utf-8"))
    root = Path(cfg["root"])
    task = cfg["hbn_task"]
    need = 4.0 * int(cfg["max_epochs_hbn"])
    if abs(a.seconds - need) > 1e-9:
        raise SystemExit(f"--seconds must equal 4 * max_epochs_hbn = {need:.0f}")

    excluded = set()
    if a.exclude_file:
        p = Path(a.exclude_file)
        if p.is_file():
            excluded = {ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()}
            print(f"exclusion list: {len(excluded)} subject(s) from {p}")
        else:
            raise SystemExit(f"--exclude-file not found: {p}")

    parts = rows_tsv(root / "participants.tsv")
    print(f"participants.tsv: {len(parts)} subjects")

    avail = [r for r in parts if r.get(task) == "available"]
    print(f"{task} marked available: {len(avail)}")
    if excluded:
        before = len(avail)
        avail = [r for r in avail if r["participant_id"] not in excluded]
        print(f"after excluding flat recordings: {len(avail)}  (removed {before-len(avail)})")

    # ---- label definition, from participants.tsv only -------------------
    if a.target == "age":
        pool = [r for r in avail if num(r, "age") is not None]
        def label(r):
            return 1 if num(r, "age") >= a.age_cut else 0
        source = f"participants.tsv:age>={a.age_cut}"
        note = f"age >= {a.age_cut} y"
    elif a.target == "attention":
        pool = [r for r in avail if num(r, "attention") is not None]
        med = statistics.median(num(r, "attention") for r in pool)
        def label(r):
            return 1 if num(r, "attention") > med else 0
        source = f"participants.tsv:attention>{med:.4f}"
        note = f"CBCL attention > cohort median ({med:.4f})"
    else:
        pool = [r for r in avail if r.get("sex") in ("M", "F")]
        def label(r):
            return 1 if r["sex"] == "M" else 0
        source = "participants.tsv:sex==M"
        note = "male"
    print(f"with a usable {a.target} value: {len(pool)}   positive class = {note}")

    # ---- resolve files and check duration --------------------------------
    manifest, labels = [], []
    missing = short = 0
    for i, r in enumerate(sorted(pool, key=lambda r: r["participant_id"])):
        sid = r["participant_id"]
        rel = f"{sid}/eeg/{sid}_task-{task}_eeg.bdf"
        p = root / rel
        if not p.is_file():
            missing += 1
            continue
        try:
            secs = duration_seconds(p)
        except Exception as e:
            print(f"  unreadable {sid}: {e}")
            missing += 1
            continue
        if secs < a.start + a.seconds:
            short += 1
            continue
        manifest.append(dict(
            include=1, path=rel, subject=sid, group=sid,
            start_s=int(a.start), stop_s=int(a.start + a.seconds),
            events="", annotation_complete=0, task=task, bytes=p.stat().st_size))
        labels.append(dict(subject=sid, label=label(r), group=sid,
                           label_source=source, task=task))
        if a.max_subjects and len(manifest) >= a.max_subjects:
            break
        if (i + 1) % 50 == 0:
            print(f"  probed {i+1} ...", flush=True)

    if not manifest:
        raise SystemExit("Nothing selected; check the release root and task name.")

    write_csv(Path(cfg["manifest"]).with_name("manifest.proposed.csv"), manifest,
              ["include", "path", "subject", "group", "start_s", "stop_s",
               "events", "annotation_complete", "task", "bytes"])
    write_csv(Path(cfg["labels"]).with_name("labels.proposed.csv"), labels,
              ["subject", "label", "group", "label_source", "task"])

    pos = sum(l["label"] for l in labels)
    src_gb = sum(m["bytes"] for m in manifest) / 1e9
    est = len(manifest) * int(a.seconds / 4) * len(cfg["channels"]) * 4 * cfg["fs"] * 4
    print()
    print(f"selected {len(manifest)} subjects   ({missing} missing/unreadable, "
          f"{short} too short, {len(excluded)} excluded as flat)")
    print(f"class balance: {pos} positive / {len(labels)-pos} negative "
          f"({100*pos/len(labels):.1f}% positive)")
    print(f"retained interval: [{int(a.start)}, {int(a.start+a.seconds)}) s "
          f"= {int(a.seconds/4)} epochs of 4 s")
    print(f"source bytes {src_gb:.2f} GB of {cfg['max_source_bytes']/1e9:.2f} GB budget")
    print(f"estimated cache {est/1e6:.0f} MB of {cfg['max_cache_bytes']/1e6:.0f} MB budget")
    print()
    print("Review local/manifest.proposed.csv and local/labels.proposed.csv, then:")
    print("  copy local\\manifest.proposed.csv local\\manifest.csv")
    print("  copy local\\labels.proposed.csv   local\\labels.csv")


if __name__ == "__main__":
    main()
