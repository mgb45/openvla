#!/usr/bin/env python3
"""
oxe_fetch.py

Downloads Open-X Embodiment RLDS datasets from the public `gs://gresearch/robotics` bucket
for transfer into an air-gapped environment.

Deliberately depends on the Python standard library ONLY -- no gcloud SDK, no gsutil, no
credentials, no pip install. Anonymous HTTPS against the GCS JSON API. It should run on any
box with python3 >= 3.8 and outbound HTTPS (including through an HTTP proxy, via the standard
`https_proxy` environment variable).

Three subcommands, run in this order:

    plan     Resolve what would be downloaded and write a manifest. Downloads nothing.
             Run this first and read the size total before committing to a transfer.
    fetch    Download everything in the manifest. Resumable -- safe to re-run after an
             interruption; files already present with the right size are skipped.
    verify   Re-check every local file against the manifest's size and MD5.

Example:

    python3 oxe_fetch.py plan  --mixture bridge_rt_1 --out /staging/oxe
    python3 oxe_fetch.py fetch --out /staging/oxe --jobs 8
    python3 oxe_fetch.py verify --out /staging/oxe

See docs/ingest-runbook.md for the full procedure.
"""

import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

BUCKET = "gresearch"
PREFIX = "robotics"
API = f"https://storage.googleapis.com/storage/v1/b/{BUCKET}/o"
MEDIA = f"https://storage.googleapis.com/{BUCKET}"

# `bridge_orig` is NOT in the OXE bucket. The mixtures in this repo deliberately use the
# original BridgeData V2 release rather than the OXE re-export (`bridge`), because the two
# have different action conventions. Downloading the wrong one produces a mixture that
# trains without error and learns the wrong thing.
BRIDGE_ORIG_URL = "https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/"

# Datasets whose byte cost is wildly out of proportion to their sampling weight. Subsampling
# these is the single largest lever on total transfer size; see docs/ingest-runbook.md.
# Measured 2026-08-19 against version 0.1.0 of each.
HEAVY = {"language_table": 429.4, "kuka": 839.1}


# ---------------------------------------------------------------------------- mixtures


def load_mixture(name: str) -> "list[str]":
    """Read a named mixture out of the repo's own mixtures.py, so this script can never
    drift from what training actually expects."""
    here = Path(__file__).resolve()
    mixtures = here.parents[2] / "prismatic" / "vla" / "datasets" / "rlds" / "oxe" / "mixtures.py"
    if not mixtures.exists():
        sys.exit(f"error: cannot find {mixtures}; run this script from inside the openvla checkout")

    src = mixtures.read_text()
    try:
        block = src.split(f'"{name}": [', 1)[1].split("],", 1)[0]
    except IndexError:
        names = re.findall(r'^\s{4}"([a-z0-9_+]+)":\s*\[', src, re.M)
        sys.exit(f"error: unknown mixture '{name}'. Available:\n  " + "\n  ".join(names))

    # Skip commented-out entries -- several mixtures carry disabled alternatives.
    out = []
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        m = re.match(r'\("([a-z0-9_]+)",', stripped)
        if m and m.group(1) not in out:
            out.append(m.group(1))
    return out


# ---------------------------------------------------------------------------- GCS listing


def http_json(url: str, retries: int = 5) -> dict:
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "oxe-fetch/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r:
                return json.load(r)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
            if attempt == retries - 1:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def find_version(dataset: str, want: "str | None" = None) -> str:
    """Resolve which version subdirectory to pull.

    Two traps live here. Some datasets (e.g. `language_table`) carry several sibling
    prefixes -- multiple semantic versions plus non-version variants like `captions` and
    `long_horizon` -- and summing across all of them badly overstates the transfer. Others
    (e.g. `cmu_stretch`) additionally keep a stray un-versioned copy of the shards at the
    dataset root, which TFDS ignores but a naive recursive copy would drag along.
    """
    q = {"prefix": f"{PREFIX}/{dataset}/", "delimiter": "/", "maxResults": "1000", "fields": "prefixes"}
    page = http_json(f"{API}?{urllib.parse.urlencode(q)}")
    versions = [p.rstrip("/").split("/")[-1] for p in page.get("prefixes", [])]
    semver = sorted(v for v in versions if re.fullmatch(r"\d+\.\d+\.\d+", v))

    if want:
        if want not in versions:
            sys.exit(f"error: {dataset} has no version '{want}' (found: {', '.join(versions) or 'none'})")
        return want
    if not semver:
        sys.exit(f"error: {dataset} has no versioned subdirectory (found: {', '.join(versions) or 'none'})")
    return semver[-1]


def list_objects(dataset: str, version: str) -> "list[dict]":
    """List every object under robotics/<dataset>/<version>/, with size and md5."""
    items, token = [], None
    while True:
        q = {
            "prefix": f"{PREFIX}/{dataset}/{version}/",
            "maxResults": "1000",
            "fields": "items(name,size,md5Hash),nextPageToken",
        }
        if token:
            q["pageToken"] = token
        page = http_json(f"{API}?{urllib.parse.urlencode(q)}")
        for it in page.get("items", []):
            # GCS represents pseudo-directories as zero-content marker objects.
            if it["name"].endswith("_$folder$"):
                continue
            items.append(it)
        token = page.get("nextPageToken")
        if not token:
            return items


# ---------------------------------------------------------------------------- shard subsampling


def is_shard(name: str) -> bool:
    return ".tfrecord-" in name


def subsample(items: "list[dict]", fraction: float) -> "tuple[list[dict], int]":
    """Keep an evenly-spaced deterministic subset of the record shards; keep all metadata.

    Evenly spaced rather than random: RLDS shards are written in episode order, so a
    contiguous prefix would bias toward whatever was collected first.

    NOTE: the kept shards must be renumbered and dataset_info.json rewritten before TFDS
    will read the result. `fetch` does that automatically and flags it in the manifest.
    """
    shards = sorted((i for i in items if is_shard(i["name"])), key=lambda i: i["name"])
    meta = [i for i in items if not is_shard(i["name"])]
    if fraction >= 1.0 or not shards:
        return items, len(shards)

    keep_n = max(1, round(len(shards) * fraction))
    step = len(shards) / keep_n
    kept = [shards[min(len(shards) - 1, int(k * step))] for k in range(keep_n)]
    return meta + kept, len(shards)


# ---------------------------------------------------------------------------- plan


def cmd_plan(args) -> None:
    datasets = args.dataset or load_mixture(args.mixture)
    fractions = dict(p.split("=", 1) for p in args.fraction)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    entries, warnings = [], []
    print(f"Resolving {len(datasets)} dataset(s) against gs://{BUCKET}/{PREFIX}/ ...\n", flush=True)

    pins = dict(p.split("=", 1) for p in args.version)
    gcs_datasets = [d for d in datasets if d != "bridge_orig"]

    def resolve(d):
        v = find_version(d, pins.get(d))
        return d, v, list_objects(d, v)

    listings = {}
    with ThreadPoolExecutor(min(8, max(1, len(gcs_datasets)))) as ex:
        for d, v, items in ex.map(resolve, gcs_datasets):
            listings[d] = (v, items)

    total = 0
    for ds in datasets:
        if ds == "bridge_orig":
            warnings.append(
                "bridge_orig is not in the OXE bucket -- `fetch` pulls it over HTTP from "
                "rail.eecs.berkeley.edu (~124 GB). Do NOT substitute the OXE `bridge` dataset."
            )
            entries.append({"dataset": ds, "source": "http_tree", "url": BRIDGE_ORIG_URL, "objects": []})
            print(f"  {'~124.0':>8} GB  {'(http)':>8}  {ds}")
            continue

        version, items = listings[ds]
        if not items:
            sys.exit(f"error: no objects found for '{ds}' -- check the dataset name")

        frac = float(fractions.get(ds, args.shard_fraction))
        items, n_orig = subsample(items, frac)
        size = sum(int(i["size"]) for i in items)
        total += size

        entries.append(
            {
                "dataset": ds,
                "source": "gcs",
                "version": version,
                "shard_fraction": frac,
                "shards_total": n_orig,
                "shards_kept": sum(1 for i in items if is_shard(i["name"])),
                "bytes": size,
                "objects": [
                    {"name": i["name"], "size": int(i["size"]), "md5": i.get("md5Hash")} for i in items
                ],
            }
        )
        flag = f"  [{frac:.0%} of {n_orig} shards]" if frac < 1.0 else ""
        print(f"  {size / 1e9:8.1f} GB  {len(items):>8}  {ds} @ {version}{flag}")

        if frac >= 1.0 and ds in HEAVY:
            warnings.append(
                f"{ds} is {HEAVY[ds]:.0f} GB at full size. Consider --fraction {ds}=0.1 "
                f"-- it is heavily downweighted in the training mixture."
            )

    manifest = {
        "version": 1,
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "mixture": args.mixture if not args.dataset else None,
        "bytes_total": total,
        "entries": entries,
    }
    path = out / "oxe_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))

    print(f"\n  {'=' * 30}")
    print(f"  {total / 1e12:8.2f} TB  total from GCS")
    print(f"\nManifest written to {path}")
    for w in dict.fromkeys(warnings):
        print(f"\n  ! {w}")


# ---------------------------------------------------------------------------- fetch


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4 << 20), b""):
            h.update(chunk)
    return base64.b64encode(h.digest()).decode()


def download(obj: dict, root: Path, check_md5: bool) -> "tuple[str, int]":
    """Returns (status, bytes_transferred). Status is 'skip', 'ok', or an error string."""
    rel = obj["name"][len(PREFIX) + 1 :]  # strip 'robotics/'
    dest = root / rel
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and dest.stat().st_size == obj["size"]:
        if not check_md5 or not obj.get("md5") or md5_of(dest) == obj["md5"]:
            return "skip", 0

    url = f"{MEDIA}/{urllib.parse.quote(obj['name'])}"
    tmp = dest.with_suffix(dest.suffix + ".partial")
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "oxe-fetch/1.0"})
            with urllib.request.urlopen(req, timeout=300) as r, tmp.open("wb") as f:
                shutil.copyfileobj(r, f, 8 << 20)
            if tmp.stat().st_size != obj["size"]:
                raise IOError(f"size mismatch: got {tmp.stat().st_size}, want {obj['size']}")
            tmp.replace(dest)
            return "ok", obj["size"]
        except Exception as e:  # noqa: BLE001 -- retry anything transient
            if attempt == 4:
                tmp.unlink(missing_ok=True)
                return f"FAILED {rel}: {e}", 0
            time.sleep(2**attempt)
    return "unreachable", 0


def renumber_subsampled(entry: dict, root: Path) -> None:
    """Rewrite dataset_info.json and rename shards so TFDS can read a subsampled dataset.

    TFDS resolves shards by the `-NNNNN-of-MMMMM` suffix and cross-checks the count against
    `shardLengths` in dataset_info.json. Both must be rewritten together or the dataset will
    not open.
    """
    ds_dir = root / entry["dataset"]
    versions = [p for p in ds_dir.iterdir() if p.is_dir()]
    for vdir in versions:
        info_path = vdir / "dataset_info.json"
        if not info_path.exists():
            continue
        info = json.loads(info_path.read_text())

        for split in info.get("splits", []):
            lengths = split.get("shardLengths")
            if not lengths:
                continue
            prefix = f"{info['name']}-{split['name']}.tfrecord"
            present = sorted(vdir.glob(f"{prefix}-*"))
            if len(present) == len(lengths):
                continue  # not subsampled

            kept_idx = [int(p.name.rsplit("-of-", 1)[0].rsplit("-", 1)[1]) for p in present]
            split["shardLengths"] = [lengths[i] for i in kept_idx]
            # numBytes/numExamples are advisory for reads but keep them honest.
            split["numExamples"] = str(sum(int(x) for x in split["shardLengths"]))

            total = len(present)
            for new_i, old in enumerate(present):
                new = vdir / f"{prefix}-{new_i:05d}-of-{total:05d}"
                if old != new:
                    old.rename(new)

        info_path.write_text(json.dumps(info, indent=2))
        print(f"  rewrote {info_path} for subsampled shards")


def cmd_fetch(args) -> None:
    root = Path(args.out)
    manifest = json.loads((root / "oxe_manifest.json").read_text())

    for entry in manifest["entries"]:
        ds = entry["dataset"]
        if entry["source"] == "http_tree":
            fetch_http_tree(entry, root)
            continue

        objs = entry["objects"]
        print(f"\n{ds}: {len(objs)} objects, {entry['bytes'] / 1e9:.1f} GB", flush=True)
        done = moved = 0
        errors = []
        with ThreadPoolExecutor(args.jobs) as ex:
            futures = [ex.submit(download, o, root, args.verify_md5) for o in objs]
            for f in as_completed(futures):
                status, n = f.result()
                done += 1
                moved += n
                if status.startswith("FAILED"):
                    errors.append(status)
                if done % 50 == 0 or done == len(objs):
                    print(f"  {done}/{len(objs)}  {moved / 1e9:.1f} GB", flush=True)
        if errors:
            print(f"  {len(errors)} FAILURES -- re-run `fetch` to retry:", file=sys.stderr)
            for e in errors[:10]:
                print(f"    {e}", file=sys.stderr)
            sys.exit(1)
        if entry.get("shard_fraction", 1.0) < 1.0:
            renumber_subsampled(entry, root)

    print("\nAll datasets fetched. Run `verify` before starting the transfer.")


def fetch_http_tree(entry: dict, root: Path) -> None:
    """BridgeData V2 is served as a plain HTTP directory tree, not from GCS."""
    dest = root / entry["dataset"]
    print(f"\n{entry['dataset']}: recursive HTTP fetch (~124 GB) from {entry['url']}", flush=True)
    if shutil.which("wget") is None:
        print(
            "  wget not found. Fetch this dataset manually:\n"
            f"    wget -r -nH --cut-dirs=4 --reject='index.html*' {entry['url']}\n"
            f"    mv bridge_dataset {dest}",
            file=sys.stderr,
        )
        sys.exit(1)
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["wget", "-r", "-nH", "--cut-dirs=4", "-c", "--reject=index.html*", "-P", str(dest.parent), entry["url"]]
    subprocess.run(cmd, check=True)
    staged = dest.parent / "bridge_dataset"
    if staged.exists() and not dest.exists():
        staged.rename(dest)  # the loader requires the name `bridge_orig`


# ---------------------------------------------------------------------------- verify


def cmd_verify(args) -> None:
    root = Path(args.out)
    manifest = json.loads((root / "oxe_manifest.json").read_text())
    bad, checked = [], 0

    def check(obj):
        rel = obj["name"][len(PREFIX) + 1 :]
        p = root / rel
        if not p.exists():
            return f"MISSING  {rel}"
        if p.stat().st_size != obj["size"]:
            return f"SIZE     {rel}"
        if args.deep and obj.get("md5") and md5_of(p) != obj["md5"]:
            return f"MD5      {rel}"
        return None

    for entry in manifest["entries"]:
        if entry["source"] != "gcs":
            print(f"{entry['dataset']}: skipped (http tree -- verify by size on both sides)")
            continue
        # A subsampled dataset has been renamed on disk; size-verify by total instead.
        if entry.get("shard_fraction", 1.0) < 1.0:
            got = sum(p.stat().st_size for p in (root / entry["dataset"]).rglob("*") if p.is_file())
            ok = abs(got - entry["bytes"]) < entry["bytes"] * 0.001
            print(f"{entry['dataset']}: {'OK' if ok else 'MISMATCH'} ({got / 1e9:.1f} GB, subsampled)")
            if not ok:
                bad.append(entry["dataset"])
            continue
        with ThreadPoolExecutor(args.jobs) as ex:
            for r in ex.map(check, entry["objects"]):
                checked += 1
                if r:
                    bad.append(r)
        print(f"{entry['dataset']}: {'OK' if not bad else 'see failures below'}")

    print(f"\nChecked {checked} objects.")
    if bad:
        print(f"{len(bad)} problems:")
        for b in bad[:40]:
            print(f"  {b}")
        sys.exit(1)
    print("All present and correct.")


# ---------------------------------------------------------------------------- cli


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    for name in ("plan", "fetch", "verify"):
        p = sub.add_parser(name)
        p.add_argument("--out", required=True, help="staging directory (becomes the RLDS data root)")
        p.add_argument("--jobs", type=int, default=8, help="parallel transfers (default 8)")
        if name == "plan":
            p.add_argument("--mixture", default="bridge_rt_1", help="named mixture from mixtures.py")
            p.add_argument("--dataset", action="append", help="explicit dataset name; repeatable, overrides --mixture")
            p.add_argument("--shard-fraction", type=float, default=1.0, help="global shard fraction, 0<f<=1")
            p.add_argument(
                "--version",
                action="append",
                default=[],
                metavar="DATASET=V",
                help="pin a dataset version, e.g. language_table=0.1.0 (default: highest semver)",
            )
            p.add_argument(
                "--fraction",
                action="append",
                default=[],
                metavar="DATASET=F",
                help="per-dataset shard fraction, e.g. language_table=0.1",
            )
        if name == "fetch":
            p.add_argument("--verify-md5", action="store_true", help="md5 already-present files instead of trusting size")
        if name == "verify":
            p.add_argument("--deep", action="store_true", help="md5 every file (slow)")

    args = ap.parse_args()
    {"plan": cmd_plan, "fetch": cmd_fetch, "verify": cmd_verify}[args.cmd](args)


if __name__ == "__main__":
    main()
