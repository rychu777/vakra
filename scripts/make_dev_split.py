#!/usr/bin/env python3
"""Build fixed, reproducible evaluation splits (dev, dev_mini) from the VAKRA train split.

Items of every capability and domain are shuffled with an RNG seeded by
"<seed>/<capability>/<domain>", and each split takes the first N items of that order,
so a smaller split is always a subset of a larger one.

The selected UUIDs are stored in manifests (experiments/splits/<split>.json) that are
committed, because data/ is git-ignored. Running the script again rebuilds identical
files from the manifests:

    python scripts/make_dev_split.py              # use existing manifests, or create them
    python scripts/make_dev_split.py --resample   # draw new splits and overwrite the manifests

The output mirrors data/train, so the runner and the evaluator can use it directly:

    data/<split>/<capability>/input/<domain>.json    -> CAPABILITY_N_DIR
    data/<split>/<capability>/output/<domain>.json   -> evaluator --gt_root
"""
from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SPLITS = ["dev=10", "dev_mini=3"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", default="data/train", help="Source split directory (default: data/train)")
    parser.add_argument("--data-root", default="data", help="Directory for the generated splits (default: data)")
    parser.add_argument(
        "--manifest-dir",
        default="experiments/splits",
        help="Directory for the UUID manifests (default: experiments/splits)",
    )
    parser.add_argument(
        "--split",
        action="append",
        default=None,
        metavar="NAME=N",
        help="Split name and maximum number of items per domain, repeatable (default: dev=10 dev_mini=3)",
    )
    parser.add_argument("--seed", type=int, default=2027, help="Base random seed (default: 2027)")
    parser.add_argument("--resample", action="store_true", help="Ignore existing manifests and draw new splits")
    return parser.parse_args()


def parse_split_specs(specs: list[str]) -> dict[str, int]:
    items_per_domain_by_split = {}
    for spec in specs:
        split_name, separator, size = spec.partition("=")
        if not separator or not split_name or not size.isdigit() or int(size) < 1:
            raise SystemExit(f"Invalid --split '{spec}', expected NAME=N with N >= 1")
        items_per_domain_by_split[split_name] = int(size)
    return items_per_domain_by_split


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def iter_domains(source_dir: Path):
    """Yield (capability, domain, input_items, groundtruth_items) for every domain of the source split."""
    for capability_dir in sorted(path for path in source_dir.glob("capability_*") if path.is_dir()):
        for input_path in sorted((capability_dir / "input").glob("*.json")):
            groundtruth_path = capability_dir / "output" / input_path.name
            if not groundtruth_path.exists():
                print(f"  WARNING: {capability_dir.name}/{input_path.stem} has no ground truth, skipped")
                continue
            yield capability_dir.name, input_path.stem, read_json(input_path), read_json(groundtruth_path)


def draw_splits(source_dir: Path, items_per_domain_by_split: dict[str, int], seed: int) -> dict:
    """Return {split: {capability: {domain: [uuid, ...]}}} with UUIDs in their source order."""
    selection = {split_name: {} for split_name in items_per_domain_by_split}
    for capability, domain, input_items, _ in iter_domains(source_dir):
        source_uuids = [item["uuid"] for item in input_items]
        shuffled_uuids = list(source_uuids)
        random.Random(f"{seed}/{capability}/{domain}").shuffle(shuffled_uuids)
        for split_name, items_per_domain in items_per_domain_by_split.items():
            chosen_uuids = set(shuffled_uuids[:items_per_domain])
            selected = [uuid for uuid in source_uuids if uuid in chosen_uuids]
            selection[split_name].setdefault(capability, {})[domain] = selected
    return selection


def materialize_split(source_dir: Path, split_dir: Path, uuids_by_capability: dict) -> dict[str, tuple[int, int]]:
    """Write the input and ground-truth files of one split. Return {capability: (domain_count, item_count)}."""
    source_items = {
        (capability, domain): (input_items, groundtruth_items)
        for capability, domain, input_items, groundtruth_items in iter_domains(source_dir)
    }
    counts = {}
    for capability, uuids_by_domain in sorted(uuids_by_capability.items()):
        item_count = 0
        for domain, uuids in sorted(uuids_by_domain.items()):
            if (capability, domain) not in source_items:
                raise SystemExit(f"The manifest refers to {capability}/{domain}, which is missing from {source_dir}")
            input_items, groundtruth_items = source_items[(capability, domain)]
            wanted_uuids = set(uuids)
            selected_inputs = [item for item in input_items if item["uuid"] in wanted_uuids]
            selected_groundtruth = [item for item in groundtruth_items if item["uuid"] in wanted_uuids]
            if len(selected_inputs) != len(wanted_uuids) or len(selected_groundtruth) != len(wanted_uuids):
                raise SystemExit(f"{capability}/{domain}: some manifest UUIDs are missing from {source_dir}")
            write_json(split_dir / capability / "input" / f"{domain}.json", selected_inputs)
            write_json(split_dir / capability / "output" / f"{domain}.json", selected_groundtruth)
            item_count += len(selected_inputs)
        counts[capability] = (len(uuids_by_domain), item_count)
    return counts


def load_or_create_manifest(manifest_path: Path, split_name: str, items_per_domain: int,
                            new_selection: dict, args: argparse.Namespace) -> dict:
    if manifest_path.exists() and not args.resample:
        manifest = read_json(manifest_path)
        print(f"\n[{split_name}] using {manifest_path.relative_to(REPO_ROOT)} "
              f"(per_domain={manifest['per_domain']}, seed={manifest['seed']})")
        if manifest["per_domain"] != items_per_domain:
            print(f"  NOTE: the manifest keeps per_domain={manifest['per_domain']}; use --resample to change it")
        return manifest

    manifest = {
        "split": split_name,
        "per_domain": items_per_domain,
        "seed": args.seed,
        "source": args.source,
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "capabilities": new_selection[split_name],
    }
    write_json(manifest_path, manifest)
    print(f"\n[{split_name}] wrote {manifest_path.relative_to(REPO_ROOT)} (per_domain={items_per_domain}, seed={args.seed})")
    return manifest


def main() -> None:
    args = parse_args()
    items_per_domain_by_split = parse_split_specs(args.split or DEFAULT_SPLITS)
    source_dir = (REPO_ROOT / args.source).resolve()
    data_root = (REPO_ROOT / args.data_root).resolve()
    manifest_dir = (REPO_ROOT / args.manifest_dir).resolve()

    if not any(source_dir.glob("capability_*/input/*.json")):
        raise SystemExit(f"No data in {source_dir}. Run 'python benchmark_setup.py --download-data' first.")
    if any((data_root / split_name).resolve() == source_dir for split_name in items_per_domain_by_split):
        raise SystemExit("A split cannot overwrite its own source directory.")

    manifest_paths = {split_name: manifest_dir / f"{split_name}.json" for split_name in items_per_domain_by_split}
    needs_sampling = args.resample or not all(path.exists() for path in manifest_paths.values())
    new_selection = draw_splits(source_dir, items_per_domain_by_split, args.seed) if needs_sampling else {}

    for split_name, items_per_domain in items_per_domain_by_split.items():
        manifest = load_or_create_manifest(
            manifest_paths[split_name], split_name, items_per_domain, new_selection, args
        )
        split_dir = data_root / split_name
        counts = materialize_split(source_dir, split_dir, manifest["capabilities"])
        total_items = 0
        for capability, (domain_count, item_count) in counts.items():
            print(f"  {capability:35} domains={domain_count:3}  items={item_count:5}")
            total_items += item_count
        print(f"  {'TOTAL':35} {'':13}items={total_items:5}  -> {split_dir.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
