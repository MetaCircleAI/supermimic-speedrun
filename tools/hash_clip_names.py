"""Replace clip names with their SHA256 in a per-clip evaluation file before publishing it.

    python3 tools/hash_clip_names.py runs/r1600_env1/eval/001600.json

The BONES-SEED license treats the dataset's metadata, including clip names, as confidential.
Each clip is keyed by hashlib.sha256(name.encode()).hexdigest(); scores and flags are unchanged.
The file is rewritten in place.
"""
import hashlib
import json
import math
import sys
from pathlib import Path


def key(name):
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def main(path):
    path = Path(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("clip_key"):
        raise SystemExit(f"{path}: clip names are already hashed")
    per_clip = {key(name): value for name, value in data["per_clip"].items()}
    if len(per_clip) != len(data["per_clip"]):
        raise SystemExit("hash collision")
    mean = math.fsum(v["norm_return"] for v in per_clip.values()) / len(per_clip)
    out = {}
    for field, value in data.items():
        if field == "per_clip":
            out["clip_key"] = "sha256 of the UTF-8 clip name"
            out["per_clip"] = {h: per_clip[h] for h in sorted(per_clip)}
        elif field == "failed":
            out["failed"] = sorted(key(name) for name in value)
        else:
            out[field] = value
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"{path}: {len(per_clip)} clips hashed, {len(out['failed'])} failed, mean={mean:.10f}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    main(sys.argv[1])
