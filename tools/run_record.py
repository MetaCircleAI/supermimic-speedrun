"""Train and evaluate a SuperMimic Speedrun run on one GPU.

    python tools/run_record.py --verify                  # check the archived record sources
    python tools/run_record.py --source UNI --motion LIB --output OUT            # current code in train/
    python tools/run_record.py --record 2026-11-26_joint_muon --source UNI --motion LIB --output OUT

Without --record the launcher runs the working copy in train/ with the recipe in
train/recipe.json. With --record it runs a record's frozen snapshot after checking its hashes.
Training is timed; the final checkpoint is then evaluated on the full library (untimed).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
DATA_ID = "5d815fd8039c4e4938a6c049935b8d712f81f483116ce69518c8a8298290017a"
UPSTREAM = "d78a5e2bc26ee26c3a94ae10418b7fb9018836ee"
RECORDS = {
    "2026-11-26_joint_muon": dict(
        code_id="dc1cb6a9ccfe4230ab7bc71c201ee7ccbc3ecae67a4f165179f4d831252f5faa",
        helper="pipeline.py",
        helper_id="84b5f4b78b9f21ec47a71cbc928fd971ecf303580514b63bf773bc93c4ca4f01",
        run="r1600_env1", iterations=1600,
        plan=dict(profile="speedrun_extra_20261126_joint_lr12e4_record",
                  overrides={"algo.pi_opt": "muon", "algo.wm_opt": "muon", "algo.lr": 0.0012,
                             "algo.ckpt_every": 1000})),
    "2026-11-09_joint_adam": dict(
        code_id="7458df4073505c1565dc545978c156f44aa5ee62dd76c5a7153268175d986f6a",
        helper="seeded_pipeline.py",
        helper_id="032e3699e7756c18c2b3a37461e4e1dc478eb59a6b8e1c5c5eea9d153b9195b5",
        run="r2600_env1", iterations=2600, plan=None),
}


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def tree_id(folder):
    files = {str(p.relative_to(folder)): digest(p) for p in folder.rglob("*")
             if p.is_file() and "__pycache__" not in p.parts}
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(), len(files)


def verify(name):
    spec = RECORDS[name]
    record = ROOT / "records" / name
    code_id, count = tree_id(record / "snapshot")
    if code_id != spec["code_id"] or digest(record / spec["helper"]) != spec["helper_id"]:
        raise SystemExit(f"{name}: archived record source checksum mismatch")
    print(f"{name}: verified {count} public source files and the launcher.", flush=True)


def resolve_target(args):
    """Return code directory, spec file, helper script, run name, iterations and plan fields."""
    if args.record:
        verify(args.record)
        spec = RECORDS[args.record]
        record = ROOT / "records" / args.record
        snapshot = record / "snapshot"
        return (snapshot / "code", snapshot / "candidate_spec.json", record / spec["helper"],
                spec["run"], spec["iterations"], spec["code_id"], spec["plan"])
    train = ROOT / "train"
    recipe = json.loads((train / "recipe.json").read_text())
    iterations = args.iterations or recipe["iterations"]
    code_id, _ = tree_id(train)
    plan = dict(profile=recipe["profile"], overrides=recipe["overrides"])
    return (train, train / "candidate_spec.json", train / "pipeline.py",
            f"r{iterations}_env1", iterations, code_id, plan)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--record", choices=sorted(RECORDS), help="frozen record to reproduce (default: train/)")
    parser.add_argument("--verify", action="store_true", help="check the record source archives and exit")
    parser.add_argument("--iterations", type=int, help="override the iteration budget of train/recipe.json")
    parser.add_argument("--source", type=Path, help="installed, pinned Uni-SuperMimic checkout")
    parser.add_argument("--motion", type=Path, help="processed BONES-SEED library")
    parser.add_argument("--output", type=Path, help="new directory for this run")
    parser.add_argument("--gpu", type=int, default=0, help="physical CUDA device index")
    args = parser.parse_args()
    if args.verify:
        for name in RECORDS:
            verify(name)
        return
    if args.record and args.iterations:
        parser.error("--iterations applies to train/ only; a record's budget is fixed")
    code, candidate, helper, run, iterations, code_id, plan_fields = resolve_target(args)
    if not (args.source and args.motion and args.output) or args.gpu < 0:
        parser.error("provide --source, --motion, --output and a nonnegative GPU index")
    source, motion, out = (p.resolve() for p in (args.source, args.motion, args.output))
    if subprocess.check_output(["git", "-C", source, "rev-parse", "HEAD"], text=True).strip() != UPSTREAM:
        parser.error("Uni-SuperMimic revision does not match the pinned revision")
    if digest(motion) != DATA_ID:
        parser.error("processed motion library checksum does not match the pinned library")
    out.mkdir(parents=True, exist_ok=False)
    (out / "source").symlink_to(source, target_is_directory=True)
    (out / "data").mkdir()
    (out / "data/bones_seed_compact.pt").symlink_to(motion)
    shutil.copyfile(candidate, out / "candidate_spec.json")
    plan = {"code_sha256": code_id, "data_sha256": DATA_ID}
    if plan_fields:
        plan.update(plan_fields, seed={"algo": 1, "env": 1})
    plan.update(run=run, iterations=iterations)
    (out / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    os.chdir(out)
    os.environ["SM_CAMP"] = str(out)
    sys.path.insert(0, str(code))
    import common
    common.ROOT = common.CAMP = out
    common.CODE = code
    common.PY = sys.executable
    original_env = common.environment
    cache = os.environ.get("TORCHINDUCTOR_CACHE_DIR", str(out / "inductor-cache"))

    def environment(gpu, run_dir):
        return {**original_env(gpu, run_dir), "TORCHINDUCTOR_CACHE_DIR": cache}

    common.environment = environment
    print(f"Training {run} for {iterations} iterations from {code.relative_to(ROOT)}", flush=True)
    sys.argv = [helper.name, run, str(args.gpu), str(iterations)]
    runpy.run_path(str(helper), run_name="__main__")


if __name__ == "__main__":
    main()
