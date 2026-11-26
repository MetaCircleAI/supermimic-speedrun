"""Check the published records.

    python3 tools/verify_records.py          # timer, seeds, evaluation scope, threshold
    python3 tools/verify_records.py --full   # also file hashes and clip-averaged scores (requires jq)
"""
import argparse
import hashlib
import json
import math
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
TARGET = 0.9363
RECORDS = [("2026-11-26_joint_muon", 1600), ("2026-11-09_joint_adam", 2600)]
BASELINE = "2026-11-09_mlp_baseline"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def verify(name, steps):
    folder = ROOT / "records" / name
    record = read(folder / "RECORD.json")
    train = read(folder / "original/training_complete.json")
    evaluation = read(folder / f"original/eval/{steps:06d}.summary.json")
    score = evaluation["norm_return"]
    seconds = train["speedrun_seconds"]
    checks = {
        "completed run": record["evaluation_returncode"] == 0 and train["status"] == "completed",
        "final iteration": steps == train["target_iterations"] == train["iteration"]
        == evaluation["checkpoint_iteration"] == record["iterations"],
        "training seeds": record["algo_seed"] == record["env_seed"] == 1,
        "evaluation": evaluation["evaluation_scope"] == "full_library"
        and evaluation["n_clips"] == 129487 and evaluation["eval_seed"] == 0
        and evaluation["evaluator_version"] == "infra_v3_terminal_bin_fix",
        "score": score == record["norm_return"] and evaluation["n_failed"] == record["n_failed"],
        "time": seconds == record["speedrun_seconds"] and seconds > train["prefill_seconds"] > 0,
        "qualification": record["target"] == TARGET and record["passes_target"] is (score > TARGET),
    }
    for check, passed in checks.items():
        if not passed:
            raise ValueError(f"{name}: inconsistent {check}")
    print(f"{name}: {seconds:.3f} s ({seconds / 60:.2f} min), norm_return={score:.10f}, qualifies={score > TARGET}")


def check_manifest(folder):
    """Verify every file listed in folder/SHA256SUMS; return {name: sha256}."""
    hashes = {}
    for line in (folder / "SHA256SUMS").read_text(encoding="ascii").splitlines():
        expected, name = line.split(maxsplit=1)
        path = (folder / name).resolve()
        if folder.resolve() not in path.parents:
            raise ValueError("Invalid checksum path")
        if sha256(path) != expected:
            raise ValueError(f"{folder}: checksum mismatch: {name}")
        hashes[name] = expected
    return hashes


def clip_mean(per_clip_file, summary):
    """Recompute the clip-averaged return from a per-clip file and check it against the summary."""
    scores, failures, failed_list, flags = {}, set(), set(), set()
    select = ('select(length == 2 and ((.[0][0] == "per_clip" and '
              '(.[0][2] == "norm_return" or .[0][2] == "failed")) or .[0][0] == "failed"))')
    with subprocess.Popen(["jq", "--stream", "-c", select, str(per_clip_file)],
                          stdout=subprocess.PIPE, text=True) as process:
        for line in process.stdout:
            key, value = json.loads(line)
            if key[0] == "failed":
                failed_list.add(value)
            elif key[2] == "failed":
                if key[1] in flags:
                    raise ValueError("Duplicate clip failure flag")
                flags.add(key[1])
                if value:
                    failures.add(key[1])
            else:
                if key[1] in scores or not math.isfinite(value):
                    raise ValueError("Duplicate or non-finite clip score")
                scores[key[1]] = value
        if process.wait():
            raise RuntimeError("Could not read the per-clip evaluation")
    if len(scores) != 129487 or scores.keys() != flags:
        raise ValueError(f"{per_clip_file}: incomplete clip scores or flags")
    if failures != failed_list or len(failures) != summary["n_failed"]:
        raise ValueError(f"{per_clip_file}: failed clips do not agree with the evaluation summary")
    score = math.fsum(scores.values()) / len(scores)
    # The evaluator reduces float32 scores on the GPU; the local mean uses double precision.
    if abs(score - summary["norm_return"]) > 1e-6:
        raise ValueError(f"{per_clip_file}: recomputed normalized return disagrees with the summary")
    return score, len(scores), len(failures)


def verify_full_record(name, steps):
    folder = ROOT / "records" / name
    archive = folder / "original"
    hashes = check_manifest(archive)
    summary = read(archive / f"eval/{steps:06d}.summary.json")
    if hashes.get(f"ckpt_{steps:06d}.pt") != summary["checkpoint_sha256"]:
        raise ValueError(f"{name}: final weights differ from the evaluated checkpoint")
    config = read(archive / "run_config.json")["config"]
    if (config["algo"]["seed"], config["env"]["seed"], config["algo"]["iterations"]) != (1, 1, steps):
        raise ValueError(f"{name}: resolved configuration differs from the fixed-seed record")
    score, clips, failed = clip_mean(archive / f"eval/{steps:06d}.json", summary)
    if score <= TARGET:
        raise ValueError(f"{name}: recomputed normalized return is not above the target")
    print(f"{name}: verified final weights, {clips} distinct clips, {failed} failed, mean={score:.10f}")


def verify_baseline():
    folder = ROOT / "records" / BASELINE
    check_manifest(folder)
    crossing = read(folder / "CROSSING.json")
    step = crossing["first_full_library_checkpoint_above_target"]
    summary = read(folder / f"eval/{step:06d}.summary.json")
    score, clips, failed = clip_mean(folder / f"eval/{step:06d}.json", summary)
    if summary["norm_return"] != crossing["norm_return"] or not score > TARGET:
        raise ValueError(f"{BASELINE}: crossing does not match the evaluation")
    print(f"{BASELINE}: iteration {step}, {crossing['learner_minutes_at_6400']:.2f} learner min, "
          f"{clips} distinct clips, {failed} failed, mean={score:.10f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--full", action="store_true",
                        help="also verify archived files and recompute scores from per-clip results (requires jq)")
    args = parser.parse_args()
    for record, steps in RECORDS:
        verify(record, steps)
    if args.full:
        for record, steps in RECORDS:
            verify_full_record(record, steps)
        verify_baseline()
