#!/usr/bin/env bash
# Train the current code in train/ (recipe in train/recipe.json) on one H100, then evaluate the
# final checkpoint on the full library.
#   ./run.sh /path/to/Uni-SuperMimic /path/to/bones_seed_compact.pt /path/to/new-run [gpu]
set -euo pipefail
if [ $# -lt 3 ]; then
  sed -n '2,4p' "$0"; exit 1
fi
cd "$(dirname "$0")"
exec python tools/run_record.py --source "$1" --motion "$2" --output "$3" --gpu "${4:-0}"
