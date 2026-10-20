#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

gpu=false
if [[ $# -eq 1 && $1 == "--gpu" ]]; then
  gpu=true
elif [[ $# -ne 0 ]]; then
  echo "usage: $0 [--gpu]" >&2
  exit 2
fi

target=${DAGGERTRACE_TARGET:-liger-rmsnorm}
output=${DAGGERTRACE_OUTPUT_DIR:-artifacts/final}

PYTHONPYCACHEPREFIX="${DAGGERTRACE_PYCACHE_DIR:-${TMPDIR:-/tmp}/daggertrace-validation-pycache}" \
  python -m compileall -q src/daggertrace evidence/holdout \
  evidence/*.py tests/*.py
python scripts/check_architecture.py
python -m ruff check src evidence tests --exclude tests/fixtures/external,evidence/results
python -m pytest -q -m "not gpu and not package"
python -m daggertrace validate

checked_packets=(
  evidence/results/bundle/liger/liger-rmsnorm
  evidence/results/bundle/art/openpipe-art-local-stats
)
for packet in "${checked_packets[@]}"; do
  if [[ ! -d "$packet" ]]; then
    echo "daggertrace: checked replay packet is missing: $packet" >&2
    exit 1
  fi
  python -B -m daggertrace replay "$packet" --json >/dev/null
done

if $gpu; then
  python -m pytest -q -m gpu \
    tests/test_liger_gpu.py tests/test_art_gpu.py
  args=(audit "$target" --gpu --output "$output")
  if [[ -n ${DAGGERTRACE_TARGET_ROOT:-} ]]; then
    args+=(--target-root "$DAGGERTRACE_TARGET_ROOT")
  fi
  python -m daggertrace "${args[@]}"
  python -m daggertrace replay "$output"
  if [[ $target == all ]]; then
    for reproducer in "$output"/*/reproducer.py; do
      [[ -f $reproducer ]] || continue
      python -B "$reproducer" >/dev/null
    done
  elif [[ -f "$output/reproducer.py" ]]; then
    python -B "$output/reproducer.py" >/dev/null
  elif [[ -f "$output/$target/reproducer.py" ]]; then
    python -B "$output/$target/reproducer.py" >/dev/null
  fi
else
  if [[ ! -e "$output" ]]; then
    echo "daggertrace: replay skipped: no replay artifact at $output"
  else
    packet=$(find "$output" -type f \( -name certificate.json -o -name source-evidence.json \) -print -quit)
    if [[ -n "$packet" ]]; then
      python -m daggertrace replay "$output"
    else
      echo "daggertrace: replay skipped: no certificate.json or source-evidence.json found under $output"
    fi
  fi
fi
