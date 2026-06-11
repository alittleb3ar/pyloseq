#!/usr/bin/env bash
# Verifies that pyloseq pip-installs cleanly on common base images.
# Usage: bash scripts/test_container_install.sh

set -uo pipefail

IMAGES=(
  "python:3.10-slim"
  "python:3.11-slim"
  "python:3.12-slim"
  "python:3.13-slim"
  "continuumio/miniconda3"
  "jupyter/scipy-notebook:latest"
)

SMOKE="import pyloseq; from pyloseq import Phyloseq, OtuTable; print('pyloseq', pyloseq.__version__, 'OK')"

pass=0
fail=0

for image in "${IMAGES[@]}"; do
  printf "%-42s" "$image"
  if output=$(docker run --rm "$image" sh -c "pip install -q pyloseq && python -c \"$SMOKE\"" 2>&1); then
    echo "PASS  — $output"
    ((pass++)) || true
  else
    echo "FAIL"
    echo "$output" | tail -5 | sed 's/^/    /'
    ((fail++)) || true
  fi
done

echo ""
echo "Results: $pass passed, $fail failed"
[[ $fail -eq 0 ]]
