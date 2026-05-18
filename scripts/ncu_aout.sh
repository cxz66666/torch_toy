#!/usr/bin/env bash
set -euo pipefail

output_prefix="${1:-ncu_report%i}"
target="${2:-./a.out}"

/usr/local/cuda/bin/ncu \
    --print-source cuda,sass \
    --import-source 1 \
    --page source \
    -o "${output_prefix}" \
    --set full \
    "${target}"
