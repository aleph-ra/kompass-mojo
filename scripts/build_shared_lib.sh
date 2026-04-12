#!/usr/bin/env bash
# Build libkompass_mojo.so from src/kompass_mojo/ffi.mojo.
#
# Requires a Mojo-supported GPU on this machine (NVIDIA sm_52+, AMD gfx*,
# Apple Metal 1-4).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/build"
SRC="${REPO_ROOT}/src/kompass_mojo/ffi.mojo"
OUT="${BUILD_DIR}/libkompass_mojo.so"

mkdir -p "${BUILD_DIR}"

# Pick up pixi from the usual install location if it isn't already on PATH.
export PATH="${HOME}/.pixi/bin:${PATH}"
if ! command -v pixi >/dev/null 2>&1; then
    echo "[build] ERROR: pixi not found on PATH" >&2
    echo "[build] install it: curl -fsSL https://pixi.sh/install.sh | bash" >&2
    exit 1
fi

echo "[build] ffi.mojo -> libkompass_mojo.so"
cd "${REPO_ROOT}"
pixi run mojo build -I src --emit shared-lib -o "${OUT}" "${SRC}"

echo "[build] verifying exports"
if command -v nm >/dev/null 2>&1; then
    nm -D --defined-only "${OUT}" | grep -E 'mojo_cost_eval_' || {
        echo "[build] WARNING: expected exports mojo_cost_eval_{create,run,destroy} not found" >&2
    }
fi

echo "[build] done: ${OUT}"
