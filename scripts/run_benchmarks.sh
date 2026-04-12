#!/usr/bin/env bash
# End-to-end driver: build the Mojo shared library, build the C++
# benchmark runner, run it against the given platform alias.
#
# Usage: ./scripts/run_benchmarks.sh <platform_name> [output_json_path]
#
# Default output path: results/<platform_name>_mojo.json
#
# Requires:
#   - pixi (will be used to invoke mojo build)
#   - cmake, a C++17 compiler
#   - An NVIDIA or AMD GPU with Mojo-supported compute capability
#     (sm_52+ for NVIDIA, gfx* for AMD)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <platform_name> [output_json_path]" >&2
    echo "  example: $0 rtx_a5000 results/rtx_a5000.json" >&2
    exit 1
fi

PLATFORM="$1"
OUTPUT="${2:-${REPO_ROOT}/results/${PLATFORM}.json}"
BUILD_DIR="${REPO_ROOT}/build"
INTEGRATION_BUILD_DIR="${REPO_ROOT}/integration/build"

mkdir -p "$(dirname "${OUTPUT}")"

# Make sure pixi is on PATH
export PATH="${HOME}/.pixi/bin:${PATH}"
if ! command -v pixi >/dev/null 2>&1; then
    echo "[run] ERROR: pixi not found on PATH" >&2
    echo "[run] install it: curl -fsSL https://pixi.sh/install.sh | bash" >&2
    exit 1
fi

# Build Mojo shared library.
echo "[run] step 1/3: building libkompass_mojo.so"
"${REPO_ROOT}/scripts/build_shared_lib.sh"

# Build C++ benchmark runner. The CMake config bakes rpath entries
# pointing at build/ and .pixi/envs/default/lib so the resulting binary
# doesn't need LD_LIBRARY_PATH.
echo "[run] step 2/3: building kompass_benchmark_mojo"
cmake -S "${REPO_ROOT}/integration" -B "${INTEGRATION_BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE=Release \
    ${ENABLE_POWER_MONITOR:+-DENABLE_POWER_MONITOR=ON}
cmake --build "${INTEGRATION_BUILD_DIR}" --parallel

# Run.
echo "[run] step 3/3: running benchmark"
"${INTEGRATION_BUILD_DIR}/kompass_benchmark_mojo" "${PLATFORM}" "${OUTPUT}"

echo "[run] results written to ${OUTPUT}"
