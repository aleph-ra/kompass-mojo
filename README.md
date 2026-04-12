# kompass-mojo

A Mojo port of the GPU compute kernels from [kompass-core](https://github.com/automatika-robotics/kompass-core), built to evaluate whether Mojo can match the cross-GPU performance of the existing SYCL (AdaptiveCpp) implementation on NVIDIA Jetson and AMD Strix Halo hardware and become a viable contender for writing the EMOS GPGPU navigation stack.

Context: [EMOS Robotics: Mojo for Robotics — Porting GPU Navigation Kernels to Jetson & Strix Halo](https://forum.modular.com/t/project-mojo-for-robotics-porting-gpu-navigation-kernels-jetson-strix-halo/2952)
## Requirements

- **Mojo-supported GPU** — NVIDIA sm_52+ (Ampere, Ada, Hopper, Blackwell, Jetson Orin `sm_87`, Jetson Thor `sm_110`), AMD gfx942/950/1030/1100-1103/1150-1152 (Strix Halo `gfx1151`)/1200/1201, Apple Metal 1-4
- **pixi** — environment manager (`curl -fsSL https://pixi.sh/install.sh | bash`)
- **cmake** ≥ 3.16 and a C++17 compiler (gcc/clang)

## Quick start

```bash
# 1. Clone and enter the repo
git clone https://github.com/automatika-robotics/kompass-mojo
cd kompass-mojo

# 2. Install Mojo and its dependencies into a local pixi environment
pixi install

# 3. Build and run the benchmark end-to-end
./scripts/run_benchmarks.sh <platform_name e.g. rtx_a5000>
```

`run_benchmarks.sh`:
1. Compiles `src/kompass_mojo/ffi.mojo` into `build/libkompass_mojo.so` (kernels + C ABI entry points)
2. Configures and builds `integration/kompass_benchmark_mojo` against the shared library
3. Runs it with 5 warmup + 50 timed iterations of the workload
4. Writes a JSON to `results/<platform>.json` matching kompass-core's benchmark schema

The binary has rpath entries pointing at `build/` and the pixi Mojo runtime so it works without `LD_LIBRARY_PATH`.

## Results format

Output JSON follows the exact schema used by [kompass-core's benchmark suite](https://github.com/automatika-robotics/kompass-core/tree/main/src/kompass_cpp/benchmarks):

```json
{
    "platform": "rtx_a5000",
    "timestamp": 1775999999,
    "benchmarks": [
        {
            "test_name": "CostEvaluator_5k_Trajs",
            "mean_ms": 15.97,
            "std_dev_ms": 0.09,
            "min_ms": 15.85,
            "max_ms": 16.27,
            "iterations": 50
        }
    ]
}
```

You can drop this file alongside any of kompass-core's `benchmark_files/*.json` and run their `plot_benchmarks.py` — both show up as bars on the same chart.

## Reference performance (RTX A5000)

First measured on an NVIDIA RTX A5000 (Ampere, 24 GB VRAM):

| Implementation | Backend | CostEvaluator_5k_Trajs |
|---|---|---|
| kompass-core SYCL (AdaptiveCpp / CUDA) | `sm_86` | 16.358 ms (±0.12) |
| kompass-mojo (Mojo 0.26.1 / CUDA) | `sm_86` | 15.973 ms (±0.09) |

Measurement protocol matches kompass-core's benchmark harness: 5 warmup iterations, 50 timed iterations, `std::chrono::high_resolution_clock`, identical synthetic trajectory generator.

Numerical parity against kompass-core's CostEvaluator was verified bit-exact during Phase 1e by linking libkompass_mojo.so directly into kompass-core's benchmark binary and feeding both code paths the same in-memory `TrajectorySamples2D` — both produce `min_cost = 0.00100002` at `min_idx = 0`. The parity harness is not shipped in this repo (it would require kompass-core as a build dep); the standalone benchmark uses an in-process synthetic generator that produces self-consistent results.

## Relationship to kompass-core

The kernels in this repo are Mojo ports of the GPU compute kernels in [kompass-core](https://github.com/automatika-robotics/kompass-core), which implements its GPGPU path in SYCL via AdaptiveCpp. Each Mojo kernel mirrors the math, memory layout, and work-group structure of its SYCL counterpart.

kompass-core is **not** a build dependency of this repo. The two projects are complementary: run each one's benchmark separately and compare JSON outputs via kompass-core's benchmark plotting script. Subsequenct work will port additional kernel groups under the same pattern.

## License

TBD.
