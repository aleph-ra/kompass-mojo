# kompass-mojo

A Mojo port of the GPU compute kernels from [kompass-core](https://github.com/automatika-robotics/kompass-core), built to evaluate whether Mojo can match the cross-GPU performance of the existing SYCL (AdaptiveCpp) implementation on NVIDIA Jetson and AMD Strix Halo hardware and become a viable contender for writing the EMOS GPGPU navigation stack.

Context: [EMOS Robotics: Mojo for Robotics — Porting GPU Navigation Kernels to Jetson & Strix Halo](https://forum.modular.com/t/project-mojo-for-robotics-porting-gpu-navigation-kernels-jetson-strix-halo/2952)

## Requirements

- **Mojo-supported GPU** — NVIDIA sm_52+ (Ampere, Ada, Hopper, Blackwell, Jetson Orin `sm_87`, Jetson Thor `sm_110`), AMD gfx942/950/1030/1100-1103/1150-1152 (Strix Halo `gfx1151`)/1200/1201, Apple Metal 1-4
- **pixi** — environment manager (`curl -fsSL https://pixi.sh/install.sh | bash`)
- A C++17 compiler (gcc/clang)

## Quick start

```bash
git clone https://github.com/aleph-ra/kompass-mojo
cd kompass-mojo
pixi install
pixi run benchmark
```

Available pixi tasks:

| Task                           | What it does                                                                                |
| ------------------------------ | ------------------------------------------------------------------------------------------- |
| `pixi run build`               | Compiles `src/kompass_mojo/ffi.mojo` into `build/libkompass_mojo.so`                        |
| `pixi run build-harness`       | Builds the C++ benchmark runner (depends on `build`)                                        |
| `pixi run benchmark`           | Runs the full benchmark pipeline and writes JSON to `results/` (depends on `build-harness`) |
| `pixi run test`                | Runs every correctness test suite below in order                                            |
| `pixi run test-cost-evaluator` | Per-kernel correctness tests for the cost evaluator                                         |
| `pixi run test-mapper`         | Per-kernel correctness tests for the local mapper (incl. an ASCII grid render)              |
| `pixi run test-critical-zone`  | Per-kernel correctness tests for the critical-zone checker                                  |

To specify a platform name for the output JSON:

```bash
PLATFORM=rtx_a5000 pixi run benchmark
# writes results/rtx_a5000.json
```

## Ported kernels

kompass-core's GPGPU stack has three kernel groups and all three are ported here. Each Mojo kernel mirrors the math, memory layout, and work-group structure of its SYCL counterpart.

| Group                 | Mojo source                            | SYCL source (kompass-core)                                          | What it does                                                                                                                                                                                                                                             |
| --------------------- | -------------------------------------- | ------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Cost evaluator        | `src/kompass_mojo/cost_evaluator.mojo` | `src/kompass_cpp/kompass_cpp/src/utils/cost_evaluator_gpu.cpp`      | 6 kernels + 2-pass min reduction scoring candidate trajectories against a reference path, a goal, smoothness / jerk penalties, and obstacle distance. Outputs the lowest-cost trajectory and its index.                                                  |
| Local mapper          | `src/kompass_mojo/local_mapper.mojo`   | `src/kompass_cpp/kompass_cpp/src/mapping/local_mapper_gpu.cpp`      | Per-ray super-cover Bresenham kernel projecting a 2D laserscan into an occupancy grid.                                                                                                                                                                   |
| Critical-zone checker | `src/kompass_mojo/critical_zone.mojo`  | `src/kompass_cpp/kompass_cpp/src/utils/critical_zone_check_gpu.cpp` | Two kernels producing a single safety factor in `[0, 1]` (1.0 = safe, 0.0 = stop). One takes pre-computed laserscan ranges via a sparse cone-index LUT; the other takes raw PointCloud2 bytes and filters + transforms + cone-tests per point on-device. |

Each group is exposed through C FFI (`integration/kompass_mojo.h`) and exercised by both the unit tests and the benchmark harness.

## Results format

Output JSON follows the exact schema used by [kompass-core's benchmark suite](https://github.com/automatika-robotics/kompass-core/tree/main/src/kompass_cpp/benchmarks):

```json
{
    "platform": "rtx_a5000",
    "timestamp": 1775999999,
    "benchmarks": [
        { "test_name": "CostEvaluator_5k_Trajs",   "mean_ms": 16.28, "std_dev_ms": 0.19, ... },
        { "test_name": "Mapper_Dense_400x400",     "mean_ms":  0.30, "std_dev_ms": 0.01, ... },
        { "test_name": "CriticalZone_Dense_Scan",  "mean_ms":  0.03, "std_dev_ms": 0.00, ... },
        { "test_name": "CriticalZone_100k_Cloud",  "mean_ms":  0.33, "std_dev_ms": 0.01, ... }
    ]
}
```

Benchmark names match kompass-core's suite exactly, so you can drop this file alongside any of kompass-core's `benchmark_files/*.json` and run their `plot_benchmarks.py` — Mojo and SYCL bars for the same kernel appear side-by-side on the same chart.

## Reference performance

### RTX A5000 (NVIDIA Ampere `sm_86`)

| Benchmark                 | kompass-core (SYCL) | kompass-mojo (Mojo nightly) |
| ------------------------- | ------------------- | --------------------------- |
| `CostEvaluator_5k_Trajs`  | 16.358 ms (±0.12)   | 15.973 ms (±0.09)           |
| `Mapper_Dense_400x400`    | 0.175 ms            | 0.297 ms                    |
| `CriticalZone_Dense_Scan` | 0.146 ms            | 0.026 ms                    |
| `CriticalZone_100k_Cloud` | 0.519 ms            | 0.331 ms                    |

## Relationship to kompass-core

kompass-core implements its GPGPU path in SYCL via AdaptiveCpp; this repo implements the same kernels in Mojo. Each port is deliberately line-by-line with its SYCL counterpart (work-group shape, memory layout, reduction pattern) so the two binaries are comparable across GPU vendors without a compiler layer in between.

kompass-core is **not** a build dependency here. The two projects are complementary: run each one's benchmark separately and overlay the JSON outputs via kompass-core's `plot_benchmarks.py`.

## License

TBD.
