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

### Jetson Orin (JetPack 6.x — driver 540.x)

Mojo 1.0+ requires NVIDIA driver 580+ / CUDA 13.x. JetPack 6.x ships driver 540.x and CUDA 12.6, so out of the box `pixi run` will hit `MAX requires a minimum driver version of 580` or `CUDA_ERROR_INVALID_IMAGE`. Install NVIDIA's `cuda-compat-orin-13-2` forward-compatibility package (no JetPack reflash, no kernel-driver upgrade) and put its libs at the front of `LD_LIBRARY_PATH`:

```bash
# 1. Download + extract (no root needed)
mkdir -p ~/cuda-compat-orin && cd ~/cuda-compat-orin
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/cuda-compat-orin-13-2_13.2.44290101-1_arm64.deb \
     -O cuda-compat-orin-13-2.deb
dpkg-deb -x cuda-compat-orin-13-2.deb extracted/

# 2. Prepend the compat libs to LD_LIBRARY_PATH (persist in ~/.bashrc)
echo 'export LD_LIBRARY_PATH=$HOME/cuda-compat-orin/extracted/usr/local/cuda-13.2/compat_orin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}' >> ~/.bashrc
source ~/.bashrc

# 3. Build + run normally
cd ~/automatika/kompass-mojo
pixi install
pixi run test
pixi run benchmark
```

System-wide install (with `sudo apt install cuda-compat-orin-13-2` after adding NVIDIA's sbsa repo) is also possible — see [NVIDIA's forum thread](https://forums.developer.nvidia.com/t/test-run-public-wheels-cuda-13-2-jetson-orin-family/364497) for the exact apt commands. Once JetPack ships a driver ≥ 580 (likely JetPack 7.2 for Orin AGX), this step becomes unnecessary.

## Ported kernels

kompass-core's GPGPU stack has three kernel groups and all three are ported here. Each Mojo kernel mirrors the math, memory layout, and work-group structure of its SYCL counterpart.

| Group                 | Mojo source                            | SYCL source (kompass-core)                                          | What it does                                                                                                                                                                                                                                             |
| --------------------- | -------------------------------------- | ------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Cost evaluator        | `src/kompass_mojo/cost_evaluator.mojo` | `src/kompass_cpp/kompass_cpp/src/utils/cost_evaluator_gpu.cpp`      | 6 kernels + 2-pass min reduction scoring candidate trajectories against a reference path, a goal, smoothness / jerk penalties, and obstacle distance. Outputs the lowest-cost trajectory and its index.                                                  |
| Local mapper          | `src/kompass_mojo/local_mapper.mojo`   | `src/kompass_cpp/kompass_cpp/src/mapping/local_mapper_gpu.cpp`      | Two kernels: (1) PointCloud2 → per-bin laserscan (one thread per point, atomic-min into the angular bin); (2) per-ray super-cover Bresenham projecting the scan into an occupancy grid. The cloud→scan stage is skipped on the laserscan-input path.    |
| Critical-zone checker | `src/kompass_mojo/critical_zone.mojo`  | `src/kompass_cpp/kompass_cpp/src/utils/critical_zone_check_gpu.cpp` | Two kernels producing a single safety factor in `[0, 1]` (1.0 = safe, 0.0 = stop). One takes pre-computed laserscan ranges via a sparse cone-index LUT; the other takes raw PointCloud2 bytes and filters + transforms + cone-tests per point on-device. |

Each group is exposed through C FFI (`integration/kompass_mojo.h`) and exercised by both the unit tests and the benchmark harness.

## Results format

Output JSON follows the exact schema used by [kompass-core's benchmark suite](https://github.com/automatika-robotics/kompass-core/tree/main/src/kompass_cpp/benchmarks):

```json
{
    "platform": "rtx_a5000",
    "timestamp": 1775999999,
    "benchmarks": [
        { "test_name": "CostEvaluator_5k_Trajs",   "mean_ms": 16.06, "std_dev_ms": 0.18, ... },
        { "test_name": "Mapper_Dense_400x400",     "mean_ms":  0.30, "std_dev_ms": 0.01, ... },
        { "test_name": "Mapper_PointCloud_100k",   "mean_ms":  0.57, "std_dev_ms": 0.01, ... },
        { "test_name": "CriticalZone_Dense_Scan",  "mean_ms":  0.02, "std_dev_ms": 0.00, ... },
        { "test_name": "CriticalZone_100k_Cloud",  "mean_ms":  0.32, "std_dev_ms": 0.00, ... }
    ]
}
```

Benchmark names match kompass-core's suite exactly, so you can drop this file alongside any of kompass-core's `benchmark_files/*.json` and run their `plot_benchmarks.py` — Mojo and SYCL bars for the same kernel appear side-by-side on the same chart.

## Reference performance

### RTX A5000 (NVIDIA Ampere `sm_86`)

| Benchmark                 | kompass-core (SYCL) | kompass-mojo (Mojo nightly) |
| ------------------------- | ------------------- | --------------------------- |
| `CostEvaluator_5k_Trajs`  | 16.63 ms            | 16.06 ms                    |
| `Mapper_Dense_400x400`    | 0.225 ms            | 0.298 ms                    |
| `Mapper_PointCloud_100k`  | 0.574 ms            | 0.567 ms                    |
| `CriticalZone_Dense_Scan` | 0.176 ms            | 0.022 ms                    |
| `CriticalZone_100k_Cloud` | 0.507 ms            | 0.319 ms                    |

### Jetson Orin AGX (NVIDIA Ampere `sm_87`, normal power profile)

| Benchmark                 | kompass-core (SYCL) | kompass-mojo (Mojo nightly) |
| ------------------------- | ------------------- | --------------------------- |
| `CostEvaluator_5k_Trajs`  | 36.10 ms            | 43.67 ms                    |
| `Mapper_Dense_400x400`    | 1.123 ms            | 0.715 ms                    |
| `Mapper_PointCloud_100k`  | 1.737 ms            | 1.041 ms                    |
| `CriticalZone_Dense_Scan` | 0.171 ms            | 0.120 ms                    |
| `CriticalZone_100k_Cloud` | 0.690 ms            | 0.373 ms                    |

### Jetson Orin AGX (NVIDIA Ampere `sm_87`, max power profile / 50 W)

| Benchmark                 | kompass-core (SYCL) | kompass-mojo (Mojo nightly) |
| ------------------------- | ------------------- | --------------------------- |
| `CostEvaluator_5k_Trajs`  | 42.77 ms            | 38.96 ms                    |
| `Mapper_Dense_400x400`    | 0.667 ms            | 0.620 ms                    |
| `Mapper_PointCloud_100k`  | 0.755 ms            | 0.975 ms                    |
| `CriticalZone_Dense_Scan` | 0.167 ms            | 0.105 ms                    |
| `CriticalZone_100k_Cloud` | 0.790 ms            | 0.441 ms                    |

Mojo runtime on Jetson is fed via the `cuda-compat-orin-13-2` forward-compatibility package because JetPack 6.x ships driver 540.x while Mojo 1.0+ requires driver 580+.

Numbers are mean over 50 iterations.

## Relationship to kompass-core

kompass-core implements its GPGPU path in SYCL via AdaptiveCpp; this repo implements the same kernels in Mojo. Each port is deliberately line-by-line with its SYCL counterpart (work-group shape, memory layout, reduction pattern) so the two binaries are comparable across GPU vendors without a compiler layer in between.

kompass-core is **not** a build dependency here. The two projects are complementary: run each one's benchmark separately and overlay the JSON outputs via kompass-core's `plot_benchmarks.py`.

## License

TBD.
