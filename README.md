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

| Task | What it does |
|---|---|
| `pixi run build` | Compiles `src/kompass_mojo/ffi.mojo` into `build/libkompass_mojo.so` |
| `pixi run build-harness` | Builds the C++ benchmark runner (depends on `build`) |
| `pixi run benchmark` | Runs the full benchmark pipeline and writes JSON to `results/` (depends on `build-harness`) |
| `pixi run test` | Runs per-kernel correctness tests on the GPU |

To specify a platform name for the output JSON:

```bash
PLATFORM=rtx_a5000 pixi run benchmark
# writes results/rtx_a5000.json
```

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

## Reference performance

| GPU | Implementation | CostEvaluator_5k_Trajs |
|---|---|---|
| RTX A5000 (Ampere `sm_86`) | kompass-core SYCL | 16.358 ms (±0.12) |
| RTX A5000 (Ampere `sm_86`) | kompass-mojo (Mojo nightly) | 15.973 ms (±0.09) |
| Strix Halo (`gfx1151`) | kompass-core SYCL | 8.23 ms |
| Strix Halo (`gfx1151`) | kompass-mojo (Mojo nightly) | 7.85 ms |

Numerical parity against kompass-core's CostEvaluator was verified bit-exact by linking `libkompass_mojo.so` directly into kompass-core's benchmark binary and feeding both code paths the same in-memory data — both produce `min_cost = 0.00100002` at `min_idx = 0`.

## Relationship to kompass-core

The kernels in this repo are Mojo ports of the GPU compute kernels in [kompass-core](https://github.com/automatika-robotics/kompass-core), which implements its GPGPU path in SYCL via AdaptiveCpp. Each Mojo kernel mirrors the math, memory layout, and work-group structure of its SYCL counterpart.

kompass-core is **not** a build dependency of this repo. The two projects are complementary: run each one's benchmark separately and compare JSON outputs via kompass-core's benchmark plotting script. Subsequent work will port additional kernel groups under the same pattern.

## License

TBD.
