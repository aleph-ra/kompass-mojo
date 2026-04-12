// Standalone benchmark runner for the Mojo cost evaluator.
//
// Mirrors the CostEvaluator_5k_Trajs scenario from kompass-core's
// src/kompass_cpp/benchmarks/benchmark_runner.cpp (lines 150-185) but
// dispatches to libkompass_mojo.so via C FFI instead of the SYCL
// CostEvaluator class. Same workload parameters, same measurement
// methodology, same output JSON schema — so the resulting file drops
// directly into plot_benchmarks.py alongside benchmark_cuda.json etc.
//
// Workload replication (matches benchmark_runner.cpp:150-185):
//   predictionHorizon = 10.0 s, timeStep = 0.01 s
//   numTrajectories = 5001, points_per_traj = 1000, velocities_count = 999
//   Reference path = 3 points [(0,0), (5,0), (10,0)] interpolated linearly
//     at 0.01 m spacing then segmented to length 1000, giving ref_size = 1001
//   Trajectory generator: center path + (pairs-1)/2 linear + angular sine
//     fluctuation variants
//   Weights: ref_path=1.0, smoothness=1.0, jerk=1.0, goal=1.0, obstacles=0.0
//   Control limits: acc_x=3, acc_y=3, acc_omega=3
//
// Build: cmake -B build integration && cmake --build build
// Run:   LD_LIBRARY_PATH=../build ./build/kompass_benchmark_mojo  <platform>  <out.json>

#include "benchmark_common.h"
#include "kompass_mojo.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using Kompass::Benchmarks::BenchmarkResult;
using Kompass::Benchmarks::measure_performance;
using Kompass::Benchmarks::save_results_to_json;

// =============================================================================
// Trajectory generator (flat float arrays)
//
// Replicates generate_heavy_trajectory_samples from benchmark_runner.cpp:37-91
// but emits flat row-major buffers directly, skipping kompass-core's
// TrajectorySamples2D type. Layout:
//   paths_x[t * N + i], paths_y[t * N + i] for t in [0, T), i in [0, N)
//   vel_vx[t * (N-1) + i], vel_vy[..], vel_omega[..]
// where T = number_of_samples, N = number_of_points.
// =============================================================================

struct FlatTrajectorySamples {
    int num_samples;
    int num_points;
    std::vector<float> paths_x;
    std::vector<float> paths_y;
    std::vector<float> vel_vx;
    std::vector<float> vel_vy;
    std::vector<float> vel_omega;
};

static FlatTrajectorySamples generate_heavy_trajectory_samples_flat(
    double predictionHorizon, double timeStep, int number_of_samples) {
    int number_of_points = static_cast<int>(predictionHorizon / timeStep);
    double v_1 = 1.0;
    double max_fluctuation = 0.5;

    FlatTrajectorySamples s;
    s.num_samples = number_of_samples;
    s.num_points = number_of_points;
    s.paths_x.assign(static_cast<size_t>(number_of_samples) * number_of_points, 0.0f);
    s.paths_y.assign(static_cast<size_t>(number_of_samples) * number_of_points, 0.0f);
    s.vel_vx.assign(static_cast<size_t>(number_of_samples) * (number_of_points - 1), 0.0f);
    s.vel_vy.assign(static_cast<size_t>(number_of_samples) * (number_of_points - 1), 0.0f);
    s.vel_omega.assign(static_cast<size_t>(number_of_samples) * (number_of_points - 1), 0.0f);

    auto store_path = [&](int t, int i, double x, double y) {
        size_t idx = static_cast<size_t>(t) * number_of_points + i;
        s.paths_x[idx] = static_cast<float>(x);
        s.paths_y[idx] = static_cast<float>(y);
    };
    auto store_vel = [&](int t, int i, double vx, double vy, double w) {
        size_t idx = static_cast<size_t>(t) * (number_of_points - 1) + i;
        s.vel_vx[idx] = static_cast<float>(vx);
        s.vel_vy[idx] = static_cast<float>(vy);
        s.vel_omega[idx] = static_cast<float>(w);
    };

    // 1. Center path (sample 0)
    for (int i = 0; i < number_of_points; ++i) {
        store_path(0, i, timeStep * v_1 * i, 0.0);
        if (i < number_of_points - 1) store_vel(0, i, v_1, 0.0, 0.0);
    }

    // 2. Generate pairs (linear + angular fluctuation)
    int pairs = (number_of_samples - 1) / 2;
    double amp_step = max_fluctuation / (pairs > 0 ? pairs : 1);

    int cursor = 1;
    for (int p = 1; p <= pairs; ++p) {
        double current_amp = p * amp_step;

        // Linear fluctuation
        for (int i = 0; i < number_of_points; ++i) {
            double fluct_v = current_amp * std::sin(2 * M_PI * i / number_of_points);
            store_path(cursor, i, timeStep * v_1 * i, timeStep * fluct_v * i);
            if (i < number_of_points - 1) store_vel(cursor, i, v_1, fluct_v, 0.0);
        }
        cursor++;
        if (cursor >= number_of_samples) break;

        // Angular fluctuation
        for (int i = 0; i < number_of_points; ++i) {
            double fluct_ang = current_amp * std::cos(2 * M_PI * i / number_of_points);
            double x = timeStep * v_1 * i * std::cos(fluct_ang);
            double y = timeStep * v_1 * i * std::sin(fluct_ang);
            store_path(cursor, i, x, y);
            if (i < number_of_points - 1) store_vel(cursor, i, v_1, 0.0, fluct_ang);
        }
        cursor++;
        if (cursor >= number_of_samples) break;
    }

    return s;
}

// =============================================================================
// Reference path generator (flat float arrays)
//
// Replicates the three-waypoint reference path from benchmark_runner.cpp:
//   Path::Path points { (0,0), (5,0), (10,0) }
//   reference_path.interpolate(0.01, LINEAR)
//   reference_path.segment(1000.0, 1000)  // length 1000m, max 1000 points
// The result is a linearly interpolated path along the x-axis with 1001
// points at 0.01 m spacing up to x=10.0 — then the segment() call caps it
// at ≤ 1000 points. For the benchmark we replicate that shape directly.
// =============================================================================

struct FlatRefPath {
    std::vector<float> x;
    std::vector<float> y;
    float length;
};

static FlatRefPath generate_ref_path_flat() {
    FlatRefPath r;
    // Match kompass-core's reference_path.interpolate(0.01, LINEAR) on waypoints
    // (0,0)-(5,0)-(10,0) which produces 1001 points at 0.01m spacing from x=0
    // to x=10.0 inclusive. segment(1000.0, 1000) keeps all 1001 since length <
    // 1000.0 and n <= 1000+1.
    int n = 1001;
    r.x.reserve(n);
    r.y.reserve(n);
    for (int i = 0; i < n; ++i) {
        r.x.push_back(static_cast<float>(i * 0.01));
        r.y.push_back(0.0f);
    }
    r.length = static_cast<float>((n - 1) * 0.01);  // 10.0 m
    return r;
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0]
                  << " <platform_name> <output_json_path>" << std::endl;
        return 1;
    }

    std::string platform_alias = argv[1];
    std::string output_path = argv[2];

    LOG_INFO("===================================================");
    LOG_INFO("  KOMPASS-MOJO BENCHMARK (cost evaluator)");
    LOG_INFO("  Target: " << platform_alias);
    LOG_INFO("===================================================");

    // --- Workload parameters (match benchmark_runner.cpp:153-177) ---
    double predictionHorizon = 10.0;
    double timeStep = 0.01;
    int numTrajectories = 5001;

    auto samples = generate_heavy_trajectory_samples_flat(
        predictionHorizon, timeStep, numTrajectories);
    auto ref_path = generate_ref_path_flat();

    LOG_INFO("Generated " << samples.num_samples << " trajectories x "
             << samples.num_points << " points; ref path "
             << ref_path.x.size() << " points, length " << ref_path.length << " m");

    // --- Config ---
    MojoCostEvalConfig cfg{};
    cfg.acc_lim_x = 3.0f;
    cfg.acc_lim_y = 3.0f;
    cfg.acc_lim_omega = 3.0f;
    cfg.ref_path_weight = 1.0f;
    cfg.goal_weight = 1.0f;
    cfg.smoothness_weight = 1.0f;
    cfg.jerk_weight = 1.0f;
    cfg.obstacles_weight = 0.0f;  // not used in this benchmark scenario
    cfg.max_obstacles_distance = 1.0f;

    // --- Create handle ---
    int max_obstacles = 1000;  // capacity even though benchmark passes 0
    MojoCostEvalHandle handle = mojo_cost_eval_create(
        samples.num_samples, samples.num_points, max_obstacles, &cfg);
    if (!handle) {
        LOG_ERROR("mojo_cost_eval_create returned null");
        return 2;
    }

    // --- Workload ---
    float out_min_cost = 0.0f;
    int32_t out_min_idx = -1;
    float goal_x = ref_path.x.empty() ? 0.0f : ref_path.x.back();
    float goal_y = ref_path.y.empty() ? 0.0f : ref_path.y.back();
    auto workload = [&]() {
        int rc = mojo_cost_eval_run(
            handle,
            samples.paths_x.data(), samples.paths_y.data(),
            samples.vel_vx.data(), samples.vel_vy.data(), samples.vel_omega.data(),
            samples.num_samples,
            ref_path.x.data(), ref_path.y.data(),
            static_cast<int32_t>(ref_path.x.size()), ref_path.length,
            goal_x, goal_y, ref_path.length,
            nullptr, nullptr, 0,  // no obstacles
            &out_min_cost, &out_min_idx);
        if (rc != 0) {
            LOG_ERROR("mojo_cost_eval_run returned " << rc);
        }
    };

    std::vector<BenchmarkResult> results;
    results.push_back(measure_performance("CostEvaluator_5k_Trajs", workload));

    LOG_INFO("Final min cost idx: " << out_min_idx << "  cost: " << out_min_cost);

    mojo_cost_eval_destroy(handle);

    save_results_to_json(platform_alias, results, output_path);
    return 0;
}
