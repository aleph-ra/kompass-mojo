// Standalone benchmark runner for the Mojo kernels.
//
// Mirrors scenarios from kompass-core's
// src/kompass_cpp/benchmarks/benchmark_runner.cpp but
// dispatches to libkompass_mojo.so via C FFI
// Build: cmake -B build integration && cmake --build build
// Run: LD_LIBRARY_PATH=../build ./build/kompass_benchmark_mojo  <platform>  <out.json>

#include "benchmark_common.h"
#include "kompass_mojo.h"

#include <cmath>
#include <cstdint>
#include <cstring>
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

// =============================================================================
// Mapping scan generator
//
// Produces a laserscan with `num_points` rays spaced evenly across 2*pi, with
// a sinusoidally-modulated range.
// =============================================================================

static void generate_mapping_scan(size_t num_points,
                                  std::vector<double>& ranges,
                                  std::vector<double>& angles) {
    ranges.resize(num_points);
    angles.resize(num_points);
    double angle_step = (2.0 * M_PI) / num_points;
    for (size_t i = 0; i < num_points; ++i) {
        angles[i] = -M_PI + (i * angle_step);
        ranges[i] = 5.0 + 2.0 * std::sin(angles[i] * 20.0);
    }
}


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

    // =========================================================================
    // TEST 1: CostEvaluator
    // =========================================================================
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
            static_cast<int32_t>(ref_path.x.size()),
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

    // =========================================================================
    // TEST 2: LocalMapper
    //
    // 400x400 grid at 0.05 m resolution. scan_size matches the generated scan
    // (3600 rays, all processed).
    // =========================================================================
    {
        int rows = 400;
        int cols = 400;
        float res = 0.05f;
        int scan_size = 3600;
        int max_points_per_line = 256;

        std::vector<double> ranges, angles;
        generate_mapping_scan(3600, ranges, angles);

        MojoLocalMapperConfig mcfg{};
        mcfg.resolution = res;
        mcfg.laserscan_orientation = 0.0f;
        mcfg.laserscan_pos_x = 0.0f;
        mcfg.laserscan_pos_y = 0.0f;
        mcfg.laserscan_pos_z = 0.0f;

        MojoLocalMapperHandle mh = mojo_local_mapper_create(
            rows, cols, scan_size, max_points_per_line, &mcfg);
        if (!mh) {
            LOG_ERROR("mojo_local_mapper_create returned null");
            return 3;
        }

        std::vector<int32_t> grid_out(static_cast<size_t>(rows) * cols);

        auto map_workload = [&]() {
            int rc = mojo_local_mapper_run(
                mh, angles.data(), ranges.data(), grid_out.data());
            if (rc != 0) {
                LOG_ERROR("mojo_local_mapper_run returned " << rc);
            }
        };

        results.push_back(measure_performance("Mapper_Dense_400x400", map_workload));

        // Sanity-log the occupied-cell count so we can spot regressions where
        // the kernel accidentally no-ops.
        size_t n_occupied = 0;
        for (int32_t v : grid_out) {
            if (v == 100) ++n_occupied;
        }
        LOG_INFO("Mapper: " << n_occupied << " OCCUPIED cells (out of "
                            << grid_out.size() << ")");

        mojo_local_mapper_destroy(mh);
    }

    // =========================================================================
    // TEST 3 & 4: CriticalZoneChecker (laserscan + pointcloud)
    //
    // Two harnesses read identical inputs:
    //   cylinder robot, radius 0.51, sensor pos (0.22, 0, 0.4),
    //   sensor rotation quaternion {0, 0, 0.99, 0} (normalised → 180° about
    //   z → 2D submatrix tf00=-1, tf11=-1, translation (0.22, 0)),
    //   critical_angle = 160° (halved to 80° = 1.396 rad),
    //   crit_dist=0.3, slow_dist=0.6, min_h=0.1, max_h=2.0.
    //
    // Both benchmarks share one CZ handle.
    // =========================================================================
    {
        MojoCritZoneConfig czcfg{};
        czcfg.tf00 = -1.0f;
        czcfg.tf01 =  0.0f;
        czcfg.tf03 =  0.22f;
        czcfg.tf10 =  0.0f;
        czcfg.tf11 = -1.0f;
        czcfg.tf13 =  0.0f;
        czcfg.robot_radius = 0.51f;
        czcfg.critical_angle = static_cast<float>(160.0 * M_PI / 180.0 / 2.0);
        czcfg.critical_distance = 0.3f;
        czcfg.slowdown_distance = 0.6f;
        czcfg.min_height = 0.1f;
        czcfg.max_height = 2.0f;

        // ---- Laserscan input: 3600 angles, ranges tuned to slowdown zone ----
        // R^2 + 0.44·cos(θ)·R - 0.8732 = 0 so every point lands at ~0.96 m
        // distance from the body origin — inside the slowdown band [0.81, 1.11]
        // between critical=0.3 and slow=0.6.
        const int scan_points = 3600;
        std::vector<double> cz_angles(scan_points);
        std::vector<float>  cz_ranges(scan_points);
        {
            const double sx = 0.22;
            const double r_target = 0.96;
            const double c_const = sx * sx - r_target * r_target;  // -0.8732
            for (int i = 0; i < scan_points; ++i) {
                double theta = -M_PI + (2.0 * M_PI) * i / scan_points;
                cz_angles[i] = theta;
                double b = 2.0 * sx * std::cos(theta);
                double disc = b * b - 4.0 * c_const;
                double r = (-b + std::sqrt(disc)) * 0.5;
                cz_ranges[i] = static_cast<float>(r);
            }
        }

        // ---- Pointcloud input: 100k random PointXYZ (x,y in [-10,10), z in [0,3)) ----
        // 16B/point, x/y/z float32 at 0/4/8.
        const int pc_count = 100000;
        const int pc_stride = 16;
        std::vector<int8_t> cz_cloud(static_cast<size_t>(pc_count) * pc_stride);
        {
            std::srand(42);  // reproducible inputs
            float* fp = reinterpret_cast<float*>(cz_cloud.data());
            for (int i = 0; i < pc_count; ++i) {
                fp[i * 4 + 0] = (std::rand() % 2000) / 100.0f - 10.0f;
                fp[i * 4 + 1] = (std::rand() % 2000) / 100.0f - 10.0f;
                fp[i * 4 + 2] = (std::rand() %  300) / 100.0f;
                fp[i * 4 + 3] = 0.0f;
            }
        }
        int pc_total_bytes = pc_count * pc_stride;

        MojoCritZoneHandle czh = mojo_crit_zone_create(
            cz_angles.data(), scan_points, pc_total_bytes, &czcfg);
        if (!czh) {
            LOG_ERROR("mojo_crit_zone_create returned null");
            return 4;
        }

        // --- TEST 3: laserscan ---
        float cz_scan_factor = 1.0f;
        auto cz_scan_workload = [&]() {
            int rc = mojo_crit_zone_run_laserscan(
                czh, cz_ranges.data(), /*forward*/ 1, &cz_scan_factor);
            if (rc != 0) {
                LOG_ERROR("mojo_crit_zone_run_laserscan returned " << rc);
            }
        };
        results.push_back(measure_performance("CriticalZone_Dense_Scan",
                                              cz_scan_workload));
        LOG_INFO("CriticalZone_Dense_Scan factor: " << cz_scan_factor);

        // --- TEST 4: pointcloud ---
        float cz_pc_factor = 1.0f;
        auto cz_pc_workload = [&]() {
            int rc = mojo_crit_zone_run_pointcloud(
                czh, cz_cloud.data(), pc_total_bytes,
                pc_stride, pc_stride * pc_count, 1, pc_count,
                0, 4, 8,
                /*forward*/ 1, &cz_pc_factor);
            if (rc != 0) {
                LOG_ERROR("mojo_crit_zone_run_pointcloud returned " << rc);
            }
        };
        results.push_back(measure_performance("CriticalZone_100k_Cloud",
                                              cz_pc_workload));
        LOG_INFO("CriticalZone_100k_Cloud factor: " << cz_pc_factor);

        mojo_crit_zone_destroy(czh);
    }

    save_results_to_json(platform_alias, results, output_path);
    return 0;
}
