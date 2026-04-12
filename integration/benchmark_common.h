// Standalone benchmark harness — adapted from kompass-core's
// src/kompass_cpp/benchmarks/benchmark_common.h. The timing methodology
// (5 warmup + 50 iterations, high_resolution_clock, JSON schema) is kept
// byte-identical so the emitted JSON drops straight into the existing
// plot_benchmarks.py pipeline. Differences from the original:
//   - logger.h replaced with plain stderr prints (no kompass-core dep)
//   - power monitor copied verbatim (hardware sensors)

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>

#ifdef ENABLE_POWER_MONITOR
#include <filesystem>
#include <regex>
#include <thread>
namespace fs = std::filesystem;
#endif

#define BM_RESET "\033[0m"
#define BM_CYAN  "\033[36m"
#define BM_BOLD  "\033[1m"

#define LOG_INFO(msg)  std::cerr << "[INFO] " << msg << std::endl
#define LOG_ERROR(msg) std::cerr << "[ERROR] " << msg << std::endl

namespace Kompass {
namespace Benchmarks {

// ============================================================================
// POWER MONITOR (copied from kompass-core, unchanged)
// ============================================================================
#ifdef ENABLE_POWER_MONITOR

class PowerMonitor {
public:
    PowerMonitor() { detect_power_source(); }

    void start() {
        if (sensors_.empty()) return;
        running_ = true;
        readings_.clear();
        monitor_thread_ = std::thread([this]() {
            while (running_.load()) {
                double total_w = 0.0;
                for (const auto& s : sensors_) total_w += read_sensor(s);
                readings_.push_back(total_w);
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
            }
        });
    }

    double stop() {
        running_ = false;
        if (monitor_thread_.joinable()) monitor_thread_.join();
        if (readings_.empty()) return 0.0;
        double sum = 0.0;
        for (double r : readings_) sum += r;
        return sum / readings_.size();
    }

private:
    enum class SensorKind { HwmonPowerUW, HwmonVoltCurrentUV_UA, PsuPowerUW, PsuVoltCurrent };
    struct Sensor {
        SensorKind kind;
        std::string path_a;
        std::string path_b;
    };
    std::vector<Sensor> sensors_;
    std::vector<double> readings_;
    std::atomic<bool> running_{false};
    std::thread monitor_thread_;

    void detect_power_source() {
        // Jetson-style INA3221 via i2c hwmon
        try {
            for (const auto& e : fs::directory_iterator("/sys/bus/i2c/drivers/ina3221")) {
                auto hwmon = e.path() / "hwmon";
                if (!fs::exists(hwmon)) continue;
                for (const auto& h : fs::directory_iterator(hwmon)) {
                    for (int ch = 0; ch < 4; ++ch) {
                        auto v = h.path() / ("in" + std::to_string(ch) + "_input");
                        auto c = h.path() / ("curr" + std::to_string(ch) + "_input");
                        if (fs::exists(v) && fs::exists(c)) {
                            sensors_.push_back({SensorKind::HwmonVoltCurrentUV_UA, v.string(), c.string()});
                        }
                    }
                }
            }
        } catch (...) {}
        if (!sensors_.empty()) return;

        // Generic hwmon with power*_input
        try {
            for (const auto& h : fs::directory_iterator("/sys/class/hwmon")) {
                for (const auto& f : fs::directory_iterator(h.path())) {
                    auto name = f.path().filename().string();
                    if (std::regex_match(name, std::regex("power[0-9]+_input"))) {
                        sensors_.push_back({SensorKind::HwmonPowerUW, f.path().string(), ""});
                    }
                }
            }
        } catch (...) {}
        if (!sensors_.empty()) return;

        // Laptop / ACPI style power_now
        try {
            for (const auto& h : fs::directory_iterator("/sys/class/power_supply")) {
                auto p = h.path() / "power_now";
                if (fs::exists(p)) {
                    sensors_.push_back({SensorKind::PsuPowerUW, p.string(), ""});
                    return;
                }
                auto v = h.path() / "voltage_now";
                auto c = h.path() / "current_now";
                if (fs::exists(v) && fs::exists(c)) {
                    sensors_.push_back({SensorKind::PsuVoltCurrent, v.string(), c.string()});
                    return;
                }
            }
        } catch (...) {}
    }

    double read_sensor(const Sensor& s) {
        auto read = [](const std::string& path) -> long {
            std::ifstream f(path);
            long v = 0;
            if (f) f >> v;
            return v;
        };
        switch (s.kind) {
            case SensorKind::HwmonPowerUW:     return read(s.path_a) / 1.0e6;
            case SensorKind::HwmonVoltCurrentUV_UA: {
                long mv = read(s.path_a);  // millivolts
                long ma = read(s.path_b);  // milliamps
                return (mv * ma) / 1.0e6;
            }
            case SensorKind::PsuPowerUW:       return read(s.path_a) / 1.0e6;
            case SensorKind::PsuVoltCurrent: {
                long uv = read(s.path_a);
                long ua = read(s.path_b);
                return (uv / 1.0e6) * (ua / 1.0e6);
            }
        }
        return 0.0;
    }
};

#else

class PowerMonitor {
public:
    void start() {}
    double stop() { return 0.0; }
};

#endif // ENABLE_POWER_MONITOR

// ============================================================================
// BENCHMARK ENGINE
// ============================================================================

struct BenchmarkResult {
    std::string test_name;
    double mean_ms;
    double std_dev_ms;
    double min_ms;
    double max_ms;
    int iterations;
    double avg_power_w;
};

template <typename Func>
BenchmarkResult measure_performance(std::string name, Func&& func,
                                    int iterations = 50, int warmup_runs = 5) {
    LOG_INFO("[Benchmark: " << name << "] Warming up (" << warmup_runs << " cycles)...");

    for (int i = 0; i < warmup_runs; ++i) func();

    std::vector<double> times;
    times.reserve(iterations);

    PowerMonitor power_mon;
    power_mon.start();

    std::cout << BM_CYAN << "       [Status] " << BM_RESET << "Running "
              << iterations << " iterations..." << std::endl;

    for (int i = 0; i < iterations; ++i) {
        std::cout << "\r" << BM_CYAN << "       [Status] " << BM_RESET
                  << "Iter " << BM_BOLD << (i + 1) << "/" << iterations
                  << BM_RESET << "..." << std::flush;

        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> ms = end - start;
        times.push_back(ms.count());
    }

    double avg_watts = power_mon.stop();
    std::cout << std::endl;

    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double mean = sum / iterations;
    double sq_sum = std::inner_product(times.begin(), times.end(), times.begin(), 0.0);
    double variance = (sq_sum / iterations) - (mean * mean);
    double std_dev = std::sqrt(variance > 0 ? variance : 0);
    double min_val = *std::min_element(times.begin(), times.end());
    double max_val = *std::max_element(times.begin(), times.end());

    std::stringstream ss_res;
    ss_res << "  -> " << std::fixed << std::setprecision(3) << mean
           << " ms (+/- " << std_dev << ")";
    if (avg_watts > 0.0) ss_res << " | Power: " << avg_watts << " W";
    LOG_INFO(ss_res.str());

    return {name, mean, std_dev, min_val, max_val, iterations, avg_watts};
}

inline void save_results_to_json(const std::string& platform_name,
                                 const std::vector<BenchmarkResult>& results,
                                 const std::string& filename) {
    // Minimal JSON emit — single-file, same schema as kompass-core's
    // benchmark_common.h so plot_benchmarks.py accepts it unchanged.
    std::ofstream f(filename);
    if (!f) { LOG_ERROR("Cannot open " << filename); return; }

    auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

    f << "{\n";
    f << "    \"platform\": \"" << platform_name << "\",\n";
    f << "    \"timestamp\": " << now << ",\n";
    f << "    \"benchmarks\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        f << "        {\n";
        f << "            \"test_name\": \"" << r.test_name << "\",\n";
        f << "            \"mean_ms\": " << r.mean_ms << ",\n";
        f << "            \"std_dev_ms\": " << r.std_dev_ms << ",\n";
        f << "            \"min_ms\": " << r.min_ms << ",\n";
        f << "            \"max_ms\": " << r.max_ms << ",\n";
        f << "            \"iterations\": " << r.iterations;
        if (r.avg_power_w > 0.0) {
            f << ",\n            \"avg_power_w\": " << r.avg_power_w;
        }
        f << "\n        }" << (i + 1 < results.size() ? "," : "") << "\n";
    }
    f << "    ]\n";
    f << "}\n";
    LOG_INFO("Saved results to " << filename);
}

} // namespace Benchmarks
} // namespace Kompass
