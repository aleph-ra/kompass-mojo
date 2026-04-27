// C FFI for the Mojo cost evaluator kernels.
//
// This header is the contract between libkompass_mojo.so (built from
// src/kompass_mojo/ffi.mojo) and the C++ benchmark runner in this
// directory. Keep in sync with the pseudocode and field order documented
// at the top of ffi.mojo.

#ifndef KOMPASS_MOJO_H
#define KOMPASS_MOJO_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct MojoCostEvalConfig {
  float acc_lim_x;
  float acc_lim_y;
  float acc_lim_omega;
  float ref_path_weight;
  float goal_weight;
  float smoothness_weight;
  float jerk_weight;
  float obstacles_weight;
  float max_obstacles_distance;
} MojoCostEvalConfig;

// Opaque handle. Created by mojo_cost_eval_create, consumed by run/destroy.
typedef void *MojoCostEvalHandle;

MojoCostEvalHandle mojo_cost_eval_create(int32_t max_trajs,
                                         int32_t points_per_traj,
                                         int32_t max_obstacles,
                                         const MojoCostEvalConfig *cfg);

// Returns 0 on success, negative on error.
//
// `host_tracked_acc` must point at `ref_path_size` floats: the absolute
// prefix arc lengths on the FULL reference path at each tracked-segment
// index (i.e. the value of `accumulated_path_length_[start_idx + i]`.
// Only consumed when goal_weight > 0; pass nullptr otherwise.
int32_t mojo_cost_eval_run(MojoCostEvalHandle handle, const float *host_paths_x,
                           const float *host_paths_y, const float *host_vel_vx,
                           const float *host_vel_vy,
                           const float *host_vel_omega, int32_t trajs_size,
                           const float *host_tracked_x,
                           const float *host_tracked_y,
                           const float *host_tracked_acc, int32_t ref_path_size,
                           float tracked_segment_length, float ref_path_length,
                           const float *host_obs_x, const float *host_obs_y,
                           int32_t obs_size, float *out_min_cost,
                           int32_t *out_min_idx);

void mojo_cost_eval_destroy(MojoCostEvalHandle handle);

// ===========================================================================
// Local mapper FFI
// Grid memory is column-major: cell (x, y) lives at flat index
// `x + y * rows`, where `rows == grid_height`. OccupancyType codes:
//   UNEXPLORED = -1,  EMPTY = 0,  OCCUPIED = 100.
// ===========================================================================

typedef struct MojoLocalMapperConfig {
  float resolution;
  float laserscan_orientation;
  float laserscan_pos_x;
  float laserscan_pos_y;
  float laserscan_pos_z; // carried for parity, unused by the kernel
} MojoLocalMapperConfig;

typedef void *MojoLocalMapperHandle;

MojoLocalMapperHandle
mojo_local_mapper_create(int32_t rows, int32_t cols, int32_t scan_size,
                         int32_t max_points_per_line,
                         const MojoLocalMapperConfig *cfg);

// Returns 0 on success, negative on error. `host_grid_out` must point at a
// buffer of at least rows*cols int32_t entries. On return, populated with
// the occupancy grid in column-major order.
int32_t mojo_local_mapper_run(MojoLocalMapperHandle handle,
                              const double *host_angles,
                              const double *host_ranges,
                              int32_t *host_grid_out);

void mojo_local_mapper_destroy(MojoLocalMapperHandle handle);

// ===========================================================================
// Critical-zone checker FFI
//
// Two modes:
// - laserscan: `scan_size > 0`; handle precomputes cos/sin + forward /
//   backward cone index LUTs at create-time. `run_laserscan` is valid.
// - pointcloud: `max_cloud_bytes > 0` to pre-allocate; otherwise the
//   device raw-bytes buffer grows on first `run_pointcloud` call.
//
// Output is a single float in [0.0, 1.0]: 1.0 = safe, 0.0 = stop,
// intermediate = slowdown factor.
// ===========================================================================

typedef struct MojoCritZoneConfig {
  // 2D submatrix of the sensor→body Isometry3f (kernels drop Z).
  float tf00;
  float tf01;
  float tf03;
  float tf10;
  float tf11;
  float tf13;
  // Safety geometry.
  float robot_radius;
  float critical_angle; // half-angle in radians (must be <= pi/2)
  float critical_distance;
  float slowdown_distance;
  // Z-filter bounds for the pointcloud path.
  float min_height;
  float max_height;
} MojoCritZoneConfig;

typedef void *MojoCritZoneHandle;

// `scan_angles` is a host buffer of `scan_size` doubles (pass nullptr /
// scan_size=0 for pointcloud-only use). `max_cloud_bytes` pre-allocates
// the raw-bytes device buffer; set to 0 to defer allocation to the first
// run_pointcloud call.
MojoCritZoneHandle mojo_crit_zone_create(const double *scan_angles,
                                         int32_t scan_size,
                                         int32_t max_cloud_bytes,
                                         const MojoCritZoneConfig *cfg);

void mojo_crit_zone_destroy(MojoCritZoneHandle handle);

// Laserscan-mode check. `host_ranges` is a float buffer of `scan_size`
// entries (caller does the double→float narrowing). `forward=1` uses the
// forward cone, `forward=0` the backward cone. `out_factor` receives the
// resulting safety factor. Returns 0 on success, negative on error.
int32_t mojo_crit_zone_run_laserscan(MojoCritZoneHandle handle,
                                     const float *host_ranges, int32_t forward,
                                     float *out_factor);

// Pointcloud-mode check. Accepts raw PointCloud2 bytes + layout
// descriptors. Only FLOAT32 x/y/z fields are supported
// Returns 0 on success, negative on error.
int32_t mojo_crit_zone_run_pointcloud(MojoCritZoneHandle handle,
                                      const int8_t *host_bytes,
                                      int32_t total_bytes, int32_t point_step,
                                      int32_t row_step, int32_t height,
                                      int32_t width, int32_t x_offset,
                                      int32_t y_offset, int32_t z_offset,
                                      int32_t forward, float *out_factor);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // KOMPASS_MOJO_H
