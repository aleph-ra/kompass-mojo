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
int32_t mojo_cost_eval_run(MojoCostEvalHandle handle, const float *host_paths_x,
                           const float *host_paths_y, const float *host_vel_vx,
                           const float *host_vel_vy,
                           const float *host_vel_omega, int32_t trajs_size,
                           const float *host_tracked_x,
                           const float *host_tracked_y, int32_t ref_path_size,
                           float tracked_segment_length, float goal_x,
                           float goal_y, float goal_path_length,
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

#ifdef __cplusplus
} // extern "C"
#endif

#endif // KOMPASS_MOJO_H
