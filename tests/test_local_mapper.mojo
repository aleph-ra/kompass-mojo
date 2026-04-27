# Unit tests for the LocalMapper GPU kernel (scan_to_grid).
#
# Each test builds a small deterministic scan, launches the kernel directly,
# reads the occupancy grid back, and asserts against hand-computed expected
# values.
# Grid layout: 21x21 cells, 0.1 m resolution — 2.1 m x 2.1 m window. With
# laserscan_pos = (0, 0, 0) and central = (round(21/2)-1, round(21/2)-1) =
# (9, 9), the robot sits at cell (9, 9).

from std.testing import assert_true, assert_almost_equal
from std.math import sqrt, cos, sin, pi, ceildiv
from std.memory import UnsafePointer
from std.gpu.host import DeviceContext, DeviceBuffer

from kompass_mojo.local_mapper import (
    OCC_UNEXPLORED,
    OCC_EMPTY,
    OCC_OCCUPIED,
    pointcloud_to_laserscan_kernel,
    scan_to_grid_kernel,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fill_f64(ctx: DeviceContext, buf: DeviceBuffer[DType.float64], values: List[Float64]) raises:
    var host = ctx.enqueue_create_host_buffer[DType.float64](len(values))
    var ptr = host.unsafe_ptr().value()
    for i in range(len(values)):
        (ptr + i)[] = values[i]
    ctx.enqueue_copy(dst_buf=buf, src_buf=host)
    ctx.synchronize()


def _read_grid(ctx: DeviceContext, grid: DeviceBuffer[DType.int32], n: Int) raises -> List[Int32]:
    var result = List[Int32]()
    with grid.map_to_host() as mapped:
        var ptr = mapped.unsafe_ptr().value()
        for i in range(n):
            result.append(Int32((ptr + i)[]))
    return result^


def _fill_distances(
    ctx: DeviceContext,
    distances: DeviceBuffer[DType.float32],
    rows: Int, cols: Int,
    resolution: Float32,
    lpos_x: Float32, lpos_y: Float32, lpos_z: Float32,
    central_x: Int32, central_y: Int32,
) raises:
    """Per-cell distance from the laserscan origin. Mirrors
    local_mapper_gpu.h:37-43 — kernel reads `distances[x + y*rows]`."""
    var host = ctx.enqueue_create_host_buffer[DType.float32](rows * cols)
    var ptr = host.unsafe_ptr().value()
    for j in range(cols):
        for i in range(rows):
            var dx = Float32(central_x - Int32(i)) * resolution - lpos_x
            var dy = Float32(central_y - Int32(j)) * resolution - lpos_y
            var dz = Float32(0.0) - lpos_z
            (ptr + (i + j * rows))[] = sqrt(dx * dx + dy * dy + dz * dz)
    ctx.enqueue_copy(dst_buf=distances, src_buf=host)
    ctx.synchronize()


def _render_grid(grid: List[Int32], rows: Int, cols: Int):
    """Print the grid with '#' for OCCUPIED, '.' for EMPTY, ' ' for UNEXPLORED.
    Column-major: cell (i, j) is at flat index i + j*rows, where i is the row
    (increasing downward) and j is the column (increasing right)."""
    var border = String("  +")
    for _ in range(cols):
        border = border + "-"
    border = border + "+"
    print(border)
    for i in range(rows):
        var line = String("  |")
        for j in range(cols):
            var v = grid[i + j * rows]
            if v == Int32(100):
                line = line + "#"
            elif v == Int32(0):
                line = line + "."
            else:
                line = line + " "
        line = line + "|"
        print(line)
    print(border)


def _count_values(grid: List[Int32], value: Int32) -> Int:
    var n = 0
    for k in range(len(grid)):
        if grid[k] == value:
            n = n + 1
    return n


# ---------------------------------------------------------------------------
# Test scaffold — allocates buffers, computes geometry constants, launches
# the kernel. Each test fills angles/ranges and then calls this helper.
# ---------------------------------------------------------------------------


def _run_mapper(
    ctx: DeviceContext,
    angles_vals: List[Float64],
    ranges_vals: List[Float64],
    rows: Int, cols: Int,
    resolution: Float32,
    laserscan_orientation: Float32,
    lpos_x: Float32, lpos_y: Float32, lpos_z: Float32,
    max_points_per_line: Int,
) raises -> List[Int32]:
    var scan_size = len(angles_vals)
    var cells = rows * cols

    var grid = ctx.enqueue_create_buffer[DType.int32](cells)
    var distances = ctx.enqueue_create_buffer[DType.float32](cells)
    var angles = ctx.enqueue_create_buffer[DType.float64](scan_size)
    var ranges = ctx.enqueue_create_buffer[DType.float32](scan_size)

    var central_x = Int32(rows // 2 - 1)
    var central_y = Int32(cols // 2 - 1)
    var start_x = central_x + Int32(Int(lpos_x / resolution))
    var start_y = central_y + Int32(Int(lpos_y / resolution))

    _fill_distances(
        ctx, distances, rows, cols, resolution,
        lpos_x, lpos_y, lpos_z, central_x, central_y,
    )
    _fill_f64(ctx, angles, angles_vals)
    # Convert host doubles → device float32 (matches kompass-core's path).
    var ranges_host = ctx.enqueue_create_host_buffer[DType.float32](scan_size)
    var rh_ptr = ranges_host.unsafe_ptr().value()
    for i in range(scan_size):
        (rh_ptr + i)[] = Float32(ranges_vals[i])
    ctx.enqueue_copy(dst_buf=ranges, src_buf=ranges_host)
    ctx.synchronize()
    grid.enqueue_fill(OCC_UNEXPLORED)
    ctx.synchronize()

    ctx.enqueue_function[scan_to_grid_kernel, scan_to_grid_kernel](
        ranges.unsafe_ptr(), angles.unsafe_ptr(),
        grid.unsafe_ptr(), distances.unsafe_ptr(),
        rows, cols,
        resolution,
        laserscan_orientation,
        lpos_x, lpos_y,
        central_x, central_y,
        start_x, start_y,
        grid_dim=scan_size,
        block_dim=max_points_per_line,
    )
    ctx.synchronize()

    return _read_grid(ctx, grid, cells)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_single_ray(ctx: DeviceContext) raises:
    """One ray at angle=0, range=0.5. Endpoint cell = central + (5, 0) =
    (14, 9). Expected: exactly one OCCUPIED at (14, 9); EMPTY trail along
    the +x-axis from (9, 9) to (14, 9); everything else UNEXPLORED."""
    comptime ROWS = 21
    comptime COLS = 21
    var grid = _run_mapper(
        ctx,
        [0.0],
        [0.5],
        ROWS, COLS,
        Float32(0.1), Float32(0.0),
        Float32(0.0), Float32(0.0), Float32(0.0),
        32,
    )

    var endpoint_idx = 14 + 9 * ROWS
    assert_true(
        grid[endpoint_idx] == Int32(100),
        "single ray: endpoint (14, 9) must be OCCUPIED",
    )

    var origin_idx = 9 + 9 * ROWS
    assert_true(
        grid[origin_idx] == Int32(0),
        "single ray: origin (9, 9) must be EMPTY (free line-of-sight)",
    )

    var n_occ = _count_values(grid, Int32(100))
    assert_true(
        n_occ == 1,
        "single ray: exactly 1 OCCUPIED cell expected, got " + String(n_occ),
    )
    print("  PASS: test_single_ray (n_occ=", n_occ, ")")


def test_four_cardinal_rays(ctx: DeviceContext) raises:
    """4 rays at 0, pi/2, pi, 3*pi/2, all range 0.5.
    Endpoints: (14, 9), (9, 14), (4, 9), (9, 4). 4 OCCUPIED cells total."""
    comptime ROWS = 21
    comptime COLS = 21
    var grid = _run_mapper(
        ctx,
        [0.0, pi / 2.0, pi, 3.0 * pi / 2.0],
        [0.5, 0.5, 0.5, 0.5],
        ROWS, COLS,
        Float32(0.1), Float32(0.0),
        Float32(0.0), Float32(0.0), Float32(0.0),
        32,
    )

    # Each of the four endpoints must be OCCUPIED.
    var endpoints = List[Int]()
    endpoints.append(14 + 9 * ROWS)   # +x
    endpoints.append(9 + 14 * ROWS)   # +y
    endpoints.append(4 + 9 * ROWS)    # -x
    endpoints.append(9 + 4 * ROWS)    # -y
    for k in range(len(endpoints)):
        assert_true(
            grid[endpoints[k]] == Int32(100),
            "four_cardinal: endpoint " + String(k) + " must be OCCUPIED",
        )

    var n_occ = _count_values(grid, Int32(100))
    assert_true(
        n_occ == 4,
        "four_cardinal: exactly 4 OCCUPIED cells expected, got " + String(n_occ),
    )
    print("  PASS: test_four_cardinal_rays (n_occ=", n_occ, ")")


def test_circle_radius_0_5(ctx: DeviceContext) raises:
    """63 rays evenly spaced [0, 2*pi), all range 0.5. Produces a ring of
    OCCUPIED cells at ~5 cells from the center. Renders to terminal."""
    comptime ROWS = 21
    comptime COLS = 21
    comptime NUM_RAYS = 63
    comptime RADIUS = 0.5

    var angles = List[Float64]()
    var ranges = List[Float64]()
    for i in range(NUM_RAYS):
        angles.append(Float64(2.0) * pi * Float64(i) / Float64(NUM_RAYS))
        ranges.append(RADIUS)

    var grid = _run_mapper(
        ctx, angles, ranges,
        ROWS, COLS,
        Float32(0.1), Float32(0.0),
        Float32(0.0), Float32(0.0), Float32(0.0),
        32,
    )

    print("  circle radius 0.5 (63 rays) on 21x21 grid:")
    _render_grid(grid, ROWS, COLS)

    var n_occ = _count_values(grid, Int32(100))
    var n_empty = _count_values(grid, Int32(0))
    assert_true(
        n_occ > 5 and n_occ < 40,
        "circle_0_5: expected 5 < n_occ < 40, got " + String(n_occ),
    )
    assert_true(n_empty > 0, "circle_0_5: expected some EMPTY cells")

    var corner_idx = 0 + 0 * ROWS
    assert_true(
        grid[corner_idx] == Int32(-1),
        "circle_0_5: top-left corner should remain UNEXPLORED",
    )
    print("  PASS: test_circle_radius_0_5 (n_occ=", n_occ, "n_empty=", n_empty, ")")


def test_circle_radius_2_0(ctx: DeviceContext) raises:
    """63 rays at range 2.0 on a 21x21 / 0.1 m grid (half-window = 1.0 m).
    All rays clip at the grid boundary. Validates out-of-bounds guarding —
    no writes past the edge, no SIGSEGV."""
    comptime ROWS = 21
    comptime COLS = 21
    comptime NUM_RAYS = 63
    comptime RADIUS = 2.0

    var angles = List[Float64]()
    var ranges = List[Float64]()
    for i in range(NUM_RAYS):
        angles.append(Float64(2.0) * pi * Float64(i) / Float64(NUM_RAYS))
        ranges.append(RADIUS)

    var grid = _run_mapper(
        ctx, angles, ranges,
        ROWS, COLS,
        Float32(0.1), Float32(0.0),
        Float32(0.0), Float32(0.0), Float32(0.0),
        32,
    )

    print("  circle radius 2.0 (clipped by grid boundary):")
    _render_grid(grid, ROWS, COLS)

    var n_empty = _count_values(grid, Int32(0))
    assert_true(
        n_empty > 30,
        "circle_2_0: expected many EMPTY cells, got " + String(n_empty),
    )
    print("  PASS: test_circle_radius_2_0 (n_empty=", n_empty, ")")


def test_out_of_bounds_range(ctx: DeviceContext) raises:
    """A single ray with range = 1e6. Endpoint is far outside the grid, so
    no OCCUPIED cells should be stamped; any in-grid cells along the ray are
    marked EMPTY. No SIGSEGV."""
    comptime ROWS = 21
    comptime COLS = 21
    var grid = _run_mapper(
        ctx,
        [0.0],
        [1.0e6],
        ROWS, COLS,
        Float32(0.1), Float32(0.0),
        Float32(0.0), Float32(0.0), Float32(0.0),
        32,
    )

    var n_occ = _count_values(grid, Int32(100))
    var n_empty = _count_values(grid, Int32(0))
    assert_true(
        n_occ == 0,
        "oob_range: endpoint is outside grid, expect 0 OCCUPIED, got " + String(n_occ),
    )
    assert_true(
        n_empty > 0,
        "oob_range: expect some in-grid EMPTY cells along the ray",
    )
    print("  PASS: test_out_of_bounds_range (n_empty=", n_empty, ")")


# ---------------------------------------------------------------------------
# Helpers — pointcloud path
# ---------------------------------------------------------------------------


def _run_pc_kernel(
    ctx: DeviceContext,
    points_xyz: List[Float32],     # flat [x0, y0, z0, x1, y1, z1, ...]
    num_points: Int,
    num_bins: Int,
    min_z: Float32,
    max_z: Float32,
    range_max: Float32,
) raises -> List[Float32]:
    """Run pointcloud_to_laserscan_kernel in isolation and return per-bin
    ranges as a host List[Float32]. Bypasses the FFI so we can inspect the
    intermediate buffer directly."""
    var stride = 16  # 4 floats: x, y, z, pad — matches benchmark layout
    var total_bytes = num_points * stride

    var raw = ctx.enqueue_create_buffer[DType.int8](total_bytes)
    var raw_host = ctx.enqueue_create_host_buffer[DType.int8](total_bytes)
    var rh_ptr = raw_host.unsafe_ptr().value().bitcast[Float32]()
    for i in range(num_points):
        (rh_ptr + (i * 4 + 0))[] = points_xyz[i * 3 + 0]
        (rh_ptr + (i * 4 + 1))[] = points_xyz[i * 3 + 1]
        (rh_ptr + (i * 4 + 2))[] = points_xyz[i * 3 + 2]
        (rh_ptr + (i * 4 + 3))[] = Float32(0.0)
    ctx.enqueue_copy(dst_buf=raw, src_buf=raw_host)

    var ranges = ctx.enqueue_create_buffer[DType.float32](num_bins)
    ranges.enqueue_fill(range_max)
    ctx.synchronize()

    var WG = 256
    var num_blocks = ceildiv(num_points, WG)
    var max_z_enabled = Int32(1) if max_z >= Float32(0.0) else Int32(0)
    var inv_two_pi_times_bins = Float32(num_bins) / (Float32(2.0) * Float32(pi))

    ctx.enqueue_function[
        pointcloud_to_laserscan_kernel, pointcloud_to_laserscan_kernel
    ](
        raw.unsafe_ptr(), total_bytes, num_points,
        stride,                       # point_step
        stride * num_points,          # row_step (contiguous)
        num_points,                   # width
        Int32(1),                     # is_contiguous
        0, 4, 8,                      # x_offset, y_offset, z_offset
        min_z, max_z, max_z_enabled,
        num_bins, inv_two_pi_times_bins,
        ranges.unsafe_ptr(),
        grid_dim=num_blocks, block_dim=WG,
    )
    ctx.synchronize()

    var out = List[Float32]()
    with ranges.map_to_host() as mapped:
        var ptr = mapped.unsafe_ptr().value()
        for i in range(num_bins):
            out.append((ptr + i)[])
    return out^


def _angle_to_bin(angle_rad: Float64, num_bins: Int) -> Int:
    """Mirror the kernel's bin assignment: normalize to [0, 2π), then
    floor((angle / 2π) * num_bins) and clamp."""
    var a = angle_rad
    if a < 0.0:
        a = a + 2.0 * pi
    var b = Int(a * Float64(num_bins) / (2.0 * pi))
    if b >= num_bins:
        b = num_bins - 1
    return b


def _run_pc_mapper(
    ctx: DeviceContext,
    points_xyz: List[Float32],
    num_points: Int,
    rows: Int, cols: Int,
    resolution: Float32,
    laserscan_orientation: Float32,
    lpos_x: Float32, lpos_y: Float32, lpos_z: Float32,
    num_bins: Int,
    max_points_per_line: Int,
    min_z: Float32 = Float32(-100.0),
    max_z: Float32 = Float32(100.0),
    range_max: Float32 = Float32(20.0),
) raises -> List[Int32]:
    """End-to-end: pointcloud → laserscan → raycast grid."""
    var stride = 16
    var total_bytes = num_points * stride

    var raw = ctx.enqueue_create_buffer[DType.int8](total_bytes)
    var raw_host = ctx.enqueue_create_host_buffer[DType.int8](total_bytes)
    var rh_ptr = raw_host.unsafe_ptr().value().bitcast[Float32]()
    for i in range(num_points):
        (rh_ptr + (i * 4 + 0))[] = points_xyz[i * 3 + 0]
        (rh_ptr + (i * 4 + 1))[] = points_xyz[i * 3 + 1]
        (rh_ptr + (i * 4 + 2))[] = points_xyz[i * 3 + 2]
        (rh_ptr + (i * 4 + 3))[] = Float32(0.0)
    ctx.enqueue_copy(dst_buf=raw, src_buf=raw_host)

    var cells = rows * cols
    var grid = ctx.enqueue_create_buffer[DType.int32](cells)
    var distances = ctx.enqueue_create_buffer[DType.float32](cells)
    var angles = ctx.enqueue_create_buffer[DType.float64](num_bins)
    var ranges = ctx.enqueue_create_buffer[DType.float32](num_bins)

    var central_x = Int32(rows // 2 - 1)
    var central_y = Int32(cols // 2 - 1)
    var start_x = central_x + Int32(Int(lpos_x / resolution))
    var start_y = central_y + Int32(Int(lpos_y / resolution))

    _fill_distances(
        ctx, distances, rows, cols, resolution,
        lpos_x, lpos_y, lpos_z, central_x, central_y,
    )

    # Per-bin angles: i * 2π/num_bins (the kernel uses the same convention).
    var ang_host = ctx.enqueue_create_host_buffer[DType.float64](num_bins)
    var ah_ptr = ang_host.unsafe_ptr().value()
    var step = (2.0 * pi) / Float64(num_bins)
    for k in range(num_bins):
        (ah_ptr + k)[] = Float64(k) * step
    ctx.enqueue_copy(dst_buf=angles, src_buf=ang_host)

    grid.enqueue_fill(OCC_UNEXPLORED)
    ranges.enqueue_fill(range_max)
    ctx.synchronize()

    var WG = 256
    var num_blocks = ceildiv(num_points, WG)
    var max_z_enabled = Int32(1) if max_z >= Float32(0.0) else Int32(0)
    var inv_two_pi_times_bins = Float32(num_bins) / (Float32(2.0) * Float32(pi))

    ctx.enqueue_function[
        pointcloud_to_laserscan_kernel, pointcloud_to_laserscan_kernel
    ](
        raw.unsafe_ptr(), total_bytes, num_points,
        stride, stride * num_points, num_points,
        Int32(1),
        0, 4, 8,
        min_z, max_z, max_z_enabled,
        num_bins, inv_two_pi_times_bins,
        ranges.unsafe_ptr(),
        grid_dim=num_blocks, block_dim=WG,
    )

    ctx.enqueue_function[scan_to_grid_kernel, scan_to_grid_kernel](
        ranges.unsafe_ptr(), angles.unsafe_ptr(),
        grid.unsafe_ptr(), distances.unsafe_ptr(),
        rows, cols, resolution,
        laserscan_orientation,
        lpos_x, lpos_y,
        central_x, central_y,
        start_x, start_y,
        grid_dim=num_bins,
        block_dim=max_points_per_line,
    )
    ctx.synchronize()

    return _read_grid(ctx, grid, cells)


# ---------------------------------------------------------------------------
# Pointcloud kernel-level tests
# ---------------------------------------------------------------------------


def test_pc_single_point_in_known_bin(ctx: DeviceContext) raises:
    """One point at (3, 0, 0.5) with 4 bins. atan2(0, 3) = 0 → bin 0.
    Expect ranges[0] ≈ 3.0; bins 1..3 untouched at range_max."""
    var pts = List[Float32]()
    pts.append(3.0); pts.append(0.0); pts.append(0.5)

    var ranges = _run_pc_kernel(
        ctx, pts, 1, 4,
        Float32(-1.0), Float32(2.0), Float32(20.0),
    )

    assert_almost_equal(ranges[0], 3.0, atol=1e-3, msg="bin0 range")
    assert_true(ranges[1] >= Float32(19.99), "bin1 untouched")
    assert_true(ranges[2] >= Float32(19.99), "bin2 untouched")
    assert_true(ranges[3] >= Float32(19.99), "bin3 untouched")
    print("  PASS: test_pc_single_point_in_known_bin")


def test_pc_atomic_min_keeps_closer(ctx: DeviceContext) raises:
    """Two points, same bin (positive x-axis): (3, 0, 0) and (1, 0, 0).
    atomic_min must keep 1.0."""
    var pts = List[Float32]()
    pts.append(3.0); pts.append(0.0); pts.append(0.5)
    pts.append(1.0); pts.append(0.0); pts.append(0.5)

    var ranges = _run_pc_kernel(
        ctx, pts, 2, 4,
        Float32(-1.0), Float32(2.0), Float32(20.0),
    )

    assert_almost_equal(ranges[0], 1.0, atol=1e-3, msg="bin0 keeps closer")
    print("  PASS: test_pc_atomic_min_keeps_closer (bin0=", ranges[0], ")")


def test_pc_z_filter_rejects(ctx: DeviceContext) raises:
    """Point at z=5.0 with max_z=2.0 must NOT update any bin."""
    var pts = List[Float32]()
    pts.append(3.0); pts.append(0.0); pts.append(5.0)

    var ranges = _run_pc_kernel(
        ctx, pts, 1, 4,
        Float32(0.1), Float32(2.0), Float32(20.0),
    )

    for b in range(4):
        assert_true(
            ranges[b] >= Float32(19.99),
            "z-filter: bin " + String(b) + " should be untouched",
        )
    print("  PASS: test_pc_z_filter_rejects")


def test_pc_origin_filter_rejects(ctx: DeviceContext) raises:
    """Point at (0, 0, 0.5) — r²=0 < 1e-6 — must not update any bin."""
    var pts = List[Float32]()
    pts.append(0.0); pts.append(0.0); pts.append(0.5)

    var ranges = _run_pc_kernel(
        ctx, pts, 1, 4,
        Float32(-1.0), Float32(2.0), Float32(20.0),
    )

    for b in range(4):
        assert_true(
            ranges[b] >= Float32(19.99),
            "origin-filter: bin " + String(b) + " should be untouched",
        )
    print("  PASS: test_pc_origin_filter_rejects")


def test_pc_atan2_quadrants(ctx: DeviceContext) raises:
    """One point per quadrant, NUM_BINS=8 (each spans π/4). Verify the
    fast atan2 polynomial bins each point into the bin holding its angle."""
    comptime NUM_BINS = 8
    # Place one point per π/3 across [0, 2π) — picks 6 distinct bins.
    var angles_test = List[Float64]()
    var radii = List[Float32]()
    angles_test.append(0.3); radii.append(2.0)
    angles_test.append(1.2); radii.append(2.0)
    angles_test.append(2.0); radii.append(2.0)
    angles_test.append(3.5); radii.append(2.0)
    angles_test.append(4.5); radii.append(2.0)
    angles_test.append(5.7); radii.append(2.0)

    var pts = List[Float32]()
    for k in range(len(angles_test)):
        var r = radii[k]
        var a = angles_test[k]
        pts.append(Float32(Float64(r) * cos(a)))
        pts.append(Float32(Float64(r) * sin(a)))
        pts.append(Float32(0.0))

    var ranges = _run_pc_kernel(
        ctx, pts, len(angles_test), NUM_BINS,
        Float32(-1.0), Float32(2.0), Float32(20.0),
    )

    for k in range(len(angles_test)):
        var expected_bin = _angle_to_bin(angles_test[k], NUM_BINS)
        assert_almost_equal(
            ranges[expected_bin], 2.0, atol=5e-3,
            msg="atan2 quadrant: bin " + String(expected_bin),
        )
    print("  PASS: test_pc_atan2_quadrants")


# ---------------------------------------------------------------------------
# Pointcloud end-to-end visual tests
# ---------------------------------------------------------------------------


def test_pc_visual_single_point(ctx: DeviceContext) raises:
    """One point at (0.5, 0, 0) with 16 bins on a 21x21 / 0.1 m grid.
    Should stamp a single OCCUPIED cell on the +x axis at x≈cell+5."""
    comptime ROWS = 21
    comptime COLS = 21
    comptime NUM_BINS = 16

    var pts = List[Float32]()
    pts.append(0.5); pts.append(0.0); pts.append(0.0)

    var grid = _run_pc_mapper(
        ctx, pts, 1,
        ROWS, COLS,
        Float32(0.1), Float32(0.0),
        Float32(0.0), Float32(0.0), Float32(0.0),
        NUM_BINS, 32,
    )

    print("  pointcloud single point at (0.5, 0, 0):")
    _render_grid(grid, ROWS, COLS)

    var n_occ = _count_values(grid, Int32(100))
    assert_true(
        n_occ >= 1 and n_occ <= 3,
        "pc_single: expected 1-3 OCCUPIED cells, got " + String(n_occ),
    )
    print("  PASS: test_pc_visual_single_point (n_occ=", n_occ, ")")


def test_pc_visual_circle(ctx: DeviceContext) raises:
    """31 points evenly spaced on a circle of radius 0.5 m, 64 angular bins.
    Renders to a ring of OCCUPIED cells at ~5 cells from centre."""
    comptime ROWS = 21
    comptime COLS = 21
    # Match #points to #bins so every bin gets a real range — otherwise the
    # leftover bins keep `range_max` and the raycast kernel sweeps the whole
    # grid for those rays, marking corners EMPTY. (Same shape as the
    # laserscan circle test above.)
    comptime NUM_PTS = 63
    comptime NUM_BINS = 63
    comptime RADIUS = 0.5

    # Place each point at the CENTER of its angular bin: (i + 0.5) * 2π/N.
    # If we placed them at bin starts (i * 2π/N), µrad-level polynomial-atan2
    # error at the boundary could flip individual points into bin i-1, leaving
    # bin i with range_max=20 — which the raycast then sweeps to the grid
    # edge, marking corner cells EMPTY. Bin-centre placement is sound and
    # reproduces the laserscan circle's "no dots outside the ring" pattern.
    var pts = List[Float32]()
    for i in range(NUM_PTS):
        var theta = Float64(2.0) * pi * (Float64(i) + 0.5) / Float64(NUM_PTS)
        pts.append(Float32(RADIUS * cos(theta)))
        pts.append(Float32(RADIUS * sin(theta)))
        pts.append(Float32(0.0))

    var grid = _run_pc_mapper(
        ctx, pts, NUM_PTS,
        ROWS, COLS,
        Float32(0.1), Float32(0.0),
        Float32(0.0), Float32(0.0), Float32(0.0),
        NUM_BINS, 32,
    )

    print("  pointcloud circle radius 0.5 (", NUM_PTS, "pts,", NUM_BINS, "bins):")
    _render_grid(grid, ROWS, COLS)

    var n_occ = _count_values(grid, Int32(100))
    var n_empty = _count_values(grid, Int32(0))
    assert_true(
        n_occ > 5 and n_occ < 40,
        "pc_circle: expected 5 < n_occ < 40, got " + String(n_occ),
    )
    assert_true(n_empty > 0, "pc_circle: expected some EMPTY cells")

    var corner_idx = 0 + 0 * ROWS
    assert_true(
        grid[corner_idx] == Int32(-1),
        "pc_circle: top-left corner should remain UNEXPLORED",
    )
    print("  PASS: test_pc_visual_circle (n_occ=", n_occ, "n_empty=", n_empty, ")")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def main() raises:
    var ctx = DeviceContext()
    print("Running local mapper tests on:", ctx.name())

    test_single_ray(ctx)
    test_four_cardinal_rays(ctx)
    test_circle_radius_0_5(ctx)
    test_circle_radius_2_0(ctx)
    test_out_of_bounds_range(ctx)

    print("--- pointcloud kernel tests ---")
    test_pc_single_point_in_known_bin(ctx)
    test_pc_atomic_min_keeps_closer(ctx)
    test_pc_z_filter_rejects(ctx)
    test_pc_origin_filter_rejects(ctx)
    test_pc_atan2_quadrants(ctx)

    print("--- pointcloud end-to-end visual ---")
    test_pc_visual_single_point(ctx)
    test_pc_visual_circle(ctx)

    print("All tests passed.")
