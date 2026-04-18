# Unit tests for the LocalMapper GPU kernel (scan_to_grid).
#
# Each test builds a small deterministic scan, launches the kernel directly,
# reads the occupancy grid back, and asserts against hand-computed expected
# values.
# Grid layout: 21x21 cells, 0.1 m resolution — 2.1 m x 2.1 m window. With
# laserscan_pos = (0, 0, 0) and central = (round(21/2)-1, round(21/2)-1) =
# (9, 9), the robot sits at cell (9, 9).

from std.testing import assert_true, assert_almost_equal
from std.math import sqrt, cos, sin, pi
from std.memory import UnsafePointer
from std.gpu.host import DeviceContext, DeviceBuffer

from kompass_mojo.local_mapper import (
    OCC_UNEXPLORED,
    OCC_EMPTY,
    OCC_OCCUPIED,
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
    var ranges = ctx.enqueue_create_buffer[DType.float64](scan_size)

    var central_x = Int32(rows // 2 - 1)
    var central_y = Int32(cols // 2 - 1)
    var start_x = central_x + Int32(Int(lpos_x / resolution))
    var start_y = central_y + Int32(Int(lpos_y / resolution))

    _fill_distances(
        ctx, distances, rows, cols, resolution,
        lpos_x, lpos_y, lpos_z, central_x, central_y,
    )
    _fill_f64(ctx, angles, angles_vals)
    _fill_f64(ctx, ranges, ranges_vals)
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

    print("All tests passed.")
