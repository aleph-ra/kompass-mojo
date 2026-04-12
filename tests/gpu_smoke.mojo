from std.sys import has_accelerator
from std.gpu.host import DeviceContext
from std.gpu import global_idx
from std.memory import UnsafePointer


fn add_kernel(
    a: UnsafePointer[Float32, MutAnyOrigin],
    b: UnsafePointer[Float32, MutAnyOrigin],
    c: UnsafePointer[Float32, MutAnyOrigin],
    n: Int,
):
    var tid = Int(global_idx.x)
    if tid < n:
        c[tid] = a[tid] + b[tid]


def main():
    var ctx = DeviceContext()
    print("device:", ctx.name())

    comptime N = 256
    var a = ctx.enqueue_create_buffer[DType.float32](N)
    var b = ctx.enqueue_create_buffer[DType.float32](N)
    var c = ctx.enqueue_create_buffer[DType.float32](N)
    a.enqueue_fill(Float32(1.0))
    b.enqueue_fill(Float32(2.0))

    ctx.enqueue_function[add_kernel, add_kernel](
        a.unsafe_ptr(), b.unsafe_ptr(), c.unsafe_ptr(), N,
        grid_dim=1,
        block_dim=N,
    )

    ctx.synchronize()
    with c.map_to_host() as host:
        print("c[0]:", host[0], "c[255]:", host[255])
