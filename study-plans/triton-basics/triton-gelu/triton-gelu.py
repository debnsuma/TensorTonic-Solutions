import torch
import triton
import triton.language as tl


@triton.jit
def gelu_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Write code here
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offset = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offset < n

    x = tl.load(x_ptr+offset, mask=mask)
    # out = 0.5 * x * (1.0 + tl.math.erf(x / tl.math.sqrt(2.0)))
    out = 0.5 * x * (1.0 + tl.math.erf(x *  0.707106771))
    
    tl.store(out_ptr + offset, out, mask=mask)


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch gelu_kernel: out = 0.5 * x * (1 + erf(x / sqrt(2)))."""
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    gelu_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)