import torch
import triton
import triton.language as tl


@triton.jit
def silu_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Write code here
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n
    
    gate = tl.load(x_ptr + offset, mask=mask, other=0.0)
    silu = gate * tl.sigmoid(gate)
    
    tl.store(out_ptr + offset, silu, mask=mask)


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch silu_kernel: out = x / (1 + exp(-x))."""
    n = x.numel()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    silu_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)