import torch
import triton
import triton.language as tl


@triton.jit
def sum_kernel(x_ptr, out_ptr, n, BLOCK_SIZE: tl.constexpr):
    # Write code here
    pid = tl.program_id(axis=0)
    start_block = pid * BLOCK_SIZE
    offset = start_block + tl.arange(0, BLOCK_SIZE)
    mask = offset < n 

    x = tl.load(x_ptr + offset, mask=mask)
    temp = tl.sum(x)

    tl.atomic_add(out_ptr, temp)
    
    
    


def solve(x: torch.Tensor, out: torch.Tensor) -> None:
    """Launch sum_kernel on the provided tensors."""
    n = x.numel()
    out.zero_()
    BLOCK_SIZE = 1024
    grid = ((n + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    sum_kernel[grid](x, out, n, BLOCK_SIZE=BLOCK_SIZE)