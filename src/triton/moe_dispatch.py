import triton
import triton.language as tl
import torch

@triton.jit
def moe_dispatch_kernel(
    x_ptr,             # pointer to [N, D] input
    idx_ptr,           # pointer to [N, k] expert indices
    out_ptrs,          # pointer array [k] of output buffers
    N, D, k, 
    stride_xn, stride_xd, 
    stride_yn, stride_yd
):
    """Dispatch tokens to expert buffers based on routing indices."""
    # block dimensions
    pid = tl.program_id(0)      # which token (0 … N-1)
    eid = tl.program_id(1)      # which expert (0 … k-1)
    
    # bounds check
    if pid >= N or eid >= k:
        return
    
    # load the token vector
    offs_x = pid * stride_xn + tl.arange(0, D) * stride_xd
    x = tl.load(x_ptr + offs_x)  # [D]
    
    # load expert index for this token
    idx = tl.load(idx_ptr + pid * k + eid)
    
    # compute pointer into the right expert buffer
    out_offs = idx * (N//k) * stride_yn + pid * stride_yn + tl.arange(0, D) * stride_yd
    
    # scatter the token into that expert's private buffer
    tl.store(out_ptrs[idx] + out_offs, x)

@triton.jit
def moe_expert_fuse_kernel(
    buf_ptr,           # [k, M, D] expert buffers
    wu_ptr, bu_ptr,    # up projection weights and bias
    wd_ptr, bd_ptr,    # down projection weights and bias
    out_ptr,           # [k, M, D] output
    k, M, D,
    stride_bk, stride_bm, stride_bd,
    stride_wud, stride_bud, stride_wdd, stride_bdd,
    stride_ok, stride_om, stride_od
):
    """Fused expert computation: up projection + activation + down projection."""
    eid = tl.program_id(0)  # which expert
    mid = tl.program_id(1)  # which token within expert
    
    # bounds check
    if eid >= k or mid >= M:
        return
    
    # load input token
    off_buf = eid * stride_bk + mid * stride_bm + tl.arange(0, D) * stride_bd
    x = tl.load(buf_ptr + off_buf)  # [D]
    
    # FFN Up: x @ W_up + b_up
    wu_offs = eid * stride_wud + tl.arange(0, D) * stride_wud
    wu = tl.load(wu_ptr + wu_offs)  # [D]
    bu = tl.load(bu_ptr + eid)      # scalar
    z = tl.dot(x, wu) + bu
    
    # Activation (SiLU/GELU)
    z = tl.silu(z)  # or tl.gelu(z) depending on your activation
    
    # FFN Down: z @ W_down + b_down
    wd_offs = eid * stride_wdd + tl.arange(0, D) * stride_wdd
    wd = tl.load(wd_ptr + wd_offs)  # [D]
    bd = tl.load(bd_ptr + eid)      # scalar
    y = tl.dot(z, wd) + bd
    
    # store result
    off_out = eid * stride_ok + mid * stride_om + tl.arange(0, D) * stride_od
    tl.store(out_ptr + off_out, y)

@triton.jit
def moe_combine_kernel(
    expert_outputs_ptr,  # [k, M, D] expert outputs
    weights_ptr,         # [N, k] routing weights
    idx_ptr,             # [N, k] expert indices
    out_ptr,             # [N, D] final output
    N, D, k,
    stride_eok, stride_eom, stride_eod,
    stride_wn, stride_wk,
    stride_idxn, stride_idxk,
    stride_on, stride_od
):
    """Combine expert outputs using routing weights."""
    pid = tl.program_id(0)  # which token
    did = tl.program_id(1)  # which dimension
    
    # bounds check
    if pid >= N or did >= D:
        return
    
    # accumulate weighted expert outputs
    acc = 0.0
    for eid in range(k):
        # load routing weight
        weight = tl.load(weights_ptr + pid * stride_wn + eid * stride_wk)
        
        # load expert index
        expert_idx = tl.load(idx_ptr + pid * stride_idxn + eid * stride_idxk)
        
        # load expert output
        expert_off = expert_idx * stride_eok + pid * stride_eom + did * stride_eod
        expert_out = tl.load(expert_outputs_ptr + expert_off)
        
        # accumulate
        acc += weight * expert_out
    
    # store final output
    out_off = pid * stride_on + did * stride_od
    tl.store(out_ptr + out_off, acc)

def moe_dispatch_triton(x, expert_indices, expert_buffers, k):
    """Python wrapper for MoE dispatch kernel."""
    N, D = x.shape
    
    # Launch dispatch kernel
    grid = (N, k)
    moe_dispatch_kernel[grid](
        x_ptr=x,
        idx_ptr=expert_indices,
        out_ptrs=expert_buffers,
        N=N, D=D, k=k,
        stride_xn=x.stride(0), stride_xd=x.stride(1),
        stride_yn=expert_buffers[0].stride(0), stride_yd=expert_buffers[0].stride(1)
    )

def moe_expert_fuse_triton(expert_buffers, weights_up, bias_up, weights_down, bias_down, k):
    """Python wrapper for fused expert computation."""
    M, D = expert_buffers[0].shape
    
    # Launch expert fusion kernel
    grid = (k, M)
    moe_expert_fuse_kernel[grid](
        buf_ptr=expert_buffers,
        wu_ptr=weights_up, bu_ptr=bias_up,
        wd_ptr=weights_down, bd_ptr=bias_down,
        out_ptr=expert_buffers,  # in-place computation
        k=k, M=M, D=D,
        stride_bk=expert_buffers[0].stride(0), stride_bm=expert_buffers[0].stride(1), stride_bd=1,
        stride_wud=weights_up.stride(0), stride_bud=1,
        stride_wdd=weights_down.stride(0), stride_bdd=1,
        stride_ok=expert_buffers[0].stride(0), stride_om=expert_buffers[0].stride(1), stride_od=1
    )

def moe_combine_triton(expert_outputs, routing_weights, expert_indices, output, k):
    """Python wrapper for MoE combine kernel."""
    N, D = output.shape
    
    # Launch combine kernel
    grid = (N, D)
    moe_combine_kernel[grid](
        expert_outputs_ptr=expert_outputs,
        weights_ptr=routing_weights,
        idx_ptr=expert_indices,
        out_ptr=output,
        N=N, D=D, k=k,
        stride_eok=expert_outputs.stride(0), stride_eom=expert_outputs.stride(1), stride_eod=expert_outputs.stride(2),
        stride_wn=routing_weights.stride(0), stride_wk=routing_weights.stride(1),
        stride_idxn=expert_indices.stride(0), stride_idxk=expert_indices.stride(1),
        stride_on=output.stride(0), stride_od=output.stride(1)
    ) 