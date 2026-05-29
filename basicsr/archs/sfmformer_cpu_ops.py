from __future__ import annotations
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.utils import _pair

# -----------------------------------------------------------------------------
# Optional CUDA fast-path. If the compiled extensions are present we keep using
# them on GPU; on CPU we always use the PyTorch implementation below.
# -----------------------------------------------------------------------------
try:
    import smm_cuda  # type: ignore
    _HAS_SMM_CUDA = True
except Exception:
    _HAS_SMM_CUDA = False


# =============================================================================
# 1) Sparse Matrix Multiplication -- QmK
# =============================================================================

def _smm_qmk_pytorch(A: Tensor, B: Tensor, index: Tensor,
                     query_chunk: int = 128) -> Tensor:
    """Pure-PyTorch implementation of SMM_QmK_forward_cuda.

    Math
    ----
        out[b, i, k] = sum_c  A[b, i, c] * B[b, c, index[b, i, k]]

    Shapes
    ------
        A     : (Bh, N, C)
        B     : (Bh, C, N)         -- note: this is K already transposed
        index : (Bh, N, K)  int32/int64
        out   : (Bh, N, K)

    Memory
    ------
        Peak temporary is  (Bh, query_chunk, K, C).  Tune `query_chunk` based
        on available RAM.  For PFT-light on Pi 5 (16 GB), 128 is safe for all
        blocks; drop to 32 if you see OOM on large input patches.
    """
    Bh, N, C = A.shape
    K = index.size(-1)
    assert B.shape == (Bh, C, N), f"B shape mismatch: {B.shape}, expected {(Bh, C, N)}"
    assert index.shape[:2] == (Bh, N), f"index shape mismatch: {index.shape}"

    # torch.gather needs int64
    idx = index.long()

    # Transpose B -> (Bh, N, C) so we can gather whole rows (one row == one
    # "column" of the original B, i.e. one key vector).
    B_rows = B.transpose(1, 2).contiguous()            # (Bh, N, C)

    # Accumulate results chunk-by-chunk along the query dim to cap peak memory.
    out_chunks = []
    for s in range(0, N, query_chunk):
        e = min(s + query_chunk, N)
        q = e - s

        idx_chunk = idx[:, s:e, :]                     # (Bh, q, K)

        # Flatten (q, K) -> q*K so a single gather call pulls all required
        # key rows at once.  Then expand to the C dim for the actual gather.
        idx_flat = idx_chunk.reshape(Bh, q * K)        # (Bh, q*K)
        idx_exp  = idx_flat.unsqueeze(-1).expand(-1, -1, C)   # (Bh, q*K, C)
        B_gath   = torch.gather(B_rows, 1, idx_exp)    # (Bh, q*K, C)
        B_gath   = B_gath.view(Bh, q, K, C)            # (Bh, q, K, C)

        # out[b, i, k] = <A[b, i, :], B_gath[b, i, k, :]>
        # A[:, s:e, :].unsqueeze(2) : (Bh, q, 1, C)
        chunk_out = (A[:, s:e, :].unsqueeze(2) * B_gath).sum(dim=-1)  # (Bh, q, K)
        out_chunks.append(chunk_out)

    return torch.cat(out_chunks, dim=1)                # (Bh, N, K)


def smm_qmk(A: Tensor, B: Tensor, index: Tensor,
            query_chunk: int = 128) -> Tensor:
    """Device-aware dispatcher for SMM_QmK.

    Uses the compiled CUDA kernel when available and inputs are on CUDA,
    otherwise falls back to the pure-PyTorch path.
    """
    A = A.contiguous()
    B = B.contiguous()
    index = index.contiguous()
    if A.is_cuda and _HAS_SMM_CUDA:
        # The original kernel expects int32 index.
        return smm_cuda.SMM_QmK_forward_cuda(A, B, index.int())
    return _smm_qmk_pytorch(A, B, index, query_chunk=query_chunk)


# =============================================================================
# 2) Sparse Matrix Multiplication -- AmV
# =============================================================================

def _smm_amv_pytorch(A: Tensor, V: Tensor, index: Tensor,
                     query_chunk: int = 128) -> Tensor:
    """Pure-PyTorch implementation of SMM_AmV_forward_cuda.

    Math
    ----
        out[b, i, c] = sum_k  A[b, i, k] * V[b, index[b, i, k], c]

    Shapes
    ------
        A     : (Bh, N, K)    attention weights
        V     : (Bh, N, C)    value vectors
        index : (Bh, N, K)    int
        out   : (Bh, N, C)
    """
    Bh, N, K = A.shape
    C = V.size(-1)
    assert V.shape[:2] == (Bh, N), f"V shape mismatch: {V.shape}"
    assert index.shape == (Bh, N, K), f"index shape mismatch: {index.shape}"

    idx = index.long()

    out_chunks = []
    for s in range(0, N, query_chunk):
        e = min(s + query_chunk, N)
        q = e - s

        idx_chunk = idx[:, s:e, :]                     # (Bh, q, K)
        idx_flat  = idx_chunk.reshape(Bh, q * K)       # (Bh, q*K)
        idx_exp   = idx_flat.unsqueeze(-1).expand(-1, -1, C)  # (Bh, q*K, C)
        V_gath    = torch.gather(V, 1, idx_exp)        # (Bh, q*K, C)
        V_gath    = V_gath.view(Bh, q, K, C)           # (Bh, q, K, C)

        # Weighted sum over K.
        # A[:, s:e, :].unsqueeze(-1): (Bh, q, K, 1)
        chunk_out = (A[:, s:e, :].unsqueeze(-1) * V_gath).sum(dim=2)  # (Bh, q, C)
        out_chunks.append(chunk_out)

    return torch.cat(out_chunks, dim=1)                # (Bh, N, C)


def smm_amv(A: Tensor, V: Tensor, index: Tensor,
            query_chunk: int = 128) -> Tensor:
    """Device-aware dispatcher for SMM_AmV."""
    A = A.contiguous()
    V = V.contiguous()
    index = index.contiguous()
    if A.is_cuda and _HAS_SMM_CUDA:
        return smm_cuda.SMM_AmV_forward_cuda(A, V, index.int())
    return _smm_amv_pytorch(A, V, index, query_chunk=query_chunk)


# -----------------------------------------------------------------------------
# autograd.Function wrappers -- drop-in replacement for original SMM_QmK /
# SMM_AmV classes. Autograd flows naturally through the PyTorch ops on CPU,
# but we keep the Function wrapper so that on CUDA we can still call the
# compiled backward kernels (they are much faster than gather+matmul gradients
# for the backward pass of top-k sparse attention).
# -----------------------------------------------------------------------------

from torch.autograd import Function
from torch.autograd.function import once_differentiable


class SMM_QmK(Function):
    """Drop-in replacement for the original SMM_QmK autograd.Function."""

    @staticmethod
    def forward(ctx, A, B, index):
        ctx.save_for_backward(A, B, index)
        ctx._on_cuda = A.is_cuda and _HAS_SMM_CUDA
        if ctx._on_cuda:
            return smm_cuda.SMM_QmK_forward_cuda(
                A.contiguous(), B.contiguous(), index.contiguous().int()
            )
        # CPU path: relies on _smm_qmk_pytorch, but because we're inside a
        # custom Function we must implement backward manually for CPU too.
        return _smm_qmk_pytorch(A.contiguous(), B.contiguous(), index.contiguous())

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        A, B, index = ctx.saved_tensors
        if ctx._on_cuda:
            grad_A, grad_B = smm_cuda.SMM_QmK_backward_cuda(
                grad_output.contiguous(), A.contiguous(),
                B.contiguous(), index.contiguous().int(),
            )
            return grad_A, grad_B, None
        # CPU backward derived from the forward math:
        #   out[b,i,k] = sum_c A[b,i,c] * B[b,c,index[b,i,k]]
        # => dA[b,i,c] = sum_k grad_out[b,i,k] * B[b,c,index[b,i,k]]
        # => dB[b,c,j] = sum over (i,k s.t. index[b,i,k]==j) of
        #                   grad_out[b,i,k] * A[b,i,c]
        grad_A, grad_B = _smm_qmk_backward_pytorch(grad_output, A, B, index)
        return grad_A, grad_B, None


class SMM_AmV(Function):
    """Drop-in replacement for the original SMM_AmV autograd.Function."""

    @staticmethod
    def forward(ctx, A, V, index):
        ctx.save_for_backward(A, V, index)
        ctx._on_cuda = A.is_cuda and _HAS_SMM_CUDA
        if ctx._on_cuda:
            return smm_cuda.SMM_AmV_forward_cuda(
                A.contiguous(), V.contiguous(), index.contiguous().int()
            )
        return _smm_amv_pytorch(A.contiguous(), V.contiguous(), index.contiguous())

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        A, V, index = ctx.saved_tensors
        if ctx._on_cuda:
            grad_A, grad_V = smm_cuda.SMM_AmV_backward_cuda(
                grad_output.contiguous(), A.contiguous(),
                V.contiguous(), index.contiguous().int(),
            )
            return grad_A, grad_V, None
        grad_A, grad_V = _smm_amv_backward_pytorch(grad_output, A, V, index)
        return grad_A, grad_V, None


# -----------------------------------------------------------------------------
# CPU-side manual backward.  Only needed if you ever train/finetune on CPU,
# which you probably won't.  They use scatter_add for the "accumulate into
# the key/value positions" part.
# -----------------------------------------------------------------------------

def _smm_qmk_backward_pytorch(grad_out: Tensor, A: Tensor, B: Tensor,
                              index: Tensor, query_chunk: int = 128
                              ) -> Tuple[Tensor, Tensor]:
    """Backward for SMM_QmK on CPU.

    grad_out : (Bh, N, K)
    returns  : grad_A (Bh, N, C),  grad_B (Bh, C, N)
    """
    Bh, N, C = A.shape
    K = index.size(-1)
    idx = index.long()

    grad_A = torch.zeros_like(A)
    # Accumulate grad_B in the transposed layout (Bh, N, C), then transpose.
    grad_B_rows = torch.zeros(Bh, N, C, device=B.device, dtype=B.dtype)
    B_rows = B.transpose(1, 2).contiguous()

    for s in range(0, N, query_chunk):
        e = min(s + query_chunk, N)
        q = e - s
        idx_chunk = idx[:, s:e, :]                              # (Bh, q, K)
        idx_flat  = idx_chunk.reshape(Bh, q * K)
        idx_exp   = idx_flat.unsqueeze(-1).expand(-1, -1, C)    # (Bh, q*K, C)

        # Gather the same key rows as in forward.
        B_gath = torch.gather(B_rows, 1, idx_exp).view(Bh, q, K, C)
        g_out  = grad_out[:, s:e, :]                            # (Bh, q, K)

        # grad_A[b,i,:] = sum_k g_out[b,i,k] * B_gath[b,i,k,:]
        grad_A[:, s:e, :] = (g_out.unsqueeze(-1) * B_gath).sum(dim=2)

        # grad_B_rows accumulates at positions index[b,i,k]:
        #   contribution from (b,i,k) is g_out[b,i,k] * A[b,i,:]
        contrib = (g_out.unsqueeze(-1) * A[:, s:e, :].unsqueeze(2))  # (Bh, q, K, C)
        contrib = contrib.reshape(Bh, q * K, C)
        grad_B_rows.scatter_add_(1, idx_exp, contrib)

    grad_B = grad_B_rows.transpose(1, 2).contiguous()           # (Bh, C, N)
    return grad_A, grad_B


def _smm_amv_backward_pytorch(grad_out: Tensor, A: Tensor, V: Tensor,
                              index: Tensor, query_chunk: int = 128
                              ) -> Tuple[Tensor, Tensor]:
    """Backward for SMM_AmV on CPU.

    grad_out : (Bh, N, C)
    returns  : grad_A (Bh, N, K),  grad_V (Bh, N, C)
    """
    Bh, N, K = A.shape
    C = V.size(-1)
    idx = index.long()

    grad_A = torch.zeros_like(A)
    grad_V = torch.zeros_like(V)

    for s in range(0, N, query_chunk):
        e = min(s + query_chunk, N)
        q = e - s
        idx_chunk = idx[:, s:e, :]                          # (Bh, q, K)
        idx_flat  = idx_chunk.reshape(Bh, q * K)
        idx_exp   = idx_flat.unsqueeze(-1).expand(-1, -1, C)

        V_gath = torch.gather(V, 1, idx_exp).view(Bh, q, K, C)
        g_out  = grad_out[:, s:e, :]                        # (Bh, q, C)

        # grad_A[b,i,k] = sum_c g_out[b,i,c] * V_gath[b,i,k,c]
        grad_A[:, s:e, :] = (g_out.unsqueeze(2) * V_gath).sum(dim=-1)

        # grad_V accumulates at positions index[b,i,k]:
        #   contribution (b,i,k) -> A[b,i,k] * g_out[b,i,:]
        a_chunk = A[:, s:e, :]                              # (Bh, q, K)
        contrib = (a_chunk.unsqueeze(-1) * g_out.unsqueeze(2))  # (Bh, q, K, C)
        contrib = contrib.reshape(Bh, q * K, C)
        grad_V.scatter_add_(1, idx_exp, contrib)

    return grad_A, grad_V


# =============================================================================
# 3) IDynamic dynamic depth-wise convolution
# =============================================================================

def _idynamic_pytorch(inp: Tensor, weight: Tensor,
                      stride=1, padding=0, dilation=1) -> Tensor:
    """Pure-PyTorch implementation of _idynamic_cuda.

    Matches the CUDA kernel indexing exactly:

        out[b, c, h, w] = sum_{kh, kw} weight[b, g, kh, kw, h, w] *
                                       inp[b, c, h_in, w_in]
        where g     = c // (C // groups)   (which group channel c belongs to)
              h_in  = h*stride_h - pad_h + kh*dilation_h
              w_in  = w*stride_w - pad_w + kw*dilation_w
        (out-of-bounds input positions are treated as zero.)

    Shapes
    ------
        inp    : (B, C, H_in, W_in)
        weight : (B, groups, kH, kW, H_out, W_out)
        out    : (B, C, H_out, W_out)

    Implementation uses F.unfold to extract all kH*kW patches around each
    output position, then does an element-wise multiply with `weight` and
    sums over the kernel dim.
    """
    stride   = _pair(stride)
    padding  = _pair(padding)
    dilation = _pair(dilation)

    B, C, H_in, W_in = inp.shape
    B2, groups, kH, kW, H_out, W_out = weight.shape
    assert B == B2, f"batch mismatch: inp {B} vs weight {B2}"
    assert C % groups == 0, f"C={C} not divisible by groups={groups}"
    gc = C // groups                                   # channels per group

    # F.unfold -> (B, C*kH*kW, L) where L = H_out_unfold * W_out_unfold.
    # With stride=1 and padding=(k-1)//2 we get L == H_in*W_in == H_out*W_out.
    unfolded = F.unfold(
        inp, kernel_size=(kH, kW),
        stride=stride, padding=padding, dilation=dilation,
    )                                                  # (B, C*kH*kW, L)
    L = unfolded.size(-1)
    assert L == H_out * W_out, (
        f"Unfold produced L={L}, but weight expects H_out*W_out={H_out*W_out}. "
        "Check stride/padding/dilation consistency with the weight tensor."
    )

    # Reshape so we can broadcast weight (which has no gc dim) against inp
    # (which has both groups and gc).
    unfolded = unfolded.view(B, groups, gc, kH * kW, L) # (B, G, gc, kHW, L)
    w_flat   = weight.view(B, groups, kH * kW, L)       # (B, G, kHW, L)
    w_exp    = w_flat.unsqueeze(2)                      # (B, G, 1, kHW, L)

    out = (unfolded * w_exp).sum(dim=3)                 # (B, G, gc, L)
    out = out.view(B, C, H_out, W_out)
    return out


def idynamic_conv(inp: Tensor, weight: Tensor,
                  bias: Optional[Tensor] = None,
                  stride: Union[int, Tuple[int, int]] = 1,
                  padding: Union[int, Tuple[int, int]] = 0,
                  dilation: Union[int, Tuple[int, int]] = 1) -> Tensor:
    """Drop-in replacement for `_idynamic_cuda` in idynamicdwconv_util.py.

    Works on both CPU and CUDA.  Shape assertions mirror the original helper.
    """
    assert inp.size(0) == weight.size(0), "batch size mismatch"
    # The original checks `input.size(-2)//stride == weight.size(-2)`; we
    # keep the semantic check but make it stride-aware via _pair.
    stride_pair = _pair(stride)
    assert inp.size(-2) // stride_pair[0] == weight.size(-2), \
        f"H mismatch: inp H={inp.size(-2)} / stride={stride_pair[0]} != weight H_out={weight.size(-2)}"
    assert inp.size(-1) // stride_pair[1] == weight.size(-1), \
        f"W mismatch: inp W={inp.size(-1)} / stride={stride_pair[1]} != weight W_out={weight.size(-1)}"

    out = _idynamic_pytorch(inp, weight,
                            stride=stride, padding=padding, dilation=dilation)
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)
    return out
