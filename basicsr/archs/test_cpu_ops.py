
from __future__ import annotations
import argparse
import torch

from sfmformer_cpu_ops import (
    _smm_qmk_pytorch, _smm_amv_pytorch, _idynamic_pytorch,
    smm_qmk, smm_amv, idynamic_conv,
    SMM_QmK, SMM_AmV,
    _HAS_SMM_CUDA,
)


# -----------------------------------------------------------------------------
# Reference implementations (obvious, slow, correct)
# -----------------------------------------------------------------------------
def smm_qmk_reference(A, B, index):
    """Dense matmul then gather — O(BhN^2C) memory, slow but simple."""
    # A (Bh, N, C), B (Bh, C, N), index (Bh, N, K)
    full = A @ B                                           # (Bh, N, N)
    idx = index.long()
    return torch.gather(full, 2, idx)                      # (Bh, N, K)


def smm_amv_reference(A, V, index):
    """Gather per-query values, weighted sum."""
    Bh, N, K = A.shape
    C = V.size(-1)
    idx = index.long()
    idx_exp = idx.unsqueeze(-1).expand(-1, -1, -1, C)      # (Bh, N, K, C)
    V_exp   = V.unsqueeze(1).expand(-1, N, -1, -1)         # (Bh, N, N, C)
    V_gath  = torch.gather(V_exp, 2, idx_exp)              # (Bh, N, K, C)
    return (A.unsqueeze(-1) * V_gath).sum(dim=2)           # (Bh, N, C)


def idynamic_reference(inp, weight, stride=1, padding=0, dilation=1):
    """Python loop over kernel positions — slow, transparent."""
    from torch.nn.modules.utils import _pair
    import torch.nn.functional as F
    sH, sW = _pair(stride)
    pH, pW = _pair(padding)
    dH, dW = _pair(dilation)

    B, C, H_in, W_in = inp.shape
    B2, G, kH, kW, H_out, W_out = weight.shape
    gc = C // G

    # Pad input so we can index freely.
    padded = F.pad(inp, (pW, pW, pH, pH))                  # (B, C, H_in+2pH, W_in+2pW)

    out = torch.zeros(B, C, H_out, W_out, dtype=inp.dtype, device=inp.device)
    for kh in range(kH):
        for kw in range(kW):
            # Sliding window positions in the padded tensor
            h_start = kh * dH
            w_start = kw * dW
            patch = padded[:, :,
                           h_start:h_start + sH * H_out:sH,
                           w_start:w_start + sW * W_out:sW]   # (B, C, H_out, W_out)
            # Broadcast weight[:, g, kh, kw, :, :] across channels-in-group
            w = weight[:, :, kh, kw, :, :]                    # (B, G, H_out, W_out)
            w = w.unsqueeze(2).expand(B, G, gc, H_out, W_out).reshape(B, C, H_out, W_out)
            out = out + w * patch
    return out


# -----------------------------------------------------------------------------
# Test cases
# -----------------------------------------------------------------------------
def test_smm_qmk():
    print("[SMM_QmK] forward numerical equivalence …")
    torch.manual_seed(0)
    # Realistic PFT-light shapes: window=32 -> N=1024, C_h=13, K=64
    Bh, N, C, K = 8, 1024, 13, 64
    A = torch.randn(Bh, N, C, dtype=torch.float64)
    B = torch.randn(Bh, C, N, dtype=torch.float64)
    index = torch.randint(0, N, (Bh, N, K), dtype=torch.int64)

    out_ref  = smm_qmk_reference(A, B, index)
    out_fast = _smm_qmk_pytorch(A, B, index, query_chunk=128)
    err = (out_ref - out_fast).abs().max().item()
    print(f"  max abs error (fp64): {err:.2e}")
    assert err < 1e-10, f"SMM_QmK forward failed, err={err}"

    # Gradcheck: verifies backward is mathematically correct to finite diff.
    print("[SMM_QmK] autograd.gradcheck …")
    Bh, N, C, K = 2, 16, 4, 5
    A = torch.randn(Bh, N, C, dtype=torch.float64, requires_grad=True)
    B = torch.randn(Bh, C, N, dtype=torch.float64, requires_grad=True)
    index = torch.randint(0, N, (Bh, N, K), dtype=torch.int64)
    ok = torch.autograd.gradcheck(
        lambda a, b: _smm_qmk_pytorch(a, b, index),
        (A, B), eps=1e-6, atol=1e-5,
    )
    print(f"  gradcheck: {ok}")

    # Also test the autograd.Function wrapper on CPU.
    print("[SMM_QmK] Function wrapper forward matches pure function …")
    out1 = _smm_qmk_pytorch(A, B, index)
    out2 = SMM_QmK.apply(A, B, index)
    err = (out1 - out2).abs().max().item()
    print(f"  max abs error: {err:.2e}")
    assert err < 1e-10


def test_smm_amv():
    print("[SMM_AmV] forward numerical equivalence …")
    torch.manual_seed(1)
    Bh, N, K, C = 8, 1024, 64, 13
    A = torch.randn(Bh, N, K, dtype=torch.float64)
    V = torch.randn(Bh, N, C, dtype=torch.float64)
    index = torch.randint(0, N, (Bh, N, K), dtype=torch.int64)

    out_ref  = smm_amv_reference(A, V, index)
    out_fast = _smm_amv_pytorch(A, V, index, query_chunk=128)
    err = (out_ref - out_fast).abs().max().item()
    print(f"  max abs error (fp64): {err:.2e}")
    assert err < 1e-10, f"SMM_AmV forward failed, err={err}"

    print("[SMM_AmV] autograd.gradcheck …")
    Bh, N, K, C = 2, 8, 4, 3
    A = torch.randn(Bh, N, K, dtype=torch.float64, requires_grad=True)
    V = torch.randn(Bh, N, C, dtype=torch.float64, requires_grad=True)
    index = torch.randint(0, N, (Bh, N, K), dtype=torch.int64)
    ok = torch.autograd.gradcheck(
        lambda a, v: _smm_amv_pytorch(a, v, index),
        (A, V), eps=1e-6, atol=1e-5,
    )
    print(f"  gradcheck: {ok}")


def test_idynamic():
    print("[IDynamic] forward numerical equivalence …")
    torch.manual_seed(2)
    # WMA-style shapes: DWT halves resolution, so at 64x64 input the DWT
    # output is 32x32.  With WMA's reduce: C' = C // 4.
    B, C, H, W = 2, 32, 16, 16
    kH = kW = 7
    groups = 4
    weight = torch.randn(B, groups, kH, kW, H, W, dtype=torch.float64)
    inp    = torch.randn(B, C, H, W, dtype=torch.float64)

    out_ref  = idynamic_reference(inp, weight, stride=1, padding=3)
    out_fast = _idynamic_pytorch(inp, weight, stride=1, padding=3)
    err = (out_ref - out_fast).abs().max().item()
    print(f"  max abs error (fp64): {err:.2e}")
    assert err < 1e-10, f"IDynamic forward failed, err={err}"

    print("[IDynamic] autograd.gradcheck …")
    B, C, H, W = 1, 4, 5, 5
    kH = kW = 3
    groups = 2
    weight = torch.randn(B, groups, kH, kW, H, W, dtype=torch.float64,
                         requires_grad=True)
    inp    = torch.randn(B, C, H, W, dtype=torch.float64, requires_grad=True)
    ok = torch.autograd.gradcheck(
        lambda i, w: _idynamic_pytorch(i, w, stride=1, padding=1),
        (inp, weight), eps=1e-6, atol=1e-5,
    )
    print(f"  gradcheck: {ok}")


def test_cuda_parity():
    """If CUDA + compiled smm_cuda are available, compare CPU vs CUDA outputs."""
    if not (torch.cuda.is_available() and _HAS_SMM_CUDA):
        print("[CUDA parity] skipped (no CUDA / no smm_cuda extension).")
        return

    print("[CUDA parity] SMM_QmK …")
    Bh, N, C, K = 4, 256, 13, 32
    A = torch.randn(Bh, N, C, dtype=torch.float32)
    B = torch.randn(Bh, C, N, dtype=torch.float32)
    index = torch.randint(0, N, (Bh, N, K), dtype=torch.int32)
    out_cpu  = smm_qmk(A, B, index)
    out_cuda = smm_qmk(A.cuda(), B.cuda(), index.cuda()).cpu()
    err = (out_cpu - out_cuda).abs().max().item()
    print(f"  max abs error (fp32): {err:.2e}")

    print("[CUDA parity] SMM_AmV …")
    A = torch.randn(Bh, N, K, dtype=torch.float32)
    V = torch.randn(Bh, N, C, dtype=torch.float32)
    out_cpu  = smm_amv(A, V, index)
    out_cuda = smm_amv(A.cuda(), V.cuda(), index.cuda()).cpu()
    err = (out_cpu - out_cuda).abs().max().item()
    print(f"  max abs error (fp32): {err:.2e}")


def bench():
    """Rough single-thread CPU timing for the three ops at PFT-light scale."""
    import time
    torch.set_num_threads(4)   # typical on Pi 5
    torch.manual_seed(42)

    # PFT-light, typical deep-block shape with K=32
    Bh, N, C, K = 16, 1024, 13, 32
    A = torch.randn(Bh, N, C)
    B = torch.randn(Bh, C, N)
    V = torch.randn(Bh, N, C)
    W = torch.randn(Bh, N, K)
    index = torch.randint(0, N, (Bh, N, K), dtype=torch.int64)

    # Warmup
    for _ in range(3):
        _smm_qmk_pytorch(A, B, index)
        _smm_amv_pytorch(W, V, index)

    reps = 5
    t0 = time.perf_counter()
    for _ in range(reps):
        _smm_qmk_pytorch(A, B, index)
    t_qmk = (time.perf_counter() - t0) / reps * 1000

    t0 = time.perf_counter()
    for _ in range(reps):
        _smm_amv_pytorch(W, V, index)
    t_amv = (time.perf_counter() - t0) / reps * 1000

    # IDynamic bench
    Bi, Ci, Hi, Wi = 1, 32, 64, 64
    kH = 7
    g = 4
    inp = torch.randn(Bi, Ci, Hi, Wi)
    wt  = torch.randn(Bi, g, kH, kH, Hi, Wi)
    for _ in range(3):
        _idynamic_pytorch(inp, wt, padding=3)
    t0 = time.perf_counter()
    for _ in range(reps):
        _idynamic_pytorch(inp, wt, padding=3)
    t_idyn = (time.perf_counter() - t0) / reps * 1000

    print("\n[Bench] per-call CPU time (4 threads):")
    print(f"  SMM_QmK  Bh={Bh} N={N} C={C} K={K}      : {t_qmk:7.2f} ms")
    print(f"  SMM_AmV  Bh={Bh} N={N} C={C} K={K}      : {t_amv:7.2f} ms")
    print(f"  IDynamic B={Bi} C={Ci} H=W={Hi} k={kH} g={g}: {t_idyn:7.2f} ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true", help="Run timing benchmark")
    args = ap.parse_args()

    test_smm_qmk()
    test_smm_amv()
    test_idynamic()
    test_cuda_parity()
    if args.bench:
        bench()
    print("\nAll tests passed ✅")


if __name__ == "__main__":
    main()
