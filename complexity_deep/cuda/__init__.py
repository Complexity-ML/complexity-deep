"""
CUDA/Triton Accelerated Kernels for Complexity
===============================================

Optional GPU-accelerated implementations of the core operations.
These are provided as reference for reproducing the paper's throughput
numbers but are NOT required for correctness -- the pure-PyTorch
implementations in core/ produce identical outputs.

Available optimizations (when Triton is installed):
- Token-Routed MLP with CGGR (Contiguous Grouped GEMM Routing)
- Fused QK-Norm + Flash Attention
- Fused RMSNorm + MLP projections
- Fused Residual + RMSNorm
- INT8 quantization with fused GEMM
- Fused Mu-QKV (mu-guided K/Q/V in a single kernel)

Note: The ``pid`` variable used throughout these kernels refers to
``tl.program_id()`` (Triton's thread-block index), not to any PiD
controller or dynamical system.
"""

try:
    import triton
    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


def get_optimization_info() -> dict:
    """Summarize available CUDA optimizations."""
    return {
        "triton_available": HAS_TRITON,
        "optimizations": {
            "fused_qk_attention": {
                "description": "Fused QK Normalization + Flash Attention",
                "speedup": "15-20%",
                "available": HAS_TRITON,
            },
            "fused_mlp": {
                "description": "Fused RMSNorm + Gate/Up/Down Projections",
                "speedup": "20-30%",
                "available": HAS_TRITON,
            },
            "persistent_cggr": {
                "description": "Persistent kernels for Token-Routed MLP",
                "speedup": "10-15%",
                "available": HAS_TRITON,
            },
            "fused_mu_qkv": {
                "description": "Fused Mu-guided K/Q/V projection",
                "speedup": "~2x vs 6 separate matmuls",
                "available": HAS_TRITON,
            },
            "int8_quantization": {
                "description": "INT8 quantization with fused GEMM",
                "speedup": "40-50% throughput, 50% memory",
                "available": HAS_TRITON,
            },
            "fused_residual": {
                "description": "Fused Residual + RMSNorm",
                "speedup": "5-10%",
                "available": HAS_TRITON,
            },
        },
    }


__all__ = ["HAS_TRITON", "get_optimization_info"]
