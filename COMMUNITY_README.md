# COMPLEXITY-DEEP: Supplementary Code

This repository contains the source code for the COMPLEXITY-DEEP architecture described in the paper.

## Structure

```
.
├── core/                     # Core model components
│   ├── attention.py          # Mu-Guided Attention implementation
│   ├── mlp.py                # SwiGLU MLP components
│   ├── layer.py              # Decoder layer with all components
│   ├── token_routed_mlp.py   # Token-Routed MLP (deterministic routing)
│   ├── normalization.py      # RMSNorm implementation
│   ├── rotary.py             # Rotary Position Embeddings (RoPE)
│   └── safety.py             # Safety clamping mechanisms
├── models/
│   ├── config.py             # Model configuration
│   ├── modeling.py           # Full model implementation
│   └── utils.py              # Utility functions
├── cuda/                     # CUDA/Triton optimizations
│   ├── triton_token_routed.py  # Triton-accelerated Token-Routed MLP
│   ├── triton_mu_qkv.py        # Triton Mu-guided attention
│   ├── fused_attention.py      # Fused attention kernels
│   ├── fused_mlp.py            # Fused MLP kernels
│   └── persistent_cggr.py      # Persistent CGGR optimization
├── training/
│   └── train_complexity.py   # Training script
├── evaluation/
│   └── run_benchmarks.py     # Benchmark evaluation script
└── configs/
    └── model_config.json     # Model configuration
```

## Key Components

### Token-Routed MLP (Deterministic MoE)

Tokens are routed to experts via Zipf-balanced bin-packing:

```python
expert_id = BinPack(token_id, frequencies)  # greedy bin-packing
output = SharedMLP(x) + Expert_e(x)         # shared + specialized
```

A shared lexical expert processes ALL tokens (universal patterns), while routed experts specialize on their lexical subsets.

### Mu-Guided Attention

A learnable latent state mu flows between layers, biasing K, Q, V projections:

```python
K = x @ W_K + mu_prev @ W_muK
Q = x @ W_Q + mu_prev @ W_muQ
V = x @ W_V + mu_prev @ W_muV
```

mu is produced after expert dispatch (MuGuidance module), so the next layer's attention adapts based on which expert processed each token.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- transformers
- datasets
- tqdm
- triton (optional, for CUDA optimizations)

## Usage

### Training
```bash
python training/train_complexity.py --size base --dataset your_dataset
```

### Evaluation
```bash
python evaluation/run_benchmarks.py --checkpoint path/to/checkpoint.pt --max-samples 500
```

## License

CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0)
