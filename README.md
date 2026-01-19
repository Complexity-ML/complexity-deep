# Complexity Deep

Complexity architecture with **INL Dynamics** for robotics-grade control.

[![PyPI version](https://badge.fury.io/py/complexity-deep.svg)](https://badge.fury.io/py/complexity-deep)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18293026.svg)](https://doi.org/10.5281/zenodo.18293026)

## Installation

```bash
pip install complexity-deep
```

## What's Different from Complexity?

Complexity Deep adds **INL Dynamics** - a robotics-inspired control layer:

```
Input -> [Attention -> MLP -> Dynamics] x N -> Output
```

The Dynamics layer provides:
- **Velocity tracking** - smooth trajectories
- **Learnable equilibrium (mu)** - stable attractors
- **Adaptive control** - alpha, beta, gate parameters

## Architecture

Each layer has 3 components:

1. **KQV Attention** (perception) - what tokens to attend to
2. **Token-Routed MLP** (transformation) - deterministic expert routing
3. **INL Dynamics** (control) - trajectory smoothing

### Token-Routed MLP (Deterministic MoE)

Unlike learned routing (Mixtral, DeepSeek), we route tokens to experts using a simple formula:

```python
expert_id = token_id % num_experts
```

**Benefits:**
- **Uniform distribution**: Each expert receives exactly 25% of tokens
- **No expert collapse**: Frequent tokens spread across all experts
- **Zero routing parameters**: No router network to learn
- **Zero load balancing loss**: Perfectly balanced by design
- **100% deterministic and parallelizable**

### INL Dynamics Equations

```python
mu_contextual = mu_base + mu_proj(h)  # contextual equilibrium (NEW!)
error = h - mu_contextual             # deviation from equilibrium
v_next = alpha * v - beta * error     # velocity update (momentum + correction)
h_next = h + dt * gate * v_next       # position update
```

Where:
- `h` = hidden state
- `v` = velocity (momentum)
- `mu_base` = global learnable equilibrium point
- `mu_proj` = context-dependent equilibrium adjustment (adapts per token)
- `alpha` = inertia (0.9 default)
- `beta` = correction strength (0.1 default)
- `gate` = amplitude control
- `dt` = integration timestep

**Contextual Mu**: Unlike a fixed equilibrium, the contextual mu adapts to each token's hidden state. This allows different tokens (code vs text vs math) to have different attractors, improving model expressivity.

## Usage

```python
from complexity_deep import DeepConfig, DeepForCausalLM, create_deep_model

# Create model by size
model = create_deep_model("base")  # ~125M params

# Or with custom config
config = DeepConfig(
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    num_key_value_heads=4,
    use_token_routed_mlp=True,
    num_experts=4,
    use_qk_norm=True,
    # INL Dynamics parameters
    dynamics_alpha=0.9,
    dynamics_beta=0.1,
    dynamics_gate=0.5,
    dynamics_dt=0.1,
)
model = DeepForCausalLM(config)

# Forward pass
outputs = model(input_ids, labels=labels)
loss = outputs.loss
```

## Model Sizes

| Size | Params | Hidden | Layers | Experts |
|------|--------|--------|--------|---------|
| tiny | ~15M | 256 | 6 | 4 |
| 20m | ~20M | 320 | 8 | 4 |
| small | ~50M | 512 | 8 | 4 |
| 150m | ~150M | 768 | 12 | 4 |
| base | ~125M | 768 | 12 | 4 |
| medium | ~350M | 1024 | 24 | 4 |
| large | ~760M | 1536 | 24 | 4 |
| 1b | ~1B | 2048 | 24 | 4 |
| 3b | ~3B | 2560 | 32 | 4 |

## Why Dynamics?

The INL Dynamics layer is inspired by robotics control theory:

1. **Smooth Trajectories** - The velocity tracking prevents sudden jumps in hidden states
2. **Contextual Attractors** - The contextual mu adapts equilibrium per token type
3. **Momentum** - Alpha parameter allows the model to maintain momentum
4. **Error Correction** - Beta parameter provides corrective feedback

This is particularly useful for:
- Generating coherent long sequences
- Maintaining consistency in outputs
- Smooth interpolation in latent space
- **Different behaviors for different content types** (code, text, math)

### Training Note

When training, exclude `mu` (the base equilibrium) from weight decay to allow it to learn freely:

```python
# In optimizer setup
if 'bias' in name or 'norm' in name or ('.mu' in name and 'mu_proj' not in name):
    no_decay_params.append(param)  # No weight decay for mu_base
else:
    decay_params.append(param)     # mu_proj.weight gets normal decay
```

## CUDA Optimizations

```python
from complexity_deep.cuda import (
    HAS_TRITON,
    get_optimization_info,
    FusedQKNormAttention,
    FusedSwiGLUMLP,
    PersistentTokenRoutedMLP,
)

# Check available optimizations
info = get_optimization_info()
print(info)
# {
#   "triton_available": True,
#   "optimizations": {
#     "fused_qk_attention": {"speedup": "15-20%"},
#     "fused_mlp": {"speedup": "20-30%"},
#     "persistent_cggr": {"speedup": "10-15%"},
#     "int8_quantization": {"speedup": "40-50%"},
#   }
# }
```

## Roadmap

Future architecture improvements under research:

| Feature | Description | Status |
|---------|-------------|--------|
| **Per-Layer Dynamics** | Different alpha/beta/gate per layer | Planned |
| **Mu Attention** | Mu as persistent memory token in attention | Research |
| **Sparse Mu** | Mu active only on top-k dimensions | Research |
| **Speculative Decoding** | 2-3x faster inference | Planned |
| **Continuous Batching** | Better throughput for serving | Planned |

## Related Packages

- **[mu-inference](https://pypi.org/project/mu-inference/)** - High-performance inference server
- **[complexity-framework](https://pypi.org/project/complexity-framework/)** - Training framework

## Citation

If you use Complexity-Deep in your research, please cite:

```bibtex
@software{peyriguere2026complexity,
  author       = {Peyriguere, Boris},
  title        = {Complexity-Deep: Token-Routed MLP with Mu-Guided Dynamics for Efficient Transformer Architectures},
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18293026},
  url          = {https://doi.org/10.5281/zenodo.18293026}
}
```

## License

CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0)

- **Research & Education**: Free to use
- **Commercial use**: Open an issue on GitHub for licensing
