# Complexity Deep

Complexity architecture with **INL Dynamics** for robotics-grade control.

[![PyPI version](https://badge.fury.io/py/complexity-deep.svg)](https://badge.fury.io/py/complexity-deep)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

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
2. **Token-Routed MLP** (transformation) - feature processing
3. **INL Dynamics** (control) - trajectory smoothing

### INL Dynamics Equations

```python
error = h - mu                      # deviation from equilibrium
v_next = alpha * v - beta * error   # velocity update (momentum + correction)
h_next = h + dt * gate * v_next     # position update
```

Where:
- `h` = hidden state
- `v` = velocity (momentum)
- `mu` = learnable equilibrium point
- `alpha` = inertia (0.9 default)
- `beta` = correction strength (0.1 default)
- `gate` = amplitude control
- `dt` = integration timestep

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
2. **Stable Attractors** - The learnable mu provides stable equilibrium points
3. **Momentum** - Alpha parameter allows the model to maintain momentum
4. **Error Correction** - Beta parameter provides corrective feedback

This is particularly useful for:
- Generating coherent long sequences
- Maintaining consistency in outputs
- Smooth interpolation in latent space

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

## Related Packages

- **complexity** - Base architecture without Dynamics
- **complexity-diffusion** - DiT for image generation
- **pyllm-inference** - Inference server with streaming

## License

MIT
