# Complexity Deep

LLM architecture with **Token-Routed MLP**, **Mu-Guided Attention**, and **Shared Lexical Expert**.

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

## Architecture

Each layer has 3 components:

```
Input
  |
  v
[RMSNorm] -> [Mu-Guided GQA] -> residual -> [RMSNorm] -> [Token-Routed MLP + Shared Expert] -> residual
                  ^                                                |
                  |                                                v
             mu_prev                                         MuGuidance(h_post_mlp)
        (from previous layer)                                      |
                                                              mu_contextual
                                                          (to next layer)
```

### 1. Token-Routed MLP (Deterministic MoE)

Tokens are routed to experts based on their token ID, not learned softmax routing:

```python
expert_id = BinPack(token_id, frequencies)  # greedy bin-packing
output = SharedMLP(x) + Expert_e(x)         # shared + specialized
```

**Zipf-balanced greedy bin-packing**: tokens sorted by corpus frequency are assigned to the expert with the lowest accumulated load. Achieves **perfect 1.0000x load balance** (vs 1.38x with round-robin).

**Shared Lexical Expert**: a dense SwiGLU MLP that ALL tokens pass through, capturing universal patterns (syntax, grammar). Routed experts specialize on their lexical subsets.

**Sparse dispatch**: only routed tokens are computed per expert (no masked waste).

### 2. Mu-Guided Attention

A learnable latent state mu flows between layers, biasing K, Q, V projections:

```python
K = x @ W_K + mu_prev @ W_muK
Q = x @ W_Q + mu_prev @ W_muQ
V = x @ W_V + mu_prev @ W_muV
```

**mu_init**: learnable parameter for layer 0 (initialized to zero), so the first layer also benefits from guidance.

**mu after MLP**: mu is produced after expert dispatch, capturing which expert processed each token. The next layer adapts its attention accordingly.

```python
mu_contextual = clamp(mu_param) + mu_proj(h_post_mlp)
```

### 3. Training Recipe

- **Dynamic warmup**: 5% of total steps (not hardcoded)
- **GPT-style init**: residual projections scaled by 1/sqrt(2*num_layers)
- **AdamW**: betas=(0.9, 0.95), weight_decay=0.1, grad_clip=1.0
- **Cosine scheduler**: min_lr = 10% of peak
- **BF16 precision**

## Usage

```python
from complexity_deep.models import ComplexityConfig, ComplexityModel

config = ComplexityConfig(
    hidden_size=768,
    num_hidden_layers=18,
    num_attention_heads=12,
    num_key_value_heads=4,
    intermediate_size=2048,
    num_experts=4,
    shared_expert=True,
    use_mu_guidance=True,
)
model = ComplexityModel(config)

# Forward
outputs = model(input_ids)
logits = outputs["logits"]

# Generate
output_ids = model.generate(input_ids, max_new_tokens=100)
```

## Key Results

At iso-param (~170M), the Token-Routed architecture converges faster than the dense baseline (avg loss 4.79 vs 4.91 over 954 steps on 500M tokens). Each expert achieves functional specialization measured by per-expert perplexity.

## Links

- **[complexity-framework](https://github.com/Complexity-ML/complexity-framework)** — Full training framework
- **Paper**: Under review at TMLR

## Citation

```bibtex
@software{peyriguere2026complexity,
  author       = {Peyriguere, Boris},
  title        = {Complexity-Deep: Token-Routed MLP with Mu-Guided Attention},
  year         = 2026,
  url          = {https://github.com/Complexity-ML/complexity-deep}
}
```

## License

CC BY-NC 4.0 (Creative Commons Attribution-NonCommercial 4.0)
