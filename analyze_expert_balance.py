"""
Analyse du déséquilibre entre experts (Token-Routed MLP).

Mesure:
1. Distribution théorique (basée sur token_id % num_experts)
2. Distribution réelle sur un corpus de texte
3. Entropie de la distribution (plus haute = plus équilibré)
4. Load factor par expert
"""

import torch
import argparse
from pathlib import Path
from collections import Counter
import numpy as np

def load_tokenizer(checkpoint_dir: str):
    """Load tokenizer from checkpoint or default."""
    from transformers import AutoTokenizer
    try:
        return AutoTokenizer.from_pretrained(checkpoint_dir)
    except:
        # Try loading from tokenizer files
        tokenizer_path = Path(checkpoint_dir).parent / "tokenizer.json"
        if tokenizer_path.exists():
            from tokenizers import Tokenizer
            return Tokenizer.from_file(str(tokenizer_path))
        raise ValueError(f"Cannot load tokenizer from {checkpoint_dir}")

def analyze_expert_balance_theoretical(vocab_size: int, num_experts: int = 4):
    """
    Analyse théorique: distribution si tous les tokens sont équiprobables.
    Avec routing modulo, chaque expert reçoit vocab_size/num_experts tokens.
    """
    tokens_per_expert = vocab_size // num_experts
    remainder = vocab_size % num_experts

    distribution = [tokens_per_expert] * num_experts
    for i in range(remainder):
        distribution[i] += 1

    total = sum(distribution)
    probs = [d / total for d in distribution]

    print("\n=== Distribution Théorique (tokens équiprobables) ===")
    for i, (count, prob) in enumerate(zip(distribution, probs)):
        print(f"  Expert {i}: {count} tokens ({prob*100:.2f}%)")

    entropy = -sum(p * np.log2(p) for p in probs if p > 0)
    max_entropy = np.log2(num_experts)
    print(f"  Entropie: {entropy:.4f} / {max_entropy:.4f} (max)")
    print(f"  Balance ratio: {entropy/max_entropy*100:.2f}%")

    return distribution

def analyze_expert_balance_empirical(
    tokenizer_path: str,
    text_samples: list,
    num_experts: int = 4,
    vocab_size: int = 100000
):
    """
    Analyse empirique: distribution réelle basée sur le texte.
    """
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except:
        print(f"Warning: Cannot load tokenizer from {tokenizer_path}")
        return None

    # Tokenize all samples
    all_token_ids = []
    for text in text_samples:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        all_token_ids.extend(tokens)

    # Count expert assignments
    expert_counts = Counter()
    for token_id in all_token_ids:
        expert_id = token_id % num_experts
        expert_counts[expert_id] += 1

    total = sum(expert_counts.values())

    print(f"\n=== Distribution Empirique ({len(all_token_ids)} tokens) ===")
    probs = []
    for i in range(num_experts):
        count = expert_counts[i]
        prob = count / total
        probs.append(prob)
        print(f"  Expert {i}: {count} tokens ({prob*100:.2f}%)")

    entropy = -sum(p * np.log2(p) for p in probs if p > 0)
    max_entropy = np.log2(num_experts)
    print(f"  Entropie: {entropy:.4f} / {max_entropy:.4f} (max)")
    print(f"  Balance ratio: {entropy/max_entropy*100:.2f}%")

    # Imbalance metrics
    min_prob = min(probs)
    max_prob = max(probs)
    imbalance_ratio = max_prob / min_prob if min_prob > 0 else float('inf')
    print(f"  Imbalance ratio (max/min): {imbalance_ratio:.2f}x")

    return expert_counts

def analyze_checkpoint_weights(checkpoint_path: str, num_experts: int = 4):
    """
    Analyse des poids des experts dans le checkpoint.
    Vérifie si certains experts ont des normes très différentes.
    """
    print(f"\n=== Analyse des poids du checkpoint ===")
    print(f"Loading: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Get state dict
    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    # Find expert weights
    expert_norms = {i: {'gate_up': [], 'down': []} for i in range(num_experts)}

    for key, tensor in state_dict.items():
        if 'mlp' in key.lower() and ('gate_up_proj' in key or 'down_proj' in key):
            # Format: layers.X.mlp.gate_up_proj or similar
            # Shape: [num_experts, hidden, intermediate*2] or [num_experts, intermediate, hidden]
            if tensor.dim() == 3 and tensor.shape[0] == num_experts:
                proj_type = 'gate_up' if 'gate_up' in key else 'down'
                for i in range(num_experts):
                    norm = tensor[i].norm().item()
                    expert_norms[i][proj_type].append(norm)

    if not any(expert_norms[0]['gate_up']):
        print("  No expert weights found (format may differ)")
        return None

    print("\n  Normes des poids par expert:")
    all_norms = []
    for i in range(num_experts):
        avg_gate_up = np.mean(expert_norms[i]['gate_up']) if expert_norms[i]['gate_up'] else 0
        avg_down = np.mean(expert_norms[i]['down']) if expert_norms[i]['down'] else 0
        total_norm = avg_gate_up + avg_down
        all_norms.append(total_norm)
        print(f"    Expert {i}: gate_up={avg_gate_up:.4f}, down={avg_down:.4f}, total={total_norm:.4f}")

    # Check for imbalance
    if all_norms:
        std_norm = np.std(all_norms)
        mean_norm = np.mean(all_norms)
        cv = std_norm / mean_norm if mean_norm > 0 else 0  # Coefficient of variation
        print(f"\n  Coefficient de variation des normes: {cv:.4f}")
        print(f"  (< 0.1 = bien équilibré, > 0.3 = déséquilibré)")

    return expert_norms

def analyze_mu_router_influence(checkpoint_path: str, num_experts: int = 4):
    """
    Analyse l'influence du mu_router sur le routage.
    Si mu_router a des poids significatifs, il peut modifier la distribution.
    """
    print(f"\n=== Analyse du Mu-Router ===")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint

    mu_router_weights = []
    for key, tensor in state_dict.items():
        if 'mu_router' in key:
            mu_router_weights.append((key, tensor))

    if not mu_router_weights:
        print("  Pas de mu_router trouvé (routage purement basé sur token_id)")
        return None

    for key, tensor in mu_router_weights:
        print(f"\n  {key}:")
        print(f"    Shape: {tensor.shape}")
        print(f"    Norm: {tensor.norm().item():.6f}")
        print(f"    Mean: {tensor.mean().item():.6f}")
        print(f"    Std: {tensor.std().item():.6f}")
        print(f"    Max abs: {tensor.abs().max().item():.6f}")

        # Si initialisé à zéro, pas d'influence
        if tensor.norm().item() < 1e-6:
            print("    -> Poids quasi-nuls: mu n'influence pas le routage")
        else:
            print("    -> Poids non-nuls: mu influence le routage!")
            # Show per-expert bias
            if tensor.dim() == 2:
                expert_bias = tensor.mean(dim=0)  # [num_experts]
                print(f"    Biais moyen par expert: {expert_bias.tolist()}")

def main():
    parser = argparse.ArgumentParser(description="Analyse du déséquilibre des experts")
    parser.add_argument("--checkpoint", type=str, default="./checkpoints/final.pt",
                        help="Path to checkpoint .pt file")
    parser.add_argument("--tokenizer", type=str, default="./checkpoints/pacific-prime-math-v2",
                        help="Path to tokenizer")
    parser.add_argument("--num-experts", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=100000)
    args = parser.parse_args()

    print("=" * 60)
    print("ANALYSE DU DÉSÉQUILIBRE DES EXPERTS")
    print("=" * 60)

    # 1. Analyse théorique
    analyze_expert_balance_theoretical(args.vocab_size, args.num_experts)

    # 2. Analyse empirique sur texte sample
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models can learn complex patterns from data.",
        "What is 25 + 17? Let me solve this step by step.",
        "The transformer architecture uses self-attention mechanisms.",
        "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
    ]

    try:
        analyze_expert_balance_empirical(args.tokenizer, sample_texts, args.num_experts, args.vocab_size)
    except Exception as e:
        print(f"\nSkipping empirical analysis: {e}")

    # 3. Analyse des poids du checkpoint
    if Path(args.checkpoint).exists():
        analyze_checkpoint_weights(args.checkpoint, args.num_experts)
        analyze_mu_router_influence(args.checkpoint, args.num_experts)
    else:
        print(f"\nCheckpoint not found: {args.checkpoint}")

    print("\n" + "=" * 60)
    print("INTERPRÉTATION")
    print("=" * 60)
    print("""
- Balance ratio > 95%: Excellent équilibre
- Balance ratio 90-95%: Bon équilibre
- Balance ratio 80-90%: Déséquilibre modéré
- Balance ratio < 80%: Déséquilibre significatif

- CV des normes < 0.1: Experts similaires
- CV des normes > 0.3: Certains experts sous/sur-entraînés

- mu_router non-nul: Le contexte influence le routage
  (peut améliorer ou dégrader l'équilibre)
""")

if __name__ == "__main__":
    main()
