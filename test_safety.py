"""
Test Safety Integration in Complexity Deep

Tests:
1. SafetyClamp basic functionality
2. High projection clamping
3. ContrastiveSafetyLoss
4. Model-level safety installation
"""

import torch
import sys
sys.path.insert(0, '.')

print("=" * 60)
print("Testing Safety Integration - Complexity Deep")
print("=" * 60)

# Test 1: SafetyClamp
print("\n[1] Testing SafetyClamp...")
from complexity_deep.core.safety import SafetyClamp

clamp = SafetyClamp(hidden_size=256, threshold=2.0, soft_clamp=False)

# Set harm direction
harm_dir = torch.randn(256)
clamp.set_harm_direction(harm_dir)
clamp.enabled = True

# Test clamping
x = torch.randn(2, 32, 256)
x_clamped = clamp(x)
print(f"  Input shape:  {x.shape}")
print(f"  Output shape: {x_clamped.shape}")
print(f"  Stats: {clamp.get_stats()}")
print("  [OK] SafetyClamp works!")

# Test 2: High projection clamping
print("\n[2] Testing high-projection clamping...")
harm_dir_norm = harm_dir / harm_dir.norm()
high_harm = harm_dir_norm.unsqueeze(0) * 5.0  # 5x threshold
clamped = clamp(high_harm)
proj_before = (high_harm @ harm_dir_norm).item()
proj_after = (clamped @ harm_dir_norm).item()
print(f"  Projection before: {proj_before:.4f}")
print(f"  Projection after:  {proj_after:.4f}")
print(f"  Threshold:         {clamp.threshold}")
assert proj_after <= clamp.threshold + 0.01, f"Clamping failed! {proj_after} > {clamp.threshold}"
print("  [OK] High-projection clamping works!")

# Test 3: ContrastiveSafetyLoss
print("\n[3] Testing ContrastiveSafetyLoss...")
from complexity_deep.core.safety import ContrastiveSafetyLoss

loss_fn = ContrastiveSafetyLoss(hidden_size=256, margin=1.0)

safe_act = torch.randn(4, 256)
harmful_act = torch.randn(4, 256)

result = loss_fn(safe_act, harmful_act)
print(f"  Loss:       {result['loss'].item():.4f}")
print(f"  Separation: {result['separation'].item():.4f}")
print("  [OK] ContrastiveSafetyLoss works!")

# Test 4: Model-level installation
print("\n[4] Testing model-level safety installation...")
from complexity_deep import DeepConfig, DeepForCausalLM, install_safety_on_model

config = DeepConfig.complexity_tiny()
model = DeepForCausalLM(config)

# Install safety on last 2 layers
install_safety_on_model(
    model,
    harm_direction=harm_dir,
    threshold=2.0,
    layers=[-2, -1]
)

# Forward pass
input_ids = torch.randint(0, 1000, (2, 16))
output = model(input_ids)

print(f"  Logits shape: {output.logits.shape}")
print("  [OK] Model-level safety works!")

# Test 5: Remove safety from model
print("\n[5] Testing model safety removal...")
from complexity_deep import remove_safety_from_model

remove_safety_from_model(model)
print("  [OK] Safety removed from model!")

print("\n" + "=" * 60)
print("All safety tests passed!")
print("=" * 60)
