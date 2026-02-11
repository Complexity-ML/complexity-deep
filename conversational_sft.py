"""
Conversational SFT for Complexity Models
=========================================

Fine-tune with multi-turn conversations using Jinja2 templates.

Supports:
- Multi-turn conversations (ChatGPT-like)
- Jinja2 chat templates (HuggingFace standard)
- Loss masking on assistant responses only
- Various datasets (OpenAssistant, ShareGPT, Dolphin, etc.)

Usage:
    # Using config file (recommended)
    python conversational_sft.py --config configs/sft/conversational.yaml

    # OpenAssistant (high quality multi-turn)
    python conversational_sft.py --checkpoint ./model --dataset OpenAssistant/oasst1

    # Dolphin (GPT-4 style)
    python conversational_sft.py --checkpoint ./model --dataset cognitivecomputations/dolphin

    # Custom dataset
    python conversational_sft.py --checkpoint ./model --dataset your/dataset --format sharegpt
"""

import os
import re
import math
import json
import yaml
import argparse
import random
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, List, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR
from torch.utils.tensorboard import SummaryWriter

from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from tqdm import tqdm
from jinja2 import Template

warnings.filterwarnings("ignore", message="Mismatch dtype between input and weight")

# Mixed precision
try:
    from torch.amp import autocast, GradScaler
    AMP_AVAILABLE = True
except ImportError:
    try:
        from torch.cuda.amp import autocast, GradScaler
        AMP_AVAILABLE = True
    except ImportError:
        AMP_AVAILABLE = False


def cleanup_old_checkpoints(output_dir: str, keep_last: int = 5) -> None:
    """
    Delete old checkpoints, keeping only the last N.

    Args:
        output_dir: Directory containing checkpoints
        keep_last: Number of recent checkpoints to keep (default: 5)
    """
    import glob

    # Find all checkpoint files
    pattern = os.path.join(output_dir, "checkpoint_epoch*.pt")
    checkpoints = glob.glob(pattern)

    if len(checkpoints) <= keep_last:
        return

    # Sort by epoch number
    def get_epoch(path):
        match = re.search(r'checkpoint_epoch(\d+)\.pt', path)
        return int(match.group(1)) if match else 0

    checkpoints.sort(key=get_epoch)

    # Delete old ones
    to_delete = checkpoints[:-keep_last]
    for ckpt in to_delete:
        try:
            os.remove(ckpt)
            print(f"Deleted old checkpoint: {ckpt}")
        except OSError as e:
            print(f"Warning: Could not delete {ckpt}: {e}")


# ============================================================================
# CHAT TEMPLATES (Jinja2)
# ============================================================================

CHAT_TEMPLATES = {
    # Default conversational template
    "default": """{% if messages[0]['role'] == 'system' %}{{ messages[0]['content'] }}

{% set messages = messages[1:] %}{% endif %}{% for message in messages %}{% if message['role'] == 'user' %}User: {{ message['content'] }}

{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}{% if not loop.last %}

{% endif %}{% endif %}{% endfor %}""",

    # ChatML format (used by many models)
    "chatml": """{% for message in messages %}<|im_start|>{{ message['role'] }}
{{ message['content'] }}<|im_end|>
{% endfor %}<|im_start|>assistant
""",

    # Llama-2 style
    "llama2": """{% if messages[0]['role'] == 'system' %}[INST] <<SYS>>
{{ messages[0]['content'] }}
<</SYS>>

{% set messages = messages[1:] %}{% endif %}{% for message in messages %}{% if message['role'] == 'user' %}[INST] {{ message['content'] }} [/INST]{% elif message['role'] == 'assistant' %} {{ message['content'] }}{% endif %}{% endfor %}""",

    # Simple format
    "simple": """{% for message in messages %}{% if message['role'] == 'user' %}User: {{ message['content'] }}
{% elif message['role'] == 'assistant' %}Assistant: {{ message['content'] }}
{% elif message['role'] == 'system' %}System: {{ message['content'] }}
{% endif %}{% endfor %}""",

    # Alpaca-style (for compatibility)
    "alpaca": """{% if messages[0]['role'] == 'system' %}{{ messages[0]['content'] }}

{% set messages = messages[1:] %}{% endif %}### Instruction:
{{ messages[0]['content'] }}

### Response:
{% if messages|length > 1 %}{{ messages[1]['content'] }}{% endif %}""",
}


def get_chat_template(template_name: str) -> str:
    """Get a chat template by name or return custom template."""
    if template_name in CHAT_TEMPLATES:
        return CHAT_TEMPLATES[template_name]
    # Assume it's a custom template string
    return template_name


# ============================================================================
# DATASET FORMATS
# ============================================================================

def convert_to_messages(example: Dict[str, Any], format_name: str) -> List[Dict[str, str]]:
    """Convert various dataset formats to unified messages format."""

    if format_name == "oasst":
        # OpenAssistant format
        messages = []
        if "messages" in example:
            for msg in example["messages"]:
                role = msg.get("role", "user")
                content = msg.get("content", msg.get("text", ""))
                messages.append({"role": role, "content": content})
        elif "prompt" in example and "response" in example:
            messages = [
                {"role": "user", "content": example["prompt"]},
                {"role": "assistant", "content": example["response"]},
            ]
        return messages

    elif format_name == "guanaco":
        # Guanaco format: single "text" field with ### Human: / ### Assistant: markers
        messages = []
        text = example.get("text", "")
        import re
        parts = re.split(r'### (Human|Assistant):', text)
        # parts[0] is before first marker (usually empty), then alternating role/content
        for i in range(1, len(parts) - 1, 2):
            role = "user" if parts[i] == "Human" else "assistant"
            content = parts[i + 1].strip()
            if content:
                messages.append({"role": role, "content": content})
        return messages

    elif format_name == "sharegpt":
        # ShareGPT format
        messages = []
        conversations = example.get("conversations", example.get("messages", []))
        for conv in conversations:
            role_map = {"human": "user", "gpt": "assistant", "system": "system"}
            role = conv.get("from", conv.get("role", "user"))
            role = role_map.get(role, role)
            content = conv.get("value", conv.get("content", ""))
            messages.append({"role": role, "content": content})
        return messages

    elif format_name == "dolphin":
        # Dolphin/Orca format
        messages = []
        if example.get("system_prompt"):
            messages.append({"role": "system", "content": example["system_prompt"]})
        if example.get("question"):
            messages.append({"role": "user", "content": example["question"]})
        if example.get("response"):
            messages.append({"role": "assistant", "content": example["response"]})
        return messages

    elif format_name == "alpaca":
        # Alpaca format
        messages = []
        instruction = example.get("instruction", "")
        input_text = example.get("input", "")
        output = example.get("output", "")

        if input_text:
            user_content = f"{instruction}\n\nInput: {input_text}"
        else:
            user_content = instruction

        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": output})
        return messages

    elif format_name == "messages":
        # Already in messages format
        return example.get("messages", [])

    elif format_name == "qa":
        # Simple Q&A format (gsm8k, MetaMath, NuminaMath, etc.)
        messages = []
        question = example.get("question",
                   example.get("prompt",
                   example.get("query",
                   example.get("problem", ""))))
        answer = example.get("answer",
                 example.get("response",
                 example.get("solution",
                 example.get("output", ""))))
        if question:
            messages.append({"role": "user", "content": question})
        if answer:
            messages.append({"role": "assistant", "content": answer})
        return messages

    elif format_name == "hh":
        # Anthropic HH-RLHF format: "Human: ...\n\nAssistant: ..."
        messages = []
        text = example.get("chosen", example.get("text", ""))
        if not text:
            return messages
        # Parse Human/Assistant turns
        turns = re.split(r'\n\n(?=Human:|Assistant:)', text)
        for turn in turns:
            turn = turn.strip()
            if turn.startswith("Human:"):
                content = turn[6:].strip()
                if content:
                    messages.append({"role": "user", "content": content})
            elif turn.startswith("Assistant:"):
                content = turn[10:].strip()
                if content:
                    messages.append({"role": "assistant", "content": content})
        return messages

    elif format_name == "mmlu":
        # MMLU format: question + choices + answer
        question = example.get("question", "")
        choices = example.get("choices", [])
        answer_idx = example.get("answer", 0)
        if isinstance(answer_idx, str):
            answer_idx = ord(answer_idx.upper()) - ord('A')

        # Build question with choices
        choice_letters = ["A", "B", "C", "D"]
        choices_text = "\n".join([f"{choice_letters[i]}) {c}" for i, c in enumerate(choices)])
        user_content = f"Question: {question}\n\nChoices:\n{choices_text}"

        # Answer is the correct choice text
        answer_text = choices[answer_idx] if answer_idx < len(choices) else choices[0]

        return [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": f"The answer is {choice_letters[answer_idx]}) {answer_text}"}
        ]

    elif format_name == "sciq":
        # SciQ format: question + correct_answer + distractors → Convert to multiple choice
        import random
        question = example.get("question", "")
        correct = example.get("correct_answer", "")
        distractor1 = example.get("distractor1", "")
        distractor2 = example.get("distractor2", "")
        distractor3 = example.get("distractor3", "")

        # Build choices list and shuffle
        choices = [correct, distractor1, distractor2, distractor3]
        # Use hash of question for deterministic shuffle (reproducibility)
        random.seed(hash(question) % (2**32))
        random.shuffle(choices)
        answer_idx = choices.index(correct)

        # Format like MMLU/ARC
        choice_letters = ["A", "B", "C", "D"]
        choices_text = "\n".join([f"{choice_letters[i]}) {c}" for i, c in enumerate(choices)])
        user_content = f"Question: {question}\n\nChoices:\n{choices_text}"

        return [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": f"The answer is {choice_letters[answer_idx]}) {correct}"}
        ]

    elif format_name == "arc":
        # ARC format: question + choices + answerKey
        question = example.get("question", "")
        choices_data = example.get("choices", {})
        texts = choices_data.get("text", [])
        labels = choices_data.get("label", [])
        answer_key = example.get("answerKey", "A")

        # Build choices text
        choices_text = "\n".join([f"{labels[i]}) {texts[i]}" for i in range(len(texts))])
        user_content = f"Question: {question}\n\nChoices:\n{choices_text}"

        # Find answer text
        try:
            answer_idx = labels.index(answer_key)
            answer_text = texts[answer_idx]
        except ValueError:
            answer_text = texts[0] if texts else ""
            answer_key = labels[0] if labels else "A"

        return [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": f"The answer is {answer_key}) {answer_text}"}
        ]

    elif format_name == "openbookqa":
        # OpenBookQA format: similar to ARC
        question = example.get("question_stem", example.get("question", ""))
        choices_data = example.get("choices", {})
        texts = choices_data.get("text", [])
        labels = choices_data.get("label", [])
        answer_key = example.get("answerKey", "A")

        choices_text = "\n".join([f"{labels[i]}) {texts[i]}" for i in range(len(texts))])
        user_content = f"Question: {question}\n\nChoices:\n{choices_text}"

        try:
            answer_idx = labels.index(answer_key)
            answer_text = texts[answer_idx]
        except ValueError:
            answer_text = texts[0] if texts else ""
            answer_key = labels[0] if labels else "A"

        return [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": f"The answer is {answer_key}) {answer_text}"}
        ]

    elif format_name == "hellaswag":
        # HellaSwag format: ctx + endings → Convert to multiple choice
        context = example.get("ctx", example.get("context", ""))
        endings = example.get("endings", [])
        label = example.get("label", 0)
        if isinstance(label, str):
            label = int(label)

        # Ensure we have 4 endings
        while len(endings) < 4:
            endings.append("")

        # Format as multiple choice like MMLU/ARC
        choice_letters = ["A", "B", "C", "D"]
        choices_text = "\n".join([f"{choice_letters[i]}) {endings[i]}" for i in range(4)])
        user_content = f"Complete the following:\n\n{context}\n\nChoices:\n{choices_text}"

        answer_text = endings[label] if label < len(endings) else endings[0]

        return [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": f"The answer is {choice_letters[label]}) {answer_text}"}
        ]

    elif format_name == "winogrande":
        # WinoGrande format: sentence with "_" placeholder + 2 options
        sentence = example.get("sentence", "")
        option1 = example.get("option1", "")
        option2 = example.get("option2", "")
        answer = example.get("answer", "1")
        if isinstance(answer, str):
            answer = int(answer) - 1  # "1" or "2" -> 0 or 1
        else:
            answer = answer - 1

        # Format as binary choice
        user_content = f"Fill in the blank with the correct option:\n\n{sentence}\n\nOptions:\nA) {option1}\nB) {option2}"

        answer_text = option1 if answer == 0 else option2
        answer_letter = "A" if answer == 0 else "B"

        return [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": f"The answer is {answer_letter}) {answer_text}"}
        ]

    else:
        # Try to auto-detect
        if "messages" in example:
            return example["messages"]
        elif "conversations" in example:
            return convert_to_messages(example, "sharegpt")
        elif "instruction" in example:
            return convert_to_messages(example, "alpaca")
        elif "question" in example and "response" in example:
            return convert_to_messages(example, "dolphin")
        elif "question" in example and "answer" in example:
            return convert_to_messages(example, "qa")
        else:
            raise ValueError(f"Unknown format: {format_name}")


# ============================================================================
# DATASET CLASS
# ============================================================================

class ConversationalDataset(Dataset):
    """Dataset for conversational fine-tuning with proper loss masking."""

    def __init__(
        self,
        dataset_name: str,
        tokenizer: PreTrainedTokenizerFast,
        chat_template: str,
        format_name: str = "auto",
        max_length: int = 2048,
        split: str = "train",
        max_samples: Optional[int] = None,
        token: Optional[str] = None,
        subset: Optional[str] = None,
        mask_user: bool = True,  # Mask loss on user messages
        _examples: Optional[List] = None,  # Pre-loaded examples (for multi-dataset)
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_user = mask_user
        self.template = Template(chat_template)
        self.format_name = format_name

        # Use pre-loaded examples if provided
        if _examples is not None:
            self.examples = _examples
            print(f"Using {len(self.examples)} pre-loaded conversations")
        else:
            # Load single dataset
            print(f"Loading dataset: {dataset_name}")
            try:
                if subset:
                    ds = load_dataset(dataset_name, subset, split=split, token=token)
                else:
                    ds = load_dataset(dataset_name, split=split, token=token)
            except Exception as e:
                print(f"Error loading dataset: {e}")
                print("Trying with trust_remote_code=True...")
                if subset:
                    ds = load_dataset(dataset_name, subset, split=split, token=token, trust_remote_code=True)
                else:
                    ds = load_dataset(dataset_name, split=split, token=token, trust_remote_code=True)

            # Limit samples
            if max_samples and len(ds) > max_samples:
                ds = ds.select(range(max_samples))

            self.examples = list(ds)
            print(f"Loaded {len(self.examples)} conversations")

        # Show sample
        if self.examples:
            # Find a non-empty sample to display
            for sample in self.examples[:10]:
                sample_format = sample.get("_format", self.format_name) if isinstance(sample, dict) else self.format_name
                sample_messages = convert_to_messages(sample, sample_format)
                if sample_messages:
                    sample_text = self.template.render(messages=sample_messages)
                    print(f"\n--- Sample conversation ---")
                    print(sample_text[:500] + "..." if len(sample_text) > 500 else sample_text)
                    print("---\n")
                    break

    @classmethod
    def from_multiple_datasets(
        cls,
        datasets_config: List[dict],
        tokenizer: PreTrainedTokenizerFast,
        chat_template: str,
        format_name: str = "auto",
        max_length: int = 2048,
        split: str = "train",
        max_samples: Optional[int] = None,
        token: Optional[str] = None,
        mask_user: bool = True,
    ):
        """
        Load and combine multiple datasets with weights.

        datasets_config: List of dicts with keys: name, weight, subset (optional)
        Example: [{"name": "openai/gsm8k", "weight": 0.3, "subset": "main"}, ...]
        """
        all_examples = []
        total_weight = sum(d.get("weight", 1.0) for d in datasets_config)

        print(f"\n{'='*60}")
        print(f"Loading {len(datasets_config)} datasets (max_samples={max_samples})")
        print(f"{'='*60}")

        for ds_config in datasets_config:
            ds_name = ds_config["name"]
            ds_weight = ds_config.get("weight", 1.0) / total_weight
            ds_subset = ds_config.get("subset", None)
            ds_split = ds_config.get("split", split)  # Use per-dataset split if specified

            # Calculate samples for this dataset
            if max_samples:
                ds_max = int(max_samples * ds_weight)
            else:
                ds_max = None

            print(f"\n[{ds_name}] weight={ds_config.get('weight', 1.0):.2f} -> {ds_max or 'all'} samples (split={ds_split})")

            try:
                if ds_subset:
                    ds = load_dataset(ds_name, ds_subset, split=ds_split, token=token)
                else:
                    ds = load_dataset(ds_name, split=ds_split, token=token)
            except Exception as e:
                print(f"  Error: {e}")
                print("  Trying with trust_remote_code=True...")
                try:
                    if ds_subset:
                        ds = load_dataset(ds_name, ds_subset, split=ds_split, token=token, trust_remote_code=True)
                    else:
                        ds = load_dataset(ds_name, split=ds_split, token=token, trust_remote_code=True)
                except Exception as e2:
                    print(f"  Failed to load {ds_name}: {e2}")
                    continue

            # Sample or limit
            ds_list = list(ds)
            if ds_max and len(ds_list) > ds_max:
                random.shuffle(ds_list)
                ds_list = ds_list[:ds_max]

            print(f"  Loaded {len(ds_list)} samples")
            # Store format with each example
            ds_format = ds_config.get("format", format_name)
            for ex in ds_list:
                ex["_format"] = ds_format
            all_examples.extend(ds_list)

        # Shuffle combined dataset
        random.shuffle(all_examples)

        print(f"\n{'='*60}")
        print(f"Total combined: {len(all_examples)} samples")
        print(f"{'='*60}\n")

        return cls(
            dataset_name="combined",
            tokenizer=tokenizer,
            chat_template=chat_template,
            format_name=format_name,
            max_length=max_length,
            mask_user=mask_user,
            _examples=all_examples,
        )

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        example = self.examples[idx]

        # Get format: per-example format (from multi-dataset) or global format
        format_to_use = example.get("_format", self.format_name) if isinstance(example, dict) else self.format_name

        # Convert to messages format
        messages = convert_to_messages(example, format_to_use)

        if not messages:
            # Fallback for empty conversations
            return self._empty_item()

        # Render full conversation
        full_text = self.template.render(messages=messages)

        # Tokenize
        tokens = self.tokenizer.encode(full_text, add_special_tokens=True)

        # Truncate
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]

        input_ids = torch.tensor(tokens, dtype=torch.long)

        # Create labels with masking
        if self.mask_user:
            labels = self._create_masked_labels(messages, tokens)
        else:
            labels = input_ids.clone()

        return {"input_ids": input_ids, "labels": labels}

    def _create_masked_labels(self, messages: List[Dict], tokens: List[int]) -> torch.Tensor:
        """Create labels with -100 for user messages (no loss computed)."""
        # Use Python list to avoid creating intermediate tensors in the loop
        labels = [-100] * len(tokens)

        # Find assistant response positions
        # We tokenize incrementally to find boundaries
        current_pos = 0

        for i, msg in enumerate(messages):
            # Render up to this message
            partial_messages = messages[:i+1]
            partial_text = self.template.render(messages=partial_messages)
            partial_tokens = self.tokenizer.encode(partial_text, add_special_tokens=True)

            if msg["role"] == "assistant":
                # Compute loss on assistant tokens
                start_pos = current_pos
                end_pos = min(len(partial_tokens), len(tokens))
                # Direct list slice assignment - no tensor creation
                labels[start_pos:end_pos] = tokens[start_pos:end_pos]

            current_pos = len(partial_tokens)
            if current_pos >= len(tokens):
                break

        # Single tensor creation at the end
        return torch.tensor(labels, dtype=torch.long)

    def _empty_item(self):
        """Return empty item for invalid examples."""
        return {
            "input_ids": torch.tensor([self.tokenizer.pad_token_id], dtype=torch.long),
            "labels": torch.tensor([-100], dtype=torch.long),
        }


def collate_fn(batch, pad_token_id: int = 0):
    """Collate with padding."""
    # Filter empty items and items with no valid labels (all -100)
    batch = [item for item in batch
             if item["input_ids"].shape[0] > 1 and (item["labels"] != -100).any()]
    if not batch:
        return None

    max_len = max(item["input_ids"].shape[0] for item in batch)

    input_ids = torch.full((len(batch), max_len), pad_token_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros(len(batch), max_len, dtype=torch.long)

    for i, item in enumerate(batch):
        seq_len = item["input_ids"].shape[0]
        input_ids[i, :seq_len] = item["input_ids"]
        labels[i, :seq_len] = item["labels"]
        attention_mask[i, :seq_len] = 1

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> tuple:
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            if batch is None:
                continue

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            attention_mask = batch["attention_mask"].to(device)

            if use_amp and AMP_AVAILABLE:
                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
            else:
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

            if not torch.isnan(loss) and not torch.isinf(loss):
                total_loss += loss.item()
                num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    perplexity = math.exp(min(avg_loss, 20))
    return avg_loss, perplexity


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    scaler: Optional[GradScaler],
    device: torch.device,
    epoch: int,
    writer: SummaryWriter,
    global_step: int,
    gradient_accumulation: int = 1,
    max_grad_norm: float = 1.0,
    use_amp: bool = True,
) -> tuple:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(pbar):
        if batch is None:
            continue

        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Forward
        if use_amp and AMP_AVAILABLE:
            with autocast(device_type="cuda", dtype=torch.bfloat16):
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss / gradient_accumulation
        else:
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / gradient_accumulation

        # Check for NaN loss (training instability)
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\nWarning: NaN/Inf loss at batch {batch_idx}, skipping...")
            optimizer.zero_grad()
            continue

        # Backward
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Gradient accumulation step
        if (batch_idx + 1) % gradient_accumulation == 0:
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            # Log
            if global_step % 10 == 0:
                lr = scheduler.get_last_lr()[0]
                current_loss = loss.item() * gradient_accumulation
                perplexity = math.exp(min(current_loss, 20))  # Cap to avoid overflow
                writer.add_scalar("train/loss", current_loss, global_step)
                writer.add_scalar("train/perplexity", perplexity, global_step)
                writer.add_scalar("train/lr", lr, global_step)
                # Flush every 100 steps to ensure logs are written to disk
                if global_step % 100 == 0:
                    writer.flush()
                    # Clear CUDA cache to prevent memory fragmentation
                    torch.cuda.empty_cache()

        total_loss += loss.item() * gradient_accumulation
        num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        ppl = math.exp(min(avg_loss, 20))  # Cap to avoid overflow
        pbar.set_postfix({
            "loss": f"{avg_loss:.4f}",
            "ppl": f"{ppl:.2f}",
            "lr": f"{scheduler.get_last_lr()[0]:.2e}",
        })

    avg_loss = total_loss / max(num_batches, 1)
    return global_step, avg_loss


def main():
    parser = argparse.ArgumentParser(description="Conversational SFT for Complexity")

    # Config file
    parser.add_argument("--config", type=str, default=None, help="YAML config file path")

    # Model
    parser.add_argument("--checkpoint", type=str, default=None, help="Model checkpoint path")
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer path")

    # Dataset
    parser.add_argument("--dataset", type=str, default=None, help="HuggingFace dataset (single)")
    parser.add_argument("--datasets-json", type=str, default=None,
                       help="JSON array of datasets with weights: [{\"name\": \"...\", \"weight\": 0.3, \"subset\": \"...\"}]")
    parser.add_argument("--subset", type=str, default=None, help="Dataset subset")
    parser.add_argument("--format", type=str, default="auto",
                       choices=["auto", "oasst", "sharegpt", "dolphin", "alpaca", "messages", "qa", "hh"],
                       help="Dataset format")
    parser.add_argument("--max-samples", type=int, default=None, help="Max samples (total across all datasets)")
    parser.add_argument("--token", type=str, default=None, help="HF token")

    # Chat template
    parser.add_argument("--template", type=str, default=None,
                       choices=list(CHAT_TEMPLATES.keys()),
                       help="Chat template name")
    parser.add_argument("--custom-template", type=str, default=None,
                       help="Custom Jinja2 template string")
    parser.add_argument("--no-mask-user", action="store_true",
                       help="Don't mask user messages (compute loss on everything)")

    # Training
    parser.add_argument("--epochs", type=int, default=3, help="Epochs")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size")
    parser.add_argument("--gradient-accumulation", type=int, default=16, help="Grad accum steps")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--max-length", type=int, default=2048, help="Max sequence length")
    parser.add_argument("--warmup-ratio", type=float, default=0.03, help="Warmup ratio")

    # Hardware
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--bf16", action="store_true", default=True, help="Use BF16")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="Enable gradient checkpointing (saves VRAM)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")

    # Output
    parser.add_argument("--output", type=str, default=None, help="Output dir")
    parser.add_argument("--save-every", type=int, default=None, help="Save every N epochs")

    # Validation
    parser.add_argument("--val-split", type=float, default=0.0, help="Validation split ratio (0 to disable, removed)")
    parser.add_argument("--eval-every", type=int, default=1, help="Evaluate every N epochs (unused)")

    args = parser.parse_args()

    # Load config file if provided
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        # Map config to args (CLI args override config)
        config_mapping = {
            # Model
            ('model', 'checkpoint'): 'checkpoint',
            ('model', 'tokenizer'): 'tokenizer',
            ('model', 'output'): 'output',
            # Training
            ('training', 'epochs'): 'epochs',
            ('training', 'batch_size'): 'batch_size',
            ('training', 'gradient_accumulation'): 'gradient_accumulation',
            ('training', 'learning_rate'): 'lr',
            ('training', 'weight_decay'): 'weight_decay',
            ('training', 'max_length'): 'max_length',
            ('training', 'warmup_ratio'): 'warmup_ratio',
            ('training', 'gradient_checkpointing'): 'gradient_checkpointing',
            ('training', 'bf16'): 'bf16',
            # Data
            ('data', 'max_samples'): 'max_samples',
            ('data', 'format'): 'format',
            # Template
            ('template', 'name'): 'template',
            ('template', 'mask_user'): None,  # Inverse of no_mask_user
            # Saving
            ('training', 'save_every'): 'save_every',
            # Validation
            ('validation', 'split'): 'val_split',
            ('validation', 'eval_every'): 'eval_every',
        }

        for (section, key), arg_name in config_mapping.items():
            if section in config and key in config[section]:
                value = config[section][key]
                if arg_name is None:
                    # Special case: mask_user -> no_mask_user (inverse)
                    if key == 'mask_user':
                        if getattr(args, 'no_mask_user', None) is None:
                            args.no_mask_user = not value
                elif getattr(args, arg_name, None) is None or \
                     (arg_name in ['epochs', 'batch_size', 'gradient_accumulation'] and
                      getattr(args, arg_name) == parser.get_default(arg_name)):
                    setattr(args, arg_name, value)

        # Handle datasets from config
        if 'data' in config and 'datasets' in config['data'] and not args.dataset and not args.datasets_json:
            args.datasets_json = json.dumps(config['data']['datasets'])

    # Apply defaults for args not set by CLI or config
    if args.output is None:
        args.output = "./checkpoints-conv-sft"
    if args.template is None:
        args.template = "default"
    if args.save_every is None:
        args.save_every = 1

    # Validate required args
    if not args.checkpoint:
        parser.error("--checkpoint is required (or set model.checkpoint in config)")

    # Setup
    device = torch.device(args.device)
    os.makedirs(args.output, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = Path(args.dataset).name if args.dataset else "multi"
    run_name = f"conv_sft_{dataset_name}_{timestamp}"
    # Use runs-sft directory (create in script directory for consistency)
    script_dir = Path(__file__).parent.parent  # Go up from complexity-deep to pacific-prime
    runs_dir = script_dir / "runs-sft" / run_name
    runs_dir.parent.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(runs_dir))
    print(f"TensorBoard logs: {runs_dir}")

    print("\n" + "=" * 60)
    print("CONVERSATIONAL SFT - COMPLEXITY MODEL")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    if args.dataset:
        print(f"Dataset: {args.dataset} (format: {args.format})")
    elif args.datasets_json:
        datasets_info = json.loads(args.datasets_json)
        print(f"Datasets: {len(datasets_info)} sources (format: {args.format})")
        for ds in datasets_info:
            print(f"  - {ds['name']} (weight: {ds.get('weight', 1.0)})")
    print(f"Template: {args.template}")
    print(f"Epochs: {args.epochs}")
    print(f"Effective batch: {args.batch_size} x {args.gradient_accumulation} = {args.batch_size * args.gradient_accumulation}")
    print(f"Learning rate: {args.lr}")
    print(f"Max length: {args.max_length}")
    print(f"Loss masking: {'User messages masked' if not args.no_mask_user else 'Full sequence'}")
    print("=" * 60 + "\n")

    # Get chat template
    if args.custom_template:
        chat_template = args.custom_template
    else:
        chat_template = get_chat_template(args.template)

    print("Chat template:")
    print("-" * 40)
    print(chat_template[:200] + "..." if len(chat_template) > 200 else chat_template)
    print("-" * 40 + "\n")

    # Load tokenizer
    tokenizer_path = args.tokenizer or args.checkpoint
    print(f"Loading tokenizer: {tokenizer_path}")
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model
    print(f"Loading model: {args.checkpoint}")
    from complexity_deep import DeepForCausalLM, DeepConfig

    # Find checkpoint - handle both file path and directory
    if os.path.isfile(args.checkpoint):
        # Direct file path provided
        checkpoint_path = args.checkpoint
        checkpoint_dir = os.path.dirname(args.checkpoint)
    else:
        # Directory provided - search for checkpoint files
        checkpoint_dir = args.checkpoint
        checkpoint_names = ["model.pt", "final.pt", "model.safetensors", "checkpoint.pt"]
        checkpoint_path = None
        for name in checkpoint_names:
            path = f"{checkpoint_dir}/{name}"
            if os.path.exists(path):
                checkpoint_path = path
                break

        if checkpoint_path is None:
            raise FileNotFoundError(f"No checkpoint in {args.checkpoint}")

    print(f"Loading: {checkpoint_path}")

    # Load config
    config_path = f"{checkpoint_dir}/config.json"
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    else:
        raise FileNotFoundError(f"No config.json in {args.checkpoint}")

    model_config = DeepConfig.from_dict(config)
    model = DeepForCausalLM(model_config)

    # Load weights
    if checkpoint_path.endswith(".safetensors"):
        from safetensors.torch import load_file
        state_dict = load_file(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))

    model.load_state_dict(state_dict)

    # Enable gradient checkpointing for VRAM savings
    if args.gradient_checkpointing:
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable()
            print("Gradient checkpointing: ENABLED (saves ~40% VRAM)")
        else:
            print("Warning: Model doesn't support gradient_checkpointing_enable()")

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params / 1e9:.2f}B")

    # Load dataset(s)
    if args.datasets_json:
        # Multiple datasets with weights
        datasets_config = json.loads(args.datasets_json)
        dataset = ConversationalDataset.from_multiple_datasets(
            datasets_config=datasets_config,
            tokenizer=tokenizer,
            chat_template=chat_template,
            format_name=args.format,
            max_length=args.max_length,
            max_samples=args.max_samples,
            token=args.token,
            mask_user=not args.no_mask_user,
        )
    elif args.dataset:
        # Single dataset
        dataset = ConversationalDataset(
            dataset_name=args.dataset,
            tokenizer=tokenizer,
            chat_template=chat_template,
            format_name=args.format,
            max_length=args.max_length,
            max_samples=args.max_samples,
            token=args.token,
            subset=args.subset,
            mask_user=not args.no_mask_user,
        )
    else:
        raise ValueError("Must provide --dataset or --datasets-json")

    # No validation split - use all data for training
    pad_token_id = tokenizer.pad_token_id or 0
    train_dataset = dataset

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_token_id),
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    # Scheduler with warmup
    total_steps = len(train_loader) * args.epochs // args.gradient_accumulation
    warmup_steps = int(total_steps * args.warmup_ratio)

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup (start from 1% to avoid dead first steps)
            return max(0.01, float(current_step) / float(max(1, warmup_steps)))
        else:
            # Cosine decay after warmup
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)
    print(f"Scheduler: Linear warmup ({warmup_steps} steps) + Cosine decay (total {total_steps} steps)")

    # Scaler
    scaler = GradScaler() if args.bf16 and AMP_AVAILABLE else None

    # Training
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        print(f"\n{'=' * 40}")
        print(f"Epoch {epoch}/{args.epochs}")
        print(f"{'=' * 40}")

        global_step, avg_loss = train_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
            epoch=epoch,
            writer=writer,
            global_step=global_step,
            gradient_accumulation=args.gradient_accumulation,
            use_amp=args.bf16,
        )

        print(f"Epoch {epoch} - Train loss: {avg_loss:.4f}")

        # Save checkpoint
        if epoch % args.save_every == 0:
            save_path = f"{args.output}/checkpoint_epoch{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": config,
                "global_step": global_step,
                "chat_template": chat_template,
            }, save_path)
            print(f"Saved: {save_path}")

            # Cleanup old checkpoints - keep only last 5
            cleanup_old_checkpoints(args.output, keep_last=5)

    # Save final model
    final_path = f"{args.output}/model_conv_sft.pt"
    torch.save({
        "model": model.state_dict(),
        "config": config,
        "chat_template": chat_template,
    }, final_path)
    print(f"\nFinal model: {final_path}")

    # Save in HF format
    hf_path = f"{args.output}/hf"
    os.makedirs(hf_path, exist_ok=True)

    # Config with chat_template
    config["chat_template"] = chat_template
    with open(f"{hf_path}/config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Tokenizer with chat_template
    tokenizer.save_pretrained(hf_path)

    # Add chat_template to tokenizer_config.json
    tokenizer_config_path = f"{hf_path}/tokenizer_config.json"
    if os.path.exists(tokenizer_config_path):
        with open(tokenizer_config_path) as f:
            tok_config = json.load(f)
        tok_config["chat_template"] = chat_template
        with open(tokenizer_config_path, "w") as f:
            json.dump(tok_config, f, indent=2)

    print(f"HuggingFace format: {hf_path}")

    # Save safetensors
    try:
        from safetensors.torch import save_file
        bf16_state = {k: v.to(torch.bfloat16) if v.dtype == torch.float32 else v
                      for k, v in model.state_dict().items()}
        save_file(bf16_state, f"{hf_path}/model.safetensors")
        print(f"Saved: {hf_path}/model.safetensors (BF16)")
    except Exception as e:
        print(f"Warning: Could not save safetensors: {e}")

    writer.close()
    print("\nConversational SFT complete!")


if __name__ == "__main__":
    main()
