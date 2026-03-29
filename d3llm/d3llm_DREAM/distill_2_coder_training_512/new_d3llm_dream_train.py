import sys
import os

sys.path.append("./distill_2_coder_training_512")
import torch
import torch.nn.functional as F
import yaml
from datasets import load_dataset
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import random
import pickle
import hashlib
import subprocess
from ast import literal_eval
from types import MethodType

# ---------------------------------------------------------------------------
# 启发式揭示顺序：与 simulate_trajectory_for_inspection.py 命令行默认一致
# （不传 --swap-frac / --long-swap-prob / --max-swap-distance 时的行为）。
# 训练时可通过 d3llm_train.yaml 的 distillation 项覆盖；未配置则使用下列写死默认值。
# ---------------------------------------------------------------------------
HEURISTIC_SWAP_FRAC = 0.5
HEURISTIC_LONG_SWAP_PROB = 0.3
HEURISTIC_MAX_SWAP_DISTANCE = 8


def load_config(config_path: str) -> Dict[str, Any]:
    """Load YAML configuration file"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def override_config(config: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Override config values from command line args like 'training.learning_rate=0.000001'"""
    for override in overrides:
        # Skip DeepSpeed/distributed training args (--local_rank, etc.)
        if override.startswith('--') or '=' not in override:
            continue
        key_path, value = override.split('=', 1)
        keys = key_path.split('.')
        
        # Navigate to nested dict
        target = config
        for key in keys[:-1]:
            target = target.setdefault(key, {})
        
        # Convert value to appropriate type
        final_key = keys[-1]
        old_value = target.get(final_key)
        
        # Try literal_eval for lists/dicts (e.g., "[16,32,32]")
        if isinstance(old_value, (list, dict)) or value.startswith(('[', '{')):
            try:
                target[final_key] = literal_eval(value)
            except (ValueError, SyntaxError):
                target[final_key] = value
        elif isinstance(old_value, bool):
            target[final_key] = value.lower() in ('true', '1', 'yes')
        elif isinstance(old_value, int):
            target[final_key] = int(value)
        elif isinstance(old_value, float):
            target[final_key] = float(value)
        else:
            target[final_key] = value
    
    return config


def get_deepspeed_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Create DeepSpeed configuration"""
    return {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "zero_allow_untested_optimizer": True,
        "bf16": {"enabled": "auto"},
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
    }


def prepare_model(config: Dict[str, Any]):
    """Prepare model and tokenizer with optional LoRA"""
    torch_dtype = getattr(torch, config["model"]["torch_dtype"])
    
    model = AutoModel.from_pretrained(
        config["model"]["name"],
        torch_dtype=torch_dtype,
        trust_remote_code=config["model"]["trust_remote_code"],
    )
    
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"], 
        trust_remote_code=config["model"]["trust_remote_code"]
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Apply LoRA if enabled in config
    lora_config_dict = config.get("lora")
    if lora_config_dict and lora_config_dict.get("enabled", False):
        print("=" * 80)
        print("Applying LoRA configuration...")
        # PEFT 在 CAUSAL_LM 等 task_type 下会访问 base_model.prepare_inputs_for_generation；
        # 远程 DreamModel 可能未实现，训练 forward 不依赖 generate，挂一个最小实现即可。
        def _dummy_prepare_inputs_for_generation(self, input_ids, **kwargs):
            return {"input_ids": input_ids, **kwargs}

        if not hasattr(model, "prepare_inputs_for_generation"):
            model.prepare_inputs_for_generation = MethodType(
                _dummy_prepare_inputs_for_generation, model
            )
        lora_config = LoraConfig(
            r=lora_config_dict.get("r", 16),
            lora_alpha=lora_config_dict.get("lora_alpha", 16),
            target_modules=lora_config_dict.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),
            lora_dropout=lora_config_dict.get("lora_dropout", 0.0),
            bias=lora_config_dict.get("bias", "none"),
            task_type=lora_config_dict.get("task_type", "CAUSAL_LM")
        )
        
        model = get_peft_model(model, lora_config)
        
        # Print the number of trainable parameters
        model.print_trainable_parameters()
        print("=" * 80)
    else:
        print("=" * 80)
        print("LoRA is disabled. Training full model.")
        print("=" * 80)
    
    return model, tokenizer


def _pick_timeline_swap_adjacent(seg_len: int, rng: random.Random) -> Tuple[int, int]:
    i = rng.randint(0, seg_len - 2)
    return i, i + 1


def _pick_timeline_swap_long(
    seg_len: int, rng: random.Random, max_dist: int
) -> Optional[Tuple[int, int]]:
    md = min(max_dist, seg_len - 1)
    if md < 2 or seg_len < 3:
        return None
    for _ in range(64):
        i = rng.randint(0, seg_len - 1)
        lo = max(0, i - md)
        hi = min(seg_len - 1, i + md)
        cand = [j for j in range(lo, hi + 1) if j != i and 2 <= abs(i - j) <= md]
        if cand:
            return i, rng.choice(cand)
    return None


def generate_local_decode_order(
    seg_len: int,
    rng: random.Random,
    swap_frac: float = HEURISTIC_SWAP_FRAC,
    num_swaps: Optional[int] = None,
    long_swap_prob: float = HEURISTIC_LONG_SWAP_PROB,
    max_swap_distance: int = HEURISTIC_MAX_SWAP_DISTANCE,
) -> List[int]:
    """Build decode/reveal order over **response-local** indices [0, seg_len).

    从恒等顺序出发；重复若干次：以概率 ``long_swap_prob`` 在揭示时间轴上交换
    相距 2..max_swap_distance 的两步，否则交换相邻两步。
    Prompt 不参与该排列。默认超参见模块级 HEURISTIC_* 常量。
    """
    if seg_len <= 0:
        return []
    if seg_len == 1:
        return [0]

    order = list(range(seg_len))
    if num_swaps is not None:
        n = max(0, int(num_swaps))
    else:
        n = max(1, int(round(swap_frac * seg_len)))

    lp = max(0.0, min(1.0, float(long_swap_prob)))
    md = max(2, int(max_swap_distance))

    for _ in range(n):
        use_long = rng.random() < lp
        if use_long:
            pair = _pick_timeline_swap_long(seg_len, rng, md)
            if pair is not None:
                i, j = pair
                order[i], order[j] = order[j], order[i]
                continue
        i, j = _pick_timeline_swap_adjacent(seg_len, rng)
        order[i], order[j] = order[j], order[i]
    return order


def heuristic_seg_mask_bool(
    seg_len: int,
    mask_ratio: float,
    rng: random.Random,
    swap_frac: float = HEURISTIC_SWAP_FRAC,
    num_swaps: Optional[int] = None,
    long_swap_prob: float = HEURISTIC_LONG_SWAP_PROB,
    max_swap_distance: int = HEURISTIC_MAX_SWAP_DISTANCE,
) -> List[bool]:
    """Binary mask over a response segment: True = still masked (not yet revealed).

    Reveal order follows ``generate_local_decode_order``. After k = round((1-mask_ratio)*L)
    reveals, positions ``order[0..k-1]`` are unmasked; the rest stay masked.
    """
    if seg_len <= 0:
        return []
    order = generate_local_decode_order(
        seg_len,
        rng,
        swap_frac=swap_frac,
        num_swaps=num_swaps,
        long_swap_prob=long_swap_prob,
        max_swap_distance=max_swap_distance,
    )
    k = int(round((1.0 - mask_ratio) * seg_len))
    k = max(0, min(seg_len, k))
    still_masked = set(order[k:])
    return [i in still_masked for i in range(seg_len)]


def random_ratio_seg_mask_bool(
    seg_len: int,
    mask_ratio: float,
    rng: random.Random,
) -> List[bool]:
    """按 mask_ratio 随机生成掩码：在响应段内均匀随机选取恰好 k 个位置置为 masked。

    k = round(mask_ratio * seg_len)，并裁剪到 [0, seg_len]。与独立 Bernoulli（use_naive_random_mask）
    不同，本策略在每个 segment 上**精确**满足（四舍五入后的）掩码个数，仅位置随机。
    """
    if seg_len <= 0:
        return []
    mr = max(0.0, min(1.0, float(mask_ratio)))
    k = int(round(mr * seg_len))
    k = max(0, min(seg_len, k))
    if k == 0:
        return [False] * seg_len
    if k == seg_len:
        return [True] * seg_len
    chosen = rng.sample(range(seg_len), k)
    masked_set = set(chosen)
    return [i in masked_set for i in range(seg_len)]


def forward_process_with_trajectory(
    input_ids,
    prompt_lengths,
    mask_token_id=151666,
    block_size=32,
    mask_ratio=0.5,
    use_blockwise=False,
    use_naive_random_mask=False,
    use_random_ratio_mask=False,
    use_heuristic_trajectory=True,
    heuristic_swap_frac=HEURISTIC_SWAP_FRAC,
    heuristic_num_swaps=None,
    heuristic_long_swap_prob=HEURISTIC_LONG_SWAP_PROB,
    heuristic_max_swap_distance=HEURISTIC_MAX_SWAP_DISTANCE,
    use_complementary_loss=False,
    eps=1e-3,
    sample_indices=None,
    global_step: int = 0,
):
    """Forward masking: heuristic pseudo-trajectory (default)、i.i.d. Bernoulli 或按比例的随机定长掩码。

    Args:
        use_blockwise: If True, only predict one block; otherwise random mask entire response
        use_naive_random_mask: If True, 每个位置独立 Bernoulli(mask_ratio)，仅期望比例接近
        use_random_ratio_mask: If True, 在段内随机选 k=round(mask_ratio*L) 个位置掩码（精确个数）
        use_heuristic_trajectory: If True (且未启用 naive / random_ratio), use local decode-order heuristic
        heuristic_swap_frac: number of transposition attempts ≈ swap_frac * segment_length
        heuristic_num_swaps: if set, fixed number of attempts (overrides swap_frac)
        heuristic_long_swap_prob: each attempt uses a long swap (timeline distance 2..max) with this probability
        heuristic_max_swap_distance: max |i-j| for long swaps on the reveal timeline
        use_complementary_loss: If True, also return complementary masked batch for dParallel loss

    优先级: use_random_ratio_mask > use_naive_random_mask > use_heuristic_trajectory > Bernoulli 回退
    """
    b, l = input_ids.shape
    device = input_ids.device
    
    noisy_batch = input_ids.clone()
    noisy_batch_rev = input_ids.clone() if use_complementary_loss else None
    masked_indices = torch.zeros_like(input_ids, dtype=torch.bool)
    masked_indices_rev = torch.zeros_like(input_ids, dtype=torch.bool) if use_complementary_loss else None
    
    # Protect prompt region from masking
    token_positions = torch.arange(l, device=device).expand(b, l)
    prompt_mask = token_positions < prompt_lengths.unsqueeze(1)
    
    noisy_batch[prompt_mask] = input_ids[prompt_mask]
    if use_complementary_loss:
        noisy_batch_rev[prompt_mask] = input_ids[prompt_mask]
    
    for i in range(b):
        prompt_len = prompt_lengths[i].item()
        response_len = l - prompt_len
        
        if response_len <= 0:
            continue
        
        # Determine mask region
        if use_blockwise:
            max_blocks = response_len // block_size
            num_blocks = random.randint(0, max_blocks)
            mask_start = prompt_len + num_blocks * block_size
            mask_end = mask_start + block_size if num_blocks < max_blocks else l
        else:
            mask_start = prompt_len
            mask_end = l
        
        seg_len = mask_end - mask_start
        sid = int(sample_indices[i].item()) if sample_indices is not None else i
        seed = (
            sid * 1000003
            + int(global_step) * 1315423911
            + mask_start * 17
            + mask_end * 31
            + int(mask_ratio * 1e6)
        ) & 0xFFFFFFFF
        rng = random.Random(seed)

        if use_random_ratio_mask:
            seg_mask = torch.tensor(
                random_ratio_seg_mask_bool(seg_len, mask_ratio, rng),
                device=device,
                dtype=torch.bool,
            )
        elif use_naive_random_mask:
            p_mask = (1 - eps) * mask_ratio + eps
            seg_mask = torch.rand(seg_len, device=device) < p_mask
        elif use_heuristic_trajectory:
            seg_mask = torch.tensor(
                heuristic_seg_mask_bool(
                    seg_len,
                    mask_ratio,
                    rng,
                    swap_frac=heuristic_swap_frac,
                    num_swaps=heuristic_num_swaps,
                    long_swap_prob=heuristic_long_swap_prob,
                    max_swap_distance=heuristic_max_swap_distance,
                ),
                device=device,
                dtype=torch.bool,
            )
        else:
            p_mask = (1 - eps) * mask_ratio + eps
            seg_mask = torch.rand(seg_len, device=device) < p_mask
        # print(f"[Debug] sample {i}, seg_mask: {seg_mask}")
        # print(f"[Debug] sample {i}, seg_len: {seg_len}")
        # print(f"[Debug] sample {i}, mask_start: {mask_start}")
        # print(f"[Debug] sample {i}, mask_end: {mask_end}")
        # print(f"[Debug] sample {i}, mask_ratio: {mask_ratio}")
        # print(f"[Debug] sample {i}, prompt_len: {prompt_len}")
        # print(f"[Debug] sample {i}, response_len: {response_len}")

        # Apply mask (same logic as dream_train.py)
        masked_indices[i, mask_start:mask_end] = seg_mask
        if use_complementary_loss:
            masked_indices_rev[i, mask_start:mask_end] = ~seg_mask
        
        noisy_batch[i, mask_start:mask_end] = torch.where(
            masked_indices[i, mask_start:mask_end], mask_token_id, input_ids[i, mask_start:mask_end]
        )
        if use_complementary_loss:
            noisy_batch_rev[i, mask_start:mask_end] = torch.where(
                masked_indices_rev[i, mask_start:mask_end], mask_token_id, input_ids[i, mask_start:mask_end]
            )
        
        # Mask future tokens
        noisy_batch[i, mask_end:l] = mask_token_id
        if use_complementary_loss:
            noisy_batch_rev[i, mask_end:l] = mask_token_id


        # print(f"[Debug] sample {i}, total length:{l}")
        # print(f"[Debug] sample {i}, prompt length:{prompt_len}")
        # print(f"[Debug] sample {i} masked_indices: {sum(masked_indices[i])}")
        # print(f"[Debug] sample {i}, mask_start:{mask_start}, mask_end:{mask_end}")
        # print(f"[Debug] sample {i} masked_indices: {sum(masked_indices[i])}")
        # print(f"[Debug] sample {i} masked_indices_rev: {sum(masked_indices_rev[i])}")
        # print(f"[Debug] sample {i} Ratio of masks in noisy_batch: {sum(noisy_batch[i, mask_start:mask_end] == mask_token_id)}/{mask_end - mask_start}")
        # print(f"[Debug] sample {i} Ratio of masks in noisy_batch_rev: {sum(noisy_batch_rev[i, mask_start:mask_end] == mask_token_id)}/{mask_end - mask_start}")
        # print(f"[Debug] sample {i} ALL Ratio of masks in noisy_batch: {sum(noisy_batch[i, :] == mask_token_id)}/{len(noisy_batch[i, :])}")
        # print(f"[Debug] sample {i} ALL Ratio of masks in noisy_batch_rev: {sum(noisy_batch_rev[i, :] == mask_token_id)}/{len(noisy_batch_rev[i, :])}")
    if use_complementary_loss:
        return noisy_batch, noisy_batch_rev, masked_indices, masked_indices_rev
    return noisy_batch, masked_indices


class DLMTrainer(Trainer):
    """Trajectory-based diffusion language model trainer for DREAM"""
    
    def __init__(
        self,
        mask_token_id=151666,
        temperature=0.5,
        entropy_weight=1.0,
        progressive_block_sizes=None,
        min_mask_ratio=0.2,
        max_mask_ratio=0.8,
        use_blockwise_loss=False,
        use_naive_random_mask=False,
        use_random_ratio_mask=False,
        use_complementary_loss=False,
        use_heuristic_trajectory=True,
        heuristic_swap_frac=HEURISTIC_SWAP_FRAC,
        heuristic_num_swaps=None,
        heuristic_long_swap_prob=HEURISTIC_LONG_SWAP_PROB,
        heuristic_max_swap_distance=HEURISTIC_MAX_SWAP_DISTANCE,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mask_token_id = mask_token_id
        self.temperature = temperature
        self.entropy_weight = entropy_weight
        self.progressive_block_sizes = progressive_block_sizes or [32]
        self.current_block_size = self.progressive_block_sizes[0]
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio
        self.use_blockwise_loss = use_blockwise_loss
        self.use_naive_random_mask = use_naive_random_mask
        self.use_random_ratio_mask = use_random_ratio_mask
        self.use_complementary_loss = use_complementary_loss
        self.use_heuristic_trajectory = use_heuristic_trajectory
        self.heuristic_swap_frac = heuristic_swap_frac
        self.heuristic_num_swaps = heuristic_num_swaps
        self.heuristic_long_swap_prob = heuristic_long_swap_prob
        self.heuristic_max_swap_distance = heuristic_max_swap_distance
    
    def get_current_block_size(self):
        """Calculate current block size based on epoch progress (linear interpolation)"""
        if self.state.epoch is None:
            return self.progressive_block_sizes[0]
        
        current_epoch = self.state.epoch
        epoch_idx = int(current_epoch)
        epoch_idx = min(epoch_idx, len(self.progressive_block_sizes) - 1)
        
        start_block_size = self.progressive_block_sizes[epoch_idx]
        
        if epoch_idx >= len(self.progressive_block_sizes) - 1:
            return int(start_block_size)
        
        end_block_size = self.progressive_block_sizes[epoch_idx + 1]
        epoch_progress = current_epoch - epoch_idx
        interpolated_size = start_block_size + epoch_progress * (end_block_size - start_block_size)
        
        return int(interpolated_size)
    
    def get_current_mask_ratio(self):
        """Calculate current mask ratio based on training progress (linear schedule)"""
        if self.state.max_steps > 0:
            current_step = self.state.global_step
            total_steps = self.state.max_steps
            progress = min(current_step / total_steps, 1.0)
            current_ratio = self.min_mask_ratio + progress * (self.max_mask_ratio - self.min_mask_ratio)
            return current_ratio
        else:
            return self.min_mask_ratio
    
    def _get_gpu_stats(self):
        """Get GPU memory and utilization statistics"""
        try:
            import json
            result = subprocess.run(
                ['gpustat', '--json'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                gpu_data = json.loads(result.stdout)
                total_memory_used = 0
                total_memory_total = 0
                total_utilization = 0
                num_gpus = len(gpu_data['gpus'])
                
                for gpu in gpu_data['gpus']:
                    total_memory_used += gpu['memory.used']
                    total_memory_total += gpu['memory.total']
                    total_utilization += gpu['utilization.gpu']
                
                if num_gpus > 0:
                    avg_memory_used = total_memory_used / num_gpus
                    avg_memory_total = total_memory_total / num_gpus
                    avg_utilization = total_utilization / num_gpus
                    memory_percent = (total_memory_used / total_memory_total * 100) if total_memory_total > 0 else 0
                    
                    return {
                        'gpu_memory_used_mb': avg_memory_used,
                        'gpu_memory_total_mb': avg_memory_total,
                        'gpu_memory_percent': memory_percent,
                        'gpu_utilization_percent': avg_utilization,
                        'num_gpus': num_gpus
                    }
        except Exception:
            pass
        
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                total_memory_used = 0
                total_memory_total = 0
                total_utilization = 0
                num_gpus = len(lines)
                
                for line in lines:
                    parts = line.split(',')
                    if len(parts) == 3:
                        total_memory_used += float(parts[0].strip())
                        total_memory_total += float(parts[1].strip())
                        total_utilization += float(parts[2].strip())
                
                if num_gpus > 0:
                    avg_memory_used = total_memory_used / num_gpus
                    avg_memory_total = total_memory_total / num_gpus
                    avg_utilization = total_utilization / num_gpus
                    memory_percent = (total_memory_used / total_memory_total * 100) if total_memory_total > 0 else 0
                    
                    return {
                        'gpu_memory_used_mb': avg_memory_used,
                        'gpu_memory_total_mb': avg_memory_total,
                        'gpu_memory_percent': memory_percent,
                        'gpu_utilization_percent': avg_utilization,
                        'num_gpus': num_gpus
                    }
        except Exception:
            pass
        
        return None
    
    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        """Override log to add GPU statistics and current mask ratio"""
        gpu_stats = self._get_gpu_stats()
        if gpu_stats:
            logs.update(gpu_stats)
        
        logs['mask_ratio'] = self.get_current_mask_ratio()
        logs['block_size'] = self.get_current_block_size()
        
        super().log(logs, *args, **kwargs)
    
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs["input_ids"]
        prompt_lengths = inputs["prompt_lengths"]
        sample_indices = inputs["sample_idx"]
        
        gs = int(self.state.global_step) if self.state is not None else 0
        
        # Get current mask ratio and block size
        current_mask_ratio = self.get_current_mask_ratio()
        current_mask_ratio = random.uniform(current_mask_ratio, self.max_mask_ratio)
        current_block_size = self.get_current_block_size()
        
        # Forward masking: heuristic pseudo-trajectory (or i.i.d. random if use_naive_random_mask)
        if self.use_complementary_loss:
            noisy_batch, noisy_batch_rev, masked_indices, masked_indices_rev = forward_process_with_trajectory(
                input_ids, prompt_lengths,
                mask_token_id=self.mask_token_id, block_size=current_block_size,
                mask_ratio=current_mask_ratio, use_blockwise=self.use_blockwise_loss,
                use_naive_random_mask=self.use_naive_random_mask,
                use_random_ratio_mask=self.use_random_ratio_mask,
                use_heuristic_trajectory=self.use_heuristic_trajectory,
                heuristic_swap_frac=self.heuristic_swap_frac,
                heuristic_num_swaps=self.heuristic_num_swaps,
                heuristic_long_swap_prob=self.heuristic_long_swap_prob,
                heuristic_max_swap_distance=self.heuristic_max_swap_distance,
                use_complementary_loss=True,
                sample_indices=sample_indices,
                global_step=gs,
            )
        else:
            noisy_batch, masked_indices = forward_process_with_trajectory(
                input_ids, prompt_lengths,
                mask_token_id=self.mask_token_id, block_size=current_block_size,
                mask_ratio=current_mask_ratio, use_blockwise=self.use_blockwise_loss,
                use_naive_random_mask=self.use_naive_random_mask,
                use_random_ratio_mask=self.use_random_ratio_mask,
                use_heuristic_trajectory=self.use_heuristic_trajectory,
                heuristic_swap_frac=self.heuristic_swap_frac,
                heuristic_num_swaps=self.heuristic_num_swaps,
                heuristic_long_swap_prob=self.heuristic_long_swap_prob,
                heuristic_max_swap_distance=self.heuristic_max_swap_distance,
                sample_indices=sample_indices,
                global_step=gs,
            )
        
        # token shift
        masked_indices = masked_indices[:, 1:]
        masked_indices_rev = masked_indices_rev[:, 1:] if self.use_complementary_loss else None
        
        # compute logits（保持与模型一致的 dtype，通常为 bf16；勿对整段 [B,T,V] 转 float32，否则显存约翻倍且 softmax 再占一张同尺寸张量易 OOM）
        outputs = model(input_ids=noisy_batch)
        logits = outputs.logits[:, :-1]

        # compute logits for complementary mask
        if self.use_complementary_loss:
            outputs_rev = model(input_ids=noisy_batch_rev)
            logits_rev = outputs_rev.logits[:, :-1]
        
        input_ids = input_ids[:, 1:]
        # Calculate loss: only calculate loss for masked tokens
        if masked_indices.sum() > 0:
            # Get the logits and labels of the masked positions
            masked_logits = logits[masked_indices]  # [num_masked, vocab_size]
            masked_labels = input_ids[masked_indices]  # [num_masked]
            
            # 仅在 masked 子集上转 FP32，数值更稳且不把整段序列升到 float32
            ce_loss = F.cross_entropy(masked_logits.float(), masked_labels)
        else:
            ce_loss = 0.0 * logits.sum()
        
        # Calculate loss: only calculate loss for masked tokens
        if self.use_complementary_loss and masked_indices_rev.sum() > 0:
            # Get the logits and labels of the masked positions
            masked_logits_rev = logits_rev[masked_indices_rev]  # [num_masked, vocab_size]
            masked_labels_rev = input_ids[masked_indices_rev]  # [num_masked]
            
            # cross entropy loss with automatic mean reduction
            ce_loss_rev = F.cross_entropy(masked_logits_rev.float(), masked_labels_rev)
        else:
            ce_loss_rev = 0.0 * logits.sum() if self.use_complementary_loss else 0.0 * logits.sum()
        
        # ---------- Apply entropy loss only to "correctly predicted" tokens ----------
        if masked_indices.sum() > 0:
            # Calculate the probability and entropy of each position
            # Note: argmax is not affected by temperature; logits/probs are equivalent.
            probs = F.softmax(logits / self.temperature, dim=-1)  # [B, T, V]，与 logits 同 dtype，避免额外 FP32 全词表张量
            H_tok = -(probs * (probs.clamp_min(1e-12)).log()).sum(dim=-1)  # [B, T]
            
            # predictions
            pred_ids = logits.argmax(dim=-1)  # [B, T]
            
            # Only keep: positions that are masked and predicted == label
            correct_mask = (pred_ids == input_ids) & masked_indices  # [B, T] bool
            
            num_correct = correct_mask.sum()
            if num_correct.item() > 0:
                # Minimize entropy only for the "correctly predicted" positions
                entropy_loss = (H_tok * correct_mask).sum() / num_correct.clamp_min(1)
            else:
                entropy_loss = 0.0 * logits.sum()
        else:
            entropy_loss = 0.0 * logits.sum()
        
        # ---------- Apply entropy loss only to "correctly predicted" tokens ----------
        if self.use_complementary_loss and masked_indices_rev.sum() > 0:
            # Calculate the probability and entropy of each position
            # Note: argmax is not affected by temperature; logits/probs are equivalent.
            probs_rev = F.softmax(logits_rev / self.temperature, dim=-1)  # [B, T, V]
            H_tok_rev = -(probs_rev * (probs_rev.clamp_min(1e-12)).log()).sum(dim=-1)  # [B, T]
            
            # predictions
            pred_ids_rev = logits_rev.argmax(dim=-1)  # [B, T]
            
            # Only keep: positions that are masked and predicted == label
            correct_mask_rev = (pred_ids_rev == input_ids) & masked_indices_rev  # [B, T] bool
            
            num_correct_rev = correct_mask_rev.sum()
            if num_correct_rev.item() > 0:
                # Minimize entropy only for the "correctly predicted" positions
                entropy_loss_rev = (H_tok_rev * correct_mask_rev).sum() / num_correct_rev.clamp_min(1)
            else:
                entropy_loss_rev = 0.0 * logits_rev.sum()
        else:
            entropy_loss_rev = 0.0 * logits.sum()
        
        # ==================== combined total loss ====================
        if self.use_complementary_loss:
            total_loss = (ce_loss + ce_loss_rev + self.entropy_weight * (entropy_loss + entropy_loss_rev)) / 4.0
        else:
            total_loss = (ce_loss + self.entropy_weight * entropy_loss) / 4.0
        
        return (total_loss, outputs) if return_outputs else total_loss


def main():
    # 1. Load configuration, model and tokenizer
    import os
    from datetime import datetime
    import shutil
    from zoneinfo import ZoneInfo
    
    config_path = os.path.join(os.path.dirname(__file__), "d3llm_train.yaml")
    config = load_config(config_path)
    
    # Override config from command line args
    config = override_config(config, sys.argv[1:])
    
    # Save modified config as d3llm_train_used.yaml for backup
    used_config_path = os.path.join(os.path.dirname(__file__), "d3llm_train_used.yaml")
    with open(used_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    # Print all configuration parameters
    print(f"=" * 80)
    print("Configuration Parameters:")
    print(f"=" * 80)
    import json
    print(json.dumps(config, indent=2, ensure_ascii=False))
    print(f"=" * 80)
    
    # Get SLURM job ID if available
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")
    
    # Use timestamp from environment (set by shell) or generate new one
    timestamp = os.environ.get("TRAINING_TIMESTAMP")
    if not timestamp:
        san_diego_tz = ZoneInfo("America/Los_Angeles")
        timestamp = datetime.now(san_diego_tz).strftime("%m%d_%H%M%S")
    
    base_output_dir = config["training"]["output_dir"]
    output_dir = f"{base_output_dir}_{slurm_job_id}_{timestamp}"
    
    # Create W&B run name with the same format
    wandb_run_name = f"{os.path.basename(base_output_dir)}_{slurm_job_id}_{timestamp}"
    
    # Update config with timestamped output_dir and run_name
    config["training"]["output_dir"] = output_dir
    config["training"]["run_name"] = wandb_run_name
    
    print(f"=" * 80)
    print(f"SLURM Job ID: {slurm_job_id}")
    print(f"Output directory: {output_dir}")
    print(f"W&B Run name: {wandb_run_name}")
    print(f"=" * 80)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Backup training code and config to output directory
    source_dir = os.path.dirname(__file__)
    backup_dir = os.path.join(output_dir, "training_code_backup")
    
    print(f"Backing up training code from {source_dir} to {backup_dir}...")
    
    try:
        shutil.copytree(source_dir, backup_dir, dirs_exist_ok=True)
        print(f"Training code backed up successfully!")
    except Exception as e:
        print(f"Warning: Failed to backup training code: {e}")
        print(f"Continuing with training anyway...")
    
    print(f"=" * 80)
    
    training_args = TrainingArguments(
        **config["training"],
        deepspeed=get_deepspeed_config(config),
        ddp_find_unused_parameters=False,
        label_names=["input_ids", "prompt_lengths", "sample_idx"],
    )
    
    model, tokenizer = prepare_model(config)
    
    distill_config = config.get("distillation", {})
    use_heuristic_trajectory = distill_config.get("use_heuristic_trajectory", True)
    if distill_config.get("heuristic_swap_frac") is not None:
        heuristic_swap_frac = float(distill_config["heuristic_swap_frac"])
    else:
        # 兼容旧键 heuristic_jump_prob（量级相近时可沿用）
        heuristic_swap_frac = float(distill_config.get("heuristic_jump_prob", HEURISTIC_SWAP_FRAC))
    _ns = distill_config.get("heuristic_num_swaps")
    heuristic_num_swaps: Optional[int] = None if _ns is None else int(_ns)
    heuristic_long_swap_prob = float(distill_config.get("heuristic_long_swap_prob", HEURISTIC_LONG_SWAP_PROB))
    heuristic_max_swap_distance = int(distill_config.get("heuristic_max_swap_distance", HEURISTIC_MAX_SWAP_DISTANCE))
    use_random_ratio_mask = bool(distill_config.get("use_random_ratio_mask", False))
    use_naive_random_mask_cfg = bool(distill_config.get("use_naive_random_mask", False))
    print(
        "Masking: "
        + (
            "random k-mask by mask_ratio (uniform positions, k=round(ratio*L))"
            if use_random_ratio_mask
            else (
                "naive i.i.d. Bernoulli per token"
                if use_naive_random_mask_cfg
                else (
                    "heuristic pseudo-trajectory "
                    f"(enabled={use_heuristic_trajectory}, swap_frac={heuristic_swap_frac}, "
                    f"num_swaps={heuristic_num_swaps}, long_swap_prob={heuristic_long_swap_prob}, "
                    f"max_swap_distance={heuristic_max_swap_distance})"
                )
            )
        )
        + ". Reveal/mask is response-local only; prompt is not reordered."
    )
    
    # 2. Load the original dataset
    # dataset = load_dataset("Zigeng/dParallel_Dream_Distill_Data", split="train")
    dataset = load_dataset("d3LLM/Ling-Coder-dParallel-merged-512-120k", split="train")
    
    # Limit dataset size for testing if max_samples is specified
    max_samples = distill_config.get("max_samples")
    if max_samples is not None and max_samples > 0:
        original_size = len(dataset)
        dataset = dataset.select(range(min(max_samples, original_size)))
        print(f"=" * 80)
        print(f"[Testing Mode] Limited dataset from {original_size} to {len(dataset)} samples")
        
        print(f"=" * 80)
    
    # 4. Check tokenized dataset cache
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache key based on dataset configuration
    cache_params = {
        "model_name": config["model"]["name"],
        "max_samples": max_samples,
        "max_length": distill_config.get("max_length", 1000),
        "dataset_size": len(dataset),
    }
    cache_key_str = str(cache_params).encode()
    cache_key = hashlib.md5(cache_key_str).hexdigest()
    cache_file_tokenized = os.path.join(cache_dir, f"tokenized_dataset_{cache_key}.pkl")
    
    # Try to load tokenized dataset from cache
    if os.path.exists(cache_file_tokenized):
        try:
            print(f"=" * 80)
            print(f"Loading tokenized dataset from cache: {cache_file_tokenized}")
            with open(cache_file_tokenized, 'rb') as f:
                tokenized_dataset = pickle.load(f)
            print(f"Successfully loaded tokenized dataset with {len(tokenized_dataset)} samples from cache!")
            print(f"=" * 80)
        except Exception as e:
            print(f"Failed to load tokenized dataset cache: {e}")
            print(f"Will tokenize from scratch...")
            tokenized_dataset = None
    else:
        print(f"Tokenized dataset cache not found. Will tokenize from scratch...")
        tokenized_dataset = None
    
    # If cache doesn't exist or failed to load, perform tokenization
    if tokenized_dataset is None:
        # Format each sample, generate the complete text and record the number of tokens in the prompt section
        def format_example(example):
            texts = []
            prompt_lengths = []
            
            for question, response in zip(example["question"], example["llm_response"]):
                # prompt text
                messages = [{"role": "user", "content": question}]
                prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                
                # response text
                answer_text = response + tokenizer.eos_token
                
                # complete text
                full_text = prompt_text + answer_text
                texts.append(full_text)
                
                # Calculate the number of tokens in the prompt part
                prompt_token_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
                prompt_lengths.append(len(prompt_token_ids))
            
            return {"text": texts, "prompt_length": prompt_lengths}
        
        print(f"Formatting dataset...")
        formatted_dataset = dataset.map(
            format_example,
            batched=True,
        )
        
        def tokenize_function(examples, indices):
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=distill_config.get("max_length", 1000),
                add_special_tokens=False,
            )
            
            tokenized["prompt_lengths"] = examples["prompt_length"]
            
            tokenized["sample_idx"] = list(indices)
            
            return tokenized
        
        print(f"Tokenizing dataset...")
        tokenized_dataset = formatted_dataset.map(
            tokenize_function,
            batched=True,
            with_indices=True,
        )
        
        # Save tokenized dataset to cache
        try:
            print(f"Saving tokenized dataset to cache: {cache_file_tokenized}")
            with open(cache_file_tokenized, 'wb') as f:
                pickle.dump(tokenized_dataset, f)
            print(f"Tokenized dataset cache saved successfully!")
        except Exception as e:
            print(f"Warning: Failed to save tokenized dataset cache: {e}")

    # Print max prompt_lengths
    max_prompt_length = max(tokenized_dataset["prompt_lengths"])
    print(f"Max prompt length: {max_prompt_length}")

    from dataclasses import dataclass
    from typing import Dict, List, Any
    
    @dataclass
    class MaskDiffusionDataCollator:
        tokenizer: Any
        
        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
            input_ids = [torch.tensor(f["input_ids"]) for f in features]
            prompt_lengths = [f["prompt_lengths"] for f in features]
            sample_indices = [f["sample_idx"] for f in features]
            
            target_length = 512 + max(prompt_lengths)
            
            pad_token_id = self.tokenizer.eos_token_id
            
            # right padding
            padded_input_ids = []
            for ids in input_ids:
                current_length = len(ids)
                if current_length < target_length:
                    # Right padding with EOS token
                    padding_length = target_length - current_length
                    padded_ids = torch.cat([
                        ids,
                        torch.full((padding_length,), pad_token_id, dtype=ids.dtype)
                    ])
                else:
                    # Truncate to target_length
                    padded_ids = ids[:target_length]
                
                padded_input_ids.append(padded_ids)
            
            batch = {
                "input_ids": torch.stack(padded_input_ids),
                "prompt_lengths": torch.tensor(prompt_lengths, dtype=torch.long),
                "sample_idx": torch.tensor(sample_indices, dtype=torch.long),
            }
            
            return batch
    
    data_collator_fixed = MaskDiffusionDataCollator(
        tokenizer=tokenizer,
    )
    
    # 5. Create trainer and train
    progressive_block_sizes = distill_config.get("progressive_block_sizes", [32])
    num_epochs = config["training"]["num_train_epochs"]
    
    # Validate progressive_block_sizes length
    if len(progressive_block_sizes) != num_epochs:
        print(f"Warning: progressive_block_sizes length ({len(progressive_block_sizes)}) != num_epochs ({num_epochs})")
        print(f"Using last block size ({progressive_block_sizes[-1]}) for remaining epochs")
        progressive_block_sizes = progressive_block_sizes + [
            progressive_block_sizes[-1]
        ] * (num_epochs - len(progressive_block_sizes))
    
    # 6. DLM Trainer
    trainer = DLMTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator_fixed,
        mask_token_id=151666,
        temperature=distill_config.get("temperature", 0.5),
        entropy_weight=distill_config.get("entropy_weight", 1.0),
        progressive_block_sizes=progressive_block_sizes,
        min_mask_ratio=distill_config.get("min_mask_ratio", 0.2),
        max_mask_ratio=distill_config.get("max_mask_ratio", 0.8),
        use_blockwise_loss=distill_config.get("use_blockwise_loss", False),
        use_naive_random_mask=distill_config.get("use_naive_random_mask", False),
        use_random_ratio_mask=distill_config.get("use_random_ratio_mask", False),
        use_complementary_loss=distill_config.get("use_complementary_loss", False),
        use_heuristic_trajectory=use_heuristic_trajectory,
        heuristic_swap_frac=heuristic_swap_frac,
        heuristic_num_swaps=heuristic_num_swaps,
        heuristic_long_swap_prob=heuristic_long_swap_prob,
        heuristic_max_swap_distance=heuristic_max_swap_distance,
    )
    
    print(f"Training with progressive block sizes: {trainer.progressive_block_sizes}")
    print(f"Starting with block size: {trainer.current_block_size}")
    print(f"Progressive mask ratio: [{trainer.min_mask_ratio}, {trainer.max_mask_ratio}]")
    print(f"Temperature: {trainer.temperature}, Entropy weight: {trainer.entropy_weight}")
    
    # 6. start training
    trainer.train()


if __name__ == "__main__":
    main()
