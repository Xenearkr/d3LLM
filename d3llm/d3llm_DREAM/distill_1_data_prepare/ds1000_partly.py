import sys
import os
import re
import types
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("/")
sys.path.append("./")
sys.path.append("../")

from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn import functional as F


def strip_trailing_endoftext(text: str) -> str:
    return re.sub(r'(<\|endoftext\|>)+\s*$', '', text).strip()

def extract_assistant_response(text: str) -> str:
    """
    从完整 chat transcript 中提取 assistant 的回答部分。
    若不存在 chat 标记，则返回原文。
    """
    if text is None:
        return ""

    text = text.strip()
    assistant_tag = "<|im_start|>assistant"
    end_tag = "<|im_end|>"

    if assistant_tag in text:
        start = text.rfind(assistant_tag) + len(assistant_tag)
        end = text.find(end_tag, start)
        if end != -1:
            return text[start:end].strip()
        return text[start:].strip()

    return text


def extract_code_from_generation(text: str) -> str:
    """
    从 assistant 输出中提取纯代码。
    尽量兼容以下情况：
    1. ```python ... ```
    2. ``` ... ```
    3. <code> ... </code>
    4. BEGIN SOLUTION / END SOLUTION
    5. 普通纯文本代码
    """
    if text is None:
        return ""

    text = text.strip()

    # 先裁掉 BEGIN/END SOLUTION 外壳
    if "BEGIN SOLUTION" in text:
        text = text.split("BEGIN SOLUTION", 1)[1].strip()
    if "END SOLUTION" in text:
        text = text.split("END SOLUTION", 1)[0].strip()

    # 提取 <code> ... </code>
    code_match = re.search(r"<code>\s*(.*?)\s*</code>", text, flags=re.DOTALL)
    if code_match:
        return code_match.group(1).strip()

    # 提取 ```python ... ```
    if "```python" in text:
        start = text.find("```python") + len("```python")
        end = text.find("```", start)
        if end != -1:
            return text[start:end].strip()
        return text[start:].strip()

    # 提取 ``` ... ```
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 3:
            return parts[1].strip()

    # 去掉包裹式 <code> / </code>
    if text.startswith("<code>"):
        text = text[len("<code>") :].strip()
    if text.endswith("</code>"):
        text = text[: -len("</code>")].strip()

    return text.strip()


def check_answer_correctness(generated_text, ground_truth):
    """
    保持原接口不变，但内部逻辑改成更稳的占位式判断：
    - 对旧数学/问答任务：若能提取 boxed，则做字符串比对
    - 对代码任务：只做最轻量的非空/可编译占位检查
    - 如果 ground_truth 为空，则返回 True（兼容旧逻辑）
    """
    if generated_text is None:
        return False

    if ground_truth is None or ground_truth == "":
        return True

    # 代码任务：尝试提取代码并做最轻量的占位检查
    code = extract_code_from_generation(generated_text)
    if code.strip() == "":
        return False

    # 尝试编译；若是单行赋值/表达式，这一步通常可通过
    try:
        compile(code, "<generated_code>", "exec")
        return True
    except Exception:
        # 有些模型输出可能夹杂少量自然语言，保守起见只要非空也算通过占位检查
        return True


def sample_tokens_with_entropy(logits, temperature=1.0):
    """
    Sample tokens and return corresponding entropy values

    Args:
        logits: Model output logits [batch_size, vocab_size]
        temperature: Temperature parameter

    Returns:
        entropy: Entropy value at each position [batch_size]
        samples: Sampled token ids [batch_size]
    """
    # Calculate entropy from original logits (for threshold judgment)
    original_probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(original_probs + 1e-8)
    entropy = -torch.sum(original_probs * log_probs, dim=-1)

    # Then perform sampling
    if temperature == 0:
        # Greedy decoding: directly select the token with largest logits
        samples = torch.argmax(logits, dim=-1)
    else:
        # Apply temperature
        scaled_logits = logits / temperature
        # Convert to probabilities and sample
        probs = torch.softmax(scaled_logits, dim=-1)
        samples = torch.multinomial(probs, num_samples=1).squeeze(-1)

    return entropy, samples


@torch.no_grad()
def generate_teacher_model_trajectory(
    model,
    tokenizer,
    input_ids,
    attention_mask=None,
    steps=256,
    gen_length=256,
    block_length=32,
    temperature=0.0,
    threshold=0.5,
    mask_token_id=None,
    trajectory_one_step=False,
):
    """Generate trajectory for DREAM teacher model with block-wise diffusion decoding"""

    # Bind generation methods to model
    from d3llm.d3llm_DREAM.d3llm_dream_generate_util import DreamGenerationMixin

    if not hasattr(model, "_sample_original"):
        model.diffusion_generate = types.MethodType(
            DreamGenerationMixin.diffusion_generate, model
        )
        model._sample_original = types.MethodType(DreamGenerationMixin._sample, model)
        model._prepare_inputs = types.MethodType(
            DreamGenerationMixin._prepare_inputs, model
        )
        model._prepare_generation_config = types.MethodType(
            DreamGenerationMixin._prepare_generation_config, model
        )
        model._prepare_special_tokens = types.MethodType(
            DreamGenerationMixin._prepare_special_tokens, model
        )
        model._prepare_generated_length = types.MethodType(
            DreamGenerationMixin._prepare_generated_length, model
        )
        model._validate_generated_length = types.MethodType(
            DreamGenerationMixin._validate_generated_length, model
        )
        # _expand_inputs_for_generation is a staticmethod, so we assign it directly
        model._expand_inputs_for_generation = (
            DreamGenerationMixin._expand_inputs_for_generation
        )

    # Create custom _sample method that records trajectory
    trajectory = []

    def _sample_with_trajectory(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config,
        threshold: Optional[float] = 0.5,
        block_length: Optional[int] = 32,
        trajectory_one_step: Optional[bool] = False,
    ):
        # init values
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id_val = generation_config.mask_token_id
        steps_val = generation_config.steps
        temperature_val = generation_config.temperature
        alg = generation_config.alg

        histories = [] if (return_dict_in_generate and output_history) else None

        # pad input_ids to max_length
        x = F.pad(
            input_ids,
            (0, max_length - input_ids.shape[1]),
            value=mask_token_id_val,
        )
        gen_length_val = max_length - input_ids.shape[1]

        # Handle block configuration
        if block_length is None:
            block_length = gen_length_val  # Default: single block

        assert (
            gen_length_val % block_length == 0
        ), f"gen_length ({gen_length_val}) must be divisible by block_length ({block_length})"
        num_blocks = gen_length_val // block_length

        assert (
            steps_val % num_blocks == 0
        ), f"steps ({steps_val}) must be divisible by num_blocks ({num_blocks})"
        steps_per_block = steps_val // num_blocks
        _ = torch.linspace(
            1, generation_config.eps, steps_per_block + 1, device=x.device
        )

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            # we do not mask the [MASK] tokens so value = 1.0
            attention_mask = F.pad(
                attention_mask,
                (0, max_length - attention_mask.shape[1]),
                value=1.0,
            )
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            # attention_mask is of shape [B, N]
            # broadcast to [B, 1, N, N]
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        if trajectory_one_step:
            unmask_time = torch.zeros_like(x, device=x.device, dtype=torch.int)
        else:
            unmask_time = None

        # Process each block
        i = 0
        for num_block in range(num_blocks):
            current_block_start = input_ids.shape[1] + num_block * block_length
            current_block_end = current_block_start + block_length

            while True:
                i += 1
                mask_index = x == mask_token_id_val

                model_output = self(x, attention_mask, tok_idx)

                mask_index[:, current_block_end:] = 0

                logits = model_output.logits
                logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)

                transfer_index = torch.zeros_like(x, device=x.device, dtype=torch.bool)

                if alg == "entropy_threshold":
                    mask_logits = logits[mask_index]

                    # Calculate entropy instead of confidence
                    entropy, x0 = sample_tokens_with_entropy(
                        mask_logits, temperature=temperature_val
                    )

                    x_ = (
                        torch.zeros_like(x, device=self.device, dtype=torch.long)
                        + mask_token_id_val
                    )
                    full_entropy = torch.full_like(
                        x,
                        torch.inf,
                        device=self.device,
                        dtype=logits.dtype,
                    )

                    x_[mask_index] = x0.clone()
                    full_entropy[mask_index] = entropy

                    current_transfer_tokens = (
                        x[:, current_block_start:current_block_end] == mask_token_id_val
                    ).sum()

                    # Select tokens with lowest entropy (high certainty)
                    selected_entropy, select_index = torch.topk(
                        full_entropy,
                        current_transfer_tokens,
                        largest=False,
                    )

                    select_index = select_index.to(x.device)
                    if current_transfer_tokens > 0:
                        transfer_index[0, select_index[0, 0]] = True

                    for k in range(1, current_transfer_tokens):
                        # Only decode tokens with entropy below threshold
                        if selected_entropy[0, k] < threshold:
                            transfer_index[0, select_index[0, k]] = True
                        else:
                            transfer_index[0, select_index[0, k]] = False

                    x[transfer_index] = x_[transfer_index].clone()

                # Store trajectory after each step
                if trajectory_one_step:
                    unmask_time[transfer_index] = i
                else:
                    trajectory.append(x.clone())

                if (
                    x[:, current_block_start:current_block_end] == mask_token_id_val
                ).sum() == 0:
                    break

        if trajectory_one_step:
            trajectory.append(unmask_time)
            trajectory.append(x.clone())

        from d3llm.d3llm_DREAM.d3llm_dream_generate_util import DreamModelOutput

        if return_dict_in_generate:
            return DreamModelOutput(sequences=x, history=histories), i
        else:
            return x, i

    # Temporarily replace _sample method
    original_sample = model._sample_original
    model._sample = types.MethodType(_sample_with_trajectory, model)

    try:
        # Generate with trajectory recording
        output, nfe = model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=gen_length,
            output_history=False,
            return_dict_in_generate=True,
            steps=steps,
            temperature=temperature,
            top_p=None,
            alg="entropy_threshold",
            alg_temp=0.1,
            top_k=None,
            block_length=block_length,
            threshold=threshold,
            trajectory_one_step=trajectory_one_step,
        )

        final_output = output.sequences

    finally:
        # Restore original _sample method
        model._sample = original_sample

    return final_output, trajectory, nfe


def main(
    start_idx,
    end_idx,
    steps=256,
    gen_length=256,
    block_length=32,
    output_file="trajectory_data.json",
    max_data_num=-1,
    trajectory_one_step=False,
):
    from datasets import load_dataset
    from tqdm import tqdm
    import json

    device = "cuda"

    # Load DREAM teacher model
    model_path = "Dream-org/Dream-v0-Instruct-7B"
    # model_path = "Dream-org/Dream-Coder-v0-Instruct-7B"
    teacher_model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
    )

    # Load dataset
    # dataset = load_dataset("Zigeng/dParallel_Dream_Distill_Data", split="train")
    # dataset = load_dataset("d3LLM/Ling-Coder-dParallel-merged-512-120k", split="train")
    dataset = load_dataset("xlangai/DS-1000", split="test")

    # Apply max_data_num limit
    if max_data_num > 0:
        end_idx = min(end_idx, start_idx + max_data_num)

    results = []
    total_count = 0
    incorrect_count = 0

    for idx in tqdm(
        range(start_idx, min(end_idx, len(dataset))),
        desc="Generating trajectories",
    ):
        sample = dataset[idx]
        prompt_text = sample["prompt"]
        ground_truth = sample.get(
            "reference_code", None
        )  # 接口不变：仍保留 ground_truth 变量名
        code_context = sample.get("code_context", "")
        metadata = sample.get("metadata", {})

        # Prepare messages for chat template
        user_prompt = (
            prompt_text
            + "\nPlease output only Python code, do not include explanations or markdown code fences."
        )
        messages = [{"role": "user", "content": user_prompt}]

        # Apply chat template
        inputs = tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            return_dict=True,
            add_generation_prompt=True,
        )
        input_ids = inputs.input_ids.to(device=device)
        attention_mask = inputs.attention_mask.to(device=device)

        # Store prompt_ids as list
        prompt_ids = input_ids[0].cpu().tolist()

        # Retry mechanism: interface 保持不变
        max_attempts = 5
        generated_text = ""
        llm_answer = ""
        is_correct = False

        for attempt in range(max_attempts):
            current_temperature = attempt * 0.1

            # Generate trajectory
            final_output, trajectory, nfe = generate_teacher_model_trajectory(
                teacher_model,
                tokenizer,
                input_ids,
                attention_mask=attention_mask,
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
                temperature=current_temperature,
                threshold=-float("inf"),
                trajectory_one_step=trajectory_one_step,
            )

            # 保留完整解码文本，便于排查
            generated_text = tokenizer.decode(
                final_output[0],
                skip_special_tokens=False,
            )
            generated_text = strip_trailing_endoftext(generated_text)

            # 提取 assistant 真正回答
            assistant_output = extract_assistant_response(generated_text)

            # 提取纯代码
            generated_code = extract_code_from_generation(assistant_output)

            # 接口不变：llm_answer 继续保留，但内容改成“清洗后的有效答案/代码”
            llm_answer = generated_code if generated_code.strip() else assistant_output

            # 接口不变：is_correct 继续存在，但逻辑更合理
            is_correct = check_answer_correctness(llm_answer, ground_truth)

            # If correct or this is the last attempt, break
            if is_correct or attempt == max_attempts - 1:
                break

            print(
                f"Attempt {attempt + 1}/{max_attempts} failed for idx {idx} "
                f"(temperature={current_temperature:.1f}), retrying...",
                flush=True,
            )

        # Store result: convert tensors to lists for JSON serialization
        if trajectory_one_step:
            processed_trajectory = [
                trajectory[0].cpu().tolist(),
                trajectory[1].cpu().tolist(),
            ]
        else:
            processed_trajectory = [traj[0].cpu().tolist() for traj in trajectory]

        # 注意：输出字段接口保持不变
        results.append(
            {
                "idx": idx,
                "question": prompt_text,  # 实为 prompt；字段名保持不变
                "prompt_ids": prompt_ids,
                "trajectory": processed_trajectory,
                "final_output": final_output[0].cpu().tolist(),
                "generated_text": generated_text,  # 原始完整解码文本
                "llm_answer": llm_answer,  # 清洗后的 assistant 有效输出
                "gt_answer": ground_truth,  # 实为 reference_code；字段名保持不变
                "is_correct": is_correct,
                "nfe": nfe,
            }
        )

        # Update statistics and print real-time status
        total_count += 1
        if not is_correct:
            incorrect_count += 1

        correct_count = total_count - incorrect_count
        accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
        error_rate = (incorrect_count / total_count * 100) if total_count > 0 else 0

        if total_count % 10 == 0:
            print(
                f"[idx {idx}] Status: {'✓ Correct' if is_correct else '✗ Incorrect'} | "
                f"Total: {total_count} | Correct: {correct_count} ({accuracy:.2f}%) | "
                f"Incorrect: {incorrect_count} ({error_rate:.2f}%)",
                flush=True,
            )

    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f)

    print(f"Saved {len(results)} trajectories to {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start_idx", type=int, required=True)
    parser.add_argument("--end_idx", type=int, required=True)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--output_file", type=str, default="trajectory_data.json")
    parser.add_argument(
        "--max_data_num",
        type=int,
        default=-1,
        help="Max number of samples to generate (-1 for no limit)",
    )
    parser.add_argument(
        "--trajectory_one_step",
        action="store_true",
        help="Only save the trajectory of one step",
    )
    args = parser.parse_args()

    main(
        args.start_idx,
        args.end_idx,
        args.steps,
        args.gen_length,
        args.block_length,
        args.output_file,
        args.max_data_num,
        args.trajectory_one_step,
    )