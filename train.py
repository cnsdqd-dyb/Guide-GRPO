"""
Implementation of GRPO, DeepSeek style training without external libraries 
"""
import os
import json
import torch
import argparse
from tqdm import tqdm
from collections import defaultdict
from transformers import PreTrainedModel, PreTrainedTokenizerBase, GenerationConfig
import random
import llms
import wandb
import utils
import evaluator
import rldatasets
import numpy as np

def eval_on_test_set(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    test_loader: rldatasets.DataLoader,
    eval_class: evaluator.RewardEvaluator,
    device: str,
    args: argparse.Namespace,
    round_num: int
) -> tuple[dict[str, float], float]:
    """
    Evaluate model performance on test set.
    
    Args:
        model: The model to evaluate
        tokenizer: Tokenizer for the model
        test_loader: DataLoader for test set
        eval_class: Evaluator for computing rewards
        device: Device to run on
        args: Training arguments
        round_num: Current training round number
        
    Returns:
        total_scores: Dictionary of average metrics
        accuracy: Accuracy on test set
    """
    print("Running evaluation on test set...")
    
    # Track metrics across all test examples
    total_scores = defaultdict(float)
    num_examples = 0
    total_accuracy = 0.0

    # Create log file for this evaluation round
    log_file = os.path.join(args.output_dir, f'eval_metrics_{round_num}.txt')
    test_loader.reset()
    
    with open(log_file, 'w') as f:
        # Run through test set
        for question, answer in tqdm(test_loader, desc="Evaluating on test set"):
            # Generate completions using same function as training
            with torch.inference_mode():
                _, _, _, completions_text, _ = generate_segment_completions(
                    model, tokenizer, question, device, args
                )
            
            # Score completions using evaluator
            mock_prompts = [[{'content': question}]] * len(completions_text)
            mock_completions = [[{'content': completion}] for completion in completions_text]
            # Make answer array same length as completions
            answers = [answer] * len(completions_text)
            rewards_per_func, metrics = eval_class.compute_rewards(
                prompts=mock_prompts,
                completions=mock_completions, 
                answer=answers,
                device=device
            )
            
            # Track accuracy and accumulate metrics
            total_accuracy += metrics['accuracy']
                
            for k, v in metrics.items():
                total_scores[k] += v
            num_examples += 1

            # Log this example
            f.write("\n" + "="*50 + "\n")
            f.write(f"Q# {num_examples}\n")
            f.write(f"Question: {question}\n")
            f.write(f"Response: {completions_text[0]}\n") # Log first completion
            f.write(f"Ground Truth: {answer}\n")
            f.write("Metrics:\n")
            for metric, value in metrics.items():
                f.write(f"{metric}: {value}\n")
            f.write(f"Total Score: {rewards_per_func.sum().item()}\n")


    # Calculate averages
    avg_scores = {k: v/num_examples for k,v in total_scores.items()}
    accuracy = total_accuracy / num_examples * 100

    # Save metrics
    metrics_path = os.path.join(args.output_dir, f'eval_metrics_{round_num}.json')
    with open(metrics_path, 'w') as f:
        json.dump({**avg_scores, 'accuracy': accuracy}, f, indent=4)

    if args.verbose:
        print("\nEvaluation Results:")
        print("-" * 20)
        print(f"Accuracy: {accuracy:.2f}%")
        for metric, value in avg_scores.items():
            print(f"{metric:15s}: {value:.4f}")
        print("-" * 20)

    return avg_scores, accuracy

def find_last_punctuation_token(token_ids: torch.Tensor, tokenizer) -> torch.Tensor:
    """找到最后一个标点符号的位置"""
    assert token_ids.dim() == 2, "Input tensor must be 2D"
    assert token_ids.size(0) == 1, "Input tensor must have batch size of 1"
    
    punct_tokens = set()
    for punct in ['\n','.', ',', '!', '?']:
        punct_ids = tokenizer(punct, add_special_tokens=False)['input_ids']
        punct_tokens.add(punct_ids[0])
    
    punct_tokens = torch.tensor(list(punct_tokens), device=token_ids.device)
    # print(punct_tokens)
    # 创建标点符号位置的mask
    cutoff_pos = -1
    for punct_id in punct_tokens:
        for i in range(token_ids.size(1)):
            if token_ids[0, i] == punct_id:
                cutoff_pos = max(cutoff_pos, i)
        if cutoff_pos != -1:
            break
    # print(cutoff_pos)
    
    return cutoff_pos


def multi_factor_weight(d, alpha=0.4, beta=0.4, gamma=0.2, lambda_=0.1, d_max=100):
    exp_component = alpha * torch.exp(-lambda_ * d)
    linear_component = beta * (1 - d/d_max)
    constant_component = gamma
    return exp_component + linear_component + constant_component


def generate_segment_completions(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase, 
    question: str,
    device: str,
    args: argparse.Namespace
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[str], str]:
    """
    Generate multiple completion sequences for a given prompt using a language model.
    
    Args:
        model: The language model to use for generation
        tokenizer: Tokenizer corresponding to the model
        question: The input question/prompt to generate completions for
        device: Device to run generation on ('cpu' or 'cuda')
        args: Namespace containing generation parameters
        
    Returns:
        prompt_completion_ids: Tensor containing the full sequence of prompt + completion token IDs
        prompt_ids: Tensor containing just the prompt token IDs
        completion_ids: Tensor containing just the completion token IDs
        attention_mask: Attention mask tensor for the full sequence
        completions_text: List of decoded completion texts
        prompt_text: The full formatted prompt text
    """
    # 1. Prepare prompting
    prompt = [
        {'role': 'system', 'content': train_loader.system_prompt},
        {'role': 'user', 'content': question}
    ]
    prompt_text = tokenizer.apply_chat_template(prompt, tokenize=False)
    prompt_inputs = tokenizer(prompt_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False)
    prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

    # Truncate prompt to max length and repeat for number of generations
    prompt_ids = prompt_ids[:, -args.max_prompt_length:] # [1, seq_len]
    prompt_mask = prompt_mask[:, -args.max_prompt_length:] # [1, seq_len]


    # Move tensors to device
    prompt_ids = prompt_ids.to(device)
    prompt_mask = prompt_mask.to(device)

    # Set up generation config
    max_new_prefix_length = random.randint(1, args.max_completion_length-128)
    generation_config_prefix = GenerationConfig(
        max_new_tokens=max_new_prefix_length,
        do_sample=True, 
        temperature=args.temperature,
        pad_token_id=tokenizer.pad_token_id,
    )
    max_new_guidance_length = random.randint(2, args.max_guide_tokens)
    generation_config_guidance = GenerationConfig(
        max_new_tokens=max_new_guidance_length,
        do_sample=True, 
        temperature=args.temperature,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    generation_config_postfix = GenerationConfig(
        max_new_tokens=args.max_completion_length,
        do_sample=True, 
        temperature=args.temperature,
        pad_token_id=tokenizer.pad_token_id,
    )

    # Generate prefix completions
    # print("Generating prefix completions...")
    max_attempts = 10  # 设置最大尝试次数，防止无限循环
    attempt = 0
    while True:
        attempt += 1
        if attempt > max_attempts:
            raise RuntimeError(f"Failed to generate valid prefix after {max_attempts} attempts")
        
        # max_new_prefix_length
        with torch.inference_mode():
            prompt_prefix_ids = model.generate(
                prompt_ids,
                attention_mask=prompt_mask,
                generation_config=generation_config_prefix
            )
        
        # Check if any completions are valid
        prefix_only = prompt_prefix_ids[:, prompt_ids.size(1):]
        is_eos = (prefix_only == tokenizer.eos_token_id).any(dim=1)
        
        if not is_eos.any():
            # Success - no EOS in prefix
            break
        
        print(f"Retrying prefix generation... (attempt {attempt}/{max_attempts})")
        max_new_prefix_length = max(max_new_prefix_length - random.randint(max_new_prefix_length//8, max_new_prefix_length//2), 0)
        generation_config_prefix.max_new_tokens = max_new_prefix_length
    
    # prompt_prefix_ids cut to the last punctuation
    cutoff_pos = find_last_punctuation_token(prompt_prefix_ids, tokenizer)
    if cutoff_pos >= prompt_ids.size(1):
        prompt_prefix_ids = prompt_prefix_ids[:, :cutoff_pos+1]
    else:
        prompt_prefix_ids = prompt_ids
    
        
    prompt_prefix_mask = torch.cat([prompt_mask,
        torch.ones_like(prompt_prefix_ids[:, prompt_ids.size(1):]).to(prompt_mask.device)], dim=1)
    
    # repeat for number of chains/generations
    prompt_prefix_ids = prompt_prefix_ids.repeat(args.num_chains, 1)
    prompt_prefix_mask = prompt_prefix_mask.repeat(args.num_chains, 1)
    
    # Generate guidance completions
    # print("Generating guidance completions...")
    rest_tokens = args.max_completion_length - prompt_prefix_ids.size(1)
    generation_config_guidance.max_new_tokens = min(generation_config_guidance.max_new_tokens, rest_tokens)
    prompt_prefix_guidance_ids = model.generate(
        prompt_prefix_ids,
        attention_mask=prompt_prefix_mask,
        generation_config=generation_config_guidance
    )
    
    prompt_prefix_guidance_mask = torch.cat([prompt_prefix_mask,
        torch.ones_like(prompt_prefix_guidance_ids[:, prompt_prefix_ids.size(1):]).to(prompt_mask.device)], dim=1)
    
    prompt_prefix_ids_length = prompt_prefix_ids.size(1)
    
    del prompt_prefix_ids
    del prompt_prefix_mask
    # Generate postfix completions
    # print("Generating postfix completions...")
    with torch.inference_mode():
        rest_tokens = args.max_completion_length - prompt_prefix_guidance_ids.size(1)
        rest_tokens = max(rest_tokens, 0)
        generation_config_postfix.max_new_tokens = rest_tokens
        if rest_tokens > 0:
            prompt_prefix_guidance_postfix_ids = model.generate(
                prompt_prefix_guidance_ids,
                attention_mask=prompt_prefix_guidance_mask,
                generation_config=generation_config_postfix
            )
        else:
            prompt_prefix_guidance_postfix_ids = prompt_prefix_guidance_ids
    
    # Extract completion ids 
    prompt_length = prompt_ids.size(1) # shape [1]

    completion_ids = prompt_prefix_guidance_postfix_ids[:, prompt_length:]
    
    # Do masking 
    is_eos = completion_ids == tokenizer.eos_token_id # [num_chains, completion_seq_len] bool
    eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device) # [num_chains, completion_seq_len] long
    eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)] # [num_chains] long
    sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
    completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
    
    # prompt_mask [0, ..., 0, 1, ..., 1] left padding
    # [1, 1, 1, 1, 0, 0, 0, 0],  # 第一个序列在第4个位置生成了EOS
    # [1, 1, 1, 1, 1, 1, 0, 0]   # 第二个序列在第6个位置生成了EOS
    attention_mask = torch.cat([prompt_mask.repeat(args.num_chains, 1), completion_mask], dim=1) # full mask
    prompt_prefix_guidance_mask = attention_mask[:, :prompt_prefix_guidance_ids.size(1)]
    average_distance = (attention_mask.sum() - prompt_prefix_guidance_mask.sum()) / attention_mask.size(0)

    # Decode completions
    completions_text = tokenizer.batch_decode(completion_ids, skip_special_tokens=True)
  
    logits_to_keep = prompt_prefix_guidance_ids.size(1) - prompt_prefix_ids_length
    
    ## 清理其他
    del completion_ids
    
    return prompt_prefix_guidance_ids, prompt_prefix_guidance_mask, logits_to_keep, completions_text, average_distance


def score_completions(
    completions_text: list[str],
    question: str,
    answer: str,
    eval_class: evaluator.RewardEvaluator,
    device: str,
    args: argparse.Namespace
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float], dict]:
    """
    Score model completions and compute advantages for training.
    
    Args:
        completions_text: List of generated completion strings
        question: Original input question/prompt
        answer: Ground truth answer
        eval_class: Evaluator class for computing rewards
        device: Device to place tensors on
        args: Training arguments
        
    Returns:
        rewards: Raw reward scores for each completion
        advantages: Computed advantages for policy gradient
        rewards_per_func: Rewards broken down by individual reward functions
        metrics: Dictionary of aggregated metrics
        log_data: Dictionary containing detailed generation and scoring data
    """
    # Build log data dictionary
    log_data = {
        'prompt': {
            'text': question,
            'answer': answer
        },
        'generations': []
    }

    # Format inputs as expected by evaluator
    mock_prompts = [[{'content': question}]] * len(completions_text)
    mock_completions = [[{'content': completion}] for completion in completions_text]
    answers = [answer] * len(completions_text)
    
    # Get rewards and metrics from evaluator
    rewards_per_func, metrics = eval_class.compute_rewards(
        prompts=mock_prompts,
        completions=mock_completions,
        answer=answers,
        device=device
    )
    rewards = rewards_per_func.sum(dim=1)

    # Store generation data
    for i, (completion, reward_scores) in enumerate(zip(completions_text, rewards_per_func)):
        generation_data = {
            'response': completion,
            'scores': {
                **eval_class.get_reward_breakdown(reward_scores),
                'total_reward': rewards[i].item()
            }
        }
        log_data['generations'].append(generation_data)

    # Compute advantages
    mean_grouped_rewards = rewards.view(-1, args.num_chains).mean(dim=1)
    std_grouped_rewards = rewards.view(-1, args.num_chains).std(dim=1)

    mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(args.num_chains, dim=0)
    std_grouped_rewards = std_grouped_rewards.repeat_interleave(args.num_chains, dim=0)

    advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
    metrics["reward_std"] = std_grouped_rewards.mean().item()

    # Store summary statistics
    log_data['summary_stats'] = {
        'mean_rewards_per_group': mean_grouped_rewards.tolist(),
        'std_rewards_per_group': std_grouped_rewards.tolist(),
        'advantages': advantages.tolist()
    }

    return rewards, advantages, rewards_per_func, metrics, log_data


def compute_loss_wo_kl(
    model: PreTrainedModel,
    prompt_completion_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    logits_to_keep: int,
    advantages: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """
    Compute the GRPO loss between current and base model.
    这里甚至连 old log_ps都懒得实现
    Args:
        model: The current model being trained
        prompt_completion_ids: Combined prompt and completion token IDs
        prompt_ids: Token IDs for just the prompt
        attention_mask: Attention mask for the full sequence
        logits_to_keep: logits_to_keep
        advantages: Advantage values for each sequence
        
    Returns:
        loss: The computed GRPO loss
        metrics: Dictionary containing additional metrics like KL divergence
    """

    # Only need the generated tokens' logits  logits_to_keep
    completion_mask = attention_mask[:,-logits_to_keep:]
    # Get training model logits
    # theta 的 per token log probabilities
    per_token_logps = utils.get_per_token_logps(model, prompt_completion_ids, attention_mask, logits_to_keep)
    

    # Compute loss with advantages
    per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
    loss = ((-per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()

    # Additional metrics
    metrics = {}
    response_length = completion_mask.sum(1).float().mean().item()
    metrics["response_length"] = response_length

    return loss, metrics


def grpo_loss(
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        question: str,
        answer: str,
        eval_class: evaluator.RewardEvaluator,
        device: str,
        round_num: int,
        training_log_dir: str, 
        args: argparse.Namespace
) -> tuple[torch.Tensor, dict[str, float], float]:
    """
    Compute GRPO loss between the current model and base model.
    
    Args:
        model: The current model being trained
        tokenizer: Tokenizer for the models
        question: Input question/prompt
        answer: Ground truth answer
        eval_class: Evaluator for computing rewards
        device: Device to run on ('cpu' or 'cuda')
        round_num: Current training round number
        training_log_dir: Directory to save training logs
        args: Training arguments
        
    Returns:
        loss: The computed GRPO loss
        metrics: Dictionary containing training metrics
        reward: The total reward for this batch
    """
    # Generate completions
    prompt_prefix_guidance_ids, prompt_prefix_guidance_mask, logits_to_keep, completions_text, average_distance = generate_segment_completions(
        model, tokenizer, question, device, args
    )

    # Score completions
    rewards, advantages, rewards_per_func, metrics, log_data = score_completions(
        completions_text, question, answer, eval_class, device, args
    )
    
    # reward normal
    # rewards = rewards * multi_factor_weight(d=average_distance , d_max=args.max_completion_length)
    
    # Write log data
    log_file = os.path.join(training_log_dir, f'{round_num}_generations.txt')
    utils.write_generation_log(log_data, log_file)

    # Compute loss
    loss, loss_metrics = compute_loss_wo_kl(
        model, prompt_prefix_guidance_ids,
        prompt_prefix_guidance_mask, logits_to_keep, advantages
    )

    # Combine metrics
    metrics.update(loss_metrics)

    return loss, metrics


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO training arguments")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct", help="Name/path of base model")
    parser.add_argument("--dataset_name", type=str, default="gsm8k", help="Dataset to use for training")
    parser.add_argument("--evaluator", type=str, default="gsm8k", help="Evaluator to use for scoring")

    # Output and logging
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save outputs")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--save_steps", type=int, default=500, help="Save model every N steps")
    parser.add_argument("--eval_iterations", type=int, default=40, help="Number of iterations for evaluation")

    # Optimization hyperparameters
    parser.add_argument("--learning_rate", type=float, default=5e-6, help="Learning rate")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="Adam beta1")
    parser.add_argument("--adam_beta2", type=float, default=0.99, help="Adam beta2") 
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    parser.add_argument("--max_grad_norm", type=float, default=0.1, help="Max gradient norm for clipping")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="Number of gradient accumulation steps")
    parser.add_argument("--warmup_percent", type=float, default=0.18, help="Percentage of total steps for warmup")
    parser.add_argument("--update_ref_model", action="store_true", help="Whether to update reference model")
    parser.add_argument("--update_ref_model_freq", type=int, default=200, help="How often to update reference model")
    parser.add_argument("--ref_model_mixup_alpha", type=float, default=0.1, help="Alpha parameter for reference model mixup")


    # Generation parameters
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--num_chains", type=int, default=16, help="Number of parallel generation chains")
    parser.add_argument("--max_prompt_length", type=int, default=256, help="Maximum prompt length")
    parser.add_argument("--max_completion_length", type=int, default=786, help="Maximum completion length")
    parser.add_argument("--max_guide_tokens", type=int, default=8, help="Maximum guide tokens")
    
    # Training parameters
    parser.add_argument("--num_train_iters", type=int, default=1000, help="Number of training iterations")
    parser.add_argument("--kl_weight_beta", type=float, default=0.04, help="KL penalty weight")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()
    return args

if __name__ == "__main__":

    # Get all args 
    args = parse_args() 
    
    # Seed everything 
    utils.seed_everything(args.seed)

    # Set device and enable bf16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
    torch.set_float32_matmul_precision('high') 

    # 1. 初始化wandb
    wandb.init(
        project="G-GRPO",    # 项目名称
        name="qwem-1.5b-math",         # 实验名称
        config={                        # 配置参数
            "learning_rate": args.learning_rate,
            "num_train_iters": args.num_train_iters,
            "num_chains": args.num_chains,
            "model_name": args.model_name,
            "max_prompt_length": args.max_prompt_length,
            "max_completion_length": args.max_completion_length,
            "temperature": args.temperature,
        }
    )
    
    ###############################
    ## Main Experiment settings ##
    ###############################

    ## Set which model to train 
    model, tokenizer = llms.get_llm_tokenizer(args.model_name, device)

    ## Set which data set 
    train_loader, test_loader = rldatasets.get_dataloaders(args.dataset_name)

    ## Set which evaluation criteria to use 
    eval_class = evaluator.get_evaluator(args.evaluator)

    ###############################


    # Setup logging 
    os.makedirs(args.output_dir, exist_ok=True)
    args_dict = vars(args)
    args_path = os.path.join(args.output_dir, 'args.json')
    with open(args_path, 'w') as f:
        json.dump(args_dict, f, indent=4)
    eval_log_dir = os.path.join(args.output_dir, 'eval_logs')
    os.makedirs(eval_log_dir, exist_ok=True)
    train_log_dir = os.path.join(args.output_dir, 'training_logs')
    os.makedirs(train_log_dir, exist_ok=True)


    # Setup optimizer for trainer agent with GRPO config settings
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.weight_decay,
        eps=1e-8
    )

    # Add linear warmup learning rate scheduler
    warmup_steps = int(args.warmup_percent * args.num_train_iters)
    def get_lr(step):
        if step < warmup_steps:
            return (step / warmup_steps)
        return 1.0
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=get_lr)


    # Begin training 
    accumulated_loss = 0
    optimizer.zero_grad()
    train_metrics_total = {}
    for round_num in tqdm(range(args.num_train_iters), desc="Training Progress"):
    
        # Evaluate on test set every so often 
        if (round_num + 1) % args.eval_iterations == 0:
            eval_metrics, eval_accuracy = eval_on_test_set(
                model=model,
                tokenizer=tokenizer, 
                test_loader=test_loader,
                eval_class=eval_class,
                device=device,
                args=args,
                round_num=round_num
            )
            
            # Save metrics to eval log dir
            metrics_path = os.path.join(eval_log_dir, f'metrics_{round_num}.json')
            with open(metrics_path, 'w') as f:
                json.dump({
                    'metrics': eval_metrics,
                    'accuracy': eval_accuracy
                }, f, indent=4)

            wandb.log({
                "eval/accuracy": eval_accuracy,
                "eval/epoch": round_num / args.num_train_iters,
                **eval_metrics
            })
                
        # Get next question
        question, answer = next(train_loader)

        # Do GRPO - generate chains, score, compute advantage, compute loss 
        total_loss, train_metrics = grpo_loss(model, tokenizer, question, answer, eval_class, device, round_num, train_log_dir, args)
        
        # Gradient accumulation
        total_loss = total_loss # / args.gradient_accumulation_steps
        total_loss.backward()
        accumulated_loss += total_loss.item()
        scheduler.step()

        # Step optimizer
        if (round_num + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()    
            torch.cuda.empty_cache()  # 清理不需要的缓存
            
        # Logs
        train_metrics["learning_rate"] = scheduler.get_last_lr()[0]
        train_metrics["loss"] = total_loss.item() * args.gradient_accumulation_steps
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf')).item()
        train_metrics["grad_norm"] = grad_norm
        train_metrics_total[round_num] = train_metrics
        with open(os.path.join(train_log_dir, "train_logs.json"), "w") as f:
            json.dump(train_metrics_total, f, indent=4)
    
        wandb.log({
            "train/loss": total_loss.item() * args.gradient_accumulation_steps,
            "train/lr": scheduler.get_last_lr()[0],
            "train/grad_norm": grad_norm,
            "train/step": round_num,
            "train/epoch": round_num / args.num_train_iters,
        })
# export HF_ENDPOINT=https://hf-mirror.com
# export CUDA_VISIBLE_DEVICES=2