import os
import math
import random
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoTokenizer, AutoModelForCausalLM

import wandb
from typing import List
import argparse
import datetime as dt
import matplotlib.pyplot as plt

from grpo import (
    rollout, 
    compute_grpo_loss, 
    normalize_rewards,
    generate_game,
    Turn, 
)

from utils import save_text_to_pdf, plot_trend


def unwrap(model):
    return model.module if hasattr(model, "module") else model


def setup_distributed():
    dist.init_process_group(backend="nccl")
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    print(f"world_size: {world_size}, rank: {rank}, local_rank: {local_rank}")
    torch.cuda.set_device(local_rank)

    seed = 42 + rank
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return world_size, rank, local_rank


def load_tokenizer_and_models(model_name, local_rank):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
        tok.pad_token = tok.eos_token
    # tok.padding_side = "left"

    policy = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).to(local_rank)
    ref = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16
    ).to(local_rank)
    ref.eval()
    for p in ref.parameters(): p.requires_grad = False

    policy = DDP(policy, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    return tok, policy, ref



def build_optimizer(policy, lr=1e-5, weight_decay=0.1):
    p = unwrap(policy)
    decay, nodecay = [], []
    for n, t in p.named_parameters():
        if not t.requires_grad: continue
        (decay if t.dim() >= 2 else nodecay).append(t)
    groups = [{"params": decay, "weight_decay": weight_decay},
              {"params": nodecay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr)



def cosine_lr(it, max_steps, warmup_frac=0.1, max_lr=1e-5, min_lr=1e-6):
    warm = max(1, int(warmup_frac * max_steps))
    if it < warm: return max_lr * (it + 1) / warm
    if it >= max_steps: return min_lr
    r = (it - warm) / (max_steps - warm)
    return min_lr + 0.5 * (1 + math.cos(math.pi * r)) * (max_lr - min_lr)



def train(
    model_name: str,
    num_groups_per_gpu: int,        
    num_per_group: int,        # K (episodes) per start
    minibatch_size: int,       # turns per optimizer step
    num_epochs: int,           # OUTER loop
    learning_rate: float,
    max_steps: int,     
    out_dir: str,       
    kl_coef: float = 0.005,
    clip_eps: float = 0.2,
    ckpt_every_groups: int = 100,
    wandb_project: str = "grpo-2048",
    wandb_run_name: str = "llm-2048-grpo",
    eval_interval_step: int = 5,
    eval_episodes: int = 5,
):
    
    world_size, rank, local_rank = setup_distributed()
    tokenizer, policy_model, ref_model = load_tokenizer_and_models(model_name, local_rank)
    optimizer = build_optimizer(policy_model, lr=1e-5, weight_decay=0.1)

    # wandb: init only on rank 0
    if rank == 0:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        wandb_run_name = f"{wandb_run_name}_{timestamp}"
        wandb.init(
            project=wandb_project,
            name=wandb_run_name,
            config=dict(
                model_name=model_name,
                num_groups_per_gpu=num_groups_per_gpu,
                num_per_group=num_per_group,
                learning_rate=learning_rate,
                minibatch_size=minibatch_size,
                num_epochs=num_epochs,
                kl_coef=kl_coef,
                clip_eps=clip_eps,
                max_steps=max_steps,
            ),
        )

    global_step = 0
    eval_rewards = []
    rollout_rewards: List[List[float]] = []  # per epoch

    for epoch in range(num_epochs):
        if rank == 0:
            print(f"=== Epoch {epoch+1}/{num_epochs} ===")

        
        # collect rollouts
        all_turns: List[Turn] = []
        rewards = []
        for group in range(num_groups_per_gpu):
            game = generate_game()
            group_turns = []
            for k in range(num_per_group):
                turns, _, _ = rollout(
                    policy_model=policy_model,
                    tokenizer=tokenizer,
                    device=local_rank,
                    game_override=game,
                )
                if rank == 0:
                    print(f"  [Group {group+1}/{num_groups_per_gpu} - Ep {k+1}/{num_per_group}] Collected {len(turns)} turns with reward: {turns[0].terminal_reward:.4f}")
                rewards.append(turns[0].terminal_reward)
                group_turns.extend(turns)
            group_turns = normalize_rewards(group_turns)
            all_turns.extend(group_turns)

        rollout_rewards.append(rewards)

        if rank == 0:
            print(f"  Collected total {len(all_turns)} turns")

            ### save rewards
            wandb.log(
                {
                    "train/rollout_min_reward": float(np.min(rewards)),
                    "train/rollout_max_reward": float(np.max(rewards)),
                    "train/rollout_mean_reward": float(np.mean(rewards)),
                    "train/rollout_std_reward": float(np.std(rewards)),
                },
            )

        # if epoch == num_epochs - 1:
        #     break   # skip update on last epoch, just collect rollouts for stats

        ### sort turns by length of ids to minimize padding
        all_turns.sort(key=lambda t: t.ids.shape[0])

       # move turns to CPU to save GPU memory
        for t in all_turns:
            t.ids    = t.ids.cpu()
            t.labels = t.labels.cpu()
            t.attn   = t.attn.cpu()    # optional if you recompute attn later

        torch.cuda.empty_cache()

        if rank == 0:
            epoch_losses = []

        for i in range(0, len(all_turns), minibatch_size):
            ###
            if global_step % eval_interval_step == 0 and rank == 0:
                eval_r = []
                for eval_i in range(eval_episodes):
                    _, r, _ = rollout(
                        policy_model=policy_model,
                        tokenizer=tokenizer,
                        device=local_rank,
                    )
                    eval_r.append(r)
                    print(f"    [Eval {eval_i+1}/{eval_episodes}] reward: {r:.4f}")
                print(f"  [Eval] reward: {np.mean(eval_r):.4f} ± {np.std(eval_r):.4f} (n={len(eval_r)})")
                eval_rewards.append(np.array(eval_r))



            j = min(i + minibatch_size, len(all_turns))
            mb_turns = all_turns[i:j]

            # lr
            lr = cosine_lr(global_step, max_steps)
            for pg in optimizer.param_groups: 
                pg["lr"] = lr

            optimizer.zero_grad(set_to_none=True)

            loss = compute_grpo_loss(
                policy_model=policy_model,
                ref_model=ref_model,
                tokenizer=tokenizer,
                device=local_rank,
                turns=mb_turns,
                kl_coef=kl_coef,
                clip_eps=clip_eps,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=0.1)
            optimizer.step()

            ### move mb_turns to CPU to save GPU memory
            for t in mb_turns:
                t.ids    = t.ids.cpu()
                t.labels = t.labels.cpu()
                t.attn   = t.attn.cpu()    # optional if you recompute attn later
            torch.cuda.empty_cache()


            # ---------- per-step logging ----------
            # reduce scalar loss across ranks for consistent logging
            with torch.no_grad():
                step_loss = loss.detach()
                dist.all_reduce(step_loss, op=dist.ReduceOp.AVG)
            if rank == 0:
                wandb.log(
                    {
                        "train/step_loss": step_loss.item(),
                        "train/lr": lr,
                        "train/global_step": global_step,
                        "train/epoch": epoch,
                    },
                    step=global_step,
                )
                print(f"    [MB {i//minibatch_size+1}/{math.ceil(len(all_turns)/minibatch_size)}] step_loss={step_loss.item():.4f} lr={lr:.6e}")
                epoch_losses.append(step_loss.item())
            global_step += 1


        

        if rank == 0:
            epoch_loss = np.mean(epoch_losses)
            wandb.log(
                {
                    "train/epoch_loss": epoch_loss,
                },
                # step=global_step-1,
            )
            print(f"[Epoch {epoch+1}] avg_loss={epoch_loss:.4f}")
        

    dist.barrier()
    if rank == 0:
        ## save 
        out_dir = os.path.join(out_dir, f"run_{wandb_run_name}")
        os.makedirs(out_dir, exist_ok=True)

        ### save eval rewards as plot
        if len(eval_rewards) > 0:
            ## eval_rewards: List[np.ndarray] -> (N, eval_episodes)
            eval_rewards = np.array(eval_rewards)
            plot_trend(
                data=eval_rewards,
                path=os.path.join(out_dir, f"eval_rewards.pdf"),
                x_interval=eval_interval_step,
            )
            plot_trend(
                data=eval_rewards,
                path=os.path.join(out_dir, f"eval_rewards_mean.pdf"),
                x_interval=eval_interval_step,
                plot_std=False,
            )
            np.save(os.path.join(out_dir, f"eval_rewards.npy"), eval_rewards)
           

        ## final evaluation
        print("=== Final Evaluation ===")
        rewards = []
        record_historys = []
        for i in range(5):
            _, reward, record_history = rollout(
                policy_model=policy_model,
                tokenizer=tokenizer,
                device=local_rank,
                )
            rewards.append(reward)
            record_historys.append(record_history)
            print(f"  [Eval {i+1}/5] reward: {reward:.4f}")

        ### save best game log to pdf
        idx = int(np.argmax(rewards))
        history = record_historys[idx]
        log_text = "\n".join(history)
        save_text_to_pdf(log_text, os.path.join(out_dir, f"final_eval_game_log.pdf"))

        # final checkpoint
        final_dir = os.path.join(out_dir, f"step{global_step}")
        os.makedirs(final_dir, exist_ok=True)
        unwrap(policy_model).save_pretrained(final_dir)
        tokenizer.save_pretrained(final_dir)
        wandb.save(os.path.join(final_dir, "*"))
        wandb.finish()

    dist.destroy_process_group()



def argparse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--num_groups_per_gpu", type=int, default=4)
    parser.add_argument("--num_per_group", type=int, default=8)
    parser.add_argument("--minibatch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_steps", type=int, default=10000)
    parser.add_argument("--out_dir", type=str, default="./checkpoints")
    parser.add_argument("--kl_coef", type=float, default=0.005)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ckpt_every_groups", type=int, default=100)
    parser.add_argument("--wandb_project", type=str, default="grpo-2048")
    parser.add_argument("--wandb_run_name", type=str, default="llm-2048-grpo")
    args = parser.parse_args()
    return args 


if __name__ == "__main__":
    args = argparse_args()
    train(
        model_name=args.model_name,
        num_groups_per_gpu=args.num_groups_per_gpu,
        num_per_group=args.num_per_group,
        minibatch_size=args.minibatch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        max_steps=args.max_steps,
        out_dir=args.out_dir,
        kl_coef=args.kl_coef,
        clip_eps=args.clip_eps,
        ckpt_every_groups=args.ckpt_every_groups,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )
