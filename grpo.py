from dataclasses import dataclass
from typing import Optional, List, Tuple
import torch
import torch.nn.functional as F
import copy, math, re
import numpy as np

from game_utils_2048 import (
    generate_game, 
    render_board,
    apply_agent_move,
    check_game_finished,
    total_board_value,
    max_cell_value,
    sys_prompt,
    TwentyFortyEightGame,
    WINNING_VALUE,
)


@dataclass
class Turn:
    # full sequence for this turn = [full chat history up to board_t] + [generated move tokens]
    ids: torch.Tensor         # LongTensor [L_t]
    attn: torch.Tensor        # LongTensor [L_t]  (ids != pad)
    labels: torch.Tensor      # LongTensor [L_t]  (-100 on history tokens, real ids on action span)
    text: str                 # decoded assistant text for this turn (e.g., "<move>left</move>")
    terminal_reward: Optional[float] = None  # set on final turn (or leave None for others)

def unwrap(model):
    return model.module if hasattr(model, "module") else model


@torch.no_grad()
def rollout(
    policy_model,
    tokenizer,
    device,
    game_override = None,
    omit_middle = True,
    max_new_tokens = 128,
    T_max = 200,
    top_p = 0.9,
    temperature = 1.0,
    record_interval = 10,
) -> Tuple[List[Turn], float, List[str]]:
    
    policy_model.eval()
    game = copy.deepcopy(game_override) if game_override is not None else generate_game()
    messages = [{"role": "system", "content": sys_prompt()}]
    turns: List[Turn] = []

    record_history = []

    reward = 0.0

    for i in range(T_max):
        rendered_board = render_board(game)
        messages.append({"role": "user", "content": rendered_board})

        if i % record_interval == 0:
            record_history.append(f"\nmove {i+1}")
            record_history.append(f"board:\n{rendered_board}")

        ## tokenize the full message history
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = tokenizer([prompt], return_tensors="pt", padding=True, truncation=True).to(device)
        prompt_len = int(enc["attention_mask"].sum(1).item())


        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out_ids = unwrap(policy_model).generate(
                input_ids=enc["input_ids"],
                attention_mask=enc["attention_mask"],
                max_new_tokens=max_new_tokens,
                do_sample=True, top_p=top_p, temperature=temperature,
                eos_token_id=tokenizer.eos_token_id,
            )  # [1, prompt_len + action_len]


        seq = out_ids[0]                          # LongTensor [L_t]
        labels = seq.clone()
        labels[:prompt_len] = -100                # only current action tokens are supervised
        attn = (seq != tokenizer.pad_token_id).long()
        action_text = tokenizer.decode(seq[prompt_len:], skip_special_tokens=True)

        turns.append(Turn(ids=seq, attn=attn, labels=labels, text=action_text))


        messages.append({"role": "assistant", "content": action_text})

      
        # add assistant reply to history and step env
        if len(messages) > 20:
            if omit_middle:
                messages = messages[:10] + [{"role": "system", "content": "...omitted..."}] + messages[-10:]

        try:
            apply_agent_move(game, action_text)
        except Exception as e:
            reward = -1.0
            break

        if i % record_interval == 0:
            record_history.append(f"agent move: {action_text}")
            record_history.append(f"updated board:\n{render_board(game)}")



        if check_game_finished(game):
        
            max_value = max_cell_value(game)
            board_value = total_board_value(game)


            ## record final board state
            record_history.append("\n" + "="*20)
            record_history.append(f"game finished in {i+1} moves")
            record_history.append(f"final board:\n{render_board(game)}")
            record_history.append(f"max value: {max_value}")
            record_history.append(f"board value: {board_value}")


            # try to get as close to the winning value as possible
            # otherwise, try to maximize number of high cells on board
            # but above all else: WIN THE GAME!
            if max_value < WINNING_VALUE:
                # scale max value logarithmically between 0 for 2 and 1 for WINNING_VALUE
                max_value_reward = (math.log(max_value, 2) - 1) / (
                    math.log(WINNING_VALUE, 2) - 1
                )
                # scale board value logarithmically between 0 for 2 * 16 and 1 for WINNING_VALUE * 16
                board_value_reward = (math.log(board_value, 2) - 1) / (
                    math.log(WINNING_VALUE * 16, 2) - 1
                )
                # combine the two rewards, with max value having a higher weight
                reward = max_value_reward + (board_value_reward * 0.2)

                record_history.append("game lost! (x_x)")
            else:
                # double reward if the agent wins
                reward = 2.0

                record_history.append("game won! \\o/")

            break

           


    if turns:
        for turn in turns:
            turn.terminal_reward = reward

    return turns, reward, record_history



def normalize_rewards(turns: List[Turn]) -> List[Turn]:
    
    rewards = [turn.terminal_reward for turn in turns if turn.terminal_reward is not None]
    mean_reward = np.mean(rewards) if rewards else 0.0
    std_reward = np.std(rewards) if rewards else 1.0
    std_reward = max(std_reward, 1e-6)  # prevent division by zero

    for turn in turns:
        if turn.terminal_reward is not None:
            turn.terminal_reward = (turn.terminal_reward - mean_reward) / std_reward

    return turns




def pad_collate_turns(
        turns: List[Turn], 
        pad_id: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
     # shapes per turn: [L_t] → stack to [T, L]
    # T = len(turns)
    L = max(t.ids.shape[0] for t in turns)
    
    def _pad(x, pad):
        out = x.new_full((L,), pad)
        out[:x.shape[0]] = x
        return out
    
    ids = torch.stack([_pad(t.ids, pad_id) for t in turns], dim=0)
    labels = torch.stack([_pad(t.labels, -100) for t in turns], dim=0)
    # recompute attn from ids to be safe
    attn = (ids != pad_id).long()
    return ids, attn, labels  # [T, L] each


def compute_token_logprobs(model, ids, labels, attn):

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(ids, attn, use_cache=False)
    logits = out.logits[:, :-1, :].contiguous()  # [B, L-1, V]
    labs = labels[:, 1:].contiguous()          # [B, L-1]
    lp = -F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labs.reshape(-1),
        reduction="none", ignore_index=-100
    ).view(logits.size(0), -1)                   # [B, L-1]
    return lp


def compute_token_logprobs_sparse(model, ids, labels, attn):

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model(ids, attn, use_cache=False)
    logits = out.logits[:, :-1, :].contiguous()  # [B, L-1, V]
    labs = labels[:, 1:].contiguous()          # [B, L-1]
    mask = (labs != -100)
    if not mask.any():
        return logits.new_zeros((0,))
    return -F.cross_entropy(logits[mask], labs[mask], reduction="none")  # [N_act]


def tail_window(ids, labels, attn, max_len: int):
    # ids, labels, attn: [B, L]
    L = ids.shape[1]
    if L <= max_len:
        return ids, labels, attn
    ids    = ids[:, -max_len:]
    labels = labels[:, -max_len:]
    attn   = attn[:, -max_len:]
    return ids, labels, attn


def compute_grpo_loss(
        policy_model, 
        ref_model, 
        tokenizer, 
        device,
        turns: List[Turn],
        kl_coef=0.005, 
        clip_eps=0.2,
        use_sparse_logp=True,
        max_seq_tokens=None,
):
    max_seq_tokens = 2048
    ids, attn, labels = pad_collate_turns(turns, tokenizer.pad_token_id)  # [T, L] each
    ids, attn, labels = ids.to(device), attn.to(device), labels.to(device)

    if max_seq_tokens is not None:
        ids, labels, attn = tail_window(ids, labels, attn, max_seq_tokens)

    # OLD/REF logprobs
    policy_model.eval()
    if use_sparse_logp:
         # NEW: sparse logprobs only on action tokens
        logp_old = compute_token_logprobs_sparse(policy_model, ids, labels, attn)   # [N_act]
        logp_ref = compute_token_logprobs_sparse(ref_model, ids, labels, attn)   # [N_act]

    else:
        logp_old = compute_token_logprobs(policy_model, ids, labels, attn)   # [T, L-1]
        logp_ref = compute_token_logprobs(ref_model, ids, labels, attn)   # [T, L-1]

   

    policy_model.train()

 
  

    if use_sparse_logp:
        # NEW (with grad)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = policy_model(ids, attn, use_cache=False)
        logits_new = out.logits[:, :-1, :].contiguous()
        labs = labels[:, 1:].contiguous()
        mask = (labs != -100)
        if not mask.any():
            logp_new = logits_new.new_zeros(())  # no supervised tokens

        logp_new = -F.cross_entropy(logits_new[mask], labs[mask], reduction="none")  # [N_act]

    else:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = policy_model(ids, attn, use_cache=False)
        logits_new = out.logits[:, :-1, :].contiguous()
        labs = labels[:, 1:].contiguous()
        logp_new = -F.cross_entropy(
            logits_new.reshape(-1, logits_new.size(-1)),
            labs.reshape(-1),
            reduction="none", ignore_index=-100
        ).view(logits_new.size(0), -1)                            # [T, L-1]



    # compute GRPO loss

    # build flat advantages per token
    if use_sparse_logp:
        B, Lm1 = labs.shape
        turn_idx = torch.arange(B, device=ids.device).unsqueeze(1).expand(B, Lm1)[mask]  # [N_act]
        adv_turn = torch.zeros(B, device=ids.device)
        for i, turn in enumerate(turns):
            if turn.terminal_reward is not None:
                adv_turn[i] = turn.terminal_reward
        adv = adv_turn[turn_idx].to(device)  # [N_act]

    else:
        ## advantages
        adv = torch.zeros_like(logp_old)
        for i, turn in enumerate(turns):
            if turn.terminal_reward is not None:
                adv[i] = turn.terminal_reward
        
        adv = adv.to(device)

    if use_sparse_logp:
        ratio      = torch.exp(logp_new - logp_old)
        ratio_clip = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        ppo_term   = torch.min(ratio * adv, ratio_clip * adv)
        dlog       = logp_ref - logp_new
        kl_term    = torch.exp(dlog) - dlog - 1.0
        grpo_loss  = - (ppo_term - kl_coef * kl_term).mean()

    else:
        valid_mask = (labels[:, 1:] != -100).float()                  # [T, L-1]

        ratio = torch.exp(logp_new - logp_old)
        ratio_clip = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
        ppo_term = torch.min(ratio * adv, ratio_clip * adv)

        dlog = logp_ref - logp_new
        kl_term = torch.exp(dlog) - dlog - 1.0

        num = ((ppo_term - kl_coef * kl_term) * valid_mask).sum(dim=1)    # [T]
        den = valid_mask.sum(dim=1).clamp_min(1.0)                         # [T]
        turn_losses = - (num / den)                                        # [T]
        grpo_loss = turn_losses.mean()
        # den = valid_mask.sum(dim=1)                                 # [T]
        # safe = den > 0
        # turn_loss = torch.zeros_like(den, dtype=torch.float32)
        # turn_loss[safe] = - (num[safe] / den[safe])
        # # average only over safe turns (and avoid div-by-zero)
        # grpo_loss = turn_loss[safe].mean() if safe.any() else torch.tensor(0.0, device=den.device)

    return grpo_loss
    



def grpo_batch_step(
        policy_model,
        ref_model,
        optimizer,
        device,
        tokenizer,
        num_per_group: int,
        num_groups: int,
        minibatch_size: int,
        kl_coef=0.005,
        clip_eps=0.2,
        logger=None,
):
    # collect rollouts
    all_turns: List[Turn] = []

    for _ in range(num_groups):
        game = generate_game()
        group_turns = []
        for _ in range(num_per_group):
            turns = rollout(
                policy_model, tokenizer, device,
                game_override=game,
            )
            group_turns.extend(turns)
        group_turns = normalize_rewards(group_turns)
        all_turns.extend(group_turns)

    
    ### sort turns by length of ids to minimize padding
    all_turns.sort(key=lambda t: t.ids.shape[0])

    losses = []

    for i in range(0, len(all_turns), minibatch_size):
        j = min(i + minibatch_size, len(all_turns))
        mb_turns = all_turns[i:j]

        optimizer.zero_grad(set_to_none=True)

        loss = compute_grpo_loss(
            policy_model, ref_model, tokenizer, device, mb_turns,
            kl_coef=kl_coef, clip_eps=clip_eps,
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_norm=0.1)
        optimizer.step()


        



    













if __name__ == "__main__":
    pass