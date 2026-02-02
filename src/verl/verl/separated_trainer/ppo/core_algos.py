# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

import numpy as np
import torch
from collections import defaultdict

import verl.utils.torch_functional as verl_F


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(config):
    if config.critic.kl_ctrl.type == 'fixed':
        kl_ctrl = FixedKLController(kl_coef=config.critic.kl_ctrl.kl_coef)
    elif config.critic.kl_ctrl.type == 'adaptive':
        assert config.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
        kl_ctrl = AdaptiveKLController(init_kl_coef=config.critic.kl_ctrl.kl_coef,
                                       target_kl=config.critic.kl_ctrl.target_kl,
                                       horizon=config.critic.kl_ctrl.horizon)
    else:
        raise ValueError('Unknown kl_ctrl type')

    return kl_ctrl

def compute_turn_level_return(turn_level_rewards: torch.Tensor, turn_mask: torch.Tensor,
                              gamma: torch.Tensor):
    """
    Compute turn-level return
    """
    with torch.no_grad():
        returns = torch.zeros_like(turn_level_rewards)
        running_return = 0

        for t in reversed(range(turn_level_rewards.shape[1])):
            running_return = turn_level_rewards[:, t] + gamma * running_return
            running_return = running_return * turn_mask[:, t]
            returns[:, t] = running_return

        return returns

def compute_gae_advantage_return(token_level_rewards: torch.Tensor, values: torch.Tensor, eos_mask: torch.Tensor,
                                 gamma: torch.Tensor, lam: torch.Tensor):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        values: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma: `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, eos_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for GRPO, operating only on Outcome reward 
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_rloo_outcome_advantage(token_level_rewards: torch.Tensor,
                                   eos_mask: torch.Tensor,
                                   index: torch.Tensor,
                                   epsilon: float = 1e-6):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num -
                                                        1) - id2mean[index[i]] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return scores, scores


def compute_reinforce_plus_plus_outcome_advantage(token_level_rewards: torch.Tensor, eos_mask: torch.Tensor,
                                                  gamma: torch.Tensor):
    """
    Compute advantage for REINFORCE++. 
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * eos_mask[:, t]

        advantages = verl_F.masked_whiten(returns, eos_mask)
        advantages = advantages * eos_mask

    return advantages, returns


def compute_remax_outcome_advantage(token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor,
                                    eos_mask: torch.Tensor):
    """
    Compute advantage for ReMax, operating only on Outcome reward 
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505

    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    
    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    with torch.no_grad():
        returns = (token_level_rewards * eos_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1).tile([1, response_length]) * eos_mask

    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def compute_policy_loss(
    old_log_prob, 
    log_prob, 
    advantages, 
    eos_mask, 
    step_id,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    log_ratio_clip_c=3.0,
    agg_mode: str = 'token',
    clip_mode: str = 'token',
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        cliprange: (float)
            The clip range used in PPO. See https://arxiv.org/abs/1707.06347

    Returns:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via PPO
        pg_clipfrac: (float)
            a float number indicating the fraction of policy gradient loss being clipped

    """
    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    assert log_ratio_clip_c >= 0.0, (
        "The lower bound of the log_ratio_clip_c for dual-clip PPO should be greater than 0.0,"
        + f" but get the value: {log_ratio_clip_c}."
    )

    assert agg_mode in ['token', 'turn', 'trajectory']
    assert clip_mode in ['token', 'turn']

    negative_approx_kl = log_prob - old_log_prob
    if clip_mode == 'token':
        if log_ratio_clip_c is not None:
            log_ratio_clipfrac = verl_F.masked_mean(torch.gt(negative_approx_kl, log_ratio_clip_c).float(), eos_mask)
            negative_approx_kl = torch.clamp(negative_approx_kl, min=None, max=log_ratio_clip_c)
        else:
            log_ratio_clipfrac = torch.tensor(0.0)
        
        ratio = torch.exp(negative_approx_kl)
        ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

        pg_losses1 = -advantages * ratio
        if cliprange_low is None:
            cliprange_low = cliprange
        if cliprange_high is None:
            cliprange_high = cliprange
        pg_losses2 = -advantages * torch.clamp(
            ratio, 1 - cliprange_low, 1 + cliprange_high
        )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
        clip_pg_losses1 = torch.maximum(
            pg_losses1, pg_losses2
        )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
        pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), eos_mask)

        # an extra clip for negative advantages
        pg_losses3 = -advantages * clip_ratio_c
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
        pg_clipfrac_lower = verl_F.masked_mean(
            torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), eos_mask
        )

        pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
        
        if agg_mode == 'token':
            pg_loss = verl_F.masked_mean(pg_losses, eos_mask)
        elif agg_mode == 'trajectory':
            seq_losses = torch.sum(pg_losses * eos_mask, dim=-1) / torch.sum(eos_mask, dim=-1)  # token-mean
            pg_loss = torch.mean(seq_losses)  # seq-mean
        elif agg_mode == 'turn':
            with torch.no_grad():
                max_turns = step_id.max().item() + 1
                turn_masks = torch.arange(max_turns, device=step_id.device).view(1, 1, -1) == step_id.unsqueeze(-1)  # (bsz, seq_len, max_turns)

                turn_counts = turn_masks.sum(dim=1)  # (bsz, max_turns)
                turn_weights = (1 / turn_counts).masked_fill(turn_counts == 0, 0).unsqueeze(1)  # (bsz, 1, max_turns)

                # Broadcast weights and compute the total weights.
                weights = (turn_masks * turn_weights).sum(dim=-1)  # (bsz, seq_len)
            # assert (weights != 0).sum() == (eos_mask != 0).sum()

            pg_losses_weighted = pg_losses * weights
            pg_loss = pg_losses_weighted.sum() / turn_counts.gt(0).sum()
        else:
            raise ValueError(f"Unknown agg_mode: {agg_mode}")
        return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, log_ratio_clipfrac
        
    elif clip_mode == 'turn':
        max_turns = step_id.max().item() + 1
        turn_masks = torch.arange(max_turns, device=step_id.device).view(1, 1, -1) == step_id.unsqueeze(-1)  # (bsz, seq_len, max_turns)
        turn_masks = turn_masks.detach()
        turn_level_eos_mask = (turn_masks.sum(1) != 0).detach()

        turn_counts = turn_masks.sum(dim=1)  # (bsz, max_turns)
        # for averaging
        turn_weights = (1 / turn_counts).masked_fill(turn_counts == 0, 0).detach()  # (bsz, max_turns)
        
        # Sum negative_approx_kl for each turn.
        # We multiply by eos_mask to ensure only valid token KLs are summed for each turn.
        # Then, we use turn_masks to sum these KLs for each turn across the sequence length.
        # The result, turn_level_negative_approx_kl, will have shape (bsz, max_turns).
        assert torch.all((step_id != -100) == (eos_mask!=0))
        negative_approx_kl_each_turn = (negative_approx_kl.unsqueeze(-1) * turn_masks) # (bsz, seq_len, max_turns)
        turn_level_avg_negative_approx_kl = negative_approx_kl_each_turn.sum(dim=1) * turn_weights # (bsz, max_turns)
        turn_level_avg_advantages = (advantages.unsqueeze(-1) * turn_masks).sum(dim=1) * turn_weights  # (bsz, max_turns)

        if log_ratio_clip_c is not None:
            log_ratio_clipfrac = verl_F.masked_mean(torch.gt(turn_level_avg_negative_approx_kl, log_ratio_clip_c).float(), turn_level_eos_mask)
            turn_level_avg_negative_approx_kl = torch.clamp(turn_level_avg_negative_approx_kl, min=None, max=log_ratio_clip_c)
        else:
            log_ratio_clipfrac = torch.tensor(0.0)
        turn_level_ratio = torch.exp(turn_level_avg_negative_approx_kl)

        ppo_kl = verl_F.masked_mean(-negative_approx_kl, eos_mask)

        pg_losses1 = -turn_level_avg_advantages * turn_level_ratio
        if cliprange_low is None:
            cliprange_low = cliprange
        if cliprange_high is None:
            cliprange_high = cliprange
        pg_losses2 = -turn_level_avg_advantages * torch.clamp(
            turn_level_ratio, 1 - cliprange_low, 1 + cliprange_high
        )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
        clip_pg_losses1 = torch.maximum(
            pg_losses1, pg_losses2
        )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
        pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), turn_level_eos_mask)

        # an extra clip for negative advantages
        pg_losses3 = -turn_level_avg_advantages * clip_ratio_c
        clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
        pg_clipfrac_lower = verl_F.masked_mean(
            torch.gt(clip_pg_losses1, pg_losses3) * (turn_level_avg_advantages < 0).float(), turn_level_eos_mask
        )
        pg_losses = torch.where(turn_level_avg_advantages < 0, clip_pg_losses2, clip_pg_losses1)
        if agg_mode == 'token':
            raise ValueError("token agg_mode is not supported for turn clip_mode")
        elif agg_mode == 'trajectory':
            seq_losses = torch.sum(pg_losses * turn_level_eos_mask, dim=-1) / torch.sum(
                turn_level_eos_mask, dim=-1).clamp(min=1)  # token-mean
            pg_loss = torch.mean(seq_losses)  # seq-mean
        elif agg_mode == 'turn':
            pg_loss = verl_F.masked_mean(pg_losses, turn_level_eos_mask)
        else:
            raise ValueError(f"Unknown agg_mode: {agg_mode}")
        return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, log_ratio_clipfrac

    else:
        raise ValueError(f"Unknown clip mode: {clip_mode}")
        


    


def compute_entropy_loss(logits, eos_mask):
    """Compute Categorical entropy loss

    Args:
        logits: `(torch.Tensor)`
            shape: (bs, response_length, vocab_size)
        eos_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = verl_F.masked_mean(entropy, mask=eos_mask)
    return entropy_loss


def compute_value_loss(vpreds, returns, values, eos_mask, cliprange_value):
    """Compute the value loss. Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (`torch.FloatTensor`):
            Predicted values of the value head, shape (`batch_size`, `response_length`)
        values (`torch.FloatTensor`):
            Old values of value head, shape (`batch_size`, `response_length`)
        returns: (`torch.FloatTensor`):
            Ground truth returns, shape (`batch_size`, `response_length`)

    Returns:
        vf_loss: a scalar (`torch.FloatTensor`):
            value function loss
        vf_clipfrac: a float
            The ratio of vf being clipped

    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns)**2
    vf_losses2 = (vpredclipped - returns)**2
    vf_loss = 0.5 * verl_F.masked_mean(torch.max(vf_losses1, vf_losses2), eos_mask)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), eos_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty == "kl":
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty == "mse":
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty == 'low_var_kl':
        kl = ref_logprob - logprob
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError
