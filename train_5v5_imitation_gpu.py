"""
Train 5v5 Imitation GPU: Learn from Legacy, then beat it (CUDA accelerated).

GPU Optimizations:
- Batched imitation training on GPU
- GPU-accelerated PPO fine-tuning
- Batched inference for episodes

Usage:
    python train_5v5_imitation_gpu.py --imitation_episodes 500 --finetune_iters 100
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from typing import List, Tuple, Dict
import time
import numpy as np

from unified_brain_v2 import (
    UnifiedBrainV2, get_state_vector_5v5, get_valid_spell_mask,
    ROLE_GUARDIAN, ROLE_STRIKER, ROLE_DISRUPTOR, STATE_DIM, NUM_SPELLS
)
from legacy_brain_v2 import LegacyBrainV2, get_legacy_state_5v5
from duel_arena_5v5 import create_team
from duel_engine import SPELL_LIST, SPELL_BOOK


# ==============================================================================
# DEVICE SETUP
# ==============================================================================

def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ==============================================================================
# PHASE 1: GPU-ACCELERATED IMITATION
# ==============================================================================

def collect_demonstrations_gpu(
    legacy_team: List[LegacyBrainV2],
    num_episodes: int = 100,
    device=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collect demonstrations and return as GPU tensors.
    """
    states_list = []
    actions_list = []
    
    for ep in range(num_episodes):
        team_a = create_team("Legacy", is_team_a=True)
        team_b = create_team("Opponent", is_team_a=False)
        
        opponent_brains = [LegacyBrainV2() for _ in range(5)]
        
        for turn in range(15):
            alive_a = [w for w in team_a if w.hp > 0]
            alive_b = [w for w in team_b if w.hp > 0]
            
            if len(alive_a) == 0 or len(alive_b) == 0:
                break
            
            for i, wiz in enumerate(team_a):
                if wiz.hp <= 0:
                    continue
                
                state = get_state_vector_5v5(wiz, team_a, team_b)
                legacy_state = get_legacy_state_5v5(wiz, team_a, team_b)
                mask = get_valid_spell_mask(wiz)
                action = legacy_team[i % len(legacy_team)].get_action(legacy_state, mask)
                
                states_list.append(state)
                actions_list.append(action)
                
                spell = SPELL_BOOK[SPELL_LIST[action]]
                hp_dmg = spell.get("hp_dmg", 0)
                target = next((e for e in team_b if e.hp > 0), None)
                if target:
                    target.hp -= hp_dmg
            
            for i, wiz in enumerate(team_b):
                if wiz.hp <= 0:
                    continue
                state = get_legacy_state_5v5(wiz, team_b, team_a)
                mask = get_valid_spell_mask(wiz)
                action = opponent_brains[i].get_action(state, mask)
                spell = SPELL_BOOK[SPELL_LIST[action]]
                hp_dmg = spell.get("hp_dmg", 0)
                target = next((e for e in team_a if e.hp > 0), None)
                if target:
                    target.hp -= hp_dmg
    
    states = torch.FloatTensor(np.array(states_list))
    actions = torch.LongTensor(actions_list)
    
    if device:
        states = states.to(device)
        actions = actions.to(device)
    
    return states, actions


def train_imitation_gpu(
    student: UnifiedBrainV2,
    states: torch.Tensor,
    actions: torch.Tensor,
    device,
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 256,  # Larger batch for GPU
) -> float:
    """GPU-accelerated imitation training."""
    
    optimizer = optim.Adam(student.parameters(), lr=lr)
    dataset_size = len(states)
    
    for epoch in range(epochs):
        perm = torch.randperm(dataset_size, device=device)
        total_loss = 0.0
        n_batches = 0
        
        for i in range(0, dataset_size, batch_size):
            idx = perm[i:i+batch_size]
            batch_states = states[idx]
            batch_actions = actions[idx]
            
            # Batched forward on GPU
            logits, _ = student.forward(batch_states, ROLE_STRIKER, 0.5)
            
            loss = F.cross_entropy(logits, batch_actions)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        avg_loss = total_loss / max(1, n_batches)
        print(f"  Epoch {epoch+1:2d} | Loss: {avg_loss:.4f}")
    
    return avg_loss


# ==============================================================================
# PHASE 2: GPU-ACCELERATED PPO
# ==============================================================================

class GPUBuffer:
    """GPU-optimized buffer."""
    
    def __init__(self, device):
        self.device = device
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.masks = []  # Valid action masks
    
    def add(self, state, action, log_prob, reward, value, done, mask=None):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        if mask is not None:
            self.masks.append(mask)
    
    def clear(self):
        self.__init__(self.device)
    
    def __len__(self):
        return len(self.states)
    
    def to_tensors(self):
        return (
            torch.FloatTensor(np.array(self.states)).to(self.device),
            torch.LongTensor(self.actions).to(self.device),
            torch.FloatTensor(self.log_probs).to(self.device),
            torch.FloatTensor(self.rewards).to(self.device),
            torch.FloatTensor(self.values).to(self.device),
            torch.FloatTensor(self.dones).to(self.device),
        )


def run_episode_gpu(
    student: UnifiedBrainV2,
    legacy_team: List[LegacyBrainV2],
    buffer: GPUBuffer,
    device,
    temperature: float = 0.5,
) -> Dict:
    """GPU-accelerated episode."""
    
    team_student = create_team("Student", is_team_a=True)
    team_legacy = create_team("Legacy", is_team_a=False)
    
    total_reward = 0.0
    
    for turn in range(15):
        alive_s = [w for w in team_student if w.hp > 0]
        alive_l = [w for w in team_legacy if w.hp > 0]
        
        if len(alive_s) == 0 or len(alive_l) == 0:
            break
        
        turn_frac = turn / 15
        hp_before_s = sum(w.hp for w in team_student)
        hp_before_l = sum(w.hp for w in team_legacy)
        
        # Collect states for alive student wizards
        states_batch = []
        masks_batch = []
        wizard_indices = []
        
        for i, wiz in enumerate(team_student):
            if wiz.hp <= 0:
                continue
            states_batch.append(get_state_vector_5v5(wiz, team_student, team_legacy))
            masks_batch.append(get_valid_spell_mask(wiz))
            wizard_indices.append(i)
        
        if states_batch:
            states_t = torch.FloatTensor(np.array(states_batch)).to(device)
            masks_t = torch.FloatTensor(np.array(masks_batch)).to(device)
            
            # Batched inference on GPU
            with torch.no_grad():
                logits, values = student(states_t, ROLE_STRIKER, turn_frac)
                
                # Safe masking
                safe_masks = masks_t.clone()
                dead = safe_masks.sum(dim=1) == 0
                safe_masks[dead, 0] = 1.0
                
                logits = logits.masked_fill(safe_masks == 0, float('-inf'))
                probs = F.softmax(logits / temperature, dim=-1)
                dist = torch.distributions.Categorical(probs)
                actions = dist.sample()
                log_probs = dist.log_prob(actions)
            
            # Store experience and execute
            for j, idx in enumerate(wizard_indices):
                buffer.add(
                    states_batch[j],
                    actions[j].item(),
                    log_probs[j].item(),
                    0.0,
                    values[j].item(),
                    0.0,
                )
                
                spell = SPELL_BOOK[SPELL_LIST[actions[j].item()]]
                hp_dmg = spell.get("hp_dmg", 0)
                target = next((e for e in team_legacy if e.hp > 0), None)
                if target:
                    target.hp -= hp_dmg
        
        # Legacy team attacks
        for i, wiz in enumerate(team_legacy):
            if wiz.hp <= 0:
                continue
            
            state = get_legacy_state_5v5(wiz, team_legacy, team_student)
            mask = get_valid_spell_mask(wiz)
            action = legacy_team[i % len(legacy_team)].get_action(state, mask)
            
            spell = SPELL_BOOK[SPELL_LIST[action]]
            hp_dmg = spell.get("hp_dmg", 0)
            target = next((e for e in team_student if e.hp > 0), None)
            if target:
                target.hp -= hp_dmg
        
        hp_after_s = sum(max(0, w.hp) for w in team_student)
        hp_after_l = sum(max(0, w.hp) for w in team_legacy)
        
        damage_dealt = hp_before_l - hp_after_l
        damage_taken = hp_before_s - hp_after_s
        reward = damage_dealt * 2 - damage_taken
        total_reward += reward
    
    # Fill rewards
    n = len(buffer)
    if n > 0:
        per_step = total_reward / n
        for i in range(len(buffer.rewards)):
            buffer.rewards[i] = per_step
    
    student_wins = sum(1 for w in team_student if w.hp > 0) > sum(1 for w in team_legacy if w.hp > 0)
    
    return {"win": student_wins, "reward": total_reward}


def ppo_update_gpu(
    student: UnifiedBrainV2,
    optimizer: optim.Adam,
    buffer: GPUBuffer,
    epochs: int = 4,
    clip_eps: float = 0.2,
    entropy_coef: float = 0.01,
    batch_size: int = 128,
) -> float:
    """GPU-accelerated PPO update with minibatching."""
    if len(buffer) < 32:
        return 0.0
    
    device = buffer.device
    states, actions, old_log_probs, rewards, values, dones = buffer.to_tensors()
    n = len(buffer)
    
    # Compute GAE on GPU
    advantages = torch.zeros(n, device=device)
    gae = 0
    
    for t in reversed(range(n)):
        next_value = 0 if t == n - 1 else values[t + 1]
        delta = rewards[t] + 0.99 * next_value * (1 - dones[t]) - values[t]
        gae = delta + 0.99 * 0.95 * (1 - dones[t]) * gae
        advantages[t] = gae
    
    returns = advantages + values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    total_loss = 0.0
    n_updates = 0
    
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        
        for start in range(0, n, batch_size):
            idx = perm[start:start+batch_size]
            
            batch_states = states[idx]
            batch_actions = actions[idx]
            batch_old_lp = old_log_probs[idx]
            batch_advs = advantages[idx]
            batch_returns = returns[idx]
            
            # Batched forward
            logits, values_pred = student(batch_states, ROLE_STRIKER, 0.5)
            values_pred = values_pred.squeeze()
            
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_lp = dist.log_prob(batch_actions)
            entropy = dist.entropy().mean()
            
            ratio = torch.exp(new_lp - batch_old_lp)
            surr1 = ratio * batch_advs
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * batch_advs
            
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values_pred, batch_returns)
            
            loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 0.5)
            optimizer.step()
            
            total_loss += loss.item()
            n_updates += 1
    
    return total_loss / max(1, n_updates)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="5v5 Imitation GPU")
    parser.add_argument("--imitation_episodes", type=int, default=200)
    parser.add_argument("--imitation_epochs", type=int, default=10)
    parser.add_argument("--finetune_iters", type=int, default=50)
    parser.add_argument("--episodes_per_iter", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_imitation_gpu")
    parser.add_argument("--legacy_dir", type=str, default="checkpoints_5v5")
    args = parser.parse_args()
    
    device = get_device()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    print("=" * 60)
    print("5V5 IMITATION + FINE-TUNING (GPU)")
    print("=" * 60)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)
    
    # Load Legacy brains
    print("\n[Phase 0] Loading Legacy brains (teacher)...")
    legacy_team = []
    for i in range(5):
        brain = LegacyBrainV2()  # Keep on CPU for demo collection
        path = os.path.join(args.legacy_dir, f"legacy_best_{i}.pth")
        if os.path.exists(path):
            brain.load_state_dict(torch.load(path, weights_only=True))
            print(f"  ✓ Loaded {path}")
        legacy_team.append(brain)
    
    # =========================================================================
    # PHASE 1: IMITATION (GPU)
    # =========================================================================
    print("\n[Phase 1] Collecting demonstrations...")
    states, actions = collect_demonstrations_gpu(legacy_team, args.imitation_episodes, device)
    print(f"  Collected {len(states)} state-action pairs on GPU")
    
    print("\n[Phase 1] Training student on GPU...")
    student = UnifiedBrainV2().to(device)
    train_imitation_gpu(student, states, actions, device, epochs=args.imitation_epochs)
    
    # Free memory
    del states, actions
    torch.cuda.empty_cache()
    
    imitated_path = os.path.join(args.checkpoint_dir, "imitation_phase1.pth")
    torch.save(student.state_dict(), imitated_path)
    print(f"\n  ✓ Saved: {imitated_path}")
    
    # =========================================================================
    # PHASE 2: FINE-TUNING (GPU)
    # =========================================================================
    print("\n[Phase 2] Fine-tuning with PPO on GPU...")
    
    optimizer = optim.Adam(student.parameters(), lr=args.lr)
    buffer = GPUBuffer(device)
    
    best_win_rate = 0.0
    
    for iteration in range(args.finetune_iters):
        iter_start = time.time()
        
        wins = 0
        total_reward = 0.0
        
        for _ in range(args.episodes_per_iter):
            stats = run_episode_gpu(student, legacy_team, buffer, device)
            if stats["win"]:
                wins += 1
            total_reward += stats["reward"]
        
        loss = ppo_update_gpu(student, optimizer, buffer)
        buffer.clear()
        
        win_rate = wins / args.episodes_per_iter
        avg_reward = total_reward / args.episodes_per_iter
        
        if win_rate > best_win_rate:
            best_win_rate = win_rate
            torch.save(student.state_dict(), os.path.join(args.checkpoint_dir, "student_best.pth"))
        
        elapsed = time.time() - iter_start
        
        if (iteration + 1) % 10 == 0 or iteration == 0:
            print(f"  Iter {iteration+1:3d} | vsLegacy: {win_rate*100:5.1f}% | "
                  f"Reward: {avg_reward:7.1f} | Loss: {loss:.4f} | {elapsed:.1f}s")
    
    final_path = os.path.join(args.checkpoint_dir, "student_final.pth")
    torch.save(student.state_dict(), final_path)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best win rate vs Legacy: {best_win_rate * 100:.1f}%")


if __name__ == "__main__":
    main()
