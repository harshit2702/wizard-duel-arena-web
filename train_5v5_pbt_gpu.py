"""
Train 5v5 PBT GPU: Population-Based Training with CUDA acceleration.

GPU Optimizations:
- Models on GPU for faster forward/backward pass
- Batched tensor operations on GPU
- Workers use CPU for game simulation, GPU for updates

Usage:
    python train_5v5_pbt_gpu.py --population 20 --iterations 100 --workers 24
"""

import os
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import copy
import numpy as np


from unified_brain_v2 import UnifiedBrainV2, get_state_vector_5v5, get_valid_spell_mask
from unified_brain_v2 import ROLE_STRIKER, NUM_SPELLS, STATE_DIM

from duel_arena_5v5 import run_5v5_duel, create_team
from duel_engine import SPELL_LIST, SPELL_BOOK
from legacy_adapter import LegacyAgentWrapper, load_legacy_model, select_legacy_action


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)


# ==============================================================================
# DEVICE SETUP
# ==============================================================================

def get_device():
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ==============================================================================
# HYPERPARAMETER MEMBER
# ==============================================================================

@dataclass
class PBTMember:
    """A member of the PBT population with hyperparameters."""
    brain: UnifiedBrainV2
    hyperparams: Dict = field(default_factory=dict)
    fitness: float = 0.0
    
    def __post_init__(self):
        if not self.hyperparams:
            self.hyperparams = {
                "lr": random.uniform(1e-4, 5e-4), # Lowered range
                "entropy_coef": random.uniform(0.01, 0.05), # Increased range
                "temperature": random.uniform(0.5, 1.0),
            }
    
    def to(self, device):
        """Move brain to device."""
        self.brain = self.brain.to(device)
        return self


def create_member(device) -> PBTMember:
    """Create a new PBT member on device."""
    brain = UnifiedBrainV2().to(device)
    return PBTMember(brain=brain)


def exploit(member: PBTMember, elite: PBTMember, device) -> PBTMember:
    """Copy weights from elite to member."""
    new_brain = elite.brain.copy().to(device)
    return PBTMember(brain=new_brain, hyperparams=copy.deepcopy(elite.hyperparams))


def explore(member: PBTMember, perturb_factor: float = 0.2) -> None:
    """Perturb hyperparameters randomly."""
    for key in member.hyperparams:
        if random.random() < 0.5:
            factor = 1.0 + random.uniform(-perturb_factor, perturb_factor)
            member.hyperparams[key] *= factor
    
    member.hyperparams["lr"] = max(1e-5, min(1e-3, member.hyperparams["lr"]))
    member.hyperparams["entropy_coef"] = max(0.005, min(0.1, member.hyperparams["entropy_coef"]))
    member.hyperparams["temperature"] = max(0.2, min(2.0, member.hyperparams["temperature"]))


# ==============================================================================
# GPU-OPTIMIZED PPO BUFFER
# ==============================================================================

class GPUBuffer:
    """GPU-accelerated PPO buffer with batched operations."""
    
    def __init__(self, device):
        self.device = device
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def add(self, state, action, log_prob, reward, value, done):
        self.states.append(state)  # Keep as numpy for now
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def to_tensors(self):
        """Convert all data to GPU tensors for batched processing."""
        if not self.states:
            return None
            
        return (
            torch.FloatTensor(np.array(self.states)).to(self.device),
            torch.LongTensor(self.actions).to(self.device),
            torch.FloatTensor(self.log_probs).to(self.device),
            torch.FloatTensor(self.rewards).to(self.device),
            torch.FloatTensor(self.values).to(self.device),
            torch.FloatTensor(self.dones).to(self.device),
        )
    
    def clear(self):
        self.__init__(self.device)
    
    def __len__(self):
        return len(self.states)


# ==============================================================================
# GPU-OPTIMIZED PPO UPDATE
# ==============================================================================

def compute_gae_gpu(rewards, values, dones, gamma=0.99, lam=0.95):
    """Compute GAE on GPU."""
    n = len(rewards)
    advantages = torch.zeros(n, device=rewards.device)
    gae = 0
    
    for t in reversed(range(n)):
        if t == n - 1:
            next_value = 0
        else:
            next_value = values[t + 1]
        
        delta = rewards[t] + gamma * next_value * (1 - dones[t]) - values[t]
        gae = delta + gamma * lam * (1 - dones[t]) * gae
        advantages[t] = gae
    
    returns = advantages + values
    return advantages, returns


def ppo_update_gpu(
    brain: UnifiedBrainV2,
    optimizer: optim.Adam,
    buffer: GPUBuffer,
    hyperparams: Dict,
    epochs: int = 4,
    batch_size: int = 64,
):
    """GPU-accelerated PPO update with minibatch processing."""
    if len(buffer) < 32:
        return 0.0
    
    device = buffer.device
    entropy_coef = hyperparams.get("entropy_coef", 0.01)
    clip_eps = 0.2
    
    tensors = buffer.to_tensors()
    if tensors is None:
        return 0.0
        
    states, actions, old_log_probs, rewards, values, dones = tensors
    
    # Compute advantages on GPU
    advantages, returns = compute_gae_gpu(rewards, values, dones)
    if advantages.std() > 1e-8:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    n = len(states)
    total_loss = 0.0
    n_updates = 0
    
    for _ in range(epochs):
        # Shuffle indices
        perm = torch.randperm(n, device=device)
        
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            idx = perm[start:end]
            
            batch_states = states[idx]
            batch_actions = actions[idx]
            batch_old_lp = old_log_probs[idx]
            batch_advs = advantages[idx]
            batch_returns = returns[idx]
            
            # Forward pass (batched on GPU)
            logits, values_pred = brain(batch_states, ROLE_STRIKER, 0.5)
            values_pred = values_pred.squeeze()
            
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(batch_actions)
            entropy = dist.entropy().mean()
            
            # PPO loss
            ratio = torch.exp(new_log_probs - batch_old_lp)
            surr1 = ratio * batch_advs
            surr2 = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * batch_advs
            
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values_pred, batch_returns)
            
            loss = policy_loss + 0.5 * value_loss - entropy_coef * entropy
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(brain.parameters(), 0.5)
            optimizer.step()
            
            total_loss += loss.item()
            n_updates += 1
    
    return total_loss / max(1, n_updates)


# ==============================================================================
# EPISODE RUNNER (CPU for game logic)
# ==============================================================================

def run_episode_gpu(
    brain: UnifiedBrainV2,
    opponent_brains: List[UnifiedBrainV2],
    buffer: GPUBuffer,
    temperature: float,
    device,
    legacy_opponents: Optional[List[LegacyAgentWrapper]] = None,
) -> float:
    """Run episode with GPU acceleration for inference."""
    
    team_a = create_team("Alpha", is_team_a=True)
    team_b = create_team("Beta", is_team_a=False)
    
    total_reward = 0.0
    use_legacy = legacy_opponents is not None
    
    for turn in range(20):
        alive_a = sum(1 for w in team_a if w.hp > 0)
        alive_b = sum(1 for w in team_b if w.hp > 0)
        
        if alive_a == 0 or alive_b == 0:
            break
        
        turn_frac = turn / 20
        
        # Batch collect states from all alive team A wizards
        states_batch = []
        masks_batch = []
        wizard_indices = []
        
        for i, wiz in enumerate(team_a):
            if wiz.hp <= 0:
                continue
            state = get_state_vector_5v5(wiz, team_a, team_b)
            mask = get_valid_spell_mask(wiz)
            states_batch.append(state)
            masks_batch.append(mask)
            wizard_indices.append(i)
        
        if not states_batch:
            continue
        
        # Batched inference on GPU
        states_t = torch.FloatTensor(np.array(states_batch)).to(device)
        masks_t = torch.FloatTensor(np.array(masks_batch)).to(device)
        
        with torch.no_grad():
            logits, values = brain(states_t, ROLE_STRIKER, turn_frac)
            logits = logits.masked_fill(masks_t == 0, float('-inf'))
            probs = F.softmax(logits / temperature, dim=-1)
            dist = torch.distributions.Categorical(probs)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
        
        # Store experience (back to CPU)
        for j, idx in enumerate(wizard_indices):
            buffer.add(
                states_batch[j],  # Keep as numpy
                actions[j].item(),
                log_probs[j].item(),
                0.0,
                values[j].item(),
                0.0,
            )
        
        # Execute combat
        hp_before_a = sum(w.hp for w in team_a)
        hp_before_b = sum(w.hp for w in team_b)
        
        # Team A attacks
        for j, idx in enumerate(wizard_indices):
            wiz = team_a[idx]
            spell = SPELL_BOOK[SPELL_LIST[actions[j].item()]]
            hp_dmg = spell.get("hp_dmg", 0)
            target = next((e for e in team_b if e.hp > 0), None)
            if target:
                target.hp -= hp_dmg
        
        # Team B attacks
        for i, wiz in enumerate(team_b):
            if wiz.hp <= 0:
                continue
            
            if use_legacy:
                agent = legacy_opponents[i % len(legacy_opponents)]
                # Legacy agent expects (me, enemies, allies)
                spell_id = agent.get_action(wiz, team_a, team_b)
            else:
                opp_brain = opponent_brains[i % len(opponent_brains)]
                state = get_state_vector_5v5(wiz, team_b, team_a)
                spell_id = opp_brain.get_action(state, ROLE_STRIKER, turn_frac, get_valid_spell_mask(wiz))
            
            spell = SPELL_BOOK[SPELL_LIST[spell_id]]
            hp_dmg = spell.get("hp_dmg", 0)
            target = next((e for e in team_a if e.hp > 0), None)
            if target:
                target.hp -= hp_dmg
        
        hp_after_a = sum(max(0, w.hp) for w in team_a)
        hp_after_b = sum(max(0, w.hp) for w in team_b)
        
        damage_dealt = hp_before_b - hp_after_b
        damage_taken = hp_before_a - hp_after_a
        
        # Reward shaping matching MAPPO
        reward = (damage_dealt * 1.5) - (damage_taken * 0.5)
        total_reward += reward
    
    # Assign rewards
    if len(buffer) > 0:
        per_step = total_reward / len(buffer)
        for i in range(len(buffer.rewards)):
            buffer.rewards[i] = per_step
    
    return total_reward


# ==============================================================================
# EVALUATION (CPU workers)
# ==============================================================================

def evaluate_member(args: Tuple) -> Tuple[int, float]:
    """Evaluate member against LEGACY AGENTS."""
    member_idx, brain_state, num_games = args
    
    brain = UnifiedBrainV2()
    brain.load_state_dict(brain_state)
    
    # Load legacy opponents only (much simpler and fairer benchmark)
    legacy_opponents = [
        LegacyAgentWrapper("brain_dps.pth", device="cpu"),
        # LegacyAgentWrapper("brain_tank.pth", device="cpu"),
        # LegacyAgentWrapper("brain_supp.pth", device="cpu"),
        # LegacyAgentWrapper("brain_baseline.pth", device="cpu"),
    ]
    # We'll use a mix inside duel runner or just fix one role?
    # run_5v5_duel supports "legacy" brain type but expects a list of agents.
    # LegacyAgentWrapper has get_action compatible with what run_5v5_duel might expect?
    # Actually run_5v5_duel handles "legacy" type by assuming it has .get_action(state)
    # But LegacyAgentWrapper wraps model.get_action.
    # Let's verify run_5v5_duel interface.
    
    # In run_5v5_duel:
    # if brain_type_b == "legacy":
    #     spell_id = brain.get_action(state, mask) 
    # It passes state and mask!
    
    # LegacyAgentWrapper.get_action signature: (me, enemies, allies) used in training loop.
    # This is incompatible with run_5v5_duel's "legacy" expectation.
    
    # However, we can use "custom" opponents if we modify run_5v5_duel, or simpler:
    # Just run a simplified duel here like in training, but for evaluation.
    # Or, rely on the fact that LegacyAgentWrapper holds a model that works with state?
    # LegacyAgentWrapper.model is the DQN. DQN.forward(state) works.
    # LegacyBrainV2.get_action(state) works.
    
    # So we should load LegacyBrainV2 (from legacy_brain_v2.py) which has the right interface?
    # No, the weights are from SquadAgent.
    # LegacyAdapter has `select_legacy_action(model, state)`.
    
    # Let's adhere to run_5v5_duel logic for "legacy" mode which uses `legacy_brain_v2.get_legacy_state_5v5`.
    # AND it assumes the brain object has .get_action(state, mask).
    
    # We need a small adapter class here that mimics LegacyBrainV2 interface but uses our loaded weights.
    
    class LegacyEvalAdapter:
        def __init__(self, wrapper):
            self.model = wrapper.model
            self.device = "cpu"
        def get_action(self, state, mask=None):
            return select_legacy_action(self.model, state, self.device)
            
    opponents = [LegacyEvalAdapter(legacy_opponents[0]) for _ in range(5)]
    team_brains = [brain] * 5
    
    total_score = 0.0
    

    for _ in range(num_games):
        # We start with team A (Unified) vs Team B (Legacy)
        # We tell run_5v5_duel that Team B is "squad" (12-dim state).
        result = run_5v5_duel(team_brains, opponents, "unified", "squad")
        total_score += result.score_diff
    
    return member_idx, total_score


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="5v5 PBT GPU Training")
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--episodes_per_iter", type=int, default=10)
    parser.add_argument("--eval_games", type=int, default=5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_pbt_gpu")
    args = parser.parse_args()
    
    device = get_device()
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    print("=" * 60)
    print("5V5 PBT (GPU) - HYBRID EVALUATION")
    print("=" * 60)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Population: {args.population}")
    print(f"Iterations: {args.iterations}")
    print("=" * 60)
    
    # Initialize population on GPU
    print("\nInitializing population on GPU...")
    population = [create_member(device) for _ in range(args.population)]
    optimizers = [optim.Adam(m.brain.parameters(), lr=m.hyperparams["lr"]) for m in population]
    
    # Load Legacy Agents for TRAINING mixture
    legacy_agents = [
        LegacyAgentWrapper("brain_dps.pth", device="cpu"),
        LegacyAgentWrapper("brain_tank.pth", device="cpu")
    ]
    
    best_fitness = float('-inf')
    
    # Training Mix Probability
    legacy_prob = 0.2
    
    for iteration in range(args.iterations):
        iter_start = time.time()
        
        # Training on GPU
        for i, member in enumerate(population):
            buffer = GPUBuffer(device)
            
            other_brains = [population[(i + j) % args.population].brain for j in range(1, 4)] # 3 opponents? loop needs 5
            # We need 5 opponents usually. The loop in run_episode modded len.
            # Just pass the whole population or list of 5 neighbors
            neighbor_brains = [population[(i + j + 1) % args.population].brain for j in range(5)]
            
            for _ in range(args.episodes_per_iter):
                # Mix in Legacy
                use_legacy = random.random() < legacy_prob
                opps = legacy_agents if use_legacy else None
                
                run_episode_gpu(member.brain, neighbor_brains, buffer, 
                               member.hyperparams["temperature"], device, legacy_opponents=opps)
            
            if len(buffer) > 0:
                optimizers[i].param_groups[0]["lr"] = member.hyperparams["lr"]
                ppo_update_gpu(member.brain, optimizers[i], buffer, member.hyperparams)
        
        # Evaluation on CPU (parallel) matching against Legacy
        tasks = [
            (i, m.brain.cpu().state_dict(), args.eval_games)
            for i, m in enumerate(population)
        ]
        
        # Move back to GPU
        for m in population:
            m.brain = m.brain.to(device)
            
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            results = list(ex.map(evaluate_member, tasks))
        
        for idx, score in results:
            population[idx].fitness = score
        
        # PBT exploit/explore
        sorted_pop = sorted(range(len(population)), key=lambda i: population[i].fitness, reverse=True)
        top_20_pct = max(1, len(population) // 5)
        bottom_20_pct = sorted_pop[-top_20_pct:]
        top_20_indices = sorted_pop[:top_20_pct]
        
        for bad_idx in bottom_20_pct:
            elite_idx = random.choice(top_20_indices)
            population[bad_idx] = exploit(population[bad_idx], population[elite_idx], device)
            explore(population[bad_idx])
            optimizers[bad_idx] = optim.Adam(
                population[bad_idx].brain.parameters(),
                lr=population[bad_idx].hyperparams["lr"]
            )
        
        # Stats
        best = max(m.fitness for m in population)
        avg = sum(m.fitness for m in population) / len(population)
        
        if best > best_fitness:
            best_fitness = best
            best_member = max(population, key=lambda m: m.fitness)
            torch.save(
                best_member.brain.state_dict(),
                os.path.join(args.checkpoint_dir, "pbt_best.pth")
            )
        
        elapsed = time.time() - iter_start
        
        best_hp = max(population, key=lambda m: m.fitness).hyperparams
        print(f"Iter {iteration+1:3d} | Best (vsLegacy): {best:.0f} | Avg: {avg:.0f} | "
              f"lr={best_hp['lr']:.5f} | {elapsed:.1f}s")
        
        if (iteration + 1) % 10 == 0:
            best_member = max(population, key=lambda m: m.fitness)
            torch.save(
                best_member.brain.state_dict(),
                os.path.join(args.checkpoint_dir, f"pbt_iter{iteration+1}.pth")
            )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best fitness: {best_fitness:.0f}")


if __name__ == "__main__":
    main()
