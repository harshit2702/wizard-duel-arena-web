
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import copy
import time

from unified_brain_v2 import UnifiedBrainV2, get_state_vector_5v5, get_valid_spell_mask
from unified_brain_v2 import ROLE_STRIKER, ROLE_GUARDIAN, ROLE_DISRUPTOR, NUM_SPELLS, STATE_DIM
from duel_arena_5v5 import create_team
from duel_engine import SPELL_LIST, SPELL_BOOK
from legacy_adapter import LegacyAgentWrapper

# ==============================================================================
# DEVICE SETUP
# ==============================================================================

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

# ==============================================================================
# REPLAY BUFFER
# ==============================================================================

@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    mask: np.ndarray
    next_mask: np.ndarray

class ReplayBuffer:
    def __init__(self, capacity: int, device):
        self.capacity = capacity
        self.device = device
        self.data: List[Transition] = []
        self.position = 0
    
    def add(self, *args):
        if len(self.data) < self.capacity:
            self.data.append(None)
        self.data[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int):
        batch = random.sample(self.data, batch_size)
        
        states = torch.FloatTensor(np.array([t.state for t in batch])).to(self.device)
        actions = torch.LongTensor([t.action for t in batch]).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor([t.reward for t in batch]).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(np.array([t.next_state for t in batch])).to(self.device)
        dones = torch.FloatTensor([1.0 if t.done else 0.0 for t in batch]).unsqueeze(1).to(self.device)
        masks = torch.FloatTensor(np.array([t.mask for t in batch])).to(self.device)
        next_masks = torch.FloatTensor(np.array([t.next_mask for t in batch])).to(self.device)
        
        return states, actions, rewards, next_states, dones, masks, next_masks
    
    def __len__(self):
        return len(self.data)


# ==============================================================================
# DQN UPDATE
# ==============================================================================

def dqn_update(
    policy_net: UnifiedBrainV2,
    target_net: UnifiedBrainV2,
    optimizer: optim.Adam,
    buffer: ReplayBuffer,
    batch_size: int = 64,
    gamma: float = 0.99,
):
    if len(buffer) < batch_size:
        return 0.0
    
    states, actions, rewards, next_states, dones, masks, next_masks = buffer.sample(batch_size)
    
    # Current Q values
    # UnifiedBrainV2 returns logits, value (for PPO). We used logits as Q-values for legacy DQN.
    # Q-Learning interpretation: Logits ARE the Q-values.
    q_values, _ = policy_net(states, ROLE_STRIKER, 0.5)
    current_q = q_values.gather(1, actions)
    
    # Next Q values (Target)
    with torch.no_grad():
        next_q_raw, _ = target_net(next_states, ROLE_STRIKER, 0.5)
        # Mask invalid actions in next state
        next_q_raw = next_q_raw.masked_fill(next_masks == 0, -1e9)
        max_next_q = next_q_raw.max(1)[0].unsqueeze(1)
        expected_q = rewards + (gamma * max_next_q * (1 - dones))
    
    loss = F.mse_loss(current_q, expected_q)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0)
    optimizer.step()
    
    return loss.item()

# ==============================================================================
# EPISODE RUNNER
# ==============================================================================

def run_dqn_episode(
    policy_net: UnifiedBrainV2,
    target_net: UnifiedBrainV2,
    buffer: ReplayBuffer,
    epsilon: float,
    device,
    legacy_opponents: Optional[List[LegacyAgentWrapper]] = None,
) -> Dict:
    team_a = create_team("Alpha", is_team_a=True)
    team_b = create_team("Beta", is_team_a=False)
    
    total_reward = 0.0
    use_legacy = legacy_opponents is not None
    
    for turn in range(20):
        alive_a = [w for w in team_a if w.hp > 0]
        alive_b = [w for w in team_b if w.hp > 0]
        
        if not alive_a or not alive_b:
            break
        
        turn_frac = turn / 20.0
        
        # --- TEAM A (TRAINING) ACTIONS ---
        obs_states = []
        obs_masks = []
        indices = []
        
        for i, wiz in enumerate(team_a):
            if wiz.hp <= 0:
                continue
            
            s = get_state_vector_5v5(wiz, team_a, team_b)
            m = get_valid_spell_mask(wiz)
            
            obs_states.append(s)
            obs_masks.append(m)
            indices.append(i)
        
        # Batch inference
        states_t = torch.FloatTensor(np.array(obs_states)).to(device)
        masks_t = torch.FloatTensor(np.array(obs_masks)).to(device)
        
        actions = []
        with torch.no_grad():
            q_vals, _ = policy_net(states_t, ROLE_STRIKER, turn_frac)
            # Epsilon Greedy
            for j in range(len(indices)):
                if random.random() < epsilon:
                    valid_indices = np.where(obs_masks[j] > 0)[0]
                    if len(valid_indices) > 0:
                        act = random.choice(valid_indices)
                    else:
                        act = 0 # Default if no mana
                else:
                    # Mask invalid
                    valid_q = q_vals[j].clone()
                    valid_q[masks_t[j] == 0] = -1e9
                    act = valid_q.argmax().item()
                actions.append(act)
        

        # --- EXECUTION PHASE ---
        hp_before_a = sum(max(0, w.hp) for w in team_a)
        hp_before_b = sum(max(0, w.hp) for w in team_b)
        
        # Execute Team A
        for j, idx in enumerate(indices):
            wiz = team_a[idx]
            spell = SPELL_BOOK[SPELL_LIST[actions[j]]]
            
            costs = spell.get("cost", 0)
            if wiz.focus >= costs:
                wiz.focus -= costs
                hp_dmg = spell.get("hp_dmg", 0)
                target = next((e for e in team_b if e.hp > 0), None)
                if target:
                    target.hp -= hp_dmg # Simple damage
            
        # Execute Team B
        for i, wiz in enumerate(team_b):
            if wiz.hp <= 0: continue
            
            if use_legacy:
                agent = legacy_opponents[i % len(legacy_opponents)]
                spell_id = agent.get_action(wiz, team_a, team_b)
            else:
                # Self play against TARGET NET for stability
                s_b = get_state_vector_5v5(wiz, team_b, team_a)
                m_b = get_valid_spell_mask(wiz)
                with torch.no_grad():
                    qs, _ = target_net(torch.FloatTensor(s_b).unsqueeze(0).to(device), ROLE_STRIKER, turn_frac)
                    qs[0][torch.FloatTensor(m_b).to(device) == 0] = -1e9
                    spell_id = qs.argmax().item()
            
            spell = SPELL_BOOK[SPELL_LIST[spell_id]]
            costs = spell.get("cost", 0)
            if wiz.focus >= costs:
                wiz.focus -= costs
                hp_dmg = spell.get("hp_dmg", 0)
                target = next((e for e in team_a if e.hp > 0), None)
                if target:
                    target.hp -= hp_dmg

        hp_after_a = sum(max(0, w.hp) for w in team_a)
        hp_after_b = sum(max(0, w.hp) for w in team_b)
        
        dmg_dealt = hp_before_b - hp_after_b
        dmg_taken = hp_before_a - hp_after_a
        
        step_reward = (dmg_dealt * 2.0) - (dmg_taken * 0.5)
        total_reward += step_reward
        
        done = (hp_after_a <= 0) or (hp_after_b <= 0)
        if done:
            if hp_after_b <= 0: step_reward += 50
            if hp_after_a <= 0: step_reward -= 20
        
        # Store transitions
        # We need next state for each agent
        
        for j, idx in enumerate(indices):
            wiz = team_a[idx]
            # If dead, next state is terminal (zeros)
            if wiz.hp <= 0:
                ns = np.zeros(STATE_DIM, dtype=np.float32)
                nm = np.zeros(NUM_SPELLS, dtype=np.float32)
                d = True
            else:
                ns = get_state_vector_5v5(wiz, team_a, team_b)
                nm = get_valid_spell_mask(wiz)
                d = done
            
            # Add to buffer: (s, a, r, s', done, mask, next_mask)
            buffer.add(
                obs_states[j],
                actions[j],
                step_reward, # Shared reward
                ns,
                d,
                obs_masks[j],
                nm
            )
            
        if done:
            break
            
    won = sum(1 for w in team_a if w.hp > 0) > sum(1 for w in team_b if w.hp > 0)
    return {"win": won, "reward": total_reward}

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--buffer_size", type=int, default=10000)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.05)
    parser.add_argument("--epsilon_decay", type=float, default=0.995)
    parser.add_argument("--target_update", type=int, default=10)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_dqn_gpu")
    args = parser.parse_args()
    
    device = get_device()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    print("="*60)
    print("5V5 DQN TRAINING (GPU)")
    print(f"Device: {device}")
    print("="*60)
    
    policy_net = UnifiedBrainV2().to(device)
    target_net = UnifiedBrainV2().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    
    optimizer = optim.Adam(policy_net.parameters(), lr=args.lr)
    buffer = ReplayBuffer(args.buffer_size, device)
    
    epsilon = args.epsilon_start
    best_win_rate = 0.0
    
    # Load Legacy
    legacy_agents = [
        LegacyAgentWrapper("brain_dps.pth", "cpu"),
        # Add others if available or duplicate
        LegacyAgentWrapper("brain_tank.pth", "cpu"),
        LegacyAgentWrapper("brain_supp.pth", "cpu"),
        LegacyAgentWrapper("brain_dps.pth", "cpu"),
        LegacyAgentWrapper("brain_baseline.pth", "cpu")
    ]
    
    for i in range(args.iterations):
        wins = 0
        tot_rew = 0
        start = time.time()
        
        for _ in range(args.episodes):
            use_legacy = random.random() < 0.5
            opps = legacy_agents if use_legacy else None
            
            res = run_dqn_episode(policy_net, target_net, buffer, epsilon, device, opps)
            if res["win"]: wins += 1
            tot_rew += res["reward"]
            
            # Update per episode or per step? Per episode update loop
            loss = dqn_update(policy_net, target_net, optimizer, buffer, args.batch_size, args.gamma)
            
        # Periodically update target
        if (i + 1) % args.target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
            
        # Decay epsilon
        epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)
        
        wr = wins / args.episodes
        if wr > best_win_rate:
            best_win_rate = wr
            torch.save(policy_net.state_dict(), os.path.join(args.checkpoint_dir, "dqn_best.pth"))
            
        print(f"Iter {i+1:3d} | Win: {wr*100:5.1f}% | Rew: {tot_rew/args.episodes:6.1f} | Eps: {epsilon:.3f} | {time.time()-start:.1f}s")

if __name__ == "__main__":
    main()
