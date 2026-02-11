"""
Train 5v5 Evolution GPU: Self-play evolution with CUDA acceleration.

GPU Optimizations:
- Models on GPU for fast inference
- Batched evaluation for fitness scoring
- Workers use serialized state_dicts for CPU evaluation

Usage:
    python train_5v5_evo_gpu.py --generations 50 --population 20 --workers 24
"""

import os
import argparse
import random
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from typing import List, Tuple, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
from dataclasses import dataclass
import numpy as np

from unified_brain_v2 import UnifiedBrainV2, get_state_vector_5v5, get_valid_spell_mask
from unified_brain_v2 import ROLE_STRIKER, NUM_SPELLS, STATE_DIM
from legacy_brain_v2 import LegacyBrainV2
from duel_arena_5v5 import run_5v5_duel, create_team
from duel_engine import SPELL_LIST, SPELL_BOOK


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
# TEAM WRAPPER (GPU-aware)
# ==============================================================================

@dataclass
class Team:
    """A team of 5 brains."""
    brains: List
    brain_type: str  # "unified" or "legacy"
    fitness: float = 0.0
    
    def to(self, device):
        """Move all brains to device."""
        for brain in self.brains:
            brain.to(device)
        return self
    
    def cpu(self):
        """Move all brains to CPU."""
        return self.to(torch.device("cpu"))
    
    def state_dicts(self):
        """Get state dicts for serialization."""
        return [b.state_dict() for b in self.brains]


def create_unified_team(device) -> Team:
    """Create team of 5 UnifiedBrain V2 on device."""
    brains = [UnifiedBrainV2().to(device) for _ in range(5)]
    return Team(brains=brains, brain_type="unified")


def create_legacy_team(device) -> Team:
    """Create team of 5 LegacyBrain V2 on device."""
    brains = [LegacyBrainV2().to(device) for _ in range(5)]
    return Team(brains=brains, brain_type="legacy")


def mutate_team(team: Team, device, rate: float = 0.1, strength: float = 0.1) -> Team:
    """Mutate all brains."""
    new_brains = []
    for brain in team.brains:
        new_brain = brain.copy().to(device)
        new_brain.mutate(rate, strength)
        new_brains.append(new_brain)
    return Team(brains=new_brains, brain_type=team.brain_type)


def crossover_teams(t1: Team, t2: Team, device) -> Team:
    """Crossover two teams."""
    new_brains = []
    for b1, b2 in zip(t1.brains, t2.brains):
        child = b1.crossover(b2).to(device)
        new_brains.append(child)
    return Team(brains=new_brains, brain_type=t1.brain_type)


# ==============================================================================
# GPU-ACCELERATED EPISODE
# ==============================================================================

def run_episode_gpu(
    team_a: Team,
    team_b: Team,
    device,
) -> Tuple[bool, float]:
    """
    Run match with GPU inference.
    
    Returns:
        (team_a_wins, score_diff)
    """
    wizards_a = create_team("Alpha", is_team_a=True)
    wizards_b = create_team("Beta", is_team_a=False)
    
    total_score = 0.0
    
    for turn in range(15):
        alive_a = [w for w in wizards_a if w.hp > 0]
        alive_b = [w for w in wizards_b if w.hp > 0]
        
        if not alive_a or not alive_b:
            break
        
        turn_frac = turn / 15
        hp_before_a = sum(w.hp for w in wizards_a)
        hp_before_b = sum(w.hp for w in wizards_b)
        
        # Team A actions (batched on GPU)
        states_a = []
        masks_a = []
        for wiz in wizards_a:
            if wiz.hp > 0:
                states_a.append(get_state_vector_5v5(wiz, wizards_a, wizards_b))
                masks_a.append(get_valid_spell_mask(wiz))
        
        if states_a:
            states_t = torch.FloatTensor(np.array(states_a)).to(device)
            masks_t = torch.FloatTensor(np.array(masks_a)).to(device)
            
            with torch.no_grad():
                all_logits = []
                for i, brain in enumerate(team_a.brains[:len(states_a)]):
                    if team_a.brain_type == "unified":
                        logits, _ = brain(states_t[i:i+1], ROLE_STRIKER, turn_frac)
                    else:
                        logits = brain(states_t[i:i+1])
                    all_logits.append(logits)
                
                logits = torch.cat(all_logits, dim=0)
                safe_masks = masks_t.clone()
                dead = safe_masks.sum(dim=1) == 0
                safe_masks[dead, 0] = 1.0
                logits = logits.masked_fill(safe_masks == 0, float('-inf'))
                probs = F.softmax(logits, dim=-1)
                actions_a = torch.multinomial(probs, 1).squeeze(-1)  # Keep as 1-dim
            
            # Execute team A attacks
            alive_idx = 0
            for wiz in wizards_a:
                if wiz.hp <= 0:
                    continue
                spell = SPELL_BOOK[SPELL_LIST[actions_a[alive_idx].item()]]
                hp_dmg = spell.get("hp_dmg", 0)
                target = next((e for e in wizards_b if e.hp > 0), None)
                if target:
                    target.hp -= hp_dmg
                alive_idx += 1
        
        # Team B actions (batched on GPU)
        states_b = []
        masks_b = []
        for wiz in wizards_b:
            if wiz.hp > 0:
                states_b.append(get_state_vector_5v5(wiz, wizards_b, wizards_a))
                masks_b.append(get_valid_spell_mask(wiz))
        
        if states_b:
            states_t = torch.FloatTensor(np.array(states_b)).to(device)
            masks_t = torch.FloatTensor(np.array(masks_b)).to(device)
            
            with torch.no_grad():
                all_logits = []
                for i, brain in enumerate(team_b.brains[:len(states_b)]):
                    if team_b.brain_type == "unified":
                        logits, _ = brain(states_t[i:i+1], ROLE_STRIKER, turn_frac)
                    else:
                        logits = brain(states_t[i:i+1])
                    all_logits.append(logits)
                
                logits = torch.cat(all_logits, dim=0)
                safe_masks = masks_t.clone()
                dead = safe_masks.sum(dim=1) == 0
                safe_masks[dead, 0] = 1.0
                logits = logits.masked_fill(safe_masks == 0, float('-inf'))
                probs = F.softmax(logits, dim=-1)
                actions_b = torch.multinomial(probs, 1).squeeze(-1)  # Keep as 1-dim
            
            alive_idx = 0
            for wiz in wizards_b:
                if wiz.hp <= 0:
                    continue
                spell = SPELL_BOOK[SPELL_LIST[actions_b[alive_idx].item()]]
                hp_dmg = spell.get("hp_dmg", 0)
                target = next((e for e in wizards_a if e.hp > 0), None)
                if target:
                    target.hp -= hp_dmg
                alive_idx += 1
        
        hp_after_a = sum(max(0, w.hp) for w in wizards_a)
        hp_after_b = sum(max(0, w.hp) for w in wizards_b)
        
        total_score += (hp_before_b - hp_after_b) - (hp_before_a - hp_after_a)
    
    alive_a = sum(1 for w in wizards_a if w.hp > 0)
    alive_b = sum(1 for w in wizards_b if w.hp > 0)
    
    team_a_wins = alive_a > alive_b or (alive_a == alive_b and total_score > 0)
    
    return team_a_wins, total_score


# ==============================================================================
# TOURNAMENT
# ==============================================================================

def run_tournament_gpu(
    unified_pop: List[Team],
    legacy_pop: List[Team],
    device,
    games_per_match: int = 3,
) -> None:
    """
    Run tournament on GPU.
    """
    for team in unified_pop + legacy_pop:
        team.fitness = 0.0
    
    for u_team in unified_pop:
        for l_team in legacy_pop:
            for _ in range(games_per_match):
                u_wins, score = run_episode_gpu(u_team, l_team, device)
                if u_wins:
                    u_team.fitness += 100 + score
                else:
                    l_team.fitness += 100 - score


# ==============================================================================
# SELECTION
# ==============================================================================

def select_parents(population: List[Team], num: int) -> List[Team]:
    """Tournament selection."""
    return sorted(population, key=lambda t: t.fitness, reverse=True)[:num]


def next_generation(population: List[Team], pop_size: int, device, 
                    mut_rate: float = 0.1, mut_strength: float = 0.1) -> List[Team]:
    """Create next generation."""
    parents = select_parents(population, max(2, pop_size // 4))
    
    next_gen = [parents[0]]  # Elitism
    
    while len(next_gen) < pop_size:
        p1, p2 = random.sample(parents, 2)
        child = crossover_teams(p1, p2, device)
        child = mutate_team(child, device, mut_rate, mut_strength)
        next_gen.append(child)
    
    return next_gen[:pop_size]


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="5v5 Evolution GPU")
    parser.add_argument("--generations", type=int, default=50)
    parser.add_argument("--population", type=int, default=10)
    parser.add_argument("--games_per_match", type=int, default=3)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints_evo_gpu")
    args = parser.parse_args()
    
    device = get_device()
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    print("=" * 60)
    print("5V5 EVOLUTION TRAINING (GPU)")
    print("=" * 60)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Generations: {args.generations}")
    print(f"Population: {args.population} per architecture")
    print("=" * 60)
    
    # Initialize on GPU
    print("\nInitializing populations on GPU...")
    unified_pop = [create_unified_team(device) for _ in range(args.population)]
    legacy_pop = [create_legacy_team(device) for _ in range(args.population)]
    
    best_unified = float('-inf')
    best_legacy = float('-inf')
    
    for gen in range(args.generations):
        gen_start = time.time()
        
        # Tournament
        run_tournament_gpu(unified_pop, legacy_pop, device, args.games_per_match)
        
        # Stats
        best_u = max(t.fitness for t in unified_pop)
        best_l = max(t.fitness for t in legacy_pop)
        avg_u = sum(t.fitness for t in unified_pop) / len(unified_pop)
        avg_l = sum(t.fitness for t in legacy_pop) / len(legacy_pop)
        
        # Save best
        if best_u > best_unified:
            best_unified = best_u
            best_team = max(unified_pop, key=lambda t: t.fitness)
            for i, brain in enumerate(best_team.brains):
                torch.save(brain.state_dict(), 
                          os.path.join(args.checkpoint_dir, f"unified_best_{i}.pth"))
        
        if best_l > best_legacy:
            best_legacy = best_l
            best_team = max(legacy_pop, key=lambda t: t.fitness)
            for i, brain in enumerate(best_team.brains):
                torch.save(brain.state_dict(),
                          os.path.join(args.checkpoint_dir, f"legacy_best_{i}.pth"))
        
        elapsed = time.time() - gen_start
        print(f"Gen {gen+1:3d} | U: {best_u:.0f}/{avg_u:.0f} | L: {best_l:.0f}/{avg_l:.0f} | {elapsed:.1f}s")
        
        # Next generation
        unified_pop = next_generation(unified_pop, args.population, device)
        legacy_pop = next_generation(legacy_pop, args.population, device)
        
        if (gen + 1) % 10 == 0:
            for arch, pop in [("unified", unified_pop), ("legacy", legacy_pop)]:
                best_team = max(pop, key=lambda t: t.fitness)
                for i, brain in enumerate(best_team.brains):
                    torch.save(brain.state_dict(),
                              os.path.join(args.checkpoint_dir, f"{arch}_gen{gen+1}_{i}.pth"))
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Best Unified: {best_unified:.0f}")
    print(f"Best Legacy: {best_legacy:.0f}")


if __name__ == "__main__":
    main()
