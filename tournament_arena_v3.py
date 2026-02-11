"""
Tournament Arena v3: Includes Legacy SquadAgent brains.

This version supports both:
- UnifiedBrain (new architecture)
- SquadAgent (legacy DQN brains)

Legacy brains only play as Enemy/Squad (since they're designed for enemy AI).

Usage:
    python tournament_arena_v3.py --games 1000 --workers 24
"""

import os
import argparse
import torch
import torch.multiprocessing as mp
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import random
import glob
import numpy as np

# Must set start method before any CUDA operations
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)


# ============================================================================
# LEGACY SQUADAGENT SUPPORT
# ============================================================================

from duel_engine import Wizard, resolve_round, SPELL_LIST, SPELL_BOOK


def load_squad_agent_brain(path: str, device: str = "cpu"):
    """Load a legacy SquadAgent DQN model."""
    import torch.nn as nn
    
    STATE_SIZE = 12
    ACTION_SIZE = len(SPELL_LIST)
    
    class DQN(nn.Module):
        def __init__(self):
            super(DQN, self).__init__()
            self.fc1 = nn.Linear(STATE_SIZE, 64)
            self.fc2 = nn.Linear(64, 64)
            self.fc3 = nn.Linear(64, ACTION_SIZE)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return self.fc3(x)
    
    model = DQN()
    model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    model.eval()
    return model


def squad_agent_get_state(me: Wizard, player: Wizard, allies: List[Wizard]) -> np.ndarray:
    """Get state vector for SquadAgent (12 features)."""
    me_hp = me.hp / me.max_hp
    me_pos = max(0, me.posture) / me.max_posture
    me_foc = me.focus / me.max_focus
    me_dist = me.dist / 2.0
    me_air = 1.0 if me.status["Airborne"] > 0 else 0.0
    
    p_hp = player.hp / player.max_hp
    p_pos = max(0, player.posture) / player.max_posture
    p_foc = player.focus / player.max_focus
    p_air = 1.0 if player.status["Airborne"] > 0 else 0.0
    p_stun = 1.0 if player.status["Frozen"] > 0 or player.status["Stunned"] > 0 else 0.0
    
    ally_low = 0.0
    ally_air = 0.0
    for a in allies:
        if a.id != me.id and a.hp > 0:
            if (a.hp / a.max_hp) < 0.3:
                ally_low = 1.0
            if a.status["Airborne"] > 0:
                ally_air = 1.0
    
    return np.array([me_hp, me_pos, me_foc, me_dist, me_air,
                     p_hp, p_pos, p_foc, p_air, p_stun,
                     ally_low, ally_air], dtype=np.float32)


def squad_agent_select_action(model, state: np.ndarray) -> int:
    """Select action using SquadAgent model."""
    with torch.no_grad():
        state_t = torch.FloatTensor(state).unsqueeze(0)
        q_values = model(state_t)
        return q_values.max(1)[1].item()


# ============================================================================
# UNIFIED BRAIN SUPPORT
# ============================================================================

from unified_brain_v2 import UnifiedBrainV2, get_state_vector_5v5, get_valid_spell_mask
from unified_brain_v2 import ROLE_GUARDIAN, ROLE_STRIKER, ROLE_DISRUPTOR, NUM_SPELLS



@dataclass
class UnifiedModel:
    path: str
    single_file: bool = False  # If True, one file shared for all 5 slots. If False, expects path_{i}.pth pattern

def load_unified_brain(model_info: UnifiedModel, slot_idx: int = 0, device: str = "cpu") -> UnifiedBrainV2:
    """Load UnifiedBrainV2 model handling single-file vs per-slot."""
    brain = UnifiedBrainV2()
    
    if model_info.single_file:
        path = model_info.path
    else:
        # Expects a pattern with {i} or similar, or just uses the path if it already has the index
        # For this system, we store the pattern in model_info.path for per-slot models
        path = model_info.path.format(i=slot_idx % 5)
    
    if os.path.exists(path):
        brain.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    else:
        print(f"Warning: Checkpoint not found {path}, using random weights")
        
    brain.eval()
    return brain


# ============================================================================
# MATCH RUNNER
# ============================================================================

@dataclass
class MatchResult:
    brain_a: str
    brain_b: str
    score_a: float
    score_b: float
    games_played: int


def run_duel_mixed(
    player_brain,
    player_type: str,  # "unified" or "squad"
    squad_brains: Dict,  # {role: (brain, type)}
    num_games: int = 1,
) -> float:
    """
    Run a duel between a player brain and a squad of brains.
    
    Returns:
        Score for player (positive = player won, negative = squad won)
    """
    total_player_damage = 0.0
    total_squad_damage = 0.0
    
    for _ in range(num_games):
        player = Wizard("Player", "prodigy", True, 99)
        enemies = [
            Wizard("Tank", "auror", False, 1),
            Wizard("DPS", "curse_specialist", False, 2),
            Wizard("Support", "death_eater", False, 3),
        ]
        
        # Map 1=Tank/Guardian, 2=DPS/Striker, 3=Support/Disruptor
        role_mapping = {1: ROLE_GUARDIAN, 2: ROLE_STRIKER, 3: ROLE_DISRUPTOR}
        
        for turn in range(15):
            if player.hp <= 0 or all(e.hp <= 0 for e in enemies):
                break
            
            turn_frac = turn / 15
            
            # --- Player action ---
            living = [e for e in enemies if e.hp > 0]
            target_e = living[0] if living else enemies[0]
            
            if player_type == "unified":
                # V2 state extraction: (me, my_team, enemy_team)
                # Player has no teammates in this 1v3 scenario, so team is just [player]
                p_state = get_state_vector_5v5(player, [player], enemies)
                p_mask = get_valid_spell_mask(player)
                
                # Turn frac needed for V2
                spell_id = player_brain.get_action(p_state, ROLE_STRIKER, turn_frac, p_mask)
                player_spell = SPELL_LIST[spell_id]
                
                # Smart targeting since V2 output is just spell_id
                # Target the first enemy or prioritize based on spell
                if player_spell == "Descendo":
                    airborne = [e for e in enemies if e.hp > 0 and e.status.get("Airborne", 0) > 0]
                    target_e = airborne[0] if airborne else target_e
                
                player_target = target_e.id
            else:  # squad type as player (shouldn't happen but fallback)
                player_spell = "Basic Cast"
                player_target = target_e.id
            
            # --- Squad actions ---
            ai_moves = {}
            ai_targets = {}
            
            for e in enemies:
                if e.hp <= 0:
                    continue
                
                brain, btype = squad_brains.get(e.id, (None, None))
                
                if btype == "unified":
                    # Squad member state: (me, my_team, enemy_team)
                    # For them, my_team is enemies list, enemy_team is [player]
                    e_state = get_state_vector_5v5(e, enemies, [player])
                    e_mask = get_valid_spell_mask(e)
                    
                    spell_id = brain.get_action(e_state, ROLE_STRIKER, turn_frac, e_mask)
                    spell = SPELL_LIST[spell_id]
                    ai_moves[e.id] = spell
                    
                    # Smart targeting for squad
                    tgt = 99 # Player ID
                    if spell == "Descendo" and player.status.get("Airborne", 0) > 0:
                        tgt = 99
                    # Could add ally targeting here if needed
                    ai_targets[e.id] = tgt
                    
                elif btype == "squad":
                    e_state = squad_agent_get_state(e, player, enemies)
                    spell_id = squad_agent_select_action(brain, e_state)
                    ai_moves[e.id] = SPELL_LIST[spell_id]
                    ai_targets[e.id] = 99  # Legacy brains always target player
                    
                else:
                    # Random fallback
                    ai_moves[e.id] = random.choice(SPELL_LIST[:5])
                    ai_targets[e.id] = 99
            
            # --- Resolve ---
            hp_before_player = player.hp
            hp_before_squad = sum(e.hp for e in enemies)
            
            resolve_round(player, enemies, player_spell, player_target, ai_moves, ai_targets)
            
            hp_after_player = player.hp
            hp_after_squad = sum(e.hp for e in enemies if e.hp > 0)
            
            total_player_damage += (hp_before_squad - hp_after_squad)
            total_squad_damage += (hp_before_player - hp_after_player)
        
        # Win bonus
        if player.hp <= 0:
            total_squad_damage += 50
        elif all(e.hp <= 0 for e in enemies):
            total_player_damage += 50
    
    return total_player_damage - total_squad_damage


def run_match_worker_v3(args: Tuple) -> MatchResult:
    """
    Worker for mixed matches.
    
    Args format: (path_a, type_a, path_b, type_b, name_a, name_b, games_per_side)
    """
    path_a, type_a, path_b, type_b, name_a, name_b, games_per_side = args
    
    # Optimize CPU usage for parallel workers
    torch.set_num_threads(1)
    
    # Load brains
    # Type is now an object for unified, or "squad" string for legacy
    
    if isinstance(type_a, UnifiedModel):
        # A is unified player
        brain_a = load_unified_brain(type_a, slot_idx=0, device="cpu")
    else:
        brain_a = load_squad_agent_brain(path_a, "cpu")
    
    if isinstance(type_b, UnifiedModel):
        # B is unified player (for mixed duel logic, usually B is squad so we load just one instance for now
        # but run_duel_mixed handles the squad loading internally if we pass the info)
        brain_b = load_unified_brain(type_b, slot_idx=0, device="cpu")
    else:
        brain_b = load_squad_agent_brain(path_b, "cpu")
    
    total_score_a = 0.0
    total_score_b = 0.0
    
    # --- A as Player, B as all 3 Squad members ---
    if isinstance(type_a, UnifiedModel):  # Only unified can be player
        # For B (squad), if it's unified we need to load per-slot if needed
        squad_b = {}
        for role_id in [1, 2, 3]:
            if isinstance(type_b, UnifiedModel):
                b_brain = load_unified_brain(type_b, slot_idx=role_id, device="cpu")
                squad_b[role_id] = (b_brain, "unified")
            else:
                b_brain = load_squad_agent_brain(path_b, "cpu") # Legacy is one file
                squad_b[role_id] = (b_brain, "squad")
                
        score = run_duel_mixed(brain_a, "unified", squad_b, games_per_side)
        total_score_a += score
        total_score_b -= score
    
    # --- B as Player, A as all 3 Squad members ---
    if isinstance(type_b, UnifiedModel):  # Only unified can be player
        squad_a = {}
        for role_id in [1, 2, 3]:
            if isinstance(type_a, UnifiedModel):
                a_brain = load_unified_brain(type_a, slot_idx=role_id, device="cpu")
                squad_a[role_id] = (a_brain, "unified")
            else:
                a_brain = load_squad_agent_brain(path_a, "cpu")
                squad_a[role_id] = (a_brain, "squad")
                
        score = run_duel_mixed(brain_b, "unified", squad_a, games_per_side)
        total_score_b += score
        total_score_a -= score
    
    # If neither can be player (both squad types), just have them fight as squads
    # If neither can be player (both squad types), just have them fight as squads
    if not isinstance(type_a, UnifiedModel) and not isinstance(type_b, UnifiedModel):
        # Both are squad-only, create random player and compare
        for _ in range(games_per_side):
            # A controls squad vs random player
            from unified_brain_v2 import UnifiedBrainV2 as UB
            random_brain = UB()
            squad_a = {}
            for role_id in [1, 2, 3]: # Legacy loader puts single brain in slot
                if isinstance(type_a, UnifiedModel):
                     b = load_unified_brain(type_a, slot_idx=role_id, device="cpu")
                     squad_a[role_id] = (b, "unified")
                else: 
                     b = load_squad_agent_brain(path_a, "cpu")
                     squad_a[role_id] = (b, "squad")
            
            score_a = run_duel_mixed(random_brain, "unified", squad_a, 1)
            total_score_a -= score_a  # Squad won if score_a is negative
            
            score_a = run_duel_mixed(random_brain, "unified", squad_a, 1)
            total_score_a -= score_a  # Squad won if score_a is negative
            
            squad_b = {}
            for role_id in [1, 2, 3]:
                if isinstance(type_b, UnifiedModel):
                     b = load_unified_brain(type_b, slot_idx=role_id, device="cpu")
                     squad_b[role_id] = (b, "unified")
                else:
                     b = load_squad_agent_brain(path_b, "cpu")
                     squad_b[role_id] = (b, "squad")
            
            score_b = run_duel_mixed(random_brain, "unified", squad_b, 1)
            total_score_b -= score_b
    
    return MatchResult(
        brain_a=name_a,
        brain_b=name_b,
        score_a=total_score_a,
        score_b=total_score_b,
        games_played=games_per_side * 2,
    )


def find_all_checkpoints() -> Dict[str, Any]:
    """Find all checkpoints (both unified and legacy)."""
    checkpoints = {}  # {name: (path, type_obj)}
    
    # 1. Evo Best (checkpoints_evo_gpu)
    if os.path.exists("checkpoints_evo_gpu/unified_best_0.pth"):
        checkpoints["Evo Best GPU"] = ("checkpoints_evo_gpu/unified_best_{i}.pth", 
                                      UnifiedModel("checkpoints_evo_gpu/unified_best_{i}.pth", False))
    
    # 2. Evo Generations (checkpoints_evo_gpu)
    if os.path.exists("checkpoints_evo_gpu"):
        gens = sorted(set(
            int(f.split("_gen")[1].split("_")[0])
            for f in os.listdir("checkpoints_evo_gpu")
            if f.startswith("unified_gen") and f.endswith(".pth")
        ))
        # Add a few key generations
        for g in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            if g in gens:
                checkpoints[f"Evo Gen {g}"] = (f"checkpoints_evo_gpu/unified_gen{g}_{{i}}.pth",
                                              UnifiedModel(f"checkpoints_evo_gpu/unified_gen{g}_{{i}}.pth", False))
    
    # 3. MAPPO (checkpoints_mappo_gpu)
    if os.path.exists("checkpoints_mappo_gpu/mappo_best.pth"):
        checkpoints["MAPPO Best"] = ("checkpoints_mappo_gpu/mappo_best.pth",
                                    UnifiedModel("checkpoints_mappo_gpu/mappo_best.pth", True))
        
    # 4. PBT (checkpoints_pbt_gpu)
    if os.path.exists("checkpoints_pbt_gpu/pbt_best.pth"):
        checkpoints["PBT Best"] = ("checkpoints_pbt_gpu/pbt_best.pth",
                                  UnifiedModel("checkpoints_pbt_gpu/pbt_best.pth", True))
    

    # 5. Imitation (checkpoints_imitation_gpu)
    if os.path.exists("checkpoints_imitation_gpu/student_best.pth"):
        checkpoints["Imitation Best"] = ("checkpoints_imitation_gpu/student_best.pth",
                                        UnifiedModel("checkpoints_imitation_gpu/student_best.pth", True))

    # 5.5. DQN (checkpoints_dqn_gpu)
    if os.path.exists("checkpoints_dqn_gpu/dqn_best.pth"):
        checkpoints["DQN Best"] = ("checkpoints_dqn_gpu/dqn_best.pth",
                                  UnifiedModel("checkpoints_dqn_gpu/dqn_best.pth", True))
    
    # 6. Original Evo (checkpoints_5v5)
    if os.path.exists("checkpoints_5v5/unified_best_0.pth"):
        checkpoints["Original Evo Best"] = ("checkpoints_5v5/unified_best_{i}.pth",
                                           UnifiedModel("checkpoints_5v5/unified_best_{i}.pth", False))
    
    # 7. Legacy SquadAgent checkpoints
    legacy = [
        ("Legacy Baseline", "brain_baseline.pth"),
        ("Legacy DPS", "brain_dps.pth"),
        ("Legacy Tank", "brain_tank.pth"),
        ("Legacy Support", "brain_supp.pth"),
    ]
    
    for name, path in legacy:
        if os.path.exists(path):
            checkpoints[name] = (path, "squad")
            
    # Add Legacy V2 Best from 5v5 dir
    if os.path.exists("checkpoints_5v5/legacy_best_0.pth"):
         # For Tournament V3 legacy support, we treat Legacy V2 as "squad" type if valid
         # But the current loader is for old DQN brains. 
         # LegacyBrainV2 is a Policy Network.
         # For simplicity, let's skip Legacy V2 in this Mixed Tournament for now
         # unless we update the loader.
         pass
            
    return checkpoints

def select_participants(checkpoints: Dict) -> Dict:
    """Let user select participants."""
    print("\nAvailable Participants:")
    names = sorted(checkpoints.keys())
    for i, name in enumerate(names):
        print(f"  {i+1}. {name}")
    
    print("\nSelect participants (comma separated IDs, e.g. '1,3,5') or 'all':")
    choice = input("> ").strip().lower()
    
    if choice == 'all':
        return checkpoints
    
    selected = {}
    try:
        indices = [int(x.strip()) - 1 for x in choice.split(",")]
        for idx in indices:
            if 0 <= idx < len(names):
                name = names[idx]
                selected[name] = checkpoints[name]
    except:
        print("Invalid input. Using all.")
        return checkpoints
        
    return selected


def print_table(headers: List[str], rows: List[List]):
    col_widths = [max(len(str(row[i])) for row in rows + [headers]) + 2 for i in range(len(headers))]
    header_row = "|".join(str(h).center(w) for h, w in zip(headers, col_widths))
    print(f"|{header_row}|")
    print("|" + "|".join("-" * w for w in col_widths) + "|")
    for row in rows:
        data_row = "|".join(str(cell).center(w) for cell, w in zip(row, col_widths))
        print(f"|{data_row}|")


def format_number(n: float) -> str:
    if abs(n) >= 1_000_000:
        return f"{n/1_000_000:.1f}M"
    elif abs(n) >= 1_000:
        return f"{n/1_000:.1f}K"
    else:
        return f"{n:.0f}"


@dataclass
class Stats:
    name: str
    brain_type: str
    wins: int = 0
    losses: int = 0
    draws: int = 0
    total_damage: float = 0.0
    games_played: int = 0
    
    @property
    def score(self):
        return self.wins * 3 + self.draws + self.total_damage * 0.001


def main():
    parser = argparse.ArgumentParser(description="Tournament Arena v3 (with Legacy Support)")
    parser.add_argument("--games", type=int, default=500, help="Games per side")
    parser.add_argument("--workers", type=int, default=24, help="Parallel workers")
    parser.add_argument("--output", type=str, default="tournament_results.md", help="Output MD file")
    args = parser.parse_args()
    
    # Default workers to CPU count if not specified (or if default 24 is high/low)
    # Actually, let's just use all cores minus 2 for system stability if high enough, else all
    # But for max performance user asked "all cores", so let's try strict cpu_count
    import multiprocessing
    max_cores = multiprocessing.cpu_count()
    if args.workers == 24: # usage default
        args.workers = max_cores
        
    print("=" * 70)
    print("     TOURNAMENT ARENA v3 (UnifiedBrain + Legacy SquadAgent)")
    print("=" * 70)
    print(f"Games per matchup: {args.games * 2:,}")
    print(f"Workers: {args.workers} (System has {max_cores})")
    
    all_checkpoints = find_all_checkpoints()
    checkpoints = select_participants(all_checkpoints)
    
    if len(checkpoints) < 2:
        print(f"Not enough brains: {list(checkpoints.keys())}")
        return
    
    print(f"\nContestants: {len(checkpoints)}")
    for name, (path, btype) in checkpoints.items():
        print(f"  - {name} [{btype}]")
    
    names = list(checkpoints.keys())
    n = len(names)
    n_matchups = n * (n - 1) // 2
    
    print(f"\nTotal matchups: {n_matchups}")
    print(f"Total games: ~{n_matchups * args.games * 2:,}")
    
    # Prepare tasks
    # Prepare tasks (chunked)
    BATCH_SIZE = 50  # Games per chunk (per side) = 100 actual games
    tasks = []
    
    for i in range(n):
        for j in range(i + 1, n):
            path_a, type_a = checkpoints[names[i]]
            path_b, type_b = checkpoints[names[j]]
            
            # Split games into chunks
            full_chunks = args.games // BATCH_SIZE
            remainder = args.games % BATCH_SIZE
            
            for _ in range(full_chunks):
                tasks.append((path_a, type_a, path_b, type_b, names[i], names[j], BATCH_SIZE))
            
            if remainder > 0:
                tasks.append((path_a, type_a, path_b, type_b, names[i], names[j], remainder))
                
    total_tasks = len(tasks)
    print(f"Total tasks (chunked): {total_tasks}")
    
    # Run matches
    print("\n" + "=" * 70)
    print("RUNNING MATCHES")
    print("=" * 70)
    
    start = time.time()
    results = []
    
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(run_match_worker_v3, t): t for t in tasks}
        done = 0
        for f in as_completed(futures):
            r = f.result()
            results.append(r)
            done += 1
            if done % 10 == 0 or done == total_tasks:
                print(f"  [{done}/{total_tasks}] Chunk completed ({r.games_played} games) - {r.brain_a} vs {r.brain_b}")
    
    elapsed = time.time() - start
    
    # Compile stats
    # Aggregate results by pairing
    aggregated = {} # (name_a, name_b) -> MatchResult
    
    for r in results:
        # Sort key to handle (A,B) vs (B,A) if logic triggers, though our loops are strict (i < j)
        # But r.brain_a and r.brain_b come from task which preserves order
        key = (r.brain_a, r.brain_b)
        if key not in aggregated:
            aggregated[key] = MatchResult(r.brain_a, r.brain_b, 0.0, 0.0, 0)
        
        aggregated[key].score_a += r.score_a
        aggregated[key].score_b += r.score_b
        aggregated[key].games_played += r.games_played

    # Compile stats from aggregated results
    stats = {n: Stats(name=n, brain_type=checkpoints[n][1]) for n in names}
    
    for r in aggregated.values():
        stats[r.brain_a].games_played += r.games_played
        stats[r.brain_b].games_played += r.games_played
        stats[r.brain_a].total_damage += r.score_a
        stats[r.brain_b].total_damage += r.score_b
        
        # Determine win based on TOTAL score across all chunks
        margin = 100 * (r.games_played / 6)
        if r.score_a > r.score_b + margin:
            stats[r.brain_a].wins += 1
            stats[r.brain_b].losses += 1
        elif r.score_b > r.score_a + margin:
            stats[r.brain_b].wins += 1
            stats[r.brain_a].losses += 1
        else:
            stats[r.brain_a].draws += 1
            stats[r.brain_b].draws += 1
    
    # Rankings
    print("\n" + "=" * 70)
    print("FINAL RANKINGS")
    print("=" * 70)
    
    ranked = sorted(stats.values(), key=lambda x: x.score, reverse=True)
    rows = []
    for i, s in enumerate(ranked, 1):
        rows.append([i, s.name, f"[{s.brain_type}]", s.wins, s.losses, s.draws, format_number(s.total_damage), f"{s.score:.1f}"])
    
    print()
    print_table(["#", "Name", "Type", "W", "L", "D", "Damage", "Score"], rows)
    
    # Save to file
    if args.output:
        try:
            with open(args.output, "w") as f:
                f.write(f"# Tournament Rankings ({time.strftime('%Y-%m-%d %H:%M:%S')})\n\n")
                f.write(f"**Settings**: {args.games}x2 games per matchup, {args.workers} workers\n\n")
                
                # Markdown Table
                headers = ["#", "Name", "Type", "W", "L", "D", "Damage", "Score"]
                f.write("| " + " | ".join(headers) + " |\n")
                f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
                for r in rows:
                    f.write("| " + " | ".join(str(x) for x in r) + " |\n")
                
                f.write(f"\n**👑 CHAMPION**: {ranked[0].name} [{ranked[0].brain_type}]\n")
                f.write(f"\n**Time**: {elapsed:.1f}s | **Throughput**: {n_matchups * args.games * 2 / elapsed:.0f} games/sec\n")
            print(f"\n✓ Results saved to {args.output}")
        except Exception as e:
            print(f"\n✗ Error saving results: {e}")
            
    print(f"\n👑 CHAMPION: {ranked[0].name} [{ranked[0].brain_type}]")
    print(f"\nTime: {elapsed:.1f}s | Throughput: {n_matchups * args.games * 2 / elapsed:.0f} games/sec")


if __name__ == "__main__":
    main()
