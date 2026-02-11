"""
Duel Arena 5v5: Game runner for 5v5 matches.

Features:
- 5v5 team battles (flexible composition)
- Limited visibility state extraction  
- Smart targeting based on visible info
- Supports both UnifiedBrain V2 and Legacy Brain V2
- Uses resolve_round_5v5 for full spell logic
"""

import random
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

from duel_engine import Wizard, resolve_round, resolve_round_5v5, SPELL_LIST, SPELL_BOOK


# ==============================================================================
# GAME RESULT
# ==============================================================================

@dataclass
class GameResult5v5:
    """Result of a 5v5 match."""
    team_a_wins: bool
    team_a_alive: int
    team_b_alive: int
    team_a_damage_dealt: float
    team_b_damage_dealt: float
    turns_played: int
    
    @property
    def score_diff(self) -> float:
        """Score difference (positive = team A won)."""
        return self.team_a_damage_dealt - self.team_b_damage_dealt


# ==============================================================================
# TEAM CREATION
# ==============================================================================

def create_team(
    team_name: str,
    is_team_a: bool = True,
    archetypes: Optional[List[str]] = None,
) -> List[Wizard]:
    """
    Create a team of 5 wizards.
    
    Args:
        team_name: Team identifier (e.g., "Alpha", "Beta")
        is_team_a: Whether this is team A (affects wizard IDs)
        archetypes: Optional list of 5 archetypes
    
    Returns:
        List of 5 Wizard objects
    """
    default_archetypes = [
        "auror",           # Guardian
        "curse_specialist", # Striker 1
        "curse_specialist", # Striker 2
        "prodigy",         # Disruptor 1
        "prodigy",         # Disruptor 2
    ]
    
    archetypes = archetypes or default_archetypes
    team = []
    
    base_id = 0 if is_team_a else 100
    
    for i in range(5):
        name = f"{team_name}_{i+1}"
        archetype = archetypes[i] if i < len(archetypes) else "prodigy"
        wiz = Wizard(name, archetype, is_team_a, base_id + i)
        team.append(wiz)
    
    return team


# ==============================================================================
# SMART TARGETING
# ==============================================================================

def get_best_enemy_target(enemies: List[Wizard], spell_name: str) -> Optional[int]:
    """
    Get best enemy target based on LIMITED visibility (only airborne status).
    
    Priority:
    1. Airborne enemy (for Descendo = guaranteed kill)
    2. First alive enemy
    """
    living = [e for e in enemies if e.hp > 0]
    if not living:
        return None
    
    # Descendo prioritizes airborne
    if spell_name == "Descendo":
        airborne = [e for e in living if e.status.get("Airborne", 0) > 0]
        if airborne:
            return airborne[0].id
    
    # Default: first living enemy
    return living[0].id


def get_best_ally_target(
    my_team: List[Wizard],
    me: Wizard,
    spell_name: str,
) -> Optional[int]:
    """
    Get best ally target based on visible info.
    
    For Descendo: save airborne ally
    """
    allies = [w for w in my_team if w.id != me.id and w.hp > 0]
    if not allies:
        return None
    
    if spell_name == "Descendo":
        airborne = [a for a in allies if a.status.get("Airborne", 0) > 0]
        if airborne:
            return airborne[0].id
    
    return None


# ==============================================================================
# 5V5 DUEL RUNNER
# ==============================================================================

def run_5v5_duel(
    team_a_brains: List,
    team_b_brains: List,
    brain_type_a: str = "unified",  # "unified" or "legacy"
    brain_type_b: str = "unified",
    max_turns: int = 20,
    verbose: bool = False,
) -> GameResult5v5:
    """
    Run a 5v5 match between two teams.
    
    Args:
        team_a_brains: 5 brain models for team A
        team_b_brains: 5 brain models for team B
        brain_type_a: "unified" or "legacy"
        brain_type_b: "unified" or "legacy"
        max_turns: Maximum turns before draw
        verbose: Print match events
    
    Returns:
        GameResult5v5 with match outcome
    """
    # Import brain modules

    from unified_brain_v2 import get_state_vector_5v5, get_valid_spell_mask, ROLE_STRIKER
    from legacy_brain_v2 import get_legacy_state_5v5, get_valid_spell_mask as legacy_mask
    from legacy_adapter import get_squad_state
    
    # Create teams
    team_a = create_team("Alpha", is_team_a=True)
    team_b = create_team("Beta", is_team_a=False)
    
    total_hp_a = sum(w.hp for w in team_a)
    total_hp_b = sum(w.hp for w in team_b)
    
    damage_dealt_a = 0.0
    damage_dealt_b = 0.0
    
    for turn in range(max_turns):
        # Check win conditions
        alive_a = sum(1 for w in team_a if w.hp > 0)
        alive_b = sum(1 for w in team_b if w.hp > 0)
        
        if alive_a == 0 or alive_b == 0:
            break
        
        turn_frac = turn / max_turns
        
        if verbose:
            print(f"\n--- Turn {turn + 1} ---")
            print(f"  Team A: {alive_a} alive")
            print(f"  Team B: {alive_b} alive")
        
        # --- Collect all actions ---
        actions = {}  # {wizard_id: (spell_name, target_id)}
        
        # Team A actions
        for i, wiz in enumerate(team_a):
            if wiz.hp <= 0:
                continue
            
            brain = team_a_brains[i] if i < len(team_a_brains) else team_a_brains[0]
            

            if brain_type_a == "unified":
                state = get_state_vector_5v5(wiz, team_a, team_b)
                mask = get_valid_spell_mask(wiz)
                spell_id = brain.get_action(state, ROLE_STRIKER, turn_frac, mask)
            elif brain_type_a == "squad":
                target = next((e for e in team_b if e.hp > 0), team_b[0])
                state = get_squad_state(wiz, target, team_a) # allies includes self in list, handled by func
                # Brain is likely a wrapper or model that takes state
                # If wrapped by adapter that takes (state, mask), ok.
                # If wrapped by LegacyEvalAdapter (PBT), it takes (state, mask).
                spell_id = brain.get_action(state, None)
            else:  # legacy
                state = get_legacy_state_5v5(wiz, team_a, team_b)
                mask = legacy_mask(wiz)
                spell_id = brain.get_action(state, mask)
            
            spell_name = SPELL_LIST[spell_id]
            target_id = get_best_enemy_target(team_b, spell_name)
            
            # Check if should target ally (Descendo for airborne ally)
            ally_target = get_best_ally_target(team_a, wiz, spell_name)
            if ally_target is not None:
                target_id = ally_target
            
            actions[wiz.id] = (spell_name, target_id or team_b[0].id)
        
        # Team B actions
        for i, wiz in enumerate(team_b):
            if wiz.hp <= 0:
                continue
            
            brain = team_b_brains[i] if i < len(team_b_brains) else team_b_brains[0]
            
            if brain_type_b == "unified":
                state = get_state_vector_5v5(wiz, team_b, team_a)
                mask = get_valid_spell_mask(wiz)
                spell_id = brain.get_action(state, ROLE_STRIKER, turn_frac, mask)
            elif brain_type_b == "squad":
                target = next((e for e in team_a if e.hp > 0), team_a[0])
                state = get_squad_state(wiz, target, team_b)
                spell_id = brain.get_action(state, None)
            else:  # legacy
                state = get_legacy_state_5v5(wiz, team_b, team_a)
                mask = legacy_mask(wiz)
                spell_id = brain.get_action(state, mask)
            
            spell_name = SPELL_LIST[spell_id]
            target_id = get_best_enemy_target(team_a, spell_name)
            
            ally_target = get_best_ally_target(team_b, wiz, spell_name)
            if ally_target is not None:
                target_id = ally_target
            
            actions[wiz.id] = (spell_name, target_id or team_a[0].id)
        
        # --- Resolve round with full spell logic ---
        hp_before_a = sum(w.hp for w in team_a)
        hp_before_b = sum(w.hp for w in team_b)
        
        all_wizards = team_a + team_b
        logs = resolve_round_5v5(all_wizards, actions)
        
        if verbose:
            for log in logs:
                caster_id, spell, target_id, hp_dmg, pos_dmg, effect = log
                if target_id is not None:
                    print(f"    {caster_id} → {spell} → {target_id}: -{hp_dmg} HP, -{pos_dmg} POS {effect}")
                else:
                    print(f"    {caster_id} → {spell}: {effect}")
        
        hp_after_a = sum(max(0, w.hp) for w in team_a)
        hp_after_b = sum(max(0, w.hp) for w in team_b)
        
        damage_dealt_a += (hp_before_b - hp_after_b)
        damage_dealt_b += (hp_before_a - hp_after_a)
    
    # Final result
    alive_a = sum(1 for w in team_a if w.hp > 0)
    alive_b = sum(1 for w in team_b if w.hp > 0)
    
    team_a_wins = alive_a > alive_b or (alive_a == alive_b and damage_dealt_a > damage_dealt_b)
    
    return GameResult5v5(
        team_a_wins=team_a_wins,
        team_a_alive=alive_a,
        team_b_alive=alive_b,
        team_a_damage_dealt=damage_dealt_a,
        team_b_damage_dealt=damage_dealt_b,
        turns_played=turn + 1,
    )


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def run_many_matches(
    team_a_brains: List,
    team_b_brains: List,
    brain_type_a: str = "unified",
    brain_type_b: str = "unified",
    num_games: int = 10,
) -> Tuple[int, int, float]:
    """
    Run multiple matches and return stats.
    
    Returns:
        (team_a_wins, team_b_wins, avg_score_diff)
    """
    a_wins = 0
    b_wins = 0
    total_diff = 0.0
    
    for _ in range(num_games):
        result = run_5v5_duel(
            team_a_brains, team_b_brains,
            brain_type_a, brain_type_b,
        )
        
        if result.team_a_wins:
            a_wins += 1
        else:
            b_wins += 1
        
        total_diff += result.score_diff
    
    return a_wins, b_wins, total_diff / num_games
