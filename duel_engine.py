"""
Duel Engine v6.1.1 (Target Override Patch)

Patch:
- resolve_round(...) now supports ai_targets_override (enemy spell+target control).
- end_round_tick now uses correct took_damage tracking (posture regen correctness).
- Backward compatible: older callers can still pass only ai_moves_override.

Everything else matches your v6.1 logic as closely as possible.
"""

import random

# ==========================
# CONSTANTS
# ==========================
CLOSE = 0
MID = 1
FAR = 2

RANGE_NAMES = {0: "CLOSE", 1: "MID", 2: "FAR"}

MIN_POSTURE = -20
POSTURE_REGEN = 5
FOCUS_REGEN = 15  # Increased from 10 to make expensive spells more viable

# ==========================
# SPELL DATA
# ==========================
SPELL_BOOK = {
    # DEFENSE (Prio 1)
    "Protego": {"type": "Defense", "prio": 1, "cost": 0},
    "Protego Maximus": {"type": "Defense", "prio": 1, "cost": 30},

    # INFO (Prio 2)
    "Revelio": {"type": "Info", "prio": 2, "cost": 10},
    "Legilimens": {"type": "Info", "prio": 2, "cost": 20},

    # CONTROL (Prio 3)
    "Levioso": {"type": "Control", "prio": 3, "cost": 15, "pos_dmg": 20},
    "Glacius": {"type": "Control", "prio": 3, "cost": 20, "pos_dmg": 0},
    "Arresto Momentum": {"type": "Control", "prio": 3, "cost": 25, "pos_dmg": 30},

    # FORCE (Prio 4)
    "Accio": {"type": "Force", "prio": 4, "cost": 15, "pos_dmg": 15, "hp_dmg": 5},
    "Depulso": {"type": "Force", "prio": 4, "cost": 15, "pos_dmg": 25, "hp_dmg": 5},
    "Descendo": {"type": "Force", "prio": 4, "cost": 20, "pos_dmg": 20, "hp_dmg": 10},

    # DAMAGE & CURSE (Prio 5)
    "Basic Cast": {"type": "Damage", "prio": 5, "cost": 0, "pos_dmg": 5, "hp_dmg": 5},
    "Incendio": {"type": "Damage", "prio": 5, "cost": 25, "pos_dmg": 5},
    "Confringo": {"type": "Damage", "prio": 5, "cost": 30, "pos_dmg": 10, "hp_dmg": 25},
    "Diffindo": {"type": "Damage", "prio": 5, "cost": 35, "pos_dmg": 5, "hp_dmg": 45},
    "Crucio": {"type": "Curse", "prio": 5, "cost": 50, "pos_dmg": 0, "hp_dmg": 0},
    "Avada Kedavra": {"type": "Curse", "prio": 5, "cost": 100, "pos_dmg": 0, "hp_dmg": 40},
}

SPELL_LIST = list(SPELL_BOOK.keys())

ARCHETYPES = {
    "auror": {"hp": 110, "pos": 55, "foc": 80, "max_foc": 150},
    "brawler": {"hp": 120, "pos": 60, "foc": 80, "max_foc": 140},
    "curse_specialist": {"hp": 100, "pos": 30, "foc": 120, "max_foc": 150},
    "death_eater": {"hp": 95, "pos": 30, "foc": 110, "max_foc": 150},
    "occlumens": {"hp": 80, "pos": 50, "foc": 100, "max_foc": 160},
    "strategist": {"hp": 100, "pos": 50, "foc": 105, "max_foc": 150},
    "prodigy": {"hp": 100, "pos": 50, "foc": 100, "max_foc": 150},
    "dueling_master": {"hp": 100, "pos": 60, "foc": 95, "max_foc": 150},
    "default": {"hp": 100, "pos": 50, "foc": 100, "max_foc": 150},
}


class Wizard:
    def __init__(self, name, archetype_key, is_player=False, id=0, team=0):
        self.name = name
        self.id = id
        self.is_player = is_player
        self.team = team  # 0 or 1 for 5v5 combat

        self.archetype_key = archetype_key
        stats = ARCHETYPES.get(archetype_key, ARCHETYPES["default"])

        self.max_hp = stats["hp"]
        self.hp = float(self.max_hp)

        self.max_posture = stats["pos"]
        self.posture = int(self.max_posture)

        self.max_focus = stats["max_foc"]
        self.focus = stats["foc"]

        self.dist = MID
        self.scan_timer = 0  # 3 rounds visibility

        self.status = {
            "Airborne": 0,
            "Frozen": 0,
            "Slowed": 0,
            "Brittle": 0,
            "Stunned": 0,
            "CursedPain": 0,
            "Shield": False,
            "MaxShield": False,
        }

    def can_cast(self, spell_name):
        cost = SPELL_BOOK[spell_name]["cost"]
        if self.focus < cost:
            return False

        # v6.1 airborne restriction
        if self.status["Airborne"] > 0:
            allowed = ["Info", "Control", "Defense"]
            if SPELL_BOOK[spell_name]["type"] not in allowed:
                return False

        if self.status["Frozen"] > 0 or self.status["Stunned"] > 0:
            return False

        return True

    def take_hit(self, hp_dmg, pos_dmg, ignore_mitigation=False):
        if self.status["Brittle"] > 0 and hp_dmg > 0:
            hp_dmg *= 2
            self.status["Brittle"] = 0

        actual_hp_dmg = hp_dmg

        if not ignore_mitigation and self.posture > 0 and hp_dmg > 0:
            actual_hp_dmg = hp_dmg * 0.5

        self.hp = max(0, self.hp - actual_hp_dmg)
        self.posture = max(MIN_POSTURE, self.posture - pos_dmg)

        return actual_hp_dmg

    def start_round_regen(self, move_name):
        if move_name not in ["Protego", "Protego Maximus"]:
            self.focus = min(self.max_focus, self.focus + FOCUS_REGEN)

    def end_round_tick(self, took_damage):
        # Airborne no longer auto-clears - must use Descendo to remove

        if self.status["Frozen"] > 0:
            self.status["Frozen"] = 0

        # Slowed clear moved after regent check to ensure it suppresses regen


        if self.status["Stunned"] > 0:
            self.status["Stunned"] -= 1

        if self.scan_timer > 0:
            self.scan_timer -= 1

        if not took_damage and self.status["Slowed"] == 0:
            self.posture = min(self.max_posture, self.posture + POSTURE_REGEN)

        # Clear Slowed after regen check
        if self.status["Slowed"] > 0:
            self.status["Slowed"] = 0

        if self.status["CursedPain"] > 0:
            tick = 15 if self.status["CursedPain"] >= 3 else (10 if self.status["CursedPain"] == 2 else 7)
            self.hp = max(0, self.hp - tick)
            self.status["CursedPain"] -= 1


# ==========================
# HEURISTIC AI
# ==========================
def get_heuristic_move(enemy, player, all_enemies):
    for ally in all_enemies:
        if ally.id != enemy.id and ally.hp > 0 and ally.status["Airborne"] > 0:
            return "Descendo"

    if enemy.hp < 30 or enemy.posture <= 0:
        return "Protego"

    if player.status["Airborne"] > 0:
        return "Descendo"

    if player.status["Frozen"] > 0:
        return "Diffindo"

    key = enemy.archetype_key

    if key == "auror":
        low_hp = any(e.hp < 50 for e in all_enemies if e.hp > 0)
        if low_hp and enemy.focus >= 30:
            return "Protego Maximus"
        return "Accio"

    if key == "curse_specialist":
        return "Avada Kedavra" if player.hp < 40 else "Crucio"

    if key == "death_eater":
        return "Levioso"

    return "Confringo" if enemy.focus > 40 else "Basic Cast"


# ==========================
# ROUND RESOLUTION
# ==========================
def resolve_round(
    player,
    enemies,
    player_move,
    player_target_id,
    ai_moves_override=None,
    ai_targets_override=None,  # NEW
):
    logs = []
    living_enemies = [e for e in enemies if e.hp > 0]

    # took_damage tracking (for posture regen correctness)
    took_damage = {player: False}
    for e in living_enemies:
        took_damage[e] = False

    def mark_damage(victim, hp_dmg, pos_dmg):
        if hp_dmg > 0 or pos_dmg > 0:
            took_damage[victim] = True

    # Start-of-round focus regen
    player.start_round_regen(player_move)

    enemy_intents = {}
    enemy_targets = {}

    id_to_enemy = {e.id: e for e in living_enemies}
    player_id = getattr(player, "id", 99)

    # Determine enemy intents + targets
    for e in living_enemies:
        move = ai_moves_override.get(e.id) if ai_moves_override else get_heuristic_move(e, player, living_enemies)
        e.start_round_regen(move)
        enemy_intents[e.id] = move

        # Default target logic (v6.1)
        tgt = player
        if move == "Descendo":
            airborne = [a for a in living_enemies if a.status["Airborne"] > 0]
            if airborne:
                tgt = airborne[0]

        # Override target if provided
        if ai_targets_override and e.id in ai_targets_override:
            tid = ai_targets_override[e.id]
            if tid == player_id:
                tgt = player
            elif tid in id_to_enemy:
                tgt = id_to_enemy[tid]

        enemy_targets[e.id] = tgt

    # Pay player cost
    p_cost = SPELL_BOOK[player_move]["cost"]
    player.focus -= p_cost

    # Validate enemy moves and pay their costs
    valid_enemy_moves = {}
    for e in living_enemies:
        move = enemy_intents[e.id]
        if move and e.can_cast(move):
            e.focus -= SPELL_BOOK[move]["cost"]
            valid_enemy_moves[e.id] = move
            logs.append(f"{e.name} casts {move}!")
        else:
            valid_enemy_moves[e.id] = None

    # Helper: Get actions by priority with Frozen check
    def get_actions(prio):
        acts = []

        # Player action
        if SPELL_BOOK[player_move]["prio"] == prio and player.status["Airborne"] == 0:
            if player.status.get("Frozen", 0) == 0:
                p_tgt_obj = next((e for e in living_enemies if e.id == player_target_id), None)
                acts.append((player, player_move, p_tgt_obj))
            else:
                logs.append("Player is Frozen! Action skipped.")

        # Enemy actions
        for e in living_enemies:
            m = valid_enemy_moves[e.id]
            if m and SPELL_BOOK[m]["prio"] == prio and e.status["Airborne"] == 0:
                if e.status.get("Frozen", 0) == 0:
                    acts.append((e, m, enemy_targets[e.id]))
                else:
                    logs.append(f"{e.name} is Frozen! Action skipped.")

        return acts

    # --- PHASE 1: DEFENSE ---
    if player_move == "Protego":
        player.status["Shield"] = True
        player.status["ShieldTargetID"] = player_target_id
    elif player_move == "Protego Maximus":
        player.status["MaxShield"] = True

    for e in living_enemies:
        m = valid_enemy_moves[e.id]
        if m == "Protego":
            e.status["Shield"] = True
            e.status["ShieldTargetID"] = enemy_targets[e.id].id if enemy_targets[e.id] else None
        elif m == "Protego Maximus":
            # Symmetric: everyone gets MaxShield (no GlobalShield)
            e.status["MaxShield"] = True

    # --- PHASE 2: INFO ---
    for c, m, t in get_actions(2):
        if not t:
            continue

        if m == "Revelio":
            # Symmetric: reveals enemy team for all casters
            enemies_of_caster = [w for w in living_enemies if w.team != c.team]
            for e in enemies_of_caster:
                if e.archetype_key != "occlumens":
                    e.scan_timer = 3
            logs.append(f"{c.name}'s Revelio reveals enemy stats for 3 rounds.")

        elif m == "Legilimens":
            if t.archetype_key == "occlumens":
                t.take_hit(0, 20)
                mark_damage(t, 0, 20)
                logs.append("Legilimens BLOCKED! Occlumens takes mental dmg.")
            else:
                t.scan_timer = 3
                logs.append(f"Mind Read: {t.name} HP:{int(t.hp)} Pos:{t.posture}")

    # --- PHASE 3: CONTROL (Reflect Logic) ---
    for c, m, t in get_actions(3):
        if not t:
            continue

        final_t = t

        if t.status["MaxShield"]:
             logs.append(f"{t.name} (MaxShield) BLOCKS {m}!")
             continue

        # Protego: Directional Check
        if t.status["Shield"]:
             # Only block if caster is the Shield Target
             shield_target_id = t.status.get("ShieldTargetID")
             if getattr(c, 'id', 999) == shield_target_id:
                logs.append(f"{t.name} REFLECTS {m}!")
                t.status["Shield"] = False  # Break shield

                if t.is_player:
                    if living_enemies:
                        final_t = random.choice(living_enemies)
                else:
                    final_t = player

                # Check if reflected target has MaxShield
                if final_t.status["MaxShield"]:
                    logs.append(f"Reflected {m} blocked by MaxShield!")
                    continue
                
                # Check directional shield on reflected target
                if final_t.status["Shield"]:
                     # If final_t is shielding against t (the reflector), block it
                     if final_t.status.get("ShieldTargetID") == getattr(t, 'id', 999):
                         logs.append(f"Reflected {m} blocked by second shield!")
                         final_t.status["Shield"] = False # Break second shield too
                         continue
             else:
                # Shield doesn't block this direction
                pass

        if m == "Levioso":
            final_t.status["Airborne"] = 1
            logs.append(f"{final_t.name} is Levitated!")

        elif m == "Glacius":
            final_t.status["Frozen"] = 1
            final_t.status["Brittle"] = 1
            logs.append(f"{final_t.name} is Frozen!")

        elif m == "Arresto Momentum":
            final_t.status["Slowed"] = 1
            logs.append(f"{final_t.name} is Slowed!")

    # --- PHASE 4: FORCE (Reflect Logic) ---
    for c, m, t in get_actions(4):
        if not t:
            continue

        final_t = t

        if t.status["MaxShield"]:
             logs.append(f"{t.name} (MaxShield) BLOCKS {m}!")
             continue

        if t.status["Shield"]:
             # Protego only reflects CONTROL. For Force/Damage it BLOCKS.
             # Wait, logic says logic says "Reflects Control spells back". 
             # Does it reflect Force? Typically Force is blocked or ignored. 
             # Logic table: "Protego: Blocks incoming damage. Reflects Control spells".
             # So Force is just blocked? Or passes through? Usually blocked. 
             # Let's assume BLOCKS Force.
             
             shield_target_id = t.status.get("ShieldTargetID")
             if getattr(c, 'id', 999) == shield_target_id:
                logs.append(f"{t.name} BLOCKS {m}!")
                t.status["Shield"] = False # Break shield
                continue

        if m == "Accio":
            final_t.dist = CLOSE

        elif m == "Descendo":
            if final_t.status["Airborne"] > 0:
                final_t.status["Airborne"] = 0

                # Symmetric: same team (self/ally) = no damage save; enemy = slam
                if c.team != final_t.team:
                    final_t.take_hit(10, 20)
                    mark_damage(final_t, 10, 20)
                    logs.append(f"{final_t.name} Slammed!")
                else:
                    logs.append(f"{c.name} saves {final_t.name}!")
            else:
                # Descendo on non-airborne enemy still deals damage
                if c.team != final_t.team:
                    final_t.take_hit(10, 20)
                    mark_damage(final_t, 10, 20)

        elif m == "Depulso":
            # Symmetric: pushes target back (not self or all enemies)
            if final_t.dist < FAR:
                final_t.dist += 1
                logs.append(f"{final_t.name} pushed back!")

    # --- PHASE 5: DAMAGE ---

    for c, m, t in get_actions(5):
        if not t:
            continue

        if m == "Basic Cast":
            c.posture = min(c.max_posture, c.posture + 5)

        targets = []

        if m == "Incendio":
            # Symmetric: AoE 30 dmg to all close enemies of caster's team
            all_combatants = [player] + living_enemies
            close_enemies = [x for x in all_combatants if x.team != c.team and x.hp > 0 and x.dist == CLOSE]
            if close_enemies:
                for x in close_enemies:
                    targets.append((x, 30))
            elif t:
                targets.append((t, 20 if t.dist == MID else 15))

        elif m == "Avada Kedavra":
            dmg = 9999 if t.posture <= 0 else 40
            targets.append((t, dmg))
            # Symmetric: chain damage to all enemies with CursedPain
            all_combatants = [player] + living_enemies
            for x in all_combatants:
                if x.id != t.id and x.team != c.team and x.status["CursedPain"] > 0:
                    targets.append((x, 40))

        else:
            bonus = 10 if m == "Confringo" and ((c.is_player and t.dist == FAR) or (not c.is_player and c.dist == FAR)) else 0
            targets.append((t, SPELL_BOOK[m].get("hp_dmg", 0) + bonus))

        for victim, dmg in targets:
            if victim is None or victim.hp <= 0:
                continue

            if victim.status["MaxShield"]:
                victim.focus = min(victim.max_focus, victim.focus + dmg)
                logs.append(f"{victim.name} absorbs {dmg} dmg to Focus!")
                continue

            if victim.status["Shield"]:
                shield_target_id = victim.status.get("ShieldTargetID")
                if getattr(c, 'id', 999) == shield_target_id:
                     logs.append(f"{victim.name} Blocks!")
                     victim.status["Shield"] = False # Break shield
                     continue

            pos_dmg = SPELL_BOOK[m].get("pos_dmg", 0)

            if m == "Basic Cast":
                if victim.posture - pos_dmg < 10:
                    pos_dmg = 0

            if m == "Crucio":
                victim.status["Stunned"] = 2
                victim.status["CursedPain"] += 3

            victim.take_hit(dmg, pos_dmg)
            mark_damage(victim, dmg, pos_dmg)

    # --- Cleanup / end tick ---
    player.end_round_tick(took_damage[player])

    for e in living_enemies:
        e.end_round_tick(took_damage[e])
        e.status["Shield"] = False
        e.status["MaxShield"] = False
        if "ShieldTargetID" in e.status:
            del e.status["ShieldTargetID"]

    player.status["Shield"] = False
    player.status["MaxShield"] = False
    if "ShieldTargetID" in player.status:
        del player.status["ShieldTargetID"]

    return logs


# ==========================
# 5V5 ROUND RESOLUTION
# ==========================

def resolve_round_5v5(all_wizards, actions):
    """
    Resolve a 5v5 turn with full spell logic.
    
    Args:
        all_wizards: List[Wizard] - All wizards (team_a + team_b)
        actions: Dict[int, (spell_name, target_id)] - Wizard id -> (spell, target_id)
    
    Returns:
        logs: List[str] - Turn event logs (caster, spell, target, damage, effects)
    """
    logs = []
    took_damage = {w: False for w in all_wizards}
    id_to_wizard = {w.id: w for w in all_wizards}
    
    def mark_damage(victim, hp_dmg, pos_dmg):
        if hp_dmg > 0 or pos_dmg > 0:
            took_damage[victim] = True
    
    # Start-of-round focus regen for all
    for wiz in all_wizards:
        if wiz.hp > 0 and wiz.id in actions:
            spell_name = actions[wiz.id][0]
            wiz.start_round_regen(spell_name)
    
    # Validate and pay spell costs
    valid_actions = {}
    for wiz_id, (spell_name, target_id) in actions.items():
        wiz = id_to_wizard.get(wiz_id)
        if wiz and wiz.hp > 0 and wiz.can_cast(spell_name):
            cost = SPELL_BOOK[spell_name]["cost"]
            wiz.focus -= cost
            valid_actions[wiz_id] = (spell_name, target_id)
        else:
            # Fall back to Basic Cast if can't cast
            if wiz and wiz.hp > 0:
                valid_actions[wiz_id] = ("Basic Cast", target_id)
    
    def get_actions_by_prio(prio):
        """Get all actions for a given priority."""
        result = []
        for wiz_id, (spell_name, target_id) in valid_actions.items():
            if SPELL_BOOK[spell_name]["prio"] == prio:
                caster = id_to_wizard[wiz_id]
                target = id_to_wizard.get(target_id)
                
                # Check Airborne and Frozen
                if caster.hp > 0 and caster.status.get("Airborne", 0) == 0:
                    if caster.status.get("Frozen", 0) > 0:
                        # Frozen skips action
                        logs.append((caster.id, spell_name, None, 0, 0, "❄️ Skips action (Frozen)"))
                        continue
                    
                    result.append((caster, spell_name, target))
        return result
    
    def is_shielded(target, attacker_id, spell_type="Damage"):
        """
        Check if target has active shield against attacker.
        Returns: (blocked, reflected, absorbs)
        """
        if not target:
            return False, False, False
            
        if target.status.get("MaxShield"):
            # Protego Maximus: Absorbs Damage, Blocks Control/Force?
            # User said "Absorbs damage as Focus". 
            # Logic: "Protego Maximus ... Absorbs damage as Focus instead of HP."
            # It blocks everything. (Absorbing damage implies block).
            return True, False, True

        if target.status.get("Shield"):
            # Protego: Directional
            shield_target = target.status.get("ShieldTargetID")
            if shield_target == attacker_id:
                # Matched direction.
                # Reflect only Control.
                if spell_type == "Control":
                    return True, True, False
                else:
                    return True, False, False
        
        return False, False, False
    
    # --- PHASE 1: DEFENSE (Priority 1) ---
    for caster, spell, target in get_actions_by_prio(1):
        if spell == "Protego":
            caster.status["Shield"] = True
            # Store target ID for directional blocking
            # The 'target' obj matches actions[caster.id][1]
            if target:
                caster.status["ShieldTargetID"] = target.id
            logs.append((caster.id, spell, target.id if target else None, 0, 0, "🛡️ Shield up (Directional)"))
        elif spell == "Protego Maximus":
            caster.status["MaxShield"] = True
            logs.append((caster.id, spell, caster.id, 0, 0, "🛡️ Max Shield up"))
    
    # --- PHASE 2: INFO (Priority 2) ---
    for caster, spell, target in get_actions_by_prio(2):
        if spell == "Revelio":
            # Reveal all enemies for 3 turns
            for w in all_wizards:
                if w.team != caster.team and w.hp > 0:
                    w.scan_timer = 3
            logs.append((caster.id, spell, None, 0, 0, "👁️ Revealed enemies"))
        elif spell == "Legilimens":
            if target and target.hp > 0:
                target.scan_timer = 5
                logs.append((caster.id, spell, target.id, 0, 0, "🧠 Mind read"))
    
    # --- PHASE 3: CONTROL (Priority 3) ---
    for caster, spell, target in get_actions_by_prio(3):
        if not target or target.hp <= 0:
            continue
            
        blocked, reflected, absorbs = is_shielded(target, caster.id, "Control")
        
        if blocked:
            if reflected:
                logs.append((caster.id, spell, target.id, 0, 0, "REFLECTED by Shield!"))
                target.status["Shield"] = False # Break shield
                
                # Reflect back to caster
                final_t = caster
                # Check if caster has shield against reflector?
                # Reflector is 'target'. Caster is 'caster'.
                b2, r2, a2 = is_shielded(final_t, target.id, "Control")
                
                if b2:
                     logs.append((target.id, "Reflect", final_t.id, 0, 0, "Reflect BLOCKED by Shield!"))
                     if not a2: # If not MaxShield, it breaks Protego
                         final_t.status["Shield"] = False
                     continue
                
                # Apply Control to Caster
                pos_dmg = float(SPELL_BOOK[spell].get("pos_dmg", 0))
                final_t.take_hit(0.0, pos_dmg)
                mark_damage(final_t, 0.0, pos_dmg)
                
                if spell == "Levioso":
                    final_t.status["Airborne"] = 2
                    logs.append((target.id, "Reflect", final_t.id, 0, pos_dmg, "[AIRBORNE] (Reflected)"))
                elif spell == "Glacius":
                    final_t.status["Frozen"] = 2
                    final_t.status["Brittle"] = 1
                    logs.append((target.id, "Reflect", final_t.id, 0, pos_dmg, "❄️ Frozen (Reflected)"))
                elif spell == "Arresto Momentum":
                    final_t.status["Slowed"] = 2
                    logs.append((target.id, "Reflect", final_t.id, 0, pos_dmg, "⏸️ Slowed (Reflected)"))
                
            else:
                logs.append((caster.id, spell, target.id, 0, 0, "BLOCKED by Shield"))
                if not absorbs:
                    target.status["Shield"] = False # Break Protego
            continue
        
        pos_dmg = float(SPELL_BOOK[spell].get("pos_dmg", 0))
        # Control spells deal direct posture damage but should trigger take_hit for consistency?
        # Actually, take_hit() reduces HP based on posture. Control spells usually do 0 HP dmg.
        # But if they did HP dmg, take_hit would handle it.
        # For pure posture damage (Levioso 20 pos), take_hit(0, 20) is fine.
        # It also marks took_damage correctly.
        target.take_hit(0.0, pos_dmg)
        mark_damage(target, 0.0, pos_dmg)
        
        if spell == "Levioso":
            target.status["Airborne"] = 2
            logs.append((caster.id, spell, target.id, 0, pos_dmg, "[AIRBORNE]"))
        elif spell == "Glacius":
            target.status["Frozen"] = 2
            target.status["Brittle"] = 1
            logs.append((caster.id, spell, target.id, 0, pos_dmg, "❄️ Frozen + Brittle"))
        elif spell == "Arresto Momentum":
            target.status["Slowed"] = 2
            logs.append((caster.id, spell, target.id, 0, pos_dmg, "⏸️ Slowed"))
    
    # --- PHASE 4: FORCE (Priority 4) ---
    for caster, spell, target in get_actions_by_prio(4):
        if not target or target.hp <= 0:
            continue
            
        blocked, reflected, absorbs = is_shielded(target, caster.id, "Force")
        if blocked:
            logs.append((caster.id, spell, target.id, 0, 0, "BLOCKED by Shield"))
            if not absorbs:
                target.status["Shield"] = False
            continue
        
        hp_dmg = float(SPELL_BOOK[spell].get("hp_dmg", 0))
        pos_dmg = float(SPELL_BOOK[spell].get("pos_dmg", 0))
        effect = ""
        
        if spell == "Accio":
            target.status["Airborne"] = 2
            target.status["Airborne"] = 2
            target.dist = CLOSE
            effect = "Pulled to CLOSE + [AIRBORNE]"
        elif spell == "Depulso":
            if target.dist < FAR:
                target.dist += 1
            effect = "Pushed"
        elif spell == "Descendo":
            if target.status.get("Airborne", 0) > 0:
                target.status["Airborne"] = 0
                
                if target.team == caster.team:
                    # Ally Save
                    hp_dmg = 0
                    pos_dmg = 0
                    effect = "🙌 Saved Ally!"
                else:
                    # Enemy Slam (Bonus: +10 HP, +20 Pos)
                    hp_dmg += 10
                    pos_dmg += 20
                    effect = "⬇️ SLAM! (+Bonus)"
            else:
                effect = "⬇️"
        
        # Check for Brittle (for UI effect string only - take_hit handles logic)
        if target.status.get("Brittle", 0) > 0:
            effect += " [Brittle 2x!]"
        
        real_dmg = target.take_hit(hp_dmg, pos_dmg)
        mark_damage(target, real_dmg, pos_dmg)
        logs.append((caster.id, spell, target.id, real_dmg, pos_dmg, effect))
    
    # --- PHASE 5: DAMAGE & CURSE (Priority 5) ---
    for caster, spell, target in get_actions_by_prio(5):
        if not target or target.hp <= 0:
            continue
            
        blocked, reflected, absorbs = is_shielded(target, caster.id, "Damage")
        
        if blocked:
            # Protego Maximus absorbs to Focus
            if absorbs:
                hp_dmg = float(SPELL_BOOK[spell].get("hp_dmg", 0))
                # Absorb: Add to Focus
                target.focus = min(target.max_focus, target.focus + hp_dmg)
                logs.append((caster.id, spell, target.id, 0, 0, f"🛡️ Absorbed {int(hp_dmg)} to Focus"))
            else:
                # Regular Protego Block
                logs.append((caster.id, spell, target.id, 0, 0, "BLOCKED by Shield"))
                target.status["Shield"] = False
            continue
        
        hp_dmg = float(SPELL_BOOK[spell].get("hp_dmg", 0))
        pos_dmg = float(SPELL_BOOK[spell].get("pos_dmg", 0))
        effect = ""
        
        if spell == "Basic Cast":
            caster.posture = min(caster.max_posture, caster.posture + 5)
            effect = ""
        elif spell == "Incendio":
            # AoE fire damage to all close enemies
            hit_in_aoe = False
            for w in all_wizards:
                if w.team != caster.team and w.hp > 0 and w.dist == CLOSE:
                    d = w.take_hit(30.0, 5.0) # Incendio base pos_dmg is 5
                    mark_damage(w, d, 5.0)
                    if w.id == target.id:
                        hit_in_aoe = True
            
            if not hit_in_aoe:
                if target.dist == MID:
                    hp_dmg = 20.0
                elif target.dist == FAR:
                    hp_dmg = 15.0
            
            effect = "🔥 Fire"
        elif spell == "Confringo":
            # Bonus +10 damage at FAR range
            # Check distance between caster and target. 
            # In 1vMany, "dist" is range from center/player?
            # Assumption: "dist" is relative to "center".
            # If Caster is Player, Target Dist FAR -> Bonus.
            # If Caster is Enemy, Caster Dist FAR -> Bonus (attacking player).
            # But 5v5 is symmetric?
            # 5v5 Logic: Simply if `target.dist == FAR`? 
            # Or distance between them? "Bonus +10 damage at FAR range".
            # Let's assume if Target is FAR from Caster.
            # But we only track `dist` which is abstract.
            # Let's Use Logic L135/L61 Incendio/Confringo.
            # L61: "Bonus +10 damage at FAR range".
            # If Caster is at FAR? Or Target is at FAR?
            # Legacy L431: "Confringo ... ((c.is_player and t.dist == FAR) or (not c.is_player and c.dist == FAR))"
            # This implies "Long Range Shot".
            # In 5v5:
            # If Caster is `curse_specialist` (usually FAR), they get bonus?
            # Let's say if Target is FAR. (Hardest to hit).
            # Wait, Legacy says: Player attacking FAR target = Bonus. Enemy attacking from FAR = Bonus.
            # So if Distance(Caster, Target) is Large?
            # But we don't have relative positions easily.
            # Let's assume: If `target.dist == FAR` (for Player casting) OR `caster.dist == FAR` (for Enemy casting on Player)?
            # In 5v5, `dist` might be "Distance from Center/Frontline".
            # Keep it simple: If Caster.dist == FAR or Target.dist == FAR ?
            # Let's use: If Target.dist == FAR.
            
            bonus = 10.0 if target.dist == FAR else 0.0
            hp_dmg += bonus
            effect = "💥 Explode" + (" (Range Bonus)" if bonus > 0 else "")
        elif spell == "Crucio":
            target.status["Stunned"] = 2
            target.status["CursedPain"] = target.status.get("CursedPain", 0) + 3
            effect = "💀 CursedPain (DoT 15→10→7)"
        elif spell == "Avada Kedavra":
            if target.posture <= 0:
                hp_dmg = 9999  # Instant kill if posture broken
                effect = "💀 INSTANT KILL!"
            else:
                effect = "💀"
            # Chain to all with CursedPain
            for w in all_wizards:
                if w.team != caster.team and w.id != target.id and w.status.get("CursedPain", 0) > 0:
                    d = w.take_hit(40.0, 0.0)
                    mark_damage(w, d, 0.0)
        
        # Check for Brittle (UI only)
        if target.status.get("Brittle", 0) > 0 and hp_dmg > 0:
            effect += " [Brittle 2x!]"
        
        real_dmg = target.take_hit(hp_dmg, pos_dmg)
        mark_damage(target, real_dmg, pos_dmg)
        logs.append((caster.id, spell, target.id, real_dmg, pos_dmg, effect))
    
    # --- END OF ROUND TICK ---
    for wiz in all_wizards:
        if wiz.hp > 0:
            wiz.end_round_tick(took_damage.get(wiz, False))
            wiz.status["Shield"] = False
            wiz.status["MaxShield"] = False
            if "ShieldTargetID" in wiz.status:
                del wiz.status["ShieldTargetID"]
    
    return logs

