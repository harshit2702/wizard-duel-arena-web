"""
game_session.py — Full rewrite for the new frontend.

Changes from original:
- ASCII art served per wizard via avatars_2.py
- Animation events include hp_dmg, pos_dmg, effect, caster_name, target_name
- Ally AI model selection (player's teammates can use different model from enemies)
- Fog of war: enemy stats hidden unless revealed by info spells
- Winner detection: "Team A" or "Team B" for frontend to show victory/defeat
"""

import sys
import os
import re
import random
import copy
from collections import defaultdict
import numpy as np
import torch

# Add parent directory to path so we can import duel_engine etc.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from duel_engine import Wizard, SPELL_BOOK, SPELL_LIST, resolve_round_5v5, ARCHETYPES
    from unified_brain_v2 import UnifiedBrainV2, get_state_vector_5v5, get_valid_spell_mask, ROLE_STRIKER
    from legacy_brain_v2 import LegacyBrainV2, get_legacy_state_5v5
    HAS_BRAINS = True
except ImportError as e:
    print(f"Warning: Brain modules or duel_engine not found: {e}")
    HAS_BRAINS = False
    if 'SPELL_LIST' not in dir():
        SPELL_LIST = ["Basic Cast"]
    if 'SPELL_BOOK' not in dir():
        SPELL_BOOK = {"Basic Cast": {"type": "Damage", "cost": 0}}

# Import avatars for ASCII art
try:
    from avatars_2 import AVATARS
except ImportError:
    try:
        from avatars import AVATARS
    except ImportError:
        AVATARS = {"default": ["(O_O)", "/|\\", "/ \\"]}


def strip_ansi(text):
    """Remove ANSI escape codes from text."""
    return re.sub(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])', '', text)


def get_ascii_art(avatar_id, is_team_a=True, hp_pct=100, status=None, hit_spell=None):
    """Get ASCII art for a wizard, cleaned of ANSI codes and colorama formatting."""
    base = avatar_id or "default"
    suffix = ""

    if status:
        if status.get("Airborne", 0) > 0:
            suffix = "_levioso_lifted"
        elif status.get("CursedPain", 0) > 0:
            suffix = "_crucio_torture"
        elif hp_pct <= 0:
            suffix = "_surrender_on_knees"
        elif hp_pct <= 30:
            suffix = "_30hp_torn_breathing"
        elif hp_pct <= 50:
            suffix = "_50hp_worried"

    if hit_spell:
        s = hit_spell.lower()
        if "avada" in s:
            suffix = "_avada_kedavra"
        elif "incendio" in s or "confringo" in s:
            suffix = "_incendio_burned"
        elif "descendo" in s:
            suffix = "_descendo_impact"
        elif "depulso" in s:
            suffix = "_depulso_staggered"
        elif "diffindo" in s:
            suffix = "_diffindo_slash"

    if not is_team_a:
        key = f"{base}_flipped{suffix}"
    else:
        key = f"{base}{suffix}"

    art = AVATARS.get(key)
    if not art:
        if not is_team_a:
            art = AVATARS.get(f"{base}_flipped", AVATARS.get("default_flipped", AVATARS.get("default", ["?"])))
        else:
            art = AVATARS.get(base, AVATARS.get("default", ["?"]))

    # Clean ANSI codes and {c} placeholders
    cleaned = []
    for line in (art or ["?"]):
        line = strip_ansi(str(line))
        line = line.replace("{c}", " ")
        cleaned.append(line)

    return "\n".join(cleaned[:8])  # Max 8 lines


class GameSession:
    def __init__(self, session_id):
        self.session_id = session_id
        self.team_a = []
        self.team_b = []
        self.brains_a = []
        self.brains_b = []
        self.model_a = "random"
        self.model_b = "random"
        self.ally_ai_config = None  # For player's teammates
        self.turn = 1
        self.max_turns = 15
        self.logs = []
        self.animation_queue = []
        self.game_over = False
        self.winner = None
        self.player_position = 0  # Which slot is the player (0-indexed)

    def setup_game(self, config):
        self.logs.append("Initializing Duel...")

        # ── TEAM A ──
        conf_a = config.get("team_a", {})
        size_a = conf_a.get("size", 1)
        self.model_a = conf_a.get("control", "player")
        avatars_a = conf_a.get("avatars", [])
        variant_a = conf_a.get("variant") or {}
        ally_ai = conf_a.get("ally_ai") or {}
        self.player_position = max(0, variant_a.get("player_position", 1) - 1)

        for i in range(size_a):
            is_player_slot = (self.model_a == "player" and i == self.player_position)

            if is_player_slot and avatars_a:
                av = avatars_a[0]
                name = av.get("name", "Player")
                archetype = av.get("archetype", "default")
                avatar_id = av.get("avatar_id", "harry")
            else:
                name = f"Ally {i+1}"
                archetype = "default"
                avatar_id = avatars_a[0].get("avatar_id", "harry") if avatars_a else "harry"

            w = Wizard(name, archetype, is_player=is_player_slot, id=i, team=0)
            w.avatar_id = avatar_id
            self.team_a.append(w)

        # Load brains for Team A
        # Player slot gets no brain, teammates get ally_ai model
        self.brains_a = [None] * size_a
        if ally_ai and ally_ai.get("control") and ally_ai["control"] != "random":
            teammate_brains = self._load_brains(
                ally_ai["control"], size_a,
                ally_ai.get("variant")
            )
            for i in range(size_a):
                if not self.team_a[i].is_player:
                    self.brains_a[i] = teammate_brains[i] if i < len(teammate_brains) else None

        # ── TEAM B ──
        conf_b = config.get("team_b", {})
        size_b = conf_b.get("size", 1)
        self.model_b = conf_b.get("control", "random")
        avatars_b = conf_b.get("avatars", [])
        variant_b = conf_b.get("variant")
        
        enemy_names = ["Draco", "Bellatrix", "Lucius", "Dolohov", "Greyback"]
        for i in range(size_b):
            av_data = avatars_b[i] if i < len(avatars_b) else {}
            name = enemy_names[i % len(enemy_names)] if av_data.get("name") == "Enemy" else av_data.get("name", f"Enemy {i+1}")
            archetype = av_data.get("archetype", "default")
            avatar_id = av_data.get("avatar_id", "voldemort")

            w = Wizard(name, archetype, is_player=False, id=i + 10, team=1)
            w.avatar_id = avatar_id
            self.team_b.append(w)

        self.brains_b = self._load_brains(self.model_b, size_b, variant_b)
        self.logs.append(f"Team A ({size_a}) vs Team B ({size_b}) — Fight!")

    def _load_brains(self, model_type, team_size, variant_info):
        brains = [None] * team_size
        if not HAS_BRAINS:
            return brains

        checkpoint_dir = "checkpoints_5v5"
        if variant_info and isinstance(variant_info, dict) and "checkpoint_dir" in variant_info:
            checkpoint_dir = variant_info["checkpoint_dir"]

        if model_type == "unified":
            if variant_info and variant_info.get("single_file"):
                brain = UnifiedBrainV2()
                path = os.path.join(checkpoint_dir, variant_info.get("file_pattern", "unified_best_0.pth"))
                if os.path.exists(path):
                    brain.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
                    print(f"  ✓ Loaded shared model: {path}")
                else:
                    print(f"  ✗ Not found: {path}")
                brains = [brain] * team_size
            else:
                pattern = variant_info.get("file_pattern", "unified_best_{i}.pth") if variant_info else "unified_best_{i}.pth"
                brains = []
                for i in range(team_size):
                    brain = UnifiedBrainV2()
                    path = os.path.join(checkpoint_dir, pattern.format(i=i % 5))
                    if os.path.exists(path):
                        brain.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
                    brains.append(brain)
                print(f"  ✓ Loaded {team_size} models from {checkpoint_dir}")

        elif model_type == "legacy":
            brains = []
            for i in range(team_size):
                brain = LegacyBrainV2()
                path = os.path.join(checkpoint_dir, f"legacy_best_{i % 5}.pth")
                if os.path.exists(path):
                    brain.load_state_dict(torch.load(path, map_location='cpu', weights_only=True))
                brains.append(brain)

        return brains

    def get_state(self, player_perspective=True):
        state = {
            "turn": self.turn,
            "max_turns": self.max_turns,
            "game_over": self.game_over,
            "winner": self.winner,
            "logs": self.logs[-10:],
            "text_logs": [],
            "team_a": [self._serialize_wizard(w, reveal=True, is_ally=True) for w in self.team_a],
            "team_b": [],
            "animation_events": self.animation_queue
        }

        self.animation_queue = []

        for w in self.team_b:
            is_revealed = (w.scan_timer > 0) or w.hp <= 0 or not player_perspective
            state["team_b"].append(self._serialize_wizard(w, reveal=is_revealed, is_ally=False))

        return state

    def _serialize_wizard(self, w, reveal=False, is_ally=True):
        hp_pct = (w.hp / w.max_hp * 100) if w.max_hp > 0 else 0
        ascii_art = get_ascii_art(
            getattr(w, 'avatar_id', 'default'),
            is_team_a=is_ally,
            hp_pct=hp_pct,
            status=dict(w.status) if hasattr(w, 'status') else {}
        )

        data = {
            "id": w.id,
            "name": w.name,
            "avatar_id": getattr(w, "avatar_id", "default"),
            "ascii_art": ascii_art,
            "max_hp": w.max_hp,
            "team": w.team,
            "is_player": w.is_player,
        }

        if reveal:
            data.update({
                "hp": round(w.hp, 1),
                "max_posture": w.max_posture,
                "posture": w.posture,
                "max_focus": w.max_focus,
                "focus": w.focus,
                "dist": w.dist,
                "status": {k: v for k, v in w.status.items() if k not in ("ShieldTargetID",)},
                "is_revealed": True,
                "scan_timer": w.scan_timer,
            })
        else:
            data.update({
                "hp": "???",
                "max_posture": "???",
                "posture": "???",
                "max_focus": "???",
                "focus": "???",
                "dist": w.dist,
                "status": {k: v for k, v in w.status.items()
                           if k in ("Airborne", "Frozen", "Stunned", "Shield", "MaxShield")},
                "is_revealed": False,
                "scan_timer": 0,
            })

        return data

    def process_turn(self, player_action=None):
        if self.game_over:
            return

        alive_a = [w for w in self.team_a if w.hp > 0]
        alive_b = [w for w in self.team_b if w.hp > 0]

        if not alive_a or not alive_b:
            self.game_over = True
            self.winner = "Team A" if alive_a else "Team B"
            self.logs.append(f"Game Over! {self.winner} wins!")
            return

        actions = {}
        turn_frac = self.turn / self.max_turns

        # ── Team A actions ──
        for idx, w in enumerate(self.team_a):
            if w.hp <= 0:
                continue

            if w.is_player and player_action:
                actions[w.id] = player_action
            else:
                brain = self.brains_a[idx] if idx < len(self.brains_a) else None
                act_idx = self._get_ai_move(brain, w, self.team_a, self.team_b, turn_frac)
                spell = SPELL_LIST[act_idx]
                target = self._get_ai_target(w, spell, self.team_a, self.team_b)
                actions[w.id] = (spell, target)

        # ── Team B actions ──
        for idx, w in enumerate(self.team_b):
            if w.hp <= 0:
                continue
            brain = self.brains_b[idx] if idx < len(self.brains_b) else None
            act_idx = self._get_ai_move(brain, w, self.team_b, self.team_a, turn_frac)
            spell = SPELL_LIST[act_idx]
            target = self._get_ai_target(w, spell, self.team_b, self.team_a)
            actions[w.id] = (spell, target)

        # ── Resolve round ──
        turn_logs = resolve_round_5v5(self.team_a + self.team_b, actions)

        # ── Process logs into animation events ──
        for log_entry in turn_logs:
            if isinstance(log_entry, tuple) and len(log_entry) >= 6:
                caster_id, spell, target_id, hp_dmg, pos_dmg, effect = log_entry

                self.animation_queue.append({
                    "type": "cast",
                    "caster_id": caster_id,
                    "target_id": target_id,
                    "spell": spell,
                    "hp_dmg": round(float(hp_dmg), 1) if hp_dmg else 0,
                    "pos_dmg": round(float(pos_dmg), 1) if pos_dmg else 0,
                    "effect": str(effect),
                    "caster_name": self._get_name(caster_id),
                    "target_name": self._get_name(target_id) if target_id is not None else "",
                    "color_type": self._get_spell_color_type(spell),
                })

                text = f"{self._get_name(caster_id)} > {spell}"
                if target_id is not None:
                    text += f" on {self._get_name(target_id)}"
                text += f": {effect}"
                self.logs.append(text)
            else:
                self.logs.append(str(log_entry))

        # ── Advance turn ──
        self.turn += 1

        # ── Check win conditions ──
        alive_a = [w for w in self.team_a if w.hp > 0]
        alive_b = [w for w in self.team_b if w.hp > 0]

        if not alive_a or not alive_b:
            self.game_over = True
            if alive_a:
                self.winner = "Team A"
            elif alive_b:
                self.winner = "Team B"
            else:
                self.winner = "Draw"
            self.logs.append(f"Game Over! {self.winner} wins!")

        if self.turn > self.max_turns and not self.game_over:
            self.game_over = True
            # Compare remaining HP
            hp_a = sum(w.hp for w in self.team_a if w.hp > 0)
            hp_b = sum(w.hp for w in self.team_b if w.hp > 0)
            if hp_a > hp_b:
                self.winner = "Team A"
            elif hp_b > hp_a:
                self.winner = "Team B"
            else:
                self.winner = "Draw (Max Turns)"
            self.logs.append(f"Max turns reached! {self.winner} wins by HP advantage!")

    def _get_name(self, wid):
        for w in self.team_a + self.team_b:
            if w.id == wid:
                return w.name
        return "?"

    def _get_spell_color_type(self, spell):
        data = SPELL_BOOK.get(spell, {})
        return data.get("type", "Unknown")

    def _get_ai_move(self, brain, wizard, team, enemies, turn_frac):
        if not brain:
            try:
                mask = get_valid_spell_mask(wizard)
                valid = np.where(mask > 0)[0]
                return random.choice(valid) if len(valid) > 0 else 0
            except Exception:
                return 0

        try:
            if isinstance(brain, UnifiedBrainV2):
                state = get_state_vector_5v5(wizard, team, enemies)
                mask = get_valid_spell_mask(wizard)
                return brain.get_action(state, ROLE_STRIKER, turn_frac, mask)
            elif isinstance(brain, LegacyBrainV2):
                state = get_legacy_state_5v5(wizard, team, enemies)
                mask = get_valid_spell_mask(wizard)
                return brain.get_action(state, mask)
        except Exception as e:
            print(f"AI error: {e}")
        return 0

    def _get_ai_target(self, wizard, spell, team, enemies):
        # Descendo: check for airborne allies first
        if spell == "Descendo":
            for ally in team:
                if ally.id != wizard.id and ally.hp > 0 and ally.status.get("Airborne", 0) > 0:
                    return ally.id
            # Then airborne enemies
            for e in enemies:
                if e.hp > 0 and e.status.get("Airborne", 0) > 0:
                    return e.id

        # Revelio: target self (no target needed)
        if spell == "Revelio":
            return wizard.id

        valid_enemies = [e for e in enemies if e.hp > 0]
        if not valid_enemies:
            return wizard.id  # fallback

        # Prefer low HP for damage spells
        spell_type = SPELL_BOOK.get(spell, {}).get("type", "")
        if spell_type in ("Damage", "Curse"):
            best = min(valid_enemies, key=lambda e: e.hp)
            return best.id

        # Control: target highest HP
        if spell_type == "Control":
            best = max(valid_enemies, key=lambda e: e.hp)
            return best.id

        return random.choice(valid_enemies).id
