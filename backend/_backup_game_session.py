import sys
import os
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
except ImportError:
    print("Warning: Brain modules or duel_engine not found. AI may not work.")
    HAS_BRAINS = False
    # Mocking for testing if missing
    if 'SPELL_LIST' not in locals():
        SPELL_LIST = ["Basic Cast"]
    if 'SPELL_BOOK' not in locals():
        SPELL_BOOK = {"Basic Cast": {"type": "Damage", "cost": 0}}

class GameSession:
    def __init__(self, session_id):
        self.session_id = session_id
        self.team_a = []
        self.team_b = []
        self.brains_a = []
        self.brains_b = []
        self.model_a = "random"
        self.model_b = "random"
        self.turn = 1
        self.max_turns = 15
        self.logs = []
        self.animation_queue = [] # Queue of events for frontend to animate
        self.game_over = False
        self.winner = None
        
        # Fog of War state
        # Map of Wizard ID -> Scan Timer (how long they are revealed to the OTHER team)
        # If A1 casts Revelio, all B wizards get scan_timer = 3
        # Logic is handled in duel_engine, but we need to read it to mask state.
        # Actually, duel_engine's Wizard class has .scan_timer! 
        # "e.scan_timer = 3" in resolve_round. 
        # So we just need to check that property.

    def setup_game(self, config):
        """
        Config dict should contain:
        {
            "team_a": {
                "size": int,
                "control": "player" | "unified" | "legacy" | "random",
                "variant": dict (optional),
                "avatars": list of dicts { "name": str, "archetype": str, "avatar_id": str }
            },
            "team_b": { ... }
        }
        """
        self.logs.append("Initializing Duel...")
        
        # Setup Team A
        conf_a = config.get("team_a", {})
        size_a = conf_a.get("size", 1)
        self.model_a = conf_a.get("control", "player")
        avatars_a = conf_a.get("avatars", [])
        variant_a = conf_a.get("variant") or {}
        player_pos = variant_a.get("player_position", 1) - 1 # 0-indexed
        
        for i in range(size_a):
            # Default values if avatar info missing
            av_data = avatars_a[i] if i < len(avatars_a) else {}
            
            is_player_slot = (self.model_a == "player" and i == player_pos)
            
            name = av_data.get("name", f"A{i+1}")
            # If player slot, use first avatar from list (or specific logic)
            # Actually frontend sends list of 1 avatar? 
            # If creating team of 5, we need 5 avatars OR default for teammates.
            
            if is_player_slot:
                # Use the provided avatar config for the player
                # Provided avatars list might just have 1 entry for player?
                # Let's assume frontend sends 1 avatar for player, rest defaults.
                if avatars_a:
                     # Use the first avatar config for the player character
                     av = avatars_a[0]
                     name = av.get("name", "Player")
                     archetype = av.get("archetype", "default")
                     avatar_id = av.get("avatar_id", "harry")
                else:
                     archetype = "default"
                     avatar_id = "harry"
            else:
                # Teammate
                name = f"A{i+1}"
                archetype = "default" 
                avatar_id = "default"

            w = Wizard(name, archetype, is_player=is_player_slot, id=i, team=0)
            w.avatar_id = avatar_id
            self.team_a.append(w)
            
        self.brains_a = self._load_brains(self.model_a, size_a, variant_a)

        # Setup Team B
        conf_b = config.get("team_b", {})
        size_b = conf_b.get("size", 1)
        self.model_b = conf_b.get("control", "random")
        avatars_b = conf_b.get("avatars", [])
        
        for i in range(size_b):
            av_data = avatars_b[i] if i < len(avatars_b) else {}
            name = av_data.get("name", f"B{i+1}")
            archetype = av_data.get("archetype", "death_eater") # Default enemy
            
            w = Wizard(name, archetype, is_player=False, id=i+10, team=1)
            w.avatar_id = av_data.get("avatar_id", "voldemort")
            self.team_b.append(w)
            
        self.brains_b = self._load_brains(self.model_b, size_b, conf_b.get("variant"))
        
        self.logs.append(f"Team A ({size_a}) vs Team B ({size_b}) ready!")

    def _load_brains(self, model_type, team_size, variant_info):
        """Reused logic from visual_duel_v8 to load brains."""
        brains = [None] * team_size
        if not HAS_BRAINS:
            return brains
            
        checkpoint_dir = "checkpoints_5v5"
        if variant_info and "checkpoint_dir" in variant_info:
            checkpoint_dir = variant_info["checkpoint_dir"]
            
        if model_type == "unified":
            if variant_info and variant_info.get("single_file"):
                # Shared model
                brain = UnifiedBrainV2()
                path = os.path.join(checkpoint_dir, variant_info["file_pattern"])
                if os.path.exists(path):
                    brain.load_state_dict(torch.load(path, weights_only=True))
                brains = [brain] * team_size
            else:
                # Per-slot models
                pattern = variant_info["file_pattern"] if variant_info else "unified_best_{i}.pth"
                brains = []
                for i in range(team_size):
                    brain = UnifiedBrainV2()
                    path = os.path.join(checkpoint_dir, pattern.format(i=i%5))
                    if os.path.exists(path):
                        brain.load_state_dict(torch.load(path, weights_only=True))
                    brains.append(brain)
                    
        elif model_type == "legacy":
            brains = []
            for i in range(team_size):
                brain = LegacyBrainV2()
                path = os.path.join(checkpoint_dir, f"legacy_best_{i%5}.pth")
                if os.path.exists(path):
                    brain.load_state_dict(torch.load(path, weights_only=True))
                brains.append(brain)
                
        return brains

    def get_state(self, player_perspective=True):
        """
        Returns the game state dict. 
        If player_perspective is True, masks enemy stats if not revealed.
        """
        state = {
            "turn": self.turn,
            "max_turns": self.max_turns,
            "game_over": self.game_over,
            "winner": self.winner,
            "logs": self.logs[-10:], # Last 10 logs
            "team_a": [self._serialize_wizard(w, reveal=True) for w in self.team_a],
            "team_b": [],
            "animation_events": self.animation_queue # Send events to animate
        }
        
        # Clear animation queue after sending (assuming polling consumes them)
        # Or we keep them with an event ID? For now, clear 'em.
        self.animation_queue = []

        for w in self.team_b:
            # Check if revealed
            # In duel_engine, scan_timer > 0 means "I can see you" basically?
            # Actually: "e.scan_timer = 3" means 'e' is revealed.
            is_revealed = (w.scan_timer > 0) or w.hp <= 0 or not player_perspective
            state["team_b"].append(self._serialize_wizard(w, reveal=is_revealed))
            
        return state

    def _serialize_wizard(self, w, reveal=False):
        """Convert Wizard object to dict."""
        data = {
            "id": w.id,
            "name": w.name,
            "avatar_id": getattr(w, "avatar_id", "default"),
            "max_hp": w.max_hp,
            "team": w.team
        }
        
        if reveal:
            data.update({
                "hp": w.hp,
                "max_posture": w.max_posture,
                "posture": w.posture,
                "max_focus": w.max_focus,
                "focus": w.focus,
                "dist": w.dist,
                "status": dict(w.status),
                "is_revealed": True
            })
        else:
            data.update({
                "hp": "???",
                "max_posture": "???",
                "posture": "???",
                "max_focus": "???",
                "focus": "???",
                "dist": w.dist, # Distance is always visible? 
                # Maybe vague distance? logic uses Close/Mid/Far. Let's show it.
                "status": {k:v for k,v in w.status.items() if k in ["Airborne", "Frozen"]}, # Only visible status?
                # Actually, status effects like Stunned/Frozen are visible physically. 
                # Internal stats like Focus/Posture are hidden.
                # Let's show all status for now, mask HP/Pos/Foc.
                "is_revealed": False
            })
            
        return data

    def process_turn(self, player_action=None):
        """
        player_action: tuple (spell_name, target_id) provided by frontend
        """
        if self.game_over:
            return

        actions = {}
        alive_a = [w for w in self.team_a if w.hp > 0]
        alive_b = [w for w in self.team_b if w.hp > 0]
        alive_all = alive_a + alive_b
        
        if not alive_a or not alive_b:
            self.game_over = True
            self.winner = "Team A" if alive_a else "Team B"
            self.logs.append(f"Game Over! {self.winner} wins!")
            return

        # 1. Collect AI Moves (Team A teammates + Team B)
        turn_frac = self.turn / self.max_turns
        
        # Team A
        for idx, w in enumerate(self.team_a):
            if w.hp <= 0: continue
            
            if w.is_player and self.model_a == "player":
                # Use provided player action
                if player_action and idx == 0: # Assuming Player is always A1 (index 0)
                    actions[w.id] = player_action
                else:
                    # Teammate AI (if any) or Default to Basic Cast
                    # For now, let's assume A1 is player, A2-A5 are AI if configured
                    if idx < len(self.brains_a) and self.brains_a[idx]:
                        act_idx = self._get_ai_move(self.brains_a[idx], w, self.team_a, self.team_b, turn_frac)
                        spell = SPELL_LIST[act_idx]
                        # Heuristic target: simple valid target
                        target = self._get_ai_target(w, spell, self.team_a, self.team_b)
                        actions[w.id] = (spell, target)
                    else:
                        # Fallback for player teammates without brains
                        actions[w.id] = ("Basic Cast", self.team_b[0].id if self.team_b else 0)
            else:
                # Full AI Team A
                brain = self.brains_a[idx] if idx < len(self.brains_a) else None
                act_idx = self._get_ai_move(brain, w, self.team_a, self.team_b, turn_frac)
                spell = SPELL_LIST[act_idx]
                target = self._get_ai_target(w, spell, self.team_a, self.team_b)
                actions[w.id] = (spell, target)

        # Team B (Always AI)
        for idx, w in enumerate(self.team_b):
            if w.hp <= 0: continue
            brain = self.brains_b[idx] if idx < len(self.brains_b) else None
            act_idx = self._get_ai_move(brain, w, self.team_b, self.team_a, turn_frac) # Note enemies param swapped
            spell = SPELL_LIST[act_idx]
            target = self._get_ai_target(w, spell, self.team_b, self.team_a)
            actions[w.id] = (spell, target)

        # 2. Resolve Round
        turn_logs = resolve_round_5v5(self.team_a + self.team_b, actions)
        
        # 3. Process logs into animation events and text
        for log_entry in turn_logs:
            # resolve_round_5v5 returns tuples: (caster_id, spell, target_id, hp_dmg, pos_dmg, effect_str)
            if isinstance(log_entry, tuple):
                caster_id, spell, target_id, hp, pos, effect = log_entry
                
                # Create animation event
                self.animation_queue.append({
                    "type": "cast",
                    "caster_id": caster_id,
                    "target_id": target_id,
                    "spell": spell,
                    "color": self._get_spell_color_type(spell)
                })
                
                # Add text log
                caster_name = self._get_name(caster_id)
                target_name = self._get_name(target_id) if target_id else "None"
                self.logs.append(f"{caster_name} > {spell} on {target_name}: {effect}")
            else:
                self.logs.append(str(log_entry))

        self.turn += 1
        if self.turn > self.max_turns:
            self.game_over = True
            self.winner = "Draw (Max Turns)"
            self.logs.append("Max turns reached!")

    def _get_name(self, wid):
        for w in self.team_a + self.team_b:
            if w.id == wid: return w.name
        return "?"

    def _get_spell_color_type(self, spell):
        # Return logical type for frontend coloring
        data = SPELL_BOOK.get(spell, {})
        return data.get("type", "Unknown")

    def _get_ai_move(self, brain, wizard, team, enemies, turn_frac):
        if not brain:
             # Random valid
             mask = get_valid_spell_mask(wizard)
             valid = np.where(mask > 0)[0]
             return random.choice(valid) if len(valid)>0 else 0
             
        if isinstance(brain, UnifiedBrainV2):
             state = get_state_vector_5v5(wizard, team, enemies)
             mask = get_valid_spell_mask(wizard)
             return brain.get_action(state, ROLE_STRIKER, turn_frac, mask)
        elif isinstance(brain, LegacyBrainV2):
             state = get_legacy_state_5v5(wizard, team, enemies)
             mask = get_valid_spell_mask(wizard)
             return brain.get_action(state, mask)
        return 0

    def _get_ai_target(self, wizard, spell, team, enemies):
        # Simple heuristic or logic to pick target ID
        # UnifiedBrain V2 doesn't output target yet, logic is inside environment or heuristic
        # We'll use a simple heuristic for now similar to duel_engine's heuristic
        
        # Descendo -> Airborne
        if spell == "Descendo":
            for e in enemies:
                if e.hp > 0 and e.status["Airborne"] > 0:
                    return e.id
        
        # Default: Lowest HP enemy or Random
        valid_enemies = [e for e in enemies if e.hp > 0]
        if not valid_enemies: return None
        
        # Prefer low HP for damage
        if SPELL_BOOK[spell]["type"] == "Damage":
            best = min(valid_enemies, key=lambda e: e.hp)
            return best.id
            
        return random.choice(valid_enemies).id

