"""
Visual Duel V8.0 - Team Battle Visualizer with Trained Models

Features:
- Team size selection (2-5 per side)
- Model type selection (Unified V2, Legacy V2, Random)
- Animated spell beams and avatars
- Audio support (if pygame available)
- Spell usage tracking for evaluation

Usage:
    python visual_duel_v8.py
"""

import os
import sys
import time
import random
import re
import glob
import numpy as np
import torch
from colorama import Fore, Style, init
import copy
from collections import defaultdict

# --- IMPORTS ---
try:
    from avatars_2 import AVATARS
except ImportError:
    try:
        from avatars import AVATARS
    except ImportError:
        AVATARS = {
            "default": ["(O_O)", "/|\\", "/ \\"],
            "default_flipped": ["(O_O)", "/|\\", "/ \\"]
        }

try:
    from duel_engine import Wizard, SPELL_BOOK, SPELL_LIST, resolve_round_5v5
except ImportError:
    print(f"{Fore.RED}CRITICAL ERROR: duel_engine.py missing.{Style.RESET_ALL}")
    exit()

try:
    from unified_brain_v2 import UnifiedBrainV2, get_state_vector_5v5, get_valid_spell_mask
    from unified_brain_v2 import ROLE_STRIKER, NUM_SPELLS
    from legacy_brain_v2 import LegacyBrainV2, get_legacy_state_5v5
    HAS_BRAINS = True
except ImportError:
    print(f"{Fore.YELLOW}WARNING: Brain modules not found. Using random AI.{Style.RESET_ALL}")
    HAS_BRAINS = False

init(autoreset=True)


# =========================
# AUDIO MANAGER
# =========================

class SoundManager:
    def __init__(self):
        self.enabled = False
        try:
            import pygame
            pygame.mixer.init()
            self.enabled = True
            self.sfx = {
                "cast": self._load("assets/cast_attack.mp3"),
                "hit": self._load("assets/hit_damage.mp3"),
                "win": self._load("assets/victory_string_1.mp3"),
                "loss": self._load("assets/defeat_string_1.mp3"),
            }
        except:
            pass

    def _load(self, path):
        import pygame
        if self.enabled and os.path.exists(path):
            return pygame.mixer.Sound(path)
        return None

    def play(self, key):
        if self.enabled and self.sfx.get(key):
            self.sfx[key].play()


audio = SoundManager()


# =========================
# VISUAL HELPERS
# =========================

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def get_clean_len(text):
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return len(ansi_escape.sub('', text))


def draw_bar(current, maximum, length=10, color=Fore.GREEN):
    pct = max(0, min(1, current / maximum))
    filled = int(pct * length)
    bar = color + "█" * filled + Fore.BLACK + Style.BRIGHT + "░" * (length - filled) + Style.RESET_ALL
    return f"{bar} {int(current)}"


def pad_str(text, width, align='left'):
    clean_len = get_clean_len(text)
    padding = max(0, width - clean_len)
    if align == 'center':
        left = padding // 2
        right = padding - left
        return (" " * left) + text + (" " * right)
    elif align == 'right':
        return (" " * padding) + text
    return text + (" " * padding)


def get_avatar_lines(wizard, hit_spell=None, is_team_a=True):
    """Get avatar art for wizard."""
    base_keys = ["harry", "voldemort", "snape", "bellatrix", "kingsley", "flitwick", "witch", "hagrid", "default"]
    base = base_keys[wizard.id % len(base_keys)]
    
    suffix = ""
    hp_pct = (wizard.hp / wizard.max_hp) * 100 if wizard.max_hp > 0 else 0
    
    if wizard.status.get("Airborne", 0) > 0:
        suffix = "_levioso_lifted"
    elif wizard.status.get("CursedPain", 0) > 0:
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
    
    # Flipped for enemies (team B)
    if not is_team_a:
        key = f"{base}_flipped{suffix}"
    else:
        key = f"{base}{suffix}"
    
    # Try with suffix, then without
    art = AVATARS.get(key)
    if not art:
        if not is_team_a:
            art = AVATARS.get(f"{base}_flipped", AVATARS.get("default_flipped", ["?", "?", "?"]))
        else:
            art = AVATARS.get(base, AVATARS.get("default", ["?", "?", "?"]))
    
    return art[:6]  # Limit height
def get_spell_color(spell_name):
    """Get color for a spell based on its type."""
    if spell_name not in SPELL_BOOK:
        return Fore.WHITE
    
    s_type = SPELL_BOOK[spell_name].get("type", "Unknown")
    
    if s_type == "Damage":
        return Fore.RED
    elif s_type == "Control":
        return Fore.CYAN
    elif s_type == "Force":
        return Fore.YELLOW
    elif s_type == "Defense":
        return Fore.GREEN
    elif s_type == "Info":
        return Fore.BLUE
    elif s_type == "Curse":
        return Fore.MAGENTA + Style.BRIGHT
    
    return Fore.WHITE

# =========================
# TEAM SETUP
# =========================

def create_wizard(name, team_id, wiz_id, is_team_a=True):
    """Create a wizard for the team using proper archetype values."""
    w = Wizard(name, "default", is_player=is_team_a, id=wiz_id)
    # Wizard class initializes from ARCHETYPES["default"]:
    # hp=100, posture=50, focus=100, max_focus=150
    # Don't override - just ensure status is a defaultdict
    w.status = defaultdict(int)
    return w


def load_brains(model_type: str, team_size: int, checkpoint_dir: str = "checkpoints_5v5",
                variant_info: dict = None):
    """Load AI brains based on model type and variant.
    
    variant_info keys:
        file_pattern: str - e.g. 'unified_best_{i}.pth' or 'mappo_best.pth'
        single_file: bool - True if one model is shared across all slots
        checkpoint_dir: str - overrides the checkpoint_dir parameter
        label: str - human-readable name for display
    """
    brains = []
    
    if variant_info and variant_info.get("checkpoint_dir"):
        checkpoint_dir = variant_info["checkpoint_dir"]
    
    if model_type == "unified" and HAS_BRAINS:
        if variant_info and variant_info.get("single_file"):
            # Single model shared across all slots (MAPPO, PBT, Imitation)
            brain = UnifiedBrainV2()
            path = os.path.join(checkpoint_dir, variant_info["file_pattern"])
            if os.path.exists(path):
                brain.load_state_dict(torch.load(path, weights_only=True))
                print(f"  ✓ Loaded {path}")
            else:
                print(f"  ✗ Not found: {path} (using random weights)")
            brains = [brain] * team_size
        else:
            # Per-slot models (Evo)
            pattern = variant_info["file_pattern"] if variant_info else "unified_best_{i}.pth"
            for i in range(team_size):
                brain = UnifiedBrainV2()
                path = os.path.join(checkpoint_dir, pattern.format(i=i % 5))
                if os.path.exists(path):
                    brain.load_state_dict(torch.load(path, weights_only=True))
                else:
                    print(f"  ✗ Not found: {path}")
                brains.append(brain)
            print(f"  ✓ Loaded {team_size} models from {checkpoint_dir}")
    elif model_type == "legacy" and HAS_BRAINS:
        for i in range(team_size):
            brain = LegacyBrainV2()
            path = os.path.join(checkpoint_dir, f"legacy_best_{i % 5}.pth")
            if os.path.exists(path):
                brain.load_state_dict(torch.load(path, weights_only=True))
            brains.append(brain)
    else:
        brains = [None] * team_size  # Random action fallback
    
    return brains


def get_brain_action(brain, model_type, wizard, team, enemies, turn_frac=0.5):
    """Get action from brain."""
    if brain is None or not HAS_BRAINS:
        # Random action
        mask = get_valid_spell_mask(wizard) if HAS_BRAINS else np.ones(16)
        valid = np.where(mask > 0)[0]
        return random.choice(valid) if len(valid) > 0 else 0
    

    if model_type in ["unified", "dqn"]:
        state = get_state_vector_5v5(wizard, team, enemies)
        mask = get_valid_spell_mask(wizard)
        return brain.get_action(state, ROLE_STRIKER, turn_frac, mask)
    else:  # legacy
        state = get_legacy_state_5v5(wizard, team, enemies)
        mask = get_valid_spell_mask(wizard)
        return brain.get_action(state, mask)


# =========================
# MAIN GAME CLASS
# =========================

class VisualDuelV8:
    def __init__(self):
        self.team_a = []
        self.team_b = []
        self.brains_a = []
        self.brains_b = []
        self.model_a = "random"
        self.model_b = "random"
        self.variant_a = None  # dict with variant details
        self.variant_b = None
        self.label_a = "RANDOM"  # display name
        self.label_b = "RANDOM"
        self.turn = 1
        self.max_turns = 15
        self.spell_counts_a = defaultdict(int)
        self.spell_counts_b = defaultdict(int)

    def setup(self):
        """Interactive setup for teams."""
        clear_screen()
        print(f"{Fore.YELLOW + Style.BRIGHT}╔══════════════════════════════════════════════════════════╗{Style.RESET_ALL}")
        print(f"{Fore.YELLOW + Style.BRIGHT}║     WIZARD DUEL v8.0 - TEAM BATTLE VISUALIZER            ║{Style.RESET_ALL}")
        print(f"{Fore.YELLOW + Style.BRIGHT}╚══════════════════════════════════════════════════════════╝{Style.RESET_ALL}")
        
        # Team A setup
        print(f"\n{Fore.GREEN}═══ TEAM A (Left Side) ═══{Style.RESET_ALL}")
        size_a = self._get_team_size("Team A")
        self.model_a = self._get_model_type("Team A", allow_player=True)
        if self.model_a == "unified":
            self.variant_a = self._get_unified_variant("Team A")
            self.label_a = self.variant_a["label"]
        elif self.model_a == "legacy":
            self.label_a = "LEGACY V2"
        elif self.model_a == "player":
            self.label_a = "PLAYER"
        else:
            self.label_a = "RANDOM"
        
        for i in range(size_a):
            self.team_a.append(create_wizard(f"A{i+1}", 0, i, is_team_a=True))
        
        if self.model_a == "player":
            # Player controls A1, teammates use AI
            if size_a > 1:
                print(f"\n{Fore.CYAN}You control A1. Choose AI for your teammates (A2-A{size_a}):{Style.RESET_ALL}")
                self.teammate_ai = self._get_model_type("Teammate", allow_player=False)
                teammate_variant = None
                if self.teammate_ai == "unified":
                    teammate_variant = self._get_unified_variant("Teammate")
                self.brains_a = load_brains(self.teammate_ai, size_a, variant_info=teammate_variant)
            else:
                self.teammate_ai = "player"
                self.brains_a = [None]
        else:
            self.teammate_ai = self.model_a
            self.brains_a = load_brains(self.model_a, size_a, variant_info=self.variant_a)
        
        # Team B setup
        print(f"\n{Fore.RED}═══ TEAM B (Right Side) ═══{Style.RESET_ALL}")
        size_b = self._get_team_size("Team B")
        self.model_b = self._get_model_type("Team B")
        if self.model_b == "unified":
            self.variant_b = self._get_unified_variant("Team B")
            self.label_b = self.variant_b["label"]
        elif self.model_b == "legacy":
            self.label_b = "LEGACY V2"
        else:
            self.label_b = "RANDOM"
        
        for i in range(size_b):
            self.team_b.append(create_wizard(f"B{i+1}", 1, i + 10, is_team_a=False))
        self.brains_b = load_brains(self.model_b, size_b, variant_info=self.variant_b)
        
        print(f"\n{Fore.CYAN}Setup complete! Press Enter to start battle...{Style.RESET_ALL}")
        input()

    def _get_team_size(self, team_name: str) -> int:
        """Get team size from user."""
        while True:
            try:
                size = input(f"{team_name} size (2-5): ").strip()
                size = int(size)
                if 2 <= size <= 5:
                    return size
                print(f"{Fore.RED}Please enter a number between 2 and 5.{Style.RESET_ALL}")
            except ValueError:
                print(f"{Fore.RED}Invalid input. Please enter a number.{Style.RESET_ALL}")

    def _get_model_type(self, team_name: str, allow_player: bool = False) -> str:
        """Get model type from user."""
        print(f"\n{team_name} control type:")
        if allow_player:
            print("  0. PLAYER (you control)")
        print("  1. Unified V2 (trained) →  pick variant next")
        print("  2. Legacy V2 (trained)")
        print("  3. Random (baseline)")
        
        while True:
            choice = input(f"Select ({0 if allow_player else 1}-3): ").strip()
            if choice == "0" and allow_player:
                return "player"
            elif choice == "1":
                return "unified"
            elif choice == "2":
                return "legacy"
            elif choice == "3":
                return "random"
            print(f"{Fore.RED}Invalid choice.{Style.RESET_ALL}")

    def _get_unified_variant(self, team_name: str) -> dict:
        """Select which unified model variant to use."""
        print(f"\n{Fore.CYAN + Style.BRIGHT}═══ {team_name} Unified Model Variant ═══{Style.RESET_ALL}")
        print(f"  1. Evo Best          (checkpoints_evo_gpu)")
        print(f"  2. Evo Generation    (pick gen 10-100)")
        print(f"  3. MAPPO Best        (checkpoints_mappo_gpu)")
        print(f"  4. MAPPO Iteration   (pick iter)")
        print(f"  5. PBT Best          (checkpoints_pbt_gpu)")

        print(f"  6. Imitation Student (checkpoints_imitation_gpu)")
        print(f"  7. Original Evo Best (checkpoints_5v5)")
        print(f"  8. New DQN Best      (checkpoints_dqn_gpu)")
        
        while True:
            choice = input(f"Select variant (1-8): ").strip()
            
            if choice == "1":
                return {"checkpoint_dir": "checkpoints_evo_gpu",
                        "file_pattern": "unified_best_{i}.pth",
                        "single_file": False, "label": "EVO BEST"}
            
            elif choice == "2":
                # Find available generations
                gens = sorted(set(
                    int(f.split("_gen")[1].split("_")[0])
                    for f in os.listdir("checkpoints_evo_gpu")
                    if f.startswith("unified_gen") and f.endswith(".pth")
                ))
                if not gens:
                    print(f"{Fore.RED}No generation checkpoints found.{Style.RESET_ALL}")
                    continue
                print(f"  Available gens: {gens}")
                gen = input(f"  Pick generation: ").strip()
                if gen.isdigit() and int(gen) in gens:
                    return {"checkpoint_dir": "checkpoints_evo_gpu",
                            "file_pattern": f"unified_gen{gen}_" + "{i}.pth",
                            "single_file": False, "label": f"EVO GEN {gen}"}
                print(f"{Fore.RED}Invalid generation.{Style.RESET_ALL}")
                continue
            
            elif choice == "3":
                return {"checkpoint_dir": "checkpoints_mappo_gpu",
                        "file_pattern": "mappo_best.pth",
                        "single_file": True, "label": "MAPPO BEST"}
            
            elif choice == "4":
                iters = sorted(set(
                    int(f.replace("mappo_iter", "").replace(".pth", ""))
                    for f in os.listdir("checkpoints_mappo_gpu")
                    if f.startswith("mappo_iter") and f.endswith(".pth")
                ))
                if not iters:
                    print(f"{Fore.RED}No MAPPO iteration checkpoints found.{Style.RESET_ALL}")
                    continue
                print(f"  Available iters: {iters}")
                it = input(f"  Pick iteration: ").strip()
                if it.isdigit() and int(it) in iters:
                    return {"checkpoint_dir": "checkpoints_mappo_gpu",
                            "file_pattern": f"mappo_iter{it}.pth",
                            "single_file": True, "label": f"MAPPO ITER {it}"}
                print(f"{Fore.RED}Invalid iteration.{Style.RESET_ALL}")
                continue
            
            elif choice == "5":
                return {"checkpoint_dir": "checkpoints_pbt_gpu",
                        "file_pattern": "pbt_best.pth",
                        "single_file": True, "label": "PBT BEST"}
            
            elif choice == "6":
                # Show available imitation files
                files = [f for f in os.listdir("checkpoints_imitation_gpu")
                         if f.endswith(".pth")]
                if not files:
                    print(f"{Fore.RED}No imitation checkpoints found.{Style.RESET_ALL}")
                    continue
                print(f"  Available: {sorted(files)}")
                print(f"  a. student_best.pth")
                print(f"  b. student_final.pth")
                print(f"  c. imitation_phase1.pth (pre-finetune)")
                sub = input(f"  Pick (a/b/c): ").strip().lower()
                fmap = {"a": "student_best.pth", "b": "student_final.pth",
                        "c": "imitation_phase1.pth"}
                if sub in fmap and fmap[sub] in files:
                    lbl = {"a": "IMIT BEST", "b": "IMIT FINAL", "c": "IMIT PHASE1"}[sub]
                    return {"checkpoint_dir": "checkpoints_imitation_gpu",
                            "file_pattern": fmap[sub],
                            "single_file": True, "label": lbl}
                print(f"{Fore.RED}Invalid choice.{Style.RESET_ALL}")
                continue
            

            elif choice == "7":
                return {"checkpoint_dir": "checkpoints_5v5",
                        "file_pattern": "unified_best_{i}.pth",
                        "single_file": False, "label": "ORIGINAL EVO"}
            
            elif choice == "8":
                return {"checkpoint_dir": "checkpoints_dqn_gpu",
                        "file_pattern": "dqn_best.pth",
                        "single_file": True, "label": "DQN BEST"}
            
            print(f"{Fore.RED}Invalid choice. Pick 1-8.{Style.RESET_ALL}")

    def render_battlefield(self, beam_info=None):
        """Render the battlefield."""
        clear_screen()
        
        print(f"{Fore.YELLOW}════════ TURN {self.turn}/{self.max_turns} ════════{Style.RESET_ALL}")
        
        # Show team info with player status
        if self.model_a == "player":
            team_a_label = f"[YOU: A1 + {self.teammate_ai.upper()} AI]"
        else:
            team_a_label = f"[{self.label_a}]"
        
        print(f"{Fore.GREEN}{team_a_label} Team A{Style.RESET_ALL}" + " " * 20 +
              f"{Fore.RED}Team B [{self.label_b}]{Style.RESET_ALL}")
        print("=" * 80)
        
        # Determine max rows to display
        max_wizards = max(len(self.team_a), len(self.team_b))
        
        for row in range(max_wizards):
            wiz_a = self.team_a[row] if row < len(self.team_a) else None
            wiz_b = self.team_b[row] if row < len(self.team_b) else None
            
            self._render_wizard_row(wiz_a, wiz_b, beam_info, row)
            print()
        
        print("=" * 80)

    def _render_wizard_row(self, wiz_a, wiz_b, beam_info, row_idx):
        """Render a single wizard row (A vs B)."""
        left_lines = []
        right_lines = []
        
        # Team A wizard
        if wiz_a and wiz_a.hp > 0:
            hit_a = None
            if beam_info and beam_info.get("source") == "b" and beam_info.get("target_idx") == row_idx and beam_info.get("frame") == 5:
                hit_a = beam_info.get("spell")
            
            # Player indicator for A1
            player_marker = "★ " if (self.model_a == "player" and row_idx == 0) else ""
            
            left_lines = [
                f"{Fore.GREEN}{player_marker}[A{row_idx+1}] HP:  {draw_bar(wiz_a.hp, wiz_a.max_hp, 8, Fore.GREEN)}{Style.RESET_ALL}",
                f"      FOC: {draw_bar(wiz_a.focus, wiz_a.max_focus, 8, Fore.CYAN)}",
                f"      POS: {draw_bar(wiz_a.posture, wiz_a.max_posture, 8, Fore.YELLOW)}",
            ]
            status = [k for k, v in wiz_a.status.items() if v > 0]
            if status:
                left_lines.append(f"{Fore.MAGENTA}[{','.join(status)}]{Style.RESET_ALL}")
            left_lines.extend(get_avatar_lines(wiz_a, hit_a, is_team_a=True))
        elif wiz_a:
            left_lines = [f"{Fore.BLACK}[A{row_idx+1}] DEFEATED{Style.RESET_ALL}"]
        
        # Team B wizard
        if wiz_b and wiz_b.hp > 0:
            hit_b = None
            if beam_info and beam_info.get("source") == "a" and beam_info.get("target_idx") == row_idx and beam_info.get("frame") == 5:
                hit_b = beam_info.get("spell")
            
            # Distance indicator
            dist_names = ["Close", "Mid", "Far"]
            dist = dist_names[min(wiz_b.dist, 2)] if hasattr(wiz_b, 'dist') else "Mid"
            
            right_lines = [
                f"{Fore.RED}[B{row_idx+1}] [{dist}] HP: {draw_bar(wiz_b.hp, wiz_b.max_hp, 8, Fore.RED)}{Style.RESET_ALL}",
                f"            POS: {draw_bar(wiz_b.posture, wiz_b.max_posture, 8, Fore.YELLOW)}",
            ]
            status = [k for k, v in wiz_b.status.items() if v > 0]
            if status:
                right_lines.append(f"{Fore.MAGENTA}[{','.join(status)}]{Style.RESET_ALL}")
            right_lines.extend(get_avatar_lines(wiz_b, hit_b, is_team_a=False))
        elif wiz_b:
            right_lines = [f"{Fore.BLACK}[B{row_idx+1}] DEFEATED{Style.RESET_ALL}"]
        
        # Build beam for this row
        beam_str = ""
        if beam_info and beam_info.get("active"):
            f = beam_info.get("frame", 0)
            color = beam_info.get("color", Fore.CYAN if beam_info.get("source") == "a" else Fore.RED)
            if beam_info.get("row") == row_idx:
                if beam_info.get("source") == "a":
                    beam_str = f"{color}{'═' * (f * 4)}⚡{Style.RESET_ALL}"
                else:
                    beam_str = f"{color}⚡{'═' * (f * 4)}{Style.RESET_ALL}"
        
        # Print row
        max_h = max(len(left_lines), len(right_lines), 1)
        for i in range(max_h):
            left = left_lines[i] if i < len(left_lines) else ""
            right = right_lines[i] if i < len(right_lines) else ""
            mid = beam_str if i == max_h // 2 else ""
            print(f"{pad_str(left, 35)}{pad_str(mid, 20, 'center')}{right}")

    def animate_attack(self, source_team, attacker_idx, target_idx, spell_name, color=None):
        """Animate an attack."""
        if color is None:
            color = Fore.CYAN if source_team == "a" else Fore.RED
            
        for frame in range(6):
            beam_info = {
                "active": True,
                "source": source_team,
                "row": attacker_idx,
                "target_idx": target_idx,
                "spell": spell_name,
                "frame": frame,
                "color": color
            }
            self.render_battlefield(beam_info)
            time.sleep(0.06)
        
        if audio.enabled:
            audio.play("hit")

    def show_spell_menu(self):
        """Display the spell menu with costs and effects (v7 style)."""
        # Get player focus if available
        player_focus = self.team_a[0].focus if self.team_a else 50
        
        print(f"\n{Fore.YELLOW + Style.BRIGHT}═══ SPELL MENU ═══ (Focus: {int(player_focus)}){Style.RESET_ALL}")
        print(f"{'#':<3} {'SPELL':<18} {'COST':<5} {'TYPE':<10} {'EFFECT'}")
        print("-" * 80)
        
        for idx, spell_name in enumerate(SPELL_LIST):
            data = SPELL_BOOK[spell_name]
            cost = data.get('cost', 0)
            spell_type = data.get('type', 'Unknown')
            
            # Cost color (green if affordable, red if not)
            c_color = Fore.GREEN if player_focus >= cost else Fore.RED + Style.DIM
            
            # Build effect description - v7 style based on type
            if spell_type == 'Damage':
                hp = data.get('hp_dmg', 0)
                pos = data.get('pos_dmg', 0)
                if spell_name == 'Incendio':
                    effect = "AoE 30 (Close) / 20 (Mid) / 15 (Far)"
                elif spell_name == 'Confringo':
                    effect = "25 HP + 10 Pos (Bonus +10 HP at Far)"
                else:
                    effect = f"{hp} HP / {pos} POS"
            elif spell_type == 'Control':
                pos = data.get('pos_dmg', 0)
                if spell_name == 'Levioso':
                    effect = f"{pos} POS + [AIRBORNE]"
                elif spell_name == 'Glacius':
                    effect = "❄️ Freeze (Interrupt) + Brittle"
                elif spell_name == 'Arresto Momentum':
                    effect = f"{pos} POS + ⏸️ Slow (No Regen)"
                else:
                    effect = f"{pos} POS + Status"
            elif spell_type == 'Force':
                hp = data.get('hp_dmg', 0)
                pos = data.get('pos_dmg', 0)
                if spell_name == 'Accio':
                    effect = f"{hp} HP, {pos} POS + Pull to CLOSE"
                elif spell_name == 'Depulso':
                    effect = f"{hp} HP, {pos} POS + Push"
                elif spell_name == 'Descendo':
                    effect = f"{hp} HP, {pos} POS ⬇️ [Bonus if AIRBORNE]"
                else:
                    effect = f"{hp} HP, {pos} POS"
            elif spell_type == 'Defense':
                if spell_name == 'Protego':
                    effect = "🛡️ Block attacks + Reflect Control"
                elif spell_name == 'Protego Maximus':
                    effect = "🛡️ Absorb Dmg to Focus"
                else:
                    effect = "Block"
            elif spell_type == 'Info':
                if spell_name == 'Revelio':
                    effect = "👁️ Reveal enemy stats"
                elif spell_name == 'Legilimens':
                    effect = "🧠 Read enemy intent"
                else:
                    effect = "Reveal"
            elif spell_type == 'Curse':
                if spell_name == 'Crucio':
                    effect = "💀 Pain (7/10/15 DoT) + Stun 2t"
                elif spell_name == 'Avada Kedavra':
                    effect = f"💀 Kill if Pos<=0, else {data.get('hp_dmg', 40)} HP"
                else:
                    effect = "💀 DARK MAGIC"
            else:
                effect = spell_type
            
            # Color code by type
            if 'curse' in spell_type.lower():
                color = Fore.MAGENTA + Style.BRIGHT
            elif 'damage' in spell_type.lower():
                color = Fore.RED
            elif 'control' in spell_type.lower():
                color = Fore.CYAN
            elif 'force' in spell_type.lower():
                color = Fore.YELLOW
            elif 'defense' in spell_type.lower():
                color = Fore.GREEN
            else:
                color = Fore.WHITE
            
            print(f"{color}{idx:<3} {spell_name:<18}{Style.RESET_ALL} {c_color}{cost:<5}{Style.RESET_ALL} {color}{spell_type:<10} {effect}{Style.RESET_ALL}")
        print("-" * 80)

    def get_player_input(self, wizard_idx: int, alive_enemies: list) -> tuple:
        """Get spell and target from player."""
        print(f"\n{Fore.GREEN}[A{wizard_idx+1}] Your turn!{Style.RESET_ALL}")
        print(f"Enemies alive: {[f'B{i+1}' for i in alive_enemies]}")
        
        # Get spell
        while True:
            try:
                spell_input = input(f"Enter spell # (0-{len(SPELL_LIST)-1}) or name: ").strip().lower()
                
                # Try as number
                if spell_input.isdigit():
                    spell_idx = int(spell_input)
                    if 0 <= spell_idx < len(SPELL_LIST):
                        break
                    print(f"{Fore.RED}Invalid spell #. Use 0-{len(SPELL_LIST)-1}.{Style.RESET_ALL}")
                    continue
                
                # Try as name match
                for idx, name in enumerate(SPELL_LIST):
                    if spell_input in name.lower():
                        spell_idx = idx
                        break
                else:
                    print(f"{Fore.RED}Unknown spell. Try again.{Style.RESET_ALL}")
                    continue
                break
            except ValueError:
                print(f"{Fore.RED}Invalid input.{Style.RESET_ALL}")
        
        spell_name = SPELL_LIST[spell_idx]
        print(f"  → Selected: {Fore.CYAN}{spell_name}{Style.RESET_ALL}")
        
        # Get target (Enemies B1-B5 OR Allies A1-A5)
        # We need to return the Target ID, not just index, because we can target both teams.
        # But execute_turn expects (spell_idx, target_idx) where target_idx is index in team_b?
        # No, execute_turn: "target_id = self.team_b[target_idx].id"
        # We need to modify execute_turn to handle ID directly if we change this.
        # Or return a tuple (is_enemy, index)? 
        # Better: return the actual Target ID. 
        # But let's check execute_turn usages.
        # It calls: action_idx, target_idx = self.get_player_input(i, alive_b)
        # Then: target_id = self.team_b[target_idx].id
        # This assumes target is always Enemy.
        # I need to refactor this too.
        # Let's return (spell_idx, target_id) from here.
        
        while True:
            target_input = input(f"Target (e.g. B1, A2): ").strip().upper()
            
            # Auto target B1 if empty?
            if not target_input and len(alive_enemies) > 0:
                 t_idx = alive_enemies[0]
                 print(f"  → Auto-Target: B{t_idx+1}")
                 return spell_idx, self.team_b[t_idx].id
            
            # Parse A# or B#
            if target_input.startswith("A"):
                try:
                    idx = int(target_input[1:]) - 1
                    if 0 <= idx < len(self.team_a):
                        if self.team_a[idx].hp > 0:
                            return spell_idx, self.team_a[idx].id
                        else:
                            print(f"{Fore.RED}A{idx+1} is defeated.{Style.RESET_ALL}")
                    else:
                        print(f"{Fore.RED}Invalid ally index.{Style.RESET_ALL}")
                except:
                    print(f"{Fore.RED}Invalid format. Use A1-A5.{Style.RESET_ALL}")
            
            elif target_input.startswith("B"):
                try:
                    idx = int(target_input[1:]) - 1
                    if 0 <= idx < len(self.team_b):
                        if self.team_b[idx].hp > 0:
                            return spell_idx, self.team_b[idx].id
                        else:
                            print(f"{Fore.RED}B{idx+1} is defeated.{Style.RESET_ALL}")
                    else:
                        print(f"{Fore.RED}Invalid enemy index.{Style.RESET_ALL}")
                except:
                    print(f"{Fore.RED}Invalid format. Use B1-B5.{Style.RESET_ALL}")
            
            # Fallback to simple number = Enemy index
            elif target_input.isdigit():
                 try:
                    idx = int(target_input) - 1
                    if idx in alive_enemies:
                        return spell_idx, self.team_b[idx].id
                    else:
                        print(f"{Fore.RED}Target B{idx+1} not valid/alive.{Style.RESET_ALL}")
                 except:
                    pass
            else:
                print(f"{Fore.RED}Unknown target. Use 'B1' for enemy 1, 'A2' for ally 2.{Style.RESET_ALL}")

    def execute_turn(self):
        """Execute a single turn using resolve_round_5v5."""
        turn_frac = self.turn / self.max_turns
        
        alive_a = [i for i, w in enumerate(self.team_a) if w.hp > 0]
        alive_b = [i for i, w in enumerate(self.team_b) if w.hp > 0]
        
        if not alive_a or not alive_b:
            return
        
        # Show spell menu for player mode
        if self.model_a == "player":
            self.render_battlefield()
            self.show_spell_menu()
        
        # Collect all actions: {wizard_id: (spell_name, target_id)}
        actions = {}
        
        # Team A actions
        for i in alive_a:
            wiz = self.team_a[i]
            
            if self.model_a == "player" and i == 0:
                # Player controls A1
                # get_player_input now returns (spell_idx, target_id)
                action_idx, target_id = self.get_player_input(i, alive_b)
                spell_name = SPELL_LIST[action_idx]
                # target_id is already the ID, no need to lookup from team_b
            else:
                # AI action
                brain = self.brains_a[i] if i < len(self.brains_a) else None
                ai_type = self.teammate_ai if self.model_a == "player" else self.model_a
                action_idx = get_brain_action(brain, ai_type, wiz, self.team_a, self.team_b, turn_frac)
                spell_name = SPELL_LIST[action_idx]
                target_idx = random.choice(alive_b)
                target_id = self.team_b[target_idx].id
            
            actions[wiz.id] = (spell_name, target_id)
            self.spell_counts_a[spell_name] += 1
        
        # Team B actions
        for i in alive_b:
            wiz = self.team_b[i]
            brain = self.brains_b[i] if i < len(self.brains_b) else None
            
            action_idx = get_brain_action(brain, self.model_b, wiz, self.team_b, self.team_a, turn_frac)
            spell_name = SPELL_LIST[action_idx]
            target_idx = random.choice(alive_a)
            target_id = self.team_a[target_idx].id
            
            actions[wiz.id] = (spell_name, target_id)
            self.spell_counts_b[spell_name] += 1
        
        # --- SNAPSHOT & RESOLVE ---
        # 1. Snapshot current state for visual replay
        snap_a = copy.deepcopy(self.team_a)
        snap_b = copy.deepcopy(self.team_b)
        
        # 2. Resolve round (modifies self.team_a/b in place)
        all_wizards = self.team_a + self.team_b
        logs = resolve_round_5v5(all_wizards, actions)
        
        # 3. Save final state
        final_team_a = self.team_a
        final_team_b = self.team_b
        
        # 4. Revert to snapshot for animation
        self.team_a = snap_a
        self.team_b = snap_b
        
        # 5. Replay logs
        print(f"\n{Fore.YELLOW}Resolving Turn...{Style.RESET_ALL}")
        
        id_to_name = {}
        for i, w in enumerate(self.team_a):
            id_to_name[w.id] = "★ YOU" if (self.model_a == "player" and i == 0) else f"A{i+1}"
        for i, w in enumerate(self.team_b):
            id_to_name[w.id] = f"B{i+1}"

        turn_log = []
        for log in logs:
            if len(log) < 6:
                continue
                
            caster_id, spell, target_id, hp_dmg, pos_dmg, effect = log
            
            # Identify caster
            source_team = "a"
            caster_idx = -1
            caster_obj = None
            
            # Check A
            for i, w in enumerate(self.team_a):
                if w.id == caster_id:
                    source_team = "a"
                    caster_idx = i
                    caster_obj = w
                    break
            

            # Check B
            for i, w in enumerate(self.team_b):
                    if w.id == caster_id:
                        source_team = "b"
                        caster_idx = i
                        caster_obj = w
                        break
            
            if caster_idx == -1:
                continue # Unknown caster
                
            # Identify target index for animation
            target_idx = -1
            target_obj = None
            opp_team = self.team_b if source_team == "a" else self.team_a
            
            if target_id is not None:
                # Check opponent team first for attacks
                for i, w in enumerate(opp_team):
                    if w.id == target_id:
                        target_idx = i
                        target_obj = w
                        break
                
                # Check own team (buffs/heals)
                if target_idx == -1:
                    own_team = self.team_a if source_team == "a" else self.team_b
                    for i, w in enumerate(own_team):
                        if w.id == target_id:
                            # For self-buffs, we usually don't animate a beam across, but we can logic it
                            target_obj = w
                            # if it's strictly self, target_idx might be irrelevant for beam
                            break
            
            # Text Log
            caster_name = id_to_name.get(caster_id, "?")
            target_name = id_to_name.get(target_id, "ALL") if target_id else "ALL"
            c_color = Fore.GREEN if source_team == "a" else Fore.RED
            
            msg = f"{c_color}{caster_name} -> {spell} -> {target_name}{Style.RESET_ALL}"
            if hp_dmg > 0:
                msg += f" {Fore.RED}-{hp_dmg} HP{Style.RESET_ALL}"
            if pos_dmg > 0:
                msg += f" {Fore.YELLOW}-{pos_dmg} POS{Style.RESET_ALL}"
            if effect:
                msg += f" {Fore.MAGENTA}{effect}{Style.RESET_ALL}"
            
            print("  " + msg)
            turn_log.append(msg.replace(Fore.GREEN, "").replace(Fore.RED, "").replace(Fore.YELLOW, "").replace(Fore.MAGENTA, "").replace(Style.RESET_ALL, ""))
            
            # Animate
            spell_color = get_spell_color(spell)
            
            # Only animate beam if it's an attack on an enemy or a unified/legacy interaction
            # Skip animation for self-buffs to avoid visual clutter, or handle them differently?
            # User asked for "show the spell color... in the order".
            # Let's animate everything that has a target.
            
            should_animate = (target_idx != -1) and (target_obj in opp_team)
            
            if should_animate:
                self.animate_attack(source_team, caster_idx, target_idx, spell, color=spell_color)
            else:
                 time.sleep(0.3) # Short pause for non-projectile actions (buffs)
            
            # Apply immediate visual updates to snapshot
            if target_obj:
                target_obj.hp = max(0, target_obj.hp - hp_dmg)
                target_obj.posture = max(-20, target_obj.posture - pos_dmg)
                # We don't easily map string effects back to status dicts without parsing
                # But HP/POS are the big ones.
                
                # Update caster focus if possible?
                # Caster focus was paid at start of round log in duel_engine.
                # Here we just show results.
                
            self.render_battlefield()
            
        
        # 6. Restore true final state
        self.team_a = final_team_a
        self.team_b = final_team_b
        
        # Final render to show updated statuses/cooldowns
        self.render_battlefield()
        if not logs:
            print("No actions taken.")
            time.sleep(1)
        
        # Show turn summary
        self.show_turn_summary(turn_log)

    def show_turn_summary(self, turn_log):
        """Display what happened this turn."""
        self.render_battlefield()
        
        print(f"\n{Fore.YELLOW + Style.BRIGHT}═══ TURN {self.turn} SUMMARY ═══{Style.RESET_ALL}")
        for line in turn_log:
            print(f"  {line}")
        
        # Check for defeats
        for i, w in enumerate(self.team_a):
            if w.hp <= 0:
                print(f"  {Fore.BLACK}💀 A{i+1} has been defeated!{Style.RESET_ALL}")
        for i, w in enumerate(self.team_b):
            if w.hp <= 0:
                print(f"  {Fore.BLACK}💀 B{i+1} has been defeated!{Style.RESET_ALL}")
        
        print(f"{'-' * 50}")
        
        # Wait for player to continue in player mode
        if self.model_a == "player":
            input(f"{Fore.CYAN}Press Enter to continue...{Style.RESET_ALL}")

    def print_spell_summary(self, team_name, spell_counts):
        """Print spell usage summary."""
        total = sum(spell_counts.values())
        if total == 0:
            print(f"\n{team_name}: No spells cast")
            return
        
        print(f"\n{Fore.YELLOW}{team_name} Spell Usage:{Style.RESET_ALL}")
        sorted_spells = sorted(spell_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        for spell, count in sorted_spells:
            pct = count / total * 100
            bar = "█" * int(pct / 5)
            print(f"  {spell:<18} {count:>3} ({pct:>5.1f}%) {bar}")

    def run(self):
        """Main game loop."""
        self.setup()
        
        while self.turn <= self.max_turns:
            alive_a = sum(1 for w in self.team_a if w.hp > 0)
            alive_b = sum(1 for w in self.team_b if w.hp > 0)
            
            if alive_a == 0 or alive_b == 0:
                break
            
            self.execute_turn()
            self.turn += 1
            
            # Brief pause between turns
            time.sleep(0.3)
        
        # Final render
        self.render_battlefield()
        
        # Determine winner
        alive_a = sum(1 for w in self.team_a if w.hp > 0)
        alive_b = sum(1 for w in self.team_b if w.hp > 0)
        total_hp_a = sum(max(0, w.hp) for w in self.team_a)
        total_hp_b = sum(max(0, w.hp) for w in self.team_b)
        
        print("\n" + "=" * 60)
        print(f"{Fore.YELLOW + Style.BRIGHT}BATTLE COMPLETE!{Style.RESET_ALL}")
        print("=" * 60)
        
        print(f"\n{Fore.GREEN}Team A [{self.label_a}]:{Style.RESET_ALL}")
        print(f"  Survivors: {alive_a}/{len(self.team_a)}")
        print(f"  Total HP: {total_hp_a}")
        
        print(f"\n{Fore.RED}Team B [{self.label_b}]:{Style.RESET_ALL}")
        print(f"  Survivors: {alive_b}/{len(self.team_b)}")
        print(f"  Total HP: {total_hp_b}")
        
        if alive_a > alive_b:
            print(f"\n{Fore.GREEN + Style.BRIGHT}🏆 TEAM A WINS! 🏆{Style.RESET_ALL}")
            audio.play("win")
        elif alive_b > alive_a:
            print(f"\n{Fore.RED + Style.BRIGHT}🏆 TEAM B WINS! 🏆{Style.RESET_ALL}")
            audio.play("loss")
        elif total_hp_a > total_hp_b:
            print(f"\n{Fore.GREEN + Style.BRIGHT}🏆 TEAM A WINS BY HP! 🏆{Style.RESET_ALL}")
        elif total_hp_b > total_hp_a:
            print(f"\n{Fore.RED + Style.BRIGHT}🏆 TEAM B WINS BY HP! 🏆{Style.RESET_ALL}")
        else:
            print(f"\n{Fore.YELLOW + Style.BRIGHT}⚖️ TIE! ⚖️{Style.RESET_ALL}")
        
        # Spell analysis
        self.print_spell_summary("Team A", self.spell_counts_a)
        self.print_spell_summary("Team B", self.spell_counts_b)
        
        print("\n" + "=" * 60)


if __name__ == "__main__":
    game = VisualDuelV8()
    game.run()
