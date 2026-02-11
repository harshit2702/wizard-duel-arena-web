"""
UnifiedBrain V2: 5v5 Version with Limited Visibility

Changes from V1:
- State: 26 features (limited visibility - only airborne/alive for others)
- Actions: 16 spells (auto-targeting)
- Roles: Flexible (Guardian, Striker, Disruptor)

Usage:
    brain = UnifiedBrainV2().to(device)
    spell_id = brain.get_action(state, temperature=0.5)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from typing import Tuple, List, Optional

from duel_engine import SPELL_LIST, SPELL_BOOK, Wizard


# ==============================================================================
# CONSTANTS
# ==============================================================================

NUM_SPELLS = len(SPELL_LIST)  # 16 spells

# Role IDs (flexible assignment, not tied to position)
ROLE_GUARDIAN = 0   # Tank/protector
ROLE_STRIKER = 1    # DPS/damage dealer
ROLE_DISRUPTOR = 2  # CC/control specialist
NUM_ROLES = 3

# State dimensions
STATE_DIM = 26  # Limited visibility state


# ==============================================================================
# STATE EXTRACTION (LIMITED VISIBILITY)
# ==============================================================================

def get_state_vector_5v5(
    me: Wizard,
    my_team: List[Wizard],
    enemy_team: List[Wizard],
) -> np.ndarray:
    """
    Extract state vector with LIMITED VISIBILITY.
    
    What we can see:
    - Self: Full info (HP, Focus, Posture, Dist, Airborne)
    - Allies: Only Airborne + Alive
    - Enemies: Only Airborne + Alive
    
    State layout (26 features):
    - Self: HP%, Focus%, Posture%, Dist, Airborne, Role = 6
    - Allies (4): Airborne, Alive = 8
    - Enemies (5): Airborne, Alive = 10
    - Turn info: TurnFrac, LastSpellType = 2
    TOTAL = 26
    """
    state = np.zeros(STATE_DIM, dtype=np.float32)
    idx = 0
    
    # --- Self (6 features) ---
    state[idx] = me.hp / me.max_hp
    idx += 1
    state[idx] = me.focus / me.max_focus
    idx += 1
    state[idx] = max(0, me.posture) / me.max_posture
    idx += 1
    state[idx] = me.dist / 2.0  # Normalized distance (0, 0.5, 1.0)
    idx += 1
    state[idx] = 1.0 if me.status.get("Airborne", 0) > 0 else 0.0
    idx += 1
    # Role will be set externally; default to 0
    state[idx] = 0.0
    idx += 1
    
    # --- Allies (4 × 2 = 8 features) ---
    allies = [w for w in my_team if w.id != me.id]
    for i in range(4):
        if i < len(allies):
            ally = allies[i]
            state[idx] = 1.0 if ally.status.get("Airborne", 0) > 0 else 0.0
            state[idx + 1] = 1.0 if ally.hp > 0 else 0.0
        else:
            state[idx] = 0.0
            state[idx + 1] = 0.0
        idx += 2
    
    # --- Enemies (5 × 2 = 10 features) ---
    for i in range(5):
        if i < len(enemy_team):
            enemy = enemy_team[i]
            state[idx] = 1.0 if enemy.status.get("Airborne", 0) > 0 else 0.0
            state[idx + 1] = 1.0 if enemy.hp > 0 else 0.0
        else:
            state[idx] = 0.0
            state[idx + 1] = 0.0
        idx += 2
    
    # --- Turn info (2 features) ---
    # These will be set by caller
    state[idx] = 0.0  # Turn fraction
    state[idx + 1] = 0.0  # Last spell type
    
    return state


def smart_target_enemy(enemy_team: List[Wizard], spell_name: str) -> Optional[int]:
    """
    Auto-target based on visible info only (airborne status).
    
    Returns:
        Enemy ID to target, or None if no valid target.
    """
    living_enemies = [e for e in enemy_team if e.hp > 0]
    if not living_enemies:
        return None
    
    spell = SPELL_BOOK.get(spell_name, {})
    spell_type = spell.get("type", "Damage")
    
    # Descendo should prioritize airborne enemies (guaranteed kill)
    if spell_name == "Descendo":
        airborne = [e for e in living_enemies if e.status.get("Airborne", 0) > 0]
        if airborne:
            return airborne[0].id
    
    # For damage spells, target a random living enemy
    return living_enemies[0].id if living_enemies else None


def smart_target_ally(my_team: List[Wizard], me: Wizard, spell_name: str) -> Optional[int]:
    """
    Auto-target ally based on visible info (airborne status).
    For ally-helping spells like Descendo on airborne ally.
    """
    allies = [w for w in my_team if w.id != me.id and w.hp > 0]
    if not allies:
        return None
    
    # Descendo can save airborne allies
    if spell_name == "Descendo":
        airborne = [a for a in allies if a.status.get("Airborne", 0) > 0]
        if airborne:
            return airborne[0].id
    
    return None


# ==============================================================================
# UNIFIED BRAIN V2 (Actor-Critic)
# ==============================================================================

class UnifiedBrainV2(nn.Module):
    """
    Neural network for 5v5 with limited visibility.
    
    Architecture:
    - Input: 26 features (limited visibility state)
    - Hidden: 128 → 128 with ReLU
    - Actor: 16 spell probabilities
    - Critic: Value estimate
    """
    
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        
        # Shared trunk
        self.fc1 = nn.Linear(STATE_DIM, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Actor head (policy)
        self.actor = nn.Linear(hidden_dim, NUM_SPELLS)
        
        # Critic head (value)
        self.critic = nn.Linear(hidden_dim, 1)
        
        # Role embedding (can influence behavior)
        self.role_embed = nn.Embedding(NUM_ROLES, 16)
        self.role_proj = nn.Linear(16, hidden_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(
        self,
        state: torch.Tensor,
        role: int = ROLE_STRIKER,
        turn_frac: float = 0.5,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            state: (batch, 26) state tensor
            role: Role ID for role embedding
            turn_frac: Turn fraction (0-1)
        
        Returns:
            logits: (batch, 16) spell logits
            value: (batch, 1) value estimate
        """
        # Add turn fraction to state
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # Shared layers
        x = F.relu(self.fc1(state))
        
        # Add role embedding
        role_t = torch.tensor([role], device=state.device)
        role_emb = self.role_embed(role_t)
        role_bias = self.role_proj(role_emb)
        x = x + role_bias
        
        x = F.relu(self.fc2(x))
        
        # Actor and critic
        logits = self.actor(x)
        value = self.critic(x)
        
        return logits, value
    
    def get_action(
        self,
        state: np.ndarray,
        role: int = ROLE_STRIKER,
        turn_frac: float = 0.5,
        valid_mask: Optional[np.ndarray] = None,
        temperature: float = 0.5,
    ) -> int:
        """
        Get action (spell ID) from state.
        
        Args:
            state: 26-feature state vector
            role: Role ID
            turn_frac: Turn fraction
            valid_mask: Optional mask for valid spells
            temperature: Exploration temperature
        
        Returns:
            spell_id: Index into SPELL_LIST
        """
        # Get device from model parameters
        device = next(self.parameters()).device
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).to(device)
            logits, _ = self.forward(state_t, role, turn_frac)
            logits = logits.squeeze()
            
            # Apply mask if provided
            if valid_mask is not None:
                mask_t = torch.FloatTensor(valid_mask).to(device)
                logits = logits.masked_fill(mask_t == 0, float('-inf'))
            
            # Temperature scaling
            probs = F.softmax(logits / max(temperature, 0.01), dim=-1)
            
            # Sample
            if temperature > 0.01:
                spell_id = torch.multinomial(probs, 1).item()
            else:
                spell_id = torch.argmax(probs).item()
            
            return spell_id
    
    def copy(self) -> "UnifiedBrainV2":
        """Create a deep copy."""
        clone = UnifiedBrainV2()
        clone.load_state_dict(copy.deepcopy(self.state_dict()))
        return clone
    
    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.1):
        """Mutate weights for evolution."""
        with torch.no_grad():
            for param in self.parameters():
                mask = torch.rand_like(param) < mutation_rate
                noise = torch.randn_like(param) * mutation_strength
                param.add_(mask.float() * noise)
    
    def crossover(self, other: "UnifiedBrainV2") -> "UnifiedBrainV2":
        """Single-point crossover with another brain."""
        child = self.copy()
        with torch.no_grad():
            params_self = list(self.parameters())
            params_other = list(other.parameters())
            params_child = list(child.parameters())
            
            # Pick crossover point
            crossover_point = len(params_self) // 2
            
            for i in range(len(params_child)):
                if i >= crossover_point:
                    params_child[i].data.copy_(params_other[i].data)
        
        return child


# ==============================================================================
# VALID SPELL MASK
# ==============================================================================

def get_valid_spell_mask(wizard: Wizard) -> np.ndarray:
    """
    Get mask of valid spells based on focus cost.
    
    Returns:
        (16,) binary mask where 1 = can cast, 0 = cannot
    """
    mask = np.zeros(NUM_SPELLS, dtype=np.float32)
    
    for i, spell_name in enumerate(SPELL_LIST):
        spell = SPELL_BOOK[spell_name]
        cost = spell.get("cost", 0)
        
        if wizard.focus >= cost:
            mask[i] = 1.0
    
    return mask


# ==============================================================================
# UTILITY
# ==============================================================================

def create_team(num_wizards: int = 5, is_player_team: bool = True) -> List[Wizard]:
    """Create a team of wizards for 5v5."""
    team = []
    archetypes = ["prodigy", "auror", "curse_specialist", "death_eater", "prodigy"]
    
    for i in range(num_wizards):
        name = f"P{i+1}" if is_player_team else f"E{i+1}"
        archetype = archetypes[i % len(archetypes)]
        wiz = Wizard(name, archetype, is_player_team, i + (0 if is_player_team else 100))
        team.append(wiz)
    
    return team
