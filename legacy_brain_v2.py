"""
Legacy Brain V2: Simple DQN for 5v5 with Limited Visibility

This is the simpler architecture (compared to UnifiedBrain V2):
- Input: 26 features (limited visibility)
- Hidden: 64 → 64 (smaller than UnifiedBrain)
- Output: 16 spells

Used as baseline comparison against UnifiedBrain V2.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from typing import List, Optional

from duel_engine import SPELL_LIST, SPELL_BOOK, Wizard


# ==============================================================================
# CONSTANTS
# ==============================================================================

STATE_DIM = 26
NUM_SPELLS = len(SPELL_LIST)  # 16


# ==============================================================================
# STATE EXTRACTION
# ==============================================================================

def get_legacy_state_5v5(
    me: Wizard,
    my_team: List[Wizard],
    enemy_team: List[Wizard],
) -> np.ndarray:
    """
    Extract limited visibility state for legacy brain.
    Same format as UnifiedBrain V2 for fair comparison.
    
    26 features:
    - Self (6): HP%, Focus%, Posture%, Dist, Airborne, Role
    - Allies (4×2=8): Airborne, Alive
    - Enemies (5×2=10): Airborne, Alive  
    - Turn info (2): TurnFrac, Reserved
    """
    state = np.zeros(STATE_DIM, dtype=np.float32)
    idx = 0
    
    # Self (6)
    state[idx] = me.hp / me.max_hp
    state[idx+1] = me.focus / me.max_focus
    state[idx+2] = max(0, me.posture) / me.max_posture
    state[idx+3] = me.dist / 2.0
    state[idx+4] = 1.0 if me.status.get("Airborne", 0) > 0 else 0.0
    state[idx+5] = 0.0  # Role placeholder
    idx += 6
    
    # Allies (4 × 2 = 8)
    allies = [w for w in my_team if w.id != me.id]
    for i in range(4):
        if i < len(allies):
            state[idx] = 1.0 if allies[i].status.get("Airborne", 0) > 0 else 0.0
            state[idx+1] = 1.0 if allies[i].hp > 0 else 0.0
        idx += 2
    
    # Enemies (5 × 2 = 10)
    for i in range(5):
        if i < len(enemy_team):
            state[idx] = 1.0 if enemy_team[i].status.get("Airborne", 0) > 0 else 0.0
            state[idx+1] = 1.0 if enemy_team[i].hp > 0 else 0.0
        idx += 2
    
    # Turn info (2)
    state[idx] = 0.0  # Turn fraction (set by caller)
    state[idx+1] = 0.0  # Reserved
    
    return state


# ==============================================================================
# LEGACY BRAIN V2 (DQN)
# ==============================================================================

class LegacyBrainV2(nn.Module):
    """
    Simple DQN for 5v5.
    
    Architecture:
    - Input: 26 features
    - Hidden: 64 → 64 (smaller than UnifiedBrain)
    - Output: 16 Q-values
    """
    
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        
        self.fc1 = nn.Linear(STATE_DIM, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, NUM_SPELLS)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            state: (batch, 26) state tensor
        
        Returns:
            q_values: (batch, 16) Q-values for each spell
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values
    
    def get_action(
        self,
        state: np.ndarray,
        valid_mask: Optional[np.ndarray] = None,
        epsilon: float = 0.0,
    ) -> int:
        """
        Get action using epsilon-greedy policy.
        
        Args:
            state: 26-feature state vector
            valid_mask: Optional mask for valid spells
            epsilon: Exploration rate
        
        Returns:
            spell_id: Index into SPELL_LIST
        """
        if np.random.random() < epsilon:
            # Random action
            if valid_mask is not None:
                valid_actions = np.where(valid_mask > 0)[0]
                if len(valid_actions) > 0:
                    return np.random.choice(valid_actions)
            return np.random.randint(0, NUM_SPELLS)
        
        # Get device from model parameters
        device = next(self.parameters()).device
        
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = self.forward(state_t).squeeze()
            
            if valid_mask is not None:
                mask_t = torch.FloatTensor(valid_mask).to(device)
                q_values = q_values.masked_fill(mask_t == 0, float('-inf'))
            
            return torch.argmax(q_values).item()
    
    def copy(self) -> "LegacyBrainV2":
        """Create a deep copy."""
        clone = LegacyBrainV2()
        clone.load_state_dict(copy.deepcopy(self.state_dict()))
        return clone
    
    def mutate(self, mutation_rate: float = 0.1, mutation_strength: float = 0.1):
        """Mutate weights for evolution."""
        with torch.no_grad():
            for param in self.parameters():
                mask = torch.rand_like(param) < mutation_rate
                noise = torch.randn_like(param) * mutation_strength
                param.add_(mask.float() * noise)
    
    def crossover(self, other: "LegacyBrainV2") -> "LegacyBrainV2":
        """Single-point crossover."""
        child = self.copy()
        with torch.no_grad():
            params_self = list(self.parameters())
            params_other = list(other.parameters())
            params_child = list(child.parameters())
            
            crossover_point = len(params_self) // 2
            
            for i in range(len(params_child)):
                if i >= crossover_point:
                    params_child[i].data.copy_(params_other[i].data)
        
        return child


# ==============================================================================
# VALID SPELL MASK
# ==============================================================================

def get_valid_spell_mask(wizard: Wizard) -> np.ndarray:
    """Get mask of valid spells based on focus."""
    mask = np.zeros(NUM_SPELLS, dtype=np.float32)
    
    for i, spell_name in enumerate(SPELL_LIST):
        cost = SPELL_BOOK[spell_name].get("cost", 0)
        if wizard.focus >= cost:
            mask[i] = 1.0
    
    return mask
