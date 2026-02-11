
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Optional
from duel_engine import Wizard, SPELL_LIST

# ==============================================================================
# LEGACY STATE & MODEL
# ==============================================================================

STATE_SIZE = 12
ACTION_SIZE = len(SPELL_LIST)

class LegacyDQN(nn.Module):
    def __init__(self):
        super(LegacyDQN, self).__init__()
        self.fc1 = nn.Linear(STATE_SIZE, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, ACTION_SIZE)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def load_legacy_model(path: str, device: str = "cpu") -> LegacyDQN:
    """Load a legacy SquadAgent DQN model."""
    model = LegacyDQN().to(device)
    try:
        model.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    except FileNotFoundError:
        print(f"Warning: Legacy checkpoint not found {path}, using random weights")
    model.eval()
    return model


def get_squad_state(me: Wizard, player: Wizard, allies: List[Wizard]) -> np.ndarray:
    """Get state vector for SquadAgent (12 features)."""
    me_hp = me.hp / me.max_hp
    me_pos = max(0, me.posture) / me.max_posture
    me_foc = me.focus / me.max_focus
    me_dist = me.dist / 2.0
    me_air = 1.0 if me.status.get("Airborne", 0) > 0 else 0.0
    
    p_hp = player.hp / player.max_hp
    p_pos = max(0, player.posture) / player.max_posture
    p_foc = player.focus / player.max_focus
    p_air = 1.0 if player.status.get("Airborne", 0) > 0 else 0.0
    p_stun = 1.0 if player.status.get("Frozen", 0) > 0 or player.status.get("Stunned", 0) > 0 else 0.0
    
    ally_low = 0.0
    ally_air = 0.0
    for a in allies:
        if a.id != me.id and a.hp > 0:
            if (a.hp / a.max_hp) < 0.3:
                ally_low = 1.0
            if a.status.get("Airborne", 0) > 0:
                ally_air = 1.0
    
    return np.array([me_hp, me_pos, me_foc, me_dist, me_air,
                     p_hp, p_pos, p_foc, p_air, p_stun,
                     ally_low, ally_air], dtype=np.float32)

def select_legacy_action(model: LegacyDQN, state: np.ndarray, device: str = "cpu") -> int:
    """Select action using SquadAgent model."""
    with torch.no_grad():
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = model(state_t)
        return q_values.max(1)[1].item()

class LegacyAgentWrapper:
    """Wrapped agent interface for compatibility."""
    def __init__(self, model_path: str, device: str = "cpu"):
        self.model = load_legacy_model(model_path, device)
        self.device = device
        

    def get_action(self, me: Wizard, enemies: List[Wizard], allies: List[Wizard]) -> int:
        # Legacy brain expects 'player' (primary enemy).
        # We usually pick the first living enemy or closest.
        target = next((e for e in enemies if e.hp > 0), enemies[0])
        
        state = get_squad_state(me, target, allies)
        return select_legacy_action(self.model, state, self.device)
