import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from ray.rllib.env.multi_agent_env import MultiAgentEnv

from duel_engine import Wizard, resolve_round, SPELL_LIST, SPELL_BOOK

# Default values if config is missing
DEFAULT_CONFIG = {
    "max_turns": 15,
    "anti_stall_N": 3,
    "stall_penalty": -5.0,
    "illegal_penalty": -1.0,
}

PLAYER_AID = "player"
TANK_AID = "tank"
DPS_AID = "dps"
SUPPORT_AID = "support"

AGENTS = [PLAYER_AID, TANK_AID, DPS_AID, SUPPORT_AID]

def _spell_id(name: str) -> int:
    return SPELL_LIST.index(name)

class Duel3v1Env(MultiAgentEnv):
    def __init__(self, config=None):
        # 1. HANDLE CONFIG SAFELY
        if config is None:
            config = DEFAULT_CONFIG.copy()
        
        # Extract values or use defaults
        self.max_turns = config.get("max_turns", 15)
        self.anti_stall_N = config.get("anti_stall_N", 3)
        self.stall_penalty = config.get("stall_penalty", -5.0)
        self.illegal_penalty = config.get("illegal_penalty", -1.0)

        super().__init__()

        # --- FIX 1: DEFINE AGENT LISTS (REQUIRED BY RAY) ---
        self.agents = [PLAYER_AID, TANK_AID, DPS_AID, SUPPORT_AID]
        self.possible_agents = [PLAYER_AID, TANK_AID, DPS_AID, SUPPORT_AID]
        self._agent_ids = set(self.agents) 
        # ---------------------------------------------------

        self.num_spells = len(SPELL_LIST)
        self.num_targets = 4  # Player(0) or Enemies(1,2,3)
        self.total_actions = self.num_spells * self.num_targets

        # Action space: MultiDiscrete [spell_id, target_idx]
        single_action_space = spaces.MultiDiscrete([self.num_spells, self.num_targets])
        
        # 14 features in _state_vec
        obs_dim = 14
        
        # Observation space: Dict with 'observations' and 'action_mask'
        # Action mask is flat: num_spells * num_targets binary values
        single_observation_space = spaces.Dict({
            "observations": spaces.Box(low=-1.0, high=2.0, shape=(obs_dim,), dtype=np.float32),
            "action_mask": spaces.Box(low=0, high=1, shape=(self.total_actions,), dtype=np.float32),
        })

        self.action_spaces = {aid: single_action_space for aid in AGENTS}
        self.observation_spaces = {aid: single_observation_space for aid in AGENTS}

        # Keep these (legacy/testing support)
        self.action_space = single_action_space
        self.observation_space = single_observation_space

    def _reset_match(self):
        self.turn = 0
        self.stall_counter = 0
        self.last_total_hp = None

        self.player = Wizard("Player", "prodigy", True, 99)
        self.enemies = [
            Wizard("Tank", "auror", False, 1),
            Wizard("DPS", "curse_specialist", False, 2),
            Wizard("Support", "death_eater", False, 3),
        ]

        self.last_spell = {aid: None for aid in AGENTS}
        self.repeat_count = {aid: 0 for aid in AGENTS}
        self.pending_setup = [] 

    # ---------- helpers ----------
    def _living_enemies(self):
        return [e for e in self.enemies if e.hp > 0]

    def _id_to_enemy(self, eid: int):
        for e in self.enemies:
            if e.id == eid:
                return e
        return None

    def _primary_enemy_for_player_obs(self):
        le = self._living_enemies()
        return le[0] if le else self.enemies[0]

    def _state_vec(self, me: Wizard, opp: Wizard, allies: list[Wizard]):
        me_hp = me.hp / me.max_hp
        me_pos = max(0, me.posture) / me.max_posture
        me_foc = me.focus / me.max_focus
        me_dist = me.dist / 2.0
        me_air = 1.0 if me.status.get("Airborne", 0) > 0 else 0.0

        op_hp = opp.hp / opp.max_hp
        op_pos = max(0, opp.posture) / opp.max_posture
        op_foc = opp.focus / opp.max_focus
        op_air = 1.0 if opp.status.get("Airborne", 0) > 0 else 0.0
        op_stun = 1.0 if (opp.status.get("Frozen", 0) > 0 or opp.status.get("Stunned", 0) > 0) else 0.0

        ally_low = 0.0
        ally_air = 0.0
        for a in allies:
            if a.id == me.id or a.hp <= 0:
                continue
            if a.hp / a.max_hp < 0.3:
                ally_low = 1.0
            if a.status.get("Airborne", 0) > 0:
                ally_air = 1.0

        return np.array(
            [me_hp, me_pos, me_foc, me_dist, me_air,
             op_hp, op_pos, op_foc, op_air, op_stun,
             ally_low, ally_air],
            dtype=np.float32
        )

    def _obs(self):
        obs = {}
        pe = self._primary_enemy_for_player_obs()
        base = self._state_vec(self.player, pe, [])
        turn_frac = np.array([self.turn / max(1, self.max_turns)], dtype=np.float32)
        last = self.last_spell[PLAYER_AID]
        last_norm = np.array([(-1.0 if last is None else last / max(1, self.num_spells - 1))], dtype=np.float32)
        obs[PLAYER_AID] = np.concatenate([base, turn_frac, last_norm], axis=0)

        for aid, eid in [(TANK_AID, 1), (DPS_AID, 2), (SUPPORT_AID, 3)]:
            me = self._id_to_enemy(eid)
            base = self._state_vec(me, self.player, self.enemies)
            last = self.last_spell[aid]
            last_norm = np.array([(-1.0 if last is None else last / max(1, self.num_spells - 1))], dtype=np.float32)
            obs[aid] = np.concatenate([base, turn_frac, last_norm], axis=0)
        return obs

    def _get_action_mask(self, wizard: Wizard, is_player: bool) -> np.ndarray:
        """Generate a binary mask for valid (spell, target) combinations.
        
        Returns a flat array of shape (num_spells * num_targets,) where:
        - Index = spell_id * num_targets + target_idx
        - 1 = valid action, 0 = invalid action
        """
        mask = np.zeros(self.total_actions, dtype=np.float32)
        
        for spell_id, spell_name in enumerate(SPELL_LIST):
            spell_data = SPELL_BOOK.get(spell_name, {})
            cost = spell_data.get("cost", 0)
            
            # Can the wizard afford this spell?
            can_afford = wizard.focus >= cost
            
            # Does this spell require a target?
            requires_target = self._spell_requires_target(spell_name)
            
            for target_idx in range(self.num_targets):
                action_index = spell_id * self.num_targets + target_idx
                
                if not can_afford:
                    mask[action_index] = 0
                    continue
                
                if not requires_target:
                    # No-target spells: only valid with target_idx=0
                    mask[action_index] = 1.0 if target_idx == 0 else 0.0
                else:
                    # Target spells: check if target is alive
                    if is_player:
                        # Player targets enemies (1, 2, 3 -> enemy IDs)
                        if target_idx == 0:
                            mask[action_index] = 0  # Player can't target self
                        else:
                            enemy = self._id_to_enemy(target_idx)
                            mask[action_index] = 1.0 if (enemy and enemy.hp > 0) else 0.0
                    else:
                        # Enemies: 0 = Player, 1-3 = allies
                        if target_idx == 0:
                            mask[action_index] = 1.0 if self.player.hp > 0 else 0.0
                        else:
                            ally = self._id_to_enemy(target_idx)
                            mask[action_index] = 1.0 if (ally and ally.hp > 0) else 0.0
        
        # Ensure at least Basic Cast is always valid (fallback)
        basic_cast_id = SPELL_LIST.index("Basic Cast")
        for t in range(self.num_targets):
            idx = basic_cast_id * self.num_targets + t
            if is_player and t > 0:
                enemy = self._id_to_enemy(t)
                if enemy and enemy.hp > 0:
                    mask[idx] = 1.0
            elif not is_player and t == 0 and self.player.hp > 0:
                mask[idx] = 1.0
        
        return mask

    def _obs_with_mask(self):
        """Returns observations in Dict format with action masks."""
        raw_obs = self._obs_raw()
        obs_dict = {}
        
        for aid in AGENTS:
            is_player = (aid == PLAYER_AID)
            if is_player:
                wizard = self.player
            else:
                eid = {TANK_AID: 1, DPS_AID: 2, SUPPORT_AID: 3}[aid]
                wizard = self._id_to_enemy(eid)
            
            mask = self._get_action_mask(wizard, is_player)
            obs_dict[aid] = {
                "observations": raw_obs[aid],
                "action_mask": mask,
            }
        
        return obs_dict

    def _obs_raw(self):
        """Returns raw observation vectors (original format)."""
        obs = {}
        pe = self._primary_enemy_for_player_obs()
        base = self._state_vec(self.player, pe, [])
        turn_frac = np.array([self.turn / max(1, self.max_turns)], dtype=np.float32)
        last = self.last_spell[PLAYER_AID]
        last_norm = np.array([(-1.0 if last is None else last / max(1, self.num_spells - 1))], dtype=np.float32)
        obs[PLAYER_AID] = np.concatenate([base, turn_frac, last_norm], axis=0)

        for aid, eid in [(TANK_AID, 1), (DPS_AID, 2), (SUPPORT_AID, 3)]:
            me = self._id_to_enemy(eid)
            base = self._state_vec(me, self.player, self.enemies)
            last = self.last_spell[aid]
            last_norm = np.array([(-1.0 if last is None else last / max(1, self.num_spells - 1))], dtype=np.float32)
            obs[aid] = np.concatenate([base, turn_frac, last_norm], axis=0)
        return obs

    def _obs(self):
        """Main observation method - returns Dict format with masks."""
        return self._obs_with_mask()

    # ---------- legality ----------
    def _spell_requires_target(self, spell_name: str) -> bool:
        return spell_name not in ["Protego", "Protego Maximus", "Revelio"]

    def _anti_repeat(self, aid: str, spell_id: int):
        last = self.last_spell[aid]
        if last == spell_id:
            self.repeat_count[aid] += 1
        else:
            self.repeat_count[aid] = 1
        self.last_spell[aid] = spell_id

        if self.repeat_count[aid] == 2:
            return -10.0
        if self.repeat_count[aid] >= 3:
            return -50.0
        return 0.0

    def _legalize_action_player(self, spell_id: int, target_idx: int):
        spell = SPELL_LIST[spell_id]
        if not self._spell_requires_target(spell):
            if not self.player.can_cast(spell):
                return _spell_id("Protego"), 0, self.illegal_penalty
            return spell_id, 0, 0.0

        eid = int(target_idx)
        enemy = self._id_to_enemy(eid)
        if enemy is None or enemy.hp <= 0:
            le = self._living_enemies()
            if not le:
                return _spell_id("Protego"), 0, self.illegal_penalty
            enemy = random.choice(le)
            eid = enemy.id

        if not self.player.can_cast(spell):
            le = self._living_enemies()
            if le:
                enemy = random.choice(le)
                return _spell_id("Basic Cast"), enemy.id, self.illegal_penalty
            return _spell_id("Protego"), 0, self.illegal_penalty

        return spell_id, eid, 0.0

    def _legalize_action_enemy(self, enemy: Wizard, spell_id: int, target_idx: int):
        spell = SPELL_LIST[spell_id]
        if not self._spell_requires_target(spell):
            if not enemy.can_cast(spell):
                return _spell_id("Basic Cast"), self.player.id, self.illegal_penalty
            return spell_id, 0, 0.0

        if not enemy.can_cast(spell):
            return _spell_id("Basic Cast"), self.player.id, self.illegal_penalty

        # Map index to ID: 0->Player(99), 1->1, 2->2, 3->3
        tgt_id = 99 if target_idx == 0 else target_idx
        
        # Verify target is alive
        valid = False
        if tgt_id == 99:
            if self.player.hp > 0:
                valid = True
        else:
            e = self._id_to_enemy(tgt_id)
            if e and e.hp > 0:
                 valid = True
        
        if not valid:
             # Fallback to Player if alive, else random enemy (game over anyway)
             tgt_id = 99

        return spell_id, tgt_id, 0.0

    def _combo_and_setup_tracking(self, actions_by_agent):
        for aid in [TANK_AID, DPS_AID, SUPPORT_AID]:
            sid, _ = actions_by_agent[aid]
            sname = SPELL_LIST[sid]
            if sname in ["Levioso", "Glacius"]:
                self.pending_setup.append({"by": aid, "expires": self.turn + 1})

    def _assist_credit(self, player_damage_this_turn):
        bonus = {aid: 0.0 for aid in AGENTS}
        if player_damage_this_turn <= 30:
            return bonus
        still = []
        for s in self.pending_setup:
            if self.turn <= s["expires"]:
                bonus[s["by"]] += 10.0
                still.append(s)
        self.pending_setup = still
        return bonus

    def _compute_combo_rewards(self, actions_by_agent, pre_status, post_status):
        """Compute combo rewards based on spell synergies.
        
        Combos:
        - Levioso → Descendo: +15 for slamming airborne target
        - Glacius → High Damage: +20 for hitting Brittle target
        - Crucio → Avada Kedavra: +25 for executing CursedPain target
        - Ally Save: +25 for using Descendo to save airborne ally
        """
        combo_rewards = {aid: 0.0 for aid in AGENTS}
        
        # Check each agent's action for combos
        for aid in [TANK_AID, DPS_AID, SUPPORT_AID]:
            sid, tgt_id = actions_by_agent[aid]
            spell_name = SPELL_LIST[sid]
            
            # Descendo on Airborne target (Levioso → Descendo combo)
            if spell_name == "Descendo":
                # Check if target was airborne BEFORE this round
                if tgt_id == 99:  # Player
                    if pre_status.get("player_airborne", False):
                        combo_rewards[aid] += 15.0  # Slam combo!
                elif tgt_id in [1, 2, 3]:  # Ally
                    # Check if we saved an ally
                    if pre_status.get(f"enemy_{tgt_id}_airborne", False):
                        combo_rewards[aid] += 25.0  # Ally save!
            
            # High damage on Brittle target (Glacius → Damage combo)
            if spell_name in ["Diffindo", "Avada Kedavra", "Confringo", "Incendio"]:
                if tgt_id == 99 and pre_status.get("player_brittle", False):
                    combo_rewards[aid] += 20.0  # Brittle combo!
            
            # Avada on CursedPain target (Crucio → Avada combo)
            if spell_name == "Avada Kedavra":
                if tgt_id == 99 and pre_status.get("player_cursed", False):
                    combo_rewards[aid] += 25.0  # Execute combo!
        
        return combo_rewards

    def _capture_status(self):
        """Capture current status for combo detection."""
        status = {
            "player_airborne": self.player.status.get("Airborne", 0) > 0,
            "player_brittle": self.player.status.get("Brittle", 0) > 0,
            "player_cursed": self.player.status.get("CursedPain", 0) > 0,
        }
        for e in self.enemies:
            status[f"enemy_{e.id}_airborne"] = e.status.get("Airborne", 0) > 0
        return status

    # ---------- gym api ----------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_match()
        self.last_total_hp = self.player.hp + sum(e.hp for e in self.enemies)
        return self._obs(), {}

    def step(self, action_dict):
        self.turn += 1
        p_hp_pre = self.player.hp
        e_hp_pre = {e.id: e.hp for e in self.enemies}
        rewards = {aid: 0.0 for aid in AGENTS}
        infos = {aid: {} for aid in AGENTS}

        # 1. Player Action
        psid, p_tid_idx = action_dict.get(PLAYER_AID, (0, 0))
        psid, p_target_id, pen = self._legalize_action_player(int(psid), int(p_tid_idx))
        rewards[PLAYER_AID] -= pen

        # 2. Enemy Actions
        ai_moves = {}
        ai_targets = {}
        for aid, eid in [(TANK_AID, 1), (DPS_AID, 2), (SUPPORT_AID, 3)]:
            enemy = self._id_to_enemy(eid)
            sid, t_idx = action_dict.get(aid, (0, 0))
            sid, tgt_id, pen = self._legalize_action_enemy(enemy, int(sid), int(t_idx))
            rewards[aid] -= pen
            ai_moves[eid] = SPELL_LIST[sid]
            ai_targets[eid] = tgt_id

        # 3. Anti-Repeat
        rewards[PLAYER_AID] += self._anti_repeat(PLAYER_AID, psid)
        rewards[TANK_AID] += self._anti_repeat(TANK_AID, _spell_id(ai_moves[1]))
        rewards[DPS_AID] += self._anti_repeat(DPS_AID, _spell_id(ai_moves[2]))
        rewards[SUPPORT_AID] += self._anti_repeat(SUPPORT_AID, _spell_id(ai_moves[3]))

        actions_by_agent = {
            PLAYER_AID: (psid, p_target_id),
            TANK_AID: (_spell_id(ai_moves[1]), ai_targets[1]),
            DPS_AID: (_spell_id(ai_moves[2]), ai_targets[2]),
            SUPPORT_AID: (_spell_id(ai_moves[3]), ai_targets[3]),
        }
        self._combo_and_setup_tracking(actions_by_agent)

        # Capture status BEFORE resolution for combo detection
        pre_status = self._capture_status()

        # 4. Resolve Round
        resolve_round(
            self.player,
            self.enemies,
            SPELL_LIST[psid],
            p_target_id,
            ai_moves_override=ai_moves,
            ai_targets_override=ai_targets,
        )

        # Capture status AFTER resolution
        post_status = self._capture_status()

        # 4b. Combo Rewards
        combo_rewards = self._compute_combo_rewards(actions_by_agent, pre_status, post_status)
        for aid, bonus in combo_rewards.items():
            rewards[aid] += bonus

        # 5. Rewards
        p_hp_post = self.player.hp
        e_hp_post = {e.id: e.hp for e in self.enemies}
        
        player_damage_taken = max(0.0, p_hp_pre - p_hp_post)
        player_damage_dealt = sum(max(0.0, e_hp_pre[i] - e_hp_post[i]) for i in e_hp_pre)

        rewards[PLAYER_AID] += player_damage_dealt - 1.5 * player_damage_taken
        for aid, eid in [(TANK_AID, 1), (DPS_AID, 2), (SUPPORT_AID, 3)]:
            dmg_taken = max(0.0, e_hp_pre[eid] - e_hp_post[eid])
            rewards[aid] += player_damage_taken - 1.5 * dmg_taken

        assist_bonus = self._assist_credit(player_damage_taken)
        for k, v in assist_bonus.items():
            rewards[k] += v

        # 6. Termination
        done = False
        if self.player.hp <= 0:
            done = True
            for aid in [TANK_AID, DPS_AID, SUPPORT_AID]:
                rewards[aid] += 100.0
            rewards[PLAYER_AID] -= 100.0

        if all(e.hp <= 0 for e in self.enemies):
            done = True
            rewards[PLAYER_AID] += 100.0
            for aid in [TANK_AID, DPS_AID, SUPPORT_AID]:
                rewards[aid] -= 100.0

        # 7. Truncation (Anti-Stall)
        truncated = False
        total_hp = self.player.hp + sum(e.hp for e in self.enemies)
        
        if self.last_total_hp is None: 
            self.last_total_hp = total_hp
            
        if abs(total_hp - self.last_total_hp) < 1e-6:
            self.stall_counter += 1
        else:
            self.stall_counter = 0
        self.last_total_hp = total_hp

        if self.stall_counter >= self.anti_stall_N and not done:
            truncated = True
            for aid in AGENTS:
                rewards[aid] += self.stall_penalty

        if self.turn >= self.max_turns and not done:
            truncated = True

        obs = self._obs()
        terminateds = {aid: done for aid in AGENTS}
        terminateds["__all__"] = done
        truncateds = {aid: truncated for aid in AGENTS}
        truncateds["__all__"] = truncated

        return obs, rewards, terminateds, truncateds, infos
