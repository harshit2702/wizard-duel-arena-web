# Draft System Design ("The Grimoire")

## Overview
A "Roguelike" drafting system where players build their spell deck **on the fly** before each match.
This replaces static loadouts with a dynamic, skill-testing preparation phase.

**Core Loop**:
1.  **Draft 6 Rounds**: In each round, the player is offered **3 Random Spells**.
2.  **Pick 1**: Player selects one spell to add to their deck.
3.  **Repeat**: Until 6 slots are filled.
4.  **Battle**: Enter the arena with the drafted deck + Basic Cast.

---

## 1. Mechanics & Rules

### The Pool
- **Total Spells**: ~13 (current implementation)
- **Draft Slots**: 6
- **Choices Per Round**: 3

### Rarity & Weighting (The "Director" AI)
To prevent "bad feels" (getting offered 3 useless spells), the draft uses a weighted system.

- **Tag System**: Every spell has tags (e.g., `Control`, `Force`, `Damage`, `Fire`, `Combo-Starter`).
- **Synergy Weights**: 
    - If you pick a **Combo-Starter** (e.g., *Levioso*), the weight of its **Combo-Finisher** (*Descendo*) increases by +50% for future rounds.
    - If you have 0 defensive spells by Round 4, the weight of *Protego* increases significantly.
    - **No Duplicates**: Once a spell is picked, it is removed from the pool.

### The "Reroll" (Optional Mechanic)
- Players get **1 Reroll** token per draft.
- Spending it refreshes the 3 choices.
- Adds a small layer of risk management.

---

## 2. Draft Logic Example

**Round 1**:
- Choices: `Incendio` (Damage), `Protego` (Defense), `Accio` (Force)
- Player picks: **Incendio**.
- *System Update*: `Fire` synergy +1. `Damage` count = 1.

**Round 2**:
- Choices now favor `Fire` or `Close-Range` (since Incendio is close range).
- Choices: `Confringo` (Fire synergy), `Glacius` (Control), `Revelio` (Info)
- Player picks: **Glacius**.
- *System Update*: `Control` count = 1. `Combo-Starter` count = 1.

**Round 3**:
- System sees `Glacius` (Combo Starter). It *really* wants to offer a payoff.
- Choices: `Diffindo` (High Dmg = Shatter Combo), `Descendo` (Force), `Basic Cast` (Wait, Basic is innate)
- Choices: `Diffindo`, `Descendo`, `Arresto Momentum`.
- Player picks: **Diffindo** (Ice Shatter Combo assembled!).

---

## 3. Implementation: Data Structures

### `DraftManager` Class
Manages the state of the draft session.

```python
class DraftManager:
    def __init__(self, spell_database):
        self.pool = list(spell_database.keys())
        self.deck = []
        self.history = [] # Track rejected spells to avoid re-offering immediately?
    
    def get_choices(self):
        # 1. Filter out already picked spells
        # 2. Calculate weights based on self.deck tags
        # 3. Return 3 weighted random choices
        pass

    def pick_card(self, spell_name):
        self.deck.append(spell_name)
        self.pool.remove(spell_name)
```

### `SpellSynergy` Data
A dictionary mapping spells to their "friends".

```python
SYNERGIES = {
    "Levioso":  {"Descendo": 5.0, "Accio": 2.0},
    "Glacius":  {"Diffindo": 5.0, "Incendio": 3.0},
    "Crucio":   {"Avada Kedavra": 10.0}, # The forbidden combo
}
```

---

## 4. AI Drafting Intelligence

The AI must also draft its deck to be competitive. It shouldn't just pick randomly.

**AI Heuristics**:
1.  **Role Preference**: If AI is `Archetype: Defender`, it forcefully picks `Protego` / `Protego Maximus` if offered.
2.  **Combo Greed**: If AI picks `Levioso`, it will prioritize `Descendo` above all else.
3.  **Balance Check**: If AI has 4 Damage spells and 0 Defense, it drastically prioritizes Defense.

*Implementation*: The `DraftManager` will have an `auto_draft(archetype)` function that simulates a player drafting with these biases.

---

## 5. UI/UX (Conceptual)

**Screen**: `DraftScreen`
- **Center**: 3 large "Cards" representing the spell choices.
- **Top**: "Your Deck" (6 slots, filling up left-to-right).
- **Hover**: Hovering a card shows its stats AND highlights interactions with your current deck (e.g., if you have *Levioso*, hovering *Descendo* makes *Levioso* glow to show synergy).
