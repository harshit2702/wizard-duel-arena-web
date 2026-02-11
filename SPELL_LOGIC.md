# Spell Logic Documentation

This document describes all spells in the 5v5 Team Duel system.

## Spell Priorities

Spells resolve in **priority order** (lower = faster):
1. **Defense** (Prio 1) - Shields activate first
2. **Info** (Prio 2) - Reveals and mind-reading
3. **Control** (Prio 3) - Levitate, Freeze, Slow
4. **Force** (Prio 4) - Push, Pull, Slam
5. **Damage/Curse** (Prio 5) - Direct damage spells

---

## Spell Details

### Defense (Priority 1)

| Spell | Cost | Effect |
|-------|------|--------|
| **Protego** | 0 | Blocks incoming damage. **Reflects** Control spells back at caster. |
| **Protego Maximus** | 30 | Absorbs damage as Focus instead of HP. 
---

### Info (Priority 2)

| Spell | Cost | Effect |
|-------|------|--------|
| **Revelio** | 10 | Reveals enemy stats for 3 rounds. (No target required) |
| **Legilimens** | 20 | Mind-reads target: shows HP/Posture.

---

### Control (Priority 3)

| Spell | Cost | Posture Dmg | Effect |
|-------|------|-------------|--------|
| **Levioso** | 15 | 20 | Target becomes **Airborne** Airborne units can't use Force/Damage spells. |Can only come down after descendo
| **Glacius** | 20 | 0 | Target becomes **Frozen** (skips next action) and **Brittle** (next hit deals 2x damage). |
| **Arresto Momentum** | 25 | 30 | Target becomes **Slowed** (no posture regen). |

---

### Force (Priority 4)

| Spell | Cost | Pos Dmg | HP Dmg | Effect |
|-------|------|---------|--------|--------|
| **Accio** | 15 | 15 | 5 | Pulls target to **CLOSE** range. |
| **Depulso** | 15 | 25 | 5 | Pushes target back one range level. |
| **Descendo** | 20 | 20 | 10 | Slams target. If target is **Airborne**, deals **bonus damage** (10 HP + 20 Posture). If friendly, **saves ally** from airborne state. |

---

### Damage (Priority 5)

| Spell | Cost | Pos Dmg | HP Dmg | Effect |
|-------|------|---------|--------|--------|
| **Basic Cast** | 0 | 5 | 5 | Weak attack. Regenerates 5 Posture on cast. Safe fallback. |
| **Incendio** | 25 | 5 | 20-30 | **AoE at CLOSE range**: hits all close enemies for 30 dmg. Otherwise single target (20 MID, 15 FAR). |
| **Confringo** | 30 | 10 | 25+10 | Explosion. **Bonus +10 damage at FAR range**. |
| **Diffindo** | 35 | 5 | 45 | High damage single target attack. |

---

### Curses (Priority 5)

| Spell | Cost | Effect |
|-------|------|--------|
| **Crucio** | 50 | Target is **Stunned** for 2 turns. Applies **Cursed Pain** (3 stacks): deals 7/10/15 damage per turn. |
| **Avada Kedavra** | 100 | If target has **broken posture** (≤0): **instant kill**. Otherwise deals 40 HP damage. Also deals 40 damage to all enemies with Cursed Pain. |

---

## Key Mechanics

### Posture
- Starts at max (e.g., 50-60 depending on archetype)
- **Broken Posture (≤0)**: Takes 100% damage from all attacks instead of 50%
- Regenerates **+5 per turn** if not hit

### Shields (Protego)
- **Reflect**: Control spells bounce back to caster
- **Block**: Damage spells are blocked entirely
- Shields only last **one turn**

### Airborne State
- Applied by **Levioso**
- Victim can only use Info/Control/Defense spells
- **Descendo** can slam airborne targets for extra damage
- Allies can use Descendo to **save** airborne teammates (no damage, just removes airborne)

### Brittle State
- Applied by **Glacius**
- Next hit deals **2x damage**
- Consumed after one hit



## Combo Rewards (RL Training)

The following combos give bonus rewards during training:

| Setup | Follow-up | Bonus | Description |
|-------|-----------|-------|-------------|
| Levioso | Descendo | +15 | Slam airborne target |
| Glacius | High Damage | +20 | Hit Brittle target (2x dmg) |
| Crucio | Avada Kedavra | +25 | Execute CursedPain target |
| Ally Airborne | Descendo (ally) | +25 | Save airborne ally |

---

## Focus Regeneration

- **Base Regen**: +15 per turn (increased from 10)
- **Shielding**: No regen when using Protego/Protego Maximus
- **Basic Cast**: +5 posture (not focus)

---

## 5v5 Symmetric Combat

All spells now work **identically** for both teams. The game is now a Many-vs-Many (5v5) team-based combat system.

### Team Structure
- Two teams of 5 wizards each (Team A vs Team B)
- Each wizard has a `team` attribute (0 or 1)
- Targeted attacks: any wizard can target any enemy wizard

### Symmetric Spell Behavior

| Spell | Behavior (same for all) |
|-------|-------------------------|
| **Protego Maximus** | Absorbs damage as Focus (no GlobalShield) |
| **Incendio** | AoE 30 dmg to ALL close enemies; otherwise single target (20 MID, 15 FAR) |
| **Avada Kedavra** | Instant kill if posture ≤0, else 40 dmg. Chains 40 dmg to ALL enemies with CursedPain |
| **Depulso** | Pushes the TARGET back one range level |
| **Revelio** | Reveals all enemy team stats for 3 rounds |

### Airborne Mechanics
- Airborne state stays until someone casts **Descendo** on the target
- **Same team** (self or ally) Descendo → removes airborne, **NO damage**
- **Enemy** Descendo → removes airborne + slam damage (10 HP, 20 Posture)

