# AI Training Session Summary

## Date: February 6-7, 2026

---

## What We Built

### Core Components

| File | Purpose |
|------|---------|
| `unified_brain.py` | Neural network with mutation/crossover for all AI roles |
| `scenarios.py` | "Drill Sergeant" tests for AI qualification |
| `duel_arena.py` | Lightweight game runner for training |

### Training Scripts

| Script | Approach | Status |
|--------|----------|--------|
| `train_evo_league.py` | Pure genetic algorithm with tournament selection | ✅ Working |
| `train_time_hive.py` | PPO + Time Turner (rewind) + Population-Based Training | ✅ Working |
| `train_gauntlet.py` | Curriculum learning with evolution fallback | ✅ Working |
| `train_vs_legacy.py` | PPO training against Legacy SquadAgent opponents | ✅ Working |

### Tournament Scripts

| Script | Purpose |
|--------|---------|
| `tournament_arena.py` | Basic tournament (UnifiedBrain only) |
| `tournament_arena_v2.py` | Parallelized with configurable game counts |
| `tournament_arena_v3.py` | Mixed tournament (UnifiedBrain + Legacy SquadAgent) |

---

## Training Results

### Evo League (100 generations, 20 population)
```
Generation 1  → Best fitness: 205
Generation 58 → Best fitness: 650 (peak)
Generation 100 → Saved to checkpoints_evo/
```

### Time Hive (50 generations, 4 islands)
```
Generation 1  → avg_score: -17.8
Generation 33 → avg_score: 85.2 (peak)
```

### Gauntlet (10 levels)
```
Level 0-6: Passed
Level 7 (Self-Play): Stuck
Final: 8/10 levels completed
```

### vs Legacy Training (200 iterations)
```
Iter 1   → Reward: -173
Iter 140 → Reward: +2.6 (breakthrough)
Iter 190 → Reward: +10.9 (peak)
Iter 200 → Reward: -10.7
```

---

## Tournament Results (72,000 games)

### UnifiedBrain Only
| Rank | Brain | W-L-D | vsRandom |
|------|-------|-------|----------|
| 1 | Evo League | 7-0-1 | 187.8 |
| 2 | Evo Gen100 | 7-0-1 | 185.4 |
| 3 | Evo Gen50 | 6-2-0 | 180.4 |

### Mixed (UnifiedBrain + Legacy)
| Rank | Brain | Type | W-L-D |
|------|-------|------|-------|
| 1 | Legacy Support | squad | 4-0-4 |
| 2 | Legacy DPS | squad | 4-0-4 |
| 3 | Legacy Tank | squad | 3-0-5 |
| 5 | Evo Gen100 | unified | 3-0-5 |
| 6 | Evo League | unified | 3-0-5 |
| 7 | vs Legacy | unified | 1-5-2 |

**Key Finding:** Legacy SquadAgent brains dominate because:
- 3v1 advantage (3 squad members vs 1 player)
- Specialized training for enemy role only
- Simpler state/action space (12 features, spell-only actions)

---

## System Info

- CPU: Intel Xeon w7-2495X (48 threads)
- GPU: CUDA available
- Tournament throughput: ~500 games/sec with 24 workers

---

## Files Created

```
oneVsMany/
├── unified_brain.py         # Core neural network
├── scenarios.py             # Drill Sergeant tests
├── duel_arena.py            # Game runner
├── train_evo_league.py      # Genetic algorithm training
├── train_time_hive.py       # PPO + PBT training
├── train_gauntlet.py        # Curriculum training
├── train_vs_legacy.py       # Train against Legacy
├── tournament_arena.py      # Basic tournament
├── tournament_arena_v2.py   # Parallelized tournament
├── tournament_arena_v3.py   # Mixed architecture tournament
├── checkpoints_evo/         # Evo League checkpoints
├── checkpoints_hive/        # Time Hive checkpoints
├── checkpoints_gauntlet/    # Gauntlet checkpoints
└── checkpoints_vs_legacy/   # vs Legacy checkpoints
```

---

## Next Steps to Consider

1. **Dedicated Player Brain** - Train for Player role only
2. **Curriculum Learning** - Beat progressively harder opponents
3. **Counter-Strategy** - Analyze and counter Legacy patterns
4. **Population vs Population** - Train squad of brains
5. **Longer Training** - 1000+ iterations for vs Legacy
6. **Better Rewards** - Bonus for kills, not just damage
