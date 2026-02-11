# 5v5 GPU Training Guide

## Quick Start

All scripts require Python with PyTorch and CUDA. Activate the venv first:
```bash
cd /home/qsdal2/Desktop/bookGame/oneVsMany
source venv/bin/activate
```

---

## 1. Evolution Training (`train_5v5_evo_gpu.py`)

**Best for:** Initial exploration, finding diverse strategies

```bash
./venv/bin/python train_5v5_evo_gpu.py --generations 100 --population 20 --games_per_match 5
```

| Parameter | Default | Description | Recommendation |
|-----------|---------|-------------|----------------|
| `--generations` | 50 | Number of evolution cycles | 100-200 for serious training |
| `--population` | 10 | Teams per architecture | 10-20 (more = slower but diverse) |
| `--games_per_match` | 3 | Games per team matchup | 3-5 for stable fitness |
| `--checkpoint_dir` | `checkpoints_evo_gpu` | Save location | — |

**Output:** Saves `unified_best_*.pth` and `legacy_best_*.pth`

---

## 2. MAPPO Training (`train_5v5_mappo_gpu.py`)

**Best for:** Team coordination, learning cooperative behaviors

```bash
./venv/bin/python train_5v5_mappo_gpu.py --iterations 500 --episodes 20 --lr 3e-4
```

| Parameter | Default | Description | Recommendation |
|-----------|---------|-------------|----------------|
| `--iterations` | 100 | Training iterations | 200-500 for strong policy |
| `--episodes` | 10 | Episodes per iteration | 10-20 (more = stable gradients) |
| `--lr` | 3e-4 | Learning rate | 1e-4 to 5e-4 |
| `--checkpoint_dir` | `checkpoints_mappo_gpu` | Save location | — |

**Output:** Saves `mappo_best.pth`, `mappo_final.pth`, `mappo_critic.pth`

---

## 3. PBT Training (`train_5v5_pbt_gpu.py`)

**Best for:** Finding optimal hyperparameters automatically

```bash
./venv/bin/python train_5v5_pbt_gpu.py --population 20 --iterations 500 --episodes_per_iter 20 --eval_games 10 --workers 32
```

| Parameter | Default | Description | Recommendation |
|-----------|---------|-------------|----------------|
| `--population` | 10 | Number of parallel agents | 10-20 |
| `--iterations` | 50 | PBT iterations | 100+ for hyperparameter discovery |
| `--episodes_per_iter` | 10 | Episodes before evaluation | 10-20 |
| `--eval_games` | 5 | Evaluation games per member | 5-10 |
| `--workers` | 8 | CPU workers for evaluation | Match CPU cores |
| `--checkpoint_dir` | `checkpoints_pbt_gpu` | Save location | — |

**Output:** Saves `pbt_best.pth`, `pbt_iter*.pth`

**Hyperparameters tuned automatically:**
- Learning rate (1e-5 to 1e-2)
- Entropy coefficient (0.001 to 0.1)
- Temperature (0.1 to 2.0)

---

## 4. Imitation + Fine-tuning (`train_5v5_imitation_gpu.py`)

**Best for:** Bootstrapping from existing Legacy agents

```bash
./venv/bin/python train_5v5_imitation_gpu.py --imitation_episodes 500 --imitation_epochs 50 --finetune_iters 500
```

| Parameter | Default | Description | Recommendation |
|-----------|---------|-------------|----------------|
| `--imitation_episodes` | 200 | Demo episodes from Legacy | 200-500 for good imitation |
| `--imitation_epochs` | 10 | Epochs of behavioral cloning | 10-20 |
| `--finetune_iters` | 50 | PPO fine-tuning iterations | 100-200 to beat teacher |
| `--episodes_per_iter` | 10 | Episodes per PPO iteration | 10-20 |
| `--lr` | 3e-4 | Learning rate | 1e-4 to 3e-4 |
| `--legacy_dir` | `checkpoints_5v5` | Where Legacy models are | Point to trained Legacy |
| `--checkpoint_dir` | `checkpoints_imitation_gpu` | Save location | — |

**Requires:** Pre-trained Legacy brains in `--legacy_dir` as `legacy_best_*.pth`

**Output:** 
- `imitation_phase1.pth` (after behavioral cloning)
- `student_best.pth` (best vs Legacy)
- `student_final.pth` (final model)

---

## 5. CPU Evolution (`train_5v5_evo.py`)

**Best for:** Parallel evaluation with many CPU workers

```bash
./venv/bin/python train_5v5_evo.py --generations 100 --population 20 --workers 32
```

| Parameter | Default | Description | Recommendation |
|-----------|---------|-------------|----------------|
| `--generations` | 50 | Evolution cycles | 50-100 |
| `--population` | 10 | Teams per architecture | 10-20 |
| `--games_per_match` | 3 | Games per matchup | 3-5 |
| `--workers` | 8 | Parallel CPU workers | Match CPU cores |
| `--checkpoint_dir` | `checkpoints_5v5` | Save location | — |

---

## Recommended Training Pipeline

### Option A: Pure Self-Play
```bash
# 1. Evolution to find initial strategies
./venv/bin/python train_5v5_evo_gpu.py --generations 100 --population 15
```
### 2. MAPPO (Multi-Agent PPO) - **STABILIZED**
**Script:** `train_5v5_mappo_gpu.py`

This script now uses **Mixed Opponent Training**:
- **50% Games vs Self:** Learns new strategies.
- **50% Games vs Legacy:** Grounds the agent against known strong baselines (DPS, Tank, Support), preventing "self-play collapse" where agents learn weird behaviors that only work against themselves.
- **Centralized Critic:** Learns a value function from the perspective of the entire team.

**Recommended Command:**
```bash
python3 train_5v5_mappo_gpu.py --iterations 200 --episodes 20 --lr 0.0001
```

### 3. Population-Based Training (PBT) - **FIXED**
**Script:** `train_5v5_pbt_gpu.py`

PBT now performs **Hybrid Evaluation**:
- **Training:** Agents play against each other (population) AND Legacy agents (20% mix).
- **Evaluation:** Agents are ranked **exclusively** by their performance against the Legacy Squad. This ensures that "high fitness" means "beats the baseline," not just "beats a random bad agent."
- **Evolution:** The bottom 20% of the population copies the top 20% and mutates hyperparameters.

**Recommended Command:**
```bash
python3 train_5v5_pbt_gpu.py --population 20 --iterations 200 --workers 32
```


### 4. Deep Q-Learning (DQN) - **NEW**
**Script:** `train_5v5_dqn_gpu.py`

Trains using **Pure Q-Learning** (off-policy, value-based) instead of PPO.
- **Pros:** Sample efficient (uses replay buffer), explicitly learns "Value of Action".
- **Cons:** Can be unstable if hyperparameters aren't tuned (we use Target Networks to fix this).
- **Opponents:** 50% Self-Playing (Target Net), 50% Legacy Squad (DPS/Tank/Supp).

**Recommended Command:**
```bash
python3 train_5v5_dqn_gpu.py --iterations 200 --episodes 20 --lr 0.0001
```

### Option B: Learn from Legacy
```bash
# 1. Train Legacy with CPU evolution (overnight)
./venv/bin/python train_5v5_evo.py --generations 200 --population 20 --workers 32

# 2. Imitate Legacy and surpass it
./venv/bin/python train_5v5_imitation_gpu.py --imitation_episodes 500 --finetune_iters 200
```

---

## GPU Memory Guidelines

| Script | Pop=5 | Pop=10 | Pop=20 |
|--------|-------|--------|--------|
| Evolution GPU | ~1GB | ~2GB | ~4GB |
| MAPPO | ~1GB | ~1GB | ~1GB |
| PBT GPU | ~2GB | ~3GB | ~5GB |
| Imitation GPU | ~1GB | ~1GB | ~1GB |

Your RTX 4500 Ada (24GB) can handle any configuration.

---

## Quick Reference

| Goal | Script | Key Params |
|------|--------|------------|
| Explore strategies | `train_5v5_evo_gpu.py` | `--generations 100 --population 15` |
| Team coordination | `train_5v5_mappo_gpu.py` | `--iterations 300 --episodes 20` |
| Auto-tune hyperparams | `train_5v5_pbt_gpu.py` | `--iterations 100 --population 10` |
| Bootstrap from Legacy | `train_5v5_imitation_gpu.py` | `--imitation_episodes 500` |
| Parallel CPU training | `train_5v5_evo.py` | `--workers 32 --population 20` |
