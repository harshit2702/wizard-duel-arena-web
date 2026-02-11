---
title: Wizard Duel 5v5 Arena
emoji: 🧙‍♂️
colorFrom: purple
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

# Wizard Duel 5v5 Arena 🧙‍♂️⚡

**A strategic 5v5 magical combat simulation powered by Deep Reinforcement Learning.**

This project simulates intense wizard duels where teams of 5 AI agents (or human players) compete using a complex spell system. The AI is trained using advanced RL techniques like **Evolutionary Strategies**, **PBT (Population-Based Training)**, **MAPPO (Multi-Agent PPO)**, and **DQN**.

---

## 🌟 Features

- **5v5 Team Combat**: Strategic battles requiring coordination, positioning, and resource management.
- **Advanced AI**: Agents trained via self-play and league-based training (Legacy Squads).
- **Web Interface**: A modern Vanilla JS + FastAPI web app to watch or play matches in the browser.
- **Real-time Visualizer**: High-performance PyGame visualizer (`visual_duel_v8.py`) for observing AI behavior.
- **Tournament System**: Automated arena to pit different AI models against each other.
- **Complex Spell Engine**: 12+ spells interacting via a priority and physics system (e.g., *Levioso* makes targets vulnerable to *Descendo*).

---

## 🚀 Installation

### Prerequisites
- Python 3.8+
- CUDA-capable GPU (Recommended for training)

### Setup
1.  **Clone the repository**:
    ```bash
    git clone <repository_url>
    cd oneVsMany
    ```

2.  **Create a virtual environment**:
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # Linux/Mac
    # venv\Scripts\activate  # Windows
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

---

## 🎮 Usage Guide

### 1. Web Interface (Play/Watch)
Run the game server to play against the AI or watch matches in a browser.

```bash
python3 backend/server.py
```
- Open **[http://localhost:8901](http://localhost:8901)** in your browser.
- **Controls**: Click spell icons to cast. Select targets by clicking on them.

### 2. Visualizer (Fast Simulation)
Watch the AI battle in real-time using the PyGame visualizer. This is the best way to debug AI behavior.

```bash
python3 visual_duel_v8.py
```
- **Keybindings**:
    - `SPACE`: Pause/Resume
    - `R`: Reset
    - `S`: Step (when paused)

### 3. Tournament Arena
Run an automated tournament to evaluate different model checkpoints.

```bash
python3 tournament_arena_v3.py
```
- Comparison includes: **Evo Best**, **MAPPO**, **PBT**, **DQN**, and **Imitation** models.
- Results are saved to `tournament_results.md`.

---

## 🧠 AI Training

We use a multi-stage training pipeline to create robust agents.

**Supported Algorithms:**
- **Evolution**: Genetic algorithms for initial strategy discovery.
- **MAPPO**: Multi-Agent PPO for team coordination.
- **PBT**: Population-Based Training for hyperparameter optimization.
- **DQN**: Deep Q-Network for value-based decision making.
- **Imitation**: Behavioral cloning from rule-based "Legacy" experts.

📖 **[Read the Full Training Guide](TRAINING_GUIDE.md)** for detailed commands and hyperparameters.

**Quick Start (Evolution on GPU):**
```bash
python3 train_5v5_evo_gpu.py --generations 50 --population 20
```

---

## ⚡ Spell System

Spells resolve based on a **Priority System** (Lower = Faster).

| Priority | Type | Examples | Description |
| :--- | :--- | :--- | :--- |
| **1** | Defense | `Protego`, `Protego Maximus` | Block damage and reflect control spells. |
| **2** | Info | `Revelio`, `Legilimens` | Reveal enemy stats and positions. |
| **3** | Control | `Levioso`, `Glacius` | Disable enemies (Airborne, Frozen). |
| **4** | Force | `Accio`, `Descendo`, `Depulso` | Move units. *Descendo* combos with *Levioso*. |
| **5** | Damage | `Confringo`, `Incendio`, `Avada Kedavra` | Deal HP damage. *Avada* kills broken posture units. |

📖 **[View Full Spell Logic](SPELL_LOGIC.md)** for damage numbers, costs, and combo mechanics.

---

## 📂 Project Structure

- **`backend/`**: FastAPI server and game session logic.
- **`frontend/`**: Web UI (HTML/CSS/JS).
- **`checkpoints_*/`**: Directory for saved model weights (e.g., `checkpoints_5v5/`).
- **`duel_engine.py`**: The core simulation engine.
- **`visual_duel_v8.py`**: PyGame visualizer.
- **`tournament_arena_v3.py`**: Model evaluation script.
- **`unified_brain_v2.py`**: Neural network architecture.

---

**Happy Dueling!** 🪄
