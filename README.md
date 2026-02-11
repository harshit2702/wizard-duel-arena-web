# Wizard Duel: One vs Many (Web Edition)

A magical 1v5 wizard duel simulation with a FastAPI backend and Vanilla JS frontend.

## 📦 Features
- **Web Interface**: Play physically or simulate AI battles in the browser.
- **Unified Brain AI**: Advanced AI trained via self-play evolution.
- **Fog of War**: Hidden enemy stats until revealed.
- **Spell System**: 12+ spells including *Avada Kedavra*, *Protego*, and *Expelliarmus*.

## 🚀 Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Game Server**:
   ```bash
   python3 backend/server.py
   ```

3. **Play**:
   Open [http://localhost:8901](http://localhost:8901) in your browser.

## 🧠 Training the AI
To train the AI models from scratch using GPU acceleration:
```bash
python3 train_5v5_evo_gpu.py --generations 50 --population 20
```
This will produce new checkpoints in `checkpoints_evo_gpu/`.

## 📂 Structure
- `backend/`: FastAPI server and game session management.
- `frontend/`: HTML/CSS/JS for the game UI.
- `duel_engine.py`: Core game mechanics and spell logic.
- `checkpoints_5v5/`: Pre-trained AI models.
