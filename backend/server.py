from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uuid
import os

from game_session import GameSession

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store sessions in memory
sessions: Dict[str, GameSession] = {}

class AvatarConfig(BaseModel):
    name: str
    archetype: str
    avatar_id: str

class AllyAIConfig(BaseModel):
    control: str = "random"
    variant: Optional[Dict[str, Any]] = None

class TeamConfig(BaseModel):
    size: int
    control: str  # "player", "unified", "random", "legacy"
    variant: Optional[Dict[str, Any]] = None
    avatars: List[AvatarConfig]
    ally_ai: Optional[AllyAIConfig] = None  # For player teammates

class GameConfig(BaseModel):
    team_a: TeamConfig
    team_b: TeamConfig

class ActionRequest(BaseModel):
    session_id: str
    spell_name: str
    target_id: int

@app.post("/api/create")
def create_game(config: GameConfig):
    session_id = str(uuid.uuid4())
    session = GameSession(session_id)
    
    # Convert Pydantic models to dicts for setup_game
    cfg_dict = config.model_dump()
    session.setup_game(cfg_dict)
    
    sessions[session_id] = session
    
    # Find player wizard ID
    player_id = 0
    for w in session.team_a:
        if w.is_player:
            player_id = w.id
            break
    
    return {"session_id": session_id, "player_id": player_id, "message": "Game created"}

@app.get("/api/state/{session_id}")
def get_state(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    return session.get_state(player_perspective=True)

@app.post("/api/action")
def submit_action(req: ActionRequest):
    if req.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[req.session_id]
    session.process_turn(player_action=(req.spell_name, req.target_id))
    
    return session.get_state(player_perspective=True)

# Mount static files (Last to avoid shadowing API routes)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "../frontend")
app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8901)
