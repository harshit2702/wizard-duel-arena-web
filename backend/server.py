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

class DraftGameConfig(BaseModel):
    """Game config that includes a drafted spell deck"""
    team_a: TeamConfig
    team_b: TeamConfig
    drafted_deck: Optional[List[str]] = None  # List of spell names the player drafted

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

    # If session has a drafted deck, validate spell is in the deck
    if hasattr(session, 'drafted_deck') and session.drafted_deck:
        if req.spell_name not in session.drafted_deck:
            raise HTTPException(status_code=400, detail=f"Spell '{req.spell_name}' not in your drafted deck")

    session.process_turn(player_action=(req.spell_name, req.target_id))
    
    return session.get_state(player_perspective=True)

@app.post("/api/create_draft")
def create_draft_game(config: DraftGameConfig):
    """Create a game with a pre-drafted spell deck"""
    session_id = str(uuid.uuid4())
    session = GameSession(session_id)
    
    cfg_dict = config.model_dump()
    drafted = cfg_dict.pop('drafted_deck', None)
    session.setup_game(cfg_dict)
    
    # Store drafted deck on session
    if drafted:
        session.drafted_deck = drafted
    
    sessions[session_id] = session
    
    player_id = 0
    for w in session.team_a:
        if w.is_player:
            player_id = w.id
            break
    
    return {
        "session_id": session_id,
        "player_id": player_id,
        "message": "Draft game created",
        "drafted_deck": drafted or []
    }

# Mount static files (Last to avoid shadowing API routes)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
FRONTEND_DIR = os.path.join(BASE_DIR, "../frontend")
ASSETS_DIR = os.path.join(BASE_DIR, "../assets")

# Serve audio assets
if os.path.isdir(ASSETS_DIR):
    app.mount("/assets", StaticFiles(directory=ASSETS_DIR), name="assets")

app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8901)
