import requests
import json
import time
import sys

BASE_URL = "http://localhost:8901"

def log(msg):
    print(f"[TEST] {msg}")

def test_flow():
    # 1. Check Index
    try:
        log("Fetching index.html...")
        r = requests.get(f"{BASE_URL}/")
        r.raise_for_status()
        log("Index OK (200)")
    except Exception as e:
        log(f"Index FAILED: {e}")
        return

    # 2. Create Game
    payload = {
        "team_a": {
            "size": 1,
            "control": "player",
            "avatars": [{"name": "Hero", "archetype": "auror", "avatar_id": "harry"}],
            "variant": {"player_position": 1}
        },
        "team_b": {
            "size": 1,
            "control": "random", # Use random to avoid brain loading issues for this test
            "avatars": [{"name": "Enemy", "archetype": "default", "avatar_id": "default"}]
        }
    }
    
    session_id = None
    try:
        log("Creating Game...")
        r = requests.post(f"{BASE_URL}/api/create", json=payload)
        if r.status_code != 200:
            log(f"Create FAILED: {r.status_code} {r.text}")
            return
        data = r.json()
        session_id = data["session_id"]
        log(f"Game Created: {session_id}")
    except Exception as e:
        log(f"Create CRASHED: {e}")
        return

    # 3. Poll State
    try:
        log(f"Polling State for {session_id}...")
        r = requests.get(f"{BASE_URL}/api/state/{session_id}")
        if r.status_code != 200:
            log(f"Poll FAILED: {r.status_code} {r.text}")
            return
        state = r.json()
        log("State Received.")
        
        # 4. Validate Structure for Frontend
        required_keys = ["turn", "max_turns", "logs", "team_a", "team_b", "animation_events"]
        missing = [k for k in required_keys if k not in state]
        if missing:
            log(f"State MISSING keys: {missing}")
            return
            
        log(f"Turn: {state['turn']}")
        log(f"Team A: {len(state['team_a'])} wizards")
        log(f"Team B: {len(state['team_b'])} wizards")
        
        # Check specific wizard fields needed by app.js
        w = state['team_a'][0]
        wiz_keys = ["id", "name", "hp", "max_hp", "posture", "max_posture", "status"]
        w_missing = [k for k in wiz_keys if k not in w]
        if w_missing:
            log(f"Wizard MISSING keys: {w_missing}")
            return
            
        log("State Structure OK.")
        
    except Exception as e:
        log(f"Poll CRASHED: {e}")
        return

if __name__ == "__main__":
    test_flow()
