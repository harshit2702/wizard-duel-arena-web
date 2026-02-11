import requests
import json
import time

BASE_URL = "http://localhost:8901"

def test_create_game():
    payload = {
        "team_a": {
            "size": 1,
            "control": "player",
            "avatars": [{"name": "Player1", "archetype": "auror", "avatar_id": "harry"}]
        },
        "team_b": {
            "size": 1,
            "control": "random",
            "avatars": [{"name": "Enemy1", "archetype": "death_eater", "avatar_id": "voldemort"}]
        }
    }
    
    print(f"POST {BASE_URL}/api/create")
    try:
        resp = requests.post(f"{BASE_URL}/api/create", json=payload)
        resp.raise_for_status()
        data = resp.json()
        print("Success:", data)
        return data["session_id"]
    except Exception as e:
        print("Failed to create game:", e)
        # return None to indicate failure
        return None

def test_get_state(session_id):
    print(f"GET {BASE_URL}/api/state/{session_id}")
    try:
        resp = requests.get(f"{BASE_URL}/api/state/{session_id}")
        resp.raise_for_status()
        data = resp.json()
        print("Success. Turn:", data["turn"])
        print("Logs:", len(data["logs"]))
        print("Team B (Masked):", data["team_b"][0])
    except Exception as e:
        print("Failed to get state:", e)

def main():
    session_id = test_create_game()
    if session_id:
        test_get_state(session_id)

if __name__ == "__main__":
    main()
