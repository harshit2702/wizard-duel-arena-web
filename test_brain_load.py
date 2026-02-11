import sys
import os
import torch

# Add path
sys.path.append(os.path.abspath('.'))

try:
    from unified_brain_v2 import UnifiedBrainV2
    print("Import successful")
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

def test_load():
    print("Attempting to init Brain...")
    try:
        brain = UnifiedBrainV2()
        print("Brain initialized.")
    except Exception as e:
        print(f"Brain init failed: {e}")
        return

    checkpoint_path = "checkpoints_5v5/unified_best_0.pth"
    if os.path.exists(checkpoint_path):
        print(f"Loading {checkpoint_path}...")
        try:
            brain.load_state_dict(torch.load(checkpoint_path, weights_only=True))
            print("Load successful.")
        except Exception as e:
            print(f"Load failed: {e}")
    else:
        print(f"{checkpoint_path} not found.")

if __name__ == "__main__":
    test_load()
