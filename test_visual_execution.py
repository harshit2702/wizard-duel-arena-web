import sys
import os
from colorama import init
init(autoreset=True)

# Ensure checking in the correct directory
sys.path.append("/home/qsdal2/Desktop/bookGame/oneVsMany")
os.chdir("/home/qsdal2/Desktop/bookGame/oneVsMany")

try:
    from visual_duel_v8 import VisualDuelV8, create_wizard
except ImportError as e:
    print(f"ImportError: {e}")
    sys.exit(1)

class TestDuel(VisualDuelV8):
    def setup(self):
        print("Mocking setup...")
        # Programmatic setup
        self.team_a = [create_wizard("A1", 0, 0, True), create_wizard("A2", 0, 1, True)]
        self.team_b = [create_wizard("B1", 1, 10, False), create_wizard("B2", 1, 11, False)]
        self.model_a = "random"
        self.model_b = "random"
        self.teammate_ai = "random"
        self.brains_a = [None, None]
        self.brains_b = [None, None]
        self.max_turns = 1
        
    def run_test(self):
        self.setup()
        print("Setup complete. Executing turn...")
        try:
            self.execute_turn()
            print("Turn executed successfully.")
        except Exception as e:
            print(f"FAILED: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    game = TestDuel()
    game.run_test()
