import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../BenchMARL')))

import torch
import numpy as np
from benchmarl.environments.lux.lux import LuxTorchRLEnv
from performance_analysis.kpi_tracker import LuxKPITracker

def main():
    print("Testing KPI Tracker Math Logic...")
    batch_size = 2
    env = LuxTorchRLEnv(batch_size=batch_size, reward_version="v2", max_steps=100)
    tracker = LuxKPITracker(batch_size=batch_size)
    
    td = env.reset()
    
    # We will simulate 100 steps of pure random actions + rulebased
    print("Simulating matches...")
    for step in range(100):
        # We just generate random actions for player 0 (Team 0)
        actions = np.random.randint(0, 5, size=(batch_size, 16))
        td["agents", "action"] = torch.tensor(actions, device=env.device)
        
        td = env.step(td)
        
        # Extract reward components
        rc = env.last_reward_components
        
        # Update KPIs
        tracker.update(env, td, rc)
        
        if td["done"][0].item():
            break
            
    print("\nSimulation Complete. Extracting KPIs...")
    results = tracker.get_results()
    
    print("\n=== BATCH 0, TEAM 0 RESULTS ===")
    for k, v in results[0].items():
        if k.startswith("team_0_agent_0_"):
            print(f"  {k}: {v:.2f}")
        elif not "agent" in k and k.startswith("team_0_"):
            print(f"{k}: {v:.2f}")

    print("\n✅ KPI Tracker math logic passed without crashing!")

if __name__ == "__main__":
    main()
