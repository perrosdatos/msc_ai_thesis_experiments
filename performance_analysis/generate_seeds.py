import os
import random
import pandas as pd

def main():
    # Setup
    output_dir = "/home/carlos/Documents/github/msc_ai_thesis_experiments/performance_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    csv_path = os.path.join(output_dir, "seeds.csv")
    
    # Requirements
    num_seeds = 50
    master_seed = 42
    
    # Reproducibility
    random.seed(master_seed)
    
    # Generate random unique seeds between 0 and 100_000
    seeds = random.sample(range(0, 100000), num_seeds)
    
    # Create DataFrame
    df = pd.DataFrame({
        "env_id": range(1, num_seeds + 1),
        "seed": seeds
    })
    
    # Export
    df.to_csv(csv_path, index=False)
    print(f"✅ Successfully generated {num_seeds} reproducible seeds at: {csv_path}")
    print(df.head())

if __name__ == "__main__":
    main()
