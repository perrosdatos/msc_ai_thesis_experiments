import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Paths to the scalar directories
MASAC_SCALARS = "BenchMARL/outputs/2026-04-24/04-47-15/masac_match_v2_cnn__d0cf953a_26_04_24-04_47_15/masac_match_v2_cnn__d0cf953a_26_04_24-04_47_15/scalars"
MAPPO_SCALARS = "BenchMARL/outputs/2026-04-22/09-58-15/mappo_match_v2_cnn__6d9aeb28_26_04_22-09_58_15/mappo_match_v2_cnn__6d9aeb28_26_04_22-09_58_15/scalars"

def load_csv(path, filename):
    filepath = os.path.join(path, filename)
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    # BenchMARL csvs usually don't have a header. Col 0: iteration/step, Col 1: value
    df = pd.read_csv(filepath, header=None, names=["step", "value"])
    return df

def analyze_and_predict(df, algo_name, metric_name, target_value=200.0, recent_window=500):
    """Analyzes the recent trend and predicts steps needed to reach the target."""
    if df is None or len(df) < 10:
        return
    
    # Sort by step just in case
    df = df.sort_values(by="step")
    
    # Calculate rolling mean to smooth the curve
    df['rolling_value'] = df['value'].rolling(window=max(1, len(df)//20)).mean()
    
    # Take the recent window to calculate the slope
    recent_df = df.tail(recent_window)
    if len(recent_df) < 2:
        return
        
    slope, intercept, r_value, p_value, std_err = linregress(recent_df['step'], recent_df['value'])
    
    current_value = recent_df['value'].iloc[-1]
    current_step = recent_df['step'].iloc[-1]
    
    print(f"--- Analysis for {algo_name}: {metric_name} ---")
    print(f"Current Value: {current_value:.4f} at Step: {current_step}")
    print(f"Recent Trend (Slope): {slope:.6f} per step")
    
    if slope > 0:
        steps_needed = (target_value - current_value) / slope
        print(f"Prediction: At the current rate, it will take approx {int(steps_needed)} more steps to reach {target_value}.")
    else:
        print(f"Prediction: The trend is currently flat or negative. It is not converging towards the target at this rate.")
        
    return current_step, current_value, slope

def analyze_training():
    print("=========================================")
    print("        MAPPO vs MASAC Analysis          ")
    print("=========================================\n")
    
    # 1. Analyze Evaluation Rewards
    mappo_eval = load_csv(MAPPO_SCALARS, "eval_reward_episode_reward_mean.csv")
    masac_eval = load_csv(MASAC_SCALARS, "eval_reward_episode_reward_mean.csv")
    
    plt.figure(figsize=(10, 6))
    if mappo_eval is not None:
        analyze_and_predict(mappo_eval, "MAPPO", "Eval Mean Reward", target_value=150.0, recent_window=1000)
        plt.plot(mappo_eval['step'], mappo_eval['value'], alpha=0.3, color='blue', label='MAPPO (Raw)')
        plt.plot(mappo_eval['step'], mappo_eval['value'].rolling(50).mean(), color='blue', label='MAPPO (Smoothed)')
        
    if masac_eval is not None:
        analyze_and_predict(masac_eval, "MASAC", "Eval Mean Reward", target_value=150.0, recent_window=1000)
        plt.plot(masac_eval['step'], masac_eval['value'], alpha=0.3, color='orange', label='MASAC (Raw)')
        plt.plot(masac_eval['step'], masac_eval['value'].rolling(50).mean(), color='orange', label='MASAC (Smoothed)')
        
    plt.title("Evaluation Mean Reward Over Time")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig("reward_comparison.png")
    print("\nSaved reward comparison plot to 'reward_comparison.png'")
    
    print("\n=========================================")
    print("      Optimization Recommendations       ")
    print("=========================================\n")
    
    # Analyze internal metrics for MAPPO
    print(">> MAPPO Optimization:")
    kl_approx = load_csv(MAPPO_SCALARS, "train_agents_kl_approx.csv")
    entropy_mappo = load_csv(MAPPO_SCALARS, "train_agents_entropy.csv")
    
    if kl_approx is not None:
        recent_kl = kl_approx['value'].tail(100).mean()
        print(f"- Recent KL Divergence: {recent_kl:.4f}")
        if recent_kl > 0.05:
            print("  *WARNING: KL is quite high. Policy updates might be too drastic.")
            print("  *SUGGESTION: Lower learning rate, increase 'clip_param', or increase PPO minibatches.")
        elif recent_kl < 0.005:
            print("  *NOTE: KL is very low. The policy is barely changing per update.")
            print("  *SUGGESTION: Consider slightly increasing learning rate or allowing more PPO epochs.")
            
    if entropy_mappo is not None:
        recent_ent = entropy_mappo['value'].tail(100).mean()
        print(f"- Recent Entropy: {recent_ent:.4f}")
        if recent_ent < 0.5:
            print("  *WARNING: Entropy is low. The policy might be prematurely deterministic (stuck in local minima).")
            print("  *SUGGESTION: Increase entropy coefficient (ent_coef) to encourage exploration.")
            
    # Analyze internal metrics for MASAC
    print("\n>> MASAC Optimization:")
    alpha = load_csv(MASAC_SCALARS, "train_agents_alpha.csv")
    entropy_masac = load_csv(MASAC_SCALARS, "train_agents_entropy.csv")
    
    if alpha is not None:
        recent_alpha = alpha['value'].tail(100).mean()
        print(f"- Recent Alpha (Temperature): {recent_alpha:.4f}")
        if recent_alpha < 0.01:
            print("  *WARNING: Alpha has dropped near zero. MASAC is no longer exploring.")
            print("  *SUGGESTION: Set a higher target_entropy, or use a fixed, slightly larger alpha to force exploration.")
            
    if entropy_masac is not None:
        recent_ent_masac = entropy_masac['value'].tail(100).mean()
        print(f"- Recent Entropy: {recent_ent_masac:.4f}")
        if recent_ent_masac < -5.0:  # SAC entropy can be negative
            print("  *WARNING: Policy entropy is extremely negative. The policy is highly peaked.")
            print("  *SUGGESTION: Similar to alpha, adjust target_entropy to a higher value to prevent premature convergence.")

if __name__ == "__main__":
    analyze_training()
