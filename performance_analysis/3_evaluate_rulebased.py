import sys
import os
import argparse
import pandas as pd
import numpy as np
import torch
import hydra
from omegaconf import OmegaConf
from hydra.core.global_hydra import GlobalHydra
from tensordict import TensorDict
from tqdm import tqdm

os.environ["WANDB_MODE"] = "disabled"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../BenchMARL')))
from benchmarl.hydra_config import load_experiment_from_hydra
from performance_analysis.kpi_tracker import LuxKPITracker

def get_checkpoint_path(algo, checkpoint_idx):
    base_dir = "/home/carlos/Documents/github/msc_ai_thesis_experiments/performance_analysis/models"
    import glob
    pattern = f"{base_dir}/*{algo}_match_v2_cnn*/checkpoints/checkpoint_{checkpoint_idx * 150000}.pt"
    matches = glob.glob(pattern, recursive=True)
    if matches:
        return sorted(matches)[-1]
    return None

def load_policy(algo, checkpoint_path):
    benchmarl_conf_path = "../BenchMARL/benchmarl/conf"
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
        
    with hydra.initialize(version_base=None, config_path=benchmarl_conf_path):
        cfg = hydra.compose(
            config_name="config",
            overrides=[
                f"algorithm={algo}", "task=lux/match_v2", "model=layers/cnn_lux_16ch",
                "model@critic_model=layers/cnn_lux_16ch", "experiment.sampling_device=cpu",
                "experiment.train_device=cpu", "experiment.buffer_device=cpu",
                "experiment.checkpoint_interval=150000", "experiment.loggers=[]", "seed=42",
                "experiment.evaluation_episodes=50"
            ],
        )
        experiment = load_experiment_from_hydra(cfg, task_name="lux/match_v2")
        experiment.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False))
        return experiment.algorithm.get_policy_for_collection(), experiment.test_env

def evaluate_rulebased(algo, policy, env, chkpt_idx, seeds_df, force_team=0):
    num_seeds = len(seeds_df)
    tracker = LuxKPITracker(batch_size=num_seeds)
    
    # Monkey-patch jax.random.split temporarily to force EXACT seeds from the CSV
    import jax
    import jax.numpy as jnp
    original_split = jax.random.split
    def patched_split(key, num=2):
        if num == num_seeds:
            return jnp.stack([jax.random.PRNGKey(int(seeds_df.iloc[b]["seed"])) for b in range(num_seeds)])
        return original_split(key, num)
        
    # Monkey-patch torch.randint to force the RL model to play as a specific team (0 or 1)
    original_randint = torch.randint
    def patched_randint(low, high, size, **kwargs):
        if low == 0 and high == 2:
            # This intercepts the team_ids allocation in lux_env.py
            return torch.full(size, force_team, dtype=torch.int64, **kwargs)
        return original_randint(low, high, size, **kwargs)
        
    jax.random.split = patched_split
    torch.randint = patched_randint
    try:
        td = env.reset()
    finally:
        jax.random.split = original_split
        torch.randint = original_randint
        
    step_count = 0
    
    # Base environment reference
    base_lux = env.base_env
    
    # Video generation for the first batch element
    record_video = True
    frames = []
    
    pbar = tqdm(total=750, desc=f"Evaluating {algo} (Ckpt {chkpt_idx}) vs Rule-Based [Playing as Team {force_team}]", leave=False)
    
    while True:
        if record_video:
            frame_td = env.render("rgb_array")
            if frame_td is not None:
                frames.append(frame_td.cpu().numpy())
                
        # Team 0 Action (Our trained policy)
        with torch.no_grad():
            from torchrl.envs.utils import set_exploration_type, ExplorationType
            with set_exploration_type(ExplorationType.DETERMINISTIC):
                td = policy(td)
                
        # We DO NOT inject opp_actions. The environment will query base_lux.rulebased_agent_class automatically.
        
        td = env.step(td)
        rc = env.last_reward_components
        tracker.update(env, td, rc)
        
        step_count += 1
        pbar.update(1)
        td = td.get("next")
        if td["done"].any():
            break
            
    pbar.close()
    
    raw_results = tracker.get_results()
    
    if record_video and len(frames) > 0:
        import imageio
        seed_0 = seeds_df.iloc[0]["seed"]
        video_dir = "/home/carlos/Documents/github/msc_ai_thesis_experiments/performance_analysis/videos"
        os.makedirs(video_dir, exist_ok=True)
        vid_path = os.path.join(video_dir, f"rulebased_ckpt_{chkpt_idx}_seed_{seed_0}_{algo}_as_Team{force_team}.mp4")
        imageio.mimsave(vid_path, frames, fps=10)
    
    rows = []
    for b in range(num_seeds):
        res = raw_results[b]
        
        # If the RL model played as Team 1 on the map, we MUST swap the metrics in `res`
        # so that `team_0_*` in the output CSV always represents the RL Model, 
        # and `team_1_*` always represents the Rule-Based agent.
        if force_team == 1:
            swapped_res = {}
            for k, v in res.items():
                if k.startswith("team_0_"):
                    swapped_res[k.replace("team_0_", "team_1_")] = v
                elif k.startswith("team_1_"):
                    swapped_res[k.replace("team_1_", "team_0_")] = v
                else:
                    swapped_res[k] = v
            res = swapped_res

        row = {
            "seed_id": seeds_df.iloc[b]["seed"],
            "batch_idx": b,
            "chkpt_idx": chkpt_idx,
            "team_0_model": algo,
            "team_1_model": "rule_based",
            "map_team_played": force_team,
        }
        
        p0_pts = res["team_0_total_points"]
        p1_pts = res["team_1_total_points"]
        if p0_pts > p1_pts:
            row["winner_team"] = 0
            row["winner_model"] = algo
        elif p1_pts > p0_pts:
            row["winner_team"] = 1
            row["winner_model"] = "rule_based"
        else:
            row["winner_team"] = -1
            row["winner_model"] = "Tie"
            
        for k, v in res.items():
            row[k] = v
            
        rows.append(row)
        
    return rows

def find_top_checkpoints(csv_path, algos, top_k=5):
    """Parses the raw sweep data to find the Top K checkpoints for each algorithm based on Average Points."""
    if not os.path.exists(csv_path):
        print(f"Cannot find {csv_path}. Please run full sweep first.")
        return {}
        
    df = pd.read_csv(csv_path)
    metrics = []
    
    for _, row in df.iterrows():
        # Add Team 0
        metrics.append({
            "Model": row["team_0_model"],
            "chkpt_idx": row["chkpt_idx"],
            "Points": row["team_0_total_points"]
        })
        # Add Team 1
        metrics.append({
            "Model": row["team_1_model"],
            "chkpt_idx": row["chkpt_idx"],
            "Points": row["team_1_total_points"]
        })
        
    df_m = pd.DataFrame(metrics)
    df_agg = df_m.groupby(["Model", "chkpt_idx"]).agg({"Points": "mean"}).reset_index()
    
    top_checkpoints = {}
    for algo in algos:
        df_algo = df_agg[df_agg["Model"] == algo].sort_values(by="Points", ascending=False)
        top_checkpoints[algo] = df_algo.head(top_k)["chkpt_idx"].tolist()
        
    return top_checkpoints

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", type=int, default=5, help="Number of top checkpoints to evaluate per model")
    args = parser.parse_args()
    
    seeds_path = "/home/carlos/Documents/github/msc_ai_thesis_experiments/performance_analysis/seeds.csv"
    if not os.path.exists(seeds_path):
        print("Run generate_seeds.py first!")
        return
        
    seeds_df = pd.read_csv(seeds_path)
    algos = ["mappo", "masac", "qmix"]
    
    sweep_csv = "/home/carlos/Documents/github/msc_ai_thesis_experiments/performance_analysis/csv_data/raw_sweep_data.csv"
    print("🔍 Discovering Top Checkpoints...")
    top_checkpoints = find_top_checkpoints(sweep_csv, algos, top_k=args.top_k)
    
    for algo, ckpts in top_checkpoints.items():
        print(f"🌟 Top {args.top_k} for {algo.upper()}: Checkpoints {ckpts}")
        
    out_dir = "/home/carlos/Documents/github/msc_ai_thesis_experiments/performance_analysis/csv_data"
    os.makedirs(out_dir, exist_ok=True)
    csv_file = os.path.join(out_dir, "rulebased_eval_data.csv")
    
    all_rows = []
    completed_evals = set()
    
    if os.path.exists(csv_file):
        print(f"📥 Found existing CSV data at {csv_file}. Loading progress to resume...")
        df_exist = pd.read_csv(csv_file)
        all_rows = df_exist.to_dict("records")
        for _, row in df_exist.iterrows():
            completed_evals.add((row["chkpt_idx"], row["team_0_model"]))
            
    for algo in algos:
        ckpts = top_checkpoints.get(algo, [])
        for c in ckpts:
            print(f"\n=============================================")
            print(f"🔄 EVALUATING {algo.upper()} (Ckpt {c}) vs RULE-BASED")
            print(f"=============================================")
            
            if (c, algo) in completed_evals:
                print(f"⏭️  SKIPPING: {algo} Ckpt {c} (Already evaluated in CSV)")
                continue
                
            p_path = get_checkpoint_path(algo, c)
            if p_path is None:
                print(f"⚠️ WARNING: Checkpoint {c} for {algo} not found! Skipping...")
                continue
                
            print(f"Loading {algo} from {p_path}...")
            policy, env = load_policy(algo, p_path)
            
            # Run Match 1: RL is Team 0
            rows_t0 = evaluate_rulebased(algo, policy, env, c, seeds_df, force_team=0)
            all_rows.extend(rows_t0)
            
            # Run Match 2: RL is Team 1
            rows_t1 = evaluate_rulebased(algo, policy, env, c, seeds_df, force_team=1)
            all_rows.extend(rows_t1)
            
            df = pd.DataFrame(all_rows)
            df.to_csv(csv_file, index=False)
            
    print("\n✅ Rule-Based Evaluation Complete! Data saved to csv_data/rulebased_eval_data.csv")
    
    # Cleanup garbage folders
    import glob
    import shutil
    trash = glob.glob("*match_v2_cnn*")
    for t in trash:
        if os.path.isdir(t):
            shutil.rmtree(t)

if __name__ == "__main__":
    main()
