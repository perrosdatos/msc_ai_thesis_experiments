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
        return sorted(matches)[-1] # get most recent if multiple
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

def evaluate_matchup(algo_a, policy_a, env_a, algo_b, policy_b, chkpt_idx, seeds_df, swap=False):
    num_seeds = len(seeds_df)
    env = env_a
    tracker = LuxKPITracker(batch_size=num_seeds)
    
    # Optional: we can try to force the seeds if the wrapper supports it,
    # but since JAX vectorizes, the environment generates its own internal keys using the env base seed.
    # For absolute exact seeds per batch, we'd need to modify `torchrl.envs.JaxEnv`. 
    # For now, the batch index itself effectively acts as our 50 reproducible seeds because `reset()` is deterministic based on the global key.
    
    # Monkey-patch jax.random.split temporarily to force EXACT seeds from the CSV for each batch element
    import jax
    import jax.numpy as jnp
    original_split = jax.random.split
    def patched_split(key, num=2):
        if num == num_seeds:
            return jnp.stack([jax.random.PRNGKey(int(seeds_df.iloc[b]["seed"])) for b in range(num_seeds)])
        return original_split(key, num)
        
    jax.random.split = patched_split
    try:
        td = env.reset()
    finally:
        jax.random.split = original_split
    step_count = 0
    
    # We must construct the action for team A and team B natively
    # Team 0 is Policy A, Team 1 is Policy B (if not swapped)
    p0_policy = policy_a if not swap else policy_b
    p1_policy = policy_b if not swap else policy_a
    
    # Base environment reference to inject opp_actions
    base_lux = env.base_env
    base_lux.rulebased_agent_class = None
    
    # Video generation for the first batch element (Seed 42)
    record_video = (not swap)  # Only record one orientation to save space
    frames = []
    
    pbar = tqdm(total=750, desc=f"Evaluating {algo_a} vs {algo_b} (Swap={swap}) Ckpt {chkpt_idx}", leave=False)
    
    while True:
        if record_video:
            frame_td = env.render("rgb_array")
            if frame_td is not None:
                frames.append(frame_td.cpu().numpy())
                
        # Team 0 Action
        with torch.no_grad():
            from torchrl.envs.utils import set_exploration_type, ExplorationType
            with set_exploration_type(ExplorationType.DETERMINISTIC):
                td = p0_policy(td)
                
        # Construct Team 1 Observation to feed to p1_policy
        t_ids_np = base_lux.team_ids.cpu().numpy()
        opp_team_ids = 1 - t_ids_np
        
        u_p0 = base_lux._get_v(base_lux.jax_obs["player_0"], "units")
        u_p1 = base_lux._get_v(base_lux.jax_obs["player_1"], "units")
        pos0 = np.asarray(base_lux._get_v(u_p0, "position"), np.int32)
        pos1 = np.asarray(base_lux._get_v(u_p1, "position"), np.int32)
        opp_pos = np.where((opp_team_ids == 0)[:, None, None, None], pos0, pos1)
        
        obs_b_array = base_lux._build_spatial_observation(base_lux.jax_obs, opp_team_ids, opp_pos)
        
        m0 = np.asarray(base_lux._get_v(base_lux.jax_obs["player_0"], "units_mask"), bool)
        m1 = np.asarray(base_lux._get_v(base_lux.jax_obs["player_1"], "units_mask"), bool)
        m_opp = np.where((opp_team_ids == 0)[:, None, None], m0, m1)
        
        active_units_B = np.zeros((num_seeds, base_lux.max_units), dtype=bool)
        for b in range(num_seeds):
            active_units_B[b, :] = m_opp[b, 0, :]
            
        action_mask_B = np.zeros((num_seeds, base_lux.max_units, 5), dtype=np.bool_)
        action_mask_B[active_units_B, :] = True
        action_mask_B[~active_units_B, 0] = True
        
        td_b = TensorDict({
            "agents": TensorDict({
                "observation": torch.tensor(obs_b_array, device=env.device),
                "action_mask": torch.tensor(action_mask_B, device=env.device)
            }, batch_size=[num_seeds, base_lux.max_units])
        }, batch_size=[num_seeds])
        
        with torch.no_grad():
            with set_exploration_type(ExplorationType.DETERMINISTIC):
                td_b = p1_policy(td_b)
                action_b = td_b.get(("agents", "action")).cpu().numpy()
                
        if action_b.ndim == 3:
            action_b = action_b.squeeze(-1)
            
        base_lux.opp_actions = action_b
        
        td = env.step(td)
        rc = env.last_reward_components
        tracker.update(env, td, rc)
        
        step_count += 1
        pbar.update(1)
        td = td.get("next")
        if td["done"].any():
            break
            
    pbar.close()
    # Extract results first so we know who was Team 0 for the video filename
    raw_results = tracker.get_results()
    
    # Determine who is Team 0 for the video (batch index 0)
    t_ids_np = base_lux.team_ids.cpu().numpy()
    p0_is_team = t_ids_np[0]
    p0_algo = algo_b if swap else algo_a
    p1_algo = algo_a if swap else algo_b
    vid_t0_algo = p0_algo if p0_is_team == 0 else p1_algo
    vid_t1_algo = p1_algo if p0_is_team == 0 else p0_algo
    
    if record_video and len(frames) > 0:
        import imageio
        seed_0 = seeds_df.iloc[0]["seed"]
        video_dir = "/home/carlos/Documents/github/msc_ai_thesis_experiments/performance_analysis/videos"
        os.makedirs(video_dir, exist_ok=True)
        vid_path = os.path.join(video_dir, f"ckpt_{chkpt_idx}_seed_{seed_0}_T0_{vid_t0_algo}_vs_T1_{vid_t1_algo}.mp4")
        imageio.mimsave(vid_path, frames, fps=10)
    
    # Format into dataframe rows
    rows = []
    for b in range(num_seeds):
        res = raw_results[b]
        
        # Determine who is Team 0 and Team 1 ON THE MAP
        # p0_policy is assigned to team_ids[b].
        t_ids_np = base_lux.team_ids.cpu().numpy()
        p0_is_team = t_ids_np[b]
        
        p0_algo = algo_b if swap else algo_a
        p1_algo = algo_a if swap else algo_b
        
        if p0_is_team == 0:
            m0_algo = p0_algo
            m1_algo = p1_algo
        else:
            m0_algo = p1_algo
            m1_algo = p0_algo
        
        row = {
            "seed_id": seeds_df.iloc[b]["seed"],
            "batch_idx": b,
            "chkpt_idx": chkpt_idx,
            "team_0_model": m0_algo,
            "team_1_model": m1_algo,
            "match_swapped": swap,
        }
        
        # Determine winner
        p0_pts = res["team_0_total_points"]
        p1_pts = res["team_1_total_points"]
        if p0_pts > p1_pts:
            row["winner_team"] = 0
            row["winner_model"] = m0_algo
        elif p1_pts > p0_pts:
            row["winner_team"] = 1
            row["winner_model"] = m1_algo
        else:
            row["winner_team"] = -1
            row["winner_model"] = "Tie"
            
        # Add all KPIs
        for k, v in res.items():
            # Convert arrays to list/str or just scalar sum? Individual KPIs are scalar per agent
            row[k] = v
            
        rows.append(row)
        
    return rows

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=str, default="1,20,40", help="Comma separated checkpoint indices (1 to 40)")
    args = parser.parse_args()
    
    chkpts = [int(x) for x in args.checkpoints.split(",")]
    
    seeds_path = "/home/carlos/Documents/github/msc_ai_thesis_experiments/performance_analysis/seeds.csv"
    if not os.path.exists(seeds_path):
        print("Run generate_seeds.py first!")
        return
        
    seeds_df = pd.read_csv(seeds_path)
    
    algos = ["mappo", "masac", "qmix"]
    pairs = [("mappo", "masac"), ("mappo", "qmix"), ("masac", "qmix")]
    
    out_dir = "/home/carlos/Documents/github/msc_ai_thesis_experiments/performance_analysis/csv_data"
    os.makedirs(out_dir, exist_ok=True)
    csv_file = os.path.join(out_dir, "raw_sweep_data.csv")
    
    all_rows = []
    completed_matchups = set()
    
    if os.path.exists(csv_file):
        print(f"📥 Found existing CSV data at {csv_file}. Loading progress to resume...")
        df_exist = pd.read_csv(csv_file)
        all_rows = df_exist.to_dict("records")
        for _, row in df_exist.iterrows():
            # Dynamically sort model names to match the alphabetical pairs structure
            models = sorted([row["team_0_model"], row["team_1_model"]])
            completed_matchups.add((row["chkpt_idx"], models[0], models[1]))
            
    
    for c in chkpts:
        print(f"\n=============================================")
        print(f"🔄 SWEEPING CHECKPOINT {c}/40")
        print(f"=============================================")
        
        # Fast-forward check: If all pairs for this checkpoint are done, skip loading policies entirely
        all_completed = True
        for a_name, b_name in pairs:
            if (c, a_name, b_name) not in completed_matchups:
                all_completed = False
                break
                
        if all_completed:
            print(f"⏭️  SKIPPING: Checkpoint {c} (All matchups already evaluated in CSV)")
            continue
            
        # Load policies for this checkpoint
        policies = {}
        for algo in algos:
            p_path = get_checkpoint_path(algo, c)
            if p_path is None:
                print(f"⚠️ WARNING: Checkpoint {c} for {algo} not found! Skipping...")
                policies[algo] = None
            else:
                print(f"Loading {algo} from {p_path}...")
                policies[algo] = load_policy(algo, p_path)
                
        for a_name, b_name in pairs:
            if (c, a_name, b_name) in completed_matchups:
                print(f"\n⏭️  SKIPPING: {a_name} vs {b_name} (Already evaluated in CSV)")
                continue
                
            if policies.get(a_name) is None or policies.get(b_name) is None:
                continue
                
            pol_a, env_a = policies.get(a_name)
            pol_b, _ = policies.get(b_name)
            
            print(f"\n⚔️  MATCHUP: {a_name} vs {b_name}")
            
            # Match 1: Normal (A=Team0, B=Team1)
            rows_normal = evaluate_matchup(a_name, pol_a, env_a, b_name, pol_b, c, seeds_df, swap=False)
            all_rows.extend(rows_normal)
            
            # Match 2: Swapped (B=Team0, A=Team1)
            rows_swap = evaluate_matchup(a_name, pol_a, env_a, b_name, pol_b, c, seeds_df, swap=True)
            all_rows.extend(rows_swap)
            
            # Save intermediate
            df = pd.DataFrame(all_rows)
            df.to_csv(os.path.join(out_dir, "raw_sweep_data.csv"), index=False)
            
    print("\n✅ Sweep Complete! Data saved to csv_data/raw_sweep_data.csv")
    
    # Cleanup garbage folders generated by BenchMARL experiment init
    import glob
    import shutil
    trash = glob.glob("*match_v2_cnn*")
    for t in trash:
        if os.path.isdir(t):
            shutil.rmtree(t)
    print("🗑️ Cleaned up empty log folders.")

if __name__ == "__main__":
    main()
