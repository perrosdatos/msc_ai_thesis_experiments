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
import json
import jax
import flax

os.environ["WANDB_MODE"] = "disabled"
os.environ["JAX_PLATFORMS"] = "cpu"

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../BenchMARL')))
from benchmarl.hydra_config import load_experiment_from_hydra
from luxai_s3.wrappers import serialize_env_states, serialize_env_actions

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
                "experiment.checkpoint_interval=150000", "experiment.loggers=[]", "seed=19941210",
                "experiment.evaluation_episodes=1"
            ],
        )
        experiment = load_experiment_from_hydra(cfg, task_name="lux/match_v2")
        experiment.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False))
        return experiment.algorithm.get_policy_for_collection(), experiment.test_env

def unbatch_jax_tree(tree, index=0):
    """Slices a batched JAX PyTree to get a single item, handling scalars."""
    def _slice(x):
        try:
            if hasattr(x, 'shape') and len(x.shape) > 0:
                return x[index]
            elif isinstance(x, (list, tuple)):
                return x[index]
            else:
                return x
        except Exception:
            return x
    return jax.tree_util.tree_map(_slice, tree)

def generate_replay(algo_a, ckpt_a, algo_b, ckpt_b, seed, output_path):
    print(f"Loading {algo_a.upper()} (Ckpt {ckpt_a})...")
    path_a = get_checkpoint_path(algo_a, ckpt_a)
    p0_policy, env = load_policy(algo_a, path_a)
    
    print(f"Loading {algo_b.upper()} (Ckpt {ckpt_b})...")
    path_b = get_checkpoint_path(algo_b, ckpt_b)
    p1_policy, _ = load_policy(algo_b, path_b)
    
    print(f"Running Match (Seed {seed})...")
    
    # We must force batch size 1 internally by manually slicing if needed, 
    # but the config overrides evaluation_episodes=1 so it's batch=1.
    
    # Setup seed
    import jax.numpy as jnp
    original_split = jax.random.split
    def patched_split(key, num=2):
        if num == 1:
            return jnp.stack([jax.random.PRNGKey(seed)])
        return original_split(key, num)
    jax.random.split = patched_split
    
    td = env.reset()
    jax.random.split = original_split
    
    base_lux = env.base_env
    base_lux.rulebased_agent_class = None
    
    episode_states = []
    episode_actions = []
    
    # Get initial state
    init_state = unbatch_jax_tree(base_lux.env_state, 0)
    episode_states.append(init_state)
    
    # Store initial params
    env_params = unbatch_jax_tree(base_lux.env_params, 0)
    
    pbar = tqdm(total=1000, desc=f"Evaluating Matchup")
    step_idx = 0
    while True:
        # Team 0 Action
        with torch.no_grad():
            from torchrl.envs.utils import set_exploration_type, ExplorationType
            with set_exploration_type(ExplorationType.DETERMINISTIC):
                td = p0_policy(td)
                
        # Construct Team 1 Observation
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
        
        active_units_B = np.zeros((1, base_lux.max_units), dtype=bool)
        active_units_B[0, :] = m_opp[0, 0, :]
            
        action_mask_B = np.zeros((1, base_lux.max_units, 5), dtype=np.bool_)
        action_mask_B[active_units_B, :] = True
        action_mask_B[~active_units_B, 0] = True
        
        td_b = TensorDict({
            "agents": TensorDict({
                "observation": torch.tensor(obs_b_array, device=env.device),
                "action_mask": torch.tensor(action_mask_B, device=env.device)
            }, batch_size=[1, base_lux.max_units])
        }, batch_size=[1])
        
        with torch.no_grad():
            with set_exploration_type(ExplorationType.DETERMINISTIC):
                td_b = p1_policy(td_b)
                action_b = td_b.get(("agents", "action")).cpu().numpy()
                
        if action_b.ndim == 3:
            action_b = action_b.squeeze(-1)
            
        base_lux.opp_actions = action_b
        
        # Save actions for replay (must format appropriately to [16, 3])
        a0 = td.get(("agents", "action")).cpu().numpy()[0]
        a1 = action_b[0]
        
        a0_3d = np.zeros((base_lux.max_units, 3), dtype=np.int32)
        a0_3d[:, 0] = a0
        
        a1_3d = np.zeros((base_lux.max_units, 3), dtype=np.int32)
        a1_3d[:, 0] = a1
        
        # Depending on which team is 0/1 in the JAX env
        team0_idx = t_ids_np[0]
        team1_idx = opp_team_ids[0]
        
        actions_dict = {"player_0": None, "player_1": None}
        if team0_idx == 0:
            actions_dict["player_0"] = a0_3d
            actions_dict["player_1"] = a1_3d
        else:
            actions_dict["player_1"] = a0_3d
            actions_dict["player_0"] = a1_3d
            
        episode_actions.append(actions_dict)
        
        # Step env
        td = env.step(td)
        
        # Store state
        episode_states.append(unbatch_jax_tree(base_lux.env_state, 0))
        
        td = td.get("next")
        pbar.update(1)
        step_idx += 1
        if td["done"].any() or step_idx >= 1000:
            break
            
    print(f"Match complete. Serializing HTML to {output_path}...")
    
    # Generate JSON dict
    replay = dict()
    replay["observations"] = serialize_env_states(episode_states)
    replay["actions"] = serialize_env_actions(episode_actions)
    replay["metadata"] = {
        "seed": seed,
        "players": {"player_0": f"{algo_a.upper()}_Ckpt{ckpt_a}", "player_1": f"{algo_b.upper()}_Ckpt{ckpt_b}"}
    }
    replay["params"] = flax.serialization.to_state_dict(env_params)

    # Write HTML
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="UTF-8" />
    <link rel="icon" type="image/svg+xml" href="https://s3vis.lux-ai.org/eye.svg" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />

    <title>Lux Eye S3</title>

    <script>
window.episode = {json.dumps(replay)};
    </script>

    <script type="module" crossorigin src="https://s3vis.lux-ai.org/index.js"></script>
  </head>
  <body>
    <div id="root"></div>
  </body>
</html>
    """.strip()
    
    with open(output_path, "w") as f:
        f.write(html_content)
        
    print("Done!")

if __name__ == "__main__":
    generate_replay(
        algo_a="mappo",
        ckpt_a=32,
        algo_b="masac",
        ckpt_b=23,
        seed=12156,
        output_path="performance_analysis/mappo_vs_masac_replay.html"
    )
