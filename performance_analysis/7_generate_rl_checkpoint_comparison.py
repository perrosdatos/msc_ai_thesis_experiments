import os
import sys
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
from kpi_tracker import LuxKPITracker

SEED_NUMBER = 659597

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
                "experiment.checkpoint_interval=150000", "experiment.loggers=[]", f"seed={SEED_NUMBER}",
                "experiment.evaluation_episodes=1"
            ],
        )
        experiment = load_experiment_from_hydra(cfg, task_name="lux/match_v2")
        experiment.load_state_dict(torch.load(checkpoint_path, map_location="cpu", weights_only=False))
        return experiment.algorithm.get_policy_for_collection(), experiment.test_env

def unbatch_jax_tree(tree, index=0):
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

def generate_replay(algo_a, ckpt_a, algo_b, ckpt_b, milestone_idx, seed, output_path):
    print(f"Loading {algo_a.upper()} (Ckpt {ckpt_a})...")
    path_a = get_checkpoint_path(algo_a, ckpt_a)
    if not path_a:
        raise FileNotFoundError(f"Checkpoint for {algo_a} at index {ckpt_a} not found.")
    p0_policy, env = load_policy(algo_a, path_a)
    
    print(f"Loading {algo_b.upper()} (Ckpt {ckpt_b})...")
    path_b = get_checkpoint_path(algo_b, ckpt_b)
    if not path_b:
        raise FileNotFoundError(f"Checkpoint for {algo_b} at index {ckpt_b} not found.")
    p1_policy, _ = load_policy(algo_b, path_b)
    
    print(f"Running Match (Seed {seed})...")
    
    base_lux = env.base_env

    # Fresh reset
    base_lux._set_seed(SEED_NUMBER)
    if hasattr(base_lux, "env_state"):
        del base_lux.env_state

    # Setup seed
    import jax.numpy as jnp
    original_split = jax.random.split
    def patched_split(key, num=2):
        if num == 1:
            return jnp.stack([jax.random.PRNGKey(seed)])
        return original_split(key, num)
    jax.random.split = patched_split

    original_randint = torch.randint
    def patched_randint(low, high, size, **kwargs):
        if low == 0 and high == 2:
            return torch.full(size, 0, dtype=torch.int64, **kwargs)
        return original_randint(low, high, size, **kwargs)
    torch.randint = patched_randint

    td = env.reset()
    base_lux.rulebased_agent_class = None

    jax.random.split = original_split
    torch.randint = original_randint
    
    t_ids_np = base_lux.team_ids.cpu().numpy()
    opp_team_ids = 1 - t_ids_np
    
    tracker = LuxKPITracker(batch_size=1)
    
    episode_states = []
    episode_actions = []
    
    # Get initial state
    init_state = unbatch_jax_tree(base_lux.env_state, 0)
    episode_states.append(init_state)
    
    # Store initial params
    env_params = unbatch_jax_tree(base_lux.env_params, 0)
    
    pbar = tqdm(total=750, desc=f"Evaluating Matchup")
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
        pos = np.asarray(base_lux._get_v(u_p0, "position"), np.int32)
        
        obs_b_array = base_lux._build_spatial_observation(base_lux.jax_obs, opp_team_ids, pos)
        
        m0 = np.asarray(base_lux._get_v(base_lux.jax_obs["player_0"], "units_mask"), bool)
        m1 = np.asarray(base_lux._get_v(base_lux.jax_obs["player_1"], "units_mask"), bool)
        m_opp = np.where((opp_team_ids == 0)[:, None, None], m0, m1)
        
        active_units_B = np.zeros((1, base_lux.max_units), dtype=bool)
        active_units_B[0, :] = m_opp[0, opp_team_ids[0], :]
            
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
        
        # Update KPI tracker
        rc = getattr(env, "last_reward_components", {})
        tracker.update(env, td, rc)
        
        # Store state
        episode_states.append(unbatch_jax_tree(base_lux.env_state, 0))
        
        td = td.get("next")
        pbar.update(1)
        step_idx += 1
        if td["done"].any() or step_idx >= 750:
            break
            
    print(f"Match complete. Serializing HTML to {output_path}...")
    
    # Extract and format final KPI metrics
    raw_res = tracker.get_results()[0]
    p0_metrics = {
        "expl": raw_res.get(f"team_0_map_exploration_prop", 0.0),
        "relic": raw_res.get(f"team_0_time_to_first_relic", -1),
        "delay": raw_res.get(f"team_0_info_propagation_delay", 750),
        "synergy": raw_res.get(f"team_0_synergy_handoffs", 0.0)
    }
    p1_metrics = {
        "expl": raw_res.get(f"team_1_map_exploration_prop", 0.0),
        "relic": raw_res.get(f"team_1_time_to_first_relic", -1),
        "delay": raw_res.get(f"team_1_info_propagation_delay", 750),
        "synergy": raw_res.get(f"team_1_synergy_handoffs", 0.0)
    }
    
    def fmt_relic(val):
        return f"{val} turns" if val != -1 else "Never"
    def fmt_delay(val):
        return f"{val} turns" if val != 750 else "Never"
        
    p0_expl_str = f"{p0_metrics['expl'] * 100:.1f}%"
    p0_relic_str = fmt_relic(p0_metrics['relic'])
    p0_delay_str = fmt_delay(p0_metrics['delay'])
    p0_synergy_str = f"{p0_metrics['synergy']:.2f}"
    
    p1_expl_str = f"{p1_metrics['expl'] * 100:.1f}%"
    p1_relic_str = fmt_relic(p1_metrics['relic'])
    p1_delay_str = fmt_delay(p1_metrics['delay'])
    p1_synergy_str = f"{p1_metrics['synergy']:.2f}"

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

    <title>Lux Eye S3 Replay - {algo_a.upper()} vs {algo_b.upper()} - Milestone {milestone_idx}</title>
    
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700;800&family=JetBrains+Mono&display=swap" rel="stylesheet">

    <style>
      body {{
        margin: 0;
        padding: 0;
        overflow: hidden;
        background-color: #0c0c0e;
      }}
      #kpi-hud {{
        position: absolute;
        top: 20px;
        right: 20px;
        width: 300px;
        background: rgba(15, 15, 20, 0.7);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 20px;
        z-index: 999999;
        color: #e0e0e0;
        font-family: 'Outfit', -apple-system, sans-serif;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      }}
      #kpi-hud.collapsed {{
        width: 40px;
        height: 40px;
        padding: 0;
        overflow: hidden;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        cursor: pointer;
        background: rgba(15, 15, 20, 0.9);
      }}
      
      /* Floating Comparison Legend (Bottom Left - Compact) */
      #comparison-legend {{
        position: absolute;
        bottom: 20px;
        left: 20px;
        background: rgba(15, 15, 20, 0.85);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.15);
        border-radius: 8px;
        padding: 8px 12px;
        z-index: 999999;
        color: #ffffff;
        font-family: 'Outfit', -apple-system, sans-serif;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.6);
        font-size: 13px;
        line-height: 1.3;
      }}
      .legend-milestone {{
        font-weight: 800;
        color: #ffc107;
        font-size: 14px;
        margin-bottom: 2px;
      }}
      .legend-checkpoint {{
        font-family: 'JetBrains Mono', monospace;
        font-weight: 600;
        color: #e2e8f0;
      }}
      
      .hud-header {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 10px;
        margin-bottom: 15px;
      }}
      .hud-title {{
        margin: 0;
        font-size: 16px;
        font-weight: 700;
        letter-spacing: 0.5px;
        color: #f1f3f5;
        text-transform: uppercase;
      }}
      .toggle-btn {{
        background: none;
        border: none;
        color: #a6a7ab;
        cursor: pointer;
        font-size: 12px;
        font-family: inherit;
        padding: 4px 8px;
        border-radius: 6px;
        transition: background 0.2s;
      }}
      .toggle-btn:hover {{
        background: rgba(255, 255, 255, 0.08);
        color: #fff;
      }}
      .team-box {{
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.04);
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 12px;
      }}
      .team-name {{
        margin: 0 0 8px 0;
        font-size: 14px;
        font-weight: 600;
        display: flex;
        align-items: center;
      }}
      .team-dot {{
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
        display: inline-block;
      }}
      .t0-color {{ color: #3b82f6; }}
      .t0-bg {{ background-color: #3b82f6; box-shadow: 0 0 8px #3b82f6; }}
      .t1-color {{ color: #ef4444; }}
      .t1-bg {{ background-color: #ef4444; box-shadow: 0 0 8px #ef4444; }}
      .metric-row {{
        display: flex;
        justify-content: space-between;
        font-size: 12px;
        margin: 6px 0;
        line-height: 1.4;
      }}
      .metric-label {{
        color: #909296;
      }}
      .metric-value {{
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        color: #f8f9fa;
      }}
      .hud-collapsed-icon {{
        display: none;
        font-size: 20px;
        color: #fff;
      }}
      #kpi-hud.collapsed .hud-collapsed-icon {{
        display: block;
      }}
      #kpi-hud.collapsed .hud-main-content {{
        display: none;
      }}
    </style>

    <script>
      window.episode = {json.dumps(replay)};
      
      function toggleHUD(e) {{
        const hud = document.getElementById('kpi-hud');
        if (hud.classList.contains('collapsed')) {{
          hud.classList.remove('collapsed');
        }} else if (e.target.classList.contains('toggle-btn') || e.target.id === 'toggle-btn') {{
          hud.classList.add('collapsed');
        }}
      }}
    </script>

    <script type="module" crossorigin src="https://s3vis.lux-ai.org/index.js"></script>
  </head>
  <body>
    <div id="root"></div>
    
    <!-- Floating Comparison Legend (Bottom Left) -->
    <div id="comparison-legend">
      <div class="legend-milestone">Milestone {milestone_idx}/4</div>
      <div class="legend-checkpoint">Ckpt {ckpt_a} vs {ckpt_b}</div>
    </div>
    
    <div id="kpi-hud" onclick="toggleHUD(event)">
      <div class="hud-collapsed-icon">📊</div>
      <div class="hud-main-content" style="width: 100%;">
        <div class="hud-header">
          <h3 class="hud-title">KPIs (Seed {seed})</h3>
          <button class="toggle-btn" id="toggle-btn">Minimize</button>
        </div>
        
        <div class="team-box">
          <h4 class="team-name t0-color">
            <span class="team-dot t0-bg"></span>{algo_a.upper()} (Ckpt {ckpt_a})
          </h4>
          <div class="metric-row">
            <span class="metric-label">Map Exploration</span>
            <span class="metric-value">{p0_expl_str}</span>
          </div>
          <div class="metric-row">
            <span class="metric-label">Time to First Relic</span>
            <span class="metric-value">{p0_relic_str}</span>
          </div>
          <div class="metric-row">
            <span class="metric-label">Info Prop. Delay</span>
            <span class="metric-value">{p0_delay_str}</span>
          </div>
          <div class="metric-row">
            <span class="metric-label">Synergy Handoffs</span>
            <span class="metric-value">{p0_synergy_str}</span>
          </div>
        </div>

        <div class="team-box" style="margin-bottom: 0;">
          <h4 class="team-name t1-color">
            <span class="team-dot t1-bg"></span>{algo_b.upper()} (Ckpt {ckpt_b})
          </h4>
          <div class="metric-row">
            <span class="metric-label">Map Exploration</span>
            <span class="metric-value">{p1_expl_str}</span>
          </div>
          <div class="metric-row">
            <span class="metric-label">Time to First Relic</span>
            <span class="metric-value">{p1_relic_str}</span>
          </div>
          <div class="metric-row">
            <span class="metric-label">Info Prop. Delay</span>
            <span class="metric-value">{p1_delay_str}</span>
          </div>
          <div class="metric-row">
            <span class="metric-label">Synergy Handoffs</span>
            <span class="metric-value">{p1_synergy_str}</span>
          </div>
        </div>
      </div>
    </div>
  </body>
</html>
    """.strip()

    with open(output_path, "w") as f:
        f.write(html_content)
        
    print("Done!")

if __name__ == "__main__":
    # 4 evenly spaced milestones spanning checkpoint 1 to checkpoint 40
    # Step = 13: 1, 14, 27, 40
    matchups = [
        (1, 1, 1, "performance_analysis/mappo_vs_masac_comparison_1.html"),
        (14, 14, 2, "performance_analysis/mappo_vs_masac_comparison_2.html"),
        (27, 27, 3, "performance_analysis/mappo_vs_masac_comparison_3.html"),
        (40, 40, 4, "performance_analysis/mappo_vs_masac_comparison_4.html")
    ]
    for ckpt_a, ckpt_b, milestone, out_path in matchups:
        print("\n" + "="*50)
        print(f"GENERATING MILESTONE {milestone}/4: MAPPO (Ckpt {ckpt_a}) vs MASAC (Ckpt {ckpt_b})")
        print("="*50)
        generate_replay(
            algo_a="mappo",
            ckpt_a=ckpt_a,
            algo_b="masac",
            ckpt_b=ckpt_b,
            milestone_idx=milestone,
            seed=SEED_NUMBER,
            output_path=out_path
        )
