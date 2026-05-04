import os
import sys
import numpy as np
import jax
import torch
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pygame
import datetime
import glob

# Ensure paths
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../BenchMARL')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import hydra
from omegaconf import OmegaConf
from hydra.core.global_hydra import GlobalHydra
from benchmarl.hydra_config import load_experiment_from_hydra
from benchmarl.environments.lux.lux_env import LuxTorchRLEnv
from benchmarl.environments.lux.reward_exploration import compute_shaped_rewards_v2
from torchrl.envs.utils import set_exploration_type, ExplorationType

SEED_NUMBER = 1994

def get_base64_image(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', transparent=True, dpi=90)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def find_best_model():
    csv_path = "/home/carlos/Documents/github/msc_ai_thesis_experiments/performance_analysis/csv_data/rulebased_eval_data.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Returning fallback mappo 30.")
        return "mappo", 30
        
    df = pd.read_csv(csv_path)
    agg = df.groupby(["team_0_model", "chkpt_idx"])["team_0_total_points"].mean().reset_index()
    best = agg.loc[agg["team_0_total_points"].idxmax()]
    return best["team_0_model"], int(best["chkpt_idx"])

def get_checkpoint_path(algo, checkpoint_idx):
    base_dir = "/home/carlos/Documents/github/msc_ai_thesis_experiments/performance_analysis/models"
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

def calculate_reward_surface():
    size = 24
    relic_centers = np.array([[3, 3], [21, 14], [5, 20]])
    relics_list = [
        [3, 3], [4, 3], [3, 4], [2, 3], [3, 2],
        [20, 14], [21, 14], [21, 13], [22, 13],
        [6, 20], [6, 21], [5, 20]
    ]
    relics = np.array(relics_list)
    spawn = [12, 12]
    
    B = size * size
    current_team_mask = np.zeros((B, 16), dtype=bool)
    current_team_mask[:, 0] = True
    actions = np.zeros((B, 16), dtype=np.int32)
    delta_points = np.zeros(B, dtype=np.float32)
    delta_visible = np.zeros(B, dtype=np.int32)
    delta_energy = np.zeros((B, 16), dtype=np.float32)
    delta_energy[:, 0] = 5.0
    spawn_pos = np.full((B, 2), spawn, dtype=np.int32)
    known_relic_mask = np.ones((B, len(relics)), dtype=bool)
    known_relic_pos = np.zeros((B, len(relics), 2), dtype=np.int32)
    for b in range(B): known_relic_pos[b] = relics
    step_count = np.ones(B, dtype=np.int32) * 50

    current_team_pos = np.zeros((B, 16, 2), dtype=np.int32)
    x_coords, y_coords = np.meshgrid(range(size), range(size), indexing='ij')
    current_team_pos[:, 0, 0] = x_coords.flatten()
    current_team_pos[:, 0, 1] = y_coords.flatten()

    shaped_global, shaped_local, components = compute_shaped_rewards_v2(
        current_team_mask, current_team_pos, actions, delta_points, delta_visible,
        delta_energy, spawn_pos, known_relic_mask, known_relic_pos, step_count
    )
    
    total_shaped_rewards = shaped_global + shaped_local[:, 0]
    surface = total_shaped_rewards.reshape((size, size))
    
    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.cm.get_cmap('plasma')
    im = ax.imshow(surface.T, cmap=cmap, origin='lower')
    ax.scatter(relic_centers[:, 0], relic_centers[:, 1], c='gold', marker='*', s=300, edgecolor='black', label="Relic Center")
    ax.scatter(spawn[0], spawn[1], c='cyan', marker='X', s=150, edgecolor='black', label="Spawn Point")
    ax.contour(surface.T, levels=15, colors='white', alpha=0.3)
    ax.set_title("Dense Exploration Reward Landscape (Vector Pathing)", color='white', fontsize=14)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    plt.colorbar(im, label="Theoretical Step Reward")
    ax.legend(loc="upper right")
    fig.patch.set_facecolor('#121212')
    ax.set_facecolor('#1e1e1e')
    for text in ax.texts: text.set_color('white')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    
    b64_img = get_base64_image(fig)
    plt.close(fig)
    return b64_img

def main():
    print("Determining Best Model...")
    algo, chkpt = find_best_model()
    print(f"Best Model: {algo.upper()} (Checkpoint {chkpt})")
    
    p_path = get_checkpoint_path(algo, chkpt)
    if p_path is None:
        print("Model checkpoint not found on disk!")
        return
        
    print("Loading RL Policy...")
    policy, _ = load_policy(algo, p_path)
    
    print("Initializing environment...")
    out_dir = "/home/carlos/Documents/github/msc_ai_thesis_experiments/performance_analysis/architecture_reports"
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "best_model_architecture_report.html")
    photo_save_path = os.path.join(out_dir, "best_model_env_photo.png")
    
    np.random.seed(SEED_NUMBER)
    torch.manual_seed(SEED_NUMBER)
    
    env = LuxTorchRLEnv(batch_size=1, max_steps=200, match_count=1, seed=SEED_NUMBER, reward_version="v2")
    
    moth_dir = "/home/carlos/Documents/github/msc_ai_thesis_marl_lux"
    if moth_dir not in sys.path:
        sys.path.append(os.path.abspath(moth_dir))
    
    try:
        from agent import Agent
        env.rulebased_agent_class = Agent
    except ImportError:
        pass
        
    # Force RL to be Team 0
    original_randint = torch.randint
    def patched_randint(low, high, size, **kwargs):
        if low == 0 and high == 2:
            return torch.full(size, 0, dtype=torch.int64, **kwargs)
        return original_randint(low, high, size, **kwargs)
        
    torch.randint = patched_randint
    td = env.reset()
    torch.randint = original_randint
    
    print(f"Simulating at least 150 steps using {algo.upper()} until a relic, nebula, and enemy are observed simultaneously...")
    max_steps = 750
    final_step = 150
    unit_idx = 0
    
    for step in range(max_steps):
        with torch.no_grad():
            with set_exploration_type(ExplorationType.DETERMINISTIC):
                td = policy(td)
        td = env.step(td).get("next")
        
        if step >= 150:
            obs = td["agents", "observation"][0].cpu().numpy()
            energy_levels = obs[:, 2, :, :].sum(axis=(1,2))
            active_agents = np.where(energy_levels > 0)[0]
            
            relic_found = False
            for u in active_agents:
                # Require strictly that Channel 7 (Visible Relics), Channel 4 (Nebula), and Channel 1 (Enemies) are populated
                c7_relics = obs[u, 7, :, :].sum() > 0.1
                c4_nebula = obs[u, 4, :, :].sum() > 0.1
                c1_enemies = obs[u, 1, :, :].sum() > 0.1
                
                if c7_relics and c4_nebula and c1_enemies:
                    relic_found = True
                    unit_idx = int(u)
                    break
                    
            if relic_found:
                print(f"Relic, Nebula, and Enemy observed simultaneously at step {step} by agent {unit_idx}!")
                final_step = step
                break
            elif step == max_steps - 1:
                unit_idx = int(np.argmax(energy_levels))
                final_step = step

    print("Capturing data...")
    unbatched_state = jax.tree_util.tree_map(lambda x: np.asarray(x[0]), env.env_state)
    env.raw_env.render(unbatched_state, env.env_params)
    surface = pygame.display.get_surface()
    pygame.image.save(surface, photo_save_path)
    photo_path = "best_model_env_photo.png"

    components_html = ""
    reward_tensor = td.get(("agents", "reward"), torch.tensor(0.0))
    total_reward = torch.sum(reward_tensor[0]).item() if reward_tensor.ndim > 1 else torch.sum(reward_tensor).item()
    
    if hasattr(env, "last_reward_components"):
        components_html += "<ul class='list-group mb-4'>"
        components_html += f"<li class='list-group-item bg-dark text-light border-primary d-flex justify-content-between align-items-center'><strong>TOTAL COMBINED REWARD</strong> <span class='badge bg-success rounded-pill'>{total_reward:.4f}</span></li>"
        for k, v in env.last_reward_components.items():
            val = float(np.sum(np.asarray(v)[0]))
            components_html += f"<li class='list-group-item bg-dark text-light d-flex justify-content-between align-items-center'>{k} <span class='badge bg-primary rounded-pill'>{val:.4f}</span></li>"
        components_html += "</ul>"

    obs = td["agents", "observation"][0].cpu().numpy()
    channels_obs = obs[unit_idx]
    
    cmaps = ['viridis', 'viridis', 'magma', 'magma', 'viridis', 'viridis', 'viridis', 'plasma', 'spring', 'cool', 'inferno', 'bwr', 'RdPu', 'Blues', 'cividis', 'Greens']
    channel_images = []
    
    for i in range(16):
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        if i in [2, 3, 10, 11, 12, 13, 14, 15]:
            if i == 11:
                im = ax.imshow(channels_obs[i].T, cmap=cmaps[i], vmin=-1.0, vmax=1.0)
            else:
                im = ax.imshow(channels_obs[i].T, cmap=cmaps[i])
        else:
            im = ax.imshow(channels_obs[i].T, cmap=cmaps[i], vmin=0.0, vmax=1.0)
        ax.axis('off')
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8, colors='white')
        
        channel_images.append(get_base64_image(fig))
        plt.close(fig)

    print("Generating Theoretical Reward Landscape...")
    landscape_b64 = calculate_reward_surface()

    html_content = f"""<!DOCTYPE html>
<html lang="en" data-bs-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{algo.upper()} Architecture & Dynamic Input Report</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ background-color: #121212; }}
        .channel-card {{ border-left: 5px solid #0d6efd; background-color: #1e1e1e; margin-bottom: 20px; transition: transform 0.2s; }}
        .channel-card:hover {{ transform: translateX(10px); background-color: #2a2a2a; }}
        .relic-card {{ border-left-color: #ffc107 !important; }}
        .self-card {{ border-left-color: #198754 !important; }}
        .memory-card {{ border-left-color: #d63384 !important; }}
        .trajectory-card {{ border-left-color: #0dcaf0 !important; }}
        .img-fluid-channel {{ max-width: 100%; border-radius: 8px; background-color: #000; cursor: pointer; transition: transform 0.2s; }}
        .img-fluid-channel:hover {{ filter: brightness(1.2); transform: scale(1.05); }}
        .section-header {{ border-bottom: 1px solid #444; padding-bottom: 10px; margin-bottom: 20px; margin-top: 40px; }}
        .environment-photo {{ border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.5); max-width: 100%; cursor: pointer; }}
    </style>
</head>
<body class="text-light p-4">
    <div class="container">
        <h1 class="display-4 text-primary">Lux AI Season 3</h1>
        <h2 class="text-muted mb-4">Architecture Report: Best Model ({algo.upper()} Checkpoint {chkpt}) vs Rule-Based</h2>
        <p class="lead">This report is dynamically generated via script at simulation step {final_step}.</p>

        <h3 class="section-header">1. Environment Render</h3>
        <p>This is the actual global state of the game from which the channels are locally extracted.</p>
        <div class="text-center mb-5">
            <img src="{photo_path}" class="environment-photo" alt="Environment Screen">
        </div>

        <h3 class="section-header">2. Reward Components at Step {final_step}</h3>
        {components_html}

        <h3 class="section-header">3. Observation Channel Breakdown (Agent {unit_idx})</h3>
        <p class="text-muted">Below is the exact specification and heatmap snapshot drawn dynamically from the Python TensorDict.</p>

        <div class="row">
            <div class="col-md-4">
                <div class="card channel-card p-3"><div class="d-flex justify-content-between"><div><h5>Channel 0: My Units Map</h5></div><img src="data:image/png;base64,{channel_images[0]}" class="img-fluid-channel" style="width: 150px;"></div></div>
                <div class="card channel-card p-3"><div class="d-flex justify-content-between"><div><h5>Channel 1: Enemy Units</h5></div><img src="data:image/png;base64,{channel_images[1]}" class="img-fluid-channel" style="width: 150px;"></div></div>
                <div class="card channel-card p-3"><div class="d-flex justify-content-between"><div><h5>Channel 2: My Energy</h5></div><img src="data:image/png;base64,{channel_images[2]}" class="img-fluid-channel" style="width: 150px;"></div></div>
                <div class="card channel-card p-3"><div class="d-flex justify-content-between"><div><h5>Channel 3: Map Energy</h5></div><img src="data:image/png;base64,{channel_images[3]}" class="img-fluid-channel" style="width: 150px;"></div></div>
                <div class="card channel-card p-3"><div class="d-flex justify-content-between"><div><h5>Channel 4: Nebula Tiles</h5></div><img src="data:image/png;base64,{channel_images[4]}" class="img-fluid-channel" style="width: 150px;"></div></div>
            </div>
            
            <div class="col-md-4">
                <div class="card channel-card p-3"><div class="d-flex justify-content-between"><div><h5>Channel 5: Asteroids 🌑</h5></div><img src="data:image/png;base64,{channel_images[5]}" class="img-fluid-channel" style="width: 150px;"></div></div>
                <div class="card channel-card p-3"><div class="d-flex justify-content-between"><div><h5>Channel 6: Sensor FoW</h5></div><img src="data:image/png;base64,{channel_images[6]}" class="img-fluid-channel" style="width: 150px;"></div></div>
                <div class="card channel-card relic-card p-3"><div class="d-flex justify-content-between"><div><h5>Channel 7: Relic Nodes ☀️</h5></div><img src="data:image/png;base64,{channel_images[7]}" class="img-fluid-channel" style="width: 150px;"></div></div>
                <div class="card channel-card relic-card p-3"><div class="d-flex justify-content-between"><div><h5>Channel 14: Relic Memory 🌟</h5><p class="mb-2 text-muted"><strong>Values:</strong> Time Decay Float [0.0, 1.0]</p></div><img src="data:image/png;base64,{channel_images[14]}" class="img-fluid-channel" style="width: 150px;"></div></div>
                <div class="card channel-card self-card p-3"><div class="d-flex justify-content-between"><div><h5>Channel 8: Self Indicator 🎯</h5></div><img src="data:image/png;base64,{channel_images[8]}" class="img-fluid-channel" style="width: 150px;"></div></div>
                <div class="card channel-card p-3"><div class="d-flex justify-content-between"><div><h5>Channel 10: Timeline ⏳</h5></div><img src="data:image/png;base64,{channel_images[10]}" class="img-fluid-channel" style="width: 150px;"></div></div>
            </div>
            
            <div class="col-md-4">
                <div class="card channel-card p-3"><div class="d-flex justify-content-between"><div><h5>Channel 9: Ghost Coord 👻</h5></div><img src="data:image/png;base64,{channel_images[9]}" class="img-fluid-channel" style="width: 150px;"></div></div>
                <div class="card channel-card p-3"><div class="d-flex justify-content-between"><div><h5>Channel 11: Score Diff 👑</h5></div><img src="data:image/png;base64,{channel_images[11]}" class="img-fluid-channel" style="width: 150px;"></div></div>
                <div class="card channel-card p-3"><div class="d-flex justify-content-between"><div><h5>Channel 15: Points Delta 💰</h5><p class="mb-2 text-muted"><strong>Values:</strong> Uniform Reward Harvest Float [0.0, 1.0]</p></div><img src="data:image/png;base64,{channel_images[15]}" class="img-fluid-channel" style="width: 150px;"></div></div>
                <div class="card channel-card memory-card p-3"><div class="d-flex justify-content-between"><div><h5>Channel 12: Memory Stigmergy 🧠</h5><p class="mb-2 text-muted"><strong>Values:</strong> Time Decay Float [0.0, 1.0]</p></div><img src="data:image/png;base64,{channel_images[12]}" class="img-fluid-channel" style="width: 150px;"></div></div>
                <div class="card channel-card trajectory-card p-3"><div class="d-flex justify-content-between"><div><h5>Channel 13: Agent Trajectory 👣</h5><p class="mb-2 text-muted"><strong>Values:</strong> Egocentric Time Decay [0.0, 1.0]</p></div><img src="data:image/png;base64,{channel_images[13]}" class="img-fluid-channel" style="width: 150px;"></div></div>
            </div>
        </div>

        <h3 class="section-header text-info mt-5">4. Theoretical Reward Landscape</h3>
        <p class="text-muted">The Dense Exploration Reward uses `Manhattan Distance` proximity to known relics, balanced by a centrifugal <b>swarm dispersion pressure</b> requiring agents to spread out to maximize exploration. Furthermore, a <b>Greedy VIP Assignment algorithm</b> restricts mining rewards to the closest 4 agents per relic. Any excess agents approaching a fully manned relic within 8 tiles receive an explicit mathematically-scaling <code>overcrowding_penalty</code>.</p>
        <div class="alert alert-warning">
            <strong>Context Simulation:</strong> This matrix assumes the agent chose Action <code>0</code> (Center/Stagnation) at every coordinate tile. Areas outside of Relic Tails strictly apply the <code>Stagnation Penalty (-0.05)</code>, plunging their reward curve drastically downward. However, within distance of invisible <code>Relic Node Tails</code> surrounding the main stars, stagnation is forgiven and transforms into a positive <code>Farming Bonus (+0.01)</code>, creating elevated reward plateaus explicitly contouring the Relic tail structures. Since only 1 agent is simulated in this matrix, the Greedy VIP Assignment grants it full access to any relic it approaches without ever triggering the overcrowding penalty.
        </div>
        <div class="text-center mb-5 mt-4">
            <img src="data:image/png;base64,{landscape_b64}" class="img-fluid rounded" style="box-shadow: 0 4px 20px rgba(0,0,0,0.5); max-width: 800px;">
        </div>

    </div>
    
    <!-- Image Modal -->
    <div class="modal fade" id="imageModal" tabindex="-1" aria-labelledby="imageModalLabel" aria-hidden="true">
      <div class="modal-dialog modal-dialog-centered modal-lg">
        <div class="modal-content bg-transparent border-0">
          <div class="modal-header border-0">
            <button type="button" class="btn-close btn-close-white" data-bs-dismiss="modal" aria-label="Close"></button>
          </div>
          <div class="modal-body text-center">
            <img id="modalImg" src="" class="img-fluid" style="border-radius: 12px; box-shadow: 0 10px 30px rgba(0,0,0,0.8);">
          </div>
        </div>
      </div>
    </div>

    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/js/bootstrap.bundle.min.js"></script>
    <script>
        document.addEventListener("DOMContentLoaded", function() {{
            const modalImg = document.getElementById("modalImg");
            const imageModal = new bootstrap.Modal(document.getElementById("imageModal"));
            document.querySelectorAll(".img-fluid-channel, .environment-photo").forEach(img => {{
                img.addEventListener("click", function() {{
                    modalImg.src = this.src;
                    imageModal.show();
                }});
            }});
        }});
    </script>
</body>
</html>"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"✅ Successfully generated unified architecture report at {report_path}")

if __name__ == "__main__":
    main()
