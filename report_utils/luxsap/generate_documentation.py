import os
import sys
import numpy as np
import jax
import torch
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pygame
import subprocess
import datetime

# Make sure we can import local modules
benchmarl_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../BenchMARL"))
sys.path.append(benchmarl_dir)
from benchmarl.environments.lux.lux_with_sap import LuxSapTorchRLEnv

SEED_NUMBER = 1994
def get_base64_image(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', transparent=True, dpi=90)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def generate_report():
    print("Initializing environment...")
    
    out_dir = os.environ.get("REPORT_OUT_DIR")
    if not out_dir:
        stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        out_dir = f"/home/carlos/Documents/github/msc_ai_thesis_experiments/html_reports/luxsap_{stamp}"
        
    os.makedirs(out_dir, exist_ok=True)
    report_path = os.path.join(out_dir, "luxsap_input_report.html")
    photo_save_path = os.path.join(out_dir, "env_photo.png")
    
    # Enforce pure determinism for documentation consistency
    np.random.seed(SEED_NUMBER)
    torch.manual_seed(SEED_NUMBER)
    
    # We use a static seed to ensure reproducible interesting behaviors (units moving, etc)
    env = LuxSapTorchRLEnv(batch_size=1, max_steps=200, match_count=1, seed=SEED_NUMBER, reward_version="v2", device="cpu")
    
    moth_dir = "/home/carlos/Documents/github/msc_ai_thesis_marl_lux"
    if moth_dir not in sys.path:
        sys.path.append(os.path.abspath(moth_dir))
    
    agent_class = None
    try:
        from rulebased_agent_main import Agent
        agent_class = Agent
        env.rulebased_agent_class = Agent
    except ImportError as e:
        print(f"FAILED TO IMPORT AGENT: {e}")
        pass
        
    td = env.reset()
    
    team_id = int(env.team_ids[0].item())
    env_cfg = {"max_units": 16, "map_width": 24, "map_height": 24}
    agent_p0 = agent_class(f"player_{team_id}", env_cfg) if agent_class else None
    
    # Fast forward 150 steps
    print("Simulating 150 steps to populate environment...")
    for step in range(150):
        if agent_p0:
            # Build pseudo obs logic for the rulebased agent
            p0 = env.jax_obs[f"player_{team_id}"]
            def get_v(o, k): return getattr(o, k) if hasattr(o, k) else o.get(k)
            u_mask = np.asarray(get_v(p0, "units_mask"))[0]
            pos = np.asarray(get_v(get_v(p0, "units"), "position"))[0]
            en = np.asarray(get_v(get_v(p0, "units"), "energy"))[0]
            r_nodes = np.asarray(get_v(p0, "relic_nodes"))[0]
            r_mask = np.asarray(get_v(p0, "relic_nodes_mask"))[0]
            t_pts = np.asarray(get_v(p0, "team_points"))[0]
            
            pseudo_obs = {
                "units_mask": u_mask.tolist(),
                "units": {"position": pos.tolist(), "energy": en.tolist()},
                "relic_nodes": r_nodes.tolist(), "relic_nodes_mask": r_mask.tolist(), "team_points": t_pts.tolist()
            }
            
            act_rule = agent_p0.act(step, pseudo_obs)
            actions = act_rule[:, 0]
        else:
            actions = np.random.randint(0, 5, size=(16,))
            
        # Randomly shoot SAP (action 5) to show combat behavior in documentation
        actions[np.random.rand(16) < 0.15] = 5
            
        td["agents", "action"] = torch.tensor(actions, dtype=torch.long, device=env.device).unsqueeze(0)
        td["action"] = td["agents", "action"]
        
        env.opp_actions = np.zeros((1, 16), dtype=np.int32)
        env.opp_actions[0, np.random.rand(16) < 0.15] = 5
        
        td = env.step(td).get("next")

    print("Capturing data...")
    # 1. Render and capture Pygame Environment Photo
    unbatched_state = jax.tree_util.tree_map(lambda x: np.asarray(x[0]), env.env_state)
    env.raw_env.render(unbatched_state, env.env_params)
    surface = pygame.display.get_surface()
    pygame.image.save(surface, photo_save_path)
    
    # Encode main photo as base64 or link locally
    photo_path = "env_photo.png"

    # 2. Extract Reward Components
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
    else:
        components_html = "<p class='text-muted'>No dynamic reward components available in this step.</p>"

    # 3. Extract 16 Channels Observation (select an active unit if possible, else 0)
    obs = td["agents", "observation"][0].cpu().numpy() # [U, 16, 24, 24]
    
    # Picking the unit that has the highest energy to guarantee it is active and on map
    energy_levels = obs[:, 2, :, :].sum(axis=(1,2))
    unit_idx = int(np.argmax(energy_levels))
    channels_obs = obs[unit_idx]
    
    # 4. Generate Images for the 16 Channels
    cmaps = ['viridis', 'viridis', 'magma', 'magma', 'viridis', 'viridis', 'viridis', 'plasma', 'spring', 'cool', 'inferno', 'bwr', 'RdPu', 'Blues', 'cividis', 'Greens']
    channel_images = []
    
    for i in range(16):
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        
        # Float numerical channels
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
        
        b64 = get_base64_image(fig)
        channel_images.append(b64)
        plt.close(fig)

    # 5. Build HTML
    html_content = f"""<!DOCTYPE html>
<html lang="en" data-bs-theme="dark">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Lux AI S3 - Combat Environment Documentation</title>
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
        <h1 class="display-4 text-primary">Lux AI Season 3 (Combat SAP Module)</h1>
        <h2 class="text-muted mb-4">Interactive MARL Observation Documentation</h2>
        <p class="lead">This report is dynamically generated via script at simulation step 150.</p>

        <h3 class="section-header">1. Environment Render</h3>
        <p>This is the actual global state of the game from which the channels are locally extracted.</p>
        <div class="text-center mb-5">
            <img src="{photo_path}" class="environment-photo" alt="Environment Screen">
        </div>

        <h3 class="section-header">2. Reward Components at Step 150 (Including Combat Logic)</h3>
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
        
    print(f"✅ Successfully updated report at {report_path}")

if __name__ == "__main__":
    generate_report()
