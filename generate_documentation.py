import os
import sys
import numpy as np
import jax
import torch
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pygame

# Make sure we can import local modules
sys.path.append(os.path.abspath("/home/carlos/Documents/github/msc_ai_thesis_experiments/BenchMARL"))
from benchmarl.environments.lux.lux_env import LuxTorchRLEnv

def get_base64_image(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', transparent=True, dpi=90)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def generate_report():
    print("Initializing environment...")
    os.makedirs("/home/carlos/Documents/github/msc_ai_thesis_experiments/html_reports", exist_ok=True)
    
    # Enforce pure determinism for documentation consistency
    np.random.seed(42)
    torch.manual_seed(42)
    
    # We use a static seed to ensure reproducible interesting behaviors (units moving, etc)
    env = LuxTorchRLEnv(batch_size=1, max_steps=200, match_count=1, seed=1994)
    td = env.reset()
    
    team_id = int(env.team_ids[0].item())
    
    # Fast forward 150 steps
    print("Simulating 150 steps to populate environment...")
    for step in range(150):
        actions = np.random.randint(0, 5, size=(16,))
        td["agents", "action"] = torch.tensor(actions, dtype=torch.long, device=env.device).unsqueeze(0)
        td["action"] = td["agents", "action"]
        td = env.step(td).get("next")

    print("Capturing data...")
    # 1. Render and capture Pygame Environment Photo
    unbatched_state = jax.tree_util.tree_map(lambda x: np.asarray(x[0]), env.env_state)
    env.raw_env.render(unbatched_state, env.env_params)
    surface = pygame.display.get_surface()
    pygame.image.save(surface, "/home/carlos/Documents/github/msc_ai_thesis_experiments/html_reports/env_photo.png")
    
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

    # 3. Extract 12 Channels Observation (select an active unit if possible, else 0)
    obs = td["agents", "observation"][0].cpu().numpy() # [U, 12, 24, 24]
    
    # Picking the unit that has the highest energy to guarantee it is active and on map
    energy_levels = obs[:, 2, :, :].sum(axis=(1,2))
    unit_idx = int(np.argmax(energy_levels))
    channels_obs = obs[unit_idx]
    
    # 3.5. Mathematical Validation for Channel 3 (Global Map Energy)
    c3 = channels_obs[3]
    print(f"\n--- VALIDATION FOR CHANNEL 3 (Global Map Energy) ---")
    print(f"Agent {unit_idx} - Shape: {c3.shape}")
    print(f"Min: {np.min(c3):.6f} | Max: {np.max(c3):.6f}")
    print(f"Mean: {np.mean(c3):.8f} | Variance: {np.var(c3):.8f}")
    
    grid_c3 = obs[:, 3, :, :]
    print(f"Global Batch Channel 3 Mean across all agents: {np.mean(grid_c3):.8f}\n")

    # 4. Generate Images for the 12 Channels
    cmaps = ['viridis', 'viridis', 'magma', 'magma', 'viridis', 'viridis', 'viridis', 'plasma', 'spring', 'cool', 'inferno', 'bwr']
    channel_images = []
    
    for i in range(12):
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        
        # Float numerical channels (Energy, Step Count, Score Diff) should auto-scale to display variance dynamically
        if i in [2, 3, 10, 11]:
            # Score Diff uses diverging colormap
            if i == 11:
                im = ax.imshow(channels_obs[i].T, cmap=cmaps[i], vmin=-1.0, vmax=1.0)
            else:
                im = ax.imshow(channels_obs[i].T, cmap=cmaps[i])
        else:
            # Binary masks stick to absolute 0-1 scale
            im = ax.imshow(channels_obs[i].T, cmap=cmaps[i], vmin=0.0, vmax=1.0)
            
        ax.axis('off')
        
        # Add colorbar tightly
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
    <title>Lux AI S3 - MARL Dynamic Input Report</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.2/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {{ background-color: #121212; }}
        .channel-card {{
            transition: transform 0.2s;
            border-left: 5px solid #0d6efd;
            background-color: #1e1e1e;
            margin-bottom: 20px;
        }}
        .channel-card:hover {{
            transform: translateX(10px);
            background-color: #2a2a2a;
        }}
        .relic-card {{ border-left-color: #ffc107 !important; }}
        .self-card {{ border-left-color: #198754 !important; }}
        .img-fluid-channel {{ max-width: 100%; border-radius: 8px; background-color: #000; }}
        .section-header {{ border-bottom: 1px solid #444; padding-bottom: 10px; margin-bottom: 20px; margin-top: 40px; }}
        .environment-photo {{ border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.5); max-width: 100%; }}
    </style>
</head>
<body class="text-light p-4">
    <div class="container">
        <h1 class="display-4 text-primary">Lux AI Season 3</h1>
        <h2 class="text-muted mb-4">Interactive MARL Observation Documentation</h2>
        <p class="lead">This report is dynamically generated via script at simulation step 150.</p>

        <h3 class="section-header">1. Environment Render</h3>
        <p>This is the actual global state of the game from which the channels are locally extracted.</p>
        <div class="text-center mb-5">
            <img src="{photo_path}" class="environment-photo" alt="Environment Screen">
        </div>

        <h3 class="section-header">2. Reward Components at Step 150</h3>
        {components_html}

        <h3 class="section-header">3. Observation Channel Breakdown (Agent {unit_idx})</h3>
        <p class="text-muted">Below is the exact specification and heatmap snapshot drawn dynamically from the Python TensorDict.</p>

        <div class="row">
            <div class="col-md-4">
                <!-- C0 -->
                <div class="card channel-card p-3">
                    <div class="d-flex justify-content-between">
                        <div>
                            <h5>Channel 0: My Units Map</h5>
                            <p class="mb-2 text-muted"><strong>Values:</strong> Binary [0.0, 1.0]</p>
                        </div>
                        <img src="data:image/png;base64,{channel_images[0]}" class="img-fluid-channel" style="width: 150px;">
                    </div>
                </div>

                <!-- C1 -->
                <div class="card channel-card p-3">
                    <div class="d-flex justify-content-between">
                        <div>
                            <h5>Channel 1: Enemy Units Map</h5>
                            <p class="mb-2 text-muted"><strong>Values:</strong> Binary [0.0, 1.0] (Fog of War)</p>
                        </div>
                        <img src="data:image/png;base64,{channel_images[1]}" class="img-fluid-channel" style="width: 150px;">
                    </div>
                </div>

                <!-- C2 -->
                <div class="card channel-card p-3">
                    <div class="d-flex justify-content-between">
                        <div>
                            <h5>Channel 2: My Units Energy</h5>
                            <p class="mb-2 text-muted"><strong>Values:</strong> Float normalized by 400.0</p>
                        </div>
                        <img src="data:image/png;base64,{channel_images[2]}" class="img-fluid-channel" style="width: 150px;">
                    </div>
                </div>

                <!-- C3 -->
                <div class="card channel-card p-3">
                    <div class="d-flex justify-content-between">
                        <div>
                            <h5>Channel 3: Global Map Energy</h5>
                            <p class="mb-2 text-muted"><strong>Values:</strong> Float Normalized by 400.0</p>
                        </div>
                        <img src="data:image/png;base64,{channel_images[3]}" class="img-fluid-channel" style="width: 150px;">
                    </div>
                </div>

                <!-- C4 -->
                <div class="card channel-card p-3">
                    <div class="d-flex justify-content-between">
                        <div>
                            <h5>Channel 4: Nebula Tiles</h5>
                            <p class="mb-2 text-muted"><strong>Values:</strong> Binary mask</p>
                        </div>
                        <img src="data:image/png;base64,{channel_images[4]}" class="img-fluid-channel" style="width: 150px;">
                    </div>
                </div>
            </div>
            
            <div class="col-md-4">
                <!-- C5 -->
                <div class="card channel-card p-3">
                    <div class="d-flex justify-content-between">
                        <div>
                            <h5>Channel 5: Asteroid Tiles</h5>
                            <p class="mb-2 text-muted"><strong>Values:</strong> Binary mask</p>
                        </div>
                        <img src="data:image/png;base64,{channel_images[5]}" class="img-fluid-channel" style="width: 150px;">
                    </div>
                </div>

                <!-- C6 -->
                <div class="card channel-card p-3">
                    <div class="d-flex justify-content-between">
                        <div>
                            <h5>Channel 6: Sensor Fog of War</h5>
                            <p class="mb-2 text-muted"><strong>Values:</strong> Binary mask (1=Visible)</p>
                        </div>
                        <img src="data:image/png;base64,{channel_images[6]}" class="img-fluid-channel" style="width: 150px;">
                    </div>
                </div>

                <!-- C7 -->
                <div class="card channel-card relic-card p-3">
                    <div class="d-flex justify-content-between">
                        <div>
                            <h5>Channel 7: Relic Nodes 🌟</h5>
                            <p class="mb-2 text-muted"><strong>Values:</strong> Binary [0.0, 1.0]</p>
                        </div>
                        <img src="data:image/png;base64,{channel_images[7]}" class="img-fluid-channel" style="width: 150px;">
                    </div>
                </div>

                <!-- C8 -->
                <div class="card channel-card self-card p-3">
                    <div class="d-flex justify-content-between">
                        <div>
                            <h5>Channel 8: Self Indicator 🎯</h5>
                            <p class="mb-2 text-muted"><strong>Values:</strong> Singular `1.0` binary.</p>
                        </div>
                        <img src="data:image/png;base64,{channel_images[8]}" class="img-fluid-channel" style="width: 150px;">
                    </div>
                </div>

                <!-- C10 -->
                <div class="card channel-card p-3">
                    <div class="d-flex justify-content-between">
                        <div>
                            <h5>Channel 10: Timeline Percent ⏳</h5>
                            <p class="mb-2 text-muted"><strong>Values:</strong> Float percentage</p>
                        </div>
                        <img src="data:image/png;base64,{channel_images[10]}" class="img-fluid-channel" style="width: 150px;">
                    </div>
                </div>
            </div>
            
            <div class="col-md-4">
               <!-- EXTRA CHANNELS COLUMN -->
                <!-- C9 -->
                <div class="card channel-card p-3">
                    <div class="d-flex justify-content-between">
                        <div>
                            <h5>Channel 9: Ghost Coordinate 👻</h5>
                            <p class="mb-2 text-muted"><strong>Values:</strong> Last Action Tracking</p>
                        </div>
                        <img src="data:image/png;base64,{channel_images[9]}" class="img-fluid-channel" style="width: 150px;">
                    </div>
                </div>
                
                <!-- C11 -->
                <div class="card channel-card p-3">
                    <div class="d-flex justify-content-between">
                        <div>
                            <h5>Channel 11: Score Differential 👑</h5>
                            <p class="mb-2 text-muted"><strong>Values:</strong> Float [-1.0, 1.0]</p>
                        </div>
                        <img src="data:image/png;base64,{channel_images[11]}" class="img-fluid-channel" style="width: 150px;">
                    </div>
                </div>
            </div>
        </div>

    </div>
</body>
</html>
    """

    report_path = "/home/carlos/Documents/github/msc_ai_thesis_experiments/html_reports/lux_input_report.html"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
        
    print(f"✅ Successfully updated report at {report_path}")

if __name__ == "__main__":
    generate_report()
