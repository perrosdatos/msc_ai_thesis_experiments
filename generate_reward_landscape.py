import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Add core path to import the algorithm
sys.path.append(os.path.abspath("/home/carlos/Documents/github/msc_ai_thesis_experiments/BenchMARL"))
from benchmarl.environments.lux.reward_exploration import compute_shaped_rewards_v2

def get_base64_image(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', transparent=True, dpi=120)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def calculate_reward_surface():
    """
    Simulates the exact logic from benchmarl/environments/lux/reward_exploration.py
    for a single agent moving across every coordinate of a 24x24 grid.
    """
    size = 24
    
    # Predefined Relics (Simulating clusters/tails in a 5x5 cross-like window)
    relic_centers = np.array([[3, 3], [21, 14], [5, 20]])
    relics_list = [
        # Top-Left Relic Cluster (cross shape) around [3,3]
        [3, 3], [4, 3], [3, 4], [2, 3], [3, 2],
        
        # Bottom-Right Relic Cluster (spread shape) around [21,14]
        [20, 14], [21, 14], [21, 13], [22, 13],
        
        # Top-Right Relic Cluster around [5,20]
        [6, 20], [6, 21], [5, 20]
    ]
    relics = np.array(relics_list)
    spawn = [12, 12] # Center spawn
    
    # Simulate Native Payload to Benchmark Code
    B = size * size
    current_team_mask = np.zeros((B, 16), dtype=bool)
    current_team_mask[:, 0] = True # Only Agent 0 is active
    
    actions = np.zeros((B, 16), dtype=np.int32) # All agents choose Action 0 (Center/Idle)
    delta_points = np.zeros(B, dtype=np.float32)
    delta_visible = np.zeros(B, dtype=np.int32)
    delta_energy = np.zeros((B, 16), dtype=np.float32)
    delta_energy[:, 0] = 5.0 # Positive energy delta to strictly simulate Farming conditions if on a node
    
    spawn_pos = np.full((B, 2), spawn, dtype=np.int32)
    known_relic_mask = np.ones((B, len(relics)), dtype=bool)
    known_relic_pos = np.zeros((B, len(relics), 2), dtype=np.int32)
    for b in range(B):
        known_relic_pos[b] = relics
        
    step_count = np.ones(B, dtype=np.int32) * 50

    # Build the 576-coordinate spatial trajectory tests
    current_team_pos = np.zeros((B, 16, 2), dtype=np.int32)
    x_coords, y_coords = np.meshgrid(range(size), range(size), indexing='ij')
    current_team_pos[:, 0, 0] = x_coords.flatten()
    current_team_pos[:, 0, 1] = y_coords.flatten()

    # Exclusively use NATIVE REWARD SIMULATION
    shaped_global, shaped_local, components = compute_shaped_rewards_v2(
        current_team_mask,
        current_team_pos,
        actions,
        delta_points,
        delta_visible,
        delta_energy,
        spawn_pos,
        known_relic_mask,
        known_relic_pos,
        step_count
    )
    
    # Combine global + local reward for Agent 0
    total_shaped_rewards = shaped_global + shaped_local[:, 0]
    
    # Reshape the output mathematically
    surface = total_shaped_rewards.reshape((size, size))
    
    # Visualize Heatmap
    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.cm.get_cmap('plasma')
    im = ax.imshow(surface.T, cmap=cmap, origin='lower')
    
    # Annotate Relics & Spawn (Only plot the centers to avoid visual clutter)
    ax.scatter(relic_centers[:, 0], relic_centers[:, 1], c='gold', marker='*', s=300, edgecolor='black', label="Relic Center")
    ax.scatter(spawn[0], spawn[1], c='cyan', marker='X', s=150, edgecolor='black', label="Spawn Point")
    
    # Contour lines to show gradient paths
    ax.contour(surface.T, levels=15, colors='white', alpha=0.3)
    
    ax.set_title("Dense Exploration Reward Landscape (Vector Pathing)", color='white', fontsize=14)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    plt.colorbar(im, label="Theoretical Step Reward")
    ax.legend(loc="upper right")
    
    # Style for Dark Mode
    fig.patch.set_facecolor('#121212')
    ax.set_facecolor('#1e1e1e')
    for text in ax.texts: text.set_color('white')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    
    b64_img = get_base64_image(fig)
    plt.close(fig)
    return b64_img

def append_to_html_report(b64_image):
    out_dir = os.environ.get("REPORT_OUT_DIR")
    if out_dir:
        report_path = os.path.join(out_dir, "lux_input_report.html")
    else:
        report_path = "/home/carlos/Documents/github/msc_ai_thesis_experiments/html_reports/lux_input_report.html"
    
    # Read current report
    with open(report_path, "r", encoding="utf-8") as f:
        html = f.read()
        
    # Construct new section
    new_section = f"""
        <h3 class="section-header text-info mt-5">4. Theoretical Reward Landscape</h3>
        <p class="text-muted">The Dense Exploration Reward uses `Manhattan Distance` proximity to known relics, balanced by a centrifugal <b>swarm dispersion pressure</b> requiring agents to spread out to maximize exploration. Furthermore, a <b>Greedy VIP Assignment algorithm</b> restricts mining rewards to the closest 4 agents per relic. Any excess agents approaching a fully manned relic within 8 tiles receive an explicit mathematically-scaling <code>overcrowding_penalty</code>.</p>
        <div class="alert alert-warning">
            <strong>Context Simulation:</strong> This matrix assumes the agent chose Action <code>0</code> (Center/Stagnation) at every coordinate tile. Areas outside of Relic Tails strictly apply the <code>Stagnation Penalty (-0.05)</code>, plunging their reward curve drastically downward. However, within distance of invisible <code>Relic Node Tails</code> surrounding the main stars, stagnation is forgiven and transforms into a positive <code>Farming Bonus (+0.01)</code>, creating elevated reward plateaus explicitly contouring the Relic tail structures. Since only 1 agent is simulated in this matrix, the Greedy VIP Assignment grants it full access to any relic it approaches without ever triggering the overcrowding penalty.
        </div>
        <div class="text-center mb-5 mt-4">
            <img src="data:image/png;base64,{b64_image}" class="img-fluid rounded" style="box-shadow: 0 4px 20px rgba(0,0,0,0.5); max-width: 800px;">
        </div>
    """
    
    # Replace the end to inject our section at the bottom of the container
    if "<!-- Image Modal -->" in html:
        html = html.replace("<!-- Image Modal -->", new_section + "\n    <!-- Image Modal -->")
    elif "</body>" in html:
        html = html.replace("</body>", new_section + "\n</body>")
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
        
    print("✅ Successfully appended theoretical mapped reward landscape to report!")

if __name__ == "__main__":
    b64 = calculate_reward_surface()
    append_to_html_report(b64)
