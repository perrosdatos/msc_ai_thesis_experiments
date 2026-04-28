import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Add core path to import the algorithm
benchmarl_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../BenchMARL"))
sys.path.append(benchmarl_dir)
from benchmarl.environments.lux.reward_combat import compute_combat_rewards

def get_base64_image(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', transparent=True, dpi=120)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode('utf-8')

def calculate_reward_surface():
    """
    Simulates the logic from reward_combat.py
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
    delta_energy[:, 0] = 5.0 # Positive energy delta
    
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

    # Combat specific dummies
    footprint_map = np.zeros((B, 24, 24), dtype=np.float32)
    delta_masks = np.zeros((B, 2, 16), dtype=np.int16)
    all_positions = np.zeros((B, 2, 16, 2), dtype=np.int32)

    # Exclusively use COMBAT REWARD SIMULATION
    shaped_global, shaped_local, components = compute_combat_rewards(
        current_team_mask,
        current_team_pos,
        actions,
        delta_points,
        delta_visible,
        delta_energy,
        spawn_pos,
        known_relic_mask,
        known_relic_pos,
        step_count,
        footprint_map,
        delta_masks,
        all_positions
    )
    
    # Combine global + local reward for Agent 0
    total_shaped_rewards = shaped_global + shaped_local[:, 0]
    
    # Reshape the output mathematically
    surface = total_shaped_rewards.reshape((size, size))
    
    # Visualize Heatmap
    fig, ax = plt.subplots(figsize=(8, 7))
    cmap = plt.cm.get_cmap('plasma')
    im = ax.imshow(surface.T, cmap=cmap, origin='lower')
    
    # Annotate Relics & Spawn
    ax.scatter(relic_centers[:, 0], relic_centers[:, 1], c='gold', marker='*', s=300, edgecolor='black', label="Relic Center")
    ax.scatter(spawn[0], spawn[1], c='cyan', marker='X', s=150, edgecolor='black', label="Spawn Point")
    
    # Contour lines to show gradient paths
    ax.contour(surface.T, levels=15, colors='white', alpha=0.3)
    
    ax.set_title("Combat + Dense Exploration Reward Landscape", color='white', fontsize=14)
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
        report_path = os.path.join(out_dir, "luxsap_input_report.html")
    else:
        report_path = "/home/carlos/Documents/github/msc_ai_thesis_experiments/html_reports/luxsap_input_report.html"
    
    try:
        with open(report_path, "r", encoding="utf-8") as f:
            html = f.read()
    except Exception:
        print(f"Could not find {report_path} to append to. Generating a standalone file instead.")
        html = "<html><body></body></html>"
        
    # Construct new section
    new_section = f"""
        <h3 class="section-header text-danger mt-5">4. Theoretical Reward Landscape (Combat + Exploration)</h3>
        <p class="text-muted">This heatmap showcases the fundamental pull of the environment. In addition to relic exploration, agents are subject to explicit combat modifiers: <code>combat_kill</code> and <code>combat_death</code>. The landscape here simulates the exploration baseline, mapping out the centripetal and centrifugal forces guiding unit pathing even before dynamic combat occurs.</p>
        <div class="alert alert-warning">
            <strong>Context Simulation:</strong> Assuming Agent 0 idles across coordinates. The landscape inherits all V2 exploration characteristics (overcrowding, dispersion, farming limits) but uses the fully extended <code>compute_combat_rewards</code> mathematical model from <code>reward_combat.py</code>.
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
        
    print("✅ Successfully appended theoretical combat mapped reward landscape to report!")

if __name__ == "__main__":
    b64 = calculate_reward_surface()
    append_to_html_report(b64)
