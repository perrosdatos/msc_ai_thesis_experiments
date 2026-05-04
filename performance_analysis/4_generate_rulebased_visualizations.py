import os
import ast
import json
import numpy as np
import pandas as pd
import plotly.express as px

def safe_parse_json(s):
    if pd.isna(s): return []
    if isinstance(s, list): return s
    try:
        return json.loads(s)
    except:
        try:
            return ast.literal_eval(s)
        except:
            return []

def extract_step_curve(df, col):
    curves = []
    for s in df[col]:
        arr = safe_parse_json(s)
        if len(arr) > 0:
            curves.append(arr)
    if len(curves) == 0:
        return []
    max_len = max(len(c) for c in curves)
    padded = []
    for c in curves:
        if len(c) < max_len:
            padded.append(c + [c[-1]] * (max_len - len(c)))
        else:
            padded.append(c)
    return np.mean(padded, axis=0).tolist()

CATEGORIES = {
    "Core Performance": [
        "total_points", "energy_spent", "efficiency", "resource_monopoly"
    ],
    "Combat Metrics": [
        "friendly_killed", "opponents_killed", "combat_dominance", "total_respawns"
    ],
    "Exploration & Synergy": [
        "map_exploration_prop", "time_to_first_relic", "dispersion_variance", "info_propagation_delay", "synergy_handoffs"
    ],
    "Individual Agent Skills": [
        "ind_avg_energy_spent", "ind_avg_points_generated", "ind_avg_tiles_explored", "ind_avg_lifespan"
    ],
    "Reward Components": [
        "rc_total_reward_total", "rc_local_point_generation_total", "rc_relic_discovery_total",
        "rc_fog_discovery_total", "rc_collision_penalty_total", "rc_dispersion_bonus_total",
        "rc_novelty_bonus_total", "rc_relic_proximity_total", "rc_relic_farming_total",
        "rc_energy_gain_total", "rc_stagnation_penalty_total", "rc_overcrowding_penalty_total"
    ]
}

MODEL_COLORS = {
    "MAPPO": "#00CC96",  # Green
    "MASAC": "#AB63FA",  # Purple
    "QMIX": "#FFA15A"    # Orange
}

def clean_value(val, metric_name):
    if pd.isna(val): return np.nan
    if metric_name == "time_to_first_relic" and val == -1: return np.nan
    return float(val)

STYLE_CSS = """
    <style>
        body { font-family: 'Inter', sans-serif; background: #0E1117; color: white; padding: 20px; }
        h1 { text-align: center; color: #19D3F3; font-weight: 800; }
        h2 { color: #FF4B4B; margin-top: 50px; border-bottom: 2px solid #333; padding-bottom: 10px; font-weight: 700; }
        .tab { display: flex; gap: 10px; margin-bottom: 20px; overflow-x: auto; padding-bottom: 10px; }
        .tab button { flex: 1; min-width: 150px; background-color: #262730; border: none; border-radius: 8px; cursor: pointer; padding: 15px; font-weight: 600; color: white; transition: 0.2s; }
        .tab button:hover { background-color: #31333F; transform: translateY(-2px); }
        .tab button.active { background-color: #19D3F3; box-shadow: 0 4px 12px rgba(25,211,243,0.3); color: #000; }
        .tabcontent { animation: fadeEffect 0.5s; }
        @keyframes fadeEffect { from {opacity: 0;} to {opacity: 1;} }
        .plot-container { display: flex; flex-wrap: wrap; gap: 20px; justify-content: center; margin-bottom: 40px; }
        .plot-box { width: 48%; border: 1px solid #333; border-radius: 10px; background: #1a1a1a; padding: 10px; box-sizing: border-box; }
        .plot-box.full { width: 100%; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; font-size: 0.9em; border-radius: 8px; overflow: hidden; background-color: #1a1a1a; }
        th, td { padding: 12px 15px; text-align: right; border-bottom: 1px solid #333; }
        th { background-color: #262730; color: #19D3F3; font-weight: bold; text-align: center; }
        tr:hover { background-color: #31333F; }
        .table-container { overflow-x: auto; margin-bottom: 30px; border: 1px solid #333; border-radius: 10px; }
    </style>
"""

JS_SCRIPT = """
    <script>
    function openTab(evt, tabName) {
        var i, tabcontent, tablinks;
        tabcontent = document.getElementsByClassName("tabcontent");
        for (i = 0; i < tabcontent.length; i++) { tabcontent[i].style.display = "none"; }
        tablinks = document.getElementsByClassName("tablinks");
        for (i = 0; i < tablinks.length; i++) { tablinks[i].className = tablinks[i].className.replace(" active", ""); }
        document.getElementById(tabName).style.display = "block";
        evt.currentTarget.className += " active";
    }
    </script>
"""

def generate_rulebased_overview(df, out_dir):
    print("Generating Rule-Based Overview Report...")
    template = "plotly_dark"
    metrics = []
    
    for _, row in df.iterrows():
        # In rule-based eval, team 0 is our Model, team 1 is BuiltinAI
        model_name = row["team_0_model"]
        c = row["chkpt_idx"]
        winner = row["winner_model"]
        
        r0 = {
            "chkpt_idx": c,
            "Model": model_name.upper(),
            "Points": row["team_0_total_points"],
            "Energy": row.get("team_0_energy_spent", np.nan),
            "Efficiency": row.get("team_0_efficiency", np.nan),
            "Points_History": row.get("team_0_points_history", "[]"),
            "Win": 1 if winner == model_name else (0.5 if winner == "Tie" else 0)
        }
        for cat, m_list in CATEGORIES.items():
            for m in m_list:
                r0[m] = clean_value(row.get("team_0_" + m, np.nan), m)
        metrics.append(r0)
        
    df_m = pd.DataFrame(metrics)
    
    agg_dict = {
        "Points": "mean",
        "Energy": "mean",
        "Efficiency": "mean",
        "Win": "mean"
    }
    for cat, m_list in CATEGORIES.items():
        for m in m_list:
            if m not in agg_dict:
                agg_dict[m] = "mean"
                
    df_agg = df_m.groupby(["Model", "chkpt_idx"]).agg(agg_dict).reset_index()
    df_agg["Win Rate (%)"] = (df_agg["Win"] * 100).round(2)
    df_agg["Configuration"] = df_agg["Model"] + " (Ckpt " + df_agg["chkpt_idx"].astype(str) + ")"
    
    # Sort for Leaderboard
    df_top10 = df_agg.sort_values(by="Points", ascending=False).head(15) # Show all 15 evaluated checkpoints
    
    df_table = df_top10[["Configuration", "Points", "Energy", "Efficiency", "Win Rate (%)"]].copy()
    df_table.rename(columns={"Points": "Avg Final Points", "Energy": "Avg Energy Spent", "Efficiency": "Avg Efficiency"}, inplace=True)
    table_html = f"<div class='table-container'>{df_table.to_html(classes='table table-striped', index=False)}</div>"
    
    fig_bar = px.bar(df_top10, x="Configuration", y="Points", color="Model",
                     color_discrete_map=MODEL_COLORS,
                     title="Top Configurations by Average Final Points against Rule-Based Agent", 
                     template=template, text_auto='.2f')
    fig_bar.update_layout(xaxis={'categoryorder':'total descending'})
    
    fig_winrate = px.bar(df_top10.sort_values(by="Win Rate (%)", ascending=False), 
                         x="Configuration", y="Win Rate (%)", color="Model",
                         color_discrete_map=MODEL_COLORS,
                         title="Top Configurations by Win Rate (%) against Rule-Based Agent", 
                         template=template, text_auto='.2f')
                         
    fig_scatter = px.scatter(df_agg, x="Energy", y="Points", color="Model", size="chkpt_idx",
                             color_discrete_map=MODEL_COLORS,
                             hover_name="Configuration", title="Efficiency Landscape vs Rule-Based",
                             template=template, opacity=0.8, size_max=20)

    tabs_buttons = f'<button class="tablinks active" onclick="openTab(event, \'tab_summary\')">Executive Summary</button>\n'
    
    tabs_content = f'''<div id="tab_summary" class="tabcontent" style="display:block;">
    <h2>Top Checkpoints vs Rule-Based Agent</h2>
    {table_html}
    <h2>Macro Visualizations</h2>
    <div class="plot-container">
        <div class="plot-box full">{fig_bar.to_html(full_html=False, include_plotlyjs=False)}</div>
        <div class="plot-box full">{fig_winrate.to_html(full_html=False, include_plotlyjs=False)}</div>
        <div class="plot-box">{fig_scatter.to_html(full_html=False, include_plotlyjs=False)}</div>
    </div>
    </div>'''

    for i, (cat, m_list) in enumerate(CATEGORIES.items()):
        safe_cat = cat.lower().replace(" & ", "_").replace(" ", "_")
        tabs_buttons += f'<button class="tablinks" onclick="openTab(event, \'tab_{safe_cat}\')">{cat}</button>\n'
        
        cat_html = f"<h2>{cat} Rankings vs Rule-Based</h2><div class='plot-container'>"
        for m in m_list:
            df_clean = df_agg.dropna(subset=[m]).copy()
            if not df_clean.empty:
                is_ascending = True if m in ["friendly_killed", "total_respawns", "time_to_first_relic", "info_propagation_delay", "dispersion_variance", "rc_collision_penalty_total", "rc_stagnation_penalty_total", "rc_overcrowding_penalty_total"] else False
                df_clean = df_clean.sort_values(by=m, ascending=is_ascending)
                df_plot = df_clean.head(15)
                
                fig = px.bar(df_plot, x="Configuration", y=m, color="Model", title=f"Ranking by {m}", template=template, text_auto='.2f', color_discrete_map=MODEL_COLORS)
                if is_ascending:
                    fig.update_layout(xaxis={'categoryorder':'total ascending'})
                else:
                    fig.update_layout(xaxis={'categoryorder':'total descending'})
                    
                cat_html += f"<div class='plot-box full'>{fig.to_html(full_html=False, include_plotlyjs=False)}</div>"
                
        cat_html += "</div>"
        tabs_content += f'<div id="tab_{safe_cat}" class="tabcontent" style="display:none;">{cat_html}</div>\n'

    html = f"""
    <html><head><title>Rule-Based Overview Report</title>
    {STYLE_CSS}
    {JS_SCRIPT}
    <script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script></head><body>
    <h1>Rule-Based Opponent Evaluation Overview</h1>
    <div class="tab">{tabs_buttons}</div>
    {tabs_content}
    </body></html>
    """
    with open(os.path.join(out_dir, "0_rulebased_overview.html"), "w") as f:
        f.write(html)

def main():
    data_path = "/home/carlos/Documents/github/msc_ai_thesis_experiments/performance_analysis/csv_data/rulebased_eval_data.csv"
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}. Run evaluate_rulebased.py first.")
        return
        
    df = pd.read_csv(data_path)
    
    # Calculate Individual Skill Averages (mean over 16 agents) for our model (team 0)
    for metric in ["energy_spent", "points_generated", "tiles_explored", "avg_lifespan"]:
        cols = [f"team_0_agent_{i}_{metric}" for i in range(16)]
        if all(c in df.columns for c in cols):
            df[f"team_0_ind_avg_{metric}"] = df[cols].mean(axis=1)
    
    out_dir = "/home/carlos/Documents/github/msc_ai_thesis_experiments/performance_analysis/html_reports/rulebased_dashboards"
    os.makedirs(out_dir, exist_ok=True)
    
    generate_rulebased_overview(df, out_dir)
        
    print(f"\n✅ Rule-Based Dashboards generated successfully at: {out_dir}")

if __name__ == "__main__":
    main()
