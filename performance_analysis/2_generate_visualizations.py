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

def generate_global_report(df, out_dir):
    print("Generating Global Tabbed Overview Report...")
    template = "plotly_dark"
    metrics = []
    
    for _, row in df.iterrows():
        t0_model, t1_model = row["team_0_model"], row["team_1_model"]
        winner = row["winner_model"]
        c = row["chkpt_idx"]
        
        # Build r0
        r0 = {
            "chkpt_idx": c,
            "Model": t0_model.upper(),
            "Points": row["team_0_total_points"],
            "Energy": row.get("team_0_energy_spent", np.nan),
            "Efficiency": row.get("team_0_efficiency", np.nan),
            "Points_History": row.get("team_0_points_history", "[]"),
            "Win": 1 if winner == t0_model else (0.5 if winner == "Tie" else 0)
        }
        for cat, m_list in CATEGORIES.items():
            for m in m_list:
                r0[m] = clean_value(row.get("team_0_" + m, np.nan), m)
        metrics.append(r0)
        
        # Build r1
        r1 = {
            "chkpt_idx": c,
            "Model": t1_model.upper(),
            "Points": row["team_1_total_points"],
            "Energy": row.get("team_1_energy_spent", np.nan),
            "Efficiency": row.get("team_1_efficiency", np.nan),
            "Points_History": row.get("team_1_points_history", "[]"),
            "Win": 1 if winner == t1_model else (0.5 if winner == "Tie" else 0)
        }
        for cat, m_list in CATEGORIES.items():
            for m in m_list:
                if cat == "Reward Components":
                    r1[m] = np.nan
                else:
                    r1[m] = clean_value(row.get("team_1_" + m, np.nan), m)
        metrics.append(r1)
        
    df_m = pd.DataFrame(metrics)
    
    # Define aggregation dictionary dynamically
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
                
    # Group by Model and Checkpoint
    df_agg = df_m.groupby(["Model", "chkpt_idx"]).agg(agg_dict).reset_index()
    df_agg["Win Rate (%)"] = (df_agg["Win"] * 100).round(2)
    df_agg["Model_Checkpoint"] = df_agg["Model"] + " (Ckpt " + df_agg["chkpt_idx"].astype(str) + ")"
    
    # Sort for Top 10 by Points
    df_top10 = df_agg.sort_values(by="Points", ascending=False).head(10)
    
    # Leaderboard Table
    df_table = df_top10[["Model_Checkpoint", "Points", "Energy", "Efficiency", "Win Rate (%)"]].copy()
    df_table.rename(columns={"Model_Checkpoint": "Configuration", "Points": "Avg Final Points", "Energy": "Avg Energy Spent", "Efficiency": "Avg Efficiency"}, inplace=True)
    table_html = f"<div class='table-container'>{df_table.to_html(classes='table table-striped', index=False)}</div>"
    
    # Step Data for Top 10
    step_data = []
    for _, row in df_top10.iterrows():
        m, c = row["Model"], row["chkpt_idx"]
        subset = df_m[(df_m["Model"] == m) & (df_m["chkpt_idx"] == c)]
        curve = extract_step_curve(subset, "Points_History")
        for step_idx, pts in enumerate(curve):
            step_data.append({
                "Step": step_idx,
                "Model_Checkpoint": row["Model_Checkpoint"],
                "Model": m,
                "Average Points": pts
            })
            
    df_steps = pd.DataFrame(step_data)
    fig_steps = px.line(df_steps, x="Step", y="Average Points", color="Model", line_group="Model_Checkpoint",
                        color_discrete_map=MODEL_COLORS,
                        title="Top 10 Configurations: Points Accumulation Over Time", template=template)
                        
    fig_bar = px.bar(df_top10, x="Model_Checkpoint", y="Points", color="Model",
                     color_discrete_map=MODEL_COLORS,
                     title="Top Configurations by Average Final Points", template=template, text_auto='.2f')
    fig_bar.update_layout(xaxis={'categoryorder':'total descending'})
    
    # Efficiency Scatter for ALL configs
    fig_scatter = px.scatter(df_agg, x="Energy", y="Points", color="Model", size="chkpt_idx",
                             color_discrete_map=MODEL_COLORS,
                             hover_name="Model_Checkpoint", title="Global Efficiency Landscape (All Checkpoints)",
                             template=template, opacity=0.8, size_max=20)

    # Building the TABS
    tabs_buttons = f'<button class="tablinks active" onclick="openTab(event, \'tab_summary\')">Executive Summary</button>\n'
    
    tabs_content = f'''<div id="tab_summary" class="tabcontent" style="display:block;">
    <h2>Detailed Configuration Leaderboard (Top 10)</h2>
    {table_html}
    <h2>Macro Visualizations</h2>
    <div class="plot-container">
        <div class="plot-box full">{fig_steps.to_html(full_html=False, include_plotlyjs=False)}</div>
        <div class="plot-box">{fig_bar.to_html(full_html=False, include_plotlyjs=False)}</div>
        <div class="plot-box">{fig_scatter.to_html(full_html=False, include_plotlyjs=False)}</div>
    </div>
    </div>'''

    for i, (cat, m_list) in enumerate(CATEGORIES.items()):
        safe_cat = cat.lower().replace(" & ", "_").replace(" ", "_")
        tabs_buttons += f'<button class="tablinks" onclick="openTab(event, \'tab_{safe_cat}\')">{cat}</button>\n'
        
        cat_html = f"<h2>{cat} Rankings</h2><div class='plot-container'>"
        for m in m_list:
            df_clean = df_agg.dropna(subset=[m]).copy()
            if not df_clean.empty:
                # Sort ascending for some metrics where lower is better, otherwise descending
                is_ascending = True if m in ["friendly_killed", "total_respawns", "time_to_first_relic", "info_propagation_delay", "dispersion_variance", "rc_collision_penalty_total", "rc_stagnation_penalty_total", "rc_overcrowding_penalty_total"] else False
                
                df_clean = df_clean.sort_values(by=m, ascending=is_ascending)
                
                # Keep Top 20 configurations to avoid unreadable charts
                df_plot = df_clean.head(20)
                
                fig = px.bar(df_plot, x="Model_Checkpoint", y=m, color="Model", title=f"Top 20 by {m}", template=template, text_auto='.2f', color_discrete_map=MODEL_COLORS)
                if is_ascending:
                    fig.update_layout(xaxis={'categoryorder':'total ascending'})
                else:
                    fig.update_layout(xaxis={'categoryorder':'total descending'})
                    
                cat_html += f"<div class='plot-box full'>{fig.to_html(full_html=False, include_plotlyjs=False)}</div>"
                
        cat_html += "</div>"
        tabs_content += f'<div id="tab_{safe_cat}" class="tabcontent" style="display:none;">{cat_html}</div>\n'

    html = f"""
    <html><head><title>Global Overview Report</title>
    {STYLE_CSS}
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script></head><body>
    <h1>Global Sweep Overview & Leaderboard</h1>
    <div class="tab">{tabs_buttons}</div>
    {tabs_content}
    {JS_SCRIPT}
    </body></html>
    """
    with open(os.path.join(out_dir, "0_global_overview.html"), "w") as f:
        f.write(html)

def generate_learning_progression_reports(df, out_dir):
    print("Generating Learning Progression Dashboards...")
    template = "plotly_dark"
    
    metrics_list = []
    for _, row in df.iterrows():
        c = row["chkpt_idx"]
        
        # Team 0
        r0 = {"chkpt_idx": c, "Model": row["team_0_model"].upper()}
        for cat, m_list in CATEGORIES.items():
            for m in m_list:
                r0[m] = clean_value(row.get("team_0_" + m, np.nan), m)
        metrics_list.append(r0)
        
        # Team 1
        r1 = {"chkpt_idx": c, "Model": row["team_1_model"].upper()}
        for cat, m_list in CATEGORIES.items():
            for m in m_list:
                # Rewards aren't logged for team_1 in this setup
                if cat == "Reward Components":
                    r1[m] = np.nan
                else:
                    r1[m] = clean_value(row.get("team_1_" + m, np.nan), m)
        metrics_list.append(r1)
        
    df_m = pd.DataFrame(metrics_list)
    df_m = df_m.sort_values(by="chkpt_idx")
    
    for cat, m_list in CATEGORIES.items():
        html = f"""
        <html><head><title>Learning Progression: {cat}</title>
        {STYLE_CSS}
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script></head><body>
        <h1>Learning Progression: {cat}</h1>
        <p style="text-align:center; color:#aaa;">Visualizing the evolutionary rate of learning for each algorithm across 40 checkpoints.</p>
        <div class="plot-container">
        """
        for m in m_list:
            df_clean = df_m.dropna(subset=[m])
            if not df_clean.empty:
                # The crucial part: x="chkpt_idx" automatically groups by x.
                # Boxplot with color="Model" will put the 3 boxes side-by-side for each checkpoint interval!
                fig = px.box(df_clean, x="chkpt_idx", y=m, color="Model", 
                             title=f"Progression of {m}", template=template, 
                             color_discrete_map=MODEL_COLORS)
                fig.update_layout(boxmode='group', xaxis_title="Checkpoint Index", yaxis_title=m)
                html += f"<div class='plot-box full'>{fig.to_html(full_html=False, include_plotlyjs=False)}</div>"
        
        html += "</div></body></html>"
        
        # Save file based on category name
        safe_name = cat.lower().replace(" & ", "_").replace(" ", "_")
        file_path = os.path.join(out_dir, f"1_progression_{safe_name}.html")
        with open(file_path, "w") as f:
            f.write(html)


def generate_individual_report(df, algo, out_dir):
    df_algo = df[(df['team_0_model'] == algo) | (df['team_1_model'] == algo)].copy()
    if df_algo.empty: return
    print(f"Generating Individual Outcomes Report for {algo.upper()}...")
    
    checkpoints = sorted(df_algo['chkpt_idx'].unique())
    tabs_buttons = ""
    tabs_content = ""
    template = "plotly_dark"
    color_map = {"Win": "#00CC96", "Loss": "#FF4B4B", "Tie": "#888888"}
    
    for i, ckpt in enumerate(checkpoints):
        df_c = df_algo[df_algo['chkpt_idx'] == ckpt]
        active_cls = "active" if i == 0 else ""
        disp_style = "display:block;" if i == 0 else "display:none;"
        tabs_buttons += f'<button class="tablinks {active_cls}" onclick="openTab(event, \'ckpt_{ckpt}\')">Checkpoint {ckpt}</button>\n'
        
        metrics_list = []
        for _, row in df_c.iterrows():
            is_t0 = row['team_0_model'] == algo
            prefix = "team_0_" if is_t0 else "team_1_"
            
            if row["winner_model"] == algo:
                outcome = "Win"
            elif row["winner_model"] == "Tie":
                outcome = "Tie"
            else:
                outcome = "Loss"
            
            row_data = {"Outcome": outcome, "Points_History": row.get(prefix+"points_history", "[]")}
            
            for cat, m_list in CATEGORIES.items():
                for m in m_list:
                    if cat == "Reward Components" and not is_t0:
                        row_data[m] = np.nan
                    else:
                        row_data[m] = clean_value(row.get(prefix + m, np.nan), m)
            metrics_list.append(row_data)
            
        df_m = pd.DataFrame(metrics_list)
        
        # Build category sections
        category_html = ""
        for cat, m_list in CATEGORIES.items():
            category_html += f"<h2>{cat}</h2>"
            
            # STATISTICAL TABLE
            df_cat = df_m[["Outcome"] + m_list].dropna(how="all", subset=m_list)
            if not df_cat.empty:
                stats = df_cat.groupby("Outcome").describe().round(2)
                stats = stats.loc[:, (slice(None), ['mean', 'std'])]
                category_html += f"<div class='table-container'>{stats.to_html(classes='table table-striped')}</div>"
            
            # BOXPLOTS
            category_html += "<div class='plot-container'>"
            for m in m_list:
                df_clean = df_m.dropna(subset=[m])
                if not df_clean.empty:
                    fig = px.box(df_clean, x="Outcome", y=m, color="Outcome", title=f"{m} by Match Outcome", template=template, color_discrete_map=color_map)
                    category_html += f"<div class='plot-box'>{fig.to_html(full_html=False, include_plotlyjs=False)}</div>"
            category_html += "</div>"
            
        # Curves
        curve = extract_step_curve(df_m, "Points_History")
        if len(curve) > 0:
            df_curve = pd.DataFrame({"Step": range(len(curve)), "Points": curve})
            f_curve = px.line(df_curve, x="Step", y="Points", title="Average Points Accumulation Over Time", template=template)
            curve_html = f"<h2>Step Progression</h2><div class='plot-container'><div class='plot-box full'>{f_curve.to_html(full_html=False, include_plotlyjs=False)}</div></div>"
        else:
            curve_html = ""
            
        tabs_content += f'''
        <div id="ckpt_{ckpt}" class="tabcontent" style="{disp_style}">
            {curve_html}
            {category_html}
        </div>
        '''
        
    html = f'''
    <html><head><title>{algo.upper()} Outcomes</title>
    {STYLE_CSS}
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script></head><body>
    <h1>{algo.upper()} Comprehensive Outcome Report</h1>
    <div class="tab">{tabs_buttons}</div>
    {tabs_content}
    {JS_SCRIPT}
    </body></html>
    '''
    with open(os.path.join(out_dir, f"{algo}_outcomes.html"), "w") as f:
        f.write(html)

def generate_comparative_report(df, algo_a, algo_b, out_dir):
    df_h2h = df[((df['team_0_model'] == algo_a) & (df['team_1_model'] == algo_b)) | 
                ((df['team_0_model'] == algo_b) & (df['team_1_model'] == algo_a))].copy()
    if df_h2h.empty: return
    print(f"Generating Comparative Report: {algo_a.upper()} vs {algo_b.upper()}...")
    
    checkpoints = sorted(df_h2h['chkpt_idx'].unique())
    tabs_buttons = ""
    tabs_content = ""
    template = "plotly_dark"
    
    for i, ckpt in enumerate(checkpoints):
        df_c = df_h2h[df_h2h['chkpt_idx'] == ckpt]
        active_cls = "active" if i == 0 else ""
        disp_style = "display:block;" if i == 0 else "display:none;"
        tabs_buttons += f'<button class="tablinks {active_cls}" onclick="openTab(event, \'ckpt_{ckpt}\')">Checkpoint {ckpt}</button>\n'
        
        metrics_list = []
        for _, row in df_c.iterrows():
            is_a_t0 = row['team_0_model'] == algo_a
            pref_a = "team_0_" if is_a_t0 else "team_1_"
            pref_b = "team_1_" if is_a_t0 else "team_0_"
            
            row_a = {"Model": algo_a.upper(), "Points_History": row.get(pref_a+"points_history", "[]")}
            row_b = {"Model": algo_b.upper(), "Points_History": row.get(pref_b+"points_history", "[]")}
            
            for cat, m_list in CATEGORIES.items():
                for m in m_list:
                    if cat == "Reward Components":
                        row_a[m] = clean_value(row.get(pref_a + m, np.nan), m) if pref_a == "team_0_" else np.nan
                        row_b[m] = clean_value(row.get(pref_b + m, np.nan), m) if pref_b == "team_0_" else np.nan
                    else:
                        row_a[m] = clean_value(row.get(pref_a + m, np.nan), m)
                        row_b[m] = clean_value(row.get(pref_b + m, np.nan), m)
            metrics_list.extend([row_a, row_b])
            
        df_m = pd.DataFrame(metrics_list)
        
        # Build category sections
        category_html = ""
        for cat, m_list in CATEGORIES.items():
            category_html += f"<h2>{cat}</h2>"
            
            # STATISTICAL TABLE
            df_cat = df_m[["Model"] + m_list].dropna(how="all", subset=m_list)
            if not df_cat.empty:
                stats = df_cat.groupby("Model").describe().round(2)
                stats = stats.loc[:, (slice(None), ['mean', 'std', 'max'])]
                category_html += f"<div class='table-container'>{stats.to_html(classes='table table-striped')}</div>"

            # BOXPLOTS
            category_html += "<div class='plot-container'>"
            for m in m_list:
                df_clean = df_m.dropna(subset=[m])
                if not df_clean.empty:
                    fig = px.box(df_clean, x="Model", y=m, color="Model", title=f"{m} Variance", template=template, color_discrete_map=MODEL_COLORS)
                    category_html += f"<div class='plot-box'>{fig.to_html(full_html=False, include_plotlyjs=False)}</div>"
            category_html += "</div>"
            
        # Curves
        curve_a = extract_step_curve(df_m[df_m["Model"] == algo_a.upper()], "Points_History")
        curve_b = extract_step_curve(df_m[df_m["Model"] == algo_b.upper()], "Points_History")
        step_len = max(len(curve_a), len(curve_b))
        if step_len > 0:
            df_curve = pd.DataFrame({
                "Step": range(step_len),
                f"{algo_a.upper()} Points": curve_a + [curve_a[-1]]*(step_len-len(curve_a)),
                f"{algo_b.upper()} Points": curve_b + [curve_b[-1]]*(step_len-len(curve_b))
            })
            f_curve = px.line(df_curve, x="Step", y=[f"{algo_a.upper()} Points", f"{algo_b.upper()} Points"],
                              title="Points Accumulation Over Time", template=template,
                              color_discrete_map={f"{algo_a.upper()} Points": MODEL_COLORS[algo_a.upper()], f"{algo_b.upper()} Points": MODEL_COLORS[algo_b.upper()]})
            curve_html = f"<h2>Step Progression</h2><div class='plot-container'><div class='plot-box full'>{f_curve.to_html(full_html=False, include_plotlyjs=False)}</div></div>"
        else:
            curve_html = ""
            
        tabs_content += f'''
        <div id="ckpt_{ckpt}" class="tabcontent" style="{disp_style}">
            {curve_html}
            {category_html}
        </div>
        '''
        
    html = f'''
    <html><head><title>{algo_a.upper()} vs {algo_b.upper()}</title>
    {STYLE_CSS}
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script></head><body>
    <h1>{algo_a.upper()} vs {algo_b.upper()} Comprehensive Head-to-Head Report</h1>
    <div class="tab">{tabs_buttons}</div>
    {tabs_content}
    {JS_SCRIPT}
    </body></html>
    '''
    with open(os.path.join(out_dir, f"{algo_a}_vs_{algo_b}.html"), "w") as f:
        f.write(html)

def main():
    data_path = "/home/carlos/Documents/github/msc_ai_thesis_experiments/performance_analysis/csv_data/raw_sweep_data.csv"
    if not os.path.exists(data_path):
        print(f"Data file not found at {data_path}. Run sweep first.")
        return
        
    df = pd.read_csv(data_path)
    
    # Calculate Individual Skill Averages (mean over 16 agents)
    for t in [0, 1]:
        for metric in ["energy_spent", "points_generated", "tiles_explored", "avg_lifespan"]:
            cols = [f"team_{t}_agent_{i}_{metric}" for i in range(16)]
            if all(c in df.columns for c in cols):
                df[f"team_{t}_ind_avg_{metric}"] = df[cols].mean(axis=1)
    
    out_dir = "/home/carlos/Documents/github/msc_ai_thesis_experiments/performance_analysis/html_reports/sweep_dashboards"
    os.makedirs(out_dir, exist_ok=True)
    
    algos = ["mappo", "masac", "qmix"]
    
    # Generate Global Overview
    generate_global_report(df, out_dir)
    
    # Generate Progression Reports
    generate_learning_progression_reports(df, out_dir)
    
    # Generate Outcomes (Old _progression)
    for algo in algos:
        generate_individual_report(df, algo, out_dir)
    
    # Generate H2H Comparative Reports
    pairs = [("mappo", "masac"), ("mappo", "qmix"), ("masac", "qmix")]
    for a, b in pairs:
        generate_comparative_report(df, a, b, out_dir)
        
    print(f"\n✅ All Interactive Dashboards generated successfully at: {out_dir}")

if __name__ == "__main__":
    main()
