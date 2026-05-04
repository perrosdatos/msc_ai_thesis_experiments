import pandas as pd
import importlib.util
spec = importlib.util.spec_from_file_location("gv", "performance_analysis/2_generate_visualizations.py")
gv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gv)

df = pd.read_csv('performance_analysis/csv_data/raw_sweep_data.csv')
algo = "mappo"
df_algo = df[(df['team_0_model'] == algo) | (df['team_1_model'] == algo)].copy()
df_c = df_algo[df_algo['chkpt_idx'] == 40]
metrics_list = []
for _, row in df_c.iterrows():
    is_t0 = row['team_0_model'] == algo
    prefix = "team_0_" if is_t0 else "team_1_"
    row_data = {"Win": 1 if row["winner_model"] == algo else 0}
    for cat, m_list in gv.CATEGORIES.items():
        for m in m_list:
            if cat == "Reward Components" and not is_t0:
                row_data[m] = float('nan')
            else:
                row_data[m] = gv.clean_value(row.get(prefix + m, float('nan')), m)
    metrics_list.append(row_data)
df_m = pd.DataFrame(metrics_list)
print("Columns:", df_m.columns.tolist()[:5])
print("Data length:", len(df_m))
for m in gv.CATEGORIES["Core Performance"]:
    print(f"Non-null for {m}:", df_m[m].notna().sum())
    
print("Points variance:", df_m["total_points"].var())
