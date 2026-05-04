import pandas as pd
import importlib.util
spec = importlib.util.spec_from_file_location("gv", "performance_analysis/2_generate_visualizations.py")
gv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gv)

df = pd.read_csv('performance_analysis/csv_data/raw_sweep_data.csv')
algo_a = "mappo"
algo_b = "masac"
df_h2h = df[((df['team_0_model'] == algo_a) & (df['team_1_model'] == algo_b)) | 
            ((df['team_0_model'] == algo_b) & (df['team_1_model'] == algo_a))].copy()
            
for i, ckpt in enumerate([40]):
    df_c = df_h2h[df_h2h['chkpt_idx'] == ckpt]
    metrics_list = []
    for _, row in df_c.iterrows():
        is_a_t0 = row['team_0_model'] == algo_a
        pref_a = "team_0_" if is_a_t0 else "team_1_"
        pref_b = "team_1_" if is_a_t0 else "team_0_"
        row_a = {"Model": algo_a.upper()}
        row_b = {"Model": algo_b.upper()}
        for cat, m_list in gv.CATEGORIES.items():
            for m in m_list:
                row_a[m] = gv.clean_value(row.get(pref_a + m, float('nan')), m)
                row_b[m] = gv.clean_value(row.get(pref_b + m, float('nan')), m)
        metrics_list.extend([row_a, row_b])
    df_m = pd.DataFrame(metrics_list)
    print(df_m.groupby("Model")["total_points"].mean())
