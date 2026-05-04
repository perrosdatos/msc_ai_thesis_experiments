import pandas as pd
df = pd.read_csv("performance_analysis/csv_data/raw_sweep_data.csv")
metrics = []
for _, row in df.iterrows():
    metrics.append({
        "chkpt_idx": row["chkpt_idx"],
        "Model": row["team_0_model"],
        "Points": row["team_0_total_points"],
    })
    metrics.append({
        "chkpt_idx": row["chkpt_idx"],
        "Model": row["team_1_model"],
        "Points": row["team_1_total_points"],
    })
df_m = pd.DataFrame(metrics)
df_agg = df_m.groupby(["Model", "chkpt_idx"])["Points"].mean().reset_index()
df_agg = df_agg.sort_values(by="Points", ascending=False).head(10)
print(df_agg)
