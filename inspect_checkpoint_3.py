import torch
path = "BenchMARL/outputs/2026-04-21/20-31-15/mappo_match_v2_cnn__3af9bf08_26_04_21-20_31_15/checkpoints/checkpoint_150000.pt"
ckpt = torch.load(path, map_location="cpu")
print("LOSS AGENTS KEYS:", list(ckpt['loss_agents'].keys())[:20])
if "collector" in ckpt:
    print("COLLECTOR KEYS:", list(ckpt['collector'].keys())[:20])
