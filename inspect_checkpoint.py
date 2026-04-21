import torch
path = "BenchMARL/outputs/2026-04-21/20-31-15/mappo_match_v2_cnn__3af9bf08_26_04_21-20_31_15/checkpoints/checkpoint_150000.pt"
ckpt = torch.load(path, map_location="cpu")
if isinstance(ckpt, dict):
    print("KEYS:", ckpt.keys())
    if "experiment" in ckpt:
        print("EXPERIMENT type:", type(ckpt["experiment"]))
    if "model" in ckpt:
        print("MODEL type:", type(ckpt["model"]))
