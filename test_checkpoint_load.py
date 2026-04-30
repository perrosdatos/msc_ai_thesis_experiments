import sys
import os
import torch
import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'BenchMARL')))
from benchmarl.hydra_config import load_experiment_from_hydra

checkpoint_path = "/home/carlos/Documents/github/msc_ai_thesis_experiments/BenchMARL/outputs/2026-04-27/22-59-06/masac_match_v2_cnn__5490d15d_26_04_27-22_59_06/checkpoints/checkpoint_4050000.pt"

if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()

benchmarl_conf_path = "BenchMARL/benchmarl/conf"
with hydra.initialize(version_base=None, config_path=benchmarl_conf_path):
    cfg = hydra.compose(
        config_name="config",
        overrides=[
            f"algorithm=masac",
            "task=lux/match_v2",
            "model=layers/cnn_lux_16ch",
            "model@critic_model=layers/cnn_lux_16ch",
            "experiment.sampling_device=cpu",
            "experiment.train_device=cpu",
            "experiment.buffer_device=cpu",
            "experiment.checkpoint_interval=120000",
            "experiment.loggers=[]",
            "seed=42"
        ],
    )
    experiment = load_experiment_from_hydra(cfg, task_name="lux/match_v2")
    policy_before = experiment.algorithm.get_policy_for_collection()
    
    # get a sample weight before
    weight_before = next(policy_before.parameters()).clone()

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    experiment.load_state_dict(state_dict)

    policy_after = experiment.algorithm.get_policy_for_collection()
    weight_after = next(policy_after.parameters()).clone()
    
    print(f"Weight before loading: {weight_before.flatten()[:5]}")
    print(f"Weight after loading: {weight_after.flatten()[:5]}")
    
    if torch.allclose(weight_before, weight_after):
         print("WARNING: Weights did NOT change! The checkpoint is NOT being loaded correctly into the policy.")
    else:
         print("SUCCESS: Weights changed. The checkpoint is loaded.")
