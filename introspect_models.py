import torch
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'BenchMARL')))
import hydra
from omegaconf import OmegaConf
from hydra.core.global_hydra import GlobalHydra
from benchmarl.hydra_config import load_experiment_from_hydra

def load_algo(algo, chkpt_path):
    benchmarl_conf_path = "BenchMARL/benchmarl/conf"
    if GlobalHydra.instance().is_initialized():
        GlobalHydra.instance().clear()
        
    with hydra.initialize(version_base=None, config_path=benchmarl_conf_path):
        cfg = hydra.compose(
            config_name="config",
            overrides=[
                f"algorithm={algo}", "task=lux/match_v2", "model=layers/cnn_lux_16ch",
                "model@critic_model=layers/cnn_lux_16ch", "experiment.sampling_device=cpu",
                "experiment.train_device=cpu", "experiment.buffer_device=cpu",
                "experiment.checkpoint_interval=150000", "experiment.loggers=[]", "seed=42"
            ],
        )
        experiment = load_experiment_from_hydra(cfg, task_name="lux/match_v2")
        # Load weights to get exact structure that was used
        experiment.load_state_dict(torch.load(chkpt_path, map_location="cpu", weights_only=False))
        return experiment.algorithm

checkpoints = {
    "mappo": "performance_analysis/models/mappo_match_v2_cnn__d20a96a5_26_04_27-09_48_59/checkpoints/checkpoint_150000.pt",
    "masac": "performance_analysis/models/masac_match_v2_cnn__5490d15d_26_04_27-22_59_06/checkpoints/checkpoint_150000.pt",
    "qmix": "performance_analysis/models/qmix_match_v2_cnn__2a7d90ea_26_04_29-22_52_16/checkpoints/checkpoint_6000000.pt"
}

for algo, path in checkpoints.items():
    print(f"\n{'='*50}\nIntrospecting {algo.upper()}\n{'='*50}")
    if not os.path.exists(path):
        print(f"Path not found: {path}")
        continue
    algorithm = load_algo(algo, path)
    policy = algorithm.get_policy_for_collection()
    print("POLICY NETWORK:")
    print(policy)
    
    if hasattr(algorithm, 'get_loss'):
        try:
            loss_module = algorithm.get_loss()
            print("\nLOSS MODULE / CRITIC / MIXER:")
            print(loss_module)
        except Exception as e:
            print("Could not get loss module:", e)
