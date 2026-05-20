from __future__ import annotations
import os
os.environ['UNSLOTH_RETURN_LOGITS'] = '1'
import wandb,os
import argparse
import copy
import logging
import random
import sys
from types import SimpleNamespace
from typing import Literal

import numpy as np
import torch


from config import get_config, TrainConfig
from sft_trainer import UnslothSFTTrainer
from eval_utils import Evaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("main")

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args() -> SimpleNamespace:
    p = argparse.ArgumentParser(description="Train or evaluate Unsloth models")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    p.add_argument("--dataset_path", type=str, default=None, help="Path to dataset (overrides config file)") 
    p.add_argument("--output_dir", type=str, default=None, help="Path to output directory (overrides config file)") 
    return p.parse_args(namespace=SimpleNamespace())



def run(cfg: TrainConfig, phase: Literal["train", "test"]):
    """Dispatches to the appropriate trainer or evaluator based on *phase*."""
    if phase == "train":
        TrainerCls = None
        if cfg.training_method == "sft":
            TrainerCls = UnslothSFTTrainer
            
        trainer = TrainerCls(cfg)
        trainer.prepare_components()
        trainer.train()
    else:  # test
        log.info("Evaluating model …")
        Evaluator(cfg).evaluate_by_temp()
        # QA eval
        # ev = Evaluator(cfg)
        # ev.evaluate_qa("csqa")
        # ev.evaluate_qa("csqa2")
        # ev.evaluate_qa("piqa")


def main():
    args = parse_args()
    base_cfg = get_config(args.config)
    if args.dataset_path is not None:
        base_cfg.dataset_path = args.dataset_path
        print(f"Overriding dataset_path to {base_cfg.dataset_path}")
    if args.output_dir is not None:
        base_cfg.training_args["output_dir"] = args.output_dir
        print(f"Overriding output_dir to {base_cfg.training_args['output_dir']}")
    
    set_global_seed(getattr(base_cfg, "seed", 42))
    modes = base_cfg.mode
    if "train" in modes:
        wandb.init(project="synthetic", 
                entity="synthetic",
                name=f"llama3_sft",
                dir  = base_cfg.training_args['output_dir'],
                settings=wandb.Settings(init_timeout=120))
        cfg_l = copy.deepcopy(base_cfg)
        
        log.info("Dataset using: %s", cfg_l.dataset_path)
        log.info("Output dir: %s", cfg_l.training_args["output_dir"])
        
        run(cfg_l, "train")
        wandb.finish()
    elif "test" in modes:
        run(base_cfg, "test")

if __name__ == "__main__":
    main()