from __future__ import annotations

import json
import logging
import torch
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence
from transformers import TrainerCallback, TrainingArguments, TrainerState, TrainerControl



# ---------------------------------------------------------------------------
# Typed history container ----------------------------------------------------
# ---------------------------------------------------------------------------
@dataclass
class History:
    train_steps:  List[int] = field(default_factory=list)
    train_loss:   List[float] = field(default_factory=list)
    train_loss_ce: List[float] = field(default_factory=list)   # NEW
    train_loss_vd: List[float] = field(default_factory=list)   # NEW

    eval_steps:  List[int] = field(default_factory=list)
    eval_loss:   List[float] = field(default_factory=list)
    eval_loss_ce: List[float] = field(default_factory=list)    # NEW
    eval_loss_vd: List[float] = field(default_factory=list)    # NEW

    train_steps_ce: List[int] = field(default_factory=list)
    train_steps_vd: List[int] = field(default_factory=list)
    eval_steps_ce: List[int] = field(default_factory=list)
    eval_steps_vd: List[int] = field(default_factory=list)


    def to_dict(self) -> Dict[str, Sequence]:
        return asdict(self)

class BaseTrainer(ABC):
    """Defines the minimal interface + common utilities for custom trainers."""

    def __init__(self, config):
        self.cfg = config  # `cfg` keeps consistency with other modules
        self.history = History()
        self.log = logging.getLogger(self.__class__.__name__)
        

    @abstractmethod
    def prepare_components(self) -> None:  
        """Load model, tokenizer, dataset and build training args."""

    @abstractmethod
    def _get_common_trainer_kwargs(self) -> Dict:
        """Return kwargs forwarded to the underlying HF/TRL Trainer."""

    @abstractmethod
    def train(self) -> None:  # noqa: D401
        """Run optimisation loop and optional evaluation."""
        
    
    def plot_history(self, output_dir: str | Path, save_json: bool = True, show: bool = False) -> None:
        import matplotlib.pyplot as plt

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        
        plt.figure(figsize=(9, 5))
        if self.history.train_steps:
            plt.plot(self.history.train_steps, self.history.train_loss, label="Train loss")
        if self.history.eval_steps:
            plt.plot(self.history.eval_steps, self.history.eval_loss, label="Eval loss")
        if self.history.train_loss_ce:
            plt.plot(self.history.train_steps_ce, self.history.train_loss_ce, linestyle="--", label="Train loss_ce")
        if self.history.train_loss_vd:
            plt.plot(self.history.train_steps_vd, self.history.train_loss_vd, linestyle="--", label="Train loss_vd")

        plt.title("Loss curves")
        plt.xlabel("Steps")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out / "loss_curve.png")
        if show:
            plt.show()
        plt.close()
        
        if save_json:
            with open(out / "loss_history.json", "w", encoding="utf-8") as fh:
                json.dump(self.history.to_dict(), fh, indent=2)

class HistoryCallback(TrainerCallback):
    def __init__(self, wrapper: BaseTrainer):
        super().__init__()
        self.wrapper = wrapper
        
    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs=None,
        **kwargs,
    ) -> None:
        if not state.log_history:
            return

        rec = state.log_history[-1]
        step = state.global_step
        for k, v in rec.items():
            if isinstance(v, torch.Tensor):
                rec[k] = v.item()

        # --- training section ---------------------------------
        if "loss" in rec:             # aggregate loss
            self.wrapper.history.train_steps.append(step)
            self.wrapper.history.train_loss.append(rec["loss"])
        if "loss_ce" in rec:
            self.wrapper.history.train_steps_ce.append(step)
            self.wrapper.history.train_loss_ce.append(rec["loss_ce"])
        if "loss_vd" in rec:
            self.wrapper.history.train_steps_vd.append(step)
            self.wrapper.history.train_loss_vd.append(rec["loss_vd"])

        # --- evaluation section -------------------------------
        if "eval_loss" in rec:
            self.wrapper.history.eval_steps.append(step)
            self.wrapper.history.eval_loss.append(rec["eval_loss"])
        if "eval_loss_ce" in rec:
            self.wrapper.history.eval_steps_ce.append(step)
            self.wrapper.history.eval_loss_ce.append(rec["eval_loss_ce"])
        if "eval_loss_vd" in rec:
            self.wrapper.history.eval_steps_vd.append(step)
            self.wrapper.history.eval_loss_vd.append(rec["eval_loss_vd"])

        self.wrapper.log.debug("%s", rec)
