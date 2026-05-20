from __future__ import annotations
from unsloth import FastLanguageModel, FastModel
from unsloth import is_bfloat16_supported
from unsloth.chat_templates import train_on_responses_only,get_chat_template

import gc
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from transformers import TrainingArguments, EarlyStoppingCallback
from trl import SFTTrainer
from datasets import load_from_disk
from transformers import DataCollatorForSeq2Seq
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)

# local imports
from base_trainer import BaseTrainer, HistoryCallback

def load_model_and_tokenizer(config):

    if "gemma" in config.model_name.lower():
        model, tokenizer = FastModel.from_pretrained(
            model_name=config.model_name,
            max_seq_length=config.max_length,
            load_in_4bit=False,
            device_map={'': local_rank},
        )
        print("Peft model loaded")
        model = FastModel.get_peft_model(model, **config.lora_args)
    else:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=config.model_name,
            max_seq_length=config.max_length,
            load_in_4bit=False,
            device_map={'': local_rank},
        )
        print("Peft model loaded")
        model = FastLanguageModel.get_peft_model(model, **config.lora_args)
        
    if config.model_type == "instruct":
        if "Qwen" in config.model_name:
            tokenizer = get_chat_template(
                tokenizer,
                chat_template="qwen2.5",
            )
        elif "llama" in config.model_name.lower():
            tokenizer = get_chat_template(
                tokenizer,
                chat_template="llama-3",
            )
        elif "gemma" in config.model_name.lower():
            tokenizer = get_chat_template(
                tokenizer,
                chat_template="gemma-3",
            )
    model.print_trainable_parameters()
    
    return model, tokenizer

def process_data(config, tokenizer):
    dataset = load_from_disk(config.dataset_path)
    
    
    print(f"[Debug Mode Status] debug={getattr(config, 'debug', False)}")
    if getattr(config, "debug", False):
        print("✅ Debug mode ACTIVE")
        dataset["train"] = dataset["train"].select(range(min(67389, len(dataset["train"]))))
        print(f"Original train size: {len(dataset['train'])}")
    else:
        print("❌ Debug mode INACTIVE")
    
    def formatting_prompts_func(examples):
        inputs = examples["input"]
        outputs = examples["output"]
        texts = []

        for inp, out in zip(inputs, outputs):
            instruction_comve = "Given an implausible or counterfactual statement, generate one short explanation that why it is implausible or counterfactual using background commonsense knowledge: "
            instruction_commongen = f"Given several keywords, generate one coherent sentence that contains all the required keywords using background commonsense knowledge: "
            
            keywords = inp.strip() if "comve" in config.dataset_path.lower() else inp.strip().replace(", ", " ").replace(" ", ", ")        
            output = out.strip() if isinstance(out, str) else out[0].strip() 
            instruction = instruction_comve if "comve" in config.dataset_path.lower() else instruction_commongen
            if "gemma" in config.model_name.lower():
                conversation = [
                    {"role": "system", "content": [{"type" : "text", "text": instruction}]},
                    {"role": "user", "content": [{"type": "text","text": keywords,}]},
                    {"role": "assistant", "content": [{"type": "text","text": output,}]},
                ]
            else:
                conversation = [
                {"role":"system", "content":instruction},
                {"role": "user", "content": keywords},
                {"role": "assistant", "content": output}
            ]
            texts.append(tokenizer.apply_chat_template(
                conversation, 
                tokenize=False, 
                add_generation_prompt=False))
        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched = True,num_proc=30)
    data_collator = DataCollatorForSeq2Seq(tokenizer = tokenizer)

    return dataset, data_collator

class UnslothSFTTrainer(BaseTrainer):

    def prepare_components(self) -> None:
        # ----------------- model & tokenizer ----------------------------
        self.model, self.tokenizer = load_model_and_tokenizer(self.cfg)
        #   ↳ handles chat_template + LoRA mounting internally

        # ----------------- dataset & collator --------------------------
        self.dataset, self.data_collator = process_data(self.cfg, self.tokenizer)
        
        
        # ----------------- training arguments --------------------------
        self._build_training_args()
       
        
    # ------------------------------------------------------------------
    def _build_training_args(self) -> None:
        ta = dict(self.cfg.training_args)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ta["output_dir"] = str(Path(ta["output_dir"]).with_suffix("") / timestamp)

        self.training_args = TrainingArguments(
            **ta,
            report_to=["wandb"],
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            remove_unused_columns = False if self.cfg.sft_method == "vendi" or self.cfg.sft_method == "var" else True,
        )
        self.cfg.training_args["output_dir"] = ta["output_dir"]
        
        self.log.info("Outputs → %s", self.training_args.output_dir)


    def _get_common_trainer_kwargs(self) -> Dict[str, Any]:
        return dict(
            model=self.model,
            tokenizer=self.tokenizer,
            train_dataset=self.dataset["train"],
            data_collator=self.data_collator,
            max_seq_length=self.cfg.max_length,
            packing=False,
            dataset_num_proc=2,
            args=self.training_args,
        )
    
    
    def train(self) -> None:
        common = self._get_common_trainer_kwargs()
        

        # ----------------- instruct model training -------------------
        self.log.info("Training *instruct* model (mask user prompts)…")
        trainer = SFTTrainer(**common, dataset_text_field="text")
        
        if "Qwen" in self.cfg.model_name:
            # Train on responses only mask the user part for qwen2.5
            trainer = train_on_responses_only(
                trainer,
                instruction_part="<|im_start|>user",
                response_part="<|im_start|>assistant",
            )
        elif "llama" in self.cfg.model_name.lower():
            # Train on responses only mask the user part for llama3.1
            trainer = train_on_responses_only(
                trainer,
                instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
                response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
        )
        elif "gemma" in self.cfg.model_name.lower():
            trainer = train_on_responses_only(
                trainer,
                instruction_part = "<start_of_turn>user\n",
                response_part = "<start_of_turn>model\n",
            )

        self.log.info("SFT starts – max_steps=%d", self.training_args.max_steps)
        self.log.info("dataset keys: %s", trainer.train_dataset[0].keys())

        trainer.train()

        out_dir = Path(self.training_args.output_dir)
        self.model.save_pretrained(out_dir)
        self.tokenizer.save_pretrained(out_dir)
        #self.plot_history(out_dir)

        del trainer
        gc.collect()
        # torch.cuda.empty_cache()
        self.log.info("✅ Training finished – GPU memory cleared.")
        
        if "test" in getattr(self.cfg, "mode", []):
            from eval_utils import Evaluator  # local import to avoid cycle
            self.log.info("🚀 Starting evaluation …")
            Evaluator(self.cfg, model=self.model, tokenizer=self.tokenizer).evaluate_by_temp()
            self.log.info("✅ Evaluation done.")
            del self.model
            del self.tokenizer
            gc.collect()
            torch.cuda.empty_cache()
    
    