import itertools
import json
from pathlib import Path
from typing import Dict, List

import ipdb
import numpy as np
from torch.utils.data import DataLoader
from transformers import PreTrainedModel
import torch
from tqdm import tqdm
from trak.projectors import CudaProjector, BasicProjector, ProjectionType
from safetensors.torch import save_file

from trl import DataCollatorForCompletionOnlyLM


class GradientComputer:
    """ wrapper class for gradient storage """
    def __init__(self, model_name, model, tokenizer):
        # hyperparameters for projector
        self.block_size = 128
        self.projector_batch_size = 16
        self.proj_dim = 1024

        self.project_interval = 4 
        self.save_interval = 500 

        self.model_name = model_name
        self.model: PreTrainedModel = model
        self.model.config.use_cache = False
        self.model.eval()
        self.device = self.model.device
        self.dtype = self.model.dtype
        self.grad_dim = GradientComputer.get_gradient_vector_size(self.model)

        self.tokenizer = tokenizer

        self.collator = None

        trak_projector = GradientComputer.get_trak_projector(self.device)
        self.projector = trak_projector(
            grad_dim=self.grad_dim,
            proj_dim=self.proj_dim,
            seed=0,
            proj_type=ProjectionType.rademacher,
            device=self.device,
            dtype=torch.float16,  # self.dtype?
            block_size=self.block_size,
            max_batch_size=self.projector_batch_size,
        )

    @staticmethod
    def get_gradient_vector_size(model: PreTrainedModel):
        """ Make sure that only lora parameters require gradients in peft models. """
        num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])

        return num_params

    @staticmethod
    def get_trak_projector(device: torch.device):
        """ Get trak projectors (https://github.com/MadryLab/trak """
        try:
            num_sms = torch.cuda.get_device_properties(
                device.index).multi_processor_count

            import fast_jl
            fast_jl.project_rademacher_8(torch.zeros(
                8, 1_000, device=device), 512, 0, num_sms)
            projector = CudaProjector
        except RuntimeError:
            projector = BasicProjector
        return projector

    def obtain_gradient(self, batch) -> torch.Tensor:
        self.model.zero_grad()
        
        loss = self.model(**batch).loss
        loss.backward()

        vectorized_gradient = torch.cat(
            [p.grad.view(-1) for n, p in self.model.named_parameters() if p.grad is not None], dim=0
        )

        return vectorized_gradient

    def project_gradients(self, gradients: Dict[str, torch.Tensor]):
        sample_ids = list(gradients.keys())
        gradients = torch.stack([gradients[sample_id] for sample_id in sample_ids]).to(torch.float16)
        projected_gradients = self.projector.project(gradients, model_id=0) / np.sqrt(self.proj_dim)

        return {k: v for k, v in zip(sample_ids, projected_gradients)}

    @staticmethod
    def save_projected_gradients(projected_gradients: Dict[str, torch.Tensor], save_filename: Path):
        if len(projected_gradients) == 0:
            return

        assert not Path(save_filename).exists()

        save_file(projected_gradients, save_filename)

        # save sample_id of each projected gradients (as the same filename as gradient)
        sample_ids = list(projected_gradients.keys())
        data = [json.dumps({"id": sample_id}) for sample_id in sample_ids]

        with open(save_filename.with_suffix(".txt"), "w") as f:
            f.write("\n".join(data) + "\n")

    def prepare_model_input(self, prompt: str, completion: str):
        """
        Return models input consisting of `input_ids`, `attention_mask`, `labels`
        * NOTE: input prompt is masked in the labels (to calculate loss only for completion)
        """
        if "qwen" in self.model_name.lower():
            model_type = "qwen-instruct"
        elif "llama" in self.model_name.lower(): 
            model_type = "llama-instruct"
        else:
            raise NotImplementedError

        if self.collator is None:
            if model_type == "qwen-instruct":
                response_template = "<|im_start|>assistant"
            elif model_type == "llama-instruct": 
                response_template = "<|start_header_id|>assistant<|end_header_id|>\n\n"
            else:
                raise NotImplementedError

            self.collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=self.tokenizer)

        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": completion},
        ]


        if model_type == "qwen-instruct":
            encoding = self.tokenizer.apply_chat_template(messages, return_dict=True, return_tensors="pt") 
            encoding = {k: v.to(self.device) for k, v in encoding.items()}
        elif model_type == "llama-instruct": 

            encoding_dict = self.tokenizer.apply_chat_template(
                messages, 
                return_dict=True, 
                return_tensors="pt"
            )
            encoding_dict['labels'] = self.collator(encoding_dict['input_ids'])['labels']
            

            encoding = {k: v.to(self.device) for k, v in encoding_dict.items()}
        else:
            ipdb.set_trace()
            raise NotImplementedError

        return encoding

    def compute_project_store_gradients(self, samples: List[Dict], save_file_prefix: str,
                                        save_directory: Path | str, global_start_idx: int):
        """
        compute the gradient, project them, then store them into files in `save_directory`
        Input:
            samples: a list of samples formatted as: {
                "id": sample id,
                "prompt": input prompt,
                "completion": output completion,
            }
        """
        full_grads = {}
        projected_grads = {}
        for sample_idx, sample in tqdm(enumerate(samples), total=len(samples)):
            sample_id = sample["id"]


            encoding = self.prepare_model_input(sample["prompt"], sample["completion"])

            sample_full_grad = self.obtain_gradient(encoding)
            full_grads[sample_id] = sample_full_grad

            if (sample_idx + 1) % self.project_interval == 0:
                new_projected_grads = self.project_gradients(full_grads)
                projected_grads.update(new_projected_grads)

                full_grads = {}

            if (sample_idx + 1) % self.save_interval == 0:
                save_filename = Path(save_directory) / f"{save_file_prefix}.{global_start_idx + sample_idx + 1 - len(projected_grads)}.safetensors"
                self.save_projected_gradients(projected_grads, save_filename)
                projected_grads = {}

        if len(full_grads) > 0:
            new_projected_grads = self.project_gradients(full_grads)
            projected_grads.update(new_projected_grads)

        save_filename = Path(save_directory) / f"{save_file_prefix}.{global_start_idx + len(samples) - len(projected_grads)}.safetensors"
        self.save_projected_gradients(projected_grads, save_filename)
