import torch

from pathlib import Path
from typing import List, Tuple

from safetensors.torch import load_file


class GradientManager:
    @staticmethod
    def load_all_gradients(gradient_files: List[Path] | Path, device=0) -> Tuple[List, torch.Tensor]:
        if isinstance(gradient_files, Path):
            # gradient_files is the parent folder of safetensor files
            gradient_files = list(gradient_files.glob("*.safetensors"))

        all_gradient_dict = {}
        for gradient_file in gradient_files:
            assert gradient_file.suffix == ".safetensors", \
                "`gradient_files` should only include `.safetensor` files."
            all_gradient_dict.update(load_file(gradient_file, device=device))

        sample_ids, sample_gradients = [], []
        for sample_id, sample_gradient in all_gradient_dict.items():
            sample_ids.append(sample_id)
            sample_gradients.append(sample_gradient)

        sample_gradients = torch.stack(sample_gradients, dim=0)

        return sample_ids, sample_gradients

    @staticmethod
    def load_gradients_for_sample_ids(gradient_files: List[Path] | Path, sample_ids: List[str], device=0) -> Tuple[List, torch.Tensor]:
        if isinstance(gradient_files, Path):
            gradient_files = list(gradient_files.glob("*.safetensors"))

        set_sample_ids = set(sample_ids)
        all_gradient_dict = {}
        for gradient_file in gradient_files:
            assert gradient_file.suffix == ".safetensors", \
                "`gradient_files` should only include `.safetensor` files."
            loaded_gradients = load_file(gradient_file, device=device)
            all_gradient_dict.update({s_id: g for s_id, g in loaded_gradients.items() if s_id in set_sample_ids})

        sample_gradients = [all_gradient_dict[sample_id] for sample_id in sample_ids]

        sample_gradients = torch.stack(sample_gradients, dim=0)

        return sample_ids, sample_gradients