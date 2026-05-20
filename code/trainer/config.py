import yaml
from dataclasses import dataclass

@dataclass
class TrainConfig:
    training_args: dict # TrainingArguments
    lora_args: dict
    model_name: str
    dataset_path: str
    max_length: int
    mode: list
    training_method: str
    sft_method: str
    debug: bool
    model_type: str
    
    @classmethod
    def from_yaml(cls, yaml_path: str):
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
            return cls(
            training_args=config['training_args'],
            lora_args=config['lora_args'],
            model_name=config['model_name'],
            dataset_path=config['dataset_path'],
            max_length=config['max_length'],
            mode = config['mode'],
            training_method = config['training_method'],
            sft_method = config['sft_method'],
            debug = config.get('debug', False),
            model_type = config.get('model_type', "instruct")
        )


def get_config(path: str) -> TrainConfig:
    with open(path, 'r') as f:
        return TrainConfig(**yaml.safe_load(f))