from datasets import Dataset, DatasetDict, load_from_disk
from collections import defaultdict
import json
import numpy as np
from tqdm import tqdm
from nlgeval import compute_metrics
from transformers import AutoModel, AutoTokenizer
import torch



def combine_dataset():
    file_1 = ""
    file_2 = ""
    file_3 = ""
    output_path = ""
    data_dict = defaultdict(list)

    with open(file_1, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line.strip())
            data_dict[line["inputs"]].extend(line["completion"].split("\t"))
    with open(file_2, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line.strip())
            data_dict[line["inputs"]].extend(line["completion"].split("\t"))
    with open(file_3, 'r', encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line.strip())
            data_dict[line["inputs"]].extend(line["completion"].split("\t"))
    
    final_data = []
    for key in data_dict:
        final_data.append({
            "inputs": key.replace(" ", ", "),
            "completion": "\t".join(list(set(data_dict[key]))),
        })
    
    with open(output_path, "w", encoding="utf-8") as f:
        for sample in final_data:
            f.write(json.dumps(sample) + "\n")


def create_dataset(path) -> DatasetDict:
    import random

    default_train = []
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = json.loads(line.strip())
            default_train.append({
                "input": line["prompt"].replace(", ", " "),
                "output": line["completion"],
                })

    gpt_train = Dataset.from_list(default_train)
            

    return DatasetDict({
        "train": gpt_train,
    })
    



    

if __name__ == "__main__":
    dataset = create_dataset(path="")
    dataset.save_to_disk("")

