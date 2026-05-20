import os
import json
import uuid
import random
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import List, Dict

import jsonlines
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from vllm_model import VLLMGenerator
from utils import save_to_file



def commongen_format_(concepts: str, labels: List[str]) -> str:
    concept_str = ", ".join(concepts.split())
    label_str = "\t".join(labels) if len(labels) > 1 else labels[0]
    return f"[[Example]]\\Keywords: {concept_str}\\nReferences: {label_str}"


def read_samples(file_path: Path, method:str) -> List[str]:
    with file_path.open("r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]
    results = []

    results = [commongen_format_(s["inputs"], s["labels"]) for s in samples]
    return results

def parse_args():
    parser = ArgumentParser()

    parser.add_argument("--model_name", type=str,  default="Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("--fewshot_filename", type=str, default="")
    parser.add_argument("--num_fewshot_samples", type=int, default=5)
    parser.add_argument("--input_filename", type=str, default="")
    parser.add_argument("--output_filename", type=str, default="")
    parser.add_argument("--ports", type=int, nargs="*", default=[8000, 8001, 8002], help="vLLM OpenAI port")
    parser.add_argument("--per_gpu_batch_size", type=int, default=128, help="per gpu batch size")


    parser.add_argument("--method", type=str, choices=["dynamic_fewshot", "fewshot", "persona", "cot"], default="cot", help="generation method, only fewshot can generate all sentences in one go")
    args = parser.parse_args()
    
    args.max_model_len, args.max_gen_len = 2048, 1024


    args.out_filename = Path(args.output_filename)
    os.makedirs(args.out_filename.parent, exist_ok=True)
    if os.path.exists(args.out_filename):
        print(f"Warning: Output file {args.out_filename} already exists. Appending results.")

    return args

if __name__ == "__main__":
    args = parse_args()
    
    models = [
        VLLMGenerator(
            model_name=args.model_name,
            max_model_len=args.max_model_len,
            max_gen_len=args.max_gen_len,
            base_hosts=[f"http://127.0.0.1:{port}"], 
            temperature=1.0, 
            top_p=0.95,
        )
        for port in args.ports
    ]
    num_gpus = len(models)
    
    with jsonlines.open(args.input_filename) as f:
        samples = list(f)

    fewshot_samples = read_samples(Path(args.fewshot_filename),args.method)


    meta_batch_size = args.per_gpu_batch_size * num_gpus
    
    for batch_start_idx in tqdm(range(0, len(samples), meta_batch_size), desc="Processing meta-batches"):
        meta_batch_samples = samples[batch_start_idx : batch_start_idx + meta_batch_size]
        if args.method in ["dynamic_fewshot", "persona", "cot"]:
            meta_batch_samples = meta_batch_samples * 4
        
        # Create N lists of samples, one for each GPU
        # Each list will contain 'per_gpu_batch_size' samples
        if args.method in ["dynamic_fewshot", "persona", "cot"]:
            adjusted_per_gpu_batch_size = args.per_gpu_batch_size * 4
        else:
            adjusted_per_gpu_batch_size = args.per_gpu_batch_size
        
        gpu_batches = [
            meta_batch_samples[i * adjusted_per_gpu_batch_size : (i + 1) * adjusted_per_gpu_batch_size]
            for i in range(num_gpus)
        ]
        # Filter out any empty batches at the very end of the dataset
        gpu_batches = [b for b in gpu_batches if b]
        if not gpu_batches:
            continue
        # --- Generate few-shot examples for each sample in each batch ---
        # This structure needs to align with gpu_batches
        
        gpu_fewshot_batches = []
        for batch in gpu_batches:
            # Each batch gets its own list of few-shot sets
            gpu_fewshot_batches.append(
                [random.sample(fewshot_samples, args.num_fewshot_samples) for _ in range(len(batch))]
            )
        
        all_gpu_results = []
        with ThreadPoolExecutor(max_workers=num_gpus) as executor:
            future_to_gpu = {
                executor.submit(models[i].batch_prompt_solution, gpu_batches[i], gpu_fewshot_batches[i], args.method): i
                for i in range(len(gpu_batches)) 
            }

            for future in as_completed(future_to_gpu):
                gpu_index = future_to_gpu[future]
                port = args.ports[gpu_index]
                try:
                    result = future.result()
                    all_gpu_results.extend(result)
                except Exception as exc:
                    print(f'Port {port}: {exc}')

        batch_result_list = defaultdict(list)
        for result in all_gpu_results:
            batch_result_list[result['inputs'].strip()].append(result)

        # -- Process and save valid solutions -- #
        out_batch_samples = []
        for inputs, result_list in batch_result_list.items():
            # Combine sentences into a single solution except fewshot
            if args.method in ["dynamic_fewshot", "persona", "cot"]:
                combined_sentences = []
                for result in result_list:
                    sentence = result["completion"].strip().replace("\n", " ")
                    combined_sentences.append(sentence)
                
                # Filter for solutions that correctly generated 4 non-empty sentences
                if len(combined_sentences) == 4 and all(combined_sentences):
                    solution_text = "\t".join(combined_sentences)
                    out_batch_samples.append({
                        "inputs": result_list[0]["inputs"],
                        "completion": solution_text,
                    })
            
            else:  # fewshot
                for result in result_list:
                    solution_text = result["completion"]
                    sentences = [s.strip() for s in solution_text.split("\t")]
                
                    # Filter for solutions that correctly generated 4 non-empty sentences
                    if len(sentences) == 4 and all(sentences):
                        out_batch_samples.append({
                            "inputs": result["inputs"],
                            "completion": solution_text,
                        })
        
        if out_batch_samples:
            save_to_file(out_batch_samples, args.out_filename, save_mode="a")