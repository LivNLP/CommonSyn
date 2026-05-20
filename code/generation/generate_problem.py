import json
import os
import random
import uuid
import sys
import numpy as np
from argparse import Namespace, ArgumentParser
from pathlib import Path
from typing import List, Dict
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from vllm_model import VLLMGenerator
from utils import *




def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str,  default="Qwen/Qwen2.5-72B-Instruct")
    parser.add_argument("--ports", type=int, nargs="*", default=[8000, 8001, 8002], help="vLLM OpenAI port")
    parser.add_argument("--input_filename", type=str,  default="")
    parser.add_argument("--output_filename", type=str, default="")

    parser.add_argument("--target_size", type=int, default=20000)
    parser.add_argument("--num_fewshot_samples", type=int, default=5)
    parser.add_argument("--num_new_problems", type=int, default=2)  
    parser.add_argument("--max_tokens", type=int, default=50)
    parser.add_argument("--per_gpu_batch_size", type=int, default=1)
    parser.add_argument("--strategy", type=int, default=128, help="") 
    args = parser.parse_args()
    
    args.input_filename = Path(args.input_filename)
    args.out_filename = Path(args.output_filename)
    os.makedirs(args.out_filename.parent, exist_ok=True)
    if os.path.exists(args.out_filename):
        # Allow overwriting for simplicity in this example
        print(f"Warning: Output file {args.out_filename} already exists. It will be overwritten.")
        os.remove(args.out_filename)

    return args

if __name__ == "__main__":
    args = parse_args()

    models = [
        VLLMGenerator(
            model_name=args.model_name,
            max_model_len=2048,
            max_gen_len=args.max_tokens,
            base_hosts=[f"http://127.0.0.1:{port}"], 
            temperature=0.95,
            top_p=0.95,
        )
        for port in args.ports
    ]
    num_gpus = len(models)

    with open(args.input_filename, "r") as f:
        original_samples = [json.loads(line) for line in f]

    num_generated = 0
    seen = set()
    used = set()
    meta_batch_size = args.per_gpu_batch_size * num_gpus
    
    samples = read_samples(args.input_filename)

    num_generated = 0
    seen = set()
    meta_batch_size = args.per_gpu_batch_size * num_gpus
    
    with tqdm(total=args.target_size, initial=num_generated) as pbar:
        while num_generated < args.target_size:
            
            meta_batch_fewshot_samples = [
                [samples[idx] for idx in np.random.choice(range(len(samples)), size=args.num_fewshot_samples, replace=False)]
                for _ in range(meta_batch_size)
            ]
            
            gpu_batches = [
                meta_batch_fewshot_samples[i * args.per_gpu_batch_size : (i + 1) * args.per_gpu_batch_size]
                for i in range(num_gpus)
            ]
            
            gpu_batches = [b for b in gpu_batches if b]
            if not gpu_batches:
                continue

            all_gpu_results = []
            with ThreadPoolExecutor(max_workers=num_gpus) as executor:
                future_to_gpu = {
                    executor.submit(models[i].batch_prompt_problem, gpu_batches[i], args.num_new_problems): i
                    for i in range(len(gpu_batches))
                }
                
                for future in as_completed(future_to_gpu):
                    gpu_index = future_to_gpu[future]
                    port = args.ports[gpu_index]
                    try:
                        result = future.result()
                        all_gpu_results.extend(result)
                    except Exception as exc:
                        print(f' {port} {exc}')
                        
            
            batch_out_samples = all_gpu_results

            unique_out_samples = []
            for out_samples in batch_out_samples:
                for out_sample in out_samples:
                    prompt = ", ".join(sorted(out_sample["inputs"].split(", ")))

                    unique_out_samples.append(out_sample)
            batch_out_samples = unique_out_samples

            for out_sample in batch_out_samples:
                out_sample["prompt_id"] = f"gen.{args.input_filename.stem}.{uuid.uuid4().hex}"

            save_to_file(batch_out_samples, args.out_filename, save_mode="a")
            num_generated += len(batch_out_samples)

            pbar.n = num_generated
            pbar.refresh()