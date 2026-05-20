import spacy
from typing import Dict, Set
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from utils import CoverageAnalyzer
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

import asyncio
from tqdm.asyncio import tqdm
import numpy as np
from utils import *


INPUT_FILE = Path("")
CS_INPUT_FILE = Path("")
OUTPUT_FILE = Path("")
SINGLE_OUTPUT_FILE = Path("")
AGGREGATION_METHOD = 'min' 
PLAUSIBILITY_THRESHOLD = 4


def calculate_average_coverage():
    """
    Calculates the overall average keyword coverage for a JSONL file.
    """
    if not INPUT_FILE.exists():
        print(f"Error: Input file not found at '{INPUT_FILE}'")
        return

    analyzer = CoverageAnalyzer()

    line_average_scores = []
    detailed_results = []

    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"Analyzing {len(lines)} samples from '{INPUT_FILE}'...")
    for line in tqdm(lines, desc="Processing samples"):
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            print(f"Warning: Skipping malformed JSON line: {line.strip()}")
            continue

        keywords = data.get("inputs")
        solution = data.get("completion")

        if not keywords or not solution:
            continue

        sentences = solution.split('\t')
        sentence_scores = []
        
        for sentence in sentences:
            if sentence.strip():
                analysis = analyzer.analyze_sentence(keywords, sentence)
                sentence_scores.append(analysis['coverage_score'])

        if sentence_scores:
            avg_score_for_line = sum(sentence_scores) / len(sentence_scores)
            line_average_scores.append(avg_score_for_line)

            if avg_score_for_line == 1:
                detailed_results.append({
                    "inputs": keywords,
                    "completion": solution
                })

    if line_average_scores:
        total_average_score = sum(line_average_scores) / len(line_average_scores)
        print("\n" + "="*50)
        print("Analysis Complete!")
        print(f"Overall Average Keyword Coverage Score: {total_average_score:.4%}")
        print("="*50)

        print(f"Saving {len(detailed_results)} high-coverage samples to '{CS_INPUT_FILE}'...")
        with open(CS_INPUT_FILE, "w", encoding="utf-8") as f:
            for item in detailed_results:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")


def calculate_commonsense_score():
    print(f"Loading candidate samples from {CS_INPUT_FILE}...")
    initial_samples = []
    with open(CS_INPUT_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            initial_samples.append(json.loads(line))
    
    print(f"Found {len(initial_samples)} samples to evaluate.")

    judge = GeminiJudge()
    
    all_rated_samples = judge.rate_sentence_plausibility(initial_samples)


    final_good_samples = []
    separate_good_samples = [] 
    for sample in all_rated_samples:
        scores = sample.get("plausibility_scores", [])
        
        if not scores or len(scores) != 4:
            continue

        decision_score = 0
        
        if AGGREGATION_METHOD == 'average':
            decision_score = np.mean(scores)
        elif AGGREGATION_METHOD == 'min':
            decision_score = min(scores)
        
        if decision_score >= PLAUSIBILITY_THRESHOLD:
            sample.pop("plausibility_scores", None) 
            final_good_samples.append(sample)
            
        for idx, score in enumerate(scores):
            if score >= PLAUSIBILITY_THRESHOLD:
                separate_good_samples.append({
                    "prompt": sample["inputs"],
                    "completion": sample["completion"].split("\t")[idx]
                })

    print(f"\nFound {len(final_good_samples)} high-quality samples where the group of 4 sentences passed the Gemini filter.")
    if final_good_samples:
        save_to_file(final_good_samples, OUTPUT_FILE, save_mode="w")
        print(f"Filtered data saved to {OUTPUT_FILE}")

    print(f"\nFound {len(separate_good_samples)} high-quality samples where individual sentences passed the Gemini filter.")
    if separate_good_samples:
        save_to_file(separate_good_samples, SINGLE_OUTPUT_FILE, save_mode="w")
        print(f"Filtered data saved to {SINGLE_OUTPUT_FILE}")




def calculate_logical_label_batch_openai():
    print(f"Loading candidate samples from {CS_INPUT_FILE}...")
    samples = []
    custom_id = 0 # Custom ID for tracking
    with open(CS_INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            temp = json.loads(line)
            temp["id"] = "CG_R2_" + str(custom_id)
            custom_id += 1
            samples.append(temp)
    print(f"Found {len(samples)} samples to evaluate (OpenAI Batch).")

    batch_jsonl = "requests_openai.jsonl"
    write_openai_batch_jsonl(samples, batch_jsonl)

    result_lines = run_openai_batch(batch_jsonl)


    outputs_single, outputs_all = parse_and_filter(samples, result_lines)

    print(f"\nFound {len(outputs_single)} high-quality samples where the sentence passed the filter.")
    print(f"\nFound {len(outputs_all)} high-quality samples where the group of 4 sentences passed the filter.")
    if outputs_single:
        save_to_file(outputs_single, SINGLE_OUTPUT_FILE, save_mode="w")
        print(f"Filtered data saved to {SINGLE_OUTPUT_FILE}")

    if outputs_all:
        save_to_file(outputs_all, OUTPUT_FILE, save_mode="w")
        print(f"All high-quality sentences saved to {OUTPUT_FILE}")
    

if __name__ == '__main__':

    calculate_average_coverage()
    #calculate_commonsense_score()