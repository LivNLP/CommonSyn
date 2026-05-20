import argparse
import json
import random
import time
from tqdm import tqdm
from pathlib import Path
from vllm_model import VLLMGenerator
from quality_filter import CoverageAnalyzer, GeminiJudge
from generate_solution import read_samples
from utils import save_to_file

def generate_valid_completions(
    generator: VLLMGenerator,
    coverage_checker: CoverageAnalyzer,
    quality_judge: GeminiJudge,
    concepts: list,
    method: str = "persona",
    max_retry_per_sentence: int = 5,
    max_retry_concept: int = 10,
    quality_threshold: int = 1,
    fewshot_samples: list = None,
    output_path: Path = None,
    n_sen : int = 4,
):
    """
    For each concept set, generate exactly n high-quality, keyword-valid sentences using retries.
    """
    final_data = []
    for sample in tqdm(concepts, desc="Generating concept sets"):
        keywords = sample["inputs"]
        completions = []
        retry_count = 0

        while len(completions) < n_sen and retry_count < max_retry_concept:
            needed = n_sen - len(completions)
            batch_samples = [sample] * needed
            if method in ["persona", "cot"]:
                batch_fewshots = [[] for _ in range(needed)]
            else:  # dynamic_fewshot
                batch_fewshots = [
                    random.sample(fewshot_samples, k=5) for _ in range(needed)
                ]
            gens = generator.batch_prompt_solution(batch_samples, batch_fewshots, method, n_sen=needed)

            for g in gens:
                sent = g["completion"]
                if not sent:
                    continue
                cover_ana = coverage_checker.analyze_sentence(keywords, sent)['coverage_score']
                if cover_ana < 1.0:
                    continue
                plaus_score = quality_judge.rate_sentence_plausibility([
                    {"inputs": keywords, "completion": sent}
                ])[0]["plausibility_scores"][0]
                if plaus_score >= quality_threshold:
                    completions.append(sent)
            

            retry_count += 1
            time.sleep(0.2)
            
        if len(completions) == n_sen:
            print(f"[INFO] Successfully generated {n_sen} completions for concept: {keywords}")
            temp = {"inputs": keywords, "completion": "\t".join(completions)}
            save_to_file([temp], Path(output_path), save_mode="a")
            
            final_data.append({
                "inputs": keywords,
                "completion": "\t".join(completions)
            })
        else:
            print(f"[WARN] Skipped concept: {keywords} after {retry_count} retries.")
    return final_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fewshot_filename", type=str, default="")
    parser.add_argument("--num_fewshot_samples", type=int, default=5)
    parser.add_argument("--input", type=str,  default="")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--method", type=str, default="cot", choices=["fewshot", "cot", "dynamic_fewshot"])
    args = parser.parse_args() 

    with open(args.input) as f:
        concept_sets = [json.loads(l) for l in f if l.strip()]

    
    fewshot_samples = read_samples(Path(args.fewshot_filename),args.method)
    generator = VLLMGenerator(base_hosts=[8000]) 
    coverage_checker = CoverageAnalyzer()
    quality_judge = GeminiJudge()

    results = generate_valid_completions(
        generator,
        coverage_checker,
        quality_judge,
        concepts=concept_sets,
        method=args.method,
        fewshot_samples=fewshot_samples,output_path=args.output,n_sen=8
    )

    with open(args.output + "_check.jsonl", "w") as fout:
        for item in results:
            fout.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    main()
