import spacy
import json
import pandas as pd
import torch
import sys
import re
import time
from typing import List, Dict
from google import genai
from google.genai import types
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Set, List
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util
from openai import OpenAI

from prompts import quality_assessment_system_template, quality_assessment_batch_template


client = OpenAI()

def save_to_file(samples: List[Dict], file_path: Path, save_mode: str = "w"):
    with file_path.open(save_mode) as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")

def commongen_format_(concepts: str, labels: List[str]) -> str:
    concept_str = ", ".join(concepts.split())
    label_str = "\t".join(labels)
    return f"[[Example]]\\Keywords: {concept_str}\\nReferences: {label_str}"

def _safe_json_loads(s: str):
    s = s.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(s)
    except Exception:
        return None
    
def _extract_text_from_generate_content_response(resp_obj: Dict) -> str:
    if "response" not in resp_obj:
        return ""
    r = resp_obj["response"]
    if isinstance(r, dict) and "text" in r and isinstance(r["text"], str):
        return r["text"]

    try:
        cands = r.get("candidates", [])
        if not cands:
            return ""
        parts = cands[0].get("content", {}).get("parts", [])
        texts = []
        for p in parts:
            t = p.get("text")
            if t:
                texts.append(t)
        return "\n".join(texts)
    except Exception:
        return ""












def address_previous_round(path, save_path):
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            sample = json.loads(line)
            data.append(sample)

    merged_data = {}
    for sample in data:
        prompt = sample["prompt"]. replace(', ', ' ')
        if prompt not in merged_data:
            merged_data[prompt] = []
        merged_data[prompt].append(sample["completion"])

    with open(save_path, 'w', encoding='utf-8') as f:
        for prompt, completions in merged_data.items():
            out_sample = {
                "inputs": prompt,
                "labels": completions
            }
            f.write(json.dumps(out_sample, ensure_ascii=False) + "\n")

def read_samples(file_path: Path) -> List[str]:
    with file_path.open("r") as f:
        samples = [json.loads(line) for line in f]
    return [commongen_format_(s["inputs"], s["labels"]) for s in samples]

class CoverageAnalyzer:
    """
    A robust class to analyze the coverage of keywords in a sentence.
    It handles lemmatization and is case-insensitive.
    """
    _nlp = None

    def __init__(self, model_name: str = 'en_core_web_lg'):
        if CoverageAnalyzer._nlp is None:
            print(f"Loading spacy model '{model_name}'... (This may take a moment)")
            try:
                CoverageAnalyzer._nlp = spacy.load(model_name)
                print("Model loaded successfully.")
            except OSError:
                print(f"Error: spacy model '{model_name}' not found.")
                print(f"Please run: python -m spacy download {model_name}")
                raise

    def _prepare_keywords(self, keywords: str) -> Set[str]:
        keyword_list = keywords.replace(",", " ").lower().split()
        return set(kw.strip() for kw in keyword_list if kw.strip())

    def analyze_sentence(self, keywords: str, sentence: str) -> Dict:
        keyword_set = self._prepare_keywords(keywords)
        if not keyword_set:
            return {'coverage_score': 1.0, 'found_words': set(), 'missing_words': set(), 'found_count': 0, 'total_count': 0}

        doc = self._nlp(sentence.lower())
        lemmatized_tokens = {token.lemma_ for token in doc}

        found_words = keyword_set.intersection(lemmatized_tokens)
        missing_words = keyword_set.difference(found_words)

        total_count = len(keyword_set)
        found_count = len(found_words)
        coverage_score = found_count / total_count if total_count > 0 else 1.0

        return {
            'coverage_score': coverage_score,
            'found_words': found_words,
            'missing_words': missing_words,
            'found_count': found_count,
            'total_count': total_count
        }


def analyze_similarity(synthetic_file: Path, original_data_lookup: Dict[str, List[str]], model):
    """
    Analyzes a synthetic data file, comparing its solutions to the original labels.
    """
    if not synthetic_file.exists():
        print(f"Error: Synthetic file not found at '{synthetic_file}'")
        return

    line_avg_scores = []
    detailed_results = []

    with open(synthetic_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    print(f"\nAnalyzing {len(lines)} samples from '{synthetic_file}'...")
    for line in tqdm(lines, desc=f"Comparing {synthetic_file.name}"):
        try:
            synth_data = json.loads(line)
        except json.JSONDecodeError:
            continue

        seed_keywords = synth_data.get("seed")
        generated_sentences = synth_data.get("solution", "").split('\t')

        if not seed_keywords or not all(s.strip() for s in generated_sentences):
            continue
            
        # Normalize the seed to create a lookup key
        lookup_key = " ".join(sorted(seed_keywords))

        # Find the corresponding original labels
        original_labels = original_data_lookup.get(lookup_key)

        if not original_labels:
            continue
            

        generated_embeddings = model.encode(generated_sentences, convert_to_tensor=True)
        original_embeddings = model.encode(original_labels, convert_to_tensor=True)


        cosine_scores = util.cos_sim(generated_embeddings, original_embeddings)

        if cosine_scores.numel() > 0:
            avg_score_for_line = torch.mean(cosine_scores).item()
            line_avg_scores.append(avg_score_for_line)
            detailed_results.append({
                "keywords": synth_data.get("keywords"),
                "seed": " ".join(seed_keywords),
                "generated_solution": "\\n".join(generated_sentences),
                "original_labels": "\\n".join(original_labels),
                "avg_similarity": avg_score_for_line
            })

    # Calculate the final overall score
    if line_avg_scores:
        total_average_score = sum(line_avg_scores) / len(line_avg_scores)
        return total_average_score
    else:
        print("No valid samples with matching seeds were found to analyze.")
        return 0.0
    
class GeminiJudge:
    def __init__(self, model_name: str = 'gemini-2.5-flash-lite'):
        
        self.client = genai.Client()
        self.model = model_name
        print(f"Gemini Judge initialized with model: {model_name}")

    
    
    def rate_sentence_plausibility(self, batch_samples: List[Dict]) -> List[Dict]:
        results = []
        for sample in batch_samples:
            keywords = sample["inputs"]
            sentences = sample["completion"].split('\t')
            
            if len(sentences) == 4:
                prompt = quality_assessment_batch_template.replace(
                    "$#$keywords$#$", keywords
                ).replace(
                    "$#$sentence_1$#$", sentences[0]
                ).replace(
                    "$#$sentence_2$#$", sentences[1]
                ).replace(
                    "$#$sentence_3$#$", sentences[2]
                ).replace(
                    "$#$sentence_4$#$", sentences[3]
                )
            else: # single sentence
                prompt = quality_assessment_batch_template.replace(
                    "$#$keywords$#$", keywords
                ).replace(
                    "$#$sentence_1$#$", sentences[0]
                )


            scores = [1, 1, 1, 1]
            try:
                # Gemini API call
                response = self.client.models.generate_content(
                    model=self.model,
                    config=types.GenerateContentConfig(
                        system_instruction=quality_assessment_system_template,
                        thinking_config=types.ThinkingConfig(thinking_budget=0)),
                    contents=prompt
                )
                # Extract the numeric score from the response text
                # A simple regex is often sufficient
                response_text = response.text.strip().replace("```json", "").replace("```", "").strip()
                response_json = json.loads(response_text)
                parsed_scores = response_json.get("scores", [])
                if isinstance(parsed_scores, list) and len(parsed_scores) == 4:
                    scores = [int(s) for s in parsed_scores]

                    #print(f"Gemini API call succeeded for keywords '{keywords}...': {scores}")
            except (json.JSONDecodeError, Exception) as e:
                print(f"API call or JSON parsing failed for keywords '{keywords}...': {e}")
                print(f"Response text: {response.text}")

            sample["plausibility_scores"] = scores # Add the list of scores to the sample
            results.append(sample)

        return results












def build_openai_messages(sample: Dict):
    keywords = sample["keywords"]
    sentences = sample["solution"].split('\t')

    prompt = combined_quality_assessment_batch_template.replace(
        "$#$keywords$#$", keywords
    ).replace(
        "$#$sentence_1$#$", sentences[0]
    ).replace(
        "$#$sentence_2$#$", sentences[1]
    ).replace(
        "$#$sentence_3$#$", sentences[2]
    ).replace(
        "$#$sentence_4$#$", sentences[3]
    )
    return [
        {"role": "system", "content": combined_quality_assessment_system_template},
        {"role": "user", "content": prompt}
    ]

def write_openai_batch_jsonl(samples: List[Dict], jsonl_path: str):
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for s in samples:
            line = {
                "custom_id": str(s["id"]),         
                "method": "POST",
                "url": "/v1/chat/completions",            
                "body": {
                    "model": "gpt-4o",
                    "messages": build_openai_messages(s),
                    "response_format": { "type": "json_object" },
                    "temperature": 0.0,
                    # "max_output_tokens": 128,
                }
            }
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def run_openai_batch(input_jsonl: str):

    up = client.files.create(file=open(input_jsonl, "rb"), purpose="batch")

    batch = client.batches.create(
        input_file_id=up.id,
        endpoint="/v1/chat/completions",
        completion_window="24h"   
    )

    terminal = {"completed", "failed", "expired", "cancelled"}
    while True:
        b = client.batches.retrieve(batch.id)
        if b.status in terminal:
            break
        time.sleep(5)

    if b.status != "completed":
        err_id = getattr(b, "error_file_id", None) or (getattr(b, "error_file_ids", []) or [None])[0]
        raise RuntimeError(f"Batch not completed: {b.status}. error_file_id={err_id}")

    out_id = getattr(b, "output_file_id", None)
    if not out_id:
        ids = getattr(b, "output_file_ids", None)
        if ids and len(ids) > 0:
            out_id = ids[0]

    if not out_id:
        b = client.batches.retrieve(batch.id)
        out_id = getattr(b, "output_file_id", None) or (getattr(b, "output_file_ids", []) or [None])[0]

    if not out_id:
        raise ValueError("Batch completed but no output file id found on object. Check the job in dashboard or use batches.retrieve to inspect fields.")

    file_stream = client.files.content(out_id)
    text = file_stream.read().decode("utf-8")
    return text.splitlines()

def parse_and_filter(samples: List[Dict], result_lines: List[str]):
    _JSON_OBJ = re.compile(r'\{.*\}', re.S)
    
    id2sample = {str(s["id"]): s for s in samples}
    outputs_all = [] # For all sentences with the same prompt rated 4 or above
    outputs_single = [] # For sentences rated 4 or above
    for ln in result_lines:
        obj = json.loads(ln)
        cid = obj.get("custom_id")
        s = id2sample.get(str(cid))
        if not s: 
            continue

        body = obj.get("response", {}).get("body", {})
        content = None
        try:
            choices = body.get("choices", [])
            if choices:
                content = choices[0].get("message", {}).get("content")
        except Exception:
            content = None

        cleaned = content.replace("```json", "").replace("```", "").strip()
        try:
            data = json.loads(cleaned)
        except Exception:
            data = {}

        scores = data.get("scores", [])
        reasons = data.get("reasons", [])
        generated_sentences = s["solution"].split('\t')
        for l, r, g in zip(scores, reasons, generated_sentences):
            if l >= 4:
                outputs_single.append({"id": s["id"], "prompt": s["keywords"], "completion": g, "reason": r, "score": l})
        
        decision_score = min(scores)
        if decision_score >= 4:
            for l, r, g in zip(scores, reasons, generated_sentences):
                outputs_all.append({"id": s["id"], "prompt": s["keywords"], "completion": g, "reason": r, "score": l})

    return outputs_single, outputs_all
