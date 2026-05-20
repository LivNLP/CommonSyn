import itertools
import numpy as np
import json
from openai import OpenAI
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import torch
import genai
import requests
from google.genai import types
from prompts import * # CommonGen templates


client = OpenAI()

SOURCE_FILE = "2seed_sent/source.jsonl"
VENDI_MODEL = "princeton-nlp/sup-simcse-roberta-large"
TOKENIZER = AutoTokenizer.from_pretrained(VENDI_MODEL)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODEL = AutoModel.from_pretrained(VENDI_MODEL).to(DEVICE)
MODEL.eval()

def min_max_norm(x):
    x = np.array(x)
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

def build_quality_prompt(concept_set: str, sentences: list) -> str:
    num = len(sentences)
    display_sentences = [sent if sent.strip() else "[EMPTY]" for sent in sentences]
    sentence_lines = [f"{i+1}. {sent}" for i, sent in enumerate(display_sentences)]
    sentence_list_str = "\n".join(sentence_lines)
    
    score_placeholders = ", ".join([f"s{i+1}" for i in range(num)])
    
    prompt = QUALITY_GPT_PROMPT_BASE.format(
        num_sentences=num,
        concept_set=concept_set,
        sentence_list=sentence_list_str,
        score_placeholders=score_placeholders
    )
    return prompt


def evaluate_quality_gpt_score_for_concept_set(sample):
    sentences = sample["completion"].split("\t")
    
    if len(sentences) < 12:
        sentences += [""] * (12 - len(sentences))
    elif len(sentences) > 12:
        sentences = sentences[:12]

    concept_set = sample["inputs"]
    prompt = build_quality_prompt(concept_set, sentences)
    
    # messages = [
    #     {"role": "system", "content": QUALITY_GPT_SYSTEM_PROMPT},
    #     {"role": "user", "content": prompt}
    # ]
    
    client = genai.Client()
    
    parsed_scores = [1] * len(sentences)
    retry_count = 0
    while retry_count < 3:
        try:
            response = client.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    config=types.GenerateContentConfig(
                        system_instruction=QUALITY_GPT_SYSTEM_PROMPT,
                        thinking_config=types.ThinkingConfig(thinking_budget=0)),
                    contents=prompt
                )
            
            clean_response = response.strip()
            if clean_response.startswith("```json"):
                clean_response = clean_response[7:-3].strip()
            elif clean_response.startswith("```"):
                clean_response = clean_response[3:-3].strip()
            
            response_json = json.loads(clean_response)
            scores = response_json.get("scores", [])
            
            if isinstance(scores, list) and len(scores) == len(sentences):
                display_sentences = [sent if sent.strip() else "[EMPTY]" for sent in sentences]
                final_scores = []
                valid = True
                for s, disp in zip(scores, display_sentences):
                    if disp == "[EMPTY]":
                        final_scores.append(1)
                    else:
                        if isinstance(s, int) and 1 <= s <= 10:
                            final_scores.append(s)
                        else:
                            valid = False
                            break
                if valid:
                    parsed_scores = final_scores
                    break
        except Exception as e:
            print(f"Error evaluating concept set '{concept_set}': {e}")
            print(f"Response: {response}")
        retry_count += 1

    sample["quality_scores"] = parsed_scores
    return sample

def compute_local_diversity_scores(sentences):
    with torch.no_grad():
        inputs = TOKENIZER(sentences, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        model_output = MODEL(**inputs)
        embeddings = model_output.pooler_output.cpu().numpy()
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    scores = []
    for i in range(len(sentences)):
        sims = embeddings[i] @ embeddings.T
        sims = np.delete(sims, i)
        diversity = 1 - sims.mean()
        scores.append(diversity)

    return scores


def select_top_per_concept_set(sample, alpha=1.0, beta=1.0, n=8):
    """
    Input sample structure same as GPT score function.
    Adds final selected 8 samples.
    """
    sentences = sample["completion"].split("\t")[:24]
    Q = sample["quality_scores"][:len(sentences)]
    
    D = min_max_norm(compute_local_diversity_scores(sentences))


    if sum(Q) > len(Q) * 1.0:  
        local_quality = [idx for idx in range(len(Q)) if Q[idx] >= 3.0]
    else:
        local_quality = list(range(len(Q)))
    
    S = [D[i] for i in local_quality]
    sorted_idx = np.argsort(S)[::-1][:n]
    sample["selected_sentences"] = [sentences[sorted_idx[i]] for i in range(len(sorted_idx))]
    sample["quality_scores"] = [Q[local_quality[i]] for i in sorted_idx] 
    
    return sample

def compute_global_embeddings(sent_list, batch_size=256):
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(sent_list), batch_size):
            batch = sent_list[i: i + batch_size]

            inputs = TOKENIZER(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=64  
            ).to(DEVICE)

            outputs = MODEL(**inputs)
            emb = outputs.pooler_output  # (B, 1024)
            emb = emb / emb.norm(dim=1, keepdim=True)

            all_embeddings.append(emb.cpu())

            del inputs, outputs, emb
            torch.cuda.empty_cache()

    return torch.cat(all_embeddings, dim=0).numpy()



def global_selection_QD(dataset_samples, target_size=80000, alpha=1.0, beta=1.0, batch_size=256):
    
    flat_sentences = []
    flat_quality = []
    pair_index = []  # (sample_id, sentence_id)

    for si, sample in enumerate(dataset_samples)
        sentences = sample["selected_sentences"]
        prompt = sample["prompt"]
        Q = min_max_norm(sample["quality_scores"][:len(sentences)])  

        for sj, sent in enumerate(sentences):
            temp_sample = prompt[sj] +" "+sent
            flat_sentences.append(temp_sample)
            flat_quality.append(Q[sj])
            pair_index.append((si, sj))


    N = len(flat_sentences)
    print(f"[info] global_selection_QD: total sentences = {N}")
    

    emb = compute_global_embeddings(flat_sentences, batch_size=batch_size)
    print("[info] embeddings computed:", emb.shape)
    
    global_center = emb.mean(axis=0)
    global_center = global_center / np.linalg.norm(global_center)

    cosine_sim = emb @ global_center  # (N,)
    D = 1.0 - cosine_sim  
    D = min_max_norm(D)
    
    Q = np.array(flat_quality)
    S = alpha * Q + beta * D
    top_idx = np.argsort(S)[::-1][:target_size]
    top_q_idx = np.argsort(Q)[::-1][:target_size]
    top_d_idx = np.argsort(D)[::-1][:target_size]
    
    
    selected = []
    selected_quality = []
    selected_diversity = []
    sample_counter = 0
    for idx in top_idx:
        si, sj = pair_index[idx]
        selected.append({
            "prompt": dataset_samples[si]["inputs"],
            "completion": flat_sentences[idx],
            "quality": float(Q[idx]),
            "diversity": float(D[idx]),
            "score": float(S[idx])
        })

    for idx in top_q_idx:
        si, sj = pair_index[idx]
        selected_quality.append({
            "prompt": dataset_samples[si]["inputs"],
            "completion": flat_sentences[idx],
            "quality": float(Q[idx]),
        })
    for idx in top_d_idx:
        si, sj = pair_index[idx]
        selected_diversity.append({
            "prompt": dataset_samples[si]["inputs"],
            "completion": flat_sentences[idx],
            "diversity": float(D[idx]),
        })
    
    return selected, selected_quality, selected_diversity
if __name__ == "__main__":
    # read source file
    samples = []
    with open(SOURCE_FILE, "r", encoding="utf-8") as f:
        for line in f:
            samples.append(json.loads(line.strip()))
    
    print(f"Loaded {len(samples)} samples from {SOURCE_FILE}")

    evaluated_samples = []
    for sample in tqdm(samples, total=len(samples), desc="Evaluating GPT quality scores"):
        evaluated_sample = evaluate_quality_gpt_score_for_concept_set(sample)
        evaluated_samples.append(evaluated_sample)
    
    print("Completed GPT quality score evaluation.")


    intermediate_file = SOURCE_FILE.replace("source.jsonl", "with_quality_scores.jsonl")
    with open(intermediate_file, "w", encoding="utf-8") as f:
        for sample in evaluated_samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Read evaluated samples with quality scores to {intermediate_file}")
    

    import random
    random.shuffle(evaluated_samples)
    evaluated_samples = evaluated_samples
    final_samples = []
    for sample in tqdm(evaluated_samples, total=len(evaluated_samples), desc="Selecting top-n per concept set"):
        final_sample = select_top_per_concept_set(sample, alpha=1.0, beta=1.0, n=18)
        final_samples.append(final_sample)
    print("Completed top-10 selection based on combined scores.")
    
    final_file = SOURCE_FILE.replace("source.jsonl", "local_dataset.jsonl")
    with open(final_file, "w", encoding="utf-8") as f:
        for sample in final_samples:
            f.write(json.dumps(sample) + "\n")

    print(f"Saved local selected top-4 samples to {final_file}")
    
    # global selection
    global_selected, _, _ = global_selection_QD(final_samples, target_size=40000, alpha=1.0, beta=1.0)
    global_file = "2seed_sent/2step.jsonl"
    with open(global_file, "w", encoding="utf-8") as f:
        for item in global_selected:
            f.write(json.dumps(item) + "\n")
    print(f"Saved global selected samples to {global_file}")