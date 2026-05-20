import os
import re
import json
import spacy
import random
import datetime
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
from datasets import load_dataset, load_from_disk
from nlgeval import compute_metrics, compute_individual_metrics
from sft_trainer import load_model_and_tokenizer,process_data
from vendi_score import text_utils
import torch
from transformers import AutoModel, AutoTokenizer
from openai import OpenAI
local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)

TEMPLATE_PATH = "commongen.md" 



VENDI_MODEL = "princeton-nlp/sup-simcse-roberta-large"
TOKENIZER = AutoTokenizer.from_pretrained(VENDI_MODEL)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = AutoModel.from_pretrained(VENDI_MODEL).to(DEVICE)
MODEL.eval()


_client = OpenAI()


nlp = None 
def analyze_words(concepts, sentence):
    global nlp 
    
    if nlp is None:
        nlp = spacy.load('en_core_web_trf') 
    doc = nlp(sentence)

    # Lemmatization and POS matching
    found_words = [tok.lemma_ for tok in doc if tok.lemma_ in concepts]
    
    # take unique items 
    found_words = list(set(found_words))
    return found_words


def normalize_choice(choice):
    """
    Robustly extract an answer letter A/B/C/D/E from a model's output.
    """
    text = choice.strip()
    text = re.sub(r"\[.*?\]", "", text)  # remove bracketed text
    text = text.upper()

    # 1. direct match at beginning (e.g., "A", "A.", "(A)")
    match = re.match(r"^\s*[\(\[]?\s*([ABCDE])\s*[\)\].]?", text)
    if match:
        return match.group(1)

    # 2. search anywhere (least restrictive)
    match = re.search(r"\b([ABCDE])\b", text)
    if match:
        return match.group(1)

    # 3. fallback for sentences like "I choose A because..."
    match = re.search(r"CHOOSE\s+([ABCDE])", text)
    if match:
        return match.group(1)

    # 4. fallback for "ANSWER: B"
    match = re.search(r"ANSWER[:\s]+([ABCDE])", text)
    if match:
        return match.group(1)

    return None



def _chat_once(prompt, num=1):
    resp = _client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role":"user","content":prompt}],
        temperature=0,
        max_tokens=5,
    )
    return resp.choices[0].message.content.strip()

def _parse_choice(s: str):
    sl = s.strip().lower()
    if "tie" in sl:
        return "tie"
    if re.search(r"\bA\b", s):
        return "A"
    if re.search(r"\bB\b", s):
        return "B"
    if "a" in sl and "b" not in sl:
        return "A"
    if "b" in sl and "a" not in sl:
        return "B"
    return "tie"

def eval_self_bleu(sentences_groups):
    hyp_list, ref_list = [], []
    for sentences in sentences_groups:
        for i in range(len(sentences)):
            hyp_list.append(sentences[i]) 
            ref_list.append('\t'.join(sentences[:i]+sentences[i+1:]))
    
    self_metrics = compute_metrics(hyp_list=hyp_list, ref_list=ref_list)
    self_metrics = {f'self_{k}': v for k, v in self_metrics.items()}
    self_mean_bleu = np.mean([self_metrics[f'self_bleu_{i}'] for i in range(1, 5)])
    self_metrics['self_bleu'] = self_mean_bleu

    return self_metrics


def eval_cosine_similarity(sentences):
    all_similarities = []
    with torch.no_grad():
        embeddings = []
        inputs = TOKENIZER(sentences, padding=True, truncation=True, return_tensors="pt").to(DEVICE)
        model_output = MODEL(**inputs)
        embeddings = model_output.pooler_output.cpu().numpy()
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        cosine_sim_matrix = np.matmul(embeddings, embeddings.T)
        for i in range(len(sentences)):
            sims = np.delete(cosine_sim_matrix[i], i)
            all_similarities.extend(sims.tolist())
    return np.mean(all_similarities)

def avg_cosine_similarity(sentences_group):
    group_sims = []
    for sentences in sentences_group:
        sim = eval_cosine_similarity(sentences)
        group_sims.append(sim)
    return np.mean(group_sims)

def vendi_score(texts, model=MODEL, tokenizer=TOKENIZER):
    simcse = []
    for sentences in texts:
        if len(sentences) == 1:
            sentences.append(sentences[0])
        simcse_temp = text_utils.embedding_vendi_score(sentences, model, tokenizer, device=DEVICE)
        simcse.append(simcse_temp)
    simcse_vs = np.mean(simcse)
    return simcse_vs

class Evaluator:
    def __init__(self, config, model=None, tokenizer=None):
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        
        self.output_dir = self.config.training_args['output_dir']
        
        print(f"Output will be saved to: {self.output_dir}")
        if model is None or tokenizer is None:
            print("Could not find the model and tokenizer, loading from pretrained...")
            model, tokenizer = FastModel.from_pretrained(self.output_dir,
                                                        max_seq_length=self.config.max_length,
                                                        load_in_4bit = False,
                                                        device_map={'': local_rank})
            self.model, self.tokenizer = model, tokenizer
        

        if "llama" in self.config.model_name.lower():
            self.tokenizer = get_chat_template(self.tokenizer, chat_template="llama-3")
        elif "qwen" in self.config.model_name.lower():
            self.tokenizer = get_chat_template(self.tokenizer, chat_template="qwen2.5")
        elif "gemma" in self.config.model_name.lower():
            self.tokenizer = get_chat_template(self.tokenizer, chat_template="gemma-3")
        
        # # For commongen eval
        if "2seed" in self.config.dataset_path.lower():
            print("Using 2-seed Commongen eval set")
        self.config.dataset_path = "dataset/commongen_eval"
        
        self.dataset = load_from_disk(self.config.dataset_path)
        
        if getattr(self.config, "debug", False):
            print("✅ Debug mode ACTIVE")
            self.dataset["test"] = self.dataset["test"].select(range(20))
            print(f"Truncated test size: {len(self.dataset['test'])}")

        FastModel.for_inference(self.model)
    
    
    def evaluate_by_temp(self, batch_size: int = 1, n_samples: int = 4):
        
        def _clean_one(text: str) -> str:
            text = (text or "").strip()
            text = re.sub(r'^\s*assistant\s*:?\s*', '', text, flags=re.I)
            text = ' '.join(text.split())
            if not text:
                return ""
            parts = re.split(r'(?<=[.!?])\s+', text)
            first = (parts[0] if parts else "").strip()
            if not first:
                return ""
            if first[-1] not in ".!?":
                first += "."
            return first
        
        test_dataset = self.dataset["test"]
        predictions = []
        grouped_data = defaultdict(lambda: {"preds": [], "labels": []})
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        pred_file = os.path.join(self.output_dir, f"{self.config.dataset_path.split('/')[-1]}_predictions_{timestamp}.txt")
        metrics_file = os.path.join(self.output_dir, f"{self.config.dataset_path.split('/')[-1]}_metrics_{timestamp}.json")
        detail_file = os.path.join(self.output_dir, f"{self.config.dataset_path.split('/')[-1]}_gpt4o_pairwise_{timestamp}.json")
        
        instruction_commongen = "Given several keywords, generate one coherent sentence that contains all the required keywords using background commonsense knowledge: "
        instruction_comve = "Given an implausible or counterfactual statement, generate one short explanation (≤ 22 words) that why it is implausible or counterfactual using background commonsense knowledge: "
        instruction_anlg = "Given an initial observation and a later observation, generate a short hypothesis (≤ 22 words) that that bridges Observation 1 and Observation 2 using background commonsense knowledge. "
        instruction_roc = "Read the following 4-sentence story context and write a short and plausible ending (≤ 22 words) to this story using background commonsense knowledge: "
        # Choose instruction based on dataset
        if "anlg" in self.config.dataset_path.lower():
            instruction = instruction_anlg
            TEMPLATE_PATH = "anlg.md"
        elif "comve" in self.config.dataset_path.lower():
            instruction = instruction_comve
            TEMPLATE_PATH = "comve.md"
        elif "roc" in self.config.dataset_path.lower():
            instruction = instruction_roc
            TEMPLATE_PATH = "roc.md"
        else:
            instruction = instruction_commongen
            TEMPLATE_PATH = "commongen.md"
        
        
        with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
            EVAL_TEMPLATE = f.read()
    
        n_samples = 3 if "comve" in self.config.dataset_path.lower() else n_samples
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        def batcher(seq, bs):
            for i in range(0, len(seq), bs):
                yield [seq[j] for j in range(i, min(i+bs, len(seq)))]
    
        for batch in tqdm(batcher(test_dataset, batch_size), desc="Evaluating (batched)"):
            conversations = []
            raw_inputs_text = []
            for sample in batch:
                
                if "anlg" in self.config.dataset_path.lower():
                    # use observation_1 and observation_2
                    input_text = (
                        f"Observation 1: {sample['observation_1'].strip()}\n"
                        f"Observation 2: {sample['observation_2'].strip()}\n"
                        f"Hypothesis:" 
                    )
                else:
                    input_text = sample["input"].strip()
                    
                if "gemma" in self.config.model_name.lower():
                    print("Using gemma chat template")
                    conv = [{"role": "system", "content": [{"type" : "text", "text": instruction}]},
                            {"role": "user", "content": [{"type" : "text", "text": input_text}]},]
                     
                else: 
                    conv = [{"role": "system", "content": instruction},
                        {"role": "user", "content": input_text},]
                    
                conversations.append(conv)
                
                if not "gemma" in self.config.model_name.lower():
                    raw_input = self.tokenizer.apply_chat_template(
                        conv, tokenize=False, add_generation_prompt=True
                    )
                    
                    raw_inputs_text.append(raw_input)

            if "gemma" in self.config.model_name.lower():
                from torch.nn.utils.rnn import pad_sequence
                encodings = [ self.tokenizer.apply_chat_template(conv, add_generation_prompt=True,tokenize=True, return_tensors="pt",return_dict=True,) for conv in conversations]
                input_ids_list = [e["input_ids"][0] for e in encodings]
                attn_mask_list = [e["attention_mask"][0] for e in encodings]
                
                pad_id = ( self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id)
                
                input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id).to(device)
                attn_mask = pad_sequence(attn_mask_list, batch_first=True, padding_value=0).to(device)
                enc = {"input_ids": input_ids, "attention_mask": attn_mask}
            else:
                enc = self.tokenizer(raw_inputs_text, padding=True, return_tensors="pt").to(device)
                input_ids = enc["input_ids"]
                attn_mask = enc["attention_mask"]
            
            input_lens = attn_mask.sum(dim=1).tolist()  # shape: [B]
            B = input_ids.size(0)
            
            gen_out = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.9,                     
                    num_return_sequences=n_samples, 
                    max_new_tokens=50,
                    use_cache=True,
                )
            
            expanded_lens = []
            for L in input_lens:
                expanded_lens.extend([L] * n_samples)
            
            all_decoded = []
            for row, L in zip(gen_out, expanded_lens):
                new_token_ids = row[L:]  
                text = self.tokenizer.decode(new_token_ids, skip_special_tokens=True)
                all_decoded.append(_clean_one(text))
                
            per_sample_outputs = [
                all_decoded[i * n_samples : (i + 1) * n_samples] for i in range(B)
            ]

            for sample, k_outputs in zip(batch, per_sample_outputs):
                if isinstance(sample["output"], str):
                    sample["output"] = sample["output"].split('\t')
                
                if "anlg" in self.config.dataset_path.lower():
                    sample_input = "Observation 1 (O1): " + sample["observation_1"] \
                        + "\nObservation 2 (O2): " + sample["observation_2"]
                else:
                    sample_input = sample["input"]
                
                grouped_data[sample_input]["labels"].extend(sample["output"])
                grouped_data[sample_input]["preds"].extend(k_outputs)

                predictions.append({
                    "input": sample_input,
                    "predictions": k_outputs,
                })

        with open(pred_file, "w", encoding="utf-8") as f:
            for item in predictions:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Predictions saved to: {pred_file}")
        

        
        gpt_tasks = []
        cover_radios = []
        
        for inp, group in tqdm(grouped_data.items(), desc="Preparing GPT evaluation tasks", total=len(grouped_data)):
            concept_list_str = inp.strip()
            ref = group["labels"][0]
            preds = [random.choice(group["preds"])]
            
            concept_tokens = concept_list_str.split()
            denom = max(1, len(concept_tokens))
            temp_coverage = np.mean([
                len(analyze_words(concept_tokens, p)) / denom for p in group["preds"]
            ])
            cover_radios.append(temp_coverage)
            
            ref = ref.strip()
            for pred in preds:
                if random.random() < 0.5:
                    A, B = pred, ref
                    assignment = {"A": "generator", "B": "human"}
                else:
                    A, B = ref, pred
                    assignment = {"A": "human", "B": "generator"}

                prompt_filled = (
                    EVAL_TEMPLATE
                    .replace('{$input}', concept_list_str)
                    .replace('{$candidate_A}', A)
                    .replace('{$candidate_B}', B)
                )
                    
                gpt_tasks.append({
                    "input": inp,
                    "prompt": prompt_filled,
                    "assignment": assignment,
                    "generator_pick": pred,
                    "human_ref": ref,
                })
                    
        print(f"[info] Prepared {len(gpt_tasks)} GPT evaluation tasks "
      f"(avg {len(gpt_tasks)/len(grouped_data):.2f} per concept set)") 
                    
        
        quality_results = []
        win = lose = tie = 0
        for t in tqdm(gpt_tasks, desc="GPT-4o judging (multi-ref single-judge)"):
            out = _chat_once(t["prompt"])
            choice = _parse_choice(out)
            if choice in ("A", "B"):
                winner_side = t["assignment"][choice]
            else:
                winner_side = "tie"

            quality_results.append({
                "input": t["input"],
                "choice": choice,
                "winner_side": winner_side,
                "generator_pick": t["generator_pick"],
                "human_ref": t["human_ref"],
            })
            
        by_concept = defaultdict(list)
        for r in quality_results:
            by_concept[r["input"]].append(r["winner_side"])
        
        concept_win_tie_rates = []
        for inp, res in by_concept.items():
            wins = res.count("generator")
            ties = res.count("tie")
            total = len(res)
            concept_win_tie_rates.append((wins + ties) / total)
        
        win = sum(1 for r in quality_results if r["winner_side"] == "generator")
        lose = sum(1 for r in quality_results if r["winner_side"] == "human")
        tie = sum(1 for r in quality_results if r["winner_side"] == "tie")

        total = len(quality_results)
        win_tie_rate = round(np.mean(concept_win_tie_rates), 4)
        
        avg_coverage = round(np.mean(cover_radios), 4)
        print(f"Average coverage: {avg_coverage}")
        
        quality_metrics = {
            "gpt4o_win": win,
            "gpt4o_lose": lose,
            "gpt4o_tie": tie,
            "gpt4o_win_tie_rate": win_tie_rate,
            "avg_coverage": avg_coverage,
        }
        print(f"[done] GPT-4o judging completed. "
            f"Win: {win}, Lose: {lose}, Tie: {tie}")
        print(f"Win-tie rate (concept-avg): {win_tie_rate}, Avg coverage: {avg_coverage:.4f}")

        sentences_groups = [group["preds"] for group in grouped_data.values()]

        
        div_bleu = eval_self_bleu(sentences_groups)
        vs = vendi_score(sentences_groups)
        print(div_bleu)
        print(f"Vendi Score: {vs}")
        
        cosine_sim = avg_cosine_similarity(sentences_groups)
        div_metrics = {**div_bleu, "vendi_simcse": float(vs), "cosine_sim": float(cosine_sim)}
        metrics_all = {**quality_metrics, **div_metrics}
        print("== GPT-4o Quality Metrics ==")
        print(json.dumps(quality_metrics, indent=2))
        print("== Diversity Metrics ==")
        print(json.dumps(div_metrics, indent=2))
        
        with open(detail_file, "w", encoding="utf-8") as f:
            json.dump(quality_results, f, indent=2, ensure_ascii=False)
        print(f"Saved GPT-4o judgments to {detail_file}")
        with open(metrics_file, "w", encoding="utf-8") as f:
            json.dump(metrics_all, f, indent=2, ensure_ascii=False)
        print(f"Metrics saved to {metrics_file}")
        
    
    def evaluate_csqa(self, task_name, batch_size=1):
        
        if task_name == "csqa":
            ds = load_dataset("tau/commonsense_qa", split="validation")
        elif task_name == "csqa2":
            ds = load_dataset("tasksource/commonsense_qa_2.0", split="validation")
        elif task_name == "piqa":
            ds = load_dataset("baber/piqa", split="validation")
        else:
            ds = load_dataset("allenai/winogrande", 'winogrande_debiased' ,split="validation")
        
        if getattr(self.config, "debug", False):
            print("✅ Debug mode ACTIVE")
            ds = ds.select(range(20))
            print(f"Truncated test size: {len(ds)}")
        
        def build_prompt(sample):
            system_prompt = "You are a helpful assistant. Answer the question correctly."
            if task_name == "csqa":
                labels = sample["choices"]["label"]
                texts = sample["choices"]["text"]
                choice_str = "\n".join(f"{l}. {t}" for l, t in zip(labels, texts))
                user = (
                    f"Question: {sample['question']}\n"
                    f"Choices:\n{choice_str}\n"
                    "Answer with exactly one letter: A, B, C, D, or E."
                )
                return [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": user}]
            elif task_name == "csqa2": # Answer is yes/no
                user = (
                f"Question: {sample['question']}\n"
                "Is the answer yes or no?\n"
                "Answer with exactly: yes or no."
                )
                return [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": user}]
            elif task_name == "piqa":
                user = (
                    f"Goal: {sample['goal']}\n"
                    f"A: {sample['sol1']}\n"
                    f"B: {sample['sol2']}\n"
                    "Which is more plausible? Answer with A or B."
                )
                return [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": user}]

            else:
                user = (
                    f"Goal: {sample['sentence']}\n"
                    f"A: {sample['option1']}\n"
                    f"B: {sample['option2']}\n"
                    "Which option is more correct to fill in? Answer with A or B."
                )
                return [{"role": "system", "content": system_prompt},
                        {"role": "user", "content": user}]

        #Golden labels
        def get_label(sample):
            if task_name == "csqa":
                return sample["answerKey"]

            if task_name == "csqa2":
                # "yes" / "no"
                return sample["answer"].upper()

            if task_name == "piqa":
                return "A" if sample["label"] == 0 else "B"
            
            if task_name == "winogrande":
                return "A" if sample["answer"] == 1 else "B"
        
        def clean_pred(text):
            t = text.strip()
            # CSQA → A/B/C/D/E
            if task_name == "csqa":
                cleaned_c = normalize_choice(t)
                if cleaned_c is not None:
                    return cleaned_c
                return "A"

            # CSQA2 → yes/no
            if task_name == "csqa2":
                if "YES" in t:
                    return "YES"
                if "no" in t:
                    return "NO"
                return "YES"

            # PIQA → A or B
            if task_name == "piqa":
                cleaned_c = _parse_choice(t)
                return cleaned_c if cleaned_c != "tie" else "A"
            
            # Winogrande → A or B
            if task_name == "winogrande":
                cleaned_c = _parse_choice(t)
                return cleaned_c if cleaned_c != "tie" else "A"
    
                
        # ====== Begin Evaluation ======
        preds = []
        gold = []
        
        device = "cuda" if torch.cuda.is_available() else "cpu"

        def batcher(seq, bs):
            for i in range(0, len(seq), bs):
                yield [seq[j] for j in range(i, min(i+bs, len(seq)))]
        
        for batch in tqdm(batcher(ds, batch_size), desc=f"Evaluating {task_name}"):
            conversations = []
            raw_inputs = []

            for sample in batch:
                conv = build_prompt(sample)
                conversations.append(conv)

                raw = self.tokenizer.apply_chat_template(
                    conv, tokenize=False, add_generation_prompt=True
                )
                raw_inputs.append(raw)

            enc = self.tokenizer(raw_inputs, padding=True, return_tensors="pt").to(device)
            input_ids = enc["input_ids"]
            attn_mask = enc["attention_mask"]
            
            input_lens = attn_mask.sum(dim=1).tolist()
            outputs = self.model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
                max_new_tokens=32,
                do_sample=False,
            )
            for row, L, sample in zip(outputs, input_lens, batch):
                new_tokens = row[L:]
                text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                pred = clean_pred(text)
                # print(f"Model raw output: {text} --> Cleaned prediction: {pred}, THE GOLD: {get_label(sample)}")
                preds.append(pred)
                gold.append(get_label(sample))
        # Compute accuracy
        correct = sum(p == g for p, g in zip(preds, gold))
        acc = correct / len(gold)
        print(f"[RESULT] {task_name} Accuracy = {acc:.4f}")
        return acc
    