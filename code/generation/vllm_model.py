import itertools
import random
import re
import sys
from typing import Dict, List, Iterable, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

import requests
from prompts import * # CommonGen templates

AGE = ["teenager", "young adult", "middle-aged", "elderly"]
GENDER = ["male", "female"]
LOCATION = ["China", "US", "Europe", "Africa", "Southeast Asia", "South America"]
RACE = ["Asian", "Black", "White", "Hispanic", "Middle Eastern"]
OCCUPATION = ["nurse", "soldier", "farmer", "software engineer", "teacher", "driver","chef", "artist", "scientist", "waiter","police", "mechanic", "musician", "writer", "athlete"]
client = OpenAI()


def generate_persona():
    """Generate one persona string from sampled attributes."""
    age = random.choice(AGE)
    gender = random.choice(GENDER)
    location = random.choice(LOCATION)
    occupation = random.choice(OCCUPATION)
    # race = random.choice(RACE)

    prompt = (
        f"You are a {gender} {occupation} living in {location}, "
    )
    return prompt


class VLLMGenerator:
    """
    vLLM via OpenAI-compatible HTTP servers (round-robin across replicas).
    Adapted for Generative Commonsense Reasoning (CommonGen-style) tasks.
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-72B-Instruct",
        max_model_len: int = 4096,
        max_gen_len: int = 256,
        base_hosts: Optional[List[str]] = None,
        temperature: float = 1.0,
        top_p: float = 0.95,
        timeout: int = 120,
    ):
        self.model_name = model_name
        self.max_model_len = max_model_len
        self.max_gen_len = max_gen_len

        if base_hosts is None:
            base_hosts = [f"http://127.0.0.1:{p}" for p in (8000, 8001, 8002)]
        print(f"[INFO] VLLMGenerator using hosts: {base_hosts}")
        self.base_hosts = base_hosts
        self._rr = itertools.cycle(self.base_hosts)

        self.temperature = temperature
        self.top_p = top_p
        self.timeout = timeout
        self.seed = random.randint(0, 99999)

    # --------------------------- HTTP helpers --------------------------- #

    def _next_base(self) -> str:
        return next(self._rr)

    def _chat_completion(self, messages: List[Dict], **overrides) -> str:
        base = self._next_base()
        url = base.rstrip("/") + "/v1/chat/completions"
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": overrides.get("temperature", self.temperature),
            "top_p": overrides.get("top_p", self.top_p),
            "max_tokens": overrides.get("max_tokens", self.max_gen_len),
            "stream": False,
        }
        r = requests.post(url, json=payload, timeout=self.timeout)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"]

    def _parmap(self, fn, items: Iterable, max_workers: int = 12):
        futures = []
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            for it in items:
                futures.append(ex.submit(fn, it))
            for fut in as_completed(futures):
                results.append(fut.result())
        return results

    # --------------------------- Problem gen --------------------------- #

    def batch_prompt_problem(self, batch_fewshot_samples: List[List[str]], num_new_problems: int) -> List[List[Dict]]:
        """
        Input: batch of few-shot lists; each inner list contains few-shot concept sets.
        Output: a 2D list aligned with input batch, each containing newly created concept sets.
        """
        prompt_list = [
            self.prepare_problem_prompt(fewshots, num_new_problems)
            for fewshots in batch_fewshot_samples
        ]

        def _do(prompt: str) -> List[Dict]:
            messages = [{"role": "user", "content": prompt}]
            text = self._chat_completion(messages, temperature=self.temperature, max_tokens=self.max_gen_len)
            problems = []
            for seg in self.parse_problem_from_generation(text):
                problems.append(seg)
            return problems

        outputs = self._parmap(_do, prompt_list)
        return outputs

    def batch_prompt_problem_completion(self, batch_tasks: List[tuple]) -> List[List[Dict]]:
        """
        Input: A batch of tasks, where each task is a tuple: (task_type, keywords, number)
        """
        
        def _do(task: tuple) -> List[Dict]:
            subtask, seed_or_keywords, num_to_add = task
            prompt = ""
            sampled_seed_for_parsing = [] 


            if subtask in {"refresh", "upsample"}:
                prompt = problem_completion_instruction_template.replace(
                    "$#$seed_keywords$#$", ", ".join(seed_or_keywords)
                ).replace(
                    "$#$num_to_add$#$", str(num_to_add)
                )
                sampled_seed_for_parsing = seed_or_keywords 
                
                

            elif subtask == "extend":
                if len(seed_or_keywords)==0:
                    prompt = problem_instruction_template.replace(
                        "$#$num_to_add$#$", str(num_to_add)
                    )
                else:
                    sampled_seed = random.sample(seed_or_keywords, 2)
                    prompt = problem_2seed_instruction_template.replace(
                        "$#$seed_keywords$#$", ", ".join(sampled_seed)
                    ).replace(
                        "$#$num_to_add$#$", str(num_to_add)
                    )
                    sampled_seed_for_parsing = sampled_seed

            if not prompt:
                return []
            
            messages = [{"role": "user", "content": prompt}]
            max_tokens = num_to_add * 10
            
            added_keywords_str = self._chat_completion(messages, temperature=1.0, top_p=0.95, max_tokens=max_tokens)
            return self.parse_problem_completion_from_generation(
                original_full_keywords=seed_or_keywords, 
                generation=added_keywords_str, 
                num_to_add=num_to_add, 
                subtask=subtask, 
                seed_used_in_prompt=sampled_seed_for_parsing
            )

        outputs = self._parmap(_do, batch_tasks)
        return outputs

    @staticmethod
    def parse_problem_completion_from_generation(original_full_keywords: List[str], generation: str, num_to_add: int, subtask: str, seed_used_in_prompt: List[str]) -> List[Dict]:
        """
        Parses the generated additional keywords and combines them with the seed.
        Now handles all strategies.
        """
        added_keywords_str = generation.strip().replace("\n", ",")
        added_keywords = [kw.strip() for kw in added_keywords_str.split(",") if kw.strip()]

        kept_seed_keywords = []
        if subtask == "upsample":
            kept_seed_keywords = random.sample(original_full_keywords, max(0,  len(original_full_keywords) - 1))
        elif subtask == "refresh":
            kept_seed_keywords = random.sample(original_full_keywords, max(0, len(original_full_keywords) - num_to_add))
        
        elif subtask in {"extend", "explore"}:
            kept_seed_keywords = seed_used_in_prompt

        full_set = sorted(list(set(kept_seed_keywords + added_keywords[:num_to_add])))
        
        if 3 <= len(full_set) <= 5:
            return [{"inputs": ", ".join(full_set), "seed": original_full_keywords, "added": added_keywords}]
        else:
            return []
    
        # Helper methods for problems
    def prepare_problem_prompt(self, fewshot_sample_prompts: List[str], num_new_problems: int) -> str:
        instruction_prompt = problem_instruction_template.replace("$#$num_new_problems$#$", str(num_new_problems))        
        random.shuffle(fewshot_sample_prompts)
        fewshot_block = "\n\n".join(fewshot_sample_prompts)
        full_prompt = instruction_prompt + fewshot_block
        
        return full_prompt

    @staticmethod
    def prepare_problem_fewshot_sample_prompt(sample: str) -> str:
        return sample.strip()

    @staticmethod
    def parse_problem_from_generation(generation: str) -> List[Dict]:
        out_samples = []
        for split_generation in generation.split("---")[1:-1]:
            seg = split_generation.strip()
            if "[[Problem]]" in seg:
                problem = seg[seg.find("[[Problem]]") + len("[[Problem]]"):].strip()
                problem = problem.replace("\n", " ").strip()
                out_samples.append({"inputs": problem})
        return out_samples





    # --------------------------- Solution gen --------------------------- #

    def batch_prompt_solution(self, samples: List[Dict], few_shot_samples_batch: List[List[str]], generate_method:str, n_sen:int) -> List[Dict]:
        """ 
        Generates solutions for a batch of problems, using corresponding few-shot examples for each.
        """
        # Pair each sample with its corresponding list of few-shot examples
        
        jobs = [(s, f, generate_method) for s, f in zip(samples, few_shot_samples_batch)]
        
        def _do(job: tuple) -> Dict:
            sample, few_shot_examples, method = job
            
            system_prompt, prompt = self.prepare_solution_prompt(sample["inputs"], few_shot_examples, method, n_sen)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            
            # Use the instance's temperature for solution generation
            text = self._chat_completion(messages, temperature=self.temperature, max_tokens=self.max_gen_len)
            return self.parse_solution_from_generation(sample, text, method, n_sen)

        results = self._parmap(_do, jobs)
        return results

    def prepare_solution_prompt(self, problem_keywords: str, few_shot_examples: List[str], method: str, n_sen:int) -> str:
        """
        Constructs the full prompt for solution generation, including instructions,
        few-shot examples, and the target keywords.
        """
        if method in ["dynamic_fewshot", "fewshot"]: 
            system_prompt = fewshot_instruction_system_template
            random.shuffle(few_shot_examples)
            fewshot_block = "\n".join(few_shot_examples)
            prompt = fewshot_instruction_template.replace("$#$problem$#$", problem_keywords).replace(
                "$#$N$#$", str(n_sen) if method == "fewshot" else "1"
            )
            prompt = prompt.replace("$#$num_to_add$#$", str(len(problem_keywords.split(", "))))
            full_prompt = prompt + fewshot_block
            return system_prompt, full_prompt
        
        elif method == "persona":
            system_prompt = generate_persona()
            prompt = persona_instruction_template.replace("$#$problem$#$", problem_keywords)
            prompt = prompt.replace("$#$num_to_add$#$", str(len(problem_keywords.split(", "))))
            # print("Persona Prompt:", system_prompt) # Debugging line
            return system_prompt, prompt
            
            
        elif method == "cot":
            system_prompt = cot_system_template
            # random shulffle the keywords for more variety
            temp_keywords = problem_keywords.split(", ")
            random.shuffle(temp_keywords)
            temp_keywords = ", ".join(temp_keywords)
            
            prompt = cot_instruction_template.replace("$#$problem$#$", temp_keywords).replace(
                "$#$N$#$", "1"
            )
            prompt = prompt.replace("$#$num_to_add$#$", str(len(problem_keywords.split(", "))))

            return system_prompt, prompt


    @staticmethod
    def parse_solution_from_generation(sample: Dict, generation: str, method:str, n_sen:int) -> Dict:
        """
        Parse n_sen tab-separated sentences for CommonGen.
        Be lenient to newlines/punctuation; output a normalized tab-joined string and list.
        """
        # print("Generation:", generation) # Debugging line

        if method not in ["dynamic_fewshot", "persona", "cot"]:
            text = generation.strip()
            # First, try splitting by the intended tab character
            parts = [p.strip() for p in text.split("\t") if p.strip()]


            if len(parts) < n_sen:
                lines = [p.strip() for p in re.split(r"[\n\r]+", text) if p.strip()]
                # Often, models might output numbered lists like "1. sentence". Remove the prefix.
                lines = [re.sub(r"^\d+\.\s*", "", line) for line in lines]
                if len(lines) >= n_sen:
                    parts = lines[:n_sen]

            # Second Fallback: if that also fails, split by sentence-ending punctuation.
            if len(parts) < n_sen:
                sents = re.split(r"(?<=[.!?])\s+", text)
                sents_cleaned = [p.strip() for p in sents if p.strip()]
                if len(sents_cleaned) >= n_sen:
                    parts = sents_cleaned[:n_sen]
        
            # Ensure we always return a list of 4, padding with empty strings if necessary
            while len(parts) < n_sen:
                parts.append("")
            completion = "\t".join(parts[:n_sen])  # Ensure only 4 are joined
        else:
            # For other methods, we assume a single sentence is generated
            if method == "cot":
                # In CoT, we need to extract the final sentence after reasoning
                lines = generation.strip().split("\n")
                sentence = ""
                for line in reversed(lines):
                    line = line.strip()
                    if line and not line.lower().startswith("let's think step by step"):
                        sentence = line
                        break
            else:      
                sentence = generation.strip().replace("\n", " ").split("\t")[0]
            completion = re.sub(r"^\d+\.\s*", "", sentence)  # Remove numbering if present
            
            # print("Parsed Completion:", completion) # Debugging line
        return {
            "inputs": sample.get("inputs"),
            "seed": sample.get("seed"),
            "added": sample.get("added"),
            "completion": completion,
        }


    # Quality Assessment
    def rate_sentence_plausibility(self, batch_jobs: List[Dict]) -> List[Dict]:
        """
        Rates the commonsense plausibility of sentences based on keywords.
        
        Args:
            batch_jobs: A list of dictionaries, each with {"keywords": str, "sentence": str}.
            
        Returns:
            A list of the input dictionaries, each updated with a "plausibility_score".
        """
        def _do(job: Dict) -> Dict:
            keywords = job["keywords"]
            sentence = job["sentence"]

            # Construct the prompt
            prompt = quality_assessment_user_template.replace(
                "$#$keywords$#$", keywords
            ).replace(
                "$#$sentence$#$", sentence
            )
            
            messages = [
                {"role": "system", "content": quality_assessment_system_template},
                {"role": "user", "content": prompt},
            ]

            # Use low temperature for deterministic scoring, max_tokens=2 is enough for "1"-"5"
            response = self._chat_completion(messages, temperature=0.01, max_tokens=3)
            
            score = 0
            try:
                # Extract the first integer from the response
                score = int(re.search(r'\d+', response).group())
            except (ValueError, AttributeError):
                score = 1 # Default to the lowest score if parsing fails

            job["plausibility_score"] = score
            return job

        # Use _parmap for efficient batch processing if needed, but here we process the list directly
        # For simplicity, we can do it in a ThreadPoolExecutor here as well.
        results = self._parmap(_do, batch_jobs)
        return results