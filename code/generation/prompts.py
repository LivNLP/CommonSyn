problem_instruction_template = """
Instruction: You are an expert in commonsense knowledge.
Your goal is to produce a concept set of $#$num_new_problems$#$ keywords that can naturally form ONE plausible daily-life event or scene,
which can be expressed in ONE single sentence under 24 words.

Rules for all keywords:
- Each keyword must be a common noun or verb in dictionary form.
- DO NOT use prepositions, articles, or pronouns.
- ADDED keywords must form a coherent *action chain* with the seed keywords.
- The combined keywords MUST be usable to construct one daily-life scene or action.
- Avoid overly abstract or static concepts.
- Output ONLY the added keywords, separated by commas.
- You MUST NOT output any keyword that appears in the seed, even if it seems logically appropriate.

Each example should be formatted exactly as:

---
[[Problem]]
keyword1, keyword2, keyword3[, keyword4, keyword5]
---

Examples:

"""

problem_2seed_instruction_template = """
Instruction: You are an expert in commonsense knowledge.
Given a seed of two keywords, complete the set by adding EXACTLY $#$num_to_add$#$ keywords.

Your goal is to produce a concept set that can naturally form ONE plausible daily-life event or scene,
which can be expressed in ONE single sentence under 24 words.

Rules for ADDED keywords:
- Each keyword must be a common noun or verb in dictionary form.
- DO NOT use prepositions, articles, or pronouns.
- ADDED keywords must form a coherent *action chain* with the seed keywords.
- The combined keywords (seed + added) MUST be usable to construct one daily-life scene or action.
- Avoid overly abstract or static concepts.
- Output ONLY the added keywords, separated by commas.
- You MUST NOT output any keyword that appears in the seed, even if it seems logically appropriate.

Here are some examples of varied lengths:


---
Example 1:
Seed: passenger, train
Output: station, run
---

---
Example 2:
Seed: commemorate, conflict
Output: statue, stand, city
---

Your Task:
Seed: $#$seed_keywords$#$
Output:
"""

problem_2seed_persona_instruction_template = """Based on your persona, lifestyle and the seed two keywords, adding EXACTLY $#$num_to_add$#$ commonsense-related keywords to describe a daily-life scene or action you experience in your daily life.

Rules:
- Think from the perspective of this person, and focus on realistic, daily actions or tools.
- Only add **nouns** (objects/entities) or **verbs** (actions), in base/dictionary form.
- The added keywords must be logically coherent with the seed and relevant to the persona.
- Do NOT repeat the seed keywords.
- Output only the new keywords, comma-separated.

---
Example 1:
Persona: A sushi chef who works late hours in a busy Tokyo restaurant.  
Seed: knife, rice  
Output: fish, cut, roll  
---

Example 2:
Persona: A retired farmer who enjoys morning chores in the countryside.  
Seed: cow, walk  
Output: barn, feed, hay  
---

Your Task:
Persona: $#$persona$#$
Seed: $#$seed_keywords$#$
Output:
"""


problem_completion_instruction_template = """
Instruction: You are an expert in commonsense knowledge.
Given a set of seed keywords, complete the set by adding EXACTLY $#$num_to_add$#$ keywords to form a plausible, daily-life scene or action.

Rules for ADDED keywords:
- Each keyword must be a common noun (object/entity) or verb (action) in its base/dictionary form.
- DO NOT use prepositions (e.g., in, on, at), articles (e.g., a, an, the), pronouns.
- The added keywords must be logically coherent with the seed keywords.
- Output ONLY the ADDED keywords, separated by commas.
- Do NOT GENERATE any of the input seed keywords.

Here are some examples of varied lengths:

---
Example 1:
Seed: discussion, question, ask
Output: student

---
Example 2:
Seed: commemorate, city, conflict
Output: statue, stand
---

---
Example 3:
Seed: ride, dog, horse
Output: run, shoot
---

Your Task:
Seed: $#$seed_keywords$#$
Output:
"""


problem_fewshot_template = """---
[[Problem]]
$#$problem$#$
---"""



fewshot_instruction_system_template = """You generate fluent and commonsense-bearing English sentences under strict formatting constraints. Do not add explanations or numbering. Output only the requested sentences."""
fewshot_instruction_template = """Instruction: Given $#$num_to_add$#$ keywords, generate exactly $#$N$#$ commmonsense-bearing and diverse English sentences. 
Each sentence MUST contain ALL the required keywords (inflectional variants are allowed, e.g., stand→stood/stands).

Requirements:
- The sentence must describe a logically possible everyday situation.
- It must contain **ALL** the provided keywords (inflectional variants allowed, e.g., stand→stood/stands).
- Keep the sentence concise (≤ 22 words).
- Separate sentences with a single TAB character (\t).
- Do not add explanations, numbering, or commentary.
- Output exactly $#$N$#$ sentence(s).

Diversity rule:
- When writing, you may vary the subject, perspective (first/third person), tone (descriptive, narrative, neutral), or setting (time/place) to make the expression distinct from other possible sentences.
- Use natural variation while maintaining plausibility.

Keywords: $#$problem$#$

Use the few-shot examples below as inspiration for style and structure, **not as templates to copy**.

"""

persona_instruction_template = """
Based on your lifestyle, and background, write ONE realistic sentence using **ALL** the provided keywords below. The sentence should describe your daily life or personal experience. Keep the sentence concise (≤ 20 words)

Keywords: $#$problem$#$

"""


cot_system_template = """
You must follow formatting constraints. Do not add numbering or commentary outside the required reasoning and final sentence. Output both a reasoning step and the final sentence for each example.
"""

cot_instruction_template = """
Instruction: Given $#$num_to_add$#$ keywords, generate exactly ONE commonsense-bearing English sentence based on your lifestyle, and background,. 
Each sentence MUST contain ALL the required keywords (inflectional variants are allowed, e.g., stand→stood/stands).

Requirements:
1. First write a reasoning or description to explain the underlying commonsense connection of the keywords. Start with **"Let's think step by step:"**
2. Then, on the next line, output ONE realistic English sentence that contains ALL the keywords.
3. Keep the sentence concise (≤ 22 words).

Formatting constraints:
- Use exactly **one reasoning paragraph (>=4 sentences) and one sentence** per generation.
- Separate each generation with a blank line.
- Do not add numbering, commentary, or bullet points.

Keywords: $#$problem$#$

"""


quality_assessment_system_template = """You are an human evaluator of commonsense knowledge. Your task is to rate a set of sentences based on how realistic and logically consistent they are 
as everyday scenarios involving all the given keywords. 
Evaluate each sentence independently and return only the scores in JSON format."""
g
quality_assessment_batch_template = """On a scale of 1 to 5, how plausible is EACH of the following sentences as a description of a common, everyday scenario involving ALL the given keywords?

**Criteria**:
1: **Impossible or nonsensical** - contradicts basic physical or logical reality (e.g., "A fish rides a bicycle.").
2: **Unrealistic** - physically possible but implausible in normal life (e.g., "A dog throws a frisbee to its owner.").
3: **Somewhat plausible** - possible but awkward, forced, or rare in real life.
4: **Realistic and reasonable** - coherent, natural, fits real-life experience.
5:  **Highly natural** - vivid, fluent, and fully aligned with everyday common sense.

**Keywords**: $#$keywords$#$

**Sentences to Evaluate**:
1. $#$sentence_1$#$
2. $#$sentence_2$#$
3. $#$sentence_3$#$
4. $#$sentence_4$#$

**Your response MUST be a JSON object with a single key "scores", containing a list of 4 integer scores. **

Example: 
{"scores": [4, 5, 3, 4]}
"""


QUALITY_GPT_SYSTEM_PROMPT = """You are a strict evaluator for synthetic commonsense sentences. 
Your job is to score each candidate sentence based only on quality, correctness, plausibility, and clarity.

Do NOT rewrite or improve the sentences.
Do NOT hallucinate information not supported by the sentence.

Output must strictly follow JSON format."""



QUALITY_GPT_SYSTEM_PROMPT = "You are an expert evaluator of sentence quality and commonsense reasoning."

QUALITY_GPT_PROMPT_BASE = """I will give you a set of concepts, and {num_sentences} candidate sentences generated using different prompting strategies.

Your task:
1. Score each sentence independently from 1 to 10.
2. Higher score = higher quality and commonsense correctness.
3. All scores must be integers.
4. Use the full range (1–10) with these guidelines:

   - **1–3 (poor):** Incorrect, implausible, ungrammatical, or fails to use the concepts meaningfully.
   - **4–6 (average):** Mostly correct but may have minor issues in clarity, grammar, or concept integration.
   - **7–8 (good):** Clear, fluent, plausible sentences that use the concepts well.
   - **9–10 (excellent):** Exceptional clarity, realism, and full concept integration. No errors.

Special instruction:
   - If a sentence is marked as "[EMPTY]", assign it a score of 1.

Evaluate based on:
   - **Commonsense correctness**: Is the event plausible and realistic?
   - **Concept coverage**: Are the concepts used meaningfully together?
   - **Clarity and grammar**: Is the sentence well-formed?

Concept set:
{concept_set}

Candidate sentences:
{sentence_list}

Output ONLY the following JSON:
{{
  "scores": [{score_placeholders}]
}}"""