Given an implausible or counterfactual statement, we ask models to generate a short explanation of *why* the statement is implausible or counterfactual.

Statement:
{$input}

Model A:
"{$candidate_A}"

**Model B:**
"{$candidate_B}"

# Your Task
Your task is to choose the better explanation. Decide which model’s output better explains why the statement is implausible.

## Rules:
- A good explanation should point out the everyday commonsense reason why the statement is unrealistic.
- Prefer simple, intuitive explanations that an person would naturally give.
- Prefer explanations that highlight the obvious mismatch with real-world scene.
- If both explanations are equally good or equally flawed, choose "tie".

Now, please output your choice ("A", "B", or "tie").

Your choice:
