# CommonSyn

This repository contains the final synthetic dataset **CommonSyn**, as presented in our paper, together with the complete pipeline used to generate it and the scripts for fine-tuning models on it.

---

## 📊 CommonSyn Dataset

### Main Dataset: `commonsyn.jsonl`

The `commonsyn.jsonl` file is the final output of our generation pipeline. Each line in this file is a JSON object containing the training data and its associated metrics.

The instruction used in our experiment is:
```bash
Given several keywords, generate one coherent sentence that contains all the required keywords using background commonsense knowledge: 
```

### Data Fields

| Key | Description |
| :--- | :--- |
| **`prompt`** | Represents the **concept set**. This is the input collection of the task. |
| **`completion`** | The generated **sentence** that covers all concepts provided in the concept set. |

### 📂 Sources Folder

The `sources/` directory contains three specific files.

These files correspond to the **three sentence sources** (generation strategies) described in **Section 3.3** of our paper. They represent the diverse candidate pools from which the final dataset was constructed. Each sample contains one concept set and 4 sentences seperated by '\t'.

---

## 🚀 Data Generation Pipeline

The generation process consists of three sequential steps. Please follow them in order to ensure the data is constructed correctly.

### Step 1: Generate Concept Set

First, initialize the concept set using the "2-seed" method. This will create the foundational problems/concepts for the dataset.

```bash
python generate_problem.py
```

### Step 2: Generate Sentence Candidates

Next, generate candidate sentences based on the concepts created in step 1. This script employs **three different sentence generation strategies** to ensure diversity in the data.

```bash
python generate_candidates.py
```

### Step 3: Final Data Creation

Finally, select the candidates and build the final CommonSyn synthetic dataset using the 2-step data selection.

```bash
python 2step_creation.py
```

---

## 🧠 Model Training

Once you have generated (or downloaded) the CommonSyn dataset, you can use it to train or fine-tune your model.

### Configuration

Before running the training script, you must update the configuration file:

1. Open `sft.yaml`.
2. Locate the data path setting.
3. Set the path to point to your CommonSyn file (either the released `commonsyn.jsonl` or the one generated from Step 3).

### Run Training

Start the Supervised Fine-Tuning (SFT) process by running the main script with your configuration:

```bash
python main.py -config sft.yaml
```
