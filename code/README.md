# CommonSyn Synthetic Data Generation

This codebase provides a complete pipeline for generating high-quality synthetic data using the CommonSyn methodology, as well as scripts for fine-tuning models on the generated datasets.

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

Once you have generated the CommonSyn dataset, you can use it to train or fine-tune your model.

### Configuration

Before running the training script, you must update the configuration file:

1. Open `sft.yaml`.
2. Locate the data path setting.
3. Set the path to point to your newly generated CommonSyn file from Step 3.

### Run Training

Start the Supervised Fine-Tuning (SFT) process by running the main script with your configuration:

```bash
python main.py -config sft.yaml
```


