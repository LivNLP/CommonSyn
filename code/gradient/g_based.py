import argparse
import json
import math
from pathlib import Path
import numpy as np
import torch

from cluster_manager import ClusterManager
from gradient_manager import GradientManager

# VENDI_MODEL = "princeton-nlp/sup-simcse-roberta-large"

def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate_dataset_filename", type=Path, required=True)
    p.add_argument("--save_filename",             type=Path, required=True)
    p.add_argument("--gradient_dir",              type=Path, required=True,)
    
    p.add_argument("--mode", choices=["small", "method1"], default="method1",
                   help="small= only smallest clusters; method1=Q+G")
    
    p.add_argument("--k", type=int, default=None)
    p.add_argument("--k_ratio", type=float, default=0.05)
    p.add_argument("--small_ratio", type=float, default=0.7)
    
    p.add_argument("--target_size", type=int, default=None)
    
    p.add_argument("--alpha", type=float, default=1.0, help="Quality weight")
    p.add_argument("--gamma", type=float, default=1.0, help="Gradient Rarity weight")
    
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def l2_normalize(x: np.ndarray) -> np.ndarray:
    eps = 1e-8
    n = np.maximum(np.linalg.norm(x, axis=1, keepdims=True), eps)
    return x / n

def compute_self_rarity_scores(labels, K):
    cluster_sizes = np.bincount(labels, minlength=K).astype(float)
    G_raw = 1.0 / np.sqrt(cluster_sizes + 1e-6)
    
    if G_raw.max() == G_raw.min():
        G_normalized = np.zeros_like(G_raw)
    else:
        G_normalized = (G_raw - G_raw.min()) / (G_raw.max() - G_raw.min() + 1e-9)
        
    G_values = G_normalized[labels]
    return G_values, cluster_sizes

def main():
    args = build_args()
    rng = np.random.RandomState(args.seed)
    
    print(f"[Info] Reading candidate data from {args.candidate_dataset_filename}...")
    all_records = []
    with args.candidate_dataset_filename.open("r") as fin:
        for line in fin:
            if not line.strip(): continue
            j = json.loads(line)
            all_records.append(j)
            
    cand_ids = [j["id"] for j in all_records]
    raw_Q = np.array([j.get("quality", 0.0) for j in all_records], dtype=float)
    if raw_Q.max() == raw_Q.min():
        Q = np.zeros_like(raw_Q)
    else:
        Q = (raw_Q - raw_Q.min()) / (raw_Q.max() - raw_Q.min() + 1e-9)

    print(f"[Info] Candidate size = {len(cand_ids)}")

    print(f"[Info] Loading gradients...")
    _, cand_mat = GradientManager.load_gradients_for_sample_ids(args.gradient_dir, cand_ids)
    cand_mat = l2_normalize(cand_mat.cpu().numpy().astype(np.float32))

    N = len(cand_ids)
    K = args.k if args.k is not None else max(2, int(round(N * args.k_ratio)))
    K = min(K, N)
    
    print(f"[Info] Clustering candidate (K={K})...")
    cand_tensor = torch.from_numpy(cand_mat).cuda()
    cand_labels, _ = ClusterManager.cluster_kmeans(cand_tensor, k=K, num_iter=20, use_tqdm=True)
    cand_labels = cand_labels.cpu().numpy()

    G, cluster_sizes = compute_self_rarity_scores(cand_labels, K)
    
    selected_indices = []
    target_n = args.target_size if args.target_size else N
    
    if args.mode == "small":
        sorted_cluster_indices = np.argsort(cluster_sizes) # index of clusters sorted by size
        num_rare_clusters = max(1, int(K * args.small_ratio))
        target_clusters = set(sorted_cluster_indices[:num_rare_clusters].tolist())
        
        print(f"[Info] Mode 'small': Selecting from the smallest {num_rare_clusters} clusters (ratio={args.small_ratio}).")
        candidates_in_rare = []
        for idx, label in enumerate(cand_labels):
            if label in target_clusters:
                candidates_in_rare.append(idx)
        
        if len(candidates_in_rare) > target_n:
            sub_Q = Q[candidates_in_rare]
            sorted_relative = np.argsort(-sub_Q) 
            selected_indices = [candidates_in_rare[i] for i in sorted_relative[:target_n]]
        else:
            selected_indices = candidates_in_rare
            
    elif args.mode == "method1":
        scores = args.alpha * Q + args.gamma * G
        top_idx = np.argsort(-scores)[:target_n]
        selected_indices = top_idx.tolist()
        print(f"[Info] Mode 'method1': Sorted by Q+G.")

    selected_set = set(selected_indices)
    idx_to_cluster = {i: int(cand_labels[i]) for i in selected_indices}
    
    kept_count = 0
    with args.save_filename.open("w") as out:
        for i in range(len(all_records)):
            if i in selected_set:
                rec = all_records[i]
                rec["cluster_id"] = idx_to_cluster[i]
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                kept_count += 1
                
    print(f"[Info] Saved {kept_count} samples.")

if __name__ == "__main__":
    main()