import json
from pathlib import Path
import jsonlines

def create_seed_file_for_gradient_collection(
    original_commongen_path: Path, 
    output_seed_path: Path
):
    """
    Converts original CommonGen data to the format required by collect_gradients.py.
    """
    print(f"Reading original data from: {original_commongen_path}")
    
    formatted_samples = []
    sample_counter = 0
    
    with jsonlines.open(original_commongen_path) as reader:
        for obj in reader:
            keywords = obj.get("inputs")
            labels = obj.get("selected_sentences")
            quality_scores = obj.get("quality_scores")
            if isinstance(labels, str):
                labels = labels.split("\t")
            else:
                labels = labels
            for i in range(len(labels)):
                if not keywords or not labels[i]:
                    continue
                
                formatted_samples.append({
                    "id": f"step1_{sample_counter}",
                    "prompt": keywords,
                    "completion": labels[i],
                    "quality": quality_scores[i] if quality_scores else 0.0,
                })
                sample_counter += 1
                
        print(f"Converted original samples into {len(formatted_samples)} prompt-completion pairs.")
    
    with jsonlines.open(output_seed_path, 'w') as writer:
        writer.write_all(formatted_samples)
        
    print(f"Formatted seed file saved to: {output_seed_path}")

if __name__ == '__main__':
    original_file = Path("") 
    seed_file = Path("") 
    
    # Create parent directories if they don't exist
    seed_file.parent.mkdir(parents=True, exist_ok=True)
    
    create_seed_file_for_gradient_collection(original_file, seed_file)