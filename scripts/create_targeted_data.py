import json
import random

def create_targeted_dataset(hard_triplets_file, gpt_file, claude_file, failure_file, output_file, num_original=1000):
    # 1. Load high-priority samples
    high_priority = []
    
    # Failure cases (30)
    with open(failure_file, 'r', encoding='utf-8') as f:
        high_priority.extend([json.loads(line) for line in f])
    
    # GPT augmented (30)
    with open(gpt_file, 'r', encoding='utf-8') as f:
        high_priority.extend([json.loads(line) for line in f])
        
    # Claude augmented (30)
    with open(claude_file, 'r', encoding='utf-8') as f:
        high_priority.extend([json.loads(line) for line in f])
        
    print(f"Loaded {len(high_priority)} high-priority samples.")
    
    # 2. Load and sample from original hard triplets (Grounding)
    with open(hard_triplets_file, 'r', encoding='utf-8') as f:
        all_original = [json.loads(line) for line in f]
    
    # Exclude high priority if they overlap (though unlikely to matter much)
    sampled_original = random.sample(all_original, min(num_original, len(all_original)))
    print(f"Sampled {len(sampled_original)} original samples for grounding.")
    
    # 3. Combine and shuffle
    final_dataset = high_priority + sampled_original
    random.shuffle(final_dataset)
    
    # 4. Save
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in final_dataset:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
    print(f"Targeted dataset saved to {output_file} (Total: {len(final_dataset)})")

if __name__ == "__main__":
    create_targeted_dataset(
        "data/hard_train_triplets.jsonl",
        "data/augmented_gpt.jsonl",
        "data/augmented_claude.jsonl",
        "data/failure_cases.jsonl",
        "data/targeted_train_triplets.jsonl"
    )
