import os
import json
import random
from tqdm import tqdm

# Settings
input_dir = "extracted_chunked_text/"
output_dir = "tasks/"
max_tokens_per_task = 8192  # Adjust based on your training setup

os.makedirs(output_dir, exist_ok=True)

# --- Optional: Use a proper tokenizer if needed ---
def estimate_tokens(text):
    """Rough token estimator: 1 token ≈ 4 characters (OpenAI-style)."""
    return max(1, len(text) // 4)

# Step 1: Load and index all paper files
paper_data = []
for filename in tqdm(sorted(os.listdir(input_dir))):
    if filename.endswith(".jsonl"):
        with open(os.path.join(input_dir, filename), "r") as f:
            for line in f:
                content = json.loads(line)
                text = content.get("text", "")
                tokens = estimate_tokens(text)
                paper_data.append({
                    "filename": filename,
                    "content": content,
                    "tokens": tokens
                })

# Step 2: Shuffle
random.shuffle(paper_data)

# Step 3: Create token-constrained task batches
task_id = 1
current_task = []
current_tokens = 0

for paper in paper_data:
    if current_tokens + paper["tokens"] > max_tokens_per_task and current_task:
        # Write current task to file
        task_filename = f"task_{task_id:05d}.json"
        with open(os.path.join(output_dir, task_filename), "w") as f:
            json.dump(current_task, f, indent=2)
        task_id += 1
        current_task = []
        current_tokens = 0

    current_task.append(paper["content"])
    current_tokens += paper["tokens"]

# Final leftover task
if current_task:
    task_filename = f"task_{task_id:05d}.json"
    with open(os.path.join(output_dir, task_filename), "w") as f:
        json.dump(current_task, f, indent=2)

print(f"✅ Created {task_id} task files in: {output_dir}")
