import os
import json
import random
from tqdm import tqdm

# Settings
input_dir = "extracted_chunked_text/"
output_dir = "tasks/"
max_tokens_per_task = 15000  # You can try increasing this to 12000–16000 for testing

os.makedirs(output_dir, exist_ok=True)

# --- Optional: Use a better tokenizer in production ---
def estimate_tokens(text):
    """Better rough estimate: 1 token ≈ 6 characters."""
    return max(1, len(text) // 6)

# Step 1: Load and index all paper files
paper_data = []

print("🔄 Loading papers...")
for filename in tqdm(sorted(os.listdir(input_dir))):
    if filename.endswith(".json"):
        with open(os.path.join(input_dir, filename), "r") as f:
            content = json.load(f)
            chunks = content.get("chunks", [])
            full_text = "\n".join(chunk.get("text", "") for chunk in chunks)
            tokens = estimate_tokens(full_text)
            paper_data.append({
                "filename": filename,
                "content": content,
                "tokens": tokens
            })
            print(f"📄 {filename}: estimated {tokens} tokens")

# Step 2: Shuffle papers
random.shuffle(paper_data)

# Step 3: Create task files with ~8192 tokens worth of papers
task_id = 1
current_task = []
current_tokens = 0

for paper in paper_data:
    paper_tokens = paper["tokens"]
    if current_tokens + paper_tokens > max_tokens_per_task and current_task:
        print(f"✅ Writing task_{task_id:05d}.json with {len(current_task)} papers, total tokens: {current_tokens}")
        with open(os.path.join(output_dir, f"task_{task_id:05d}.json"), "w") as f:
            json.dump(current_task, f, indent=2)
        task_id += 1
        current_task = []
        current_tokens = 0

    current_task.append(paper["content"])
    current_tokens += paper_tokens

# Final leftover task
if current_task:
    print(f"✅ Writing task_{task_id:05d}.json with {len(current_task)} papers, total tokens: {current_tokens}")
    with open(os.path.join(output_dir, f"task_{task_id:05d}.json"), "w") as f:
        json.dump(current_task, f, indent=2)

print(f"\n🎯 Done! Created {task_id} task files in: {output_dir}")
