import os
import json
import random
from tqdm import tqdm
from config import Task_INPUT_DIR, Task_OUTPUT_DIR

class TaskCreation:
    MAY_TOKENS_PER_TASK = 8192  # You can try increasing this to 12000–16000 for testing

    paper_data = []

    # --- Optional: Use a better tokenizer in production ---
    def estimate_tokens(self, text):
        """Better rough estimate: 1 token ≈ 6 characters."""
        return max(1, len(text) // 6)

    def load_and_index_papers(self):
        print("🔄 Loading papers...")
        for filename in tqdm(sorted(os.listdir(Task_INPUT_DIR))):
            if filename.endswith(".json"):
                with open(os.path.join(Task_INPUT_DIR, filename), "r") as f:
                    content = json.load(f)
                    chunks = content.get("chunks", [])
                    full_text = "\n".join(chunk.get("text", "") for chunk in chunks)
                    tokens = self.estimate_tokens(full_text)
                    self.paper_data.append({
                        "filename": filename,
                        "content": content,
                        "tokens": tokens
                    })
                    print(f"📄 {filename}: estimated {tokens} tokens")

    def create_tasks(self):
        task_id = 1
        current_task = []
        current_tokens = 0

        for paper in self.paper_data:
            paper_tokens = paper["tokens"]
            if current_tokens + paper_tokens > self.MAY_TOKENS_PER_TASK and current_task:
                print(f"✅ Writing task_{task_id:05d}.json with {len(current_task)} papers, total tokens: {current_tokens}")
                with open(os.path.join(Task_OUTPUT_DIR, f"task_{task_id:05d}.json"), "w") as f:
                    json.dump(current_task, f, indent=2)
                task_id += 1
                current_task = []
                current_tokens = 0

            current_task.append(paper["content"])
            current_tokens += paper_tokens

        # Final leftover task
        if current_task:
            print(f"Writing task_{task_id:05d}.json with {len(current_task)} papers, total tokens: {current_tokens}")
            with open(os.path.join(Task_OUTPUT_DIR, f"task_{task_id:05d}.json"), "w") as f:
                json.dump(current_task, f, indent=2)

        print(f"\nDone! Created {task_id} task files in: {Task_OUTPUT_DIR}")

    def main(self):
        # Step 0: Ensuring all directories are created
        os.makedirs(Task_INPUT_DIR, exist_ok=True)
        os.makedirs(Task_OUTPUT_DIR, exist_ok=True)

        # Step 1: Load and index all paper files
        self.load_and_index_papers()
        
        # Step 2: Shuffle papers
        random.shuffle(self.paper_data)

        # Step 3: Create task files with the max_tokens worth of papers
        self.create_tasks()