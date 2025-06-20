from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import torch
import math
from config import BASE_MODEL_PATH, LORA_MODEL_PATH

class PPL_Evaluator:
    def __init__(self, device='cuda'):
        self.device = device

    def compute_perplexity(self, base_path, texts, lora_path=None):
        # 4-bit quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        tokenizer = AutoTokenizer.from_pretrained(base_path)
        model = AutoModelForCausalLM.from_pretrained(
            base_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )

        if lora_path:
            print("This is the LORA Path:", lora_path)
            model = PeftModel.from_pretrained(model, lora_path, device_map="auto", local_files_only=True)

        model.eval()

        losses = []
        with torch.no_grad():
            for text in texts:
                inputs = tokenizer(text, return_tensors="pt").to(self.device)
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss
                losses.append(loss.item())

        avg_loss = sum(losses) / len(losses)
        perplexity = math.exp(avg_loss)
        return perplexity

    def main(self):
        medical_texts = [
            "Toll-like receptor 4 (TLR4) is a key player in airway inflammation.",
            "The Knodell score assesses necroinflammatory injury and fibrosis in chronic hepatitis.",
            "Asthma is widely considered to stem from allergen-specific Thelper type 2 (Th2) responses that result in eosinophilic inflammation but, in many individuals, neutrophils are the predominant leukocytes in the airway.",
            "Epidermal growth factor receptor seems to be a direct target of the pathway, and its activation might contribute toward mitogenic effects of increased β-catenin in the liver.",
        ]

        base_ppl = self.compute_perplexity(BASE_MODEL_PATH, medical_texts)
        pretrained_ppl = self.compute_perplexity(BASE_MODEL_PATH, medical_texts, lora_path=LORA_MODEL_PATH)

        print(f"Base model perplexity: {base_ppl:.2f}")
        print(f"Pretrained model perplexity: {pretrained_ppl:.2f}")

