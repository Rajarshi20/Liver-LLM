
#Extract Paper
ExtractPaper_PDF_DIR = "papers/"
ExtractPaper_OUTPUT_DIR = "extracted_chunked_text/"

#Task Creation
Task_INPUT_DIR = "extracted_chunked_text/"
Task_OUTPUT_DIR = "tasks/"

#Evaluation
BASE_MODEL_PATH = "meta-llama/Meta-Llama-3-8B-Instruct"
LORA_MODEL_PATH = "./lora-liver-finetuned"
QA_DATASET_PATH = "./qa_dataset.json"

#Pretrain Model
pretrain_model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct"
pretrain_data_dir: str = "data"
pretrain_output_dir: str = "llama4-medical-finetuned"
saved_liver_llm_model: str = f"{pretrain_output_dir}/final_model"
saved_liver_llm_qa_model: str = "llama4-qa-finetuned"