
#Extract Paper
ExtractPaper_PDF_DIR = "papers/"
ExtractPaper_OUTPUT_DIR = "extracted_chunked_text/"

#Task Creation
Task_INPUT_DIR = "extracted_chunked_text/"
Task_OUTPUT_DIR = "tasks/"

#Evaluation
BASE_MODEL_PATH = "meta-llama/Meta-Llama-3-8B-Instruct"
LORA_MODEL_PATH = r"/hkfs/work/workspace/scratch/st_st191428-LiverLLM/Liver LLM/llama3_medical_finetuned/final_model"
QA_FINETUNED_MODEL_PATH = r"/hkfs/work/workspace/scratch/st_st191428-LiverLLM/Liver LLM/llama4-qa-finetuned"
QA_DATASET_PATH = "liver_qa_data.jsonl"
QA_FINETUNED_DATASET_PATH = "ft_eval.jsonl"

#Pretrain Model
pretrain_model_id: str = "meta-llama/Meta-Llama-3-8B-Instruct"
pretrain_data_dir: str = "data"
pretrain_output_dir: str = "llama4-medical-finetuned"