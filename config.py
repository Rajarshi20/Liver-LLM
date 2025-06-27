#Extract Paper
ExtractPaper_PDF_DIR = "papers/"
ExtractPaper_OUTPUT_DIR = "extracted_chunked_text/"

#Task Creation
Task_INPUT_DIR = "extracted_chunked_text/"
Task_OUTPUT_DIR = "tasks/"

#Evaluation
BASE_MODEL_PATH = "meta-llama/Meta-Llama-3-8B"
#LORA_MODEL_PATH = r"/hkfs/work/workspace/scratch/st_st191428-LiverLLM/Liver LLM/llama3_qa_finetuned/checkpoint-16221"
LORA_MODEL_PATH = r"/hkfs/work/workspace/scratch/st_st191428-LiverLLM/Liver LLM/llama3_medical_finetuned/checkpoint_7000" #pretrained model
QA_FINETUNED_MOA_MODEL_PATH = r"/hkfs/work/workspace/scratch/st_st191428-LiverLLM/Liver LLM/llama3-qa-finetuned/moa_model"
QA_FINETUNED_MCQ_MODEL_PATH = r"/hkfs/work/workspace/scratch/st_st191428-LiverLLM/Liver LLM/llama3_qa_mcq_model"
QA_DATASET_PATH = r"/hkfs/work/workspace/scratch/st_st191428-LiverLLM/Liver LLM/qa_dataset/qa_dataset_shuffled.jsonl" #QA Fine tuning dataset
QA_FINETUNED_DATASET_PATH = "ft_eval.jsonl"   #Remove
QA_FINETUNED_MOA_VALIDATION_DATASET = r"/hkfs/work/workspace/scratch/st_st191428-LiverLLM/Liver LLM/qa_validation_dataset/MOA_val.jsonl" #MOA
QA_FINETUNED_MCQ_VALIDATION_DATASET = r"/hkfs/work/workspace/scratch/st_st191428-LiverLLM/Liver LLM/qa_validation_dataset/MCQ_val.jsonl" #MCQ

#Pretrain Model
pretrain_model_id: str = "meta-llama/Meta-Llama-3-8B"
pretrain_data_dir: str = "data"
pretrain_output_dir: str = "llama4-medical-finetuned"


saved_liver_llm_model: str = f"{pretrain_output_dir}/final_model"
saved_liver_llm_qa_model: str = "llama3-qa-finetuned"
