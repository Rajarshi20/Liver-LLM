#Extract Paper
ExtractPaper_PDF_DIR = "papers/"
ExtractPaper_OUTPUT_DIR = "extracted_chunked_text/"

#Task Creation
Task_INPUT_DIR = "extracted_chunked_text/"
Task_OUTPUT_DIR = "tasks/"

#Finetuning and Evaluation
BASE_MODEL_PATH = "meta-llama/Meta-Llama-3-8B"
LORA_MODEL_PATH = r"/path_to_GPU/home/Liver-LLM/llama3_medical_finetuned" #pretrained model
QA_FINETUNED_MOA_MODEL_PATH = r"/path_to_GPU/home/Liver-LLM/llama3-qa-moa_model"
QA_FINETUNED_MCQ_MODEL_PATH = r"/path_to_GPU/home/Liver-LLM/llama3_qa_mcq_model"
CONTINUAL_LEARNING_MODEL_PATH = r"/path_to_GPU/home/llama3_medical_continual_learning/final_model" #continual learning model
QA_DATASET_PATH = r"/path_to_GPU/home/qa_dataset/qa_dataset_shuffled.jsonl" #QA Fine tuning dataset

QA_FINETUNED_MOA_VALIDATION_DATASET = r"/path_to_GPU/home/qa_validation_dataset/MOA_val.jsonl" #MOA
QA_FINETUNED_MCQ_VALIDATION_DATASET = r"/path_to_GPU/home/qa_validation_dataset/MCQ_val.jsonl" #MCQ

QA_BASEMODEL_MOA_EVAL = r"/path_to_GPU/home/eval_outputs/moa_eval/moa_qa_predictions.json"