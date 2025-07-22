# 🧠 Liver-LLM: Advancing Liver Cancer Diagnosis and Treatment with Continual Learning​

<!-- This project implements a **Retrieval-Augmented Generation (RAG)** pipeline to perform question answering in the **liver disease medical domain**, using a fine-tuned [LLaMA 3](https://ai.meta.com/llama/) model with **LoRA adapters**, **FAISS** for retrieval, and **Sentence Transformers** for embedding scientific text chunks. -->
In this work, we present the fine-tuning of the **Llama-3-8B** model on a curated dataset comprising liver cancer research publications and a specialized question-answering corpus focused on liver diseases. The resulting model, named **LiverLlama_FineTuned**, is tailored to support clinicians and researchers by delivering accurate, context-aware, and up-to-date responses to complex medical questions. To overcome the challenge of static knowledge in large language models, we incorporate a continual learning mechanism that enables users to contribute new research articles for ongoing model refinement. This approach ensures the model remains aligned with the latest medical advancements, enhancing its value as both a clinical decision-support system and a research aid. Furthermore, we integrate a Retrieval-Augmented Generation (RAG) pipeline, enabling the model to ground its responses in user-provided documents at inference time—thereby significantly improving the contextual relevance and accuracy of its outputs.


---
<!-- ├── papers/ # PDF research papers -- we haven't uploaded the papers directory to GitHub for security purposes but a directory can be created in the root folder to store the research papers
├── scripts/ # Preprocessing and training scripts
├── tasks/ # JSON-formatted training data chunks
├── models/ # LoRA fine-tuned model artifacts
├── faiss_index/ # Vector store for retrieval
├── final_model/ # Trained model for inference
└── README.md # You are here -->

## 🧰 Features

- 🧾 Around 40k **Liver Research Papers** to pretrain the base Llama 3 model `meta-llama/Meta-Llama-3-8B`
- ❓ **QA dataset** with around 50k open-vocabulary and MCQ questions for fine-tuning the model further
- 📑 **PDF parsing** and chunking with PyMuPDF
- 🦙 **LoRA fine-tuning** of the models to improve training efficiency and save space
- 🤖 End-to-end **RAG pipeline** with real-time document retrieval and generation using [FAISS](https://github.com/facebookresearch/faiss) vector database and `all-MiniLM-L6-v2` embedding model

---

## 📂 Project Structure
```bash
LIVER-LLM/
├── steps/ # PDF download, text extraction, task creation, model training and fine-tuning, and evaluation scripts
│ ├── __init__.py
│ ├── evaluate_qa_finetuned_mcq.py # Evaluates the finetuned model on the MCQ validation set
│ ├── evaluate_qa_finetuned.py # Evaluates the finetuned model (LiverLlama_Finetuned) on the open-vocabulary questions (MOA) validation dataset
│ ├── evaluate_qa_moa_llama3.py # Evaluates the base model (Llama 3) on the open-vocabulary questions (MOA) validation dataset
│ ├── extract_paper.py # Extracts and cleans text from all the research papers and stores the content on individual JSON files created in the extracted_chunk_text directory
│ ├── new_pretrain_LLM.py # Executes the pretraining on the base model using the generated tasks and continual learning loop
│ ├── paper_download.py # Script to download the papers using Unpaywall API and the DOIs provided in the hop2 and hop3 files
│ ├── perplexity_pretrained_model.py # Script to evaluate the perplexity score of the base model and 
│ ├── qa_finetuning_mcq.py # Finetunes the pretrained model on the MCQ dataset using a custom classification layer
│ ├── qa_finetuning_moa.py # Finetunes the pretrained model on the open-vocabulary questions (MOA) dataset
│ ├── rag.py # Implements the RAG pipeline
│ └── task_creation.py # Creates the task by combining the text content from multiple papers, that is used for the continual learning of the model
│
├── utils/ # Utility functions and helpers
│ ├── __init__.py
│ └── cleaner.py # Series of text cleaning functions
│
├── config.py # Path configurations
├── continual_learning.py # Implements the final Streamlit UI's continual learning logic
├── continual_learning.sh # SLURM script to execute continual learning when a user uploads a paper
├── job.sh # SLURM script to execute the main pretraining and finetuning pipeline in main.py
├── main.py # Executes the entire pretraining and finetuning pipeline and evaluation of the model
├── run_rag.sh # SLURM script to run the RAG pipeline
├── run_rag.py # Python script to invoke RAG inference for the paper uploaded by the user
├── requirements.txt # Python dependencies
└── README.md
```

---


## 🚀 Quick Start

### 1.  Clone the Repository
Clone or download this repository using the command below:
```bash
git clone https://github.com/anniechakraborty/Liver-LLM.git
```

### 2. Install dependencies
Create a virtual environment and install the required dependencies. You can set up the virtual environment based on your system configuration and whether you're using Python or Anaconda to manage environments.

After creating the virtual environment, activate it and install the dependencies with the following command:
```bash
pip install -r requirements.txt
```

### 3. Data Collection
Inside the root directory of the `Liver-LLM` repository, create the following directories:
- `papers` directory which serves as the input directory to store the research papers on liver cancer and diseases
- `qa_dataset` directory which stores the QA train dataset used for finetuning the pretrained model on the MCQ and open-vocabulary (MOA) questions
- `qa_validation_dataset` directory with the MCQ and the MOA validation datasets for evaluation

To prepare the data for pretraining:
- Create a directory called `source_csv` in the root of the `Liver-LLM` repository.  
   This directory should contain the `hop2.csv`, `hop3.csv`, or any other CSV files with DOI links to liver-related research papers.

- Running the `paper_download.py` script (automatically triggered from `main.py`) will parse these CSV files and use the DOI links to download the corresponding papers into the `papers` directory.

⚠️ Note: If you already have a collection of liver research papers, you can skip the paper download step. Simply upload your PDFs into the `papers` directory in the root. In this case, comment out the paper download step in main.py by skipping the DownloadPapers() execution as follows, and start directly from `Step 2: Extract the paper content` using the ExtractPaper class
```python
# Step 1 : Download the papers
# papers = DownloadPapers()
# papers.main()
```



### 4. Run the pipeline
To execute the pretraining and finetuning pipeline, run the `main.py` file using the `job.sh` SLURM Script. This schedules a job in the GPU cluster to execute the `main.py`.

```bash
sbatch job.sh
```
The different stages of the pipeline are executed sequentially as shown in the excerpt from the main.py script below. The GPU time allocations must be updated accordingly in the SLURM script before scheduling a job. 

```python
from steps import DownloadPapers, ExtractPaper, TaskCreation
from steps import NewPretrainLLM
from steps import PPL_Evaluator
from steps import QA_Finetuning_MCQ
from steps import QA_Finetuning_MOA
from steps import FineTunedModelEvaluationMOA
from steps import FineTunedModelEvaluationMCQ
def main():
    # Step 1 : Download the papers
    papers = DownloadPapers()
    papers.main()
    
    # Step 2: Extract the paper content
    extract_papers = ExtractPaper()
    extract_papers.main()

    # Step 3: Task Creation
    tasks = TaskCreation()
    tasks.main()

    # Step 4: Pretrain the LLAMA 3 model
    trainer = NewPretrainLLM()
    trainer.continual_loop()

    # Step 5: Evaluating the pretrained model and base model on Perplexity
    ppl_eval = PPL_Evaluator()
    ppl_eval.main()

    # Step 6: Finetuning on open-vocab (MOA) and MCQ questions
    qa_ft_mcq=QA_Finetuning_MCQ()
    qa_ft_mcq.main()
    
    qa_ft_new=QA_Finetuning_MOA()
    qa_ft_new.main()

    # Step 7: Evaluation of finetuned models
    fteval_mcq = FineTunedModelEvaluationMCQ()
    fteval_mcq.main()

    fteval_moa = FineTunedModelEvaluationMOA()
    fteval_moa.main()
  
if __name__ == '__main__':
    main()

```

### 5. User Interaction
The user can interact with the `LiverLlama_FineTuned` model in two modes.
- Retrieval-Augmented Generation (RAG) pipeline
- Continual Learning

To implement the RAG inference, schedule the `run_rag.sh` SLURM Script using sbatch. Inside the SLURM script update the arguments with the path to the PDF file you want to run the inference on, and the question you want to ask.
```bash
python run_rag.py --pdf /mnt/beegfs/home/st191428/A_multicenter_prospective_study.pdf --question "What is HCC?"
```

To run the continual learning loop on the newly uploaded paper, schedule the `continual_learning.sh` SLURM Script using sbatch. Inside the SLURM script update the arguments with the path to the PDF file you want to train the model on.
```bash
python continual_learning.py --pdf /mnt/beegfs/home/st191428/Treatment_of_Liver_Cancer.pdf
```

---
## 🙌 Acknowledgements
Huge thank you to Prof. Mojataba Nayyeri for offering this project in the Deep Learning Lab course and for your continuous support and insights. Also, thanks to the open-source community behind Hugging Face, Meta AI, and the contributors of the liver cancer research datasets.