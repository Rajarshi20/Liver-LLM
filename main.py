import os
import pandas as pd
from steps import DownloadPapers, ExtractPaper, TaskCreation, PretrainLLM, PretrainDeepseekLLM, ModelEvaluation, FineTunedModelEvaluation

def main():
    # Step 1 : Download the papers
    # papers = DownloadPapers()
    # papers.main()
    
    # Step 2: Extract the paper content
    # extract_papers = ExtractPaper()
    # extract_papers.main()

    # Step 3: Task Creation
    """ tasks = TaskCreation()
    tasks.main() """

    # Step 4: Pretrain the LLAMA 4 Scout model
    """ trainer = PretrainLLM(
        data_dir="tasks",
        output_dir="llama4_medical_finetuned"
    )
    trainer.train() """

    # Step 5: Pretrain the Deepseek R1 model
    """  deepseek = PretrainDeepseekLLM(
        model_id="deepseek-ai/deepseek-llm-r1",
        data_dir="tasks",
        output_dir="deepseek-liver-llm"
    )
    deepseek.train()
 """
    """ evaluator = ModelEvaluation()
    evaluator.main() """

    fteval = FineTunedModelEvaluation()
    fteval.main()

if __name__ == '__main__':
    main()
