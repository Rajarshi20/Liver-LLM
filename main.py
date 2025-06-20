import os
import pandas as pd
from steps import DownloadPapers, ExtractPaper, TaskCreation, PretrainLLM, PPL_Evaluator, BaseModelEvaluation, FineTunedModelEvaluation, Finetuned_ModelEval
from steps import NewPretrainLLM

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

    trainer = NewPretrainLLM()
    trainer.continual_loop()

    """ evaluator = ModelEvaluation()
    evaluator.main() """

    """ fteval = FineTunedModelEvaluation()
    fteval.main() """

    #TO compare base model and fine tuned model evaluation
    """ ft_model_eval = Finetuned_ModelEval()
    ft_model_eval.main()

    base_model_eval = BaseModelEvaluation()
    base_model_eval.main()
    """

    """ ppl_eval = PPL_Evaluator()
    ppl_eval.main()
 """
    
if __name__ == '__main__':
    main()
