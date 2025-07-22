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
