import os
import pandas as pd
import streamlit as st
# from steps import DownloadPapers, ExtractPaper, TaskCreation, PretrainLLM, PretrainDeepseekLLM, ModelEvaluation, FineTunedModelEvaluation, ModelEvaluation_compare
from steps import RetrievalAugmentedGeneration

def main():
    # Step 1 : Download the papers
    # papers = DownloadPapers()
    # papers.main()
    
    # Step 2: Extract the paper content
    # extract_papers = ExtractPaper()
    # extract_papers.main()

    # Step 3: Task Creation
    # """ tasks = TaskCreation()
    # tasks.main() """

    # # Step 4: Pretrain the LLAMA 3 Scout model
    # """ trainer = PretrainLLM(
    #     data_dir="tasks",
    #     output_dir="llama3_medical_finetuned"
    # )
    # trainer.train() """

    # # Step 5: Pretrain the Deepseek R1 model
    # """  deepseek = PretrainDeepseekLLM(
    #     model_id="deepseek-ai/deepseek-llm-r1",
    #     data_dir="tasks",
    #     output_dir="deepseek-liver-llm"
    # )
    # deepseek.train()
    # """

    # # Step 6: Evaluating the inferences drawn on the pretrained LLAMA 3 model (step 4)
    # """ evaluator = ModelEvaluation()
    # evaluator.main() """

    # # Step 7: Finetuning the pretrained LLAMA model on the QA dataset
    # qa_finetuned = QA_Finetuning()

    # # Step 8: Evaluating the inferences drawn on the QA finetuned model
    # fteval = FineTunedModelEvaluation()
    # fteval.main()
    print()


def create_streamlit_app():
    if "qa_history" not in st.session_state:
        st.session_state.qa_history = []

    st.title("🩺 Liver LLM: Summarization, QA & Continual Learning")

    # Upload paper
    uploaded_file = st.file_uploader("Upload your paper (PDF)", type=["pdf"])
    upload_clicked = st.button("Upload Paper")

    if upload_clicked:
        if uploaded_file is not None:
            st.success("Paper uploaded successfully!")
            # Here you can call your backend or processing logic
        else:
            st.error("Please upload a paper before clicking upload.")

    # Choose mode
    mode = st.radio("Choose mode", options=["Ask Questions", "Continued Learning"])
    if mode == "Ask Questions":
        rag = RetrievalAugmentedGeneration(uploaded_file_content=uploaded_file)
        with st.form(key="question_form"):
            user_question = st.text_input("Ask a question about the paper")
            submit_question = st.form_submit_button("Submit Question")

            if submit_question and user_question:
                # Replace this with your actual backend response logic
                res = rag.main(query=user_question)
                answer = f"📘 {res}"
                st.session_state.qa_history.append((user_question, answer))

    # Show Q&A History
    if mode == "Ask Questions" and st.session_state.qa_history:
        st.markdown("### 🗂️ Q&A History")
        for q, a in reversed(st.session_state.qa_history):
            st.markdown(f"**Q:** {q}")
            st.markdown(f"**A:** {a}")
            st.markdown("---")

    elif mode == "Continued Learning":
        st.text_area("Continue learning notes or reflections here...")


if __name__ == '__main__':
    # main()
    create_streamlit_app()
