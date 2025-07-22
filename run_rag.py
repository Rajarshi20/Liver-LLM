import os
import pandas as pd
import time
import logging
import traceback
# import streamlit as st
import argparse

from steps import RetrievalAugmentedGeneration

# def create_streamlit_app():
#     if "qa_history" not in st.session_state:
#         st.session_state.qa_history = []

#     st.title("🩺 Liver LLM: Summarization, QA & Continual Learning")

#     # Upload paper
#     uploaded_file = st.file_uploader("Upload your paper (PDF)", type=["pdf"])
#     upload_clicked = st.button("Upload Paper")

#     if upload_clicked:
#         if uploaded_file is not None:
#             st.success("Paper uploaded successfully!")
#             st.session_state.rag = RetrievalAugmentedGeneration(model_path="./liver-llm", lora_path="./lora")
#             chunks = st.session_state.rag.extract_text_from_pdf(uploaded_file)
#             index, emb = st.session_state.rag.build_faiss_index(chunks)
#             st.session_state.chunks = chunks
#             st.session_state.index = index
#             st.session_state.embeddings = emb
#         else:
#             st.error("Please upload a paper before clicking upload.")

#     # Choose mode
#     mode = st.radio("Choose mode", options=["Ask Questions", "Continued Learning"])
#     if mode == "Ask Questions":
#         print('MODE: Asking question')
#         rag = RetrievalAugmentedGeneration(uploaded_file_content=uploaded_file)
#         with st.form(key="question_form"):
#             user_question = st.text_input("Ask a question about the paper")
#             submit_question = st.form_submit_button("Submit Question")

#             if submit_question and user_question:
#                 # Replace this with your actual backend response logic
#                 # res = rag.main(query=user_question)
#                 retrieved = st.session_state.rag.retrieve_chunks(user_question, st.session_state.chunks, st.session_state.embeddings, st.session_state.index)
#                 answer = st.session_state.rag.generate_response(user_question, retrieved)
#                 answer = f"📘 {answer}"
#                 st.session_state.qa_history.append((user_question, answer))

#     # Show Q&A History
#     if mode == "Ask Questions" and st.session_state.qa_history:
#         st.markdown("### 🗂️ Q&A History")
#         for q, a in reversed(st.session_state.qa_history):
#             st.markdown(f"**Q:** {q}")
#             st.markdown(f"**A:** {a}")
#             st.markdown("---")

#     elif mode == "Continued Learning":
#         st.text_area("Continue learning notes or reflections here...")
    
def implement_rag():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pdf', type=str, required=True, help='Path to PDF file')
    parser.add_argument('--question', type=str, required=True, help='Question to ask')
    args = parser.parse_args()

    pdf_path = args.pdf
    question = args.question

    try:
        # Initialize RAG with model and LoRA paths
        rag = RetrievalAugmentedGeneration()

        # Extract text from the given PDF file path
        print("[INFO] Extracting text from PDF...")
        chunks = rag.extract_text_from_pdf(pdf_path)

        # Build FAISS index and embeddings
        print("[INFO] Building FAISS index...")
        index, embeddings = rag.build_faiss_index(chunks)

        # Retrieve relevant chunks
        print("[INFO] Retrieving relevant chunks...")
        retrieved_chunks = rag.retrieve_chunks(question, chunks, embeddings, index)

        # Generate response
        print("[INFO] Generating response...")
        answer = rag.generate_response(question, retrieved_chunks)

        print('---------------------------------')

        # Print the final answer
        print("\n🩺 Question:", question)
        print("\n📘Answer:")
        print(answer)
        print('---------------------------------')

    except Exception as e:
        logging.error("An error occurred:")
        traceback.print_exc()

if __name__ == '__main__':
    # create_streamlit_app()
    implement_rag()

