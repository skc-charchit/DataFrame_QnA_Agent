import os
import streamlit as st
from utility import get_answer

working_dir = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(
    page_title="Document QA",
    layout="centered",
)

st.title("Document QA with Ollama")

upload_file = st.file_uploader(label="Upload a document", type=["pdf"])
user_query = st.text_input("Enter your question")

if st.button("Run QA"):
    if upload_file is not None and user_query:
        bytes_data = upload_file.read()
        file_name = upload_file.name
        
        # Save the file to the working directory
        file_path = os.path.join(working_dir, file_name)
        with open(file_path, "wb") as f:
            f.write(bytes_data)
        
        answer = get_answer(file_name, user_query)
        st.success(answer)
    else:
        st.warning("Please upload a document and enter a question.")
