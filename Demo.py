import streamlit as st
import google.generativeai as genai
import faiss
import pickle
import os
import numpy as np

def setup_gemini():
    genai.configure(api_key="AIzaSyAOZRzjgX6LSv6FuG3pCmg-kmXJ8guYIdk")
    return genai.GenerativeModel('gemini-pro')

def load_data():
    TEMP_DIR = "temp_data" # Update this path as needed
    index = faiss.read_index(os.path.join(TEMP_DIR, "faiss_index.bin"))
    with open(os.path.join(TEMP_DIR, "texts.pkl"), "rb") as f:
        texts = pickle.load(f)
    return index, texts


def reconstruct_document(chunks):
    """
    Reconstruct the document from overlapping chunks
    
    Args:
        chunks (list): List of text chunks
        
    Returns:
        str: Reconstructed document text
    """
    if not chunks:
        return ""
    
    # Simple concatenation - you might want to implement more sophisticated 
    # deduplication for overlapping content
    return "\n".join(chunks)

# Example usage:
def retrieve_pdf_content(query=None):
    """
    Retrieve PDF content, optionally filtered by a search query
    
    Args:
        query (str, optional): Search query to filter content
        
    Returns:
        str: Retrieved PDF content
    """
    # Load saved data
    index, texts = load_data()
    
    if query:
        # If query is provided, return relevant chunks
        return texts
    else:
        # If no query, return full document
        return reconstruct_document(texts)
    
full_content = retrieve_pdf_content()
model = setup_gemini()

st.title("AI-Powered Task Execution")

task = st.text_area("Enter the task you want to perform:")
student_answer = st.text_area("Enter the student's answer (if applicable):")

if st.button("Execute Task"):
    
    reference_answer=full_content
    task_prompt = f"Generate a prompt to perform the following task: {task}\nReference: {reference_answer}\n"
    generated_prompt = model.generate_content(task_prompt).text

    if student_answer:
        generated_prompt += f"Reference: {reference_answer}\n\nData to compare: {student_answer}\n\n"

    response = model.generate_content(generated_prompt)
    
    st.subheader("AI Response:")
    st.write(response.text)