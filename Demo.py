import streamlit as st
import google.generativeai as genai
import faiss
import pickle
import os
import numpy as np
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, firestore

def setup_gemini():
    genai.configure(api_key="AIzaSyAOZRzjgX6LSv6FuG3pCmg-kmXJ8guYIdk")
    return genai.GenerativeModel('gemini-pro')

def load_data():
    TEMP_DIR = "temp_data"  # Update this path as needed
    index = faiss.read_index(os.path.join(TEMP_DIR, "faiss_index.bin"))
    with open(os.path.join(TEMP_DIR, "texts.pkl"), "rb") as f:
        texts = pickle.load(f)
    return index, texts

def reconstruct_document(chunks):
    if not chunks:
        return ""
    return "\n".join(chunks)

def initialize_firebase():
    """Initialize Firebase Admin SDK if not already initialized"""
    if not firebase_admin._apps:
        cred = credentials.Certificate("Credentials.json")
        firebase_admin.initialize_app(cred)
    return firestore.client()

def fetch_student_answers():
    """
    Fetch all student answers from Firestore
    
    Returns:
        dict: Dictionary mapping student IDs to their answers
    """
    db = initialize_firebase()
    student_answers = {}
    
    # Get all documents from the 'answers' collection
    answers_ref = db.collection('answers').stream()
    
    for doc in answers_ref:
        data = doc.to_dict()
        student_id = data.get('student_id')
        answer = data.get('answer')
        if student_id and answer:
            student_answers[student_id] = answer
            
    return student_answers

def process_all_submissions(reference_content, student_answers, task, model):
    """
    Process all student submissions and generate results
    
    Args:
        reference_content (str): Reference content
        student_answers (dict): Dictionary of student answers
        task (str): Task description
        model: Gemini model instance
        
    Returns:
        dict: Results for each submission
    """
    results = {}
    
    for student_id, answer in student_answers.items():
        # Generate task-specific response
        task_prompt = f"Generate a prompt to perform the following task: {task}\nReference: {reference_content}\n"
        generated_prompt = model.generate_content(task_prompt).text
        generated_prompt += f"Perform the task specified between Reference and \nData to compare: {answer}."
        response1 = model.generate_content(generated_prompt)
        
        # Generate similarity metrics
        similarity_prompt = f"Use metrics like Cosine Similarity and BLEU score and perform the similarity analysis between Reference: {reference_content} and Data to Compare: {answer}. Just return cosine similarity and BLEU scores. Nothing more.\n\n"
        response2 = model.generate_content(similarity_prompt)
        
        # Store results in Firestore
        db = initialize_firebase()
        results_ref = db.collection('evaluation_results').document()
        results_ref.set({
            'student_id': student_id,
            'task_response': response1.text,
            'similarity_metrics': response2.text,
            'timestamp': datetime.now(),
            'task': task
        })
        
        results[student_id] = {
            'task_response': response1.text,
            'similarity_metrics': response2.text
        }
    
    return results

# Streamlit UI
st.title("AI-Powered Batch Task Evaluation")

# Input field for task
task = st.text_area("Enter the task you want to perform:")

if st.button("Process All Submissions"):
    if task:
        try:
            # Setup and load data
            model = setup_gemini()
            reference_content = reconstruct_document(load_data()[1])
            
            # Fetch all student answers from Firestore
            student_answers = fetch_student_answers()
            
            if not student_answers:
                st.error("No student answers found in the database.")
            else:
                # Process all submissions
                results = process_all_submissions(reference_content, student_answers, task, model)
                
                # Display results
                st.subheader("Results:")
                for student_id, result in results.items():
                    st.write(f"### Student ID: {student_id}")
                    st.write("Task Analysis:")
                    st.write(result['task_response'])
                    st.write("Similarity Metrics:")
                    st.write(result['similarity_metrics'])
                    st.divider()
                
                # Option to download results
                if st.button("Download Results"):
                    result_text = "\n\n".join([
                        f"Student ID: {student_id}\n\nTask Analysis:\n{result['task_response']}\n\nSimilarity Metrics:\n{result['similarity_metrics']}"
                        for student_id, result in results.items()
                    ])
                    st.download_button(
                        label="Download Results as Text",
                        data=result_text,
                        file_name="batch_results.txt",
                        mime="text/plain"
                    )
                    
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
    else:
        st.warning("Please provide a task description.")