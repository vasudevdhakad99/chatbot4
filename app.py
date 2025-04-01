import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import pandas as pd
import tempfile
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

PDF_DIRECTORY = "E:\\chatbot"  # Change this to your desired directory E:\chatbot\Hexa_CompanyprofileCM.pdf

# Use a temporary file for chat history
def create_temp_excel_file():
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.xlsx')
    return temp_file.name

# Function to save chat history
def save_chat_history(user_id, chat_history):
    # Create a temporary file
    temp_file_path = create_temp_excel_file()
    df = pd.read_excel(temp_file_path) if os.path.exists(temp_file_path) else pd.DataFrame(columns=["User ID", "Role", "Message"])
    
    new_data = pd.DataFrame([
        {"User ID": user_id, "Role": message["Role"], "Message": message["Message"]}
        for message in chat_history
    ])
    
    df = pd.concat([df, new_data], ignore_index=True)
    df.to_excel(temp_file_path, index=False)

# Function to load chat history
def load_chat_history(user_id):
    # Use the temporary file path to load the history
    temp_file_path = create_temp_excel_file()
    if os.path.exists(temp_file_path):
        df = pd.read_excel(temp_file_path)
        user_messages = df[df["User ID"] == user_id][["Role", "Message"]].to_dict("records")
        return user_messages if user_messages else []
    return []

def get_pdf_text(pdf_directory):
    text = ""
    for pdf_file in os.listdir(pdf_directory):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, pdf_file)
            pdf_reader = PdfReader(pdf_path)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
    return text

def get
