import streamlit as st
import pyodbc 
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import pandas as pd
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Database Connection
def get_db_connection():
    conn = pyodbc.connect(
        f"DRIVER={os.getenv('SQL_DRIVER')};"
        f"SERVER={os.getenv('SQL_SERVER')};"
        f"DATABASE={os.getenv('SQL_DATABASE')};"
        f"UID={os.getenv('SQL_USERNAME')};"
        f"PWD={os.getenv('SQL_PASSWORD')}"
    )
    return conn

# Create the chat history table if it doesn't exist
def initialize_database():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='chat_history' AND xtype='U')
        CREATE TABLE chat_history (
            id int IDENTITY PRIMARY KEY,
            user_id NVARCHAR(255),
            role NVARCHAR(250),
            message NVARCHAR(MAX),
            timestamp DATETIME DEFAULT GETDATE()
        )
    ''')
    conn.commit()
    conn.close()

initialize_database()  # Ensure table is created on startup

# Function to save chat history to SQL Server
def save_chat_history(user_id, chat_history):
    conn = get_db_connection()
    cursor = conn.cursor()
    for message in chat_history:
        cursor.execute(
            "INSERT INTO chat_history (user_id, role, message) VALUES (?, ?, ?)",
            user_id, message["Role"], message["Message"]
        )
    conn.commit()
    conn.close()

# Function to load chat history from SQL Server
def load_chat_history(user_id):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT role, message FROM chat_history WHERE user_id = ? ORDER BY timestamp ASC", (user_id,))
    messages = [{"Role": row[0], "Message": row[1]} for row in cursor.fetchall()]
    conn.close()
    return messages if messages else []

# Function to get PDF text
def get_pdf_text(pdf_directory):
    text = ""
    for pdf_file in os.listdir(pdf_directory):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, pdf_file)
            pdf_reader = PdfReader(pdf_path)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
    return text

# Function to get text chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
    return text_splitter.split_text(text)

# Function to get vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", dimensions=256)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# Function to get conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "Answer is not available in the context."
    
    Context:
    {context}?
    
    Question: 
    {question}
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, streaming=True)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Function to process user input and get a response
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    
    # Perform similarity search
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

def main():
    st.title("Hexa AI Assistant")

    # User login
    user_email = st.text_input("Enter your email to start chat:")
    
    if user_email:
        st.session_state.user_id = user_email
        st.write(f"Welcome, `{st.session_state.user_id}`! How can I assist you?")

        # Load chat history
        if 'messages' not in st.session_state:
            st.session_state.messages = load_chat_history(st.session_state.user_id)

        # Display past chat messages
        for message in st.session_state.messages:
            role = message["Role"].lower()  # Ensure role is lowercase ('user' or 'assistant')
            st.chat_message(role).markdown(message["Message"])

        # User input processing
        prompt = st.chat_input("Ask me anything about the PDF...")
        if prompt:
            st.chat_message('user').markdown(prompt)
            st.session_state.messages.append({'Role': 'user', 'Message': prompt})

            try:
                response = user_input(prompt)
                st.chat_message('assistant').markdown(response)
                st.session_state.messages.append({'Role': 'assistant', 'Message': response})

                # Save chat history to SQL Server
                save_chat_history(st.session_state.user_id, st.session_state.messages)

            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
