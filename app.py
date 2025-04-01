import streamlit as st
from PyPDF2 import PdfReader
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

# File paths
PDF_DIRECTORY = "E:\\chatbot"  # Update with your PDF directory
CHAT_HISTORY_FILE = "chat_history.xlsx"  # Excel file for chat storage

# Ensure chat history file exists
if not os.path.exists(CHAT_HISTORY_FILE):
    df = pd.DataFrame(columns=["User ID", "Role", "Message"])
    df.to_excel(CHAT_HISTORY_FILE, index=False)

# Function to save chat history
def save_chat_history(user_id, chat_history):
    df = pd.read_excel(CHAT_HISTORY_FILE)
    new_data = pd.DataFrame([
        {"User ID": user_id, "Role": message["Role"], "Message": message["Message"]}
        for message in chat_history
    ])
    df = pd.concat([df, new_data], ignore_index=True)
    df.to_excel(CHAT_HISTORY_FILE, index=False)

# Function to load chat history
def load_chat_history(user_id):
    df = pd.read_excel(CHAT_HISTORY_FILE)
    user_messages = df[df["User ID"] == user_id][["Role", "Message"]].to_dict("records")
    return user_messages if not user_messages else []

# Function to extract text from PDFs
def get_pdf_text(pdf_directory):
    text = ""
    for pdf_file in os.listdir(pdf_directory):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, pdf_file)
            pdf_reader = PdfReader(pdf_path)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
    return text

# Function to create and store vector embeddings
def get_vector_store():
    raw_text = get_pdf_text(PDF_DIRECTORY)
    text_chunks = raw_text.split("\n")  # Simple split (or use RecursiveCharacterTextSplitter)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", dimensions=256)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# Load or create vector store
def load_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    if os.path.exists("faiss_index"):
        return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    else:
        return get_vector_store()

# Function to set up the conversational AI chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "Answer is not available in the context."
    
    Context:
    {context}
    
    Question: 
    {question}
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, streaming=True)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Function to handle user input
def user_input(user_question, vector_store):
    docs = vector_store.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

# Main Streamlit App
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

        # Load vector store
        vector_store = load_vector_store()

        # User input processing
        prompt = st.chat_input("Ask me anything about the PDF...")
        if prompt:
            st.chat_message('user').markdown(prompt)
            st.session_state.messages.append({'Role': 'user', 'Message': prompt})

            try:
                response = user_input(prompt, vector_store)
                st.chat_message('assistant').markdown(response)
                st.session_state.messages.append({'Role': 'assistant', 'Message': response})

                # Save chat history to Excel
                save_chat_history(st.session_state.user_id, st.session_state.messages)

            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()
