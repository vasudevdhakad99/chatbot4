import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
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

def get_pdf_text(pdf_directory):
    text = ""
    for pdf_file in os.listdir(pdf_directory):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join(pdf_directory, pdf_file)
            pdf_reader = PdfReader(pdf_path)
            for page in pdf_reader.pages:
                text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
    return text_splitter.split_text(text)


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",dimensions=256)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    
    return vector_store

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
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3,streaming =True)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

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
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        st.chat_message(message['role']).markdown(message['content'])
    
    prompt = st.chat_input("Ask me anything about the PDF...")
    
    if prompt:
        st.chat_message('user').markdown(prompt)
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        
        try:
            response = user_input(prompt)
            st.chat_message('assistant').markdown(response)
            st.session_state.messages.append({'role': 'assistant', 'content': response})
        except Exception as e:
            st.error(f"Error: {str(e)}")
    
    with st.sidebar:
        st.title("Menu:")
        if st.button("Load and Process PDFs"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(PDF_DIRECTORY)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("PDFs Processed Successfully!")

if __name__ == "__main__":
    main()

