import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter ,CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
import os

# Load environment variables
load_dotenv(dotenv_path='.env')

# Initialize session state variables
if "conversational" not in st.session_state:
    st.session_state.conversational = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = None

def get_pdf_text(pdf):
    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

def get_chunks(text):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200, length_function=len)
    chunks = text_splitter.split_text(text=text)
    return chunks

def get_vectorize(text_chunks):
    embeddings = HuggingFaceInstructEmbeddings(model_name='hkunlp/instructor-base')
    # embeddings = OpenAIEmbeddings() # if you have API access to OpenAI
    vectorstore = FAISS.from_texts(text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorized_text):
    # llm = ChatOpenAI()  # Uncomment if using OpenAI's LLM
    llm = HuggingFaceHub(repo_id='google/flan-t5-small', model_kwargs={'temperature': 0.7, 'max_length': 512},
                         huggingfacehub_api_token='enter your api')
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorized_text.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversational({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    st.write(css, unsafe_allow_html=True)
    
    st.header("Chat with PDF")
    user_question = st.text_input("Write your question:")
    
    if user_question:
        handle_userinput(user_question)
    
    with st.sidebar:
        pdf = st.file_uploader("Upload your PDF", type="pdf")
        if st.button('Process'):
            with st.spinner('Processing...'):
                raw_text = get_pdf_text(pdf)
                chunks = get_chunks(raw_text)
                vector = get_vectorize(chunks)
                st.session_state.conversational = get_conversation_chain(vector)

if __name__ == "__main__":
    main()
