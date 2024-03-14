import streamlit as st
import os
import openai
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA

# Load environment variables for OpenAI API key
load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

# Function to initialize and load the PDF document
def init_and_load_pdf(url):
    loader = PyPDFLoader(url)
    pages = loader.load()
    return pages

# Function to embed documents and create a retriever
def create_retriever_and_qa_chain(pages):
    persist_directory = 'db'
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(documents=pages, embedding=embeddings, persist_directory=persist_directory)
    retriever = vectordb.as_retriever()
    qa_chain = RetrievalQA(llm=OpenAI(), retriever=retriever, return_source_documents=True)
    return qa_chain

# Streamlit UI
st.title('Document Query App')

# Load and prepare the PDF document
pdf_url = "https://see.stanford.edu/materials/aimlcs229/transcripts/MachineLearning-Lecture01.pdf"
pages = init_and_load_pdf(pdf_url)
qa_chain = create_retriever_and_qa_chain(pages)

# User query input
user_query = st.text_input("Enter your query about the document:")

if user_query:
    # Process the query and display the answer
    llm_response = qa_chain(user_query)
    answer = llm_response['result']
    st.write("Answer:", answer)

    # Optionally display sources
    st.write("\n\nSources:")
    for source in llm_response["source_documents"]:
        st.write(source.metadata['source'])
