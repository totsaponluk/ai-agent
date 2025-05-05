import os
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()
embedding = OpenAIEmbeddings()

def create_vector_db(file_path="docs/BRD_auction_system.pdf"):
    loader = UnstructuredPDFLoader(file_path)
    data = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_documents(data)

    db = FAISS.from_documents(chunks, embedding)
    db.save_local("vector_db")

if __name__ == "__main__":
    create_vector_db()
