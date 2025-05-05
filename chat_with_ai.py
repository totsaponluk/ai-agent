import os
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings()
db = FAISS.load_local("vector_db", embedding)

qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-4", temperature=0),
    retriever=db.as_retriever()
)

while True:
    query = input("คุณอยากถามอะไรจากเอกสาร BRD: ")
    result = qa.run(query)
    print("AI:", result)
