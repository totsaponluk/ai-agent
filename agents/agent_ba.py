from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def get_ba_answer(query):
    print("✅ รับคำถาม:", query)

    embedding = OpenAIEmbeddings()
    try:
        db = FAISS.load_local("vector_db/ba", embedding, allow_dangerous_deserialization=True)
    except Exception as e:
        print("❌ โหลดเวกเตอร์ไม่สำเร็จ:", str(e))
        return "เกิดข้อผิดพลาดในการโหลดข้อมูลเวกเตอร์"

    retriever = db.as_retriever()
    docs = retriever.get_relevant_documents(query)
    context_text = "\n\n".join([doc.page_content for doc in docs])

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
คุณคือ Business Analyst (BA) ที่มีหน้าที่วิเคราะห์ระบบ เขียน Use Case, BRD, และสรุป Requirement ต่าง ๆ

Context:
{context}

คำถาม:
{question}

กรุณาตอบคำถามโดยอิงจากเอกสารที่ให้มา และหากเกี่ยวข้องให้จัดทำ Use Case หรือ Requirement ที่เหมาะสม
"""
    )

    llm = ChatOpenAI(temperature=0, model="gpt-4o")
    chain = LLMChain(llm=llm, prompt=prompt)

    try:
        result = chain.invoke({"context": context_text, "question": query})
        return result["text"]
    except Exception as e:
        print("❌ Error ขณะสร้างคำตอบ:", str(e))
        return "เกิดข้อผิดพลาดขณะสร้างคำตอบ"
