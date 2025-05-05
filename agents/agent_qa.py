# agents/agent_qa.py

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def get_qa_answer(query):
    print("✅ รับคำถาม:", query)

    # Load FAISS Vector DB
    embedding = OpenAIEmbeddings()
    try:
        db = FAISS.load_local("vector_db/qa", embedding, allow_dangerous_deserialization=True)
    except Exception as e:
        print("❌ โหลดเวกเตอร์ไม่สำเร็จ:", str(e))
        return "เกิดข้อผิดพลาดในการโหลดข้อมูลเวกเตอร์"

    retriever = db.as_retriever()
    docs = retriever.get_relevant_documents(query)
    context_text = "\n\n".join([doc.page_content for doc in docs])

    # QA-specific prompt for generating Test Scenario, Test Case, and Test Step
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
คุณคือผู้เชี่ยวชาญด้าน QA ที่มีหน้าที่สร้างเอกสารทดสอบระบบ เช่น Test Scenario, Test Case และ Test Step ตาม requirement ที่กำหนด

Context:
{context}

คำถาม:
{question}

กรุณาตอบกลับโดยสร้างรายการ Test Scenario ที่ชัดเจน พร้อมรายละเอียดของ Test Case และ Test Step แต่ละขั้นตอน เพื่อให้สามารถนำไปทดสอบระบบได้จริง
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
