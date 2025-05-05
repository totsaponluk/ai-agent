from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

def get_all_answer(query):
    print("✅ รับคำถาม:", query)

    embedding = OpenAIEmbeddings()
    try:
        db = FAISS.load_local("vector_db/all", embedding, allow_dangerous_deserialization=True)
    except Exception as e:
        print("❌ โหลดเวกเตอร์ไม่สำเร็จ:", str(e))
        return "เกิดข้อผิดพลาดในการโหลดข้อมูลเวกเตอร์"

    retriever = db.as_retriever()
    docs = retriever.get_relevant_documents(query)
    context_text = "\n\n".join([doc.page_content for doc in docs])

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
คุณคือผู้ช่วยอัจฉริยะที่สามารถวิเคราะห์เอกสารทั้งหมดของโครงการ เช่น Requirement, Test, Timeline, และ Resource เพื่อให้คำตอบได้ครอบคลุมทุกบทบาท

Context:
{context}

คำถาม:
{question}

กรุณาตอบคำถามนี้โดยอิงจากข้อมูลทุกมิติ และให้คำแนะนำอย่างครบถ้วนทั้งมุมมองของ BA, PM และ QA
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
