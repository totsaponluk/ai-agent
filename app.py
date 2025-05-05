# app.py

import streamlit as st
import os
from dotenv import load_dotenv
load_dotenv()  # ⬅️ ต้องโหลด .env ก่อน

print("🔐 OpenAI API Key:", os.getenv("OPENAI_API_KEY"))  # ตรวจสอบว่ามีค่า

from agents.agent_ba import get_ba_answer
from agents.agent_pm import get_pm_answer
from agents.agent_qa import get_qa_answer
from agents.agent_all import get_all_answer
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from io import BytesIO
from pypdf.errors import EmptyFileError

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

embedding = OpenAIEmbeddings()

# Utility function
def build_all_vector():
    embedding = OpenAIEmbeddings()

    vector_paths = {
        "ba": "vector_db/ba",
        "pm": "vector_db/pm",
        "qa": "vector_db/qa"
    }

    vector_dbs = []

    for role, path in vector_paths.items():
        index_path = os.path.join(path, "index.faiss")
        if os.path.exists(index_path):
            db = FAISS.load_local(path, embedding, allow_dangerous_deserialization=True)
            vector_dbs.append(db)
        else:
            print(f"⚠️ ข้าม {role.upper()} เพราะไม่มีไฟล์เวกเตอร์: {index_path}")

    if not vector_dbs:
        print("❌ ไม่พบเวกเตอร์ใดๆ ที่จะรวม")
        return

    # เริ่มจาก db แรก
    db_all = vector_dbs[0]
    for db in vector_dbs[1:]:
        db_all.merge_from(db)

    db_all.save_local("vector_db/all")
    print("✅ รวมเวกเตอร์ทั้งหมดเรียบร้อยแล้ว")


def process_files(uploaded_files, agent="ba"):
    folder_path = f"docs/{agent}"
    os.makedirs(folder_path, exist_ok=True)
    all_docs = []

    for file in uploaded_files:
        if hasattr(file, "name") and hasattr(file, "getbuffer"):
            filepath = os.path.join(folder_path, os.path.basename(file.name))
            with open(filepath, "wb") as f:
                f.write(file.getbuffer())
        else:
            filepath = file.name

        if os.path.getsize(filepath) == 0:
            st.warning(f"⚠️ ข้ามไฟล์ว่าง: {filepath}")
            continue

        try:
            if filepath.endswith(".csv"):
                loader = CSVLoader(file_path=filepath)
            else:
                loader = PyPDFLoader(file_path=filepath)
            data = loader.load()
            all_docs.extend(data)
        except EmptyFileError:
            st.error(f"❌ ไม่สามารถอ่านไฟล์ว่างได้: {filepath}")
        except Exception as e:
            st.error(f"❌ โหลดไฟล์ล้มเหลว: {filepath}\n{str(e)}")

    if not all_docs:
        st.warning("⚠️ ไม่มีเอกสารที่สามารถโหลดได้")
        return

    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(all_docs)
    db = FAISS.from_documents(chunks, embedding)
    db.save_local(f"vector_db/{agent}")

# UI
st.set_page_config(page_title="AI Project Assistant", layout="wide")
st.title("🤖 AI Project Assistant")

# Sidebar - Upload
st.sidebar.markdown("## 📁 Upload Documents")

def make_fake_file(path):
    with open(path, "rb") as f:
        content = f.read()
    class FakeFile:
        def __init__(self, name, content):
            self.name = name
            self._content = content
        def getbuffer(self):
            return BytesIO(self._content).getbuffer()
    return FakeFile(os.path.basename(path), content)

with st.sidebar.expander("📄 Upload for BA Agent"):
    uploaded_ba = st.file_uploader("Upload PDF/DOCX for BA", type=["pdf", "docx"], accept_multiple_files=True)
    if uploaded_ba:
        process_files(uploaded_ba, agent="ba")
        st.success("✅ BA documents processed!")

    if st.button("🔁 สร้างเวกเตอร์ BA จากไฟล์ที่มีอยู่"):
        ba_folder = "docs/ba"
        existing_files = [make_fake_file(os.path.join(ba_folder, f)) for f in os.listdir(ba_folder) if f.endswith((".pdf", ".docx", ".csv"))]
        process_files(existing_files, agent="ba")
        st.success("✅ โหลดไฟล์เก่าและสร้างเวกเตอร์ BA สำเร็จแล้ว")

with st.sidebar.expander("📄 Upload for PM Agent"):
    uploaded_pm = st.file_uploader("Upload PDF/CSV for PM", type=["pdf", "csv"], accept_multiple_files=True, key="pm")
    if uploaded_pm:
        process_files(uploaded_pm, agent="pm")
        st.success("✅ PM documents processed!")

    if st.button("🔁 สร้างเวกเตอร์ PM จากไฟล์ที่มีอยู่"):
        pm_folder = "docs/pm"
        existing_files = [make_fake_file(os.path.join(pm_folder, f)) for f in os.listdir(pm_folder) if f.endswith((".pdf", ".docx", ".csv"))]
        process_files(existing_files, agent="pm")
        st.success("✅ โหลดไฟล์เก่าและสร้างเวกเตอร์ PM สำเร็จแล้ว")

with st.sidebar.expander("📄 Upload for QA Agent"):
    uploaded_qa = st.file_uploader("Upload PDF/CSV for QA", type=["pdf", "csv"], accept_multiple_files=True, key="qa")
    if uploaded_qa:
        process_files(uploaded_qa, agent="qa")
        st.success("✅ QA documents processed!")

    if st.button("🔁 สร้างเวกเตอร์ QA จากไฟล์ที่มีอยู่"):
        qa_folder = "docs/qa"
        existing_files = [make_fake_file(os.path.join(qa_folder, f)) for f in os.listdir(qa_folder) if f.endswith((".pdf", ".docx", ".csv"))]
        process_files(existing_files, agent="qa")
        st.success("✅ โหลดไฟล์เก่าและสร้างเวกเตอร์ QA สำเร็จแล้ว")

with st.sidebar.expander("📄 รวมเวกเตอร์ All-in-One Agent"):
    if st.button("🔁 รวมเวกเตอร์ทั้งหมด"):
        try:
            build_all_vector()
            st.success("✅ รวมเวกเตอร์ทั้งหมดสำเร็จแล้ว")
        except Exception as e:
            st.error(f"❌ ไม่สามารถรวมเวกเตอร์: {str(e)}")


# Chat Interface
agent_choice = st.selectbox("เลือก Agent ที่จะถาม:", ["BA", "PM", "QA", "All-in-One"])
query = st.text_input("❓ ถามคำถามที่เกี่ยวกับโปรเจกต์:")

if st.button("ถามเลย") and query:
    with st.spinner("AI กำลังคิด..."):
        try:
            if agent_choice == "BA":
                answer = get_ba_answer(query)
            elif agent_choice == "PM":
                answer = get_pm_answer(query)
            elif agent_choice == "QA":
                answer = get_qa_answer(query)
            else:
                answer = get_all_answer(query)

            st.markdown(f"### 💬 คำตอบ:\n{answer}")
        except Exception as e:
            st.error(f"❌ เกิดข้อผิดพลาด: {str(e)}")
