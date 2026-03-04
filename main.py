import os
import json
import streamlit as st
import pandas as pd
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import (
    WebBaseLoader,
    Docx2txtLoader,
    PyPDFLoader
)
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings


# LOAD ENV
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("❌ GROQ_API_KEY not found in environment variables.")
    st.stop()


# PAGE CONFIG
st.set_page_config(page_title="AI ChatBot", layout="wide")

st.markdown("""
<style>
#MainMenu {visibility: hidden;}
header {visibility: hidden;}
footer {visibility: hidden;}

.stApp {
    padding-top: 60px;
}

.fixed-top-banner {
    position: fixed;
    top: 0;
    left: 150px;
    right: 0;
    z-index: 9999;
    background: #f8fafc;
    border-bottom: 1px solid #e5e7eb;
    padding: 10px 0;
    text-align: center;
    font-size: 15px;
    color: #333;
    font-weight: 500;
}

.chat-scroll {
    height: calc(100vh - 300px);
    overflow-y: auto;
    padding: 10px 0;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="fixed-top-banner">
    Welcome to Multi-Document AI Chat Bot
</div>
""", unsafe_allow_html=True)


# LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=800,
    groq_api_key=groq_api_key
)


# SIDEBAR
st.sidebar.title("📂 Data Sources")

st.sidebar.subheader("Enter Article URLs")
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    if url:
        urls.append(url)

st.sidebar.subheader("Upload Word Documents")
uploaded_docs = st.sidebar.file_uploader(
    "Upload .docx files",
    type=["docx"],
    accept_multiple_files=True
)

st.sidebar.subheader("Upload PDF Documents")
uploaded_pdfs = st.sidebar.file_uploader(
    "Upload .pdf files",
    type=["pdf"],
    accept_multiple_files=True
)

st.sidebar.subheader("Upload Excel Documents")
uploaded_excels = st.sidebar.file_uploader(
    "Upload .xlsx files",
    type=["xlsx"],
    accept_multiple_files=True
)

st.sidebar.subheader("Upload JSON Documents")
uploaded_jsons = st.sidebar.file_uploader(
    "Upload .json files",
    type=["json"],
    accept_multiple_files=True
)

process_clicked = st.sidebar.button("⚙️ Process Data")


# PROCESS DATA
if process_clicked:
    with st.spinner("Processing documents..."):
        documents = []

        # LOAD URLS
        for url in urls:
            if not url.startswith("http"):
                st.sidebar.warning(f"Invalid URL skipped: {url}")
                continue
            try:
                loader = WebBaseLoader(url)
                loader.requests_kwargs = {
                    "headers": {"User-Agent": "Mozilla/5.0"}
                }
                docs = loader.load()
                for d in docs:
                    d.metadata["source"] = url
                documents.extend(docs)
            except Exception:
                st.sidebar.warning(f"Failed to load URL: {url}")

        # LOAD WORD
        if uploaded_docs:
            for file in uploaded_docs:
                file_path = f"/tmp/{file.name}"
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())

                try:
                    loader = Docx2txtLoader(file_path)
                    docs = loader.load()
                    for d in docs:
                        d.metadata["source"] = file.name
                    documents.extend(docs)
                except Exception:
                    st.sidebar.warning(f"Failed to load Word file: {file.name}")

        # LOAD PDF
        if uploaded_pdfs:
            for file in uploaded_pdfs:
                file_path = f"/tmp/{file.name}"
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())

                try:
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    for d in docs:
                        d.metadata["source"] = file.name
                    documents.extend(docs)
                except Exception:
                    st.sidebar.warning(f"Failed to load PDF file: {file.name}")

        # LOAD EXCEL
        if uploaded_excels:
            for file in uploaded_excels:
                file_path = f"/tmp/{file.name}"
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())

                try:
                    excel_data = pd.read_excel(file_path, sheet_name=None)
                    for sheet_name, df in excel_data.items():
                        df = df.fillna("")
                        for index, row in df.iterrows():
                            row_text = f"Sheet: {sheet_name}\n"
                            for column in df.columns:
                                row_text += f"{column}: {row[column]}\n"

                            documents.append(
                                Document(
                                    page_content=row_text,
                                    metadata={
                                        "source": file.name,
                                        "sheet": sheet_name,
                                        "row": index
                                    }
                                )
                            )
                except Exception:
                    st.sidebar.warning(f"Failed to load Excel file: {file.name}")

        # LOAD JSON
        if uploaded_jsons:
            for file in uploaded_jsons:
                file_path = f"/tmp/{file.name}"
                with open(file_path, "wb") as f:
                    f.write(file.getbuffer())

                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        json_data = json.load(f)

                    if isinstance(json_data, list):
                        for index, item in enumerate(json_data):
                            documents.append(
                                Document(
                                    page_content=json.dumps(item, indent=2),
                                    metadata={
                                        "source": file.name,
                                        "record": index
                                    }
                                )
                            )
                    elif isinstance(json_data, dict):
                        documents.append(
                            Document(
                                page_content=json.dumps(json_data, indent=2),
                                metadata={"source": file.name}
                            )
                        )
                except Exception:
                    st.sidebar.warning(f"Failed to load JSON file: {file.name}")

        # FINAL
        if not documents:
            st.sidebar.error("No valid URLs or documents found.")
        else:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )

            split_docs = splitter.split_documents(documents)

            # LIGHTWEIGHT EMBEDDINGS (NO TORCH)
            embeddings = FastEmbedEmbeddings()

            vectorstore = FAISS.from_documents(split_docs, embeddings)

            st.session_state.vectorstore = vectorstore
            st.sidebar.success("✅ Data processed successfully!")


# CHAT
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask something about your uploaded data...")

if query:
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.markdown(query)

    if "vectorstore" in st.session_state:
        vectorstore = st.session_state.vectorstore

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 6}),
            return_source_documents=False
        )

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = qa({"query": query})
                answer = result["result"]
                st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
    else:
        with st.chat_message("assistant"):
            st.warning("⚠️ Please process URLs or documents first.")



