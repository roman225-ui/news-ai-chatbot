import os
import json
import gc
import requests
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from bs4 import BeautifulSoup

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings

from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader
)

# -------------------
# ENV
# -------------------

load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error("❌ GROQ_API_KEY not found")
    st.stop()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# -------------------
# PAGE
# -------------------

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

# -------------------
# LLM
# -------------------

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=800,
    groq_api_key=groq_api_key
)

# -------------------
# SIDEBAR
# -------------------

st.sidebar.title("📂 Data Sources")
st.sidebar.subheader("Enter Article URLs")

urls = []

for i in range(2):
    u = st.sidebar.text_input(f"URL {i+1}")
    if u:
        urls.append(u)

uploaded_docs = st.sidebar.file_uploader(
    "Upload Word (.docx)",
    type=["docx"],
    accept_multiple_files=True
)

uploaded_pdfs = st.sidebar.file_uploader(
    "Upload PDF",
    type=["pdf"],
    accept_multiple_files=True
)

uploaded_excels = st.sidebar.file_uploader(
    "Upload Excel",
    type=["xlsx"],
    accept_multiple_files=True
)

uploaded_jsons = st.sidebar.file_uploader(
    "Upload JSON",
    type=["json"],
    accept_multiple_files=True
)

process_clicked = st.sidebar.button("⚙️ Process Data")

# -------------------
# PROCESS DATA
# -------------------

if process_clicked:

    documents = []

    with st.spinner("Processing documents..."):

        # -------- URLs --------

        for url in urls:

            try:

                st.sidebar.write(f"Loading {url}")

                headers = {"User-Agent": "Mozilla/5.0"}

                response = requests.get(url, headers=headers, timeout=10)

                soup = BeautifulSoup(response.text, "html.parser")

                text = soup.get_text(separator=" ", strip=True)

                text = text[:10000]  # limit memory

                documents.append(
                    Document(
                        page_content=text,
                        metadata={"source": url}
                    )
                )

            except Exception:
                st.sidebar.warning(f"Failed: {url}")

        # -------- WORD --------

        if uploaded_docs:

            for file in uploaded_docs[:2]:

                path = f"/tmp/{file.name}"

                with open(path, "wb") as f:
                    f.write(file.getbuffer())

                try:

                    loader = Docx2txtLoader(path)
                    docs = loader.load()

                    documents.extend(docs)

                except:
                    st.sidebar.warning(f"Failed {file.name}")

        # -------- PDF --------

        if uploaded_pdfs:

            for file in uploaded_pdfs[:2]:

                path = f"/tmp/{file.name}"

                with open(path, "wb") as f:
                    f.write(file.getbuffer())

                try:

                    loader = PyPDFLoader(path)
                    docs = loader.load()

                    documents.extend(docs)

                except:
                    st.sidebar.warning(f"Failed {file.name}")

        # -------- EXCEL --------

              if uploaded_excels:

            for file in uploaded_excels[:1]:

                path = f"/tmp/{file.name}"

                with open(path, "wb") as f:
                    f.write(file.getbuffer())

                try:

                    excel_data = pd.read_excel(path, sheet_name=None)

                    for sheet_name, df in excel_data.items():

                        df = df.fillna("")
                        df.columns = df.columns.str.strip()

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

        # -------- JSON --------

        if uploaded_jsons:

            for file in uploaded_jsons[:1]:

                path = f"/tmp/{file.name}"

                with open(path, "wb") as f:
                    f.write(file.getbuffer())

                try:

                    with open(path) as f:
                        data = json.load(f)

                    text = json.dumps(data)

                    text = text[:8000]

                    documents.append(
                        Document(
                            page_content=text,
                            metadata={"source": file.name}
                        )
                    )

                except:
                    st.sidebar.warning(f"Failed {file.name}")

        # -------- LIMIT DOCS --------

        documents = documents[:5]

        if not documents:

            st.sidebar.error("No documents loaded")

        else:

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=300,
                chunk_overlap=50
            )

            split_docs = splitter.split_documents(documents)

            split_docs = split_docs[:100]

            embeddings = FastEmbedEmbeddings(
                model_name="BAAI/bge-small-en-v1.5"
            )

            vectorstore = FAISS.from_documents(
                split_docs,
                embeddings
            )

            st.session_state.vectorstore = vectorstore

            gc.collect()

            st.sidebar.success("✅ Documents processed")

# -------------------
# CHAT
# -------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:

    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask about your documents")

if query:

    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

    with st.chat_message("user"):
        st.markdown(query)

    if "vectorstore" in st.session_state:

        retriever = st.session_state.vectorstore.as_retriever(
            search_kwargs={"k": 3}
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever
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

        st.warning("⚠️ Please process documents first.")




