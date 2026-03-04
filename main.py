import os
import json
import streamlit as st
import pandas as pd

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


# ================= PAGE =================
st.set_page_config(page_title="Multi-Document AI ChatBot", layout="wide")

st.title("Multi-Document AI ChatBot")


# ================= CONSTANTS =================
VECTORSTORE_PATH = "faiss_index"
UPLOAD_DIR = "uploaded_docs"

os.makedirs(UPLOAD_DIR, exist_ok=True)


# ================= EMBEDDINGS =================
@st.cache_resource
def load_embeddings():
    return FastEmbedEmbeddings()

embeddings = load_embeddings()


# ================= LLM =================
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)


# ================= SIDEBAR =================
st.sidebar.header("Upload Data Sources")


# URL INPUT
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:
        urls.append(url)


# WORD
uploaded_docs = st.sidebar.file_uploader(
    "Upload Word Files",
    type=["docx"],
    accept_multiple_files=True
)

# PDF
uploaded_pdfs = st.sidebar.file_uploader(
    "Upload PDFs",
    type=["pdf"],
    accept_multiple_files=True
)

# EXCEL
uploaded_excels = st.sidebar.file_uploader(
    "Upload Excel Files",
    type=["xlsx"],
    accept_multiple_files=True
)

# JSON
uploaded_jsons = st.sidebar.file_uploader(
    "Upload JSON Files",
    type=["json"],
    accept_multiple_files=True
)

process_clicked = st.sidebar.button("Process Data")


# ================= PROCESS DATA =================
if process_clicked:

    documents = []

    with st.spinner("Processing documents..."):

        # ---------- URLS ----------
        for url in urls:
            try:
                docs = WebBaseLoader(url).load()
                for d in docs:
                    d.metadata["source"] = url
                documents.extend(docs)
            except:
                st.warning(f"Failed to load {url}")

        # ---------- WORD ----------
        if uploaded_docs:
            for file in uploaded_docs:

                path = os.path.join(UPLOAD_DIR, file.name)

                with open(path, "wb") as f:
                    f.write(file.getbuffer())

                try:
                    docs = Docx2txtLoader(path).load()
                    documents.extend(docs)
                except:
                    st.warning(f"Failed Word file {file.name}")

        # ---------- PDF ----------
        if uploaded_pdfs:
            for file in uploaded_pdfs:

                path = os.path.join(UPLOAD_DIR, file.name)

                with open(path, "wb") as f:
                    f.write(file.getbuffer())

                try:
                    docs = PyPDFLoader(path).load()
                    documents.extend(docs)
                except:
                    st.warning(f"Failed PDF {file.name}")

        # ---------- EXCEL ----------
        if uploaded_excels:
            for file in uploaded_excels:

                path = os.path.join(UPLOAD_DIR, file.name)

                with open(path, "wb") as f:
                    f.write(file.getbuffer())

                try:
                    df = pd.read_excel(path)

                    for _, row in df.iterrows():
                        text = " ".join([str(x) for x in row])

                        documents.append(
                            Document(page_content=text)
                        )

                except:
                    st.warning(f"Failed Excel {file.name}")

        # ---------- JSON ----------
        if uploaded_jsons:
            for file in uploaded_jsons:

                try:
                    data = json.load(file)

                    if isinstance(data, list):
                        for item in data:
                            documents.append(
                                Document(page_content=json.dumps(item))
                            )
                    else:
                        documents.append(
                            Document(page_content=json.dumps(data))
                        )

                except:
                    st.warning(f"Failed JSON {file.name}")

        if not documents:
            st.error("No valid documents found")
        else:

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100
            )

            docs = splitter.split_documents(documents)

            vectorstore = FAISS.from_documents(
                docs,
                embeddings
            )

            # SAVE FAISS INDEX
            vectorstore.save_local(VECTORSTORE_PATH)

            st.success("Documents processed successfully")


# ================= CHAT =================
query = st.chat_input("Ask something about your documents")

if query:

    if os.path.exists(VECTORSTORE_PATH):

        vectorstore = FAISS.load_local(
            VECTORSTORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True
        )

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever()
        )

        with st.spinner("Thinking..."):

            result = qa({"query": query})

            st.write(result["result"])

    else:

        st.warning("Please process your documents first.")
