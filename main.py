import os
import pickle
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
from langchain_community.embeddings import HuggingFaceEmbeddings


# ================= PAGE CONFIG =================
st.set_page_config(page_title="AI ChatBot", layout="wide")


# ================= CUSTOM CSS =================
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
}

.chat-scroll {
    height: calc(100vh - 300px);
    overflow-y: auto;
}
</style>
""", unsafe_allow_html=True)


# ================= LOAD ENV =================
load_dotenv()


# ================= HEADER =================
st.markdown("""
<div class="fixed-top-banner">
Welcome to Multi-Document AI Chat Bot
</div>
""", unsafe_allow_html=True)


# ================= SIDEBAR =================
st.sidebar.title("📂 Data Sources")

# URL INPUT
st.sidebar.subheader("Enter Article URLs")
urls = []

for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    if url:
        urls.append(url)


# WORD UPLOAD
st.sidebar.subheader("Upload Word Documents")
uploaded_docs = st.sidebar.file_uploader(
    "Upload .docx files",
    type=["docx"],
    accept_multiple_files=True
)

# PDF UPLOAD
st.sidebar.subheader("Upload PDF Documents")
uploaded_pdfs = st.sidebar.file_uploader(
    "Upload .pdf files",
    type=["pdf"],
    accept_multiple_files=True
)

# EXCEL UPLOAD
st.sidebar.subheader("Upload Excel Documents")
uploaded_excels = st.sidebar.file_uploader(
    "Upload .xlsx files",
    type=["xlsx"],
    accept_multiple_files=True
)

# JSON UPLOAD
st.sidebar.subheader("Upload JSON Documents")
uploaded_jsons = st.sidebar.file_uploader(
    "Upload .json files",
    type=["json"],
    accept_multiple_files=True
)

process_clicked = st.sidebar.button("⚙️ Process Data")


# ================= CONSTANTS =================
VECTORSTORE_PATH = "faiss_store.pkl"
UPLOAD_DIR = "uploaded_docs"

os.makedirs(UPLOAD_DIR, exist_ok=True)


# ================= LLM =================
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0,
    max_tokens=800
)


# ================= EMBEDDING CACHE =================
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )


# ================= PROCESS DOCUMENTS =================
if process_clicked:

    with st.spinner("Processing documents..."):

        documents = []

        # ----- LOAD URLS -----
        for url in urls:
            if not url.startswith("http"):
                st.sidebar.warning(f"Invalid URL skipped: {url}")
                continue

            try:
                docs = WebBaseLoader(url).load()

                for d in docs:
                    d.metadata["source"] = url

                documents.extend(docs)

            except Exception as e:
                st.sidebar.warning(f"Failed to load URL: {url}")


        # ----- WORD FILES -----
        if uploaded_docs:

            for file in uploaded_docs:

                path = os.path.join(UPLOAD_DIR, file.name)

                with open(path, "wb") as f:
                    f.write(file.getbuffer())

                try:
                    docs = Docx2txtLoader(path).load()

                    for d in docs:
                        d.metadata["source"] = file.name

                    documents.extend(docs)

                except:
                    st.sidebar.warning(f"Failed to load {file.name}")


        # ----- PDF FILES -----
        if uploaded_pdfs:

            for file in uploaded_pdfs:

                path = os.path.join(UPLOAD_DIR, file.name)

                with open(path, "wb") as f:
                    f.write(file.getbuffer())

                try:
                    docs = PyPDFLoader(path).load()

                    for d in docs:
                        d.metadata["source"] = file.name

                    documents.extend(docs)

                except:
                    st.sidebar.warning(f"Failed to load {file.name}")


        # ----- EXCEL FILES -----
        if uploaded_excels:

            for file in uploaded_excels:

                path = os.path.join(UPLOAD_DIR, file.name)

                with open(path, "wb") as f:
                    f.write(file.getbuffer())

                try:

                    excel_data = pd.read_excel(path, sheet_name=None)

                    for sheet, df in excel_data.items():

                        df = df.fillna("")

                        for i, row in df.iterrows():

                            text = f"Sheet: {sheet}\n"

                            for col in df.columns:
                                text += f"{col}: {row[col]}\n"

                            documents.append(
                                Document(
                                    page_content=text,
                                    metadata={
                                        "source": file.name,
                                        "sheet": sheet
                                    }
                                )
                            )

                except:
                    st.sidebar.warning(f"Excel load failed: {file.name}")


        # ----- JSON FILES -----
        if uploaded_jsons:

            for file in uploaded_jsons:

                path = os.path.join(UPLOAD_DIR, file.name)

                with open(path, "wb") as f:
                    f.write(file.getbuffer())

                try:

                    with open(path) as f:
                        data = json.load(f)

                    if isinstance(data, list):

                        for i, item in enumerate(data):

                            documents.append(
                                Document(
                                    page_content=json.dumps(item, indent=2),
                                    metadata={
                                        "source": file.name,
                                        "record": i
                                    }
                                )
                            )

                    else:

                        documents.append(
                            Document(
                                page_content=json.dumps(data, indent=2),
                                metadata={"source": file.name}
                            )
                        )

                except:
                    st.sidebar.warning(f"JSON load failed: {file.name}")


        # ----- CHECK DOCUMENTS -----
        if not documents:

            st.sidebar.error("No valid documents found")

        else:

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=100
            )

            split_docs = splitter.split_documents(documents)

            embeddings = load_embeddings()

            vectorstore = FAISS.from_documents(split_docs, embeddings)

            with open(VECTORSTORE_PATH, "wb") as f:
                pickle.dump(vectorstore, f)

            st.sidebar.success("Data processed successfully!")


# ================= CHAT HISTORY =================
if "messages" not in st.session_state:
    st.session_state.messages = []


chat_container = st.container()

for msg in st.session_state.messages:

    with chat_container.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ================= CHAT INPUT =================
query = st.chat_input("Ask something about your data...")


if query:

    st.session_state.messages.append({"role": "user", "content": query})

    with chat_container.chat_message("user"):
        st.markdown(query)


    if os.path.exists(VECTORSTORE_PATH):

        with open(VECTORSTORE_PATH, "rb") as f:
            vectorstore = pickle.load(f)

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vectorstore.as_retriever(search_kwargs={"k": 6}),
            return_source_documents=True
        )

        with chat_container.chat_message("assistant"):

            with st.spinner("Thinking..."):

                result = qa({"query": query})

                answer = result["result"]

                st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )

    else:

        with chat_container.chat_message("assistant"):
            st.warning("Please process documents first")
