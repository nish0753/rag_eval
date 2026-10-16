import os
import re
import glob

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

load_dotenv()  # loads OPENAI_API_KEY from .env

DATA_DIR = "data"
DB_DIR = "chroma_store"


# 1. LOAD ---- read each transcript, throw away the VTT timestamps
def load_transcripts():

    docs = []
    for path in glob.glob(f"{DATA_DIR}/*.vtt"):
        lines = []
        for line in open(path):
            line = line.strip()
            if not line or line == "WEBVTT" or "-->" in line:
                continue
            lines.append(line)
        text = " ".join(lines)

        session = re.search(r"Session[ _]*(\d+)", path).group(1)

        docs.append(Document(page_content=text, metadata={"session": session}))

    return docs


# 2. BUILD ---- chunk, embed once, and keep it on disk so we don't re-embed
def load_store():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    if os.path.exists(DB_DIR):
        print(f"⚡ Loading existing Chroma store from '{DB_DIR}'...")
        return Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

    print("📂 Reading transcripts from data/...")
    docs = load_transcripts()
    print(f"📄 Loaded {len(docs)} transcripts.")

    print("✂️  Splitting transcripts into chunks...")
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
    ).split_documents(docs)
    print(f"🧠 Generating OpenAI embeddings for {len(chunks)} chunks and saving to disk (one-time initial build, takes ~1 min)...")

    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory=DB_DIR)
    print("✅ Vector store created and saved successfully!")
    return vectorstore


def build_retriever():
    return load_store().as_retriever(search_kwargs={"k": 5})


# 3. TRY IT ---- python src/retriever.py
if __name__ == "__main__": 

    retriever = build_retriever()

    results = retriever.invoke("what is regression testing?")
    
    for r in results:
        print(f"[Session {r.metadata['session']}] {r.page_content[:150]}...\n")