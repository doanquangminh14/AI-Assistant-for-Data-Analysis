from torch._C._instruction_counter import end
import os
from typing import List
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

def load_documents(file_path: str) -> List[Document]:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in [".docx", ".doc"]:
        loader = Docx2txtLoader(file_path)
    elif ext in [".txt", ".md", "markdown"]:
        loader = TextLoader(file_path, encoding="utf-8") 
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    docs = loader.load()
    return docs
     
def split_documents(docs: List[Document], 
                    chunk_size: int = 1000,
                    chunk_overlap: int = 200) -> List[Document]:
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " ", ""],
        is_separator_regex=False
    )
    chunks = text_splitter.split_documents(docs)
    return chunks

def get_embedding_model(provider: str = "google"):
    if provider == "google":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found")
        return GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=api_key
        )
    elif provider == "local":
<<<<<<< HEAD
        return HuggingFaceEmbeddings(model_name = "all-MiniLM-L6-v2")
=======
        return HuggingFaceEmbeddings(model_name = "paraphrase-multilingual-MiniLM-L12-v2")
>>>>>>> db3fdd4 (update document processor and test, ignore chroma_db)
    else:
        raise ValueError(f"Unsupported embedding provider: {provider}")

def store_in_vector_db(chunks: List[Document],
                       persist_directory: str = "./chroma_db",
                       collection_name: str = "knowledge_base",
                       embedding_provider: str = "google") -> Chroma:
    embeddings = get_embedding_model(provider = embedding_provider)
    vector_db = Chroma.from_documents(
        documents = chunks,
        embedding = embeddings,
        collection_name = collection_name,
        persist_directory = persist_directory
        )
    return vector_db
    

                       
