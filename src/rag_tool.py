from Crypto.Util import Counter
from Crypto.Util import Padding
import  os 
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from src.document_processor import get_embedding_model
from langchain_chroma import Chroma

def get_retriever(persist_directory: str = "./chroma_db",
                 collection_name: str = "knowledge_base",
                 k: int = 4):
    embeddings = get_embedding_model(provider="local")
    vector_db = Chroma(persist_directory=persist_directory,
                       collection_name=collection_name,
                       embedding_function=embeddings)
    return vector_db.as_retriever(search_kwargs={"k": k})

retriever = get_retriever(k = 4)

RAG_CONTEXT_TEMPLATE = PromptTemplate.from_template("""
Dưới đây là các đoạn thông tin trích xuất từ tài liệu nội bộ:
---------------------
{context}
---------------------
Quy tắc:
- Chỉ sử dụng các thông tin có trong phần tài liệu trên để trả lời.
- Nếu tài liệu không chứa đủ dữ liệu để trả lời câu hỏi, hãy nói rõ rằng bạn không tìm thấy thông tin này trong tài liệu.

""")



def search_with_threshold(query: str, k: int = 4, distance_threshold: float = 18.0):
    """
    Tìm kiếm tài liệu bằng khoảng cách Euclidean (càng nhỏ càng giống).
    Khoảng cách < 18.0 là nội dung liên quan.
    """
    embeddings = get_embedding_model(provider="local")
    vector_db = Chroma(
        persist_directory="./chroma_db",
        collection_name="knowledge_base",
        embedding_function=embeddings
    )
    
    results_with_scores = vector_db.similarity_search_with_score(query, k=k)
    filtered_docs = [
        doc for doc, score in results_with_scores 
        if score <= distance_threshold
    ]
    
    return filtered_docs, results_with_scores

