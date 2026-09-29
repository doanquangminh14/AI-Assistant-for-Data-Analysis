from Crypto.Util import Counter
from Crypto.Util import Padding
import  os 
from langchain_core.tools import tool
from langchain_core.prompts import PromptTemplate
from src.document_processor import get_embedding_model
from langchain_chroma import Chroma

def get_retriver(persist_directory: str = "./chroma_db",
                 collection_name: str = "knowledge_base",
                 k: int = 4):
    embeddings = get_embedding_model(provider="local")
    vector_db = Chroma(persist_directory=persist_directory,
                       collection_name=collection_name,
                       embedding_function=embeddings)
    return vector_db.as_retriever(search_kwargs={"k": k})

retriever = get_retriver(k = 4)

RAG_CONTEXT_TEMPLATE = PromptTemplate.from_template("""
Dưới đây là các đoạn thông tin trích xuất từ tài liệu nội bộ:
---------------------
{context}
---------------------
Quy tắc:
- Chỉ sử dụng các thông tin có trong phần tài liệu trên để trả lời.
- Nếu tài liệu không chứa đủ dữ liệu để trả lời câu hỏi, hãy nói rõ rằng bạn không tìm thấy thông tin này trong tài liệu.

""")

@tool
def tra_cuu_thong_tin_tu_documents(query: str) -> str:
    docs = retriever.invoke(query)
    if not docs:
        return "Not found relevant documents"
    context_text = "\n\n".join(
        [f"[Trích đoạn #{i+1} - Nguồn: {os.path.basename(doc.metadata.get('source', 'Tài liệu'))}]:\n{doc.page_content.strip()}" 
         for i, doc in enumerate(docs)]
    )

    return RAG_CONTEXT_TEMPLATE.format(context=context_text)


    