import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from src.rag_tool import tra_cuu_tai_lieu
from typing import List, Literal
from pydantic import BaseModel, Field
from langchain_core.documents import Document
from src.rag_tool import get_retriever

load_dotenv()

class RAGResponse(BaseModel):
    is_greeting: bool = Field(
        default=False,
        description="True nếu câu hỏi chỉ là chào hỏi, cảm ơn, tán gẫu thông thường."
    )
    status: Literal["DU_DU_LIEU", "KHONG_DU_DU_LIEU"] = Field(
        description="'DU_DU_LIEU' nếu tài liệu có đủ thông tin để trả lời; 'KHONG_DU_DU_LIEU' nếu tài liệu thiếu hoặc không liên quan."
    )
    reason: str = Field(
        description="Lý do ngắn gọn 1 câu tại sao đủ hoặc thiếu dữ liệu."
    )
    answer: str = Field(
        description="Nội dung câu trả lời tự nhiên, thẳng thắn theo persona dựa vào tài liệu (nếu DU_DU_LIEU), hoặc lời chào thân mật (nếu is_greeting), hoặc giải thích rõ là tài liệu chưa có thông tin này (nếu KHONG_DU_DU_LIEU)."
    )
    sources: List[str] = Field(
        default=[],
        description="Danh sách các file nguồn đã sử dụng để trả lời."
    )


SYSTEM_PROMPT = """Bạn là một người bạn thân thiết, cực kỳ am hiểu và có kiến thức sâu rộng về Machine Learning và Phân tích dữ liệu.
Phong cách giao tiếp:
- Tự nhiên, thẳng thắn, không màu mè sáo rỗng, giải thích dễ hiểu.
Nhiệm vụ của bạn trong 1 lần xử lý:
1. Nếu câu hỏi là chào hỏi, cảm ơn, tán gẫu: Đặt is_greeting = True, status = "DU_DU_LIEU", và viết câu trả lời thân thiện vào `answer`.
2. Nếu là câu hỏi kiến thức:
   - Đọc kỹ phần [Tài liệu nội bộ] được cung cấp dưới đây.
   - Nếu tài liệu CÓ ĐỦ thông tin: Đặt status = "DU_DU_LIEU", viết câu trả lời chi tiết vào `answer`, kèm danh sách `sources`.
   - Nếu tài liệu KHÔNG ĐỦ hoặc KHÔNG LIÊN QUAN: Đặt status = "KHONG_DU_DU_LIEU", nói rõ trong `answer` rằng tài liệu nội bộ chưa có thông tin này. TUYỆT ĐỐI không tự ý bịa đặt kiến thức ngoài tài liệu.
[Tài liệu nội bộ]:
---------------------
{context}
---------------------
"""

retriever = get_retriever(k=4)

def get_rag_chain():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        temperature= 0.2,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    structured_llm = llm.with_structured_output(RAGResponse)
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "Câu hỏi: {question}")
    ])
    return prompt | structured_llm
    
rag_chain = get_rag_chain()


def extract_text(content) -> str:
    """Hàm phụ trợ trích xuất văn bản thuần từ response của Gemini"""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        texts = [item.get("text", "") for item in content if isinstance(item, dict) and "text" in item]
        return "\n".join(texts)
    return str(content)


def chat_with_agent(question: str) -> RAGResponse:
    docs: List[Document] = retriever.invoke(question)
    context_text = "\n\n".join([
        f"[Đoạn #{i+1} - Nguồn: {os.path.basename(doc.metadata.get('source', 'Tài liệu'))}]:\n{doc.page_content.strip()}" 
        for i, doc in enumerate(docs)
    ])
    response: RAGResponse = rag_chain.invoke({
        "context": context_text,
        "question": question
    })
    return response

    

if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
       
    print("=== CHATBOT AI ASSISTANT - GỘP KIỂM ĐỊNH & TRẢ LỜI ===")
    print("Chat with your AI Assistant (type 'exit' to quit)\n")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Chatbot exited.")
            break
        if not user_input:
            continue
            
        res = chat_with_agent(user_input)
        if not res.is_greeting:
            print(f" [Đánh giá: {res.status} | Lý do: {res.reason}]")
            
        print(f"Bot: {res.answer}")
        
        if res.sources:
            print(f" Nguồn: {', '.join(res.sources)}")
        print("-" * 60)


    
    
    
    
