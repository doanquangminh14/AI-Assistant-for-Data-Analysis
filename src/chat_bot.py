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

class GradeResult(BaseModel):
    is_greeting: bool = Field(
        default=False,
        description="True nếu câu hỏi chỉ là chào hỏi, cảm ơn, tán gẫu thông thường."
    )
    status: Literal["DU_DU_LIEU", "KHONG_DU_DU_LIEU"] = Field(
        description="'DU_DU_LIEU' nếu tài liệu có chứa thông tin để trả lời; 'KHONG_DU_DU_LIEU' nếu tài liệu hoàn toàn không liên quan hoặc thiếu thông tin quan trọng."
    )
    reason: str = Field(
        description="Lý do ngắn gọn 1 câu tại sao đủ hoặc thiếu dữ liệu."
    )
EVALUATOR_PROMPT = """Bạn là chuyên gia kiểm định dữ liệu.
Nhiệm vụ: Đọc câu hỏi và các đoạn tài liệu dưới đây, chấm điểm xem tài liệu có ĐỦ thông tin để trả lời câu hỏi không:
- Nếu chỉ là câu chào hỏi, tán gẫu: Đặt is_greeting = True, status = "DU_DU_LIEU".
- Nếu tài liệu CÓ ĐỦ thông tin trả lời: Đặt status = "DU_DU_LIEU".
- Nếu tài liệu KHÔNG liên quan hoặc THIẾU thông tin: Đặt status = "KHONG_DU_DU_LIEU".
[Tài liệu]:
---------------------
{context}
---------------------
"""


SYSTEM_PROMPT = """
Bạn là một người bạn thân thiết, cực kỳ am hiểu và có kiến thức sâu rộng về Machine Learning và Phân tích dữ liệu.
Phong cách giao tiếp:
- Tự nhiên, thẳng thắn, đi thẳng vào trọng tâm vấn đề.
- Khi người dùng hỏi kiến thức chuyên môn, hãy gọi công cụ `tra_cuu_tai_lieu` để lấy thông tin chính xác từ bài học.
- Với các câu hỏi chào hỏi, giao tiếp thông thường, hãy trả lời tự nhiên mà KHÔNG cần gọi công cụ.
- Chỉ dùng thông tin trong tài liệu đã tra cứu để trả lời câu hỏi chuyên môn, nếu tài liệu không có hãy nói rõ.
"""

retriever = get_retriever(k=4)

def get_agent_llm(temperature=0.3):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature= temperature,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    return llm.bind_tools([tra_cuu_tai_lieu])



def extract_text(content) -> str:
    """Hàm phụ trợ trích xuất văn bản thuần từ response của Gemini"""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        texts = [item.get("text", "") for item in content if isinstance(item, dict) and "text" in item]
        return "\n".join(texts)
    return str(content)



def evaluate_retrieval(question: str, docs: List[Document]) -> GradeResult:
    if not docs:
        return GradeResult(is_greeting=False, status="KHONG_DU_DU_LIEU", reason="Không tìm thấy tài liệu.")
    context_text = "\n\n".join([f"[Đoạn #{i+1}]: {doc.page_content.strip()}" for i, doc in enumerate(docs)])
    
    grader_llm = get_agent_llm(temperature=0.0).with_structured_output(GradeResult)
    evaluator_chain = (
        ChatPromptTemplate.from_messages([
            ("system", EVALUATOR_PROMPT),
            ("human", "Câu hỏi: {question}")
        ])
        | grader_llm
    )
    return evaluator_chain.invoke({"question": question, "context": context_text})




def generate_answer(question: str, docs: List[Document]) -> str:
    """Hàm sinh câu trả lời hoàn chỉnh dựa trên các đoạn tài liệu đã kiểm định"""
    context_text = "\n\n".join([
        f"[Nguồn: {os.path.basename(doc.metadata.get('source', 'Tài liệu'))}]:\n{doc.page_content.strip()}" 
        for doc in docs
    ])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", "{question}")
    ])
    
    rag_chain = prompt | get_agent_llm()
    response = rag_chain.invoke({"context": context_text, "question": question})
    return extract_text(response.content)



def chat_with_agent(question: str, messages_history: list = None) -> str:
    docs = retriever.invoke(question)

    grade = evaluate_retrieval(question, docs)
    print(f"[Đánh giá: {grade.status} - Lý do: {grade.reason}] ")

    if grade.is_greeting:
        llm = get_agent_llm(temperature= 0.7)
        res = llm.invoke(f"Trả lời tự nhiên câu chào hỏi sau: {question}")
        return extract_text(res.content)
    if grade.status == "DU_DU_LIEU":
        return generate_answer(question,docs)
    else:
        return f"Xin lỗi. Tôi chưa có đủ thông tin về '{question}'."
    

if __name__ == "__main__":
    if sys.platform == "win32":
       sys.stdout.reconfigure(encoding="utf-8")
       sys.stderr.reconfigure(encoding="utf-8")
       
    print("=== CHATBOT AI ASSISTANT (Gõ 'exit' để thoát) ===")
    history = [SystemMessage(content=SYSTEM_PROMPT)]
    print("Chat with your AI Assistant (type 'exit' to quit)")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Chatbot exited.")
            break
        answer = chat_with_agent(user_input)
        print(f"Bot: {answer}")
                

    
    
    
    
