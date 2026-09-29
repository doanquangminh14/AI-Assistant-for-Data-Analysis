import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from src.rag_tool import tra_cuu_tai_lieu

load_dotenv()

SYSTEM_PROMPT = """
Bạn là một người bạn thân thiết, cực kỳ am hiểu và có kiến thức sâu rộng về Machine Learning và Phân tích dữ liệu.
Phong cách giao tiếp:
- Tự nhiên, thẳng thắn, đi thẳng vào trọng tâm vấn đề.
- Khi người dùng hỏi kiến thức chuyên môn, hãy gọi công cụ `tra_cuu_tai_lieu` để lấy thông tin chính xác từ bài học.
- Với các câu hỏi chào hỏi, giao tiếp thông thường, hãy trả lời tự nhiên mà KHÔNG cần gọi công cụ.
- Chỉ dùng thông tin trong tài liệu đã tra cứu để trả lời câu hỏi chuyên môn, nếu tài liệu không có hãy nói rõ.
"""

def get_agent_llm():
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )
    return llm.bind_tools([tra_cuu_tai_lieu])

def chat_with_agent(question: str, messages_history: list = None) -> str:
    llm_with_tools = get_agent_llm()
    if messages_history is None:
        messages_history = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=question)
        ]
    else:
        messages_history.append(HumanMessage(content=question))
    
    ai_msg = llm_with_tools.invoke(messages_history)
    messages_history.append(ai_msg)
    
    if ai_msg.tool_calls:
        for tool_call in ai_msg.tool_calls:
            if tool_call['name'] == "tra_cuu_tai_lieu":
                print(f"Agent muốn sử dụng công cụ tra cứu với nội dung: {tool_call['args']['query']}")

                tool_output = tra_cuu_tai_lieu.invoke(tool_call["args"])
                messages_history.append(ToolMessage(content=tool_output, tool_call_id=tool_call['id']))

        final_response = llm_with_tools.invoke(messages_history)
        messages_history.append(final_response)
        return final_response.content
    else:
        return ai_msg.content


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
        answer = chat_with_agent(user_input,history)
        print(f"Bot: {answer}")
                

    
    
    
    
