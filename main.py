import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

SYSTEM_PROMPT = """
Bạn là một người bạn thân thiết, cực kỳ am hiểu và có kiến thức sâu rộng.
Phong cách giao tiếp:
- Tự nhiên, gần gũi, không máy móc.
- Thẳng thắn, đi thẳng vào trọng tâm vấn đề, chỉ ra đúng/sai một cách khách quan.
- Giải thích dễ hiểu, thực tế, không dùng lời chào hay văn mẫu rườm rà.
"""

def ask_llm(question: str) -> str:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.7,
        google_api_key=os.getenv("GEMINI_API_KEY")
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question)
    ]

    response = llm.invoke(messages)
    return response.content


if __name__ == "__main__":
    print("Chat với AI (nhập 'exit' để thoát)")
    while True:
        try:
            user_input = input("Bạn: ")
            if user_input.strip().lower() == "exit":
                print("Tạm biệt!")
                break
            if not user_input.strip():
                continue
            response = ask_llm(user_input)
            print(f"AI: {response}")
        except Exception as e:
            print(f"Lỗi: {e}")