import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from src.chat_bot import chat_with_agent


def run_agent_tests():
    print("=" * 65)
    print("BẮT ĐẦU KIỂM TRA TÍNH NĂNG TOOL CALLING CỦA AGENT")
    print("=" * 65)

    test_cases = [
        {
            "category": "1. Câu hỏi giao tiếp (KHÔNG ĐƯỢC GỌI TOOL)",
            "query": "Chào bạn, bạn có khỏe không? Bạn có thể giúp gì cho tôi?"
        },
        {
            "category": "2. Câu hỏi kiến thức CÓ trong tài liệu (PHẢI GỌI TOOL)",
            "query": "Overfitting là gì và có những cách khắc phục nào?"
        },
        {
            "category": "3. Câu hỏi kiến thức khác CÓ trong tài liệu (PHẢI GỌI TOOL)",
            "query": "Kỹ thuật RAG gồm những bước chính nào?"
        },
        {
            "category": "4. Câu hỏi CẦN KIỂM CHỨNG (Không có trong tài liệu)",
            "query": "Thủ đô của nước Pháp là gì và diện tích là bao nhiêu?"
        }
    ]

    for tc in test_cases:
        print(f"\n Test Case: [{tc['category']}]")
        print(f"   Câu hỏi: \"{tc['query']}\"")
        
        response = chat_with_agent(tc["query"])
        print(f"\n   AI trả lời:\n{response}")
        print("-" * 65)

    print("\n" + "=" * 65)
    print("HOÀN THÀNH TẤT CẢ TEST CASES!")
    print("=" * 65)


if __name__ == "__main__":
    run_agent_tests()
