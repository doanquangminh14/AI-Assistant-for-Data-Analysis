import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from src.document_processor import load_documents, split_documents, get_embedding_model, store_in_vector_db
def run_test_on_data():
    data_dir = "./data"
    supported_extensions = (".txt", ".md", ".pdf", ".docx", ".doc")
    if not os.path.exists(data_dir):
        print(f"Error: '{data_dir}' is not a directory")
        return

    valid_files = [f for f in os.listdir(data_dir)
                   if f.lower().endswith(supported_extensions)]
    if not valid_files:
        print(f"No Supported Documents: {data_dir}")
        return
    print("\n" + "=" * 60)
    print(f"Start processing {len(valid_files)} files in {data_dir}/")
    print("=" * 60)
    all_chunk = []
    for idx, file_name in enumerate(valid_files, start = 1):
        file_path = os.path.join(data_dir, file_name)
        print(f"[{idx}/{len(valid_files)}] Loading {file_name}...")
        try:
            docs = load_documents(file_path)
            print(f"Loaded {len(docs)} Document")
            print(f"Total word {len(docs[0].page_content.split())}")

            chunks = split_documents(docs,chunk_size=1000, chunk_overlap=200)
            print(f"Total chunks {len(chunks)}")
            
            all_chunk.extend(chunks)
            print("-" * 60)
        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            print("-" * 60)
    
    print(f"\n Total Document: {len(all_chunk)}")
    if not all_chunk:
        print("No Chunks")
        return
   
    print("\n" + "-" * 60)
    print("Create embedding model save into vector DB")
    print("-" * 60)
    try:
        vector_db = store_in_vector_db(chunks=all_chunk,
                                        persist_directory="./chroma_db",
                                        collection_name="knowledge_base",
                                        embedding_provider="local")
        print("Saved in chromadb ok!")
        
        print("\n" + "=" * 60)
        print("TEST Query Semantic Search(enter 'exit' to exit)")
        print("=" * 60)

        while True:
            query = input("\nQuery: ").strip()
            if not query or query.lower() == "exit":
                print("Exit!")
                break

            results = vector_db.similarity_search(query, k=4)
            if results:
                print("\n Result of query :\n")
                for i, doc in enumerate(results, start=1):
                    print(f"Document:{doc.metadata['source']}\n Content:{doc.page_content}\n")
            else:
                print("No results found")
    except Exception as e:
        print(f"Error during query: {e}")

    print("Vector DB persisted!")
    print("=" * 60)

if __name__ == "__main__":
    run_test_on_data()


    
        
    
        


    


    



    