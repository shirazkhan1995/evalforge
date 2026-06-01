"""Quick sanity check that retrieval works."""

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

def main():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-2")
    vector_store = Chroma(
        collection_name="evalforge",
        embedding_function=embeddings,
        persist_directory="./chroma_db"
    )

    query = "What is the procedure for implementation of the system driven disclosures?"
    results = vector_store.similarity_search_with_score(query, k=3)
    
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n--- Result {i} (score: {score:.4f}) ---")
        print(f"Source: {doc.metadata.get('source_file')} (page {doc.metadata.get('page')})")
        print(f"Content: {doc.page_content}[:300]...")

if __name__ == "__main__":
    main()
