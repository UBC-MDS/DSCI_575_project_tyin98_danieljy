from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import argparse


def load_faiss_index(index_path):
    """Load a FAISS vector store from disk.

    Args:
        index_path: Path to the directory containing the saved FAISS index files.

    Returns:
        A FAISS vector store
    """
    print(f"Loading FAISS index from {index_path}...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    return vector_store


def semantic_search(query, vector_store, k=10):
    """Return the vector store document for the most similar search results

    Args:
        query: The query to search for.
        vector_store: A FAISS vector store.
        k: Number of top results to return, default 10.

    Returns:
        A list of (page_content, score) tuples sorted by descending relevance.
    """
    results = vector_store.similarity_search_with_score(query, k)
    return [(doc.page_content, score) for doc, score in results]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", '-q', type=str, default=None)
    parser.add_argument("--k", '-k', type=int, default=10)
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    index_path = project_root / "data" / "processed" / "faiss_index"

    if not (index_path / "index.faiss").exists():
        print("FAISS index not found. Run utils.py first.")
        exit(1)
        
    vector_store = load_faiss_index(index_path)
    results = semantic_search(args.query, vector_store, args.k)
    for doc, score in results:
        print(f"Score: {score:.4f}, Product: {doc}")
    
