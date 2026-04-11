import pickle
from pathlib import Path
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import argparse


def load_faiss_index(index_path, data_dir):
    """Load a FAISS vector store and the products dictionary from disk.

    Args:
        index_path: Path to the directory containing the saved FAISS index files.
        data_dir: Path to products.pkl.

    Returns:
        A tuple (FAISS vector store, list of product dicts)
    """
    print(f"Loading FAISS index and products dictionaries...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    with open(Path(data_dir) / "products.pkl", "rb") as f:
        products = pickle.load(f)

    return vector_store, products

def semantic_search(query, vector_store, products, k=10):
    """Return list of (product, score) sorted by score ascending

    Args:
        query: The query to search for.
        vector_store: A FAISS vector store.
        products: A list of product dicts.
        k: Number of top results to return, default 10.

    Returns:
        A list of (product_dict, score) tuples.
    """
    results = vector_store.similarity_search_with_score(query, k)
    return [(products[doc.metadata["index"]], float(score)) for doc, score in results]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", '-q', type=str, default=None)
    parser.add_argument("--k", '-k', type=int, default=10)
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "processed"
    index_path = data_dir / "faiss_index"

    if not (index_path / "index.faiss").exists():
        print("FAISS index not found. Run utils.py first.")
        exit(1)

    vector_store, products= load_faiss_index(index_path, data_dir)
    results = semantic_search(args.query, vector_store, products, args.k)
    for product, score in results:
        print(f"""

Score: {score:.4f}
Product title: {product['title']}
--------------------------------------
Description: {product['description']}
--------------------------------------
Features: {product['features']}
--------------------------------------
Reviews: {product['reviews']}""")
    
