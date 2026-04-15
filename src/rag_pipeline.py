from semantic import semantic_retriever, load_faiss_index
import argparse
from pathlib import Path

class RAG_Pipeline:
    def __init__(self, vector_store , products):
        self.vector_store = vector_store
        self.products = products

    def retrieve(self, query, k=5):
        return [idx for idx, score in semantic_retriever(query, self.vector_store, k)]

    def build_context(self, docs):
        context_parts = []
        for idx in docs:
            product = self.products[idx]
            part = (
                f"ID: {idx}\n"
                f"Title: {product.get('title', '')}\n"
                f"Rating: {product.get('average_rating', 'N/A')}/5\n"
                f"Description: {product.get('description', '')}\n"
                f"Features: {product.get('features', '')}\n"
                f"Reviews: {product.get('reviews', '')}"
            )
            context_parts.append(part)
        return "\n\n".join(context_parts)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", '-q', type=str, default="")
    parser.add_argument("--k", '-k', type=int, default=10)
    parser.add_argument("--show_thinking", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "processed"
    index_path = data_dir / "faiss_index"

    if not (index_path / "index.faiss").exists():
        print("FAISS index not found. Run utils.py first.")
        exit(1)

    vector_store, products= load_faiss_index(index_path, data_dir)

    pipeline = RAG_Pipeline(vector_store, products)
    docs = pipeline.retrieve(args.query, args.k)
    print(pipeline.build_context(docs))
    
        
