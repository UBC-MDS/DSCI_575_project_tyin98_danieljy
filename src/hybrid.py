import argparse
import pickle
from pathlib import Path

from bm25 import bm25_retriever, load_index as load_bm25_index
from semantic import semantic_retriever, load_faiss_index


class HybridRetriever:
    """Retriever that merges FAISS semantic search and BM25 keyword search via RRF.

    Attributes:
        vector_store: A LangChain FAISS vector store used for semantic retrieval.
        bm25_index:   A BM25Okapi index used for keyword retrieval.
        products:     A list of product dicts loaded from products.pkl.
        rrf_k:        RRF smoothing constant (default 60). Higher values reduce
                      the impact of top-ranked documents.
    """

    def __init__(self, vector_store, bm25_index, products, rrf_k=60):
        """Initialize hybrid retriever with FAISS and BM25 indexes"""
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.products = products
        self.rrf_k = rrf_k

    def retrieve(self, query, k=10):
        """Return the top-k product indices ranked by hybrid RRF score.

        Runs semantic and BM25 retrieval in parallel, then fuses the two ranked
        lists using Reciprocal Rank Fusion so that documents ranked highly by
        both retrievers float to the top.

        Args:
            query: The user's search query string.
            k:     Number of top results to return (default 10).

        Returns:
            A list of up to k integer indices into self.products, ordered by
            descending RRF score (best match first).
        """

        # --- Step 1: Run both retrievers independently ---
        semantic_results = semantic_retriever(query, self.vector_store, k)
        bm25_indices, _ = bm25_retriever(query, self.bm25_index, k)

        # --- Step 2: Build rank lookup dicts (rank 1 = best) ---
        semantic_ranks = {
            product_idx: rank + 1
            for rank, (product_idx, _score) in enumerate(semantic_results)
        }

        bm25_ranks = {
            product_idx: rank + 1
            for rank, product_idx in enumerate(bm25_indices)
        }

        # --- Step 3: Collect all unique product indices seen by either retriever ---
        all_indices = set(semantic_ranks.keys()) | set(bm25_ranks.keys())

        # --- Step 4: Compute RRF score for every candidate document ---
        # If a document is missing from one list it only contributes one term.
        # Documents appearing in both lists receive the highest combined scores.
        rrf_scores = {}
        for idx in all_indices:
            score = 0.0
            if idx in semantic_ranks:
                score += 1.0 / (self.rrf_k + semantic_ranks[idx])
            if idx in bm25_ranks:
                score += 1.0 / (self.rrf_k + bm25_ranks[idx])
            rrf_scores[idx] = score

        # --- Step 5: Sort by RRF score descending and return top-k indices ---
        ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        return [idx for idx, _score in ranked[:k]]


def load_indexes(data_dir, index_path):
    """Load all indexes and return a ready-to-use HybridRetriever instance.

    Convenience function for app.py and rag_pipeline.py so they don't have to
    call load_faiss_index() and load_bm25_index() separately.

    Args:
        data_dir:   Path to the directory containing products.pkl and bm25_index.pkl.
        index_path: Path to the directory containing the saved FAISS index files.

    Returns:
        A HybridRetriever instance with all indexes loaded.
    """
    # Load FAISS vector store and products list
    vector_store, products = load_faiss_index(index_path, data_dir)

    # Load BM25 index (products already loaded above, so we only need the index)
    bm25_index, _ = load_bm25_index(data_dir)

    return HybridRetriever(vector_store, bm25_index, products)


if __name__ == "__main__":
    # CLI entry point for quick manual testing, e.g.:

    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "processed"
    index_path = data_dir / "faiss_index"

    parser = argparse.ArgumentParser()
    parser.add_argument("--query", "-q", type=str, required=True, help="Search query")
    parser.add_argument("--k", "-k", type=int, default=10, help="Number of results")
    args = parser.parse_args()

    if not (index_path / "index.faiss").exists():
        print("FAISS index not found. Run utils.py first.")
        exit(1)

    if not (data_dir / "bm25_index.pkl").exists():
        print("BM25 index not found. Run utils.py first.")
        exit(1)

    retriever = load_indexes(data_dir, index_path)
    results = retriever.retrieve(args.query, args.k)

    for rank, idx in enumerate(results, start=1):
        product = retriever.products[idx]
        print(f"\nRank {rank}")
        print(f"Product title: {product['title']}")
        print(f"Rating:        {product.get('average_rating', 'N/A')}/5")
        print(f"ASIN:          {product.get('parent_asin', 'N/A')}")
        print("--------------------------------------")