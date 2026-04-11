import pickle
import argparse
from pathlib import Path
from rank_bm25 import BM25Okapi
from utils import tokenize


def load_index(input_dir):
    """Load the BM25 index and product list from disk.

    Args:
        input_dir: Path to the directory containing bm25_index.pkl and products.pkl.

    Returns:
        A tuple of (bm25_index, list[products dicts])
    """
    input_dir = Path(input_dir)
    with open(input_dir / "bm25_index.pkl", "rb") as f:
        bm25_index = pickle.load(f)
    with open(input_dir / "products.pkl", "rb") as f:
        products = pickle.load(f)
    return bm25_index, products


def search(query, bm25_index, products, top_k=10):
    """
    Return list of (product, score) sorted by score descending
    """
    tokenized_query = tokenize(query)
    scores = bm25_index.get_scores(tokenized_query)
    ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    return [(products[i], scores[i]) for i in ranked_idx]


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    output_dir=project_root / "data" / "processed"

    parser = argparse.ArgumentParser()
    parser.add_argument("--query", '-q', type=str, default=None)
    parser.add_argument("--k", '-k', type=int, default=10)
    args = parser.parse_args()

    if not (output_dir / "bm25_index.pkl").exists():
        print("BM25 index not found. Run utils.py first.")
        exit(1)

    bm25_index, products = load_index(project_root / "data" / "processed")
    results = search(args.query, bm25_index, products, args.k)
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
