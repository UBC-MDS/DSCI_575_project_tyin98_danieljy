import argparse
import time
import heapq
import pickle
import re
import nltk
import pandas as pd
import numpy as np
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from pathlib import Path
from tqdm import tqdm
from rank_bm25 import BM25Okapi

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")

def tokenize(text):
    """Lowercase, strip punctuation, remove stop-words, and stem.

    Args:
        text: Raw text string

    Returns:
        A list of tokens
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    return [stemmer.stem(w) for w in text.split() if w not in stop_words]

# def review_matching(review_path, k=3):
#     """Reads a parquet review file and keeps the k reviews with the most helpful votes per product.

#     Args:
#         review_path: Path to the reviews parquet file.
#         k: Number of most helpful reviews to keep per product, default 3

#     Returns:
#         A dict mapping parent_asin to a list of (votes, text) tuples
#     """
#     print(f"Filtering top {k} helpful reviews for all products...")
#     df = pd.read_parquet(review_path, engine="pyarrow")

#     top_reviews_dict = defaultdict(list)
#     for _, row in df.iterrows():
#         asin = row['parent_asin']
#         votes = row.get('helpful_vote', 0)
#         # min heap to keep top k reviews
#         heap = top_reviews_dict[asin]
#         if len(heap) < k:
#             heapq.heappush(heap, (votes, row['text']))
#         else:
#             heapq.heappushpop(heap, (votes, row['text']))
#     return top_reviews_dict

# Usinge groupby + nlargest:
def review_matching(review_path, k=3):
    df = pd.read_parquet(review_path, columns=["parent_asin", "helpful_vote", "text"])
    df["helpful_vote"] = df["helpful_vote"].fillna(0)
    top = (
        df.groupby("parent_asin", group_keys=False)
          .apply(lambda g: g.nlargest(k, "helpful_vote"))
    )
    return top.groupby("parent_asin")["text"].agg(list).to_dict()

def build_corpus_index(meta_path, review_path, output_dir, k=3, max_products=None):
    """Loads product metadata and reviews, creates a corpus containing title, features,
    description, and top reviews, encodes the corpus with all-MiniLM-L6-v2, and saves
    everything as pickle files plus a built FAISS index.

    Args:
        meta_path: Path to the product metadata parquet file.
        review_path: Path to the reviews parquet file.
        output_dir: Directory where pickle files and the FAISS index will be saved.
        k: Number of top reviews per product, default 3.
        max_products: If set, only process the first n products.
    """
    reviews_dict = review_matching(review_path, k)

    print("Loading data from parquet...")

    # Only load the columns we need, and convert features and description to lists if they are stored as arrays
    df = pd.read_parquet(meta_path, 
                         engine="pyarrow", 
                         columns=["parent_asin", "title", "features", "description"])
    if max_products:
        df = df.head(max_products)
    products = df.to_dict('records')
    for p in products:
        if isinstance(p.get('features'), np.ndarray):
            p['features'] = p['features'].tolist()
        if isinstance(p.get('description'), np.ndarray):
            p['description'] = p['description'].tolist()
    print(f"Loaded {len(products)} products")

    print("Building corpus...")
    corpus = []
    for product in products:
        asin = product['parent_asin']
        reviews = " ".join(x[1] for x in reviews_dict[asin])
        part = f"""Title: {product.get("title", "")} |
Features: {", ".join(product.get("features", []))} |
Description: {". ".join(product.get("description", []))} |
Reviews: {reviews}
"""
        corpus.append(part)
        product['reviews'] = reviews

    print("Pickling product dictionary with reviews...")
    with open(output_dir / "products.pkl", "wb") as f:
        pickle.dump(products, f)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    # model = SentenceTransformer("paraphrase-MiniLM-L6-v2") # smaller model = faster

    print("Generating embeddings...")
    print(f"Using device: {model.device}")
    print("Running this on the full dataset could take hours if you don't have CUDA/MPS.")
    print("Adjust batch size according to RAM/VRAM, and use --max-products to limit amount of products processed")
    start = time.time()
    embeddings = model.encode(
        corpus, 
        batch_size=256, 
        show_progress_bar=True, 
        normalize_embeddings=True
        )
    elapsed = time.time() - start
    print(f"Embedding generation took {elapsed:.2f} seconds")

    print("Pickling generated embeddings...")
    with open(output_dir / "embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)
    
    build_faiss_index(corpus, embeddings, output_dir=output_dir / "faiss_index")

    return

def build_faiss_index(corpus, embeddings, output_dir):
    """Create and save a FAISS vector store from corpus text and embeddings.

    Args:
        corpus: List of strings.
        embeddings: Generated embedding vectors.
        output_dir: Directory where the FAISS index will be saved.
    """
    print(f"Building FAISS index...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    metadatas = [{"index": i} for i in range(len(corpus))]
    vector_store = FAISS.from_embeddings(
        text_embeddings=list(zip(corpus, embeddings)),
        embedding=embedding_model,
        metadatas=metadatas
    )

    print(f"Saving FAISS index...")
    vector_store.save_local(output_dir)

    return

def build_bm25_index(output_dir):
    """Loads products.pkl, tokenizes each product's title, features, and description, then creates a BM25 index.
    Saves the index and tokenized corpus to disk.

    Args:
        output_dir: Directory containing products.pkl and where bm25_index.pkl and corpus.pkl will be saved.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading pickled products...")
    products = None
    with open(output_dir / "products.pkl", "rb") as f:
        products = pickle.load(f)
    if not products:
        print("products.pkl not found. Run semantic.py first.")

    corpus = []
    for product in tqdm(products, desc = "Tokenizing corpus"):
        parts = [product.get("title", "")]
        parts.extend(product.get("features", []))
        parts.extend(product.get("description", []))
        corpus.append(tokenize(" ".join(parts)))

    print("Building BM25 index...")
    bm25_index = BM25Okapi(corpus)

    print("Pickling data...")
    with open(output_dir / "bm25_index.pkl", "wb") as f:
        pickle.dump(bm25_index, f)
    with open(output_dir / "corpus.pkl", "wb") as f:
        pickle.dump(corpus, f)
    #with open(output_dir / "products.pkl", "wb") as f: # we also pickle this so we don't have to load the json every time
    #    pickle.dump(products, f)   

    print(f"Saved pickle files to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-products", type=int, default=None)
    parser.add_argument("--rebuild", action="store_true")
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    meta_path= project_root / "data" / "processed" / "filtered_meta.parquet"
    review_path= project_root / "data" / "processed" / "filtered_reviews.parquet"
    output_dir=project_root / "data" / "processed"
    index_path = output_dir / "faiss_index"

    if not (index_path / "index.faiss").exists() or args.rebuild:
        build_corpus_index(meta_path, review_path, output_dir, max_products=args.max_products)
    else:
        print("FAISS index already exists, skipping")

    if not (output_dir / "bm25_index.pkl").exists() or args.rebuild:
        build_bm25_index(output_dir)
    else:
        print("BM25 index already exists, skipping")

    print("All done!")