"""Faster version of utils.py for building the product corpus and indexes.
 
Output artifacts (products.pkl, embeddings.pkl, faiss_index/, bm25_index.pkl,
corpus.pkl) are identical in schema to utils.py — downstream code in semantic.py,
bm25.py, and hybrid.py does not need to change.
 
Changes vs utils.py:
- review_matching: vectorized with sort + groupby.head(k) instead of per-row
  iterrows + heapq; returns {asin: [text, ...]} rather than (votes, text) tuples.
- Parquet load: reads only the 5 columns actually used downstream.
- Embedding device: explicit CUDA > MPS > CPU selection (upstream auto-detect
  skips MPS on some versions, silently falling back to CPU on Apple Silicon).
- Embedding precision: fp16 on CUDA (~2x faster on tensor cores, quality delta
  below retrieval noise); fp32 elsewhere since CPU/MPS fp16 is slower or flaky.
- Embedding batch size: tuned per device (512 CUDA / 128 MPS / 64 CPU) instead
  of a fixed 256 that can be suboptimal depending on the device.
"""
import argparse
import time
import pickle
import re
import torch
import nltk
import pandas as pd
import numpy as np
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
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
    """Normalize text into BM25-ready tokens.

    Lowercases, strips punctuation (keeps hyphens and alphanumerics), drops
    English stop-words, and applies Snowball stemming.

    Args:
        text: Raw text string.

    Returns:
        A list of stemmed tokens suitable for BM25 indexing.
    """
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    return [stemmer.stem(w) for w in text.split() if w not in stop_words]

def review_matching(review_path, k=3):
    """Select the top-k most helpful reviews per product from a parquet file.

    Vectorized replacement for the old iterrows+heap version: loads only the
    needed columns, fills missing helpful_vote with 0, sorts once globally,
    then takes the first k rows of each parent_asin group.

    Args:
        review_path: Path to the reviews parquet file.
        k: Number of top reviews to keep per product, default 3.

    Returns:
        A dict mapping parent_asin (str) to a list of review text strings,
        ordered from most to least helpful.
    """
    df = pd.read_parquet(
        review_path,
        columns=["parent_asin", "helpful_vote", "text"],
        engine="pyarrow",
    )
    df["helpful_vote"] = df["helpful_vote"].fillna(0)
    # Sort once, then take head(k) per group
    df = df.sort_values(["parent_asin", "helpful_vote"], ascending=[True, False])
    top = df.groupby("parent_asin", sort=False).head(k)
    return top.groupby("parent_asin", sort=False)["text"].agg(list).to_dict()

def build_corpus_index(meta_path, review_path, output_dir, k=3, max_products=None):
    """Build the semantic corpus and FAISS index from product metadata and reviews.

    Joins each product's title, features, description, and top-k helpful reviews
    into a single document, encodes the documents with all-MiniLM-L6-v2, and
    writes three artifacts to output_dir: products.pkl (metadata with reviews
    attached), embeddings.pkl (raw vectors), and faiss_index/ (FAISS store).

    Uses CUDA+fp16 when available, MPS or CPU with fp32 otherwise; batch size
    is auto-tuned per device.

    Args:
        meta_path: Path to the product metadata parquet file.
        review_path: Path to the reviews parquet file.
        output_dir: Directory where pickle files and the FAISS index will be saved.
        k: Number of top reviews per product, default 3.
        max_products: If set, only process the first n products (useful for dev).
    """
    reviews_dict = review_matching(review_path, k)

    print("Loading data from parquet...")

    # Only load the columns we need, and convert features and description to lists if they are stored as arrays
    df = pd.read_parquet(meta_path, 
                         engine="pyarrow", 
                         columns=["parent_asin", "title", "features", "description", "average_rating"])
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
        reviews = " ".join(reviews_dict[asin])  # 
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

    # Pick the fastest available device explicitly. SentenceTransformer's
    # auto-detect doesn't always check for MPS, so Apple Silicon can silently
    # fall back to CPU. CUDA > MPS > CPU.
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
              else "cpu")
    # Batch size tuned per device: big on CUDA (tensor cores love it, VRAM is plenty
    # for a tiny model), moderate on MPS (unified memory pressure), small on CPU
    # (fits in L2/L3 cache instead of spilling to main memory).
    batch_size = {"cuda": 512, "mps": 128, "cpu": 64}[device]
    
    model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    if device == "cuda":
        model = model.half()
    print(f"Generating embeddings on {device} (batch size={batch_size}, "
          f"precision={'fp16' if device == 'cuda' else 'fp32'})...")

    print("Running this on the full dataset could take hours if you don't have CUDA/MPS.")
    print("Adjust batch size according to RAM/VRAM, and use --max-products to limit amount of products processed")
    start = time.time()
    embeddings = model.encode(
        corpus, 
        batch_size=batch_size, 
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
    """Persist a FAISS vector store from precomputed embeddings.

    Wraps the corpus texts and embeddings in a LangChain FAISS store, attaching
    each document's original position as metadata["index"] so retrievers can
    look up the corresponding product dict later.

    Args:
        corpus: List of document strings (parallel to embeddings).
        embeddings: 2D array-like of shape (len(corpus), embedding_dim).
        output_dir: Directory where the FAISS index files will be saved.
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
    """Build and persist a BM25 keyword index from the pickled product list.

    Reads products.pkl, tokenizes each product's title + features + description
    with tokenize(), fits a BM25Okapi model, and writes bm25_index.pkl and
    corpus.pkl to output_dir. Must be run after build_corpus_index(), which
    produces products.pkl.

    Args:
        output_dir: Directory containing products.pkl and where bm25_index.pkl
            and corpus.pkl will be written.
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