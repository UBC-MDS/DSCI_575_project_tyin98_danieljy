import json
import pickle
import re
from pathlib import Path
from rank_bm25 import BM25Okapi
from tqdm import tqdm
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))
stemmer = SnowballStemmer("english")

def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s-]", "", text)
    return [stemmer.stem(w) for w in text if w not in stop_words]


def build_and_save_index(output_dir):
    #data_path = Path(data_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    #print("Loading data from json...")
    #products = []
    #with open(data_path, "r") as f:
    #    for line in f:
    #        if line.strip():
    #            products.append(json.loads(line))
    #print(f"Loaded {len(products)} products")
    
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


def load_index(input_dir) -> tuple[BM25Okapi, list[dict]]:
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
    data_path=project_root / "data" / "raw" / "meta_Musical_Instruments.jsonl"
    output_dir=project_root / "data" / "processed"

    if not (output_dir / "bm25_index.pkl").exists():
        build_and_save_index(output_dir)

    bm25_index, products = load_index(project_root / "data" / "processed")
    results = search("digital piano hammer action", bm25_index, products, top_k=10)
    for product, score in results:
        print(f"Score: {score:.4f}, Product: {product}")
