from pathlib import Path
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import json
import heapq
import pickle

def review_matching(review_path, k=3):
    print(f"Filtering top {k} helpful reviews for all products...")
    top_reviews_dict = defaultdict(list)
    with open(review_path, "r") as f:
        for line in f:
            if not line.strip(): continue
            review = json.loads(line)
            # filter out low quality reviews
            if not review.get('verified_purchase', False): continue
            if review.get('rating', 0) < 3.0: continue
            text = f'{review.get('title', '')}: {review.get('text', '')}'
            if not text: continue
            if len(text) < 50 or len(text) > 300: continue

            asin = review['parent_asin']
            votes = review.get('helpful_vote', 0)

            # min heap to keep top k reviews
            heap = top_reviews_dict[asin]            
            if len(heap) < k:
                heapq.heappush(heap, (votes, text))
            else:
                heapq.heappushpop(heap, (votes, text))
    return top_reviews_dict
    
def build_corpus_index(meta_path, review_path, output_dir, k=3):

    reviews_dict = review_matching(review_path, k)

    print("Loading data from json...")
    products = []
    with open(meta_path, "r") as f:
        for line in f:
            if line.strip():
                products.append(json.loads(line))
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

    print("pickling product dictionary with reviews...")
    with open(output_dir / "products.pkl", "wb") as f:
        pickle.dump(products, f)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Generating embeddings...")
    print(f"Using device: {model.device}")
    print("This could take hours if you don't have CUDA/MPS.")
    print("Adjust batch size according to RAM/VRAM")
    embeddings = model.encode(corpus, batch_size=256, show_progress_bar=True)

    print("pickling generated embeddings...")
    with open(output_dir / "embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)
    
    build_faiss_index(corpus, embeddings, output_dir= output_dir / "faiss_index")

    return embeddings, products


def build_faiss_index(corpus, embeddings, output_dir):
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_community.vectorstores import FAISS

    print(f"Building FAISS index...")
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_embeddings(
        text_embeddings=list(zip(corpus, embeddings)),
        embedding=embedding_model
    )

    print(f"Saving FAISS index...")
    vector_store.save_local(output_dir)

    return vector_store


def load_faiss_index(index_path):

    print(f"Loading FAISS index from {index_path}...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)

    return vector_store


def semantic_search(query, vector_store, k=10):
    results = vector_store.similarity_search_with_score(query, k)
    return [(doc.page_content, score) for doc, score in results]


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    meta_path= project_root / "data" / "raw" / "meta_Musical_Instruments.jsonl"
    review_path= project_root / "data" / "raw" / "Musical_Instruments.jsonl"
    output_dir=project_root / "data" / "processed"
    index_path = output_dir / "faiss_index"
    
    if not (index_path / "index.faiss").exists():
        build_corpus_index(meta_path, review_path, output_dir)

    vector_store = load_faiss_index(index_path)
    results = semantic_search("digital piano hammer action", vector_store, k=10)
    for doc, score in results:
        print(f"Score: {score:.4f}, Product: {doc}")
    
