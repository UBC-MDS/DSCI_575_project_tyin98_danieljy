from pathlib import Path
from collections import defaultdict
from sentence_transformers import SentenceTransformer
import json
import heapq

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
            text = review.get('text', '')
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
    
def build_corpus(meta_path, review_path, output_dir, k=3):

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
        reviews = reviews_dict[asin]
        part = f"""Title: {product.get("title", "")} |
Features: {", ".join(product.get("features", []))} |
Description: {". ".join(product.get("description", []))} |
Reviews: {" ".join(x[1] for x in reviews)}
"""
        corpus.append(part)

    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("Generating embeddings...")
    print(f"Using device: {model.device}")
    print("This could take hours if you don't have CUDA/MPS.")
    print("If you crash due to OOM or notice it's using swap space, reduce batch size")
    embeddings = model.encode(corpus, batch_size=256, show_progress_bar=True)

    print("pickling generated embeddings...")
    with open(output_dir / "embeddings.pkl", "wb") as f:
        pickle.dump(embeddings, f)
    return embeddings
    
if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    meta_path= project_root / "data" / "raw" / "meta_Musical_Instruments.jsonl"
    review_path= project_root / "data" / "raw" / "Musical_Instruments.jsonl"
    output_dir=project_root / "data" / "processed"
    
    corpus = build_corpus(meta_path, review_path, output_dir)
    
