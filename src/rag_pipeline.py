from semantic import semantic_retriever, load_faiss_index
import argparse
import os
from pathlib import Path
from prompts import build_prompt
from hybrid import HybridRetriever, load_indexes

from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq

class RAG_Pipeline:
    def __init__(self, vector_store , products):
        self.vector_store = vector_store
        self.products = products

    def retrieve(self, query, k=10):
        return [idx for idx, score in semantic_retriever(query, self.vector_store, k)]

    def build_context(self, docs):
        context_parts = []
        for idx in docs:
            product = self.products[idx]
            part = (
                f"ASIN: {product.get('parent_asin', '')}\n"
                f"Title: {product.get('title', '')}\n"
                f"Rating: {product.get('average_rating', 'N/A')}/5\n"
                f"Description: {product.get('description', '')}\n"
                f"Features: {product.get('features', '')}\n"
                f"Reviews: {product.get('reviews', '')}"
            )
            context_parts.append(part)
        return "\n\n".join(context_parts)

    def query(self, query, k=10):
        # the function that runs the entire RAG pipeline
        if "GROQ_API_KEY" not in os.environ:
            print("GROQ_API_KEY not found. Please add it in the .env file.")
            exit(1) 

        llm = ChatGroq(
            model="qwen/qwen3-32b",
            temperature=0.3,
            max_tokens=None,
            reasoning_format="parsed",
            timeout=None,
            max_retries=2,
        )

        docs = self.retrieve(query, k)
        context = self.build_context(docs)
        prompt = build_prompt(query, context)
        
        return llm.invoke(prompt).content, docs

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
    print(pipeline.query(args.query, args.k))
    
class HybridRAGPipeline(RAG_Pipeline):
    """RAG pipeline using a hybrid retriever (FAISS semantic + BM25) via RRF.
 
    Inherits build_context() and query() from RAG_Pipeline unchanged.
    Only __init__ and retrieve() are overridden to swap in the HybridRetriever.
 
    Attributes:
        hybrid_retriever: A HybridRetriever instance combining FAISS and BM25.
        products:         Inherited from RAG_Pipeline — list of product dicts.
    """
 
    def __init__(self, hybrid_retriever: HybridRetriever):
        # Pass the products list up to the parent so build_context() works as-is.
        # vector_store is not needed directly — the HybridRetriever owns it.
        super().__init__(
            vector_store=hybrid_retriever.vector_store,
            products=hybrid_retriever.products,
        )
        self.hybrid_retriever = hybrid_retriever
 
    def retrieve(self, query, k=10):
        """Return top-k product indices using the hybrid RRF retriever.
 
        Overrides RAG_Pipeline.retrieve(). Delegates to HybridRetriever.retrieve()
        which internally fuses FAISS semantic results and BM25 keyword results.
 
        Args:
            query: The user's search query string.
            k:     Number of results to retrieve (default 10).
 
        Returns:
            A list of integer indices into self.products, ranked by RRF score.
        """
        return self.hybrid_retriever.retrieve(query, k)
 
if __name__ == "__main__":
    # CLI entry point for testing both pipelines side by side
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", "-q", type=str, required=True, help="Search query")
    parser.add_argument("--k", "-k", type=int, default=10, help="Number of results")
    parser.add_argument(
        "--mode",
        choices=["semantic", "hybrid", "both"],
        default="both",
        help="Which pipeline to run (default: both)",
    )
    args = parser.parse_args()
 
    project_root = Path(__file__).parent.parent
    data_dir = project_root / "data" / "processed"
    index_path = data_dir / "faiss_index"
 
    if not (index_path / "index.faiss").exists():
        print("FAISS index not found. Run utils.py first.")
        exit(1)
 
    if not (data_dir / "bm25_index.pkl").exists():
        print("BM25 index not found. Run utils.py first.")
        exit(1)
 
    # load_indexes() handles loading FAISS + BM25 + products in one call
    hybrid_retriever = load_indexes(data_dir, index_path)
 
    if args.mode in ("semantic", "both"):
        print("\n" + "=" * 60)
        print("SEMANTIC RAG PIPELINE")
        print("=" * 60)
        semantic_pipeline = RAG_Pipeline(
            hybrid_retriever.vector_store, hybrid_retriever.products
        )
        answer, docs = semantic_pipeline.query(args.query, args.k)
        print(answer)
 
    if args.mode in ("hybrid", "both"):
        print("\n" + "=" * 60)
        print("HYBRID RAG PIPELINE")
        print("=" * 60)
        hybrid_pipeline = HybridRAGPipeline(hybrid_retriever)
        answer, docs = hybrid_pipeline.query(args.query, args.k)
        print(answer)        