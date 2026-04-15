import streamlit as st
import sys
import re
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from src.bm25 import load_index as bm25_load, bm25_search
from src.semantic import load_faiss_index, semantic_search
from src.rag_pipeline import RAG_Pipeline

data_dir = project_root / "data" / "processed"


@st.cache_resource
def load_bm25():
    return bm25_load(data_dir)


@st.cache_resource
def load_semantic():
    index_path = data_dir / "faiss_index"
    return load_faiss_index(index_path, data_dir)


bm25_index, products_bm25 = load_bm25()
vector_store, products_sem = load_semantic()
rag_pipeline = RAG_Pipeline(vector_store, products_sem)

search, rag = st.tabs(["Search", "RAG"])

with search:
    method = st.radio("Search method", ["BM25", "Semantic"])
    query = st.text_input("Query")

    if query:
        if method == "BM25":
            results = bm25_search(query, bm25_index, products_bm25, top_k=3)
        else:
            results = semantic_search(query, vector_store, products_sem, k=3)

        for product, score in results:
            st.write(f"Title: {product.get('title', '')}")
            reviews = product.get("reviews", "")
            if isinstance(reviews, str):
                truncated = reviews[:200] + ("..." if len(reviews) > 200 else "")
            else:
                truncated = str(reviews)[:200] + "..."
            st.write(f"Review: {truncated}")
            st.write(f"Rating: {product.get('average_rating', 'N/A')}")
            st.write(f"Score: {score:.4f}")
            st.divider()

with rag:
    query = st.text_input("RAG Query")
    if query:
        results, docs = rag_pipeline.query(query)
        if not results:
            st.write("LLM API call failed. Please check your API key or internet connection and try again later.")
        else:
            results = re.split(r'<rank>(.*?)</rank>', results)
            st.write(results[2])
            ranks = [r.strip() for r in results[1].split(",")]
            st.divider()
            for idx in docs:
                product = products_sem[idx]
                title = product.get('title', '')
                asin = product.get('parent_asin', '')
                rating = product.get('average_rating', 'N/A')
                reviews = product.get("reviews", "")
                if asin in ranks:
                    st.success(f"Ranked by RAG to be #{ranks.index(asin) + 1}")
                if isinstance(reviews, str):
                    truncated = reviews[:200] + ("..." if len(reviews) > 200 else "")
                else:
                    truncated = str(reviews)[:200] + "..."
                st.write(f"[{title}](http://www.amazon.com/dp/{asin}/ref=nosim)")
                st.write(f"ASIN: {asin}")
                st.write(f"Review: {truncated}")
                st.write(f"Rating: {rating}")
                st.divider()