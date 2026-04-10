# DSCI_575_project_tyin98_danieljy

-   Put `Musical_Instruments.jsonl` and `meta_Musical_Instruments.jsonl` under `data/raw/`

-   Run `python src/utils.py` to build all necessary index files, embeddings, and product dictionaries.

    > Using the full dataset could take hours on CPU. Use `--max-products` to limit the products processed e.g. `python src/utils.py --max-products 2000` to only process the first 2000 products. If you want to change `--max-products` , make sure the `faiss_index` folder and the `bm25_index.pkl` file are deleted, or use `--rebuild` .

-   Run `python src/semantic.py -q "your search query" -k 10` to perform a test search of top 10 matching products using semantic search.

-   Run `python src/bm25.py``-q "your search query" -k 10` to perform a test search of top 10 matching products using BM25 search.

Some design choices: Semantic search included the reviews as part of the corpus but not BM25 because with basic keyword matching including the review might skew the model too much. We sorted the reviews by helpful_vote and only kept the good reviews (rating \>= 3).
