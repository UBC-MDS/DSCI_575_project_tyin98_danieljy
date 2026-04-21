# Final Discussion

## Step 1: Improve Your Workflow

### Dataset Scaling
- Number of products used: 

Currently the filtered parquet files contain about ~1,000,000 products. However, when building the indexes for BM25 and semantic search, we would need to use ~5000 products to have things build in less than ~5 minutes. So, we did not change the sampling method, but we focused on improving our utils.py file to be more efficient in generating the indexes for our semantic and BM25 models. 

We created a 'utils2.py' file which focuses on greater efficiency. The following is a summary of the changes relative to 'utils.py':

- review_matching: vectorized with sort + groupby.head(k) instead of per-row iterrows + heapq; returns {asin: [text, ...]} rather than (votes, text) tuples.
- Parquet load: reads only the 5 columns actually used downstream.
- Embedding device: explicit CUDA > MPS > CPU selection (upstream auto-detect skips MPS on some versions, silently falling back to CPU on Apple Silicon).
- Embedding precision: fp16 on CUDA (~2x faster on tensor cores, quality delta below retrieval noise); fp32 elsewhere since CPU/MPS fp16 is slower or flaky.
- Embedding batch size: tuned per device (512 CUDA / 128 MPS / 64 CPU) instead of a fixed 256 that can be suboptimal depending on the device.

- Changes to sampling strategy (if any): N/A

### LLM Experiment
- Models compared (name, family, size)
- Results and discussions
    - Prompt used (copy it here)
    - Results
- Which model you chose and why

## Step 2: Additional Feature (Web Search Tool Call)

### What You Implemented

Implemented Tavily web search into the RAG pipeline. The LLM dynamically decides whether a query requires a web search by asking it to respond "yes" or "no". If needed, the web search results are added before the context from our database, allowing the LLM to incorporate additional real-time information.

### Example Queries

1. "Hammer action piano under 500"
   The response combined Amazon catalog matches with web searched pricing information, making sure all selected products are under $500.

2. "guitar pedals released in 2020"
   Web results improved the response by informing the LLM when models are released. All selected options are released or re-released in 2020.

3. "acoustic guitar with the best reputation on reddit"
   This improved the reply by incorporating non-Amazon information, using reddit to advise on the guitar's affordability, performance, and quality control.

We see an improvement in all three queries, proving that having those additional information is very beneficial. However there are several trade offs: there is now a slightly longer latency due to the web search, it requires a Tavily API as well as one additional call to the LLM, which would increase the cost.
  
## Step 3: Improve Documentation and Code Quality

### Documentation Update

- README already have clear setup instructions and usage examples
- New feature (web search) briefly described in the README
- Added the new optimized workflow for 'utils2.py' to the index building step.

### Code Quality Changes

- Deleted some placeholder files that are no longer needed
- All functions now have a doc string
- No hardcoded paths in source code
- No API keys in source code, `.env` file excluded in `.gitignore`
- Also excluded python and jupyter cache files

## Step 4: Cloud Deployment Plan

We would deploy the Amazon product recommendation tool on **AWS**, using **S3** for storage, **EC2** for serving the app, and **EMR with Spark** for the periodic heavy-lifting index-rebuild job. This mirrors the split we already have in the codebase: a cheap, always-on serving layer that runs `semantic.py` / `bm25.py` / `hybrid.py` / `rag_pipeline.py` against pre-built indexes, plus an occasional batch job that regenerates those indexes from the raw parquet files using our `utils2.py` pipeline.

### 1. Data Storage

All data and artifacts would live in **S3**, organized by prefix so the serving and rebuild jobs can find them predictably. Both EC2 and EMR can read from S3 directly at high throughput. A similar folder  structure as the GitHUb Repository can be used.

Overall, the new data extraction workflow will be: develop a new python script file which fetches the data from the UCSD McAuley Lab Amazon Reviews 2023 and then performs the filtering and transforming actions of the 'milestone1_exploration.ipynb' file. This script can be re-run daily or weekly (depending on need and resources) to get new reviews + metadata. 

**Raw data**: For the raw data (Musical_Instruments.jsonl.gz and meta_Musical_Instruments.jsonl.gz), we will fetch the data from the UCSD McAuley Lab Amazon Reviews 2023 dataset site. The download itself is just a single HTTP request so it doesn't need Spark workers — we'd run it as a shell step on the EMR primary node before the Spark filter stage kicks in. The gzipped JSONL files go directly to s3://data/raw/. These files are large, immutable once downloaded, and only re-fetched when we want newer data from McAuley Lab — so they use the S3 Standard-Infrequent Access storage class.

**Processed data**: In the same scrpt file we will save the filtered data parquet files to the data/processed folder within S3. We could use hive-partitioned parquet files for each week or month of new data so that we can compare new reviews to old reviews on the same product, for example.  

**Vector and BM25 index**: All the other pickle and index files will be saved under the data/processed folder in subfolders 'index' and 'pkl' respectively as well. At EC2 startup, the app downloads this directory to local disk and loads it into memory via FAISS.load_local(), which is already what semantic.py does. The bm25_index.pkl, corpus.pkl, produced by build_bm25_index) lives under s3://data/processed/index/bm25/. Same pattern as FAISS: downloaded once at EC2 startup, loaded into memory, and held there for the lifetime of the server.

### 2. Compute

**Where the app runs.** We package the Streamlit/FastAPI frontend plus the retrieval code into a single Python environment and run it on an **EC2 instance** (something like a `t3.large` or `m5.large` — we need enough RAM to hold the FAISS index, BM25 index, and `products.pkl` simultaneously, which is ~1–2GB for our dataset). The instance pulls the indexes from S3 at boot via a startup script, loads them into memory, and starts the app server.

**Concurrency (multiple users).** A single EC2 instance handling one request at a time would obviously not scale past a handful of users. We'd handle this in two layers:

1. **Within one EC2 instance:** run the app server (e.g. FastAPI with `uvicorn --workers 10`, or Streamlit with multi-session support) so multiple requests can be served in parallel on the instance's vCPUs. Because our indexes are read-only after load, there's no locking required across requests — they all share the same in-memory FAISS and BM25 objects safely.

2. **Across instances:** run an **Auto Scaling Group** of identical EC2 instances behind an **Elastic Load Balancer**. Each instance independently loads the indexes from S3 at boot and serves queries. The ASG scales out (adds instances) when average CPU crosses ~70% and scales in when it drops. Because all instances read the same read-only indexes from S3, and no instance writes any shared state, horizontal scaling is completely safe.

**LLM inference.** We use the **Groq hosted API** (`ChatGroq` calling `qwen/qwen3-32b`, as `rag_pipeline.py` already does) rather than hosting the LLM ourselves. Reasoning:

- Self-hosting a 32B-parameter model requires a GPU EC2 instance like `g5.xlarge` (roughly $1/hour) running 24/7 whether or not we're serving traffic. Groq charges per token, so we pay only for what we use.
- Groq's specialized inference hardware is faster than what we could achieve self-hosting on a commodity GPU.
- We avoid all the operational overhead of model weights, CUDA drivers, batching, and failover.

The `GROQ_API_KEY` and `TAVILY_API_KEY` are stored as environment variables on the EC2 instances, injected at launch from AWS Secrets Manager — never baked into the image.

### 3. Streaming / Updates

New Amazon products appear constantly, and reviews accumulate on existing ones. Assuming we need weekly updates we could do the following:

**Periodic full rebuild via EMR + Spark.** The full indexing pipeline would start with extracting new data from the datasets API, filtering and converting to hive-partitioned parquet files, read in filtered parquet files, join products with reviews, build a derived corpus, compute embeddings, write artifacts. This is a natural fit for Spark running on an EMR cluster because we can utilize parallel processing for chunking the data. We could do a scheduled rebuild (say, weekly) which would:

1. Spin up a transient EMR cluster with a primary node and a handful of core and task nodes (at least one primary node and two core nodes with 16 GB RAM each).
2. Run a PySpark version of our pipeline:
   - `spark.read.parquet("s3://data/raw/filtered_reviews.parquet")` — Spark reads directly from S3 and parallelizes across workers.
   - The `review_matching` step (top-k helpful reviews per product) is already a `groupBy` + `orderBy` + `head(k)` pattern — this maps cleanly onto Spark's DataFrame API and runs in parallel across workers, which matters because the reviews parquet is where the row-count is largest.
   - The corpus-building step (concatenating title + features + description + reviews into one string per product) is a simple per-row transformation that parallelizes nicely.
   - The embedding step (`model.encode(...)`) runs on Spark workers using a call so each worker loads the MiniLM model once and encodes its partition of the corpus. 
   - The BM25 step is harder to parallelize because `rank_bm25`'s `BM25Okapi` needs global IDF statistics. We'd do the tokenization in parallel on Spark workers, then collect the tokenized corpus to the driver and fit `BM25Okapi` there.
3. Write the new artifacts to a versioned S3 prefix like `s3://our-bucket/indexes/v2026-04-21/` and update a pointer file `s3://our-bucket/indexes/current.json`.
4. Tear down the EMR cluster (only paying for the hours it ran).