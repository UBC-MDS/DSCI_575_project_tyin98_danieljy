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

### Code Quality Changes

- Deleted some placeholder files that are no longer needed
- All functions now have a doc string
- No hardcoded paths in source code
- No API keys in source code, `.env` file excluded in `.gitignore`
- Also excluded python and jupyter cache files

## Step 4: Cloud Deployment Plan
(See Step 4 above for required subsections)