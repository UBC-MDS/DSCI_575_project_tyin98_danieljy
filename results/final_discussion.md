# Final Discussion

## Step 1: Improve Your Workflow

### Dataset Scaling
- Number of products used
- Changes to sampling strategy (if any)

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