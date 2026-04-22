# Final Discussion

## Step 1: Improve Your Workflow

### Dataset Scaling

- Number of products used: Up to the user.

Currently the filtered parquet files contain about ~1,000,000 products. However, when building the indexes for BM25 and semantic search, we would need to use ~5000 products to have things build in less than ~5 minutes. So, we did not change the sampling method, but we focused on improving our utils.py file to be more efficient in generating the indexes for our semantic and BM25 models. 

We created a 'utils2.py' file which focuses on greater efficiency. The following is a summary of the changes relative to 'utils.py':

- review_matching: vectorized with sort + groupby.head(k) instead of per-row iterrows + heapq; returns {asin: [text, ...]} rather than (votes, text) tuples.
- Parquet load: reads only the 5 columns actually used downstream.
- Embedding device: explicit CUDA > MPS > CPU selection (upstream auto-detect skips MPS on some versions, silently falling back to CPU on Apple Silicon).
- Embedding precision: fp16 on CUDA (~2x faster on tensor cores, quality delta below retrieval noise); fp32 elsewhere since CPU/MPS fp16 is slower or flaky.
- Embedding batch size: tuned per device (512 CUDA / 128 MPS / 64 CPU) instead of a fixed 256 that can be suboptimal depending on the device.

With these changes, we can load 10,000+ products in under 5 minutes using a cpu and 16GB RAM. On a mid-level GPU, loading the full dataset took about 30 minutes.

- Changes to sampling strategy (if any): N/A

### LLM Experiment

I compared our current LLM (Qwen3-32B ) and a smaller more lightweight model, Llama-3.1-8B.  The Qwen model has 32 billion parameters and the Llama has 8 billion (so a quarter of the parameters). 

**Results**:
| Query | Qwen3-32B Output | Llama-3.1-8B Output |
|-------|------------------|---------------------|
| quiet instrument for apartment practice so I don't annoy my neighbors | The top three products for quiet apartment practice are the KIBAGA Silent Drum Practice Pad (B08GQXTHRH), Meinl Drum Mute (B083YRDJ3P), and PAMPET Trumpet Practice Mute (B0796WYW7D). The drum pad (4.8/5) is rated highest for eliminating noise during snare drum practice with a realistic rebound, making it ideal for late-night sessions. The Meinl drum mute (4.6/5) effectively dampens drum volume while maintaining playability, perfect for full drum kits. The PAMPET trumpet mute (4.1/5) is praised for drastically reducing brass instrument noise without distorting sound, essential for trumpet/cornet players. These products prioritize noise reduction while retaining practice quality, making them optimal for apartment living. | I ranked these products as the top 3 because they are all designed to reduce noise levels during instrument practice, making them suitable for apartment use. The Meinl Percussion 18” Drum Mute (B083YRDJ3P) is a top choice because it is specifically designed for drummers, allowing them to practice quietly without disturbing neighbors. It is made of soft rubber, which greatly reduces attack and volume, making it an excellent option for apartment dwellers. The SUNTRADE Lightweight Aluminum Practice Trumpet Mute Silencer (B07N8WY2LS) is another top choice because it is designed for trumpet players and allows them to practice quietly without disturbing others. It is lightweight and easy to use, making it a great option for those who want to practice their instrument without disturbing others.cThe PAMPET Lightweight Plastic Practice Trumpet Mute Silencer (B0796WYW7D) is also a great option because it is designed for trumpet players and allows them to practice quietly. It is made of lightweight plastic and is easy to use, making it a great option for those who want to practice their instrument without disturbing others.cOverall, I highly recommend these products for anyone who wants to practice their instrument quietly in an apartment setting. |
| gift for my 8 year old niece who just started music lessons | The Donner Kids Guitar (B09FJY76PN) is ranked first because it’s a 3-string mini acoustic guitar specifically designed for beginners, with free lessons and accessories included. Its simplicity and age-appropriate size make it ideal for an 8-year-old just starting music lessons. The SAFEPLUS Full Size Acoustic Guitar (B07KF76Y3F) follows as a strong second choice, offering a full-sized instrument with a lightweight build and all necessary accessories, suitable for a child who may outgrow smaller instruments quickly. The Luvay Kalimba (B06WP475Y4) ranks third as a unique, portable alternative that’s easy to learn and has a high rating for kid-friendly appeal. Overall, the Donner Kids Guitar is the most practical and engaging option for a beginner, while the Kalimba adds a creative twist for variety. | I ranked these products as the top three gift options for your 8-year-old niece who just started music lessons. Donner Kids Guitar 3 String Mini Acoustic Guitar for Beginner (B09FJY76PN) - This product is a great gift for a beginner, as it's easy to play and comes with a tuner, pick, and other accessories. The fact that it's a 3-string guitar makes it perfect for learning the basics of guitar playing. Toys for 1-10 Year Old Girls Gifts Best Birthday for Kids Boys Girls Toys Age 2 3 4 5 6 7 8 9 Karaoke Microphone for Kids - Rose Gold (B07H4VPCL4) - This product is a fun and interactive way for your niece to express her love for music. It's easy to carry and use, and the Bluetooth and wireless microphone feature make it a great gift for kids who love to sing and perform. I Couldn’t Pick A Better Nephew Musical Guitar Pick Jewelry Gift for Nephew From Aunt Uncle Birthday Christmas Gifts Musician Guitar Player Gifts (B08NY7R738) - This product is a thoughtful and unique gift for your niece who loves music. It's a musical guitar pick that can be worn as a necklace or a charm, and it's a great way to show off her love for music. Overall, I recommend the Donner Kids Guitar 3 String Mini Acoustic Guitar for Beginner (B09FJY76PN) as the top gift option for your 8-year-old niece who just started music lessons. It's a great instrument for beginners, and it comes with all the accessories she needs to start playing music. |
| drum sticks for my left-handed 12 year old son who plays metal | For a left-handed 12-year-old playing metal, the On Stage 5A Maple Drum Sticks (B00H7Y2ZKK) are ideal due to their durability and nylon tips, which handle heavy metal playing. The Vic Firth Kidsticks (B0745KXRSL) are shorter and lighter, suitable for younger players but may still work for a 12-year-old who prefers lightweight sticks. The GSHLLO Lollipop Drum Mallets (B08PPC9687) are softer and safer for practice but less suited for metal's intensity. Overall, prioritize the On Stage 5A for metal's demands, then consider the Vic Firth for comfort, and the GSHLLO for casual play. | I ranked these products as the top three based on the query's requirements. GSHLLO 4 PCS Lollipop Drum Mallet Sticks Hammer Percussion Sticks Drumstick with Wood Handle for Kids (B08PPC9687): Although this product is marketed as a kid-friendly drum mallet, it seems to be a good option for a left-handed 12-year-old metal drummer. The product has a 4.5-star rating and is made of wood and plastic, which might provide a good balance between durability and feel. Vic Firth Kidsticks w/ PINK FINISH (B0745KXRSL): This product is designed for younger drummers, but it might still be suitable for a left-handed 12-year-old metal drummer. The product has a 4.8-star rating and is made of premium USA Hickory, which is a good material for drumsticks. Ludwig LC178X025 Questlove Pocket Kit 4-Piece Drum Set-Red Wine Sparkle Finish (B01F4SU520): Although this product is a drum set, not just drumsticks, it seems to be a good option for a left-handed 12-year-old metal drummer. The product has a 4.7-star rating and is designed for ages 4 through 10 years old, but it might still be suitable for a 12-year-old. Overall, I would recommend the GSHLLO 4 PCS Lollipop Drum Mallet Sticks Hammer Percussion Sticks Drumstick with Wood Handle for Kids (B08PPC9687) as the top product, as it seems to be the most suitable option for a left-handed 12-year-old metal drummer. However, it's essential to note that the product is marketed as a kid-friendly drum mallet, and it might not provide the same level of performance as a traditional drumstick. |
| cheap microphone under $50 for podcasting  | The SOUIDMY USB Microphone (B085ZM5GNS) ranks first due to its high 4.2/5 rating, positive reviews highlighting excellent sound quality and ease of use for podcasting, and features like a mute button and headphone jack that are ideal for streaming/recording. The Avanson USB Condenser Microphone (B09DFN4753) follows at second place with a 4.2/5 rating, praised for its budget-friendly performance and metal build, though some note minor build quality concerns. The QUANFUN Lavalier Microphone (B08MVS3ZS6) is third, offering a 3.9/5 rating and versatility for multiple devices, though lavalier mics may be less optimal for traditional podcasting setups compared to desktop options. Overall, the SOUIDMY is the strongest recommendation for its balance of affordability, functionality, and podcast-specific features. | I ranked these products as the top 3 based on their relevance to the query "cheap microphone under $50 for podcasting". Avanson USB Condenser Microphone (B09DFN4753) - This product is a budget-friendly USB microphone that is highly recommended for its audio quality and is suitable for podcasting. It has a rating of 4.2/5 and is priced under $50.  USB Microphone (CVL-2004) (B01ERLN4MI) - This product is a professional condenser mic that is suitable for podcasting and has a rating of 3.2/5. It is priced under $50 and has good audio pick-up quality. Lavalier Microphone, QUANFUN (B08MVS3ZS6) - This product is a lavalier microphone that is suitable for recording podcasts and has a rating of 3.9/5. It is priced under $50 and comes with a 2-pack bundle. Overall, I recommend the Avanson USB Condenser Microphone (B09DFN4753) as the top pick due to its high rating and suitability for podcasting. |
| I keep breaking high E strings, what am I doing wrong and what should I buy  | The Hricane 20-Pack High E strings (B098RRDHPS) are ranked first because they directly address the user’s issue of frequently breaking high E strings. The bulk pack offers cost-effective replacements, and the product emphasizes durability with rustproof coating and "good toughness," aligning with the review noting only 2 strings used out of 20 despite frequent breakage. The Martin Darco Electric D930 set (B0886ZP9V6) follows as a second choice, offering extra-light gauge strings (including a .009 high E) at a low price, though reviews mention they are basic and may not match premium options. The D'Addario EXL110 3-pack (B000XZZ2U0) ranks third due to its 5.0 rating and reputation for quality, though its .010 high E gauge is slightly heavier than the user’s current .010, and the full set may be unnecessary if only the high E is problematic. Recommendation: Prioritize the Hricane pack for targeted, budget-friendly replacements. If the user wants a full set of extra-light strings (e.g., for tuning stability), the Martin Darco set is a solid value. For premium quality, the D'Addario set is ideal but may cost more than needed for a single string issue. Also, consider checking for guitar burrs or technique adjustments as mentioned in the context. | I ranked these products as the top three based on their relevance to the query.Hricane 20-Pack Economy Single Acoustic Guitar Strings Bulk (.010 High E) - This product is specifically designed for replacing high E strings, which is the exact issue the query is facing. It's a bulk pack, making it a great value for the price. The reviews also mention that these strings are reliable and less prone to breaking.D'Addario EXL110 Electric Guitar Strings - 3 pack - This product is a popular choice among guitar players and is known for its bright tone and excellent intonation. Although it's not specifically designed for high E strings, it's a great option for players who experience frequent string breakage. Stringjoy BAL98 8 String Signatures Nickel Electric Guitar Strings - This product is designed for 8-string electric guitars, but its features, such as balanced tension and smooth performance, make it a great option for players who experience inconsistent tone and string breakage. Overall, I recommend the Hricane 20-Pack Economy Single Acoustic Guitar Strings Bulk (.010 High E) as the top choice. It's specifically designed for replacing high E strings and is a great value for the price. |

**Observations:**
Across all five queries, Qwen3-32B consistently produces higher-quality responses compared to the Llama model. Qwen follows the three-product ranking structure cleanly, justifies each pick with specific product attributes (gauge, material, rating, feature fit), and frequently adds meta-guidance that goes beyond the catalog (e.g., flagging guitar burrs or technique issues for the broken-string query, or noting that lavalier mics are suboptimal for traditional podcasting). Llama-3.1-8B, by contrast, visibly struggles with constraint filtering and relevance judgment — it surfaces a karaoke microphone and a guitar-pick necklace as music-lesson gifts for an 8-year-old, recommends a full drum kit (rated for ages 4–10) for a 12-year-old metal drummer, and suggests an 8-string electric guitar string set for a high-E breakage problem. Its justifications are also noticeably shallower, often just restating ratings and price without connecting features to the specific query, and its prose has mechanical artifacts like stray "c" characters between sentences.

- Which model you chose and why
 Overall, we decided to stick with the Qwen model. While both models reliably emit the rank format and avoid leaking ASINs into the prose body, Qwen dealt much better with ambigious queries. Retrieval carries the 8B on narrow queries where the top candidates are obviously right, but on anything requiring constraint juggling (age-appropriateness, genre fit, problem diagnosis), Qwen's reasoning pass produces meaningfully better picks.

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