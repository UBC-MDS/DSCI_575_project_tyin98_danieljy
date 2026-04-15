# Milestone 2 Discussion

## Model selection

Through some exploration, we discovered that the model that we can run locally (Qwen3.5-0.8B) is not capable enough for our task. It is also quite slow (\~10 seconds, unsatisfactory for a search function). Since we lack a GPU, it is doubtful that using methods like quantization and sage attention would provide a significant enough boost to justify local models over API.

For this project, we picked qwen3-32b, because Groq provides a rate limit of 60 requests/6K token per minute and 1K requests/500K tokens per day, and the usage of our project should fit comfortably within that.

Another strong contender is kimi-k2-instruct, which will likely have better performance than Qwen3 32B. We decided to use Qwen because as a reasoning model, it can conveniently have the reasoning process outside of the main output, leaving only clean and structured responses fi we need to (e.g. only the index or ASIN). The performance of Qwen3 32B seems adequate enough and we don't need to seek better models for this task.

## Designing the prompt template

In this step, we compared 3 different system prompts (see `milestone2_rag.ipynb`). The first one we tried is the most basic prompt with no extra designs. Even this version performed well and we didn't see any hallucinations. The second prompt is a variation of the first version with some anti-hallucination prompt. Even when the first version did not show any hallucinations, this version do seem to perform better, being slightly more concise. The last prompt is a slightly different task (explicitly ranking the top 3 products) with prompts for structured output. The model we choose seems to be able to consistently follow this structure, so this is the version we will be using in the pipeline.

We decided to use a k of 10. Selecting the best from the top 10 gives us a good chance of finding what we want. Including too few products could make us miss the best option, while including too many products could fill up the context window, be more expensive to run, and make the model confused.

### A note on RAG search result ordering

In the web app, the best option ranked by the LLM is not at the top of the retrieved products. This can be counterintuitive for the users but is by design. The retrieved products are ordered based on the semantic search scores, by keeping the RAG ranked products at their original position, we highlight the difference between relying only on semantic search and the RAG approach.
