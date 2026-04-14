# Milestone 2 Discussion

## Model selection

Through some exploration, we discovered that the model that we can run locally (Qwen3.5-0.8B) is not capable enough for our task. It is also quite slow (\~10 seconds, unsatisfactory for a search function). Since we lack a GPU, it is doubtful that using methods like quantization and sage attention would provide a significant enough boost to justify local models over API.

For this project, we picked qwen3-32b, because Groq provides a rate limit of 60 requests/6K token per minute and 1K requests/500K tokens per day, and the usage of our project should fit comfortably within that.

Another strong contender is kimi-k2-instruct, which will likely have better performance than Qwen3 32B. We decided to use Qwen because as a reasoning model, it can conveniently have the reasoning process outside of the main output, leaving only clean and structured responses. The performance of Qwen3 32B seems adequate enough and we don't need to seek better models for this task.
