# Milestone 2 Discussion

## Model selection

Through some exploration, we discovered that the model that we can run locally (Qwen3.5-0.8B) is not capable enough for our task. It is also quite slow (\~10 seconds, unsatisfactory for a search function). Since we lack a GPU, it is doubtful that using methods like quantization and sage attention would provide a significant enough boost to justify local models over API.

For this project, we picked qwen3-32b, because Groq provides a rate limit of 60 requests/6K token per minute and 1K requests/500K tokens per day, and the usage of our project should fit comfortably within that.

Another strong contender is kimi-k2-instruct, which will likely have better performance than Qwen3 32B. We decided to use Qwen because as a reasoning model, it can conveniently have the reasoning process outside of the main output, leaving only clean and structured responses if we need to (e.g. only the index or ASIN). The performance of Qwen3 32B seems adequate enough and we don't need to seek better models for this task.

## Designing the prompt template

In this step, we compared 3 different system prompts (see `milestone2_rag.ipynb`). The first one we tried is the most basic prompt with no extra designs. Even this version performed well and we didn't see any hallucinations. The second prompt is a variation of the first version with some anti-hallucination prompt. Even when the first version did not show any hallucinations, this version do seem to perform better, being slightly more concise. The last prompt is a slightly different task (explicitly ranking the top 3 products) with prompts for structured output. The model we choose seems to be able to consistently follow this structure, so this is the version we will be using in the pipeline.

We decided to use a k of 5. Selecting the best from the top 5 gives us a good chance of finding what we want. Including too few products could make us miss the best option, while including too many products could fill up the context window, be more expensive to run, and make the model confused. We found that using k = 10 would hit the token ceiling for the RAG model. 

### A note on RAG search result ordering

In the web app, the best option ranked by the LLM is not at the top of the retrieved products. This can be counterintuitive for the users but is by design. The retrieved products are ordered based on the semantic search scores, by keeping the RAG ranked products at their original position, we highlight the difference between relying only on semantic search and the RAG approach.

## Qualitative Evaluation of Hybrid RAG Pipeline

| Query | LLM Overall Recommendation | Accuracy (Yes/No) | Completeness (Yes/No) | Fluency (Yes/No) |
|-------|:-----------------:|:-----------------:|:---------------------:|:----------------:|
| acoustic guitar strings  | For most users, the Martin M535 Phosphor Bronze is the top choice due to its balanced performance and versatility. If you prefer a signature sound and lighter gauge, the Eric Clapton's Choice is excellent. For budget-conscious players needing frequent replacements, the Martin Authentic Acoustic bulk set offers great value. Avoid the Petift set (B07J66379B) unless you specifically need the winder/picks, as the strings lack the same professional-grade reputation. | yes | yes | yes |
| something to connect my guitar to an amp | If you need a full amp with a cable, the Fender Mustang I is ideal. For portable headphone practice, the Vox amPlug is excellent. If you play an acoustic guitar, the pickup is a niche but useful option. | yes | yes | yes |
| what drumsticks should I get for loud rock drumming without tiring my wrists | For loud rock drumming with wrist comfort, the Hornets Raw drumsticks (B0016CU0SI) are top-ranked due to their patented o-rings that reduce wrist strain and hickory construction for durability. Vic Firth NOVA (B08YNC2D3K) follows as 5B hickory sticks offer a balanced weight for powerful playing while remaining ergonomic. ChromaCast (B07BMWJBY2) rounds out third with dense hickory and a teardrop tip for loud, precise hits. Avoid FITSTIX (B07CZJHWHW) as they're lightweight fitness sticks, and the AW set (B014QW1FLA) is a beginner drum kit. Prioritize Hornets for long sessions and Vic Firth for classic rock power. | yes | somewhat | yes |
| what's a good weighted keyboard for a classical pianist transitioning from acoustic | For a classical pianist transitioning from acoustic, the Korg SP-250 (B000WGMPNM) is ideal due to its RH3 graded action keyboard and concert-grade piano sound, closely replicating acoustic piano feel. The Donner DEP-45 (B0BFCTV72R) follows as a portable option with semi-weighted keys and full 88-key size, though some reviews note uneven key responsiveness. The Hammond SK2 (B007L9OH5C) ranks third for its authentic Hammond organ and piano voices but is heavier and less focused on pure piano realism. Overall, prioritize the Korg SP-250 for its tactile accuracy and rich sound. | yes | yes | yes |
| digital piano 88 key | The Donner DEP-45 (B0BFCTV72R) is the top choice as it directly matches the query for an 88-key digital piano with features like portability, sound quality, and Bluetooth connectivity. The Fuystiulyo (B0BR8WMLPT) and AOYNG (B0BQHNMN6R) note guides are included as secondary options, though they are accessories rather than pianos. They are ranked lower because the query explicitly asks for a digital piano, but since no other pianos are listed, these guides are the next best matches for beginners needing learning tools. Overall, the Donner piano is the best standalone product for the request. | yes | yes | yes |
| best strings for a beginner acoustic guitarist who plays folk music | The Martin Guitar Eric Clapton's Choice strings (B0BBNCY2VC) are the top choice for a beginner folk guitarist due to their phosphor bronze construction, light gauge, and explicit design for folk/roots genres. The Yamaha FG700S bundle (B005PYAM5Y) follows as a strong second option since it includes steel-string acoustic guitar with a full-size body and solid tonewood, ideal for folk playing. The Music Alley MA-34-PNK (B07FNLCPVB) ranks third as a nylon-stringed starter guitar, though nylon strings are less common in folk music compared to steel strings. For pure folk authenticity, the Martin strings are unmatched, while the Yamaha offers a complete beginner setup with quality steel strings. | yes | yes | yes |
| what do I need to plug my first electric guitar into a small practice amp | The Squier by Fender bundle (B00D6U4SWM) ranks first as it provides a branded electric guitar with tremolo, a 10W practice amp, gig bag, and accessories, making it a complete beginner-friendly setup. The PylePro kit (B006O64JMY) follows due to its 10W amp, quality wood construction, and inclusion of a travel bag and tuner, though it lacks the Fender brand appeal. The TMS bundle (B01F7NI3SA) ranks third for its 10W amp and full-size guitar but has fewer accessories and less brand recognition. All three offer essential components for plugging in an electric guitar, with the top choices prioritizing brand reputation and comprehensive included items. | yes | yes | yes |
| strings that stay in tune longer | Prioritize D'Addario for bass players needing reliable tuning, GHS for banjo players seeking professional-grade stability, and Mxfans for lyre enthusiasts willing to invest time in tuning. | yes | yes | yes |
| sticks good for hitting hard  | For pure drumming or general hard-hitting needs, the On-Stage sticks are ideal. For specialized percussion or steel drum use, the Kakesa mallets excel. The Eastrock cowbell is best for adding bold, attention-grabbing sounds to a setup. | no | yes | yes |

## Evaluation Discussion

### Key observations
The Hybrid RAG pipeline performed strongly across most queries, with the majority of answers scoring Yes on all three dimensions. The LLM consistently produced fluent, well-structured responses and stayed grounded in the retrieved context without hallucinating. The one notable weakness was completeness — the "drumsticks" query received a "somewhat" rating because the answer included irrelevant products (a beginner drum kit) that diluted the response quality, suggesting the retriever occasionally surfaces loosely related products.

### Limitations
1. **Retrieval quality bottleneck:** 
The RAG pipeline can only be as good as the products retrieved. In the "sticks good for hitting hard" query, the system retrieved percussion mallets and a cowbell instead of drumsticks, leading to a factually inaccurate answer (Accuracy: No). Since the LLM is instructed to answer using only the provided context, a poor retrieval result directly produces a poor answer with no way to self-correct. For improved accuracy, one would need to build a larger product index (e.g., when running utils.py, we would use 'max-products 100,000'). 

2. **Fixed k and no query understanding:** 
The pipeline always retrieves exactly k=5 products regardless of query complexity or specificity. Broad queries like "digital piano 88 key" and narrow queries like "strings that stay in tune longer" are treated identically. This means some queries get too few relevant products while others get diluted with loosely related ones, as seen in the drumsticks and "sticks good for hitting hard" cases.

### Suggestions for improvement

- Re-ranking the retrieved documents before passing them to the LLM (e.g. using a cross-encoder model) would help filter out irrelevant products like the drum kit and cowbell that slipped through retrieval. 

- Uploading the FAISS index file to a cloud service (like HuggingFace) and using an API to call it would allow users to query the full product index ( ~1,000,000 products) without the slow processing locally. This would improve product suggestions and overall RAG output. 

- Additionally, query expansion — automatically rewriting the user's query into more specific search terms before retrieval — could improve recall for ambiguous queries like "sticks good for hitting hard", which the retrievers interpreted too literally.