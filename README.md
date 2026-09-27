# Local RAG with an Open-Source LLM

[Bahasa Indonesia](README.id.md)

A RAG pipeline that runs entirely on a Colab GPU: a Llama 3.1 8B model fine-tuned for Indonesian generates the answers, and nomic-embed-text-v1.5 handles retrieval. The notebook then evaluates it on five categories of questions, with and without a similarity threshold, to see where a small quantized model goes wrong.

## Setup

| Part | Choice |
|---|---|
| LLM | [`rubythalib33/llama3_1_8b_finetuned_bahasa_indonesia`](https://huggingface.co/rubythalib33/llama3_1_8b_finetuned_bahasa_indonesia), `unsloth.Q4_K_M.gguf` |
| Embedding | [`nomic-ai/nomic-embed-text-v1.5-GGUF`](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF), Q4_K_M, 768-dim |
| Runtime | llama-cpp-python 0.3.35 built with CUDA, all layers on GPU, `n_ctx=4096` |
| Prompt | raw completion in the Alpaca format the model was fine-tuned on |
| Retrieval | cosine similarity over 20 short documents about running, top-3 |

Four details matter more than they look:

- **CUDA build.** A plain `pip install llama-cpp-python` builds a CPU-only wheel, and `n_gpu_layers=-1` is then ignored without any error. The notebook checks `llama_supports_gpu_offload()` and stops if it is `False`.
- **`n_ctx`.** Left unset, the context window defaults to 512 tokens, which a RAG prompt plus a 256-token answer can already exceed.
- **Task prefixes.** nomic-embed expects `search_document: ` on stored text and `search_query: ` on questions.
- **Prompt format.** This GGUF carries no chat template, so `create_chat_completion` falls back to the Llama 2 format. The notebook calls the model as a raw completion with the Alpaca template instead.

## Results

All numbers come from the executed notebook, run on a Colab NVIDIA RTX PRO 6000 Blackwell GPU.

![Top-1 score vs threshold](images/top-score-vs-threshold.png)

### Without a threshold

| Category | Question | Answer |
|---|---|---|
| In-domain | Apa manfaat interval training? | Correct: improves speed and VO2 max capacity |
| Compound | apa itu lari dan apa saja jenis-jenisnya? | Lists run types, but the document about run types was **not** retrieved |
| Out-of-domain (AI term) | apa itu Rag dan bagaimana cara kerjanya | **"Rag adalah istirahat dan hari tanpa latihan..."** (hallucination) |
| Out-of-domain | Berapa harga saham Tesla hari ini? | Refused |
| Greeting | halo siapa kamu | Refused |

### With `THRESHOLD = 0.70`

| Category | LLM called | Answer |
|---|---|---|
| In-domain | yes (1 chunk) | Correct |
| Compound | yes (3 chunks) | Same issue as above |
| Out-of-domain (AI term) | **no** | Refused |
| Out-of-domain | **no** | Refused |
| Greeting | **no** | Refused |

Generation speed ranged from 112 to 182 tokens per second, including prompt evaluation. End-to-end latency was 0.09 to 0.49 seconds per question.

## What I learned

- **The dangerous question was the unfamiliar term, not the unrelated topic.** The model refused the stock-price question and the greeting on its own, but it glued the unknown word "Rag" onto whatever chunk came first and answered with confidence.
- **The prompt alone was not enough.** The instruction already told the model to reply with a fixed refusal sentence when the context did not contain the answer. The hallucination happened anyway.
- **The threshold is what stopped it, but only just.** Every question scored between 0.60 and 0.72. The lowest in-domain score (0.7134) and the highest out-of-domain score (0.6979) are only 0.0155 apart, so 0.70 works for these five questions and may not hold for the next one.
- **A threshold does not fix a retrieval miss.** The compound question passed the threshold, but the chunk it needed was not in the top 3, and the model filled the gap from its own knowledge.

## Limitations

- Five test questions are enough to show the failure modes, not to measure a hallucination rate.
- The threshold of 0.70 sits in a very narrow gap and was chosen from these same five questions.
- The GPU used here is much larger than the T4 commonly available on free Colab; throughput on smaller GPUs will be lower.

## Run it

1. Open `rag_open_source_llm_local.ipynb` in Google Colab with a GPU runtime.
2. Run the install cell (building llama-cpp-python with CUDA can take 10 to 20 minutes).
3. Run the evaluation without a threshold first, read the `top_score` column, then fill in `THRESHOLD` and continue.
4. The results are written to `hasil_rag_lokal.csv`.

## Tech stack

llama-cpp-python (CUDA) · Llama 3.1 8B fine-tuned for Indonesian (GGUF) · nomic-embed-text-v1.5 (GGUF) · NumPy · pandas · Google Colab

## License

[MIT](LICENSE)
