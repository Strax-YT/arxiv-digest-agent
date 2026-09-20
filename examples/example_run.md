# Example run — offline demo

This is **real output from `python scripts/demo_offline.py`**, pasted verbatim.

It runs the actual state graph, the actual PyMuPDF parser, the actual chunker,
the actual vector store and the actual grounded-QA path. Two things are faked so
the demo needs no network, no API key and no local model:

* the arXiv API (a canned Atom feed, plus a small PDF generated at runtime), and
* the LLM (a scripted stand-in — see `DemoLLM` in the script).

Because the LLM is scripted, treat the *prose* below as placeholder text. What
the demo actually demonstrates is the machinery around it: query planning,
the relaxation ladder, hybrid ranking, section-aware parsing and chunking,
retrieval, citation verification, and abstention on an unanswerable question.

> For a live run against real arXiv with a real model, run
> `./scripts/capture_example.sh 2401.12345` and paste the result into
> `examples/real_run.md`.

```text
==============================================================================
INPUT:  recent work on KV-cache compression for LLMs
==============================================================================
Session s1789804117-dd5cdd · mock:demo

--- what the graph did ---
      ↳ LLM plan: wants recent method papers on shrinking the KV cache
  understand_query   ok       0.00s
      ↳ 2 candidates after 2 query variant(s)
  search_papers      ok       0.00s
      ↳ selected 2402.09876: PagedCache: Block-Sparse KV Cache Compression for Long-Context LLMs
  select_paper       ok       0.00s
      ↳ parser=pymupdf quality=0.98 sections=5 chars=4321 degraded=False
  fetch_and_parse    ok       0.01s
      ↳ embedded 7 chunks (avg 676 chars) into p_2402_09876_hashing_384 via hashing:384 on flat
  chunk_and_embed    ok       0.01s
      ↳ briefing written to /tmp/arxiv-demo-4i27neql/home/sessions/s1789804117-dd5cdd/briefing.md
  summarize          ok       0.00s

--- queries sent to the arXiv API ---
  L0: (all:"KV cache compression" AND all:"long context inference") AND (cat:cs.CL OR cat:cs.LG) AND submittedDate:[202401010000 TO 299912312359]  -> 2 hits
  L1: (all:"KV cache compression" AND all:"long context inference") AND (cat:cs.CL OR cat:cs.LG)  -> 2 hits

--- selection ---
  top of 2 candidates (score 0.716, gap to runner-up 0.259); components={'similarity': 0.67, 'arxiv_rank': 1.0, 'recency': 0.179, 'llm': 0.95}; LLM: concrete compression method
  runner-up: 0.457  A Survey of Memory-Efficient Inference

==============================================================================
# PagedCache: Block-Sparse KV Cache Compression for Long-Context LLMs

**arXiv:** [2402.09876v2](https://arxiv.org/abs/2402.09876)  
**Authors:** A. Researcher, B. Coauthor  
**Published:** 2024-02-15 (updated 2024-03-01)  
**Categories:** cs.CL, cs.LG  
**PDF:** http://arxiv.org/pdf/2402.09876v2  
**Author comment:** 14 pages, 6 figures  

---

## Why this paper matters

Long-context serving is memory bound, and this work shows the KV cache can be compressed in place rather than evicted, keeping recall while cutting memory 4x. That matters to anyone running long-context inference on fixed hardware budgets.

## Problem statement

The KV cache grows linearly with context length and dominates GPU memory. Existing eviction policies discard tokens permanently and hurt retrieval recall.

## Method / approach

- Partition the KV cache into fixed 64-token blocks
- Score blocks by mean attention mass over a 512-query sliding window
- Quantize low-scoring blocks to 4 bits with per-block affine scales
- Keep the top 12 percent of blocks in fp16

## Key results and claims

- 4.1x memory reduction at 128k context with 0.3 PPL degradation _[C4]_
- 1.7x throughput improvement at batch size 32 _[C4]_

## Limitations

- English-only evaluation
- Only two model families below 10B parameters
- No mixture-of-experts results

## Suggested follow-up questions

1. How does PagedCache interact with speculative decoding?
2. What is the overhead at short context lengths?

---

## Provenance

- Briefing confidence: **high** — method and results sections parsed cleanly
- PDF parser: `pymupdf`
- Pages parsed: `3/3`
- Chunks indexed: `7`
- Embedding model: `hashing:384`
- Vector store: `flat`
- LLM: `mock:demo`
- Session id: `s1789804117-dd5cdd`

_Ask follow-up questions with:_ `python -m arxiv_agent chat s1789804117-dd5cdd`

==============================================================================

you > How does the block scoring actually work?

PagedCache splits the KV cache into fixed blocks of 64 tokens and scores each block by the mean attention mass it receives over a sliding window of 512 queries [C1]. The top 12% of blocks stay in fp16; the rest are quantized to 4 bits with per-block affine scales [C1].

Sources:
  [C1] 3 Method (p.2-2) · sim 0.102
  grounded=True  abstained=False  top_sim=0.102

you > What datasets and baselines did they compare against?

Evaluation covers WikiText-103, LongBench and Needle-in-a-Haystack with Llama-3-8B and Mistral-7B, against H2O, StreamingLLM and a full-cache fp16 baseline [C1]. PagedCache reaches 4.1x memory reduction versus 2.2x for H2O at equal quality [C1].

Sources:
  [C1] 1 Introduction (p.1-2) · sim 0.14
  grounded=True  abstained=False  top_sim=0.14

you > What was the carbon cost of training?

Not found in the retrieved sections of this paper. The retrieved text covers memory and throughput measurements but never reports training or inference energy use.

(abstained — nothing similar enough was retrieved)
  grounded=True  abstained=True  top_sim=0.21

==============================================================================
session:    s1789804117-dd5cdd
state file: /tmp/arxiv-demo-4i27neql/home/sessions/s1789804117-dd5cdd/state.json
qa turns:   3  (all persisted)
==============================================================================
```
