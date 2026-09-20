# 4-minute video reflection — talking points

Not a script to read aloud. These are the things worth spending your four
minutes on, with rough timings. They cut hard because the reviewers stop
watching at 4:00.

**Open with the demo, not the architecture (0:00–0:35).**
Screen-share a live run. Type the topic, let it find the paper, show the
briefing, ask the two good questions and then the unanswerable one so the
abstention is visible on screen. Say almost nothing over it. A working thing is
the strongest first 30 seconds you have.

**Why the graph is hand-written (0:35–1:20).**
The honest version: nine nodes, simple routing, ~150 lines of runtime. The
dependency would have cost more to explain than to replace, and I wanted the
retry policy, the criticality flags and the checkpoint boundary visible in one
file. Mention that the public API mirrors LangGraph's so the port is mechanical,
and that routing lives outside the nodes — which is why `python -m arxiv_agent
graph` can draw the diagram from live wiring instead of me maintaining a picture
that rots. Show that command running if you have the seconds.

**The one design decision I'd defend hardest (1:20–2:15).**
Pick grounding. Walk the four layers quickly: the similarity gate that abstains
*before* any LLM call, the context-only prompt, the citation contract that gets
verified against what was actually retrieved, and MMR so the model isn't looking
at five copies of the same paragraph. The line to land: an answer that cites a
chunk id I can check is a different kind of object from an answer that sounds
right. Mention that bibliographic fields never come from the model at all.

**Failure handling (2:15–3:00).**
Two things. The relaxation ladder — show `query_attempts` in the output so they
see the agent trying five renderings before giving up, and giving up cleanly
rather than inventing a paper. Then `fetch_and_parse` being the one deliberately
non-critical node: broken PDF degrades to abstract-only, labels itself, forces
confidence to low. Say the contract out loud: that node never ends the run.

**What I'd do next (3:00–3:40).**
Lead with evaluation, because it's the real gap: right now correctness is
asserted structurally (is it grounded, does the citation exist) but not
semantically (is it right). A 20-paper golden set with retrieval hit-rate@k.
Then hybrid BM25 + dense, because dense embeddings are weak on exactly the
tokens people ask about — model names, dataset names, hyperparameters — and you
can see that failure in the demo. Then table extraction.

**Close (3:40–4:00).**
One sentence on something the tests caught that you hadn't predicted — e.g. the
briefing normaliser silently dropping dict-shaped `key_results` through a length
filter meant for junk bullets. It shows the tests are load-bearing rather than
decorative.

---

## Things to cut if you're running long

- The provider abstraction. It's four HTTP calls; nobody needs 40 seconds on it.
- The chunker's sentence-boundary overlap. Interesting, but it's a detail.
- Listing directory structure. They can read.

## Things to have on screen, not in speech

- `python -m arxiv_agent graph` output
- a `state.json` open in an editor, scrolled to `trace` and `qa_history`
- the Provenance block at the bottom of a briefing

## If they ask a follow-up, the likely three

1. *Why per-paper collections?* Re-embedding is wasted work, and namespacing by
   embedder name means changing models can't mix two vector spaces.
2. *Why is the overlap sentence-based?* So every chunk begins at a readable
   boundary; a raw character slice starts mid-word and embeds badly.
3. *What happens on a 200-page paper?* Page cap at 60 with truncation reported,
   then map-reduce over sections in importance order, so the appendix is what
   gets dropped rather than the method.
