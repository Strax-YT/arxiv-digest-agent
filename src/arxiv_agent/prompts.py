"""All prompts live here so they can be diffed, reviewed and tuned in one place.

Three rules run through every template:
  1. The model never supplies bibliographic metadata — that comes from the API.
  2. The model must cite chunk labels for anything it asserts about the paper.
  3. "Not stated in the paper" is an explicitly *allowed and expected* answer.
"""

QUERY_PLAN_SYSTEM = "You convert research questions into arXiv search plans. Reply with JSON only."

QUERY_PLAN_USER = """Convert this research request into an arXiv search plan.

Request: {request}

Return JSON with exactly these keys:
{{
  "phrases": ["2-4 word technical concepts, most specific first, max 3"],
  "terms": ["single-word keywords and acronyms, max 6"],
  "categories": ["arXiv category codes such as cs.CL, cs.LG, cs.CV, stat.ML, max 3"],
  "authors": ["surnames only if the request names an author, else empty"],
  "since_year": 2023 or null,
  "notes": "one sentence on how you read the request"
}}

Rules:
- Use the paper's own vocabulary, not the user's paraphrase (e.g. "KV cache compression", not "making LLMs use less memory").
- Do not invent an author. Do not invent a year unless the request implies recency.
- Prefer 2 strong phrases over 5 weak ones."""

RERANK_SYSTEM = "You rank arXiv papers by how well they answer a research need. JSON only."

RERANK_USER = """Research need: {request}

Candidate papers:
{candidates}

Score each candidate 0.0-1.0 for how directly it addresses the research need.
A survey scores lower than a paper that proposes a concrete method, unless the
request asks for an overview.

Return JSON: {{"scores": [{{"index": 0, "score": 0.0, "reason": "max 12 words"}}]}}
Include every candidate exactly once."""

MAP_SECTION_SYSTEM = "You compress one section of a research paper into dense factual notes."

MAP_SECTION_USER = """Paper: {title}

Section: {section}

<section_text>
{text}
</section_text>

Write at most {bullets} factual bullet points capturing what THIS SECTION says.
Keep numbers, dataset names, model names, and baselines verbatim. No praise, no
speculation, no content from outside this section. If the section is boilerplate,
reply with a single bullet "- (no substantive content)"."""

BRIEFING_SYSTEM = (
    "You are a meticulous research analyst writing an executive briefing for a "
    "busy engineer. You only state what the provided text supports. Reply with JSON only."
)

BRIEFING_USER = """Write an executive briefing for this arXiv paper.

TITLE: {title}
CATEGORIES: {categories}
ABSTRACT:
{abstract}

EVIDENCE FROM THE PAPER (each block is labelled; cite these labels):
{evidence}

Return JSON with exactly these keys:
{{
  "why_it_matters": "one paragraph, 60-110 words, plain English, no jargon-for-its-own-sake, aimed at someone deciding whether to read the full paper",
  "problem_statement": "2-3 sentences: what breaks without this work",
  "method": ["4-7 bullets describing the actual mechanism, concrete enough that a reader could sketch the system"],
  "key_results": [{{"claim": "one specific result with its number/dataset", "evidence": "C3"}}],
  "limitations": ["3-5 bullets"],
  "follow_up_questions": ["4-6 questions a critical reader would ask next"],
  "confidence": "high | medium | low",
  "confidence_reason": "one sentence"
}}

Hard rules:
- Do NOT output the title, authors, arXiv id, date or URL. Those are filled in from arXiv metadata.
- Every entry in key_results must carry the label (e.g. "C3") of the evidence block it came from.
- limitations MUST be non-empty. Prefer limitations the authors state themselves. If the
  authors state none, give limitations that are evident from the setup (e.g. single language,
  one model family, no human evaluation) and prefix each with "(reviewer-inferred)".
- If the evidence is thin, say so via "confidence": "low" rather than padding with generalities."""

LIMITATIONS_RETRY_USER = """The briefing you produced for "{title}" had an empty or trivial
limitations list. That is not acceptable — every paper has limits of scope.

Evidence:
{evidence}

Return JSON: {{"limitations": ["3-5 specific bullets"]}}
Draw first on any text discussing failure cases, assumptions, "we leave to future work",
compute budgets, single-dataset evaluation, or model-size constraints. Mark anything not
stated by the authors with the prefix "(reviewer-inferred)"."""

CONDENSE_SYSTEM = "You rewrite a follow-up question into a standalone question. Reply with the question only."

CONDENSE_USER = """Conversation so far:
{history}

Follow-up question: {question}

Rewrite the follow-up as a standalone question that makes sense with no conversation
history, resolving pronouns like "it", "they", "that approach". Keep it short. Output
the rewritten question only, with no preamble."""

QA_SYSTEM = (
    "You answer questions about one specific research paper using only the retrieved "
    "excerpts you are given. You would rather say 'the paper does not say' than guess."
)

QA_USER = """Paper: {title} (arXiv:{arxiv_id})

Retrieved excerpts:
{context}

Question: {question}

Answer using ONLY the excerpts above.
- Cite the label of every excerpt you rely on, inline, like [C2]. An answer with no
  citation is invalid.
- If the excerpts do not contain the answer, reply exactly:
  "Not found in the retrieved sections of this paper." and then, in one sentence,
  say what the paper *does* discuss nearby, or which section would likely hold the answer.
- Do not use outside knowledge about this paper, its authors, or related work, even if
  you recognise it.
- Be direct. 2-6 sentences unless the question needs a list. Keep numbers exact."""

QA_STRICT_RETRY = """Your previous answer cited no excerpts, which is not allowed.

{original}

Re-answer using ONLY these excerpts and cite labels like [C1]:
{context}

Question: {question}"""
