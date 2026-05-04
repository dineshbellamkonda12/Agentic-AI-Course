# Week 7: RAG Pipelines

> **Goal:** Build production-quality Retrieval-Augmented Generation (RAG) systems. This is where embeddings, search, and LLMs come together to create the apps that companies actually pay for.

---

## Table of Contents

1. [What Is RAG and Why Does It Matter](#what-is-rag-and-why-does-it-matter)
2. [The Full RAG Architecture](#1-the-full-rag-architecture)
3. [Chunking Strategies](#2-chunking-strategies)
4. [Hybrid Search (BM25 + Dense)](#3-hybrid-search-bm25--dense)
5. [Reranking](#4-reranking)
6. [Query Rewriting and Expansion](#5-query-rewriting-and-expansion)
7. [Citations and Source Tracking](#6-citations-and-source-tracking)
8. [Putting It All Together](#7-putting-it-all-together)
9. [Common RAG Mistakes](#8-common-rag-mistakes)
10. [Final Project](#final-project)
11. [Self-Check Questions](#self-check-questions)

---

## What Is RAG and Why Does It Matter

### The Problem RAG Solves

LLMs are smart but have three big limitations:
1. **No knowledge of your private data** (your company docs, your codebase, etc.)
2. **Can't access fresh information** (anything after training cutoff)
3. **Hallucinate** when they don't know something

### The RAG Solution

**Retrieval-Augmented Generation:**

1. **Retrieve** relevant documents from your data
2. **Augment** the LLM's prompt with these documents
3. **Generate** an answer based on the documents

### A Simple Example

**Without RAG:**
```
User: "What's our company's vacation policy?"
LLM: "I don't have access to your company's specific policies..."
```

**With RAG:**
```
User: "What's our company's vacation policy?"
↓ (RAG system retrieves the HR doc)
LLM gets: "Question: What's our vacation policy?
          Context: [your HR policy doc]"
LLM: "According to your HR policy, employees get 25 days of paid leave..."
```

### Why RAG Beats Fine-Tuning (Most of the Time)

| | Fine-Tuning | RAG |
|---|------------|-----|
| **Update knowledge** | Re-train (expensive) | Update DB (instant) |
| **Hallucinations** | Still happens | Cite sources to verify |
| **Cost to set up** | High (GPU time) | Low (just API calls) |
| **Custom domain knowledge** | Good | Excellent |
| **Audit trail** | None | Full source citations |

For 90% of "AI on my data" use cases — **use RAG, not fine-tuning.**

---

## 1. The Full RAG Architecture

### The Big Picture

There are two phases:

**Phase 1: Indexing (done once)**
```
Documents → Clean → Chunk → Embed → Store in Vector DB
```

**Phase 2: Query (every user question)**
```
Query → Rewrite → Retrieve → Rerank → Build Prompt → LLM → Cite Sources
```

### The Production-Grade Diagram

```
INDEXING:
  Raw Docs (PDFs, Markdown, etc.)
    ↓
  Clean & Parse
    ↓
  Chunk (200-500 tokens each)
    ↓
  Embed with OpenAI/Cohere
    ↓
  Store in Vector DB (Qdrant/pgvector)
    + BM25 index (for hybrid search)


QUERY:
  User Question
    ↓
  Rewrite (optional: expand, decompose)
    ↓
  ┌─────────────┬──────────────┐
  Vector search   BM25 search    (parallel)
  ↓               ↓
  └──────┬────────┘
         ↓
  Reciprocal Rank Fusion (combine results)
         ↓
  Rerank top-50 → top-5
         ↓
  Build prompt with context
         ↓
  LLM generates answer
         ↓
  Add citations
         ↓
  Return to user
```

This week we'll build each component, then put them all together.

---

## 2. Chunking Strategies

### Why Chunking Matters

You can't put a whole 200-page PDF into one embedding — the meaning gets averaged out into mush. You also can't put it into one LLM prompt (context window).

**The solution:** Split documents into smaller, meaningful chunks.

### How Big Should Chunks Be?

Most production RAG systems use **200-500 tokens per chunk** (roughly 150-400 words).

Tradeoffs:

| Chunk size | Pros | Cons |
|-----------|------|------|
| Small (100 tokens) | Precise retrieval | Loses context |
| Medium (300 tokens) | Balanced | (Sweet spot for most cases) |
| Large (1000 tokens) | Rich context | Less precise, more noise |

### Chunking Strategy 1: Fixed Size

Simple. Just split every N characters.

```python
def chunk_fixed(text: str, chunk_size: int = 1000, overlap: int = 100) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap  # Overlap for context continuity
    return chunks
```

**Use when:** Quick prototyping. Don't use in production.

### Chunking Strategy 2: Recursive Character

Splits by paragraphs first, then sentences, then words. Avoids breaking in the middle of meaningful text.

```python
def chunk_recursive(text: str, chunk_size: int = 1000, overlap: int = 100) -> list[str]:
    separators = ["\n\n", "\n", ". ", " ", ""]
    return _recursive_split(text, separators, chunk_size, overlap)


def _recursive_split(text: str, separators: list[str], chunk_size: int, overlap: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    sep = separators[0]
    parts = text.split(sep) if sep else list(text)

    chunks = []
    current = ""
    for part in parts:
        candidate = current + sep + part if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(part) > chunk_size:
                # Recurse with finer separator
                chunks.extend(_recursive_split(part, separators[1:], chunk_size, overlap))
                current = ""
            else:
                current = part

    if current:
        chunks.append(current)

    # Add overlap between adjacent chunks
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i-1][-overlap:] if len(chunks[i-1]) > overlap else chunks[i-1]
            overlapped.append(prev_tail + chunks[i])
        chunks = overlapped

    return chunks
```

**Use when:** Most general-purpose RAG. **This is the default.**

### Chunking Strategy 3: Semantic Chunking

Embed each sentence, then group sentences that are semantically similar.

```python
import numpy as np
from openai import OpenAI

client = OpenAI()

def semantic_chunk(text: str, threshold: float = 0.7) -> list[str]:
    sentences = [s.strip() for s in text.split(". ") if s.strip()]
    if not sentences:
        return []

    # Embed each sentence
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=sentences
    )
    embeddings = [d.embedding for d in response.data]

    # Group consecutive sentences with high similarity
    chunks = []
    current = [sentences[0]]
    for i in range(1, len(sentences)):
        sim = cosine_similarity(embeddings[i-1], embeddings[i])
        if sim > threshold:
            current.append(sentences[i])
        else:
            chunks.append(". ".join(current) + ".")
            current = [sentences[i]]

    if current:
        chunks.append(". ".join(current) + ".")

    return chunks


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
```

**Use when:** Documents have mixed topics that should be separated.

**Cost:** Higher (one embedding per sentence during indexing).

### Chunking Strategy 4: Document-Aware

Use the document's natural structure (headings, sections, code blocks).

```python
import re

def chunk_markdown(text: str) -> list[dict]:
    """Chunks markdown by H2 sections, keeping the heading as context."""
    sections = re.split(r'\n## ', text)
    chunks = []

    for section in sections:
        if not section.strip():
            continue

        lines = section.split('\n')
        heading = lines[0].strip()
        content = '\n'.join(lines[1:]).strip()

        if content:
            chunks.append({
                "heading": heading,
                "content": f"## {heading}\n\n{content}"
            })

    return chunks
```

**Use when:** Working with structured documents (Markdown, code, technical docs).

### Practical Recommendation

For your final project this week:
1. **Start with recursive character splitting** (300-500 token chunks, 50 token overlap)
2. **Use document-aware chunking** if your docs are well-structured (Markdown, etc.)
3. **Try semantic chunking** if you have time (it's slower but often better)

### LangChain's Text Splitter (Easy Mode)

You don't have to write chunking from scratch:

```bash
pip install langchain-text-splitters
```

```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)

chunks = splitter.split_text(your_long_document)
```

This is what most production RAG systems use.

---

## 3. Hybrid Search (BM25 + Dense)

### The Core Insight

Vector search is great at semantics ("fix car" → "auto repair"), but **bad at exact matches**:

```
Query: "Error code E-1234"
Vector search: returns generic error articles, missing the exact code
BM25: nails it because "E-1234" appears literally
```

**Solution:** Combine both! This is **hybrid search**.

### What's BM25?

BM25 (Best Match 25) is the classic keyword search algorithm — what Google used before deep learning. It's based on term frequency.

For our purposes, BM25 = **exact keyword matching with smart scoring**.

### The Hybrid Strategy

1. Run vector search → get top 50
2. Run BM25 search → get top 50
3. Combine results with **Reciprocal Rank Fusion (RRF)**

### Reciprocal Rank Fusion (RRF)

RRF gives a score based on a document's **rank** in each list. Documents ranked highly in both lists get the best combined score.

```python
def reciprocal_rank_fusion(
    rankings: list[list[str]],   # Each list is [doc_id1, doc_id2, ...] in rank order
    k: int = 60
) -> list[tuple[str, float]]:
    """RRF: combines multiple ranked lists into one."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)

    # Sort by combined score
    return sorted(scores.items(), key=lambda x: -x[1])


# Example
vector_results = ["doc_3", "doc_1", "doc_5", "doc_7"]
bm25_results = ["doc_5", "doc_3", "doc_2", "doc_8"]

combined = reciprocal_rank_fusion([vector_results, bm25_results])
# doc_3 and doc_5 rank highest because they're in both lists
```

### Implementing BM25

The `rank-bm25` library is simple:

```bash
pip install rank-bm25
```

```python
from rank_bm25 import BM25Okapi

# Tokenize your docs
docs = ["The quick brown fox", "Lazy dogs sleep", "Pizza is from Italy"]
tokenized_docs = [doc.lower().split() for doc in docs]

# Build BM25 index
bm25 = BM25Okapi(tokenized_docs)

# Search
query = "quick fox".lower().split()
scores = bm25.get_scores(query)

# Get top results
import numpy as np
top_indices = np.argsort(scores)[::-1][:5]
top_results = [(docs[i], scores[i]) for i in top_indices]
```

### Production Tip: Use Postgres for BM25

If you're using pgvector, Postgres has built-in full-text search. Combine vector + full-text in one query:

```sql
SELECT
    content,
    ts_rank_cd(to_tsvector('english', content), query) AS bm25_score,
    1 - (embedding <=> %s::vector) AS vector_score
FROM documents, plainto_tsquery('english', %s) query
WHERE to_tsvector('english', content) @@ query
   OR embedding <=> %s::vector < 0.5
ORDER BY (bm25_score + vector_score) DESC
LIMIT 20;
```

---

## 4. Reranking

### The Problem

Even hybrid search returns noisy results. Top-50 contains the right answer, but it might be at position 30, not 1.

### The Solution: Rerank

Use a more powerful (and slower) model to re-score the top results.

```
Initial search: top 50 docs (fast, slightly noisy)
       ↓
Reranker scores each one carefully
       ↓
Final top 5 docs (slow but very precise)
```

### Why Two Stages?

- **Vector/BM25 search:** Fast, scales to millions of docs, but approximate
- **Reranker:** Slow, won't scale to millions, but very precise on small sets

By using both, you get speed AND accuracy.

### Cohere Rerank (Easiest)

```bash
pip install cohere
```

```python
import cohere

co = cohere.Client(api_key="...")

def rerank(query: str, documents: list[str], top_k: int = 5) -> list[dict]:
    response = co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=documents,
        top_n=top_k
    )

    return [
        {
            "index": r.index,
            "document": documents[r.index],
            "score": r.relevance_score
        }
        for r in response.results
    ]


# Example
query = "How do I fix a Python import error?"
candidates = [
    "Python uses 'import' keyword for modules",
    "ImportError occurs when modules can't be found",
    "Pizza is delicious",
    "To fix import errors, check your PYTHONPATH",
    "Common ImportError causes: typos, missing packages, circular imports"
]

reranked = rerank(query, candidates, top_k=3)
for r in reranked:
    print(f"{r['score']:.3f}: {r['document']}")

# Top results will be the relevant ImportError ones, regardless of original order
```

### Open-Source Alternative: BGE Reranker

Free, runs locally, near-Cohere quality.

```bash
pip install sentence-transformers
```

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-large")

def rerank_local(query: str, docs: list[str], top_k: int = 5) -> list[tuple]:
    pairs = [(query, doc) for doc in docs]
    scores = reranker.predict(pairs)

    indexed = list(enumerate(scores))
    indexed.sort(key=lambda x: -x[1])

    return [(docs[i], float(s)) for i, s in indexed[:top_k]]
```

### When to Rerank

Always rerank if you can afford the latency. Reranking typically improves retrieval accuracy by **10-30%**.

The latency cost is small (~100-300ms for 20 docs), but the accuracy gain is huge.

### Cost Comparison

| Method | Cost (per 1K queries with 20 docs each) | Latency |
|--------|----------------------------------------|---------|
| No rerank | $0 (just embedding) | ~100ms |
| Cohere rerank | ~$0.20 | ~200-400ms |
| Local BGE rerank | $0 (own GPU) | ~100-200ms |

---

## 5. Query Rewriting and Expansion

### The Problem with Raw Queries

Users write messy, ambiguous, or terse questions. RAG works much better if you rewrite them first.

### Technique 1: Query Rewriting

Rewrite the user's vague query into something clearer.

```python
async def rewrite_query(user_query: str) -> str:
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Rewrite this user query into a clearer, more searchable question.
Keep all important keywords. Add context if needed.

User query: "{user_query}"

Rewritten query:"""
        }],
        temperature=0
    )
    return response.choices[0].message.content.strip()


# Example
await rewrite_query("how fix it")
# Output: "How do I fix the [specific issue from context] error?"
```

### Technique 2: Query Expansion (Multiple Queries)

Generate multiple variations of the query, search with each, combine results.

```python
async def expand_query(user_query: str, n: int = 3) -> list[str]:
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Generate {n} different ways to phrase this question for searching.
Vary the wording to capture different terminology.

Question: "{user_query}"

Output as a JSON list of strings."""
        }],
        response_format={"type": "json_object"},
        temperature=0.5
    )

    import json
    data = json.loads(response.choices[0].message.content)
    return data.get("queries", [user_query])


# Example
await expand_query("how to debug Python")
# Output:
# [
#   "How to find errors in Python code",
#   "Python debugging techniques",
#   "Tools for fixing Python bugs"
# ]
```

### Technique 3: HyDE (Hypothetical Document Embeddings)

Instead of embedding the question, **generate a fake answer** and embed THAT. Then search for similar documents.

Why? Because the fake answer looks more like a document than a question does.

```python
async def hyde_search(query: str) -> str:
    # Generate hypothetical answer
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"Write a paragraph that would answer this question:\n\n{query}"
        }],
        temperature=0
    )
    hypothetical_answer = response.choices[0].message.content

    # Embed THAT (not the question)
    return await embed_text(hypothetical_answer)
```

**When HyDE helps:** Short, vague queries. Underperforms on simple keyword queries.

### Technique 4: Decomposition (For Complex Queries)

Break a multi-part query into subqueries, answer each, combine.

```python
async def decompose_query(query: str) -> list[str]:
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": f"""Break this complex question into 2-4 simpler sub-questions
that can be answered independently.

Question: "{query}"

Output as JSON: {{"sub_questions": ["...", "..."]}}"""
        }],
        response_format={"type": "json_object"},
        temperature=0
    )

    import json
    return json.loads(response.choices[0].message.content)["sub_questions"]


# Example
await decompose_query(
    "Compare the pricing and features of OpenAI and Anthropic APIs"
)
# Output:
# [
#   "What is OpenAI's API pricing?",
#   "What features does OpenAI's API offer?",
#   "What is Anthropic's API pricing?",
#   "What features does Anthropic's API offer?"
# ]
```

### When to Use What

| Query type | Technique |
|-----------|-----------|
| Vague, terse | Rewrite |
| Simple but uses unusual terms | Expand |
| Conceptual question | HyDE |
| Complex / multi-part | Decompose |
| Most production cases | Just rewrite (it's cheap and helps a lot) |

---

## 6. Citations and Source Tracking

### Why Citations Matter

In production RAG, citations are **non-negotiable**:

1. Users need to verify the answer
2. Hallucinations get caught
3. Compliance/audit requirements
4. Trust

### How to Add Citations

#### Method 1: Numbered References

```python
def build_context_with_refs(chunks: list[dict]) -> str:
    """Format chunks with [1], [2], [3] markers."""
    context = ""
    for i, chunk in enumerate(chunks, 1):
        source = chunk["metadata"].get("source", "unknown")
        context += f"[{i}] (Source: {source})\n{chunk['content']}\n\n"
    return context


PROMPT_TEMPLATE = """
Answer the question using only the context below.
After each fact, cite the source as [1], [2], etc.

CONTEXT:
{context}

QUESTION: {question}

ANSWER:
"""
```

The LLM produces:
```
The vacation policy allows 25 days of paid leave [1].
Sick days are unlimited but require a doctor's note after 3 consecutive days [2].
```

#### Method 2: Structured Output with Citations

Use Pydantic to require citations:

```python
from pydantic import BaseModel
from typing import Literal


class Citation(BaseModel):
    chunk_index: int  # Which chunk supports this claim
    quote: str        # Quoted text from the chunk


class AnswerWithCitations(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: Literal["high", "medium", "low"]


# Use with instructor
import instructor
from openai import OpenAI

client = instructor.from_openai(OpenAI())

result = client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=AnswerWithCitations,
    messages=[
        {
            "role": "user",
            "content": f"Question: {question}\n\nContext:\n{context}\n\nAnswer with citations."
        }
    ]
)

print(result.answer)
for c in result.citations:
    print(f"  [{c.chunk_index}] \"{c.quote}\"")
```

#### Method 3: Highlight in UI

Track which character ranges in the answer correspond to which sources. This lets your UI highlight cited spans.

(Implementation depends on your frontend.)

### Anti-Hallucination Prompt

Add this to your prompt:

```
RULES:
1. Only use information from the provided context.
2. If the context doesn't contain the answer, say "I don't have information about that."
3. Never make up facts. Never use general knowledge outside the context.
4. Cite sources for every claim using [1], [2], etc.
```

This dramatically reduces hallucinations.

---

## 7. Putting It All Together

### The Full Pipeline

```python
async def rag_query(user_question: str) -> dict:
    # 1. Rewrite the query
    rewritten = await rewrite_query(user_question)

    # 2. Embed the rewritten query
    query_embedding = await embed_text(rewritten)

    # 3. Run vector search + BM25 in parallel
    vector_results, bm25_results = await asyncio.gather(
        vector_search(query_embedding, top_k=30),
        bm25_search(rewritten, top_k=30)
    )

    # 4. Combine with RRF
    fused = reciprocal_rank_fusion([
        [r["id"] for r in vector_results],
        [r["id"] for r in bm25_results]
    ])
    top_ids = [doc_id for doc_id, _ in fused[:20]]
    top_chunks = await fetch_chunks(top_ids)

    # 5. Rerank
    reranked = rerank(rewritten, [c["content"] for c in top_chunks], top_k=5)
    final_chunks = [top_chunks[r["index"]] for r in reranked]

    # 6. Build prompt with citations
    context = build_context_with_refs(final_chunks)

    # 7. Generate answer with structured output
    answer = await generate_answer_with_citations(user_question, context)

    return {
        "question": user_question,
        "answer": answer.answer,
        "citations": answer.citations,
        "sources": final_chunks,
        "confidence": answer.confidence
    }
```

### Latency Budget (Typical)

| Step | Latency |
|------|---------|
| Query rewriting | 200-500ms |
| Embedding | 50-100ms |
| Vector + BM25 search | 50-200ms |
| Reranking | 100-300ms |
| LLM generation | 1-3s |
| **Total** | **~2-4s** |

Most of the time is the final LLM call. Streaming makes this feel instant.

### Optimization Tips

1. **Cache embeddings** of common queries
2. **Use `gpt-4o-mini` for rewriting**, `gpt-4o` only for the final answer
3. **Skip rewriting** for short/clear queries (cost optimization)
4. **Stream the final answer** (perceived latency drops to ~500ms)
5. **Use semantic caching** to skip the whole pipeline for repeated queries

---

## 8. Common RAG Mistakes

### Mistake 1: Bad Chunking

Chunks too big = noisy retrieval. Chunks too small = missing context.

**Fix:** Recursive character splitting with 300-500 tokens, 50 token overlap.

### Mistake 2: No Hybrid Search

Pure vector search misses exact matches (codes, names, IDs).

**Fix:** Always combine vector + BM25 with RRF.

### Mistake 3: Skipping Reranking

Top-1 vector result is often not the best. Top-5 reranked is much better.

**Fix:** Always rerank top 20-50 down to 3-5.

### Mistake 4: Stuffing Everything Into Context

LLMs perform worse with too much context. Quality > quantity.

**Fix:** 3-5 highly relevant chunks beats 20 mediocre ones.

### Mistake 5: No Evaluation

You can't improve what you don't measure. Most teams build RAG and hope.

**Fix:** Build a golden test set (we'll cover RAG eval in Month 4 with RAGAS).

### Mistake 6: Not Cleaning Documents

Embedding HTML tags, headers, footers, page numbers = noisy embeddings.

**Fix:** Pre-process aggressively. Strip boilerplate.

### Mistake 7: Mixing Document Types Without Metadata

Throwing PDFs, code, and emails in the same collection without metadata = bad retrieval.

**Fix:** Add `doc_type`, `source`, `date` metadata. Filter when relevant.

### Mistake 8: Ignoring Stale Data

Documents change. Old vectors point to outdated info.

**Fix:** Track `updated_at`. Re-index changed docs. Show users the date.

---

## Final Project

### Project: Production RAG System

Build a complete RAG system with all the techniques from this week. Index real documents (your notes, OpenSolar docs, anything you have) and query them.

### Requirements

1. Recursive character chunking (300-500 tokens, 50 overlap)
2. Hybrid search (vector + BM25) with RRF
3. Reranking (Cohere or BGE)
4. Query rewriting
5. Structured citations with Pydantic
6. Async pipeline
7. CLI interface
8. Basic latency tracking

### Project Structure

```
week-07-rag-pipeline/
├── README.md
├── requirements.txt
├── .env
├── docker-compose.yml
├── ingest.py               # Index documents
├── rag.py                  # Main RAG pipeline
├── chunking.py             # Chunking strategies
├── search.py               # Hybrid search + reranking
├── prompts/
│   └── answer_v1.md
├── data/
│   └── docs/
└── notes.md
```

### Starter Code

**`chunking.py`**

```python
from dataclasses import dataclass


@dataclass
class Chunk:
    content: str
    metadata: dict


def recursive_chunk(
    text: str,
    chunk_size: int = 1500,  # ~400 tokens
    overlap: int = 200,      # ~50 tokens
    separators: list[str] = None
) -> list[str]:
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    if len(text) <= chunk_size:
        return [text]

    sep = separators[0]
    parts = text.split(sep) if sep else list(text)
    chunks = []
    current = ""

    for part in parts:
        candidate = current + sep + part if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            if len(part) > chunk_size and len(separators) > 1:
                chunks.extend(recursive_chunk(part, chunk_size, overlap, separators[1:]))
                current = ""
            else:
                current = part

    if current:
        chunks.append(current)

    # Add overlap
    if overlap > 0 and len(chunks) > 1:
        result = [chunks[0]]
        for i in range(1, len(chunks)):
            tail = chunks[i-1][-overlap:]
            result.append(tail + " " + chunks[i])
        chunks = result

    return chunks


def chunk_document(text: str, source: str) -> list[Chunk]:
    raw_chunks = recursive_chunk(text)
    return [
        Chunk(
            content=chunk,
            metadata={"source": source, "chunk_index": i}
        )
        for i, chunk in enumerate(raw_chunks)
    ]
```

**`search.py`**

```python
import asyncio
import os
from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi
import numpy as np


class HybridSearcher:
    def __init__(self, collection: str = "rag_docs"):
        self.qdrant = QdrantClient("http://localhost:6333")
        self.openai = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.collection = collection
        self.bm25: BM25Okapi | None = None
        self.bm25_docs: list[str] = []
        self.bm25_ids: list[str] = []

    def build_bm25_index(self, ids: list[str], docs: list[str]) -> None:
        self.bm25_ids = ids
        self.bm25_docs = docs
        tokenized = [d.lower().split() for d in docs]
        self.bm25 = BM25Okapi(tokenized)

    async def vector_search(self, query: str, top_k: int = 30) -> list[dict]:
        emb_response = await self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=query
        )
        query_vector = emb_response.data[0].embedding

        results = self.qdrant.query_points(
            collection_name=self.collection,
            query=query_vector,
            limit=top_k
        ).points

        return [
            {"id": str(r.id), "content": r.payload["content"], "score": r.score, "metadata": r.payload}
            for r in results
        ]

    def bm25_search(self, query: str, top_k: int = 30) -> list[dict]:
        if not self.bm25:
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [
            {
                "id": self.bm25_ids[i],
                "content": self.bm25_docs[i],
                "score": float(scores[i])
            }
            for i in top_indices
            if scores[i] > 0
        ]

    @staticmethod
    def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
        scores = {}
        for ranking in rankings:
            for rank, doc_id in enumerate(ranking, start=1):
                scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank)
        return sorted(scores.items(), key=lambda x: -x[1])

    async def hybrid_search(self, query: str, top_k: int = 20) -> list[dict]:
        # Run both in parallel
        vector_task = self.vector_search(query, top_k=30)
        bm25_results = self.bm25_search(query, top_k=30)
        vector_results = await vector_task

        # Build ID -> content map
        all_results = {}
        for r in vector_results:
            all_results[r["id"]] = r
        for r in bm25_results:
            if r["id"] not in all_results:
                all_results[r["id"]] = r

        # RRF
        vector_ids = [r["id"] for r in vector_results]
        bm25_ids = [r["id"] for r in bm25_results]
        fused = self.reciprocal_rank_fusion([vector_ids, bm25_ids])

        return [
            {**all_results[doc_id], "fused_score": score}
            for doc_id, score in fused[:top_k]
            if doc_id in all_results
        ]


async def rerank_with_cohere(query: str, docs: list[dict], top_k: int = 5) -> list[dict]:
    """Use Cohere rerank API."""
    import cohere

    co = cohere.AsyncClient(api_key=os.getenv("COHERE_API_KEY"))

    response = await co.rerank(
        model="rerank-english-v3.0",
        query=query,
        documents=[d["content"] for d in docs],
        top_n=top_k
    )

    return [
        {**docs[r.index], "rerank_score": r.relevance_score}
        for r in response.results
    ]
```

**`rag.py`**

```python
import asyncio
import time
import os
from openai import AsyncOpenAI
from pydantic import BaseModel
from typing import Literal
from dotenv import load_dotenv
import instructor

from search import HybridSearcher, rerank_with_cohere

load_dotenv()


class Citation(BaseModel):
    chunk_index: int
    quote: str


class RAGAnswer(BaseModel):
    answer: str
    citations: list[Citation]
    confidence: Literal["high", "medium", "low"]


structured_client = instructor.from_openai(AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")))
plain_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))


async def rewrite_query(query: str) -> str:
    response = await plain_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": (
                "Rewrite this query into a clear, searchable question. "
                "Preserve all keywords. Output ONLY the rewritten query.\n\n"
                f"Query: {query}\n\nRewritten:"
            )
        }],
        temperature=0,
        max_tokens=100
    )
    return response.choices[0].message.content.strip()


def build_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.get("metadata", {}).get("source", "unknown")
        parts.append(f"[{i}] (Source: {source})\n{chunk['content']}")
    return "\n\n---\n\n".join(parts)


async def generate_answer(query: str, context: str) -> RAGAnswer:
    return await structured_client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=RAGAnswer,
        messages=[{
            "role": "user",
            "content": (
                "Answer the question using ONLY the context below. "
                "Cite specific chunks. If the context doesn't have the answer, say so.\n\n"
                f"CONTEXT:\n{context}\n\n"
                f"QUESTION: {query}"
            )
        }],
        temperature=0
    )


async def run_rag(query: str, searcher: HybridSearcher, verbose: bool = True) -> dict:
    timings = {}

    # 1. Rewrite
    t0 = time.time()
    rewritten = await rewrite_query(query)
    timings["rewrite"] = time.time() - t0
    if verbose:
        print(f"📝 Rewritten: {rewritten}")

    # 2. Hybrid search
    t0 = time.time()
    candidates = await searcher.hybrid_search(rewritten, top_k=20)
    timings["search"] = time.time() - t0
    if verbose:
        print(f"🔎 Found {len(candidates)} candidates")

    # 3. Rerank
    t0 = time.time()
    if os.getenv("COHERE_API_KEY"):
        top_chunks = await rerank_with_cohere(rewritten, candidates, top_k=5)
    else:
        top_chunks = candidates[:5]  # Skip reranking if no API key
    timings["rerank"] = time.time() - t0
    if verbose:
        print(f"🎯 Top {len(top_chunks)} after reranking")

    # 4. Build context and generate
    context = build_context(top_chunks)
    t0 = time.time()
    answer = await generate_answer(query, context)
    timings["generate"] = time.time() - t0

    return {
        "question": query,
        "rewritten": rewritten,
        "answer": answer.answer,
        "citations": [c.model_dump() for c in answer.citations],
        "confidence": answer.confidence,
        "sources": [{"source": c["metadata"].get("source"), "preview": c["content"][:200]} for c in top_chunks],
        "timings": timings
    }


async def main():
    searcher = HybridSearcher()

    # Load BM25 index from Qdrant
    print("📚 Loading BM25 index...")
    all_points = searcher.qdrant.scroll(
        collection_name="rag_docs",
        limit=10000
    )[0]
    ids = [str(p.id) for p in all_points]
    docs = [p.payload["content"] for p in all_points]
    searcher.build_bm25_index(ids, docs)
    print(f"✓ Loaded {len(docs)} documents\n")

    print("🤖 RAG ready. Type your questions ('quit' to exit)\n")

    while True:
        query = input("You: ").strip()
        if query.lower() == "quit":
            break
        if not query:
            continue

        try:
            result = await run_rag(query, searcher)

            print(f"\n💬 Answer (confidence: {result['confidence']}):")
            print(result["answer"])

            if result["citations"]:
                print("\n📎 Citations:")
                for c in result["citations"]:
                    print(f"  [{c['chunk_index']}] \"{c['quote'][:100]}...\"")

            print(f"\n⏱️  Timing: {sum(result['timings'].values()):.2f}s total")
            for step, ms in result["timings"].items():
                print(f"   - {step}: {ms*1000:.0f}ms")

            print("\n" + "─" * 60 + "\n")
        except Exception as e:
            print(f"❌ Error: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
```

**`ingest.py`** (similar to Week 6's, but with chunking)

```python
import asyncio
import sys
from pathlib import Path
from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from dotenv import load_dotenv
import os
import uuid

from chunking import chunk_document

load_dotenv()
client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
qdrant = QdrantClient("http://localhost:6333")


async def embed_batch(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = await client.embeddings.create(
            model="text-embedding-3-small",
            input=batch
        )
        all_embeddings.extend([d.embedding for d in response.data])
        print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)}")
    return all_embeddings


async def main(folder: str):
    # Setup collection
    try:
        qdrant.delete_collection("rag_docs")
    except Exception:
        pass

    qdrant.create_collection(
        collection_name="rag_docs",
        vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
    )

    # Load and chunk all documents
    print(f"📂 Loading documents from {folder}...")
    all_chunks = []
    for path in Path(folder).rglob("*"):
        if path.suffix in {".txt", ".md"} and path.is_file():
            content = path.read_text(errors="replace")
            if 50 < len(content) < 500_000:
                chunks = chunk_document(content, source=str(path))
                all_chunks.extend(chunks)

    print(f"✓ Created {len(all_chunks)} chunks from documents")

    # Embed
    print("\n🧮 Generating embeddings...")
    contents = [c.content for c in all_chunks]
    embeddings = await embed_batch(contents)

    # Insert
    print("\n💾 Inserting into Qdrant...")
    points = [
        PointStruct(
            id=str(uuid.uuid4()),
            vector=emb,
            payload={"content": chunk.content, **chunk.metadata}
        )
        for chunk, emb in zip(all_chunks, embeddings)
    ]
    qdrant.upsert(collection_name="rag_docs", points=points)

    print(f"\n✅ Indexed {len(points)} chunks")


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "data/docs"
    asyncio.run(main(folder))
```

**`requirements.txt`**

```
openai>=1.0
qdrant-client>=1.7
rank-bm25>=0.2.2
cohere>=5.0
instructor>=1.0
pydantic>=2.0
python-dotenv>=1.0
numpy>=1.24
```

**`.env`**

```
OPENAI_API_KEY=sk-...
COHERE_API_KEY=...   # Optional but highly recommended
```

### What You Should Learn from This Project

- **Recursive chunking** — production-grade splitting
- **Hybrid search** — combining sparse (BM25) and dense (vector) signals
- **RRF (Reciprocal Rank Fusion)** — the standard way to combine rankings
- **Reranking pipeline** — two-stage retrieval for accuracy
- **Query rewriting** — preprocessing user input
- **Structured citations** — non-negotiable for production
- **Latency tracking** — knowing your bottlenecks

### Stretch Goals

- Add HyDE for vague queries
- Try semantic chunking and compare retrieval quality
- Implement query decomposition for complex questions
- Add streaming for the final answer
- Try different chunk sizes (200, 500, 1000) and measure impact

---

## Self-Check Questions

1. What's the difference between RAG and fine-tuning?
2. Why is chunking important?
3. What's the typical chunk size in production?
4. Why combine vector search with BM25?
5. What does Reciprocal Rank Fusion do?
6. Why rerank if you already have hybrid search?
7. What problem does HyDE solve?
8. Why are citations non-negotiable in production RAG?
9. What's the typical RAG pipeline latency budget?
10. What are 3 common RAG mistakes?

### Answers

1. RAG retrieves relevant docs and feeds them to the LLM. Fine-tuning bakes knowledge into model weights. RAG is cheaper, easier to update, and more transparent.
2. Whole documents make poor embeddings (averaged meaning) and don't fit in context. Chunks give precise, retrievable units.
3. 200-500 tokens (~150-400 words), with 50-100 token overlap.
4. Vector handles semantics; BM25 handles exact matches (codes, names, IDs). Together they cover both cases.
5. Combines multiple ranked lists by giving each document a score based on its rank in each list. Documents ranked highly in multiple lists score best.
6. Even hybrid search returns noisy top results. A reranker (slower, more accurate) re-scores the top 20-50 to find the actual best 5.
7. Vague queries don't look like document text. HyDE generates a hypothetical answer (which looks more like a document) and embeds that for better retrieval.
8. Without citations, users can't verify answers, hallucinations go undetected, and you can't audit responses for compliance.
9. ~2-4 seconds total. Most is the final LLM call. Streaming the answer makes it feel instant.
10. (Pick any 3) Bad chunking, no hybrid search, skipping rerank, no eval, dirty input docs, ignoring metadata, stale data.

---

## Resources

### Must-Read
- [Anthropic's "Contextual Retrieval"](https://www.anthropic.com/news/contextual-retrieval) — Their cutting-edge RAG technique
- [Cohere Rerank Docs](https://docs.cohere.com/docs/rerank)
- [LangChain Text Splitters](https://python.langchain.com/docs/concepts/text_splitters/)

### Recommended Papers
- "Lost in the Middle: How Language Models Use Long Contexts" (Liu et al., 2023)
- "Precise Zero-Shot Dense Retrieval without Relevance Labels" (HyDE paper, 2022)
- "RA-DIT: Retrieval-Augmented Dual Instruction Tuning" (2023)

### Tools
- [LlamaIndex](https://docs.llamaindex.ai/) — RAG-focused framework (worth knowing)
- [Ragatouille](https://github.com/AnswerDotAI/RAGatouille) — Easy ColBERT-based retrieval
- [Vespa](https://vespa.ai/) — Production search engine with vectors
- [Anthropic's Contextual Embeddings cookbook](https://github.com/anthropics/anthropic-cookbook/tree/main/skills/contextual-embeddings)

---

## Next Up

**Week 8:** Advanced RAG — agentic RAG, multi-hop retrieval, contextual retrieval (Anthropic's technique), metadata filtering at scale.

You've now built a real RAG system. Next week we'll make it smarter — letting the LLM decide when and how to search, handling complex multi-step questions, and using state-of-the-art techniques.
