# Week 6: Embeddings and Vector Search

> **Goal:** Understand how AI systems "remember" and "find" information by meaning, not just keywords. Embeddings and vector search are the foundation of RAG, semantic search, and most production AI apps.

---

## Table of Contents

1. [The Big Picture](#the-big-picture)
2. [What Are Embeddings?](#1-what-are-embeddings)
3. [How to Generate Embeddings](#2-how-to-generate-embeddings)
4. [Cosine Similarity Explained Simply](#3-cosine-similarity-explained-simply)
5. [Embedding Models Compared](#4-embedding-models-compared)
6. [What Is a Vector Database?](#5-what-is-a-vector-database)
7. [pgvector (Start Here)](#6-pgvector-start-here)
8. [Qdrant (Specialized Vector DB)](#7-qdrant-specialized-vector-db)
9. [Indexing Strategies (HNSW, IVF, Flat)](#8-indexing-strategies-hnsw-ivf-flat)
10. [Distance Metrics](#9-distance-metrics)
11. [Common Pitfalls](#10-common-pitfalls)
12. [Final Project](#final-project)
13. [Self-Check Questions](#self-check-questions)

---

## The Big Picture

### Why This Week Matters

Every "ChatGPT for your documents" app — every RAG system, every semantic search engine, every recommendation system — runs on **embeddings + vector search**.

The flow:
```
Document → Embedding → Stored in Vector DB
                            ↓
User question → Embedding → Search vector DB → Find similar docs → Send to LLM
```

By the end of this week, you'll be able to build a **search engine that understands meaning**, not just keywords.

### Concrete Example

Old keyword search:
```
Query: "fix car"
Result: Only documents containing the words "fix" AND "car"
Misses: "auto repair", "vehicle maintenance", "broken transmission"
```

Semantic search (with embeddings):
```
Query: "fix car"
Result: All documents about auto repair, vehicle maintenance,
        broken transmissions — even if they don't use the word "fix" or "car"
```

That's the magic. Now let's understand how it works.

---

## 1. What Are Embeddings?

### The Simple Explanation

An **embedding** is a list of numbers that represents the meaning of text.

Words/sentences with similar meaning have similar number lists.

```python
"king"  → [0.21, -0.45, 0.83, ..., 0.12]   # 1536 numbers
"queen" → [0.19, -0.43, 0.85, ..., 0.10]   # Very close to "king"
"pizza" → [0.92, 0.31, -0.55, ..., 0.78]   # Very different
```

### Why It Works (Intuition)

Imagine plotting words on a 2D map:

```
        cat ●
            ● kitten
            ● dog            ● bicycle
                                ● motorcycle
                                    ● car

   ● apple
       ● banana       ● mango
```

Similar concepts cluster together. Embeddings do this — but in **1536 dimensions** (or whatever the model uses), not 2.

### Key Properties

| Property | Description |
|----------|-------------|
| **Length (dimensions)** | Fixed for a given model. OpenAI's `text-embedding-3-small` = 1536 numbers. |
| **Distance = meaning** | Similar texts → similar vectors → small distance. |
| **Model-specific** | An embedding from OpenAI is NOT comparable to one from Cohere. Always use the same model. |
| **Cost** | Cheap. ~$0.02 per million tokens. |

### Real Numbers Example

```python
from openai import OpenAI

client = OpenAI()

def embed(text: str) -> list[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

vec1 = embed("dog")
vec2 = embed("puppy")
vec3 = embed("car")

print(len(vec1))  # 1536
print(vec1[:5])   # [0.012, -0.034, 0.089, ...] (first 5 of 1536 numbers)
```

`vec1` and `vec2` will be close. `vec3` will be far away.

---

## 2. How to Generate Embeddings

### OpenAI (Most Common)

```bash
pip install openai
```

```python
from openai import OpenAI

client = OpenAI()

# Single text
response = client.embeddings.create(
    model="text-embedding-3-small",
    input="The quick brown fox"
)
embedding = response.data[0].embedding

# Multiple texts (batch - much faster!)
response = client.embeddings.create(
    model="text-embedding-3-small",
    input=[
        "The quick brown fox",
        "Lazy dogs sleep all day",
        "Pizza is delicious"
    ]
)
embeddings = [d.embedding for d in response.data]
```

### Async Version

```python
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def embed(texts: list[str]) -> list[list[float]]:
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [d.embedding for d in response.data]
```

### Open-Source Alternative (Free, Runs Locally)

```bash
pip install sentence-transformers
```

```python
from sentence_transformers import SentenceTransformer

# Downloads model first time (~90 MB)
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

embeddings = model.encode([
    "The quick brown fox",
    "Lazy dogs sleep all day"
])

print(embeddings.shape)  # (2, 384) — 384 dimensions
```

Pros: Free, fast, no API limits, runs offline.
Cons: Slightly lower quality than OpenAI for many tasks.

### Cohere (Best Multilingual)

```bash
pip install cohere
```

```python
import cohere

co = cohere.Client(api_key="...")
response = co.embed(
    texts=["The quick brown fox"],
    model="embed-english-v3.0",
    input_type="search_document"  # or "search_query"
)
```

### Important: Document vs Query Embeddings

Some models (like Cohere) want you to specify whether you're embedding a **document** (to be searched) or a **query** (to search with). They optimize differently.

OpenAI doesn't make this distinction — same embedding for both.

### Cost and Speed Cheat Sheet

| Model | Cost (per 1M tokens) | Dimensions | Speed |
|-------|---------------------|------------|-------|
| OpenAI text-embedding-3-small | $0.02 | 1536 | Fast |
| OpenAI text-embedding-3-large | $0.13 | 3072 | Fast |
| Cohere embed-english-v3.0 | $0.10 | 1024 | Fast |
| BGE-small (open source) | Free | 384 | Very fast (local) |
| BGE-large (open source) | Free | 1024 | Fast (local, GPU helps) |

> Always check current pricing — providers update frequently.

---

## 3. Cosine Similarity Explained Simply

### The Question

Given two embeddings, **how similar are they?**

### The Answer: Cosine Similarity

It measures the angle between two vectors. Range: -1 to +1.

```
Cosine similarity = 1.0    → Identical meaning
Cosine similarity = 0.8    → Very similar
Cosine similarity = 0.5    → Somewhat related
Cosine similarity = 0.0    → Unrelated
Cosine similarity = -1.0   → Opposite (rare in practice)
```

### Visual Intuition

```
        ↑ vector A
       /
      /
     /
    ●———————→ vector B   (Small angle = high similarity)


        ↑ vector A
        |
        |
        |
        ●—————→ vector B   (90° angle = no similarity)
```

### The Formula (Don't Memorize)

```
cosine_similarity(a, b) = (a · b) / (|a| × |b|)
```

In Python:

```python
import numpy as np

def cosine_similarity(a: list[float], b: list[float]) -> float:
    a = np.array(a)
    b = np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# Example
vec1 = embed("dog")
vec2 = embed("puppy")
vec3 = embed("car")

print(cosine_similarity(vec1, vec2))  # ~0.85 (similar)
print(cosine_similarity(vec1, vec3))  # ~0.20 (different)
```

### Practical Threshold

For most use cases:

| Similarity | Interpretation |
|-----------|----------------|
| > 0.85 | Near-duplicate |
| 0.70 – 0.85 | Strongly related |
| 0.50 – 0.70 | Loosely related |
| < 0.50 | Probably unrelated |

These thresholds depend on the model. Always **calibrate on your own data**.

### Tip: Normalize Vectors for Speed

If your vectors are normalized (length = 1), cosine similarity becomes a simple dot product (faster):

```python
def normalize(v):
    return v / np.linalg.norm(v)

# OpenAI embeddings are already normalized!
# For others, normalize first
```

---

## 4. Embedding Models Compared

### My Recommendations

For learning:
- **Start with OpenAI `text-embedding-3-small`** — cheap, reliable, easy.

For production:
- **OpenAI `text-embedding-3-large`** — best quality.
- **BGE-large** — free, near-OpenAI quality, runs locally.
- **Cohere** — if you need multilingual.

### The MTEB Leaderboard

The official benchmark for embedding quality is **MTEB (Massive Text Embedding Benchmark)**:

🔗 https://huggingface.co/spaces/mteb/leaderboard

Check it before choosing a model in production.

### Model Selection Decision Tree

```
Is cost a concern?
├── Yes → Open-source (BGE)
└── No → OpenAI text-embedding-3-large

Need to run offline?
├── Yes → Open-source
└── No → OpenAI or Cohere

Multilingual content?
├── Yes → Cohere multilingual or BGE-multilingual
└── No → OpenAI is fine

Massive scale (>100M vectors)?
├── Yes → Smaller dimensions (384) to save storage
└── No → Larger dimensions (1536) for quality
```

### Important: Don't Mix Models

```python
# ❌ WRONG
doc_embeddings = openai_embed(docs)
query_embedding = bge_embed(query)
similarity = cosine(doc_embeddings, query_embedding)
# This produces meaningless results!

# ✅ RIGHT
doc_embeddings = openai_embed(docs)
query_embedding = openai_embed(query)
similarity = cosine(doc_embeddings, query_embedding)
```

If you change your embedding model, you must **re-embed everything**.

---

## 5. What Is a Vector Database?

### The Problem

Imagine you have 1 million documents, each with an embedding.
A user query comes in. You need to find the 5 most similar documents.

**Naive approach:** Compare query to all 1M docs. = Slow (seconds-minutes).

**Vector DB approach:** Smart indexing finds top-5 in milliseconds.

### What Vector DBs Do

1. **Store embeddings** efficiently
2. **Index them** for fast similarity search
3. **Filter by metadata** (e.g., "find similar docs from 2024 only")
4. **Scale** to billions of vectors

### Popular Vector DBs

| DB | Best For | Hosting |
|------|----------|---------|
| **pgvector** | If you already use Postgres | Self-hosted (or any Postgres) |
| **Qdrant** | Pure vector DB, fast, open-source | Self-hosted or cloud |
| **Pinecone** | Easy managed service | Cloud only |
| **Weaviate** | Built-in modules (embedding, etc.) | Self-hosted or cloud |
| **Chroma** | Lightweight, good for prototyping | Self-hosted, embedded |
| **LanceDB** | Embedded, file-based | Local files |

### My Recommendation

- **Learning/prototyping:** Chroma (zero setup) or pgvector
- **Production:** pgvector (if Postgres-based) or Qdrant
- **Massive scale:** Qdrant or Pinecone

We'll cover **pgvector** and **Qdrant** in detail since those are most common in the wild.

---

## 6. pgvector (Start Here)

### Why pgvector?

You already know SQL. pgvector adds vector search to PostgreSQL — the database you (probably) already use.

### Setup

```bash
# Install Postgres if you haven't (Mac):
brew install postgresql@16
brew services start postgresql@16

# Or use Docker (recommended):
docker run -d --name pgvector \
  -e POSTGRES_PASSWORD=mysecret \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

```bash
pip install psycopg pgvector
```

### Enable the Extension

```sql
CREATE EXTENSION vector;
```

### Create a Table With Vectors

```sql
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding VECTOR(1536),  -- 1536 dims for OpenAI
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Add an index for fast search
CREATE INDEX ON documents USING hnsw (embedding vector_cosine_ops);
```

### Insert Embeddings (Python)

```python
import psycopg
from openai import OpenAI

client = OpenAI()
conn = psycopg.connect("postgresql://postgres:mysecret@localhost/postgres")

def embed(text: str) -> list[float]:
    return client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    ).data[0].embedding


def insert_document(content: str, metadata: dict | None = None):
    embedding = embed(content)
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO documents (content, embedding, metadata) VALUES (%s, %s, %s)",
            (content, embedding, metadata or {})
        )
    conn.commit()


# Add some docs
insert_document("Python is a programming language", {"category": "tech"})
insert_document("Lions are large carnivores", {"category": "animals"})
insert_document("Pizza is from Italy", {"category": "food"})
```

### Search by Similarity

```python
def search(query: str, limit: int = 5):
    query_embedding = embed(query)

    with conn.cursor() as cur:
        cur.execute("""
            SELECT
                id,
                content,
                metadata,
                1 - (embedding <=> %s::vector) AS similarity
            FROM documents
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        """, (query_embedding, query_embedding, limit))

        return cur.fetchall()


# Try it
results = search("Tell me about coding")
for row in results:
    print(f"Score: {row[3]:.3f} | {row[1]}")

# Output:
# Score: 0.781 | Python is a programming language
# Score: 0.234 | Lions are large carnivores
# Score: 0.198 | Pizza is from Italy
```

### Operators You'll See

```sql
embedding <=> query   -- Cosine distance (lower = more similar)
embedding <-> query   -- Euclidean distance
embedding <#> query   -- Negative inner product
```

For cosine similarity, use `<=>` and remember:
- Result is **distance** (0 = identical, 2 = opposite)
- Convert to similarity: `1 - distance`

### Filtering with Metadata

```sql
SELECT content, 1 - (embedding <=> %s::vector) AS similarity
FROM documents
WHERE metadata->>'category' = 'tech'  -- Filter first
ORDER BY embedding <=> %s::vector
LIMIT 5;
```

This is huge — you can combine SQL filters with vector search.

---

## 7. Qdrant (Specialized Vector DB)

### Why Qdrant?

Built specifically for vector search. Often faster than pgvector at scale. Great Python SDK.

### Setup with Docker

```bash
docker run -d --name qdrant \
  -p 6333:6333 \
  -p 6334:6334 \
  qdrant/qdrant
```

```bash
pip install qdrant-client
```

### Create a Collection

```python
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

client = QdrantClient("localhost", port=6333)

client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(
        size=1536,
        distance=Distance.COSINE  # or DOT, EUCLID
    )
)
```

### Insert Documents

```python
from qdrant_client.models import PointStruct
import uuid

def insert_documents(docs: list[dict]):
    """docs = [{'content': '...', 'metadata': {...}}, ...]"""
    points = []
    for doc in docs:
        embedding = embed(doc["content"])
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "content": doc["content"],
                **doc.get("metadata", {})
            }
        ))

    client.upsert(collection_name="documents", points=points)


insert_documents([
    {"content": "Python is a programming language", "metadata": {"category": "tech"}},
    {"content": "Lions are large carnivores", "metadata": {"category": "animals"}},
])
```

### Search

```python
def search(query: str, limit: int = 5, category_filter: str | None = None):
    query_vector = embed(query)

    filter_condition = None
    if category_filter:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        filter_condition = Filter(
            must=[FieldCondition(
                key="category",
                match=MatchValue(value=category_filter)
            )]
        )

    results = client.query_points(
        collection_name="documents",
        query=query_vector,
        limit=limit,
        query_filter=filter_condition
    ).points

    return [(r.score, r.payload["content"]) for r in results]


# Search all
print(search("programming"))

# Search with filter
print(search("programming", category_filter="tech"))
```

### Why Qdrant Over pgvector?

| Use case | Recommendation |
|----------|---------------|
| You already use Postgres | pgvector |
| Need >10M vectors | Qdrant (faster at scale) |
| Want a managed service | Qdrant Cloud or Pinecone |
| Pure vector search app | Qdrant |
| Mixed structured + vector queries | pgvector |

---

## 8. Indexing Strategies (HNSW, IVF, Flat)

### The Problem

With 10 million vectors, comparing the query to every single one is too slow. Indexes solve this by **approximating** the search — trading a tiny bit of accuracy for huge speed gains.

### Three Main Indexes

#### Flat (Brute Force)

- Compare query to every vector
- 100% accurate
- Slow for large datasets
- **Use when:** <10K vectors, or accuracy is critical

#### IVF (Inverted File Index)

- Cluster vectors into groups
- Search query only against nearest groups
- Fast, slightly less accurate
- **Use when:** Medium scale (10K–1M), need balanced speed/accuracy

#### HNSW (Hierarchical Navigable Small World)

- Builds a graph of vectors
- Navigates graph to find nearest neighbors
- Very fast, very accurate
- **Use when:** Large scale, low latency required (most production cases)

### Quick Comparison

| Index | Build Time | Search Speed | Memory | Accuracy |
|-------|-----------|--------------|--------|----------|
| Flat | None | Slow | Low | 100% |
| IVF | Medium | Fast | Low | ~95% |
| HNSW | Slow | Very fast | High | ~99% |

### What to Use

- **Default: HNSW** — best speed/accuracy tradeoff for most cases
- Both pgvector and Qdrant support HNSW

### HNSW Tuning Parameters

You don't need to memorize these, but you'll see them:

| Parameter | What it does | Default |
|-----------|--------------|---------|
| `m` | Connections per node (higher = better recall, more memory) | 16 |
| `ef_construction` | Build quality (higher = slower build, better quality) | 64–200 |
| `ef_search` | Search quality (higher = slower, more accurate) | 40–200 |

For learning, defaults are fine.

---

## 9. Distance Metrics

Three common ways to measure "distance" between vectors:

### Cosine Similarity

- Measures **angle** between vectors
- Range: -1 to +1 (or distance: 0 to 2)
- **Use for:** Most text embeddings (default)

### Euclidean Distance

- Measures **straight-line distance** between points
- Range: 0 to ∞
- **Use for:** Image embeddings, when magnitude matters

### Dot Product (Inner Product)

- Like cosine but doesn't normalize
- **Use for:** When vectors are already normalized (faster than cosine)

### Quick Decision

```
Are your vectors normalized?
├── Yes → Use Dot Product (fastest)
└── No  → Use Cosine Similarity (safest)

Working with images or where magnitude matters?
└── Use Euclidean Distance
```

### OpenAI's Embeddings

OpenAI returns **already-normalized** vectors, so:
- Cosine and dot product give the same result
- Use dot product for slight speed boost

---

## 10. Common Pitfalls

### Pitfall 1: Embedding the Wrong Granularity

**Bad:** Embed an entire 100-page PDF as one vector.
The vector becomes a vague average — poor retrieval.

**Good:** Split into 200–500 word chunks, embed each.
We'll cover chunking strategies in detail in Week 7.

### Pitfall 2: Mixing Embedding Models

If you re-embed with a different model, **all old embeddings become useless**. Pick a model and commit, or budget for re-embedding.

### Pitfall 3: Ignoring Metadata Filters

Vector search alone often returns irrelevant results. **Combine with metadata filters** (category, date, user_id) for better results.

### Pitfall 4: Not Normalizing for Cosine

If using a metric that needs normalization, ensure vectors are normalized — or you'll get nonsense results.

### Pitfall 5: Using Tiny Embeddings for Long Text

384-dim embeddings are great for short text, but lose nuance on long passages. Match dimensions to content complexity.

### Pitfall 6: Trusting Top-1 Results

Top-1 is rarely the best result for nuanced queries. Retrieve top-5 to top-20 and **rerank** (we'll cover this in Week 7).

### Pitfall 7: Embedding Bad Text

If you embed messy text (HTML tags, weird Unicode, encoding errors), you get garbage embeddings. **Clean text first.**

```python
import re

def clean_text(text: str) -> str:
    text = re.sub(r'\s+', ' ', text)              # Normalize whitespace
    text = re.sub(r'<[^>]+>', '', text)           # Strip HTML
    text = text.strip()
    return text
```

---

## Final Project

### Project: Semantic Search Engine

Build a CLI semantic search engine over a corpus of your choice (your notes, articles, code files, anything).

### Requirements

1. Index at least 100 documents
2. Use OpenAI embeddings
3. Use pgvector OR Qdrant (you choose)
4. Support metadata filtering
5. Show similarity scores
6. Implement async batch embedding (for speed)
7. Compare results with at least 5 different queries
8. Track total cost of indexing

### Project Structure

```
week-06-semantic-search/
├── README.md
├── requirements.txt
├── .env
├── .gitignore
├── docker-compose.yml      # If using pgvector/Qdrant via Docker
├── search.py               # Main CLI
├── embeddings.py           # Embedding utilities
├── store.py                # Vector store abstraction
├── ingest.py               # Bulk indexing
├── data/
│   └── docs/               # Your corpus
└── notes.md
```

### Starter Code (using Qdrant)

**`docker-compose.yml`**

```yaml
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - ./qdrant_data:/qdrant/storage
```

Run: `docker-compose up -d`

**`embeddings.py`**

```python
import asyncio
import os
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
COST_PER_1M_TOKENS = 0.02


async def embed_text(text: str) -> list[float]:
    response = await client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding


async def embed_batch(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """Embed many texts efficiently in batches."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = await client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=batch
        )
        all_embeddings.extend([d.embedding for d in response.data])
        print(f"  Embedded {min(i + batch_size, len(texts))}/{len(texts)}")
    return all_embeddings


def estimate_cost(total_tokens: int) -> float:
    return (total_tokens / 1_000_000) * COST_PER_1M_TOKENS
```

**`store.py`**

```python
import uuid
from typing import Any
from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct,
    Filter, FieldCondition, MatchValue
)


class VectorStore:
    def __init__(self, collection_name: str, dim: int = 1536, url: str = "http://localhost:6333"):
        self.client = QdrantClient(url=url)
        self.collection = collection_name
        self.dim = dim

    def setup(self, recreate: bool = False) -> None:
        if recreate:
            try:
                self.client.delete_collection(self.collection)
            except Exception:
                pass

        existing = [c.name for c in self.client.get_collections().collections]
        if self.collection not in existing:
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE)
            )
            print(f"✓ Created collection '{self.collection}'")
        else:
            print(f"✓ Using existing collection '{self.collection}'")

    def insert(self, content: str, embedding: list[float], metadata: dict | None = None) -> str:
        point_id = str(uuid.uuid4())
        self.client.upsert(
            collection_name=self.collection,
            points=[PointStruct(
                id=point_id,
                vector=embedding,
                payload={"content": content, **(metadata or {})}
            )]
        )
        return point_id

    def insert_batch(
        self,
        contents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict] | None = None
    ) -> list[str]:
        ids = [str(uuid.uuid4()) for _ in contents]
        metadatas = metadatas or [{}] * len(contents)

        points = [
            PointStruct(
                id=pid,
                vector=emb,
                payload={"content": c, **m}
            )
            for pid, c, emb, m in zip(ids, contents, embeddings, metadatas)
        ]

        self.client.upsert(collection_name=self.collection, points=points)
        return ids

    def search(
        self,
        query_embedding: list[float],
        limit: int = 5,
        metadata_filter: dict | None = None
    ) -> list[dict]:
        filter_obj = None
        if metadata_filter:
            filter_obj = Filter(must=[
                FieldCondition(key=k, match=MatchValue(value=v))
                for k, v in metadata_filter.items()
            ])

        results = self.client.query_points(
            collection_name=self.collection,
            query=query_embedding,
            limit=limit,
            query_filter=filter_obj
        ).points

        return [
            {
                "id": r.id,
                "score": r.score,
                "content": r.payload.get("content"),
                "metadata": {k: v for k, v in r.payload.items() if k != "content"}
            }
            for r in results
        ]

    def count(self) -> int:
        return self.client.count(self.collection).count
```

**`ingest.py`**

```python
import asyncio
import sys
from pathlib import Path
from dotenv import load_dotenv

from embeddings import embed_batch, estimate_cost
from store import VectorStore

load_dotenv()


def load_documents(folder: str) -> list[dict]:
    """Load .txt and .md files from a folder."""
    docs = []
    for path in Path(folder).rglob("*"):
        if path.suffix in {".txt", ".md"} and path.is_file():
            content = path.read_text(errors="replace")
            # Skip empty or huge files
            if 50 < len(content) < 100_000:
                docs.append({
                    "content": content[:5000],  # Truncate for now
                    "metadata": {
                        "filename": path.name,
                        "path": str(path),
                        "extension": path.suffix
                    }
                })
    return docs


async def main(folder: str):
    print(f"📂 Loading documents from {folder}...")
    docs = load_documents(folder)
    print(f"✓ Loaded {len(docs)} documents")

    if not docs:
        print("No documents found.")
        return

    # Setup store
    store = VectorStore("my_docs")
    store.setup(recreate=True)

    # Embed in batches
    print(f"\n🧮 Generating embeddings...")
    contents = [d["content"] for d in docs]
    metadatas = [d["metadata"] for d in docs]

    embeddings = await embed_batch(contents, batch_size=50)

    # Estimate cost
    total_chars = sum(len(c) for c in contents)
    approx_tokens = total_chars // 4
    cost = estimate_cost(approx_tokens)
    print(f"💵 Approximate cost: ${cost:.4f} ({approx_tokens:,} tokens)")

    # Insert
    print(f"\n💾 Inserting into vector store...")
    store.insert_batch(contents, embeddings, metadatas)
    print(f"✓ Indexed {store.count()} documents")


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else "data/docs"
    asyncio.run(main(folder))
```

**`search.py`**

```python
import asyncio
from dotenv import load_dotenv

from embeddings import embed_text
from store import VectorStore

load_dotenv()


async def search_cli():
    store = VectorStore("my_docs")
    print(f"🔍 Search engine ready ({store.count()} docs indexed)")
    print("Type 'quit' to exit, or 'filter:key=value query' to filter\n")

    while True:
        user_input = input("Query: ").strip()
        if user_input.lower() == "quit":
            break
        if not user_input:
            continue

        # Parse optional filter
        metadata_filter = None
        query = user_input
        if user_input.startswith("filter:"):
            try:
                filter_part, query = user_input[7:].split(" ", 1)
                key, value = filter_part.split("=")
                metadata_filter = {key: value}
            except ValueError:
                print("Invalid filter syntax. Use: filter:key=value query")
                continue

        # Embed the query
        query_embedding = await embed_text(query)

        # Search
        results = store.search(
            query_embedding,
            limit=5,
            metadata_filter=metadata_filter
        )

        # Display
        print(f"\n📊 Top {len(results)} results:")
        for i, r in enumerate(results, 1):
            score = r["score"]
            preview = r["content"][:200].replace("\n", " ")
            filename = r["metadata"].get("filename", "?")

            score_emoji = "🎯" if score > 0.8 else "✅" if score > 0.6 else "⚠️"
            print(f"\n{i}. {score_emoji} Score: {score:.3f} | {filename}")
            print(f"   {preview}...")

        print("\n" + "─" * 60)


if __name__ == "__main__":
    asyncio.run(search_cli())
```

**`requirements.txt`**

```
openai>=1.0
qdrant-client>=1.7
python-dotenv>=1.0
```

**`.env`**

```
OPENAI_API_KEY=sk-...
```

### Test Queries to Try

After indexing, try these to see semantic search in action:

```
> programming language
> how to fix bugs
> machine learning
> filter:extension=.md python
> what's the meaning of life
```

### What You Should Learn from This Project

- **Embedding generation** at scale (batching matters!)
- **Vector store abstraction** — swap Qdrant for pgvector by changing one class
- **Metadata filtering** combined with vector search
- **Cost estimation** before expensive operations
- **The end-to-end flow** of every RAG-style system

### Stretch Goals

- Add a `--metric` flag to switch between cosine and dot product
- Implement a `compare.py` that shows similarity between any two pieces of text
- Add re-indexing with detection of new/changed files
- Try the same project with pgvector and compare performance

---

## Self-Check Questions

1. What is an embedding, in one sentence?
2. Why can't you compare an OpenAI embedding to a Cohere embedding?
3. What does cosine similarity of 0.9 mean? Of 0.1?
4. What's the difference between Flat, IVF, and HNSW indexes?
5. When would you choose pgvector over Qdrant?
6. Why is batching embedding requests faster than one at a time?
7. What does the `<=>` operator do in pgvector?
8. Why is metadata filtering important in vector search?
9. If your vectors are already normalized, which distance metric is fastest?
10. What's the typical embedding dimension for OpenAI's `text-embedding-3-small`?

### Answers

1. A list of numbers that represents the meaning of text, where similar meanings produce similar numbers.
2. They live in different "vector spaces" — different models produce different coordinate systems with no shared meaning.
3. 0.9 = very similar in meaning. 0.1 = essentially unrelated.
4. Flat = brute force, 100% accurate, slow. IVF = clusters, fast, ~95% accurate. HNSW = graph-based, very fast, ~99% accurate.
5. When you already use Postgres or need tightly integrated SQL filters with vector search.
6. One API call processes many texts in parallel — fewer network round trips, lower latency, lower cost.
7. Computes cosine distance between two vectors (lower = more similar).
8. Combines structured filters with semantic search to drastically improve relevance and reduce false matches.
9. Dot product — skips normalization since it's already done.
10. 1536 dimensions.

---

## Resources

### Must-Read
- [OpenAI Embeddings Guide](https://platform.openai.com/docs/guides/embeddings)
- [pgvector README](https://github.com/pgvector/pgvector)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [MTEB Leaderboard](https://huggingface.co/spaces/mteb/leaderboard) (always check this before choosing a model)

### Recommended
- "What Are Embeddings?" by Vicki Boykis (free book) — https://vickiboykis.com/what_are_embeddings/
- Sentence Transformers documentation — https://www.sbert.net/

### Tools to Bookmark
- [Embedding Projector](https://projector.tensorflow.org/) — Visualize embeddings in 3D
- [Chroma](https://www.trychroma.com/) — Easiest vector DB for prototyping
- [LanceDB](https://lancedb.com/) — File-based vector DB (no server needed)

---

## Next Up

**Week 7:** RAG Pipelines — chunking strategies, hybrid search (BM25 + vectors), reranking, query rewriting, citations.

You can now turn any text into searchable vectors. Next week we'll build a real RAG system that retrieves the right context for an LLM to answer questions accurately.
