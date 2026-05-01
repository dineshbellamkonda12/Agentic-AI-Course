# Week 1-2: Python for AI Engineering

> **Goal:** Build a strong Python foundation that you'll need every single day as an AI engineer. By the end of these 2 weeks, you should be able to write async code, validate data with Pydantic, and call APIs efficiently.

---

## Table of Contents

1. [Why These Topics Matter](#why-these-topics-matter)
2. [Async/Await and Asyncio](#1-asyncawait-and-asyncio)
3. [Type Hints](#2-type-hints)
4. [Pydantic v2](#3-pydantic-v2)
5. [Generators](#4-generators)
6. [Context Managers](#5-context-managers)
7. [Decorators](#6-decorators)
8. [HTTP Clients (httpx and requests)](#7-http-clients)
9. [Error Handling and Retries](#8-error-handling-and-retries)
10. [Final Project](#final-project)
11. [Self-Check Questions](#self-check-questions)

---

## Why These Topics Matter

Before diving in, here's **why** each topic is critical for AI engineering:

| Topic | Why You Need It |
|-------|-----------------|
| **Async/await** | LLM API calls take 1-30 seconds. Without async, your app freezes. |
| **Type hints** | Modern AI libraries (Pydantic, FastAPI) require them. |
| **Pydantic** | Used to validate LLM outputs and define tool schemas. |
| **Generators** | Streaming LLM responses come as generators. |
| **Context managers** | Managing database/API connections cleanly. |
| **Decorators** | Adding retry logic, logging, caching to functions. |
| **HTTP clients** | Every LLM API call is an HTTP request. |
| **Error handling** | LLM APIs fail often. You must handle this gracefully. |

---

## 1. Async/Await and Asyncio

### What is Async?

**Simple explanation:** Imagine you're cooking. You put rice on the stove (takes 20 mins), then you start chopping vegetables (takes 10 mins). You don't STAND and watch the rice — you do other work while it cooks.

That's async. Your code doesn't wait idly while waiting for slow operations (like API calls).

### Synchronous vs Asynchronous

**Synchronous (slow):**
```python
import time

def get_weather(city):
    time.sleep(2)  # Pretend this is an API call
    return f"Weather in {city}: Sunny"

# This takes 6 seconds (2 + 2 + 2)
print(get_weather("London"))
print(get_weather("Mumbai"))
print(get_weather("Tokyo"))
```

**Asynchronous (fast):**
```python
import asyncio

async def get_weather(city):
    await asyncio.sleep(2)  # Pretend this is an API call
    return f"Weather in {city}: Sunny"

async def main():
    # All 3 run in parallel - takes only 2 seconds total!
    results = await asyncio.gather(
        get_weather("London"),
        get_weather("Mumbai"),
        get_weather("Tokyo")
    )
    for r in results:
        print(r)

asyncio.run(main())
```

### Key Concepts

**1. `async def`** — Defines an async function (called a "coroutine")
```python
async def fetch_data():
    return "data"
```

**2. `await`** — Pauses the function until the awaited operation completes
```python
async def main():
    result = await fetch_data()  # Wait for this to finish
    print(result)
```

**3. `asyncio.run()`** — Runs an async function from sync code
```python
asyncio.run(main())
```

**4. `asyncio.gather()`** — Run multiple async functions in parallel
```python
results = await asyncio.gather(task1(), task2(), task3())
```

### Common Pitfall

You **cannot** call an async function directly:

```python
# ❌ WRONG
result = fetch_data()  # Returns a coroutine object, NOT the data!

# ✅ CORRECT
result = await fetch_data()  # Inside another async function
# OR
result = asyncio.run(fetch_data())  # From sync code
```

### Real Example: Calling 3 LLM APIs in Parallel

```python
import asyncio
import httpx

async def call_llm(prompt, model_url):
    async with httpx.AsyncClient() as client:
        response = await client.post(model_url, json={"prompt": prompt})
        return response.json()

async def compare_models(prompt):
    results = await asyncio.gather(
        call_llm(prompt, "https://api.openai.com/..."),
        call_llm(prompt, "https://api.anthropic.com/..."),
        call_llm(prompt, "https://api.gemini.com/...")
    )
    return results

# Without async: 3 calls × 5 seconds = 15 seconds
# With async: max(5s, 5s, 5s) = 5 seconds
```

### Practice Exercise

Write an async function that fetches data from 5 different URLs in parallel and prints the time saved compared to sequential calls.

---

## 2. Type Hints

### What Are Type Hints?

Type hints tell Python (and other developers) what type of data a function expects and returns.

**Without type hints:**
```python
def add(a, b):
    return a + b

# Could be numbers? Strings? Lists? Who knows!
```

**With type hints:**
```python
def add(a: int, b: int) -> int:
    return a + b

# Clear: takes two ints, returns an int
```

### Common Type Hints

```python
from typing import List, Dict, Optional, Union, Tuple, Any, Callable

# Basic types
name: str = "Dinesh"
age: int = 30
price: float = 99.99
is_active: bool = True

# Collections
names: list[str] = ["Alice", "Bob"]           # List of strings
scores: dict[str, int] = {"Alice": 90}         # Dict with str keys, int values
coords: tuple[float, float] = (12.97, 77.59)   # Tuple of two floats

# Optional (can be None)
middle_name: Optional[str] = None  # Can be str or None
# Same as: middle_name: str | None = None  (Python 3.10+)

# Union (multiple possible types)
user_id: Union[int, str] = "abc123"
# Same as: user_id: int | str = "abc123"  (Python 3.10+)

# Functions
def greet(name: str) -> str:
    return f"Hello, {name}"

# Function as argument
def apply(func: Callable[[int], int], x: int) -> int:
    return func(x)
```

### Why Type Hints Matter for AI

Pydantic, FastAPI, and modern AI libraries **require** type hints to work:

```python
from pydantic import BaseModel

class LLMResponse(BaseModel):
    text: str        # Must be string
    tokens: int      # Must be integer
    cost: float      # Must be float
```

### Practice Exercise

Add type hints to this function:
```python
def process_users(users, threshold):
    result = []
    for user in users:
        if user["score"] > threshold:
            result.append(user["name"])
    return result
```

**Answer:**
```python
def process_users(users: list[dict], threshold: float) -> list[str]:
    result: list[str] = []
    for user in users:
        if user["score"] > threshold:
            result.append(user["name"])
    return result
```

---

## 3. Pydantic v2

### What is Pydantic?

Pydantic validates data. Give it a schema, and it ensures incoming data matches that schema — or it raises clear errors.

**Why it's critical for AI:**
- Validate LLM outputs (LLMs can return weird/wrong data)
- Define tool schemas for function calling
- Type-safe configuration

### Installation

```bash
pip install pydantic
```

### Basic Example

```python
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int
    email: str

# Valid - works fine
user = User(name="Dinesh", age=30, email="d@example.com")
print(user.name)  # "Dinesh"

# Invalid - raises ValidationError
user = User(name="Dinesh", age="thirty", email="d@example.com")
# ❌ ValidationError: age must be an integer
```

### Field Validation

```python
from pydantic import BaseModel, Field, EmailStr

class User(BaseModel):
    name: str = Field(..., min_length=2, max_length=50)
    age: int = Field(..., ge=0, le=120)  # ge = greater equal, le = less equal
    email: EmailStr  # Validates email format
    bio: str = Field(default="", max_length=500)  # Optional with default
```

### Real AI Example: Validating LLM Output

```python
from pydantic import BaseModel, Field
from typing import Literal

class SentimentAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str

# When LLM returns JSON, validate it:
llm_output = '{"sentiment": "positive", "confidence": 0.95, "reasoning": "..."}'
result = SentimentAnalysis.model_validate_json(llm_output)

print(result.sentiment)    # "positive"
print(result.confidence)   # 0.95

# If LLM returns bad data, you get an error - not a silent failure
bad_output = '{"sentiment": "happy", "confidence": 1.5}'
SentimentAnalysis.model_validate_json(bad_output)
# ❌ ValidationError: sentiment must be positive/negative/neutral
```

### Nested Models

```python
class Address(BaseModel):
    street: str
    city: str
    country: str

class User(BaseModel):
    name: str
    addresses: list[Address]  # List of Address objects

user = User(
    name="Dinesh",
    addresses=[
        {"street": "123 Main", "city": "London", "country": "UK"},
        {"street": "456 Park", "city": "Mumbai", "country": "India"}
    ]
)
```

### Custom Validators

```python
from pydantic import BaseModel, field_validator

class Product(BaseModel):
    name: str
    price: float

    @field_validator("price")
    @classmethod
    def price_must_be_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError("Price must be positive")
        return v
```

### Converting To/From JSON

```python
# Object to JSON
user = User(name="Dinesh", age=30, email="d@example.com")
json_str = user.model_dump_json()

# Object to dict
user_dict = user.model_dump()

# JSON to object
user = User.model_validate_json(json_str)

# Dict to object
user = User.model_validate(user_dict)
```

### Practice Exercise

Create a Pydantic model for an LLM tool call:
```python
class ToolCall(BaseModel):
    tool_name: str
    arguments: dict
    call_id: str
```

---

## 4. Generators

### What Are Generators?

Generators produce values one at a time, instead of all at once. Perfect for handling large data or **streaming responses** (like LLM streaming).

### Basic Generator

```python
def count_up_to(n):
    for i in range(n):
        yield i  # 'yield' instead of 'return'

# Use it
gen = count_up_to(5)
for num in gen:
    print(num)  # 0, 1, 2, 3, 4
```

### Why Use Generators?

**Without generator (uses lots of memory):**
```python
def get_million_numbers():
    return [i for i in range(1_000_000)]  # All in memory at once!

numbers = get_million_numbers()  # Uses ~40 MB
```

**With generator (memory-efficient):**
```python
def get_million_numbers():
    for i in range(1_000_000):
        yield i  # One at a time

numbers = get_million_numbers()  # Uses almost no memory
for n in numbers:
    print(n)
```

### Real AI Example: Streaming LLM Response

```python
def stream_llm_response(prompt):
    # Pretend this calls the LLM
    response_chunks = ["Hello", " there", "!", " How", " can", " I", " help?"]
    for chunk in response_chunks:
        yield chunk

# Print as it "arrives"
for chunk in stream_llm_response("Hi"):
    print(chunk, end="", flush=True)
# Output: Hello there! How can I help?
```

### Async Generators (For Real LLM Streaming)

```python
import asyncio

async def stream_from_llm(prompt):
    chunks = ["Hello", " world", "!"]
    for chunk in chunks:
        await asyncio.sleep(0.5)  # Simulate network delay
        yield chunk

async def main():
    async for chunk in stream_from_llm("Hi"):
        print(chunk, end="", flush=True)

asyncio.run(main())
```

---

## 5. Context Managers

### What Are Context Managers?

Context managers handle setup and cleanup automatically. The `with` statement is the syntax.

**Without context manager:**
```python
file = open("data.txt")
content = file.read()
file.close()  # Easy to forget!
```

**With context manager:**
```python
with open("data.txt") as file:
    content = file.read()
# File closes automatically, even if there's an error
```

### Why It Matters for AI

You'll use context managers for:
- Database connections
- HTTP client sessions
- File operations
- Tracing/logging spans

### Real Example: HTTP Session

```python
import httpx

async def fetch_data():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://api.example.com")
        return response.json()
    # Client automatically closes
```

### Creating Your Own Context Manager

```python
from contextlib import contextmanager
import time

@contextmanager
def timer(label: str):
    start = time.time()
    yield  # Code inside `with` runs here
    end = time.time()
    print(f"{label} took {end - start:.2f} seconds")

# Use it
with timer("LLM call"):
    # ... do some work ...
    time.sleep(2)
# Output: LLM call took 2.00 seconds
```

---

## 6. Decorators

### What Are Decorators?

Decorators wrap a function to add extra behavior — like logging, retries, or caching — without changing the function itself.

### Basic Decorator

```python
def log_call(func):
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        result = func(*args, **kwargs)
        print(f"Done")
        return result
    return wrapper

@log_call
def greet(name):
    return f"Hello, {name}"

greet("Dinesh")
# Output:
# Calling greet
# Done
```

### Real AI Example: Retry Decorator

```python
import time
from functools import wraps

def retry(max_attempts=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise  # Final attempt failed
                    print(f"Attempt {attempt + 1} failed: {e}. Retrying...")
                    time.sleep(delay)
        return wrapper
    return decorator

@retry(max_attempts=3, delay=2)
def call_llm(prompt):
    # If this fails, it retries up to 3 times
    response = some_api_call(prompt)
    return response
```

### Common Decorators You'll Use

```python
from functools import lru_cache, wraps

# Cache results (faster repeated calls)
@lru_cache(maxsize=100)
def expensive_computation(x: int) -> int:
    return x ** 2

# Always preserve original function metadata
def my_decorator(func):
    @wraps(func)  # ← Important!
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper
```

---

## 7. HTTP Clients

### Two Main Libraries

- **`requests`** — Synchronous, simple. Good for scripts.
- **`httpx`** — Both sync and async. **Use this for AI work.**

### Installation

```bash
pip install httpx requests
```

### Sync Example with httpx

```python
import httpx

response = httpx.get("https://api.github.com/users/torvalds")
print(response.status_code)  # 200
print(response.json())       # {"login": "torvalds", ...}
```

### Async Example with httpx

```python
import httpx
import asyncio

async def get_user(username: str):
    async with httpx.AsyncClient() as client:
        response = await client.get(f"https://api.github.com/users/{username}")
        return response.json()

async def main():
    user = await get_user("torvalds")
    print(user["name"])

asyncio.run(main())
```

### POST Request (Like Calling an LLM)

```python
async def call_llm(prompt: str):
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": "Bearer YOUR_API_KEY",
                "Content-Type": "application/json"
            },
            json={
                "model": "gpt-4",
                "messages": [{"role": "user", "content": prompt}]
            }
        )
        response.raise_for_status()  # Raise error if HTTP error
        return response.json()
```

### Important Settings

```python
# Set timeout (LLM calls can be slow)
async with httpx.AsyncClient(timeout=30.0) as client:
    pass

# Reuse client for multiple requests (more efficient)
async with httpx.AsyncClient() as client:
    response1 = await client.get("...")
    response2 = await client.get("...")
    response3 = await client.get("...")
```

---

## 8. Error Handling and Retries

### Basic Try/Except

```python
try:
    response = await call_llm("Hello")
except httpx.TimeoutException:
    print("Request timed out")
except httpx.HTTPStatusError as e:
    print(f"HTTP error: {e.response.status_code}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Exponential Backoff

When an API fails, don't retry immediately. Wait longer each time.

```python
import asyncio
import random

async def call_with_backoff(func, max_attempts=5):
    for attempt in range(max_attempts):
        try:
            return await func()
        except Exception as e:
            if attempt == max_attempts - 1:
                raise

            # Exponential backoff with jitter
            wait = (2 ** attempt) + random.uniform(0, 1)
            print(f"Attempt {attempt + 1} failed. Waiting {wait:.1f}s...")
            await asyncio.sleep(wait)

# Wait times: 1s, 2s, 4s, 8s, 16s (plus random jitter)
```

### Using `tenacity` Library (Recommended)

```bash
pip install tenacity
```

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10)
)
async def call_llm(prompt: str):
    response = await some_api_call(prompt)
    return response

# Automatically retries with exponential backoff
```

### What to Retry vs Not Retry

**✅ Retry these:**
- Network timeouts
- 429 (rate limit)
- 500, 502, 503 (server errors)
- Connection errors

**❌ Don't retry these:**
- 400 (bad request - your fault)
- 401 (auth error - won't fix itself)
- 404 (not found)

```python
from tenacity import retry, retry_if_exception_type

@retry(
    retry=retry_if_exception_type((httpx.TimeoutException, httpx.NetworkError)),
    stop=stop_after_attempt(3)
)
async def call_llm(prompt: str):
    pass
```

---

## Final Project

### Project: Async API Aggregator

Build an async script that calls 3 different APIs in parallel, validates responses with Pydantic, and aggregates results.

### Requirements

1. Use `httpx.AsyncClient` for HTTP calls
2. Call at least 3 free public APIs in parallel
3. Validate each response with Pydantic
4. Add retry logic with exponential backoff
5. Use type hints everywhere
6. Measure and print time saved vs sequential calls

### Suggested APIs (all free, no auth)

- `https://api.github.com/users/{username}` — GitHub user info
- `https://api.coindesk.com/v1/bpi/currentprice.json` — Bitcoin price
- `https://api.agify.io/?name={name}` — Predict age from name
- `https://catfact.ninja/fact` — Random cat fact

### Starter Template

```python
import asyncio
import time
from typing import Any
import httpx
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential


class GitHubUser(BaseModel):
    login: str
    name: str | None
    public_repos: int


class CatFact(BaseModel):
    fact: str
    length: int


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
async def fetch_github_user(client: httpx.AsyncClient, username: str) -> GitHubUser:
    response = await client.get(f"https://api.github.com/users/{username}")
    response.raise_for_status()
    return GitHubUser.model_validate(response.json())


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=5))
async def fetch_cat_fact(client: httpx.AsyncClient) -> CatFact:
    response = await client.get("https://catfact.ninja/fact")
    response.raise_for_status()
    return CatFact.model_validate(response.json())


async def fetch_all_parallel(username: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        results = await asyncio.gather(
            fetch_github_user(client, username),
            fetch_cat_fact(client),
            # Add more APIs here
        )
    return {
        "github_user": results[0],
        "cat_fact": results[1],
    }


async def fetch_all_sequential(username: str) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=10.0) as client:
        github = await fetch_github_user(client, username)
        cat = await fetch_cat_fact(client)
    return {"github_user": github, "cat_fact": cat}


async def main():
    # Sequential
    start = time.time()
    seq_results = await fetch_all_sequential("torvalds")
    seq_time = time.time() - start

    # Parallel
    start = time.time()
    par_results = await fetch_all_parallel("torvalds")
    par_time = time.time() - start

    print(f"Sequential: {seq_time:.2f}s")
    print(f"Parallel:   {par_time:.2f}s")
    print(f"Speedup:    {seq_time / par_time:.1f}x")
    print()
    print("GitHub:", par_results["github_user"].name)
    print("Cat fact:", par_results["cat_fact"].fact)


if __name__ == "__main__":
    asyncio.run(main())
```

### What to Submit (to your GitHub)

```
week-01-02-python-foundations/
├── README.md           # What you learned, how to run
├── requirements.txt    # httpx, pydantic, tenacity
├── api_aggregator.py   # The main project
└── notes.md            # Your personal notes
```

---

## Self-Check Questions

Test your understanding before moving to Week 3:

1. What's the difference between `def` and `async def`?
2. Why can't you call an async function directly without `await`?
3. What does `asyncio.gather()` do?
4. What's the difference between `Optional[str]` and `str | None`?
5. How does Pydantic help with LLM outputs?
6. What's the advantage of generators over lists for large data?
7. Why use a context manager with `httpx.AsyncClient`?
8. Write a decorator that logs the execution time of any function.
9. Which HTTP errors should you retry, and which shouldn't you?
10. What is exponential backoff and why use it?

### Answers (Quick Reference)

1. `def` is sync (blocks). `async def` is async (can await).
2. It returns a coroutine object, not the result. You need `await`.
3. Runs multiple coroutines in parallel and waits for all to finish.
4. They mean the same thing. `|` syntax requires Python 3.10+.
5. Validates structure/types. Catches LLM mistakes early instead of silent failures.
6. Generators produce one value at a time — uses far less memory.
7. Ensures the client connection is properly closed, even on errors.
8. See the timer example in the Decorators section.
9. Retry: timeouts, 429, 500, 502, 503. Don't retry: 400, 401, 404.
10. Wait progressively longer between retries (1s, 2s, 4s, 8s) to avoid hammering a struggling server.

---

## Resources

### Official Docs
- [Python asyncio docs](https://docs.python.org/3/library/asyncio.html)
- [Pydantic v2 docs](https://docs.pydantic.dev/latest/)
- [httpx docs](https://www.python-httpx.org/)
- [tenacity docs](https://tenacity.readthedocs.io/)

### Recommended Reading
- "Real Python" articles on async/await
- Pydantic's "Migration Guide" (v1 to v2)

### Videos
- Search YouTube: "asyncio in 100 seconds"
- Search YouTube: "Pydantic crash course"

---

## Next Up

**Week 3:** LLM API Fundamentals — calling OpenAI, Anthropic, and Gemini directly without frameworks.

You'll be glad you mastered async — every LLM call benefits from it.
