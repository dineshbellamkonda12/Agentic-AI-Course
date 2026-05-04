# Week 3: LLM API Fundamentals

> **Goal:** Learn to talk directly to LLMs (OpenAI, Anthropic, Gemini) using their official SDKs — no LangChain, no frameworks. Once you understand the raw APIs, every framework will make sense instantly.

---

## Table of Contents

1. [Why Learn Raw APIs First](#why-learn-raw-apis-first)
2. [How LLM APIs Actually Work](#1-how-llm-apis-actually-work)
3. [The Messages Format](#2-the-messages-format)
4. [System Prompts](#3-system-prompts)
5. [Key Parameters Explained](#4-key-parameters-explained)
6. [OpenAI SDK](#5-openai-sdk)
7. [Anthropic SDK](#6-anthropic-sdk)
8. [Google Gemini SDK](#7-google-gemini-sdk)
9. [Streaming Responses](#8-streaming-responses)
10. [Token Counting and Costs](#9-token-counting-and-costs)
11. [When to Use Which Provider](#10-when-to-use-which-provider)
12. [Final Project](#final-project)
13. [Self-Check Questions](#self-check-questions)

---

## Why Learn Raw APIs First

Most beginners jump straight to LangChain or LlamaIndex. **Bad idea.** Here's why:

| If you skip raw APIs | If you learn raw APIs first |
|---------------------|----------------------------|
| Frameworks feel like magic | You understand exactly what's happening |
| Hard to debug failures | You can debug at the network level |
| Locked into one framework | You can use any framework or none |
| Don't know real costs | You understand pricing precisely |
| Can't optimize | You know where to optimize |

**Rule:** If you can't build it without a framework, you don't understand it.

---

## 1. How LLM APIs Actually Work

### The Big Picture

Every LLM API call is just an **HTTP POST request** with a JSON body. That's it.

```
Your Code  →  HTTPS POST  →  Provider Server  →  Returns JSON
```

### What Actually Gets Sent

```http
POST https://api.openai.com/v1/chat/completions
Authorization: Bearer sk-...
Content-Type: application/json

{
  "model": "gpt-4o-mini",
  "messages": [
    {"role": "user", "content": "What is Python?"}
  ]
}
```

### What Comes Back

```json
{
  "id": "chatcmpl-abc123",
  "model": "gpt-4o-mini",
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "Python is a high-level programming language..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 8,
    "completion_tokens": 50,
    "total_tokens": 58
  }
}
```

The SDK is just a wrapper that makes this nicer to use in Python.

---

## 2. The Messages Format

### The Conversation as a List

LLMs are stateless. **They have no memory.** Every call, you send the entire conversation as a list of messages.

### Three Message Roles

| Role | Purpose | Example |
|------|---------|---------|
| `system` | Instructions for the LLM (its personality, rules) | "You are a helpful tutor." |
| `user` | What the human said | "Explain recursion." |
| `assistant` | What the LLM previously said | "Recursion is when..." |

### Single Turn

```python
messages = [
    {"role": "user", "content": "What is 2 + 2?"}
]
# LLM responds with "4"
```

### Multi-Turn (Conversation)

To have a conversation, append every response to the messages list:

```python
messages = [
    {"role": "system", "content": "You are a math tutor."},
    {"role": "user", "content": "What is 2 + 2?"},
    {"role": "assistant", "content": "2 + 2 equals 4."},
    {"role": "user", "content": "Now multiply by 3."}
]
# LLM uses ALL previous messages as context, responds with "12"
```

### Critical Concept: Context Window

Every model has a **context window** — the max number of tokens it can handle in one call (input + output combined).

| Model | Context Window |
|-------|---------------|
| GPT-4o-mini | 128,000 tokens |
| Claude Sonnet | 200,000 tokens |
| Gemini 1.5 Pro | 2,000,000 tokens |

**1 token ≈ 0.75 English words.** So 100,000 tokens ≈ 75,000 words ≈ 150 pages.

If your conversation grows too big, you need to **summarize old messages** or **drop them**.

---

## 3. System Prompts

### What's a System Prompt?

The system prompt sets the LLM's behavior, tone, and rules. It's the most important prompt in any AI app.

```python
messages = [
    {
        "role": "system",
        "content": "You are a Python tutor. Always explain with simple examples. Never write more than 3 sentences."
    },
    {"role": "user", "content": "What is a list?"}
]
```

### Good vs Bad System Prompts

**❌ Bad — too vague:**
```
You are helpful.
```

**✅ Good — specific:**
```
You are a Python tutor for beginners. Always:
- Explain concepts using a simple real-world analogy first
- Then show a code example with comments
- Keep code under 10 lines
- End with one practice exercise
Never use jargon without explaining it.
```

### Tips for Writing System Prompts

1. **Be specific** about format, length, tone
2. **Use lists** for clearer instructions
3. **Give examples** of what you want
4. **Define what NOT to do** (negative instructions)
5. **Set persona** if relevant ("You are a senior data engineer...")

### Real Example: A Code Reviewer Bot

```python
SYSTEM_PROMPT = """
You are a strict but kind senior Python code reviewer.

For every code snippet you receive:
1. Identify bugs (if any)
2. Suggest performance improvements
3. Suggest readability improvements
4. Rate the code: poor / okay / good / excellent

Format your response as:
**Bugs:** ...
**Performance:** ...
**Readability:** ...
**Rating:** ...

Be honest. Don't sugarcoat issues.
"""
```

---

## 4. Key Parameters Explained

These parameters control how the LLM responds. Knowing them well is essential.

### `temperature` (0.0 to 2.0)

Controls randomness/creativity.

- `0.0` = Deterministic. Same input → same output. Use for: extraction, classification, code.
- `0.7` = Balanced (default). Use for: chat, general Q&A.
- `1.0+` = Creative. Use for: storytelling, brainstorming.

```python
# Extracting structured data — use 0
response = client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=0,
    messages=[...]
)

# Writing a poem — use 1.0
response = client.chat.completions.create(
    model="gpt-4o-mini",
    temperature=1.0,
    messages=[...]
)
```

### `max_tokens`

Maximum tokens the LLM can generate in its response.

```python
max_tokens=500  # Response capped at ~375 words
```

**Why it matters:**
- Cost control (longer = more expensive)
- Prevents runaway responses
- Required by Anthropic (must specify)

### `top_p` (0.0 to 1.0)

Alternative to temperature. **Only change one, not both.** Most people just use temperature.

### `stop` (Stop Sequences)

Strings that, if generated, stop the response immediately.

```python
stop=["\n\n", "END"]
# Stops if it generates two newlines or "END"
```

### `presence_penalty` and `frequency_penalty` (-2.0 to 2.0)

Discourage repeated words. Rarely needed — leave at 0.

### Cheat Sheet

| Use case | temperature | max_tokens |
|----------|------------|------------|
| Data extraction | 0 | 500 |
| Code generation | 0.2 | 2000 |
| Q&A | 0.5 | 1000 |
| Chat | 0.7 | 1000 |
| Creative writing | 1.0 | 2000 |

---

## 5. OpenAI SDK

### Installation

```bash
pip install openai
```

### Setup

Get your API key from https://platform.openai.com/api-keys.

Store it in a `.env` file (never commit this!):

```bash
# .env
OPENAI_API_KEY=sk-...
```

```python
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
```

### Basic Call

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "You are a helpful tutor."},
        {"role": "user", "content": "Explain recursion in one sentence."}
    ],
    temperature=0.5,
    max_tokens=100
)

print(response.choices[0].message.content)
print(f"Tokens used: {response.usage.total_tokens}")
```

### Async Version

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()

async def ask(prompt: str):
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Call in parallel
async def main():
    results = await asyncio.gather(
        ask("What is Python?"),
        ask("What is JavaScript?"),
        ask("What is Rust?")
    )
    for r in results:
        print(r, "\n---")

asyncio.run(main())
```

### Common OpenAI Models

| Model | Use Case | Cost (per 1M tokens) |
|-------|----------|---------------------|
| `gpt-4o` | Best general model | $2.50 in / $10 out |
| `gpt-4o-mini` | Fast and cheap | $0.15 in / $0.60 out |
| `o1-mini` | Reasoning tasks | $3 in / $12 out |

> Pricing changes — always check https://openai.com/pricing.

---

## 6. Anthropic SDK

### Installation

```bash
pip install anthropic
```

### Setup

Get key from https://console.anthropic.com/.

```python
import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
```

### Basic Call

**Note:** Anthropic puts the system prompt as a **separate parameter**, not inside messages.

```python
response = client.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=1000,  # ⚠️ Required for Anthropic!
    system="You are a helpful tutor.",
    messages=[
        {"role": "user", "content": "Explain recursion in one sentence."}
    ]
)

print(response.content[0].text)
print(f"Tokens used: {response.usage.input_tokens + response.usage.output_tokens}")
```

### Key Differences from OpenAI

| Feature | OpenAI | Anthropic |
|---------|--------|-----------|
| System prompt | In messages list | Separate `system` param |
| `max_tokens` | Optional | **Required** |
| Response path | `response.choices[0].message.content` | `response.content[0].text` |
| Roles allowed | system, user, assistant | user, assistant only |

### Async Version

```python
from anthropic import AsyncAnthropic

client = AsyncAnthropic()

async def ask_claude(prompt: str):
    response = await client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text
```

### Common Anthropic Models

| Model | Use Case |
|-------|----------|
| `claude-opus-4-7` | Most capable, complex tasks |
| `claude-sonnet-4-5` | Best balance of speed and intelligence |
| `claude-haiku-4-5` | Fast and cheap |

> Always check current pricing at https://www.anthropic.com/pricing.

---

## 7. Google Gemini SDK

### Installation

```bash
pip install google-generativeai
```

### Setup

Get key from https://aistudio.google.com/apikey.

```python
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
```

### Basic Call

Gemini's API is structured differently — you create a model object first.

```python
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    system_instruction="You are a helpful tutor."
)

response = model.generate_content(
    "Explain recursion in one sentence.",
    generation_config={
        "temperature": 0.5,
        "max_output_tokens": 100
    }
)

print(response.text)
print(f"Tokens used: {response.usage_metadata.total_token_count}")
```

### Multi-Turn Chat

```python
model = genai.GenerativeModel("gemini-1.5-flash")
chat = model.start_chat()

response1 = chat.send_message("What is 2 + 2?")
print(response1.text)

response2 = chat.send_message("Multiply that by 3.")
print(response2.text)
# Gemini remembers the previous turn automatically
```

### Common Gemini Models

| Model | Use Case |
|-------|----------|
| `gemini-1.5-pro` | Most capable, huge context window |
| `gemini-1.5-flash` | Fast and cheap |
| `gemini-2.0-flash` | Latest, multimodal |

---

## 8. Streaming Responses

### Why Streaming?

Without streaming, you wait 10 seconds, then get the full response.
With streaming, you see text appear word by word — like ChatGPT does.

**Better UX = mandatory for chat apps.**

### OpenAI Streaming

```python
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Tell me a story"}],
    stream=True  # ← The magic flag
)

for chunk in stream:
    delta = chunk.choices[0].delta.content
    if delta:
        print(delta, end="", flush=True)
```

### Anthropic Streaming

```python
with client.messages.stream(
    model="claude-sonnet-4-5",
    max_tokens=1000,
    messages=[{"role": "user", "content": "Tell me a story"}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

### Gemini Streaming

```python
model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("Tell me a story", stream=True)

for chunk in response:
    print(chunk.text, end="", flush=True)
```

### Async Streaming (OpenAI Example)

```python
async def stream_response(prompt: str):
    stream = await async_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=True
    )

    async for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            print(delta, end="", flush=True)
```

---

## 9. Token Counting and Costs

### What is a Token?

A token is roughly a chunk of a word.

- "hello" → 1 token
- "antidisestablishmentarianism" → 6 tokens
- "I love coding" → 3 tokens

**Rule of thumb: 1 token ≈ 4 characters ≈ 0.75 English words.**

### Why Count Tokens?

1. **Cost** — You pay per token (input + output separately)
2. **Context limits** — Avoid exceeding model's max
3. **Performance** — Fewer tokens = faster responses

### Counting Tokens (OpenAI)

```bash
pip install tiktoken
```

```python
import tiktoken

def count_tokens(text: str, model: str = "gpt-4o-mini") -> int:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))

text = "Hello, how are you doing today?"
print(count_tokens(text))  # 8 tokens
```

### Counting Tokens for Messages

```python
def count_message_tokens(messages: list, model: str = "gpt-4o-mini") -> int:
    encoding = tiktoken.encoding_for_model(model)
    total = 0
    for msg in messages:
        total += 4  # Per-message overhead
        total += len(encoding.encode(msg["content"]))
    return total
```

### Calculating Cost

```python
# Example: GPT-4o-mini pricing
INPUT_COST_PER_1M = 0.15   # dollars per 1M input tokens
OUTPUT_COST_PER_1M = 0.60  # dollars per 1M output tokens

def calculate_cost(input_tokens: int, output_tokens: int) -> float:
    input_cost = (input_tokens / 1_000_000) * INPUT_COST_PER_1M
    output_cost = (output_tokens / 1_000_000) * OUTPUT_COST_PER_1M
    return input_cost + output_cost

# After an API call
input_tok = response.usage.prompt_tokens
output_tok = response.usage.completion_tokens
cost = calculate_cost(input_tok, output_tok)
print(f"This call cost ${cost:.6f}")
```

### Sample Cost Calculations

| Tokens | GPT-4o-mini | GPT-4o | Claude Sonnet |
|--------|------------|--------|---------------|
| 1K in / 1K out | $0.0008 | $0.0125 | $0.018 |
| 100K in / 10K out | $0.021 | $0.35 | $0.45 |
| 1M in / 100K out | $0.21 | $3.50 | $4.50 |

> Always set token budgets in production. A bug can cost you hundreds of dollars overnight.

---

## 10. When to Use Which Provider

No provider is best at everything. Here's an honest breakdown:

### OpenAI
**Strengths:**
- Strong general performance
- Best ecosystem (most tools support it)
- Good function calling
- Reasoning models (o1)

**Weaknesses:**
- Smaller context window than Gemini

### Anthropic (Claude)
**Strengths:**
- Best for long, nuanced writing
- Excellent at following instructions
- Strong at coding
- 200K context window
- Best safety/refusal behavior

**Weaknesses:**
- Slightly more expensive than equivalent OpenAI models
- Smaller ecosystem than OpenAI

### Google Gemini
**Strengths:**
- Massive 2M context window (read entire books!)
- Cheapest of the three
- Strong multimodal (vision, audio)
- Free tier

**Weaknesses:**
- Less consistent for complex reasoning
- API is slightly more awkward

### Practical Recommendation

For learning/portfolio:
- **Default:** OpenAI (`gpt-4o-mini` for cheap, `gpt-4o` for quality)
- **Long documents:** Gemini Flash
- **Complex writing/reasoning:** Claude Sonnet
- **Multi-provider apps:** Use all three to compare

---

## Final Project

### Project: Multi-Provider CLI Chatbot

Build a command-line chatbot that lets you swap between OpenAI, Anthropic, and Gemini using the same interface.

### Requirements

1. Same code interface for all 3 providers (use a base class)
2. Streaming responses to terminal
3. Multi-turn conversation (remembers history)
4. Track total tokens and cost across the session
5. Async implementation
6. `/switch <provider>` command to change models mid-conversation

### Project Structure

```
week-03-llm-apis/
├── README.md
├── requirements.txt
├── .env                    # API keys (don't commit!)
├── .gitignore
├── chatbot.py              # Main CLI app
├── providers/
│   ├── __init__.py
│   ├── base.py             # Abstract base class
│   ├── openai_provider.py
│   ├── anthropic_provider.py
│   └── gemini_provider.py
└── notes.md
```

### Starter Code

**`providers/base.py`**

```python
from abc import ABC, abstractmethod
from typing import AsyncIterator
from pydantic import BaseModel


class Message(BaseModel):
    role: str
    content: str


class TokenUsage(BaseModel):
    input_tokens: int
    output_tokens: int
    cost: float


class LLMProvider(ABC):
    """Base class — all providers must implement these methods."""

    @abstractmethod
    async def stream_chat(
        self,
        messages: list[Message],
        system: str | None = None
    ) -> AsyncIterator[str]:
        """Stream a chat response chunk by chunk."""
        ...

    @abstractmethod
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost in USD."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...
```

**`providers/openai_provider.py`**

```python
import os
from typing import AsyncIterator
from openai import AsyncOpenAI
from .base import LLMProvider, Message


class OpenAIProvider(LLMProvider):
    INPUT_COST_PER_1M = 0.15
    OUTPUT_COST_PER_1M = 0.60

    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    @property
    def name(self) -> str:
        return f"OpenAI ({self.model})"

    async def stream_chat(
        self,
        messages: list[Message],
        system: str | None = None
    ) -> AsyncIterator[str]:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.extend([{"role": m.role, "content": m.content} for m in messages])

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=msgs,
            stream=True
        )

        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            (input_tokens / 1_000_000) * self.INPUT_COST_PER_1M
            + (output_tokens / 1_000_000) * self.OUTPUT_COST_PER_1M
        )
```

**`providers/anthropic_provider.py`**

```python
import os
from typing import AsyncIterator
from anthropic import AsyncAnthropic
from .base import LLMProvider, Message


class AnthropicProvider(LLMProvider):
    INPUT_COST_PER_1M = 3.0   # Sonnet pricing example
    OUTPUT_COST_PER_1M = 15.0

    def __init__(self, model: str = "claude-sonnet-4-5"):
        self.client = AsyncAnthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model = model

    @property
    def name(self) -> str:
        return f"Anthropic ({self.model})"

    async def stream_chat(
        self,
        messages: list[Message],
        system: str | None = None
    ) -> AsyncIterator[str]:
        msgs = [{"role": m.role, "content": m.content} for m in messages]

        async with self.client.messages.stream(
            model=self.model,
            max_tokens=2000,
            system=system or "",
            messages=msgs
        ) as stream:
            async for text in stream.text_stream:
                yield text

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            (input_tokens / 1_000_000) * self.INPUT_COST_PER_1M
            + (output_tokens / 1_000_000) * self.OUTPUT_COST_PER_1M
        )
```

**`providers/gemini_provider.py`**

```python
import os
from typing import AsyncIterator
import google.generativeai as genai
from .base import LLMProvider, Message


class GeminiProvider(LLMProvider):
    INPUT_COST_PER_1M = 0.075
    OUTPUT_COST_PER_1M = 0.30

    def __init__(self, model: str = "gemini-1.5-flash"):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model_name = model

    @property
    def name(self) -> str:
        return f"Gemini ({self.model_name})"

    async def stream_chat(
        self,
        messages: list[Message],
        system: str | None = None
    ) -> AsyncIterator[str]:
        model = genai.GenerativeModel(
            model_name=self.model_name,
            system_instruction=system
        )

        # Convert message format for Gemini
        history = []
        for m in messages[:-1]:
            role = "model" if m.role == "assistant" else "user"
            history.append({"role": role, "parts": [m.content]})

        chat = model.start_chat(history=history)
        response = await chat.send_message_async(
            messages[-1].content,
            stream=True
        )

        async for chunk in response:
            if chunk.text:
                yield chunk.text

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            (input_tokens / 1_000_000) * self.INPUT_COST_PER_1M
            + (output_tokens / 1_000_000) * self.OUTPUT_COST_PER_1M
        )
```

**`chatbot.py`**

```python
import asyncio
from dotenv import load_dotenv
from providers.base import Message
from providers.openai_provider import OpenAIProvider
from providers.anthropic_provider import AnthropicProvider
from providers.gemini_provider import GeminiProvider

load_dotenv()

SYSTEM_PROMPT = "You are a helpful, concise assistant. Keep answers under 5 sentences."

PROVIDERS = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "gemini": GeminiProvider,
}


async def main():
    provider = OpenAIProvider()
    messages: list[Message] = []
    total_cost = 0.0

    print(f"\n🤖 Chatbot started with {provider.name}")
    print("Commands: /switch <openai|anthropic|gemini>, /clear, /quit\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input == "/quit":
            print(f"\n💰 Total session cost: ${total_cost:.6f}")
            break

        if user_input == "/clear":
            messages = []
            print("📝 Conversation cleared.\n")
            continue

        if user_input.startswith("/switch"):
            parts = user_input.split()
            if len(parts) == 2 and parts[1] in PROVIDERS:
                provider = PROVIDERS[parts[1]]()
                print(f"✓ Switched to {provider.name}\n")
            else:
                print("Usage: /switch <openai|anthropic|gemini>\n")
            continue

        messages.append(Message(role="user", content=user_input))

        print(f"\n{provider.name}: ", end="", flush=True)
        full_response = ""

        async for chunk in provider.stream_chat(messages, system=SYSTEM_PROMPT):
            print(chunk, end="", flush=True)
            full_response += chunk

        print("\n")
        messages.append(Message(role="assistant", content=full_response))

        # Rough token estimate (you can use tiktoken for precision)
        approx_input = sum(len(m.content) for m in messages) // 4
        approx_output = len(full_response) // 4
        cost = provider.calculate_cost(approx_input, approx_output)
        total_cost += cost
        print(f"   💵 Approx cost: ${cost:.6f} | Session total: ${total_cost:.6f}\n")


if __name__ == "__main__":
    asyncio.run(main())
```

**`requirements.txt`**

```
openai>=1.0
anthropic>=0.40
google-generativeai>=0.8
python-dotenv>=1.0
pydantic>=2.0
tiktoken>=0.7
```

**`.env`**

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
```

**`.gitignore`**

```
.env
__pycache__/
*.pyc
venv/
.venv/
```

### What You Should Learn from This Project

- The **abstract base class pattern** — write code once, swap providers
- **Streaming async generators** — async for loops on chunks
- Differences between provider APIs (system prompt, message format, response shape)
- Real-world cost tracking

---

## Self-Check Questions

1. What three roles can a message have, and what does each do?
2. Why are LLMs called "stateless"?
3. What's the difference between `temperature=0` and `temperature=1`?
4. Why does Anthropic require `max_tokens` but OpenAI doesn't?
5. What's a context window and why does it matter?
6. Roughly, how many tokens is 1000 English words?
7. What's the main benefit of streaming responses?
8. If a user sends 500 tokens and the model returns 300 tokens, and pricing is $1/1M input and $3/1M output, what's the cost?
9. When would you choose Gemini over OpenAI?
10. Why learn raw APIs before using LangChain?

### Answers

1. `system` (instructions), `user` (human messages), `assistant` (LLM responses).
2. They have no memory — every call must include the full conversation history.
3. `0` is deterministic (same output every time). `1` is creative/varied.
4. Anthropic API design choice — they want you to be explicit about output limits.
5. Max tokens (input + output) the model can handle in one call. Hit it = error.
6. About 1300 tokens (1 token ≈ 0.75 words).
7. Better UX — user sees response progressively instead of waiting silently.
8. (500 / 1,000,000 × $1) + (300 / 1,000,000 × $3) = $0.0005 + $0.0009 = **$0.0014**.
9. Massive context window (reading whole books), cheapest, free tier for prototyping.
10. So you understand what frameworks do under the hood, can debug, and aren't locked in.

---

## Resources

### Official Docs
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)
- [Anthropic API Reference](https://docs.anthropic.com/en/api)
- [Gemini API Reference](https://ai.google.dev/gemini-api/docs)

### Tools
- [tiktoken](https://github.com/openai/tiktoken) — OpenAI's tokenizer
- [LiteLLM](https://github.com/BerriAI/litellm) — Unified interface (good to study after building your own)

### Recommended Reading
- OpenAI's "Prompt Engineering" guide
- Anthropic's "Building with Claude" docs

---

## Next Up

**Week 4:** Prompt Engineering — chain-of-thought, few-shot, ReAct, structured outputs.

You now have the plumbing. Next week we make the LLM actually behave well.
