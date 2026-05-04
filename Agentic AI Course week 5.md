# Week 5: Function Calling and Tool Use

> **Goal:** Make LLMs interact with the real world. Once you give an LLM tools (databases, APIs, files), it stops being a chatbot and starts being an agent.

---

## Table of Contents

1. [What Is Function Calling?](#what-is-function-calling)
2. [How Tool Use Works Under the Hood](#1-how-tool-use-works-under-the-hood)
3. [Tool Schemas with JSON Schema](#2-tool-schemas-with-json-schema)
4. [Your First Tool-Using LLM](#3-your-first-tool-using-llm)
5. [Multiple Tools](#4-multiple-tools)
6. [Forcing vs Letting the Model Decide](#5-forcing-vs-letting-the-model-decide)
7. [Parallel Tool Calls](#6-parallel-tool-calls)
8. [Handling Tool Errors](#7-handling-tool-errors)
9. [Writing Good Tool Descriptions](#8-writing-good-tool-descriptions)
10. [Multi-Turn Tool Use (Mini Agent)](#9-multi-turn-tool-use-mini-agent)
11. [Final Project](#final-project)
12. [Self-Check Questions](#self-check-questions)

---

## What Is Function Calling?

### The Simple Explanation

You give the LLM a list of "tools" it can use. Each tool is a Python function. When the LLM thinks a tool would help, it tells you which one to call and with what arguments. **You** call it. You give the result back to the LLM. The LLM uses that to answer.

### Important Truth

> **The LLM never actually executes code or makes API calls. It only DECIDES which tool to call. YOU execute it.**

This is critical to understand. The flow is:

```
1. User asks question
2. LLM says: "Call get_weather('London')"
3. YOUR CODE calls get_weather("London") → "12°C cloudy"
4. You send the result back to the LLM
5. LLM responds: "The weather in London is 12°C and cloudy"
```

### Why It Matters

This pattern unlocks everything in agentic AI:
- LLMs that query your database
- LLMs that send emails
- LLMs that book flights
- LLMs that write to files
- LLMs that call any API

It's the **foundation of every agent framework** (LangGraph, AutoGen, CrewAI, etc.).

### Names You'll Hear

- **Function calling** (OpenAI's term)
- **Tool use** (Anthropic's term)
- **Tool calling** (modern unified term)

They all mean the same thing.

---

## 1. How Tool Use Works Under the Hood

### The Conversation Flow

Here's what actually happens in messages:

**Turn 1: User asks**
```python
messages = [
    {"role": "user", "content": "What's the weather in London?"}
]
```

**Turn 2: LLM decides to use a tool**
```python
# LLM responds with a tool_call instead of text:
{
    "role": "assistant",
    "content": null,
    "tool_calls": [{
        "id": "call_abc123",
        "type": "function",
        "function": {
            "name": "get_weather",
            "arguments": '{"city": "London"}'
        }
    }]
}
```

**Turn 3: Your code runs the function**
```python
result = get_weather("London")  # Returns "12°C cloudy"
```

**Turn 4: You add the result to the conversation**
```python
messages.append({
    "role": "tool",
    "tool_call_id": "call_abc123",
    "content": "12°C cloudy"
})
```

**Turn 5: LLM uses result to answer**
```python
# LLM responds:
{"role": "assistant", "content": "The weather in London is 12°C and cloudy."}
```

### Key Insight

The LLM's "tool call" is just **text it generates** that looks like JSON. Your code parses it, runs the function, and sends the result back. The LLM never leaves the conversation.

---

## 2. Tool Schemas with JSON Schema

### What's a Schema?

A schema describes what a tool does and what arguments it takes. The LLM reads this schema to know what tools are available.

### The Schema Anatomy

```python
{
    "type": "function",
    "function": {
        "name": "get_weather",                      # Tool name
        "description": "Get current weather...",     # What it does
        "parameters": {                              # Arguments (JSON Schema format)
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name like 'London' or 'Mumbai'"
                },
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit"
                }
            },
            "required": ["city"]                     # Which args are mandatory
        }
    }
}
```

### JSON Schema Types You'll Use

| Type | Example | Notes |
|------|---------|-------|
| `string` | `"Hello"` | Use `enum` to restrict values |
| `integer` | `42` | Whole numbers only |
| `number` | `3.14` | Any number (int or float) |
| `boolean` | `true` | true/false |
| `array` | `["a", "b"]` | Specify `items` for inner type |
| `object` | `{"key": "value"}` | Nested object with `properties` |

### Example: Multiple Argument Types

```python
{
    "type": "function",
    "function": {
        "name": "search_products",
        "description": "Search the product catalog",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search keywords"
                },
                "max_price": {
                    "type": "number",
                    "description": "Max price in USD"
                },
                "in_stock_only": {
                    "type": "boolean",
                    "description": "Only show in-stock items",
                    "default": false
                },
                "categories": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter by these categories"
                }
            },
            "required": ["query"]
        }
    }
}
```

### Pro Tip: Use Pydantic to Generate Schemas

Writing JSON Schema by hand is painful. Use Pydantic instead:

```python
from pydantic import BaseModel, Field
from typing import Literal

class WeatherArgs(BaseModel):
    city: str = Field(..., description="City name like 'London'")
    units: Literal["celsius", "fahrenheit"] = "celsius"

# Auto-generates JSON Schema
print(WeatherArgs.model_json_schema())
```

---

## 3. Your First Tool-Using LLM

Let's build a simple example: an LLM that can check the weather.

### Step 1: Define the Tool (a Python Function)

```python
def get_weather(city: str, units: str = "celsius") -> str:
    """Pretend weather API."""
    fake_weather = {
        "London": "12°C, cloudy",
        "Mumbai": "32°C, humid",
        "Tokyo": "18°C, clear"
    }
    return fake_weather.get(city, "Weather data not available")
```

### Step 2: Define the Schema

```python
WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "Get the current weather for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name (e.g., 'London', 'Mumbai')"
                },
                "units": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "default": "celsius"
                }
            },
            "required": ["city"]
        }
    }
}
```

### Step 3: Wire It All Together

```python
import json
from openai import OpenAI

client = OpenAI()

def run_tool_call(name: str, args: dict) -> str:
    """Dispatch to the right Python function."""
    if name == "get_weather":
        return get_weather(**args)
    return f"Error: unknown tool {name}"


def chat_with_tools(user_message: str):
    messages = [{"role": "user", "content": user_message}]

    # First call - LLM decides if it needs a tool
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=[WEATHER_TOOL]
    )

    msg = response.choices[0].message

    # Did the LLM call a tool?
    if msg.tool_calls:
        # Add the assistant's message to history
        messages.append(msg)

        # Execute each tool call
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            result = run_tool_call(name, args)

            # Add tool result to history
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })

        # Second call - LLM uses tool results to answer
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        return response.choices[0].message.content
    else:
        # LLM didn't need any tool
        return msg.content


# Try it
print(chat_with_tools("What's the weather in London?"))
# Output: "The weather in London is 12°C and cloudy."

print(chat_with_tools("What is 2+2?"))
# Output: "2 + 2 equals 4." (no tool needed)
```

### What's Happening

1. We send the user message + tool definitions
2. LLM decides whether to use a tool
3. If yes, we run the Python function and send the result back
4. LLM gives the final answer using the tool result

---

## 4. Multiple Tools

Real agents have many tools. The LLM picks the right one (or several).

### Example: A Multi-Tool Assistant

```python
import json
from datetime import datetime
from openai import OpenAI

client = OpenAI()

# === Tool implementations ===

def get_weather(city: str) -> str:
    weather = {"London": "12°C cloudy", "Mumbai": "32°C humid"}
    return weather.get(city, "Unknown city")


def get_current_time(timezone: str = "UTC") -> str:
    return f"Current time in {timezone}: {datetime.now().isoformat()}"


def calculate(expression: str) -> str:
    try:
        # Use eval cautiously - in prod, use a safe math parser
        result = eval(expression, {"__builtins__": {}})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# === Tool schemas ===

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time",
            "parameters": {
                "type": "object",
                "properties": {
                    "timezone": {"type": "string", "default": "UTC"}
                }
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Evaluate a math expression like '2 + 2' or '15 * 7.5'",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"}
                },
                "required": ["expression"]
            }
        }
    }
]

# === Dispatcher ===

TOOL_FUNCTIONS = {
    "get_weather": get_weather,
    "get_current_time": get_current_time,
    "calculate": calculate
}


def execute_tool(name: str, args: dict) -> str:
    func = TOOL_FUNCTIONS.get(name)
    if not func:
        return f"Unknown tool: {name}"
    try:
        return str(func(**args))
    except Exception as e:
        return f"Error executing {name}: {e}"
```

### Test It

```python
# LLM picks weather tool
chat_with_tools("How's the weather in Mumbai?")

# LLM picks calculator
chat_with_tools("What is 247 multiplied by 89?")

# LLM picks time tool
chat_with_tools("What time is it now?")

# LLM might pick MULTIPLE tools (next section)
chat_with_tools("What time is it, and what's the weather in London?")
```

---

## 5. Forcing vs Letting the Model Decide

You have three options for how the model uses tools:

### Option 1: Auto (default) — LLM decides

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=TOOLS,
    tool_choice="auto"  # Default
)
```

LLM may or may not use a tool, based on the question.

### Option 2: Required — LLM must use SOME tool

```python
response = client.chat.completions.create(
    ...,
    tool_choice="required"
)
```

Useful when you know a tool is needed (e.g., extraction tasks).

### Option 3: Specific — Force a particular tool

```python
response = client.chat.completions.create(
    ...,
    tool_choice={
        "type": "function",
        "function": {"name": "get_weather"}
    }
)
```

Forces the LLM to call exactly this tool. Used for **structured extraction** (we saw this in Week 4 for JSON outputs).

### When to Use Each

| Mode | Use Case |
|------|----------|
| `auto` | General assistants, agents |
| `required` | When you guarantee a tool should be called |
| Specific tool | Structured outputs, single-purpose pipelines |

---

## 6. Parallel Tool Calls

### The Power Feature

Modern models can call **multiple tools in one response**. You execute them in parallel.

### Example

```python
# User asks something needing 3 tool calls
chat_with_tools(
    "Tell me the weather in London, Mumbai, and Tokyo"
)
```

The LLM might respond with **3 tool calls at once**:

```python
msg.tool_calls = [
    {"id": "call_1", "function": {"name": "get_weather", "arguments": '{"city": "London"}'}},
    {"id": "call_2", "function": {"name": "get_weather", "arguments": '{"city": "Mumbai"}'}},
    {"id": "call_3", "function": {"name": "get_weather", "arguments": '{"city": "Tokyo"}'}}
]
```

### Executing in Parallel (Async)

```python
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI()

# Make tools async
async def get_weather_async(city: str) -> str:
    await asyncio.sleep(1)  # Simulate API call
    weather = {"London": "12°C", "Mumbai": "32°C", "Tokyo": "18°C"}
    return weather.get(city, "Unknown")


async def execute_tool_async(name: str, args: dict) -> str:
    if name == "get_weather":
        return await get_weather_async(**args)
    return "Unknown tool"


async def chat_with_parallel_tools(user_message: str):
    messages = [{"role": "user", "content": user_message}]

    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        tools=TOOLS
    )

    msg = response.choices[0].message

    if msg.tool_calls:
        messages.append(msg)

        # Execute ALL tool calls in parallel
        async def run(tc):
            args = json.loads(tc.function.arguments)
            result = await execute_tool_async(tc.function.name, args)
            return tc.id, result

        results = await asyncio.gather(*[run(tc) for tc in msg.tool_calls])

        # Add all results to messages
        for tool_call_id, result in results:
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": result
            })

        # Final response
        final = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages
        )
        return final.choices[0].message.content
    else:
        return msg.content
```

### Why It Matters

If you have 3 weather calls @ 1 second each:
- **Sequential:** 3 seconds
- **Parallel:** 1 second

For agents that make many API calls, this is a 5–10x speedup.

---

## 7. Handling Tool Errors

### LLMs Are Robust to Errors — If You Tell Them About Them

If your tool fails, **don't crash**. Send the error message back to the LLM. It will often recover gracefully.

### Bad: Crash on Error

```python
def get_weather(city: str) -> str:
    response = httpx.get(f"https://api.weather.com/{city}")
    return response.json()["temp"]  # Crashes if city not found!
```

### Good: Return Error as String

```python
def get_weather(city: str) -> str:
    try:
        response = httpx.get(f"https://api.weather.com/{city}")
        response.raise_for_status()
        return response.json()["temp"]
    except httpx.HTTPStatusError:
        return f"Error: City '{city}' not found. Try a different city name."
    except Exception as e:
        return f"Error: weather service unavailable ({e})"
```

### What the LLM Does With Errors

When the LLM receives `"Error: City 'Lndn' not found. Try a different city name."`, it often realizes the typo and retries:

```
Tool call: get_weather(city="Lndn")
Result: Error: City 'Lndn' not found. Try a different city name.

Thought: I had a typo. Let me try "London".
Tool call: get_weather(city="London")
Result: 12°C cloudy
```

### Error Handling Pattern

```python
def safe_tool_execution(func, args):
    try:
        result = func(**args)
        return {"status": "success", "data": result}
    except ValueError as e:
        return {"status": "error", "type": "validation", "message": str(e)}
    except TimeoutError:
        return {"status": "error", "type": "timeout", "message": "Tool timed out"}
    except Exception as e:
        return {"status": "error", "type": "unknown", "message": str(e)}
```

### Tip: Set Max Retries

To prevent infinite loops, limit how many times an agent can call tools:

```python
MAX_TOOL_CALLS = 10

call_count = 0
while msg.tool_calls and call_count < MAX_TOOL_CALLS:
    # ... process tool calls ...
    call_count += 1

if call_count >= MAX_TOOL_CALLS:
    return "I've tried multiple approaches but couldn't complete the task."
```

---

## 8. Writing Good Tool Descriptions

### Why Descriptions Matter

The LLM **only knows what your description tells it**. A bad description = wrong tool selection.

### Bad Description

```python
{
    "name": "search",
    "description": "Search stuff",
    "parameters": {
        "properties": {
            "q": {"type": "string"}
        }
    }
}
```

The LLM has no idea what this searches, what to put in `q`, or when to use it.

### Good Description

```python
{
    "name": "search_products",
    "description": (
        "Search our e-commerce product catalog by keywords. "
        "Use this when the user asks about products, prices, or availability. "
        "Do NOT use for FAQs or shipping questions — use search_help_articles instead."
    ),
    "parameters": {
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Search keywords. Use product-related terms like "
                    "'red running shoes' or 'wireless headphones'. "
                    "Don't include filters here — use the filter parameters."
                )
            },
            "max_results": {
                "type": "integer",
                "description": "Number of results (1-50). Default 10.",
                "default": 10
            }
        },
        "required": ["query"]
    }
}
```

### Description Best Practices

1. **Say what it does** in plain language
2. **Say when to use it** (positive instructions)
3. **Say when NOT to use it** (especially if multiple tools overlap)
4. **Give example arguments** in the description
5. **Describe the return format**

### Real-World Example

```python
{
    "name": "query_database",
    "description": (
        "Run a SQL SELECT query against the user_events database. "
        "Returns up to 1000 rows as JSON.\n\n"
        "USE THIS WHEN: User asks about user behavior, event counts, "
        "trends over time, or specific user activity.\n\n"
        "DO NOT USE FOR: aggregations spanning millions of rows "
        "(use get_metrics_dashboard instead).\n\n"
        "Example: SELECT COUNT(*) FROM events WHERE event_type = 'click' "
        "AND created_at > '2025-01-01'"
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "sql": {
                "type": "string",
                "description": "Valid PostgreSQL SELECT query. No INSERT/UPDATE/DELETE."
            }
        },
        "required": ["sql"]
    }
}
```

### Pro Tip: Test Descriptions Empirically

Try a few prompts and see if the LLM picks the right tool. If not, **fix the description, not the prompt.**

---

## 9. Multi-Turn Tool Use (Mini Agent)

### The Real Pattern

Agents don't make one tool call. They:
1. Call a tool
2. Read the result
3. Decide what to do next
4. Maybe call another tool
5. Eventually answer

This is the ReAct loop you saw in Week 4.

### Implementation

```python
import json
from openai import OpenAI

client = OpenAI()


def run_agent(user_message: str, max_iterations: int = 10):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant. Use tools when needed. "
                "After getting tool results, decide if you need more tools "
                "or if you can answer the user."
            )
        },
        {"role": "user", "content": user_message}
    ]

    for iteration in range(max_iterations):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS
        )

        msg = response.choices[0].message
        messages.append(msg)

        # No tool calls = LLM is done
        if not msg.tool_calls:
            return msg.content

        # Execute each tool call
        for tool_call in msg.tool_calls:
            name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)
            print(f"🔧 Calling {name}({args})")

            result = execute_tool(name, args)
            print(f"   Result: {result}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": str(result)
            })

    return "Max iterations reached."


# Test it
result = run_agent(
    "What's the weather in cities where I should go on vacation? "
    "I want somewhere warm. Try London, Mumbai, and Tokyo."
)
print(result)
```

### What This Does

```
🔧 Calling get_weather({'city': 'London'})
   Result: 12°C cloudy

🔧 Calling get_weather({'city': 'Mumbai'})
   Result: 32°C humid

🔧 Calling get_weather({'city': 'Tokyo'})
   Result: 18°C clear

Final answer: For a warm vacation, Mumbai is ideal at 32°C.
Tokyo is mild at 18°C. London is too cold at 12°C.
```

### Congrats — You've Built a Mini Agent

This loop is the foundation of all agent frameworks. Everything else is:
- Better state management
- Memory across sessions
- More tools
- Multi-agent coordination

---

## Final Project

### Project: Personal Productivity Agent

Build an agent with **5 tools** that helps with daily tasks. The agent should pick tools intelligently and chain them when needed.

### Required Tools

1. **`get_current_time`** — Returns current date/time
2. **`search_files`** — Search for files in a folder by name pattern
3. **`read_file`** — Read contents of a file (limit to 5000 chars)
4. **`web_search`** — Search the web (use a free API like Tavily or DuckDuckGo)
5. **`calculate`** — Safe math evaluator

### Requirements

1. All 5 tools implemented and well-described
2. Async with parallel tool calls
3. Error handling for every tool
4. Max iterations limit (prevent infinite loops)
5. Verbose mode showing every tool call
6. Track total cost per session

### Project Structure

```
week-05-tool-use/
├── README.md
├── requirements.txt
├── .env
├── .gitignore
├── agent.py                # Main agent loop
├── tools/
│   ├── __init__.py
│   ├── base.py             # Tool registry
│   ├── time_tool.py
│   ├── file_tools.py
│   ├── web_search.py
│   └── calculator.py
├── schemas.py              # JSON schemas for all tools
└── notes.md
```

### Starter Code

**`tools/base.py`**

```python
from typing import Any, Callable, Awaitable
from pydantic import BaseModel


class Tool(BaseModel):
    """Wraps a tool function with its schema."""
    name: str
    description: str
    parameters: dict
    function: Callable[..., Awaitable[str]]

    class Config:
        arbitrary_types_allowed = True

    def to_openai_schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters
            }
        }


class ToolRegistry:
    """Holds all tools and dispatches calls."""

    def __init__(self):
        self.tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self.tools[tool.name] = tool

    def get_schemas(self) -> list[dict]:
        return [t.to_openai_schema() for t in self.tools.values()]

    async def execute(self, name: str, args: dict) -> str:
        tool = self.tools.get(name)
        if not tool:
            return f"Error: tool '{name}' not found"
        try:
            return await tool.function(**args)
        except TypeError as e:
            return f"Error: invalid arguments to {name}: {e}"
        except Exception as e:
            return f"Error executing {name}: {type(e).__name__}: {e}"
```

**`tools/time_tool.py`**

```python
from datetime import datetime
from zoneinfo import ZoneInfo
from .base import Tool


async def get_current_time(timezone: str = "UTC") -> str:
    try:
        tz = ZoneInfo(timezone)
        now = datetime.now(tz)
        return now.strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return f"Error: unknown timezone '{timezone}'. Use names like 'UTC', 'America/New_York', 'Asia/Kolkata'."


TIME_TOOL = Tool(
    name="get_current_time",
    description=(
        "Get the current date and time in a specific timezone. "
        "USE WHEN: User asks about current time, today's date, or scheduling. "
        "Default timezone is UTC if not specified."
    ),
    parameters={
        "type": "object",
        "properties": {
            "timezone": {
                "type": "string",
                "description": (
                    "IANA timezone name like 'UTC', 'America/New_York', "
                    "'Europe/London', 'Asia/Kolkata'. Default: UTC"
                )
            }
        }
    },
    function=get_current_time
)
```

**`tools/calculator.py`**

```python
import ast
import operator
from .base import Tool


# Safe operators
OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.USub: operator.neg
}


def _safe_eval(node):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.BinOp):
        return OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        return OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


async def calculate(expression: str) -> str:
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        return str(result)
    except Exception as e:
        return f"Error: invalid math expression '{expression}': {e}"


CALCULATE_TOOL = Tool(
    name="calculate",
    description=(
        "Evaluate a math expression. Supports +, -, *, /, **, %. "
        "USE WHEN: User asks to compute numbers, percentages, or simple math. "
        "Examples: '247 * 89', '1500 * 0.18', '(100 + 50) / 2'"
    ),
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Math expression like '2 + 2' or '15 * 7.5'"
            }
        },
        "required": ["expression"]
    },
    function=calculate
)
```

**`tools/file_tools.py`**

```python
from pathlib import Path
from .base import Tool


async def search_files(folder: str, pattern: str = "*") -> str:
    try:
        path = Path(folder).expanduser()
        if not path.exists():
            return f"Error: folder '{folder}' does not exist"
        if not path.is_dir():
            return f"Error: '{folder}' is not a folder"

        matches = list(path.rglob(pattern))[:50]  # Limit to 50 results
        if not matches:
            return f"No files matching '{pattern}' in {folder}"

        return "\n".join(str(m) for m in matches)
    except Exception as e:
        return f"Error searching files: {e}"


async def read_file(path: str, max_chars: int = 5000) -> str:
    try:
        file_path = Path(path).expanduser()
        if not file_path.exists():
            return f"Error: file '{path}' does not exist"
        if not file_path.is_file():
            return f"Error: '{path}' is not a file"

        content = file_path.read_text(errors="replace")
        if len(content) > max_chars:
            content = content[:max_chars] + f"\n\n[... truncated, file has {len(content)} total chars]"
        return content
    except Exception as e:
        return f"Error reading file: {e}"


SEARCH_FILES_TOOL = Tool(
    name="search_files",
    description=(
        "Search for files in a folder using a glob pattern. "
        "USE WHEN: User wants to find files by name. "
        "Examples: search '*.py' in '~/projects' to find Python files."
    ),
    parameters={
        "type": "object",
        "properties": {
            "folder": {
                "type": "string",
                "description": "Folder path (supports ~)"
            },
            "pattern": {
                "type": "string",
                "description": "Glob pattern like '*.txt' or '*.py'. Default: '*' (all)"
            }
        },
        "required": ["folder"]
    },
    function=search_files
)

READ_FILE_TOOL = Tool(
    name="read_file",
    description=(
        "Read the contents of a text file (up to 5000 chars). "
        "USE WHEN: User asks about a file's content, after using search_files."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Full path to the file"
            }
        },
        "required": ["path"]
    },
    function=read_file
)
```

**`tools/web_search.py`**

```python
import os
import httpx
from .base import Tool


async def web_search(query: str, max_results: int = 5) -> str:
    """Uses Tavily (free tier). Sign up at tavily.com to get an API key."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY not set"

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                "https://api.tavily.com/search",
                json={
                    "api_key": api_key,
                    "query": query,
                    "max_results": max_results
                }
            )
            response.raise_for_status()
            data = response.json()

            if not data.get("results"):
                return f"No web results for '{query}'"

            output = []
            for r in data["results"]:
                output.append(f"- {r['title']}\n  {r['url']}\n  {r['content'][:200]}...")
            return "\n\n".join(output)
    except Exception as e:
        return f"Error searching web: {e}"


WEB_SEARCH_TOOL = Tool(
    name="web_search",
    description=(
        "Search the web for current information. "
        "USE WHEN: User asks about current events, recent news, or facts you don't know. "
        "DO NOT USE for: math, file operations, or general knowledge questions."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query"
            },
            "max_results": {
                "type": "integer",
                "description": "Number of results (1-10). Default 5",
                "default": 5
            }
        },
        "required": ["query"]
    },
    function=web_search
)
```

**`agent.py`**

```python
import asyncio
import json
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

from tools.base import ToolRegistry
from tools.time_tool import TIME_TOOL
from tools.calculator import CALCULATE_TOOL
from tools.file_tools import SEARCH_FILES_TOOL, READ_FILE_TOOL
from tools.web_search import WEB_SEARCH_TOOL

load_dotenv()

client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Build registry
registry = ToolRegistry()
for tool in [TIME_TOOL, CALCULATE_TOOL, SEARCH_FILES_TOOL, READ_FILE_TOOL, WEB_SEARCH_TOOL]:
    registry.register(tool)


SYSTEM_PROMPT = """
You are a helpful productivity assistant with access to tools.

Guidelines:
- Use tools when needed for facts, math, files, or current info
- For unknown information, prefer web_search over guessing
- Combine tools when needed (e.g., search files, then read one)
- After using tools, give a concise final answer
"""


async def run_agent(user_message: str, max_iterations: int = 10, verbose: bool = True):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message}
    ]

    total_input_tokens = 0
    total_output_tokens = 0

    for iteration in range(max_iterations):
        if verbose:
            print(f"\n--- Iteration {iteration + 1} ---")

        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=registry.get_schemas()
        )

        total_input_tokens += response.usage.prompt_tokens
        total_output_tokens += response.usage.completion_tokens

        msg = response.choices[0].message
        messages.append(msg)

        # Done?
        if not msg.tool_calls:
            if verbose:
                cost = (total_input_tokens / 1_000_000) * 0.15 + \
                       (total_output_tokens / 1_000_000) * 0.60
                print(f"\n💵 Tokens: {total_input_tokens} in / {total_output_tokens} out")
                print(f"💵 Cost: ${cost:.6f}")
            return msg.content

        # Execute tools in parallel
        if verbose:
            for tc in msg.tool_calls:
                print(f"🔧 {tc.function.name}({tc.function.arguments})")

        async def run_one(tc):
            args = json.loads(tc.function.arguments)
            result = await registry.execute(tc.function.name, args)
            return tc.id, result

        results = await asyncio.gather(*[run_one(tc) for tc in msg.tool_calls])

        for tool_call_id, result in results:
            if verbose:
                snippet = result[:200] + "..." if len(result) > 200 else result
                print(f"   → {snippet}")

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call_id,
                "content": result
            })

    return "Max iterations reached."


async def main():
    print("🤖 Productivity Agent (type 'quit' to exit)\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            break
        if not user_input:
            continue

        try:
            response = await run_agent(user_input)
            print(f"\nAgent: {response}\n")
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
```

**`requirements.txt`**

```
openai>=1.0
pydantic>=2.0
python-dotenv>=1.0
httpx>=0.25
```

**`.env`**

```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

### Test Cases to Try

```
> What time is it in Mumbai?
> Find all Python files in ~/projects
> What's 1500 multiplied by 0.18 (18% tax)?
> Search the web for the latest AI news
> Find Python files in ~/projects, then read the first one
> What time is it, and what's 25 * 4?  (parallel calls)
```

### What You Should Learn from This Project

- **Tool registration pattern** — clean separation of tool definition and execution
- **Parallel tool execution** with `asyncio.gather`
- **Error handling** at the tool level
- **Iteration loop** with max-iteration safety
- **Cost tracking** in production agents

### Stretch Goals

- Add a 6th tool of your choice (e.g., send email, query a database)
- Add streaming so partial responses show as the LLM types
- Persist conversation history to a JSON file
- Show a tree of tool calls (for visualizing complex agent runs)

---

## Self-Check Questions

1. Who actually executes the tool — the LLM or your code?
2. What 4 fields make up a tool schema?
3. What does `tool_choice="required"` do?
4. How do you handle a tool that throws an exception?
5. Why does writing good tool descriptions matter?
6. What's the role of the `tool_call_id` in the messages list?
7. What's the difference between sequential and parallel tool calls?
8. How do you prevent an agent from running forever?
9. What happens if your tool description overlaps with another tool's?
10. When would you use `tool_choice="auto"` vs forcing a specific tool?

### Answers

1. Your code. The LLM only generates the request; you parse it and execute.
2. `name`, `description`, `parameters` (JSON Schema), and the function itself.
3. Forces the LLM to call SOME tool (it can't reply with plain text).
4. Catch the exception and return the error as a string. The LLM can recover from text errors but not from crashes.
5. The LLM uses descriptions to decide which tool to use. Bad description = wrong tool selection.
6. It links a tool result to the specific call that requested it (essential for parallel calls).
7. Sequential = one at a time (slow). Parallel = all at once with `asyncio.gather` (fast).
8. Set a `max_iterations` limit on the loop.
9. The LLM may pick the wrong one. Disambiguate with explicit "USE WHEN" / "DO NOT USE" language.
10. Auto for general assistants. Force a specific tool for structured extraction or pipelines.

---

## Resources

### Must-Read
- [OpenAI Function Calling Guide](https://platform.openai.com/docs/guides/function-calling)
- [Anthropic Tool Use Docs](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview)
- [JSON Schema Reference](https://json-schema.org/learn/getting-started-step-by-step)

### Tools to Try
- [Tavily](https://tavily.com/) — Free web search API for agents
- [Composio](https://composio.dev/) — Pre-built tools for hundreds of apps
- [E2B](https://e2b.dev/) — Sandboxed code execution for agents

### Recommended Reading
- "Toolformer" paper (Meta, 2023) — How LLMs learn to use tools
- Anthropic's "Computer Use" announcement — Tool use applied to UI

---

## Next Up

**Week 6:** Embeddings and Vector Search — the foundation of RAG. We'll cover what embeddings actually are, how to use vector databases, and how to build semantic search.

Now that your LLM can act, next week we'll teach it to **remember and retrieve** — turning it into a system that knows your data.
