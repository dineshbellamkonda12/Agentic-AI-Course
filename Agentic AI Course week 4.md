# Week 4: Prompt Engineering

> **Goal:** Learn how to make LLMs reliably do what you want. Prompt engineering is the difference between an AI app that works 60% of the time and one that works 95% of the time.

---

## Table of Contents

1. [What Prompt Engineering Really Is](#what-prompt-engineering-really-is)
2. [The Anatomy of a Good Prompt](#1-the-anatomy-of-a-good-prompt)
3. [Zero-Shot Prompting](#2-zero-shot-prompting)
4. [Few-Shot Prompting](#3-few-shot-prompting)
5. [Chain-of-Thought (CoT)](#4-chain-of-thought-cot)
6. [ReAct Pattern](#5-react-pattern)
7. [Structured Outputs](#6-structured-outputs)
8. [Prompt Templates and Versioning](#7-prompt-templates-and-versioning)
9. [Reflection and Self-Critique](#8-reflection-and-self-critique)
10. [Common Mistakes to Avoid](#9-common-mistakes-to-avoid)
11. [Final Project](#final-project)
12. [Self-Check Questions](#self-check-questions)

---

## What Prompt Engineering Really Is

### The Honest Definition

Prompt engineering is **writing instructions for the LLM clearly enough that it does what you want, consistently.**

It's not magic. It's not a "secret prompt" that unlocks superpowers. It's careful, deliberate communication.

### Why It Matters

The same LLM with two different prompts can give you:

| Bad prompt | Good prompt |
|-----------|-------------|
| 60% accuracy | 95% accuracy |
| Inconsistent format | Reliable JSON |
| Hallucinations | Cited facts |
| 500 tokens wasted | 50 tokens used |

This week you learn the techniques that move you from one to the other.

### The Core Principle

> **LLMs are like brilliant interns: smart, eager to please, but they need very clear instructions and examples.**

Treat them that way. Give clear context, examples, and format requirements.

---

## 1. The Anatomy of a Good Prompt

A great prompt usually has 5 parts:

```
1. ROLE       — Who is the LLM?
2. TASK       — What should it do?
3. CONTEXT    — Background info it needs
4. FORMAT     — How should the output look?
5. EXAMPLES   — (Optional) Show what good looks like
```

### Bad Example

```
Summarize this.
```

### Good Example

```
You are a technical writer for engineering blogs.

TASK: Summarize the article below for senior software engineers.

CONTEXT: The audience is technical, so use precise terminology.
They will skim this in 30 seconds — make every word count.

FORMAT:
- 3 bullet points
- Each bullet starts with a verb (e.g., "Reduces", "Enables")
- Maximum 15 words per bullet
- No marketing language

ARTICLE:
{article_text}
```

### Practice: Improve This Prompt

**Original:**
```
Help me write code.
```

**Improved version (you write):**
```
You are a senior Python developer with 10 years of experience.

TASK: Write a function that reads a CSV file and returns
the rows where the "status" column equals "active".

REQUIREMENTS:
- Use type hints
- Handle file-not-found errors
- Return an empty list if no rows match
- Add a docstring

Provide only the code with no explanation.
```

---

## 2. Zero-Shot Prompting

### What Is Zero-Shot?

You ask the LLM to do a task **without giving any examples**. It uses general knowledge to figure it out.

### Example

```python
prompt = """
Classify this email as either "spam" or "not_spam".

Email: "Congratulations! You won $1,000,000! Click here NOW!"

Classification:
"""
```

The LLM responds: `spam`

### When to Use Zero-Shot

- Simple tasks (classification, basic Q&A, translation)
- When your task is well-known (the LLM has seen it in training)
- When you want minimal prompt size

### When NOT to Use Zero-Shot

- Custom formats the LLM hasn't seen
- Domain-specific tasks
- When consistency matters

### Tip: Be Specific About Output Format

**Bad zero-shot:**
```
Is this email spam?
```

**Good zero-shot:**
```
Classify this email as exactly one of: "spam" or "not_spam".
Respond with only the label, no other text.

Email: ...
```

---

## 3. Few-Shot Prompting

### What Is Few-Shot?

Give the LLM **2-5 examples** of input → output pairs. It learns the pattern from examples.

### Why It Works

LLMs are **pattern-matching machines**. Show them the pattern, they replicate it.

### Example: Sentiment with Custom Categories

Suppose you want a non-standard sentiment label set: `frustrated`, `confused`, `satisfied`, `excited`. Few-shot is perfect.

```python
prompt = """
Classify the customer feedback into one of:
frustrated, confused, satisfied, excited.

Examples:

Feedback: "I can't figure out how this works. The docs are unclear."
Label: confused

Feedback: "This is exactly what I needed! Saved me hours."
Label: satisfied

Feedback: "I've tried 3 times and it still doesn't work. This is ridiculous."
Label: frustrated

Feedback: "Wow, the new update is amazing! Can't wait to try more features!"
Label: excited

Now classify:

Feedback: "I followed the steps but I'm not sure what's supposed to happen next."
Label:
"""
```

LLM responds: `confused` ✓

### Best Practices for Few-Shot

1. **Cover edge cases** — Include the tricky examples, not just easy ones
2. **Keep examples diverse** — Avoid repetition
3. **Use 3-5 examples** — More usually doesn't help and costs more tokens
4. **Format examples consistently** — Same structure every time
5. **Order matters** — LLMs sometimes weight later examples more

### Few-Shot for Data Extraction

```python
prompt = """
Extract structured data from messy customer support emails.

Example 1:
Input: "Hi, my order #12345 hasn't arrived. Ordered last Tuesday. Email: john@x.com"
Output: {"order_id": "12345", "issue": "delivery_delay", "email": "john@x.com"}

Example 2:
Input: "I want a refund for order ABC-789. The product is broken."
Output: {"order_id": "ABC-789", "issue": "refund_request", "email": null}

Now extract:
Input: "Order 555 came damaged. Please help. Reach me at sara@test.org."
Output:
"""
```

---

## 4. Chain-of-Thought (CoT)

### What Is Chain-of-Thought?

You ask the LLM to **think step-by-step** before answering. This dramatically improves performance on reasoning, math, and logic tasks.

### Why It Works

LLMs generate one token at a time. If they have to "show work," each reasoning step becomes input for the next step. They reason their way to better answers.

### Without CoT (Often Wrong)

```
Q: A store sold 23 apples on Monday, twice as many on Tuesday,
and 15 fewer on Wednesday than Tuesday. How many total?

A: 84
```

(Actually wrong — let's check: 23 + 46 + 31 = 100)

### With CoT (More Accurate)

```
Q: A store sold 23 apples on Monday, twice as many on Tuesday,
and 15 fewer on Wednesday than Tuesday. How many total?

Let's think step by step:
- Monday: 23 apples
- Tuesday: twice Monday = 23 × 2 = 46 apples
- Wednesday: 15 fewer than Tuesday = 46 - 15 = 31 apples
- Total: 23 + 46 + 31 = 100 apples

A: 100 apples
```

### Two Ways to Trigger CoT

**Method 1: Magic phrase (zero-shot CoT)**

```python
prompt = """
Q: {question}

Let's think step by step.
"""
```

The phrase **"Let's think step by step"** is famous for unlocking reasoning.

**Method 2: Show CoT examples (few-shot CoT)**

```python
prompt = """
Q: There are 5 cars. 2 more arrive. How many cars now?
A: Start with 5 cars. 2 more arrive: 5 + 2 = 7. Answer: 7.

Q: A box has 12 apples. 3 are taken. How many remain?
A: Start with 12. Remove 3: 12 - 3 = 9. Answer: 9.

Q: {your_question}
A:
"""
```

### When to Use CoT

✅ Use it for:
- Math problems
- Logic puzzles
- Multi-step reasoning
- Code debugging
- Complex decisions

❌ Don't use it for:
- Simple lookups
- Format conversions
- Tasks where speed matters more than accuracy
- (Modern reasoning models like o1 already do CoT internally)

---

## 5. ReAct Pattern

### What Is ReAct?

ReAct = **Reasoning + Acting**

The LLM alternates between:
- **Thought** — what to do next
- **Action** — call a tool
- **Observation** — see the result
- (repeat)

This is the foundation of agents.

### The ReAct Loop

```
Thought: I need to find the current weather in London.
Action: search_web("current weather London")
Observation: 12°C, cloudy, light rain.

Thought: Now I have the weather. Let me answer.
Final Answer: It's 12°C and cloudy in London with light rain.
```

### A Simple Example

```python
prompt = """
You are a research assistant. Answer questions by reasoning step-by-step
and using these tools when needed:

- search_web(query): returns search results
- calculator(expression): evaluates math
- finish(answer): provides the final answer

Use this exact format:
Thought: <your reasoning>
Action: <tool_name>(<arguments>)
Observation: <will be filled in for you>

Repeat until you can answer.

Question: What is the population of Tokyo divided by the population of London?

"""
```

LLM might respond:
```
Thought: I need both populations. Let me search for Tokyo first.
Action: search_web("Tokyo population 2025")
Observation: Tokyo has approximately 13.96 million people.

Thought: Now I need London's population.
Action: search_web("London population 2025")
Observation: London has approximately 9.0 million people.

Thought: Now divide them.
Action: calculator(13960000 / 9000000)
Observation: 1.551

Action: finish("Tokyo's population is about 1.55x larger than London's.")
```

### Why It Matters

This is **how agents work under the hood**. LangGraph, AutoGPT, and every agent framework is essentially a ReAct loop with extra structure.

We'll go deeper in Month 3, but understand the pattern now.

---

## 6. Structured Outputs

### Why Structured Outputs?

In production, you rarely want freeform text. You want **JSON you can parse**.

```python
# ❌ Bad - have to parse free text
response = "The sentiment is positive with 90% confidence"

# ✅ Good - directly usable
response = {"sentiment": "positive", "confidence": 0.9}
```

### Three Ways to Get Structured Outputs

#### Method 1: JSON Mode (OpenAI, Anthropic)

Tell the model to respond ONLY in JSON.

```python
response = client.chat.completions.create(
    model="gpt-4o-mini",
    response_format={"type": "json_object"},  # ← Forces JSON output
    messages=[
        {
            "role": "system",
            "content": "Respond only with JSON in this format: "
                       '{"sentiment": "positive|negative|neutral", "confidence": 0.0-1.0}'
        },
        {"role": "user", "content": "I love this product!"}
    ]
)

import json
result = json.loads(response.choices[0].message.content)
print(result["sentiment"])  # "positive"
```

#### Method 2: Function Calling / Tool Use (Best Method)

Define a "tool" with a schema, the LLM fills it in.

```python
from openai import OpenAI

client = OpenAI()

tools = [{
    "type": "function",
    "function": {
        "name": "report_sentiment",
        "description": "Report the sentiment of the text",
        "parameters": {
            "type": "object",
            "properties": {
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral"]
                },
                "confidence": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1
                },
                "key_phrases": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            },
            "required": ["sentiment", "confidence", "key_phrases"]
        }
    }
}]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "I love this product!"}],
    tools=tools,
    tool_choice={"type": "function", "function": {"name": "report_sentiment"}}
)

import json
args = json.loads(response.choices[0].message.tool_calls[0].function.arguments)
print(args)
# {"sentiment": "positive", "confidence": 0.95, "key_phrases": ["love"]}
```

#### Method 3: Pydantic + Instructor (Cleanest)

The `instructor` library combines Pydantic with function calling for type-safe outputs.

```bash
pip install instructor
```

```python
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Literal

class SentimentAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0, le=1)
    key_phrases: list[str]
    reasoning: str

client = instructor.from_openai(OpenAI())

result = client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=SentimentAnalysis,  # ← Pydantic model
    messages=[{"role": "user", "content": "Analyze: I love this product!"}]
)

print(result.sentiment)        # "positive" (typed!)
print(result.confidence)       # 0.95
print(result.key_phrases)      # ["love"]
print(type(result))            # SentimentAnalysis
```

This is **my recommended method** — clean, type-safe, validated automatically.

### Real Example: Extract Resume Info

```python
from pydantic import BaseModel
from typing import Optional

class WorkExperience(BaseModel):
    company: str
    role: str
    start_year: int
    end_year: Optional[int]  # None if current job
    achievements: list[str]

class Resume(BaseModel):
    name: str
    email: str
    skills: list[str]
    experiences: list[WorkExperience]

resume_text = """
Dinesh Kumar
dinesh@example.com

Skills: Python, Django, AWS, PostgreSQL

Experience:
- OpenSolar UK, Software Developer, 2023 - present
  * Built imagery integration system
  * Improved API response time by 40%

- TechCorp, Junior Dev, 2021 - 2023
  * Developed REST APIs
  * Mentored 2 interns
"""

resume = client.chat.completions.create(
    model="gpt-4o-mini",
    response_model=Resume,
    messages=[
        {"role": "system", "content": "Extract structured resume info."},
        {"role": "user", "content": resume_text}
    ]
)

print(resume.experiences[0].company)  # "OpenSolar UK"
```

---

## 7. Prompt Templates and Versioning

### Why Templates?

In real apps, prompts have variable parts (user input, retrieved context, etc.). Templates separate **structure** from **content**.

### Simple Template

```python
PROMPT_TEMPLATE = """
You are a customer support agent for {company_name}.

Respond to this customer message:
{customer_message}

Use a {tone} tone. Keep response under {max_words} words.
"""

filled = PROMPT_TEMPLATE.format(
    company_name="OpenSolar",
    customer_message="My quote isn't loading",
    tone="friendly",
    max_words=50
)
```

### Better: Use a Class

```python
from pydantic import BaseModel
from string import Template

class PromptTemplate(BaseModel):
    name: str
    version: str
    template: str

    def format(self, **kwargs) -> str:
        return Template(self.template).safe_substitute(**kwargs)


SUPPORT_PROMPT_V1 = PromptTemplate(
    name="customer_support",
    version="1.0",
    template="""
You are a support agent for $company.
Respond to: $message
Tone: $tone
""".strip()
)

prompt = SUPPORT_PROMPT_V1.format(
    company="OpenSolar",
    message="Help",
    tone="friendly"
)
```

### Prompt Versioning

When you change a prompt in production, **track the version**. This is critical for evals (we'll cover in Month 4).

```python
# prompts/customer_support_v1.txt
# prompts/customer_support_v2.txt
# prompts/customer_support_v3.txt

# Or store in code with metadata:
PROMPTS = {
    "support_v1": {
        "version": "1.0",
        "created": "2025-01-15",
        "template": "...",
        "notes": "Initial version"
    },
    "support_v2": {
        "version": "2.0",
        "created": "2025-02-01",
        "template": "...",
        "notes": "Added refund policy mention. +12% satisfaction."
    }
}
```

### Production Tip

Store prompts in **separate files** (not buried in code). Makes them easier to:
- Edit without touching code
- Version with git
- A/B test
- Hand off to non-developers

```
prompts/
├── customer_support_v1.md
├── code_reviewer_v1.md
└── summarizer_v3.md
```

---

## 8. Reflection and Self-Critique

### What Is Reflection?

Ask the LLM to **review its own output** and improve it.

### Pattern

```
Step 1: Generate answer
Step 2: Critique the answer
Step 3: Improve based on critique
```

### Example

```python
# Step 1: Initial answer
initial_response = ask_llm(f"Write a function to {task}")

# Step 2: Self-critique
critique = ask_llm(f"""
Review this code for bugs, edge cases, and improvements:

{initial_response}

Be brutally honest. List specific issues.
""")

# Step 3: Improved answer
final_response = ask_llm(f"""
Original code:
{initial_response}

Critique:
{critique}

Rewrite the code addressing every issue in the critique.
""")
```

### When It Helps

- Code generation (catches bugs)
- Long-form writing (improves quality)
- Complex reasoning (catches errors)

### When It Hurts

- Simple tasks (overkill)
- Time-sensitive apps (3x slower)
- Cost-sensitive apps (3x more expensive)

### Reflexion Pattern

A more advanced version: the LLM **remembers past mistakes** and avoids them next time. We'll cover this in Month 3.

---

## 9. Common Mistakes to Avoid

### ❌ Mistake 1: Vague Instructions

```
Bad:  "Write a good summary"
Good: "Write a 3-sentence summary focusing on the technical impact"
```

### ❌ Mistake 2: No Output Format

```
Bad:  "Extract the entities"
Good: "Extract entities and return as JSON: {persons: [], orgs: [], dates: []}"
```

### ❌ Mistake 3: Negative-Only Instructions

```
Bad:  "Don't be too long. Don't use jargon. Don't be casual."
Good: "Write 3 short bullets in formal but accessible language."
```

LLMs handle positive instructions better than negative ones.

### ❌ Mistake 4: Mixing Languages

If your system prompt is in English but examples are in Hindi, the LLM gets confused. Stay consistent.

### ❌ Mistake 5: Too Much in One Prompt

```
Bad: "Summarize this article, translate to Spanish, find sentiment,
      extract entities, and rate the writing quality."

Good: Break into 5 separate calls, or use structured output to get
      all 5 in one well-defined schema.
```

### ❌ Mistake 6: Not Using Examples

For non-standard tasks, **always** provide 2-3 examples. It's the single biggest accuracy boost you can get.

### ❌ Mistake 7: Ignoring Temperature

Generating code with `temperature=0.9`? You'll get random nonsense.
Writing creative fiction with `temperature=0`? You'll get boring text.

Match temperature to the task.

### ❌ Mistake 8: Forgetting to Test Edge Cases

Always test:
- Empty input
- Very long input
- Adversarial input ("Ignore previous instructions")
- Inputs in different languages
- Ambiguous inputs

---

## Final Project

### Project: Resume Analyzer

Build a CLI tool that takes a resume (text or PDF) and produces:

1. Structured extraction (Pydantic-validated)
2. A scored evaluation (technical depth, clarity, achievements)
3. 3 specific improvement suggestions

This combines **few-shot prompting**, **structured outputs**, **CoT**, and **reflection**.

### Requirements

1. Use `instructor` for structured outputs
2. Use Pydantic models for all schemas
3. Implement at least 3 prompt techniques (few-shot, CoT, reflection)
4. Templates stored as separate files
5. Track prompt version
6. Async (parallelize the analyses)

### Project Structure

```
week-04-prompt-engineering/
├── README.md
├── requirements.txt
├── .env
├── .gitignore
├── analyzer.py             # Main CLI app
├── models.py               # Pydantic schemas
├── prompts/
│   ├── extract_v1.md
│   ├── evaluate_v1.md
│   └── suggest_v1.md
├── examples/
│   └── sample_resume.txt
└── notes.md
```

### Starter Code

**`models.py`**

```python
from pydantic import BaseModel, Field
from typing import Optional, Literal


class WorkExperience(BaseModel):
    company: str
    role: str
    start_year: int
    end_year: Optional[int] = None
    achievements: list[str] = []


class Education(BaseModel):
    institution: str
    degree: str
    year: Optional[int] = None


class ExtractedResume(BaseModel):
    """Structured data extracted from a resume."""
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None
    summary: Optional[str] = None
    skills: list[str] = []
    experiences: list[WorkExperience] = []
    education: list[Education] = []


class CategoryScore(BaseModel):
    score: int = Field(ge=1, le=10)
    reasoning: str


class Evaluation(BaseModel):
    """Multi-dimensional resume evaluation."""
    technical_depth: CategoryScore
    clarity: CategoryScore
    achievements_quality: CategoryScore
    overall_score: int = Field(ge=1, le=10)
    overall_reasoning: str


class Suggestion(BaseModel):
    issue: str
    recommendation: str
    priority: Literal["high", "medium", "low"]


class Improvements(BaseModel):
    """3 actionable improvement suggestions."""
    suggestions: list[Suggestion] = Field(min_length=3, max_length=5)


class ResumeAnalysis(BaseModel):
    """The complete output of the analyzer."""
    extracted: ExtractedResume
    evaluation: Evaluation
    improvements: Improvements
```

**`prompts/extract_v1.md`**

```markdown
# Resume Extraction Prompt v1.0

You are an expert resume parser. Extract structured information from the resume below.

## Rules:
- Extract only what is explicitly written. Don't make up info.
- For experiences with no end date, set end_year to null.
- For each experience, list 1-5 specific achievements (use action verbs).
- Skills should be specific (e.g., "Python", "PostgreSQL") not vague (e.g., "programming").

## Examples:

### Input:
"Software Engineer at Google, 2020 - present.
Worked on search infrastructure, reduced latency by 30%."

### Output:
{
  "company": "Google",
  "role": "Software Engineer",
  "start_year": 2020,
  "end_year": null,
  "achievements": ["Reduced search infrastructure latency by 30%"]
}

---

## Resume to extract:
{resume_text}
```

**`prompts/evaluate_v1.md`**

```markdown
# Resume Evaluation Prompt v1.0

You are a senior tech recruiter evaluating resumes for software engineering roles.

## Task:
Score this resume across 3 dimensions on a 1-10 scale.

## Scoring guide:
- **Technical depth**: How specific and advanced are the technical skills/projects?
  - 1-3: Vague, no specific technologies
  - 4-6: Lists technologies but no depth
  - 7-9: Shows specific projects with technical detail
  - 10: Exceptional, with measurable impact

- **Clarity**: How easy is it to scan and understand?
  - 1-3: Confusing, walls of text
  - 4-6: Readable but verbose
  - 7-9: Clear, well-structured
  - 10: Crystal clear, easy to scan

- **Achievements quality**: Are accomplishments specific and measurable?
  - 1-3: Generic ("worked on", "helped with")
  - 4-6: Some specifics, no metrics
  - 7-9: Specific with some metrics
  - 10: Strong metrics throughout

## Approach:
Think step-by-step. For each dimension, identify specific examples from the resume,
then assign a score with reasoning.

## Resume:
{resume_text}
```

**`prompts/suggest_v1.md`**

```markdown
# Improvement Suggestions Prompt v1.0

You are a brutally honest career coach.

## Task:
Based on the resume and evaluation below, give 3-5 SPECIFIC improvements.

## Rules:
- Be specific. "Add more detail" is bad. "In your OpenSolar role, quantify the imagery system's impact (e.g., users served, latency reduced)" is good.
- Reference exact phrases or sections of the resume.
- Prioritize: high (must fix), medium (should fix), low (nice to fix).
- Don't suggest things already done well.

## Resume:
{resume_text}

## Evaluation:
{evaluation_json}

## Provide your suggestions as structured output.
```

**`analyzer.py`**

```python
import asyncio
import os
from pathlib import Path
import instructor
from openai import AsyncOpenAI
from dotenv import load_dotenv
from models import (
    ExtractedResume, Evaluation, Improvements, ResumeAnalysis
)

load_dotenv()

# Use the async instructor client
client = instructor.from_openai(AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY")))

PROMPTS_DIR = Path(__file__).parent / "prompts"


def load_prompt(name: str) -> str:
    return (PROMPTS_DIR / f"{name}.md").read_text()


async def extract_resume(resume_text: str) -> ExtractedResume:
    prompt = load_prompt("extract_v1").replace("{resume_text}", resume_text)
    return await client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=ExtractedResume,
        temperature=0,  # Extraction = deterministic
        messages=[{"role": "user", "content": prompt}]
    )


async def evaluate_resume(resume_text: str) -> Evaluation:
    prompt = load_prompt("evaluate_v1").replace("{resume_text}", resume_text)
    return await client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=Evaluation,
        temperature=0.3,  # Some creativity in reasoning
        messages=[{"role": "user", "content": prompt}]
    )


async def suggest_improvements(
    resume_text: str,
    evaluation: Evaluation
) -> Improvements:
    prompt = load_prompt("suggest_v1") \
        .replace("{resume_text}", resume_text) \
        .replace("{evaluation_json}", evaluation.model_dump_json(indent=2))

    return await client.chat.completions.create(
        model="gpt-4o-mini",
        response_model=Improvements,
        temperature=0.5,
        messages=[{"role": "user", "content": prompt}]
    )


async def analyze_resume(resume_text: str) -> ResumeAnalysis:
    """Full pipeline: extract → evaluate → suggest."""
    # Run extraction and evaluation in parallel
    extracted, evaluation = await asyncio.gather(
        extract_resume(resume_text),
        evaluate_resume(resume_text)
    )

    # Suggestions need the evaluation, so this is sequential
    improvements = await suggest_improvements(resume_text, evaluation)

    return ResumeAnalysis(
        extracted=extracted,
        evaluation=evaluation,
        improvements=improvements
    )


def print_report(analysis: ResumeAnalysis) -> None:
    print("\n" + "=" * 60)
    print(f"📋 RESUME ANALYSIS: {analysis.extracted.name}")
    print("=" * 60)

    print(f"\n📧 Email: {analysis.extracted.email}")
    print(f"💼 Experiences: {len(analysis.extracted.experiences)}")
    print(f"🛠️  Skills: {', '.join(analysis.extracted.skills[:10])}")

    print("\n" + "-" * 60)
    print("📊 EVALUATION")
    print("-" * 60)
    e = analysis.evaluation
    print(f"Technical depth:      {e.technical_depth.score}/10")
    print(f"Clarity:              {e.clarity.score}/10")
    print(f"Achievements:         {e.achievements_quality.score}/10")
    print(f"OVERALL:              {e.overall_score}/10")
    print(f"\n💭 {e.overall_reasoning}")

    print("\n" + "-" * 60)
    print("✨ TOP IMPROVEMENTS")
    print("-" * 60)
    for i, s in enumerate(analysis.improvements.suggestions, 1):
        priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}[s.priority]
        print(f"\n{i}. {priority_icon} {s.issue}")
        print(f"   → {s.recommendation}")


async def main():
    import sys
    if len(sys.argv) < 2:
        print("Usage: python analyzer.py <resume_file>")
        return

    resume_path = Path(sys.argv[1])
    if not resume_path.exists():
        print(f"File not found: {resume_path}")
        return

    resume_text = resume_path.read_text()

    print("🔍 Analyzing resume...")
    analysis = await analyze_resume(resume_text)
    print_report(analysis)


if __name__ == "__main__":
    asyncio.run(main())
```

**`requirements.txt`**

```
openai>=1.0
instructor>=1.0
pydantic>=2.0
python-dotenv>=1.0
```

### What You Should Learn from This Project

- **Few-shot prompting** in extraction
- **CoT** in evaluation (forcing reasoning before scoring)
- **Reflection** in improvements (using evaluation as input)
- **Structured outputs** with Pydantic + instructor
- **Prompt templating** with separate files
- **Parallelization** with asyncio.gather

### Stretch Goals

- Add support for PDF input (use `pypdf` or `pdfplumber`)
- Compare 2 resumes side by side
- Generate an improved version of the resume
- Add cost tracking per call

---

## Self-Check Questions

1. What are the 5 parts of a well-structured prompt?
2. Difference between zero-shot and few-shot?
3. Why does Chain-of-Thought improve performance?
4. What does the ReAct loop alternate between?
5. Name 3 ways to get structured outputs from an LLM.
6. Why is `instructor` better than raw JSON mode?
7. What's the right temperature for code generation? For poetry?
8. Why prefer positive instructions over negative ones?
9. What's prompt versioning and why does it matter?
10. When should you NOT use reflection?

### Answers

1. Role, Task, Context, Format, Examples.
2. Zero-shot = no examples. Few-shot = 2-5 examples to teach the pattern.
3. By generating reasoning tokens before the answer, each step builds on the previous, leading to more accurate conclusions.
4. Thought (reasoning) → Action (tool call) → Observation (result), repeating until done.
5. JSON mode, function calling/tool use, instructor (Pydantic).
6. Type-safe, validated, less boilerplate, automatic retries on validation errors.
7. Code: 0 to 0.2 (deterministic). Poetry: 0.8 to 1.2 (creative).
8. LLMs follow positive instructions more reliably; negatives sometimes get ignored.
9. Tracking which prompt version produced which results. Critical for measuring improvements and rollback.
10. Simple tasks, time-sensitive apps, cost-sensitive apps — reflection adds latency and cost.

---

## Resources

### Must-Read
- [OpenAI Prompt Engineering Guide](https://platform.openai.com/docs/guides/prompt-engineering)
- [Anthropic's Prompting Guide](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview)
- [Instructor Library Docs](https://python.useinstructor.com/)

### Papers (Read at Least One)
- "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models" (Wei et al.)
- "ReAct: Synergizing Reasoning and Acting in Language Models" (Yao et al.)
- "Reflexion: Language Agents with Verbal Reinforcement Learning" (Shinn et al.)

### Tools to Try
- [Anthropic's Prompt Generator](https://console.anthropic.com/dashboard) — Auto-improves your prompts
- [PromptLayer](https://promptlayer.com/) — Prompt versioning and tracking
- [Langfuse Prompt Management](https://langfuse.com/) — Open source alternative

---

## Next Up

**Week 5:** Function Calling and Tool Use — making LLMs interact with real systems (databases, APIs, files).

You've now learned how to make LLMs **think well**. Next, we make them **act**.
