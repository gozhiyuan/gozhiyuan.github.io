---
layout: post
title: "Agent Harness Engineering Part 4: Context and Memory"
subtitle: Context windows, compaction, session notes, persistent memory, and drift
categories: Large-Language-Model Agents Harness-Engineering
tags: [Blogs]
banner: "/assets/images/banners/yuanpang-wa-iceburg2.jpg"
---

# Agent Harness Engineering Part 4: Context and Memory

This post is the fourth part of my notes on [Agent Harness Engineering: A Survey of LLM Infrastructure](https://picrew.github.io/LLM-Harness/main.pdf). [Part 1]({% post_url 2026-05-23-harness-engineering-01 %}) covered **E: Execution Environment and Sandbox**. [Part 2]({% post_url 2026-05-23-harness-engineering-02 %}) covered **T: Tool Interface and Protocol Layer**. [Part 3]({% post_url 2026-05-25-harness-engineering-03 %}) covered **L: Lifecycle and Orchestration**.

This post focuses on **C: Context and Memory Management** from section 5 of the paper: what information the model sees at each step, how that information is arranged, what gets persisted outside the context window, and how long-running agents avoid losing the thread.

The core problem is simple to state and hard to execute:

> Give the model exactly the right information at each step, and nothing more.

Too little context and the agent cannot act correctly. Too much context and the agent becomes slower, more expensive, and less reliable. Larger context windows help, but they do not remove the engineering problem.

## 1. Context Is Not Free

It is tempting to treat context as a large bucket. If the model accepts 200K, 1M, or more tokens, why not include everything that might matter?

The answer is that context is a scarce resource for three separate reasons.

### 1.1 Attention Cost

Transformers compare tokens to other tokens through self-attention. For `n` tokens, the naive attention pattern has `n^2` pairwise relationships. Modern kernels and serving tricks reduce the practical cost, but they do not make long context free.

This matters for harness design because agent context grows naturally:

```text
system instructions
+ tool definitions
+ user task
+ previous assistant messages
+ tool calls
+ tool results
+ retrieved files
+ generated plans
+ intermediate summaries
+ memory records
```

Every extra token competes for compute, latency, cost, and attention.

### 1.2 Position Sensitivity

Long-context models do not use all positions equally well. Liu et al.'s "Lost in the Middle" result showed that models often perform best when relevant information is near the beginning or end of the input, and worse when the same information appears in the middle.

The practical lesson is direct: retrieval is not enough. An agent can fetch the right information and still fail if the harness places it where the model is unlikely to use it.

That gives context engineering two jobs:

1. Select the right information.
2. Put it where the model can use it.

### 1.3 Context Rot

Chroma's 2025 context-rot report evaluated 18 models, including GPT-4.1, Claude 4, Gemini 2.5, and Qwen3 variants. The report held task difficulty mostly fixed while increasing input length, and found that model behavior became less reliable as inputs grew.

The important detail is that degradation can happen well before the advertised maximum window. A model with a very large context window may still perform worse when the harness fills it with stale tool outputs, repeated observations, or semantically distracting material.

This is why the context layer exists. It is not just a convenience layer. It is reliability infrastructure.

## 2. Prompt Engineering Versus Context Engineering

Prompt engineering mostly asks:

> What words should I put in this instruction?

Context engineering asks a broader question:

> What full information state should the model receive on this inference step?

Anthropic defines context engineering as strategies for curating and maintaining the optimal set of tokens during inference, including information outside the prompt itself. In an agent, that includes:

| Context component | Example |
| --- | --- |
| System prompt | Role, constraints, safety policy, output format |
| Tool definitions | Tool names, descriptions, schemas, examples |
| Runtime state | Current phase, task status, active branch, budgets |
| Conversation history | Prior user and assistant messages |
| Tool history | Shell output, file reads, browser observations |
| Retrieved content | Source snippets, docs, web pages, memory records |
| Working notes | Plans, todos, open questions, implementation state |

The difference from prompt engineering is that context engineering is continuous. A harness makes context decisions before every model call:

```text
current task state
+ available tools
+ recent observations
+ retrieved records
+ memory / notes
+ token budget
-> context projection for this step
```

So the context layer is a controller. It decides what to preload, what to retrieve just in time, what to summarize, what to drop, and what to persist elsewhere.

## 3. A Three-Tier Memory Model

The paper organizes context and memory by time horizon. I find this the most useful framing:

| Tier | Time horizon | Storage location | Main harness question |
| --- | --- | --- | --- |
| Short-term active context | Current call or short turn sequence | Model context window | What should the model see right now? |
| Mid-term session state | Current session or next run | Files, notes, task artifacts, compact summaries | What must survive a reset or `/clear`? |
| Long-term persistent memory | Many sessions, tasks, or users | Indexed memory store, vector DB, graph DB, KV store | What should be retrievable later? |

This resembles an operating-system memory hierarchy:

```text
Active context window  -> RAM-like working set
Session files / notes  -> local checkpoint state
Persistent memory      -> disk / database
```

The analogy is imperfect, but useful. Fast memory is small and expensive. Slow memory is larger but must be retrieved, summarized, and validated.

## 4. Short-Term: Managing the Active Context Window

Short-term context management controls what the model sees during the next inference step. This tier is the most immediate and usually the highest-leverage.

### 4.1 System Prompt Calibration

The system prompt is always present, so every unnecessary token is paid repeatedly. But an underspecified prompt creates unreliable behavior.

Anthropic's guidance is to tune the prompt at the right "altitude":

| Prompt failure | What it looks like |
| --- | --- |
| Too specific | Brittle if/else instructions, edge-case lists, maintenance-heavy logic |
| Too vague | High-level values without concrete operational guidance |
| Better | Clear sections for background, instructions, tool guidance, and output format |

The workflow should be empirical:

1. Start with the smallest prompt that gives the model enough task framing.
2. Run real or representative tasks.
3. Identify failure modes.
4. Add targeted instructions or examples only where they address observed failures.

This is the opposite of preloading every edge case into the system prompt.

### 4.2 Token-Efficient Tool Design

Tool definitions are part of context. A large tool menu can consume thousands or tens of thousands of tokens before the user request even begins.

The design rule from Part 2 applies again here: a tool is a contract. From the context layer's perspective, a good tool is not only callable. It is concise, distinct, and easy for the model to choose.

| Tool design choice | Context effect |
| --- | --- |
| Many overlapping narrow tools | More tokens, more ambiguity |
| Few expressive tools | Smaller menu, clearer choices |
| Verbose raw outputs | Context pollution |
| Structured compact outputs | Easier state projection |
| Dynamic tool removal | Can break cache reuse and confuse prior tool history |
| Phase-gated tools | Keeps behavior constrained; can be implemented by filtering tools, rejecting illegal calls, or decoder constraints |

Manus's context-engineering writeup makes the cache implication especially clear: because tool definitions sit near the front of serialized context, changing them can invalidate cached prefixes for every later turn.

### 4.3 Just-in-Time Retrieval

Good agents do not load the whole world up front. They keep pointers and fetch details when needed.

Examples:

| Pointer | Loaded only when useful |
| --- | --- |
| File path | Relevant file contents |
| Search query | Matching source snippets |
| Issue URL | Issue body and comments |
| Database query | Query results |
| Memory key | Related prior experience |
| Web link | Page text or metadata |

Anthropic calls this progressive disclosure. Claude Code's pattern is a concrete example: project-level memory files such as `CLAUDE.md` can be loaded early, while `glob`, `grep`, file reads, and shell commands let the agent inspect the repository incrementally.

This mirrors how engineers work. We do not paste the entire repository into our heads. We use filenames, directories, tests, git history, and search as an external index.

### 4.4 KV-Cache-Aware Context

Production agents often make many model calls with a long shared prefix. Manus makes this point sharply: in an agent loop, the input is often huge and the output is usually just a short tool call, so the prefill-to-decode ratio is much higher than in ordinary chat. Manus reports an average input-to-output token ratio around `100:1`. That means the expensive part is often not generating the next action. It is repeatedly reprocessing the same growing context before the model emits that action.

Manus identifies KV-cache hit rate as a critical production metric and gives three practical rules:

| Rule | Reason |
| --- | --- |
| Keep the prompt prefix stable | A small early change invalidates the cache after that point |
| Treat context as append-only | Rewriting earlier turns creates a different prefix |
| Serialize deterministically | Unstable JSON key ordering can silently break cache reuse |

The terminology is easy to blur, so I separate four related ideas:

| Term | What it means | What it optimizes |
| --- | --- | --- |
| KV cache | The per-layer key/value tensors produced while processing prior tokens | Avoid recomputing attention state inside one request or continuation |
| Prefix caching / prompt caching | Reuse KV cache across requests that share an identical token prefix | Time-to-first-token, prefill cost, throughput |
| Provider prompt caching | API-level feature where a provider caches marked or detected prompt prefixes and bills cached tokens differently | Cost and latency through a managed API |
| Response prefill | Start the assistant response with a fixed partial output such as `<tool_call>` | Output control, not prompt reuse |

So "KV cache" is the underlying inference artifact. "Prefix caching" is a serving policy that reuses that artifact when prompts share a prefix. "Prompt caching" is often the product/API name for the same idea, sometimes with explicit cache breakpoints. "Response prefill" is different: it seeds the beginning of the output so the model continues from a constrained position.

The harness implication is that context should be structured intentionally:

```text
stable system prompt
+ stable tool definitions
+ stable policy / role instructions
+ append-only conversation and observations
+ compacted or retrieved state after cache-safe boundaries
```

The prefix that should remain stable is usually the front of the request:

```text
Cache-friendly layout:

[system prompt: stable]
[tool definitions: stable and deterministically serialized]
[developer / policy instructions: stable]
[session history: append-only]
[latest user message or observation: append-only suffix]
```

The bad pattern is putting high-churn material at the front:

```text
Cache-hostile layout:

[timestamp: 2026-05-27T10:12:45.182931]
[random request id: 8f91...]
[system prompt]
[tools serialized with non-stable key order]
[history]
```

A timestamp in the first line can make every request look like a different prompt. A JSON serializer that changes key ordering can do the same thing more quietly.

In Python, that means the context builder should be boring on purpose:

```python
import json


SYSTEM_PROMPT = """You are a coding agent.
Follow repository conventions. Use tools only when needed."""

TOOLS = [
    {
        "name": "browser_open",
        "description": "Open a URL in the browser.",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
        },
    },
    {
        "name": "shell_exec",
        "description": "Run a shell command in the sandbox.",
        "parameters": {
            "type": "object",
            "properties": {"cmd": {"type": "string"}},
            "required": ["cmd"],
        },
    },
]


STABLE_TOOL_BLOCK = json.dumps(
    TOOLS,
    sort_keys=True,
    separators=(",", ":"),
)


def build_prompt(history: list[str], latest_observation: str) -> str:
    # The front of this prompt is byte-stable across turns.
    return "\n".join(
        [
            "<system>",
            SYSTEM_PROMPT,
            "</system>",
            "<tools>",
            STABLE_TOOL_BLOCK,
            "</tools>",
            "<history>",
            "\n".join(history),
            latest_observation,
            "</history>",
            "<|im_start|>assistant",
        ]
    )
```

With vLLM, automatic prefix caching can be enabled on the engine. The vLLM docs describe automatic prefix caching as reusing the KV cache when a new query shares a prefix with an existing query, and show `enable_prefix_caching=True` for the engine:

```python
from vllm import LLM, SamplingParams


llm = LLM(
    model="NousResearch/Hermes-3-Llama-3.1-8B",
    enable_prefix_caching=True,
)

sampling = SamplingParams(temperature=0, max_tokens=128)

SHARED_PREFIX = build_prompt(
    history=[
        "User: Analyze this repository.",
        "Assistant: I will inspect the project structure.",
        "Tool: shell_exec {\"cmd\":\"rg --files\"}",
        "Observation: src/agent.py, src/context.py, tests/test_agent.py",
    ],
    latest_observation="",
)

prompts = [
    SHARED_PREFIX + "\nUser: What file should I read next?",
    SHARED_PREFIX + "\nUser: Which tests cover src/context.py?",
]

outputs = llm.generate(prompts, sampling)
```

The second request can reuse the cached KV blocks for `SHARED_PREFIX` if the prefix tokenization is identical and still resident in the serving worker's cache.

That last phrase matters in distributed serving. Prefix caches are usually local to a worker or to a cache tier connected to that worker. If turn 1 for a session goes to worker A and turn 2 goes to worker B, worker B may not have the cached prefix. In a self-hosted setup, route requests with sticky session affinity:

```python
import hashlib


WORKERS = [
    "http://vllm-worker-0:8000",
    "http://vllm-worker-1:8000",
    "http://vllm-worker-2:8000",
    "http://vllm-worker-3:8000",
]


def route_for_session(session_id: str) -> str:
    digest = hashlib.sha256(session_id.encode()).digest()
    index = int.from_bytes(digest[:4], "big") % len(WORKERS)
    return WORKERS[index]
```

An equivalent Nginx-style pattern is:

```nginx
upstream vllm_backend {
    hash $http_x_session_id consistent;
    server vllm-worker-0:8000;
    server vllm-worker-1:8000;
    server vllm-worker-2:8000;
    server vllm-worker-3:8000;
}
```

The same idea applies above vLLM: if the agent runtime knows `session_id`, the inference gateway should keep that session's turns on a worker likely to retain the relevant prefix cache.

#### Stable Tools and Phase Gates

This point is easy to misunderstand. In many agent frameworks, tools look like they are initialized once at the beginning. The model server is initialized once, but the model request is usually rebuilt on every agent step:

```text
1. Model / server initialization
   Load weights once.

2. Agent-step request construction
   Send system prompt + messages + tool definitions for this step.
```

Because tool definitions usually serialize near the front of the prompt, changing them can reduce prefix-cache reuse. But "masking tools" should not be used as a vague phrase. There are three different mechanisms:

| Mechanism | Where it happens | Typical implementation | Tradeoff |
| --- | --- | --- | --- |
| Visibility filter | Before generation | Pass only currently allowed tools in the request | Most portable; lower cache reuse if the tool block changes often |
| Harness gate | After generation, before execution | Pass stable tools, but reject illegal tool calls | Cache-friendly; model may waste a turn proposing a disallowed action |
| Decoder constraint | During generation | Grammar, JSON schema enum, allowed-function field, or logits mask | Best control and cache behavior; requires serving-stack support |

For most hosted APIs, the visibility filter is the simple and correct default:

```python
TOOLS = [file_read, file_edit, shell_exec, run_tests]

while True:
    allowed = allowed_tools_for_phase(phase)
    visible_tools = [tool for tool in TOOLS if tool.name in allowed]

    response = model.chat(
        messages=messages,
        tools=visible_tools,
    )
```

For cache-sensitive deployments, keep a stable role-level tool block and enforce legality in the harness:

```python
TOOLS = [file_read, file_edit, shell_exec, run_tests]

while True:
    allowed = allowed_tools_for_phase(phase)

    response = model.chat(
        messages=messages,
        tools=TOOLS,  # stable for this agent role
    )

    if response.tool_call and response.tool_call.name not in allowed:
        messages.append(
            tool_error(
                response.tool_call,
                f"{response.tool_call.name} is not allowed during {phase}. "
                f"Allowed tools: {sorted(allowed)}",
            )
        )
        continue

    if response.tool_call:
        result = run_tool(response.tool_call)
        messages.append(tool_result(response.tool_call, result))
        continue

    break
```

This is not a decoder mask. The model can still propose `shell_exec`; the harness simply refuses to execute it if the current phase only allows `file_read`.

True decoder constraints require provider or inference-stack support. The harness still chooses the allowed action space from lifecycle state; the model chooses within that space:

```text
phase = "browser_investigation"
allowed_tools = ["browser_click", "browser_extract"]
model chooses one allowed tool call
harness validates and executes it
```

If the model should decide the phase, split that into a separate structured routing step:

```text
model chooses next_phase from ["investigate", "edit", "verify"]
harness maps next_phase to allowed_tools
model chooses a tool within allowed_tools
```

With a vLLM-style structured-output path, the constraint can be a JSON schema enum rather than a hand-written grammar:

```python
allowed_tools = ["browser_click", "browser_extract"]

schema = {
    "type": "object",
    "properties": {
        "tool_name": {"type": "string", "enum": allowed_tools},
        "arguments": {"type": "object"},
    },
    "required": ["tool_name", "arguments"],
}

completion = client.chat.completions.create(
    model=model,
    messages=messages,
    tools=TOOLS,  # stable documentation of the role's tool universe
    extra_body={"structured_outputs": {"json": schema}},
)

call = json.loads(completion.choices[0].message.content)
assert call["tool_name"] in allowed_tools
```

Here "grammar" means a formal decoding constraint, not natural-language grammar. A toy version would be:

```text
tool_call := '{"name":"' tool_name '","arguments":' json_object '}'
tool_name := "browser_click" | "browser_extract"
```

During decoding, invalid continuations are masked out. If the output has started `{"name":"browser_`, only continuations that can finish `browser_click` or `browser_extract` remain legal. The application still performs a final policy check before executing the selected tool.

The same design applies to multi-agent systems. The stable unit is usually the agent role, not the whole product:

```python
coder = Agent(
    system="You are a coding agent.",
    tools=[file_read, file_edit, shell_exec, run_tests],
)

browser = Agent(
    system="You are a browser agent.",
    tools=[browser_open, browser_click, browser_extract],
)

reviewer = Agent(
    system="You are a reviewer.",
    tools=[inspect_diff, run_tests, comment],
)
```

Each role can keep a stable prompt and tool block across its own loop, while the harness still gates which actions are legal for the current phase.

#### File System Context and Recitation

Manus also connects caching to memory layout. If an observation is huge, irreversible compression is risky because the agent may need a detail ten steps later. Their pattern is to make compression restorable: drop a web page body from the active context only if the URL remains; drop a document body only if the file path remains; drop a log only if the log file remains available in the sandbox.

```text
Active context:
  "The crawl result is stored at artifacts/pages/pricing.html.
   Relevant finding: enterprise plan mentions SSO and audit logs."

External context:
  artifacts/pages/pricing.html
  artifacts/logs/crawl-2026-05-27.txt
```

This is a cache-friendly memory policy. The active prompt stays smaller and more stable, while the full evidence is still recoverable through tools.

The `todo.md` behavior Manus describes is related but slightly different. Rewriting a todo list pushes the global plan near the end of the context, where models tend to attend better than the middle. It is a natural-language attention control mechanism:

```markdown
# todo.md

- [x] Inspect repository structure
- [x] Find the context builder
- [ ] Patch deterministic tool serialization
- [ ] Run focused tests
- [ ] Summarize validation and risks
```

An agent can recite the current todo after each major observation:

```text
Observation: tests/test_context.py failed because tool order changed.

Current todo:
- [x] Inspect repository structure
- [x] Find the context builder
- [ ] Patch deterministic tool serialization
- [ ] Run focused tests
- [ ] Summarize validation and risks
```

This does cost tokens, but it spends them at a high-attention position. The harness design principle is not "always summarize everything." It is "keep recoverable evidence outside the window, and keep the current objective visible where the model is likely to use it."

Anthropic's context-management release also turns some of this into product-level infrastructure: context editing clears stale tool calls and results as the window approaches limits, while the memory tool stores information outside the context window.

## 5. Mid-Term: Session State and Cross-Run Persistence

Mid-term memory is the layer between active context and a full indexed memory system. It handles continuity across resets, compactions, crashes, and follow-up runs.

This is where the boundary between **context**, **agent-authored working memory**, and **lifecycle state** from Part 3 becomes important. The model should not be the only place that knows what happened. The harness should persist important state externally, decide which part to re-inject, and keep the authoritative lifecycle separate from ordinary notes.

The clean split is:

| Layer | Who owns it | Example | Authority |
| --- | --- | --- | --- |
| Active context | Harness projection | Current prompt, selected files, recent tool results | What the model can see now |
| Working memory files | Agent or user | `todo.md`, `findings.md`, `validation.md` | Useful notes, not authoritative state |
| Lifecycle state | Harness | `phase`, `attempts`, `branch`, `last_test_passed` | Authoritative operational truth |
| Long-term memory | Memory system | Vector/keyword/graph records | Retrieved background knowledge |

The practical rule is: **agent writes notes; harness writes state**. The model can propose an update to lifecycle state, but the harness should validate and commit it.

### 5.1 File-Backed Session Memory

The simplest mid-term pattern is to move working memory into files. This includes lightweight note-taking, task planning, and durable evidence references. These files help the agent recover after `/clear`, compaction, a crash, or a later follow-up.

The lightweight version is a small set of markdown files:

```text
NOTES.md
TODO.md
findings.md
progress.md
```

The more operational version is a task workspace:

```text
.agent-task/
  notes.md             # agent-authored working memory
  todo.md              # current plan and open steps
  findings.md          # evidence discovered during investigation
  validation.md        # commands run and results
  decisions.md         # durable design choices and tradeoffs
  artifacts/
    pytest-001.txt     # full logs or bulky evidence
```

The agent reads these files at the beginning of a run, updates them at milestones, and uses them to recover when conversation history is unavailable. This works because it externalizes working memory. Instead of relying on a long transcript, the agent records durable facts:

| File-backed state | Example |
| --- | --- |
| Goal | "Migrate parser from regex to AST walker." |
| Decision | "Use existing `ParserContext` instead of new global state." |
| Open issue | "Two CLI tests still fail because fixture paths differ." |
| Validation | "`pytest tests/test_cli.py -q` passed on commit X." |
| Next step | "Update docs after API name is final." |

Projects such as `planning-with-files` keep markdown plans, findings, and progress files on disk so an agent can survive context loss, `/clear`, or crashes. Trellis uses repo-local specs, task files, and workspace journals so each new coding session starts with relevant project memory rather than a blank slate.

The harness can then project only the relevant pieces:

```text
full files on disk
        |
        v
short model-facing summary
  - current objective
  - completed steps
  - blocking issues
  - next command or edit target
```

This is much cheaper than replaying the whole conversation.

The point is not that Markdown is magical. The point is that the harness gives the agent a low-cost external state surface.

### 5.2 Lifecycle State Is Harness-Owned

Lifecycle state is different from notes. It is the internal task record the orchestrator uses to decide what phase can run next. In a custom agent application, this is often just a Python dict, dataclass, row in SQLite, or JSON blob owned by the harness.

For a text-to-SQL agent, a minimal lifecycle object might look like:

```python
from dataclasses import dataclass, field


@dataclass
class TextToSqlLifecycle:
    phase: str = "generate_sql"
    question: str = ""
    attempts: int = 0
    max_repairs: int = 2
    last_sql: str | None = None
    last_error: str | None = None
    validated_sql: str | None = None
    executed: bool = False
    tables_used: list[str] = field(default_factory=list)
```

The harness mutates this state around model calls and tool calls:

```python
state = TextToSqlLifecycle(question=question)

while state.attempts <= state.max_repairs:
    state.attempts += 1

    response = model_generate_sql(
        question=question,
        state=project_state_for_model(state),
    )
    state.last_sql = response.sql

    validation = validate_sql(response.sql)
    if not validation.ok:
        state.phase = "repair_sql"
        state.last_error = validation.reason
        continue

    state.validated_sql = validation.sql
    state.tables_used = validation.tables

    try:
        result = adapter.execute(validation.sql)
    except QueryError as exc:
        state.phase = "repair_sql"
        state.last_error = str(exc)
        continue

    state.phase = "done"
    state.executed = True
    break
```

The model should see only a projection, not the whole authoritative object:

```python
def project_state_for_model(state: TextToSqlLifecycle) -> str:
    return f"""
Current phase: {state.phase}
Attempt: {state.attempts}/{state.max_repairs + 1}
Last error: {state.last_error or "none"}
Last SQL: {state.last_sql or "none"}
"""
```

For long-running tasks, persist the lifecycle state:

```text
.agent-task/
  state.json         # harness-owned, structured, authoritative
  notes.md           # agent-authored, useful but not authoritative
  artifacts/
    pytest-001.txt
```

Example `state.json`:

```json
{
  "phase": "verify",
  "branch": "fix/parser-bug",
  "attempt_count": 2,
  "changed_files": ["src/parser.py", "tests/test_parser.py"],
  "last_test_command": "pytest tests/test_parser.py -q",
  "last_test_passed": true,
  "approval_to_publish": false
}
```

This distinction matters because notes can be stale or aspirational. Lifecycle state drives permissions, retries, approvals, and completion. A model-written note that says "tests passed" is not enough; the harness should record which command passed, on which revision, and whether that evidence satisfies the current phase.

### 5.3 Cross-Run Injection and Claude Code Memory

Another mid-term pattern is to summarize one run and inject the useful residue into the next.

Claude Code has two official memory surfaces: [`CLAUDE.md` files](https://code.claude.com/docs/en/memory) and auto memory. The docs describe `CLAUDE.md` as user-written persistent instructions and auto memory as notes Claude writes itself from corrections, preferences, and project learnings. Both are loaded as context, not enforced configuration; hard enforcement belongs in hooks or settings.

```text
CLAUDE.md
  - written by user/team/org
  - coding standards, workflows, architecture, commands
  - loaded into sessions as instructions

auto memory
  - written by Claude
  - debugging insights, discovered commands, preferences
  - loaded or recalled across sessions as context
```

The newer Claude Code docs also document a `MEMORY.md` entrypoint for auto memory. Auto memory lives under a project-specific directory such as:

```text
~/.claude/projects/<project>/memory/
  MEMORY.md          # concise index, loaded into every session up to a limit
  debugging.md       # detailed topic files
  api-conventions.md
```

`MEMORY.md` acts as an index. Claude keeps it concise and can read topic files on demand with file tools. This is a good example of mid-term memory: it carries useful project knowledge across sessions, but it is still context, not an authority layer. The docs also note that `/memory` lets you view and edit loaded memory files and auto memory.

Third-party systems such as [`claude-mem`](https://github.com/thedotmack/claude-mem) extend the same idea through Claude Code plugin hooks plus a separate local memory backend. In practical terms, the hooks are the integration surface, not the whole memory system.

The Claude Code hooks are the moments where `claude-mem` can observe or inject context:

| Hook moment | What `claude-mem` can do |
| --- | --- |
| `SessionStart` | Ensure the worker is running and inject relevant context from previous sessions |
| `UserPromptSubmit` | Create or update the current session record and store the raw prompt |
| `PostToolUse` | Capture tool observations such as file reads, shell output, edits, and browser results |
| `Stop` | Summarize the session and extract reusable learnings |
| `SessionEnd` / cleanup | Mark session lifecycle state in its own database |

The hook scripts call into a worker service. That worker is the local web service behind the memory layer: it exposes HTTP endpoints and a viewer UI, writes SQLite/FTS5 records, optionally syncs to a Chroma semantic index, and uses an LLM/agent SDK path to compress observations into summaries.

```text
Claude Code lifecycle hooks
  -> local worker service / HTTP API
  -> SQLite / FTS5 sessions, observations, summaries
  -> optional Chroma semantic index
  -> context-generator hook output
  -> mem-search skill or MCP-style search tools
```

The important architectural point is that the hooks are thin boundary adapters. They receive lifecycle events from Claude Code, then delegate durable memory work to the worker/database/search system. If the worker is already running, the hooks reuse it. If not, the session-start path can start it before context injection.

Its search tools then use progressive disclosure:

```text
search -> compact result IDs
timeline -> surrounding context
get_observations -> full details for selected IDs
```

So is `claude-mem` "just hooks"? No. The hook registration is the way it attaches to Claude Code, but the memory behavior comes from the backend that stores, summarizes, indexes, and retrieves those hook observations. Is it "mid-term memory only"? Not exactly. Architecturally, it starts as a cross-run continuity layer, which is mid-term memory. But because it stores observations in SQLite, supports semantic search through Chroma, and exposes searchable memory tools, it overlaps with long-term project memory. The important limitation is different: it is still a memory/retrieval layer, not authoritative lifecycle state.

The useful classification is:

| System | Best classification | Why |
| --- | --- | --- |
| `todo.md`, `findings.md` | Mid-term working memory | Agent-readable notes for the current task or near-future continuation |
| Harness `state.json` / DB row | Lifecycle state | Authoritative task phase, attempts, validation, approval, artifact status |
| Claude Code `CLAUDE.md` | Persistent instruction memory | User/team-authored rules loaded as context |
| Claude Code auto memory / `MEMORY.md` | Mid-term to project memory | Claude-authored notes and indexes shared across sessions for a repo |
| `claude-mem` | Mid-term plus searchable project memory | Captures observations, summarizes, stores, and retrieves relevant context |

This tier is valuable because it is simple to start and easy to inspect. It does, however, have limits. Forward-injected summaries can become lossy or stale. Searchable memory can over-retrieve or retrieve outdated facts. Lifecycle state can become dangerous if the model can edit it without validation. The harness should therefore keep provenance, timestamps, source links, and policy boundaries around all memory writes.

## 6. Long-Term: Persistent Memory Systems

Long-term memory systems store information across many sessions and retrieve it later by query. This tier is where the agent's experience can accumulate over time.

The standard loop is:

```text
write:   extract and store useful memory
read:    retrieve memories relevant to the current task
manage:  update, merge, decay, delete, or reorganize memories
```

### 6.1 MemGPT: Memory as Virtual Context

MemGPT made the operating-system analogy explicit. It treats the model's context window like RAM and external storage like disk. The agent can page information in and out, giving the illusion of a larger working memory than the physical context window.

That framing is powerful because it separates two ideas:

| Concept | Meaning |
| --- | --- |
| Physical context | Tokens actually visible to the model now |
| Virtual context | Larger external memory the agent can access through tools |

Harness engineering lives in the gap between the two.

### 6.2 Generative Agents: Observation, Reflection, Retrieval

Park et al.'s generative agents introduced a memory-stream architecture for social simulation agents. The system stores observations as natural-language records, scores them, retrieves relevant memories, and periodically reflects over them to synthesize higher-level conclusions.

The durable pattern is:

```text
observe -> store event
retrieve -> surface relevant memories
reflect -> synthesize higher-level insight
plan -> act using current state plus memory
```

This became a template for later memory systems because it recognizes that memory is not only recall. A useful agent also consolidates experience into abstractions.

### 6.3 MemoryBank: Forgetting and User Modeling

MemoryBank adds two ideas that matter in production:

1. Memories should decay or strengthen over time.
2. Long-term interaction should build a user model, not just a pile of facts.

Its forgetting mechanism is inspired by the Ebbinghaus forgetting curve: memories weaken with time, can be reinforced through access, and can be updated as new information arrives.

This matters because persistent memory can become a liability. A memory store that never forgets will accumulate stale preferences, contradicted facts, and irrelevant observations.

### 6.4 Mem0: Extract, Store, Retrieve, Manage

Mem0 is a useful concrete example of a long-term memory layer because it turns memory into explicit API operations rather than relying on transcript replay. Its docs describe the core loop as `add`, `search`, `update`, and `delete`, with both managed Platform and open-source deployments sharing the same mental model.

The write path starts with conversation messages or facts:

```text
messages / fact
  -> memory extraction by an LLM
  -> scoped memory records with metadata
  -> vector / keyword / optional graph stores
```

In code, the app explicitly writes memory when something durable has been learned:

```python
from mem0 import MemoryClient

client = MemoryClient(api_key="...")

client.add(
    messages=[
        {"role": "user", "content": "I prefer SQLite examples for demos."},
        {"role": "assistant", "content": "Got it. I will use SQLite examples."},
    ],
    user_id="alice",
    metadata={"category": "developer_preferences"},
)
```

Mem0's `add` flow can infer structured memories from messages, or store raw payloads when inference is disabled. For most agent systems, inferred memories are more useful than raw transcripts because they compress the interaction into retrievable facts, preferences, decisions, or goals.

The read path is a scoped retrieval pipeline:

```text
natural-language query
  -> query processing
  -> embedding search
  -> filters by user / agent / app / run / metadata
  -> optional reranking or thresholds
  -> memories returned to the agent
```

Example:

```python
results = client.search(
    "What demo database does Alice prefer?",
    filters={"user_id": "alice"},
)
```

The scope is not optional engineering hygiene. A memory store without `user_id`, `agent_id`, `app_id`, or `run_id` boundaries can leak irrelevant or wrong memories across users, agents, or workflows.

For agent harnesses, Mem0 can be placed behind a memory tool:

```text
agent asks: "What do I know about this user's SQL preferences?"
  -> memory_search(query, user_id, app_id)
  -> selected memories injected into current context
  -> model answers or acts with retrieved preferences
```

Mem0 also illustrates that long-term memory is not only vector search. Its documented features include metadata filtering, reranking, temporal reasoning, memory decay, custom categories, and graph memory. Graph memory adds relationship structure around entities, which helps when a question depends on links between people, projects, preferences, or events rather than one isolated fact.

For example:

```text
Alice -> prefers -> SQLite demos
Alice -> works_on -> text-to-SQL agent
text-to-SQL agent -> uses -> Chinook database
```

A graph-enhanced memory layer can retrieve across these relationships instead of relying only on text similarity.

The lifecycle distinction still applies. Mem0 can remember that Alice prefers SQLite examples or that a project usually validates with `pytest tests/test_cli.py -q`. It should not be the source of truth for whether the current run's tests passed. That remains lifecycle state owned by the harness.

Mem0 can also be attached directly to coding agents. Its Claude Code integration has two modes:

| Mode | What you get | What is missing |
| --- | --- | --- |
| MCP-only | Memory tools such as `add_memory`, `search_memories`, `get_memories`, `update_memory`, and `delete_memory` | No lifecycle hooks or SDK skill; the agent must call tools explicitly |
| Full plugin | MCP server, lifecycle hooks, and Mem0 SDK skill | Requires plugin install and Mem0 API key |

This makes Mem0 similar to `claude-mem` at the integration shape, but not identical:

| System | Integration with Claude Code | Backend shape | Typical memory scope |
| --- | --- | --- | --- |
| `claude-mem` | Claude Code hooks capture prompts/tool use and inject context; local worker handles storage/search | Local worker, SQLite/FTS5, optional Chroma | Project/session continuity from observed Claude Code activity |
| Mem0 MCP-only | Claude Code gets explicit memory tools over MCP | Mem0 Platform or self-hosted Mem0 | Agent-controlled save/search/update/delete operations |
| Mem0 full plugin | MCP tools plus lifecycle hooks and skill | Mem0 Platform memory layer | Automatic and tool-driven memory across Claude Code sessions |

So the design choice is not "Mem0 or hooks." Mem0 can be used as a plain memory API, as an MCP memory toolset, or through a Claude Code plugin with lifecycle hooks. `claude-mem` is more specifically a hook-triggered local recorder for Claude Code sessions.

### 6.5 A-MEM, Hindsight, and Structured Memory

Recent systems move from "store snippets and retrieve top-k" toward more structured memory.

| System | Main idea |
| --- | --- |
| A-MEM | Use Zettelkasten-style notes with keywords, tags, contextual descriptions, links, and evolving memory networks |
| Hindsight | Treat memory as a structured reasoning substrate with retain, recall, and reflect operations |

Mem0's paper reports better accuracy on LOCOMO than several memory baselines while reducing token cost compared with full-context approaches. A-MEM emphasizes dynamic linking and memory evolution: a new memory can update the interpretation of older memories. Hindsight separates memory into logical networks and reports strong results on LongMemEval and LoCoMo.

The common direction is clear: long-term memory is becoming less like a vector search cache and more like an actively managed knowledge system.

### 6.6 Shared and Project Memory

Some memory is personal. Some belongs to a project or organization.

Coding-agent harnesses increasingly expose project memory as files or repository artifacts:

| Memory scope | Example |
| --- | --- |
| User memory | Preferences, writing style, recurring goals |
| Project memory | Architecture, conventions, validation commands |
| Task memory | Current plan, findings, open issues |
| Team memory | Shared standards and lessons learned |

Trellis's repo-local specs and journals are an example of project memory. ECC and awesome Claude Code style repositories show a broader ecosystem of skills, hooks, and memory conventions for coding agents. Context Space frames MCP-style integrations, tool discovery, memory, and context optimization as context-engineering infrastructure.

The open problem is governance. Shared memory needs provenance, ownership, expiry, and conflict resolution. Otherwise it becomes another source of stale context.

## 7. Long-Horizon Techniques

Long-running agents need all three tiers at once. A large migration, multi-session research project, or autonomous coding task cannot rely only on the active context window.

### 7.1 Open-Source Long-Horizon Agent Projects

There is no single "long-horizon agent framework" that removes harness design. Open-source projects usually cover one layer of the stack:

| Project | Layer | What it gives you | What still belongs to your harness |
| --- | --- | --- | --- |
| LangGraph | Durable agent workflow runtime | Stateful graphs, persistence, interrupts, retries, human-in-the-loop control, long-running execution | Domain tools, product-specific state, approval policy, sandboxing, evaluation |
| OpenHands | Software-agent harness and SDK | Coding-agent loop, workspace interaction, shell/browser/file tools, local and cloud execution paths | Your product workflow, memory policy, repo-specific validation, deployment governance |
| AutoGen | Multi-agent conversation framework | AgentChat-style agents, group chat patterns, agent-as-tool composition, model and tool extensions | Lifecycle state, durable memory, security boundary, production authentication and policy |
| CrewAI | Multi-agent role/task framework | Crews for agent collaboration, Flows for controlled event-driven workflows, tasks, tools, memory, guardrails | Authoritative state, persistence strategy, side-effect approval, tests, observability |
| Mem0 | Long-term memory layer | Memory add/search/update/delete, metadata, graph memory, MCP and Claude Code integration options | Orchestration, phase control, lifecycle state, validation |
| Temporal or Prefect | Workflow engine | Durable jobs, retries, schedules, resumability, operational visibility | LLM loop, tool schemas, prompt/context policy, agent-specific reasoning |

AutoGen and CrewAI do implement a lot of the multi-agent platform for you, but at the agent/team orchestration layer. They can decide which specialist agent runs, pass messages between agents, run tools, and represent higher-level workflows. They do not remove the need for a harness to decide what state is authoritative, when to compact, which tool calls are allowed, how side effects are approved, how failures resume, and how success is verified.

Their open-source status is also nuanced:

| Project | Open-source status | Practical reading |
| --- | --- | --- |
| AutoGen | Open-source repository; code under MIT, docs/assets under CC-BY-4.0. Microsoft now marks it maintenance mode and recommends Microsoft Agent Framework for new projects. | Good reference for multi-agent patterns and existing AutoGen users, but not the default greenfield choice if you want active Microsoft roadmap support. |
| CrewAI | Core framework is open source under MIT. CrewAI also has commercial/hosted platform products. | The framework is usable as open-source infrastructure, but "CrewAI platform" can mean more than the OSS package. |
| OpenHands | Core work is MIT-licensed; its docs distinguish source-available enterprise pieces. | Strong open-source reference for coding-agent harness design, with commercial cloud/enterprise options around it. |
| LangGraph | Open-source library with managed platform offerings. | Use the OSS runtime when you want control; use managed infrastructure when operations matter more than self-hosting. |

So the practical answer is: use frameworks where they buy down complexity, but be precise about the layer. LangGraph buys durable state machines. AutoGen and CrewAI buy multi-agent composition. OpenHands buys a concrete coding-agent harness. Mem0 buys memory operations. None of them makes context policy, lifecycle state, tool safety, and evaluation disappear.

### 7.2 Compaction

Compaction summarizes a full or noisy context window and starts a fresh window with the compressed state.

The safe direction is recall first, precision second:

1. Preserve everything that might be needed later.
2. Test whether the agent can continue correctly.
3. Remove redundant content only after continuity is reliable.

A compaction summary should preserve:

| Keep | Drop or compress |
| --- | --- |
| User goal and constraints | Raw repeated shell output |
| Architectural decisions | Superseded reasoning |
| Files changed and why | Full file contents already summarized |
| Unresolved bugs | Tool outputs already acted on |
| Validation evidence | Stale search results |
| Next steps | Duplicate conversation turns |

Tool-result clearing is the lightest version: after the agent has used a large tool output, replace it with a compact reference.

```text
Before:
  full 4,000-line test log in context

After:
  "pytest failed in tests/test_cli.py::test_config_path.
   Full log is available at .agent/logs/pytest-2026-05-27.txt."
```

The context gets smaller, but the evidence is still available.

### 7.3 Sub-Agent Context Isolation

Subagents are also a context-management tool.

If a subtask requires large exploration, the main agent should not have to carry every intermediate search. A subagent can explore with its own clean window and return a compact result.

```text
orchestrator context
  - user goal
  - current plan
  - high-level state
        |
        +--> subagent context
             - focused task
             - local searches
             - large intermediate evidence
             - condensed result
        |
        v
orchestrator receives summary, artifact path, or patch
```

Anthropic's multi-agent research system uses this pattern for breadth-first research. Subagents act as independent filters that search and condense information before returning findings to the lead agent.

This is not always worth it. Anthropic notes that multi-agent systems use many more tokens than ordinary chats, and coding tasks often have more dependency between steps than open-ended research. The harness should use subagents when isolation and parallelism justify the coordination cost.

### 7.4 Hybrid Decision Framework

A practical context controller uses several techniques together:

| Situation | Technique |
| --- | --- |
| Always-needed instructions | Preload stable prompt and policy |
| Potentially relevant files | Keep paths and retrieve just in time |
| Large tool results | Store externally and summarize |
| Window approaching limit | Compact |
| Deep independent exploration | Spawn subagent |
| Session reset or crash | Reload notes and lifecycle state |
| Repeated user/project facts | Retrieve from persistent memory |

The important design choice is that context management should be a harness responsibility. The model can help write notes or summaries, but the harness should decide when to compact, where to store artifacts, which memory records to retrieve, and which state is authoritative.

## 8. Context Drift

Context rot is a single-call problem: the input gets longer and the model uses it worse.

Context drift is a trajectory problem: over many turns, the agent's behavior moves away from the original task.

Common symptoms:

| Symptom | Example |
| --- | --- |
| Repetition | The agent re-investigates something already resolved |
| Contradiction | It reverses an earlier decision without noticing |
| Goal loss | It optimizes a subtask while forgetting the user request |
| Summary error | A compaction slightly changes meaning, then later steps build on the wrong version |
| Retrieval failure | The agent does not know what memory it needs to ask for |

Current techniques reduce drift but do not eliminate it.

| Technique | What it helps | Remaining weakness |
| --- | --- | --- |
| Compaction | Keeps the active window bounded | Summaries are lossy |
| Retrieval | Recovers older information | Query must be well formed |
| Notes | Preserves explicit state | Notes can be incomplete or stale |
| Subagents | Isolates exploratory clutter | Orchestrator can still drift |
| Long-term memory | Supports recall across sessions | Memory may be wrong, stale, or over-retrieved |

This is where context engineering connects back to the rest of the harness. Long-horizon reliability needs verification loops, lifecycle checkpoints, observability, and governance. The context layer can decide what the model sees. It cannot by itself prove that the agent is still solving the right task.

Recent benchmarks such as MemBench and MemoryArena are useful because they move beyond simple recall. MemoryArena, for example, evaluates agents across interdependent multi-session tasks where memories from earlier sessions must guide later decisions. That is closer to real agent work than a single long-context lookup.

## 9. Design Lessons

The practical conclusions for the context and memory layer are:

1. **Bigger windows do not remove context engineering.** Attention cost, position sensitivity, and context rot still make selection and placement matter.
2. **Context is a projection, not a dump.** The harness should expose the smallest high-signal subset of state needed for the next model step.
3. **Tool outputs should not accumulate unchecked.** Store large evidence externally, keep compact references, and reload details only when needed.
4. **Memory needs tiers.** Active context, session notes, and persistent indexed memory solve different problems.
5. **Compaction should preserve recall before optimizing brevity.** Lost state is hard to recover after the agent has moved on.
6. **Subagents are context isolation, not just parallelism.** Their value is often that they keep exploratory context out of the orchestrator.
7. **Persistent memory needs management.** Write, retrieve, update, decay, and delete are all part of the system.
8. **Context drift is not solved by context alone.** Long-horizon agents need checkpoints, validation, traces, and sometimes human review.

Part 1 answered: **where can the agent safely act?** Part 2 answered: **what can it call, and through which protocol boundary?** Part 3 answered: **how does repeated action become a completed workflow?** This post answers: **what should the model know right now, and what should survive outside the window?**

## References

- [Agent Harness Engineering: A Survey of LLM Infrastructure](https://picrew.github.io/LLM-Harness/main.pdf)
- [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)
- [Managing context on the Claude Developer Platform](https://claude.com/blog/context-management)
- [Context Engineering for AI Agents: Lessons from Building Manus](https://manus.im/blog/Context-Engineering-for-AI-Agents-Lessons-from-Building-Manus)
- [vLLM: Automatic Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)
- [vLLM: Custom Logits Processors](https://docs.vllm.ai/en/latest/examples/features/logits_processor/)
- [How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system)
- [Building Effective AI Agents](https://www.anthropic.com/engineering/building-effective-agents)
- [LangGraph Overview](https://docs.langchain.com/oss/python/langgraph/overview)
- [OpenHands Introduction](https://docs.openhands.dev/overview/introduction)
- [AutoGen](https://github.com/microsoft/autogen)
- [CrewAI](https://github.com/crewAIInc/crewAI)
- [Temporal Docs](https://docs.temporal.io/)
- [Prefect Introduction](https://docs.prefect.io/v3/get-started/index)
- [Claude Code: How Claude remembers your project](https://code.claude.com/docs/en/memory)
- [Context Rot: How Increasing Input Tokens Impacts LLM Performance](https://www.trychroma.com/research/context-rot)
- [Lost in the Middle: How Language Models Use Long Contexts](https://arxiv.org/abs/2307.03172)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [MemGPT: Towards LLMs as Operating Systems](https://arxiv.org/abs/2310.08560)
- [Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442)
- [MemoryBank: Enhancing Large Language Models with Long-Term Memory](https://arxiv.org/abs/2305.10250)
- [Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory](https://arxiv.org/abs/2504.19413)
- [Mem0 Docs: Add Memory](https://docs.mem0.ai/core-concepts/memory-operations/add)
- [Mem0 Docs: Search Memory](https://docs.mem0.ai/core-concepts/memory-operations/search)
- [Mem0 Docs: Graph Memory](https://docs.mem0.ai/platform/features/graph-memory)
- [Mem0 Docs: Claude Code Integration](https://docs.mem0.ai/integrations/claude-code)
- [Mem0 Docs: MCP Server](https://docs.mem0.ai/platform/mem0-mcp)
- [A-MEM: Agentic Memory for LLM Agents](https://arxiv.org/abs/2502.12110)
- [Hindsight is 20/20: Building Agent Memory that Retains, Recalls, and Reflects](https://arxiv.org/abs/2512.12818)
- [MemBench: Towards More Comprehensive Evaluation on the Memory of LLM-based Agents](https://arxiv.org/abs/2506.21605)
- [MemoryArena: Benchmarking Agent Memory in Interdependent Multi-Session Agentic Tasks](https://arxiv.org/abs/2602.16313)
- [Planning with Files: Persistent File-Based Planning for AI Coding Agents](https://github.com/OthmanAdi/planning-with-files)
- [Trellis: An Out-of-the-Box Engineering Framework for AI Coding](https://github.com/mindfold-ai/trellis)
- [Claude-Mem: Persistent Context Across Sessions for Every Agent](https://github.com/thedotmack/claude-mem)
- [Claude-Mem Docs: Persistent Memory Compression System for Claude Code](https://docs.claude-mem.ai/)
- [ECC: Agent Harness Performance Optimization System](https://github.com/affaan-m/ECC)
- [Context Space: Context Engineering Infrastructure for MCPs and Integrations](https://github.com/context-space/context-space)
