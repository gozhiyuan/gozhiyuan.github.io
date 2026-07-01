---
layout: post
title: "Agent Harness Engineering Part 3: Lifecycle and Orchestration"
subtitle: Agent loops, subagents, state, and issue-to-pull-request workflows
categories: Large-Language-Model Agents Harness-Engineering
tags: [Blogs]
banner: "/assets/images/banners/yuanpang-wa-iceburg2.jpg"
---

# Agent Harness Engineering Part 3: Lifecycle and Orchestration

This post is the third part of my notes on [Agent Harness Engineering: A Survey of LLM Infrastructure](https://picrew.github.io/LLM-Harness/main.pdf). [Part 1]({% post_url 2026-05-23-harness-engineering-01 %}) covered **E: Execution Environment and Sandbox**. [Part 2]({% post_url 2026-05-23-harness-engineering-02 %}) covered **T: Tool Interface and Protocol Layer**.

This post focuses on **L: Lifecycle and Orchestration** from section 6 of the paper: how the harness carries a task across calls, state, failure, delegation, validation, and delivery.

The orchestration layer is closely connected to tools, but it is not the same thing. A tool interface gives an agent verbs such as "read file", "query issue tracker", or "run test". Orchestration decides which actor should use which verb, in what order, with which state, until the job is complete.

I will leave the paper's section 5, **C: Context and Memory Management**, for a separate post. It is adjacent to lifecycle state, so this post briefly separates the two concepts where necessary.

## 1. From Sandboxed Actions to Useful Work

Part 1 described where actions run. Part 2 described what operations an agent can call. Lifecycle and orchestration answer a different question: **how does a task move from request to verified completion?**

The clean separation is:

| Layer | Owns | Main question |
| --- | --- | --- |
| Tool layer | Callable operations and their schemas | What can be called? |
| Lifecycle | Durable task state and phase boundaries | Where is this task in its journey? |
| Orchestration | Control decisions that advance the lifecycle | What should happen next? |

Lifecycle and orchestration are related to the agent loop, but they are not identical to it. The agent loop is the repeated inner mechanism:

```text
build context -> model proposes action -> execute action -> observe result -> repeat
```

Lifecycle and orchestration sit around and across those loop iterations:

```text
task state -> choose phase / actor / context / capabilities
           -> run one or more agent-loop steps
           -> record result
           -> decide next transition
```

In other words, the agent loop is an execution pattern. Lifecycle is the durable task record that survives across loop iterations. Orchestration is the control policy that reads that record and decides which loop, actor, phase, context, tool set, retry, delegation, approval, or completion transition should happen next.

In this framing, **lifecycle** is the task's operational record. It is the durable structure that says: this request was accepted, this workspace or branch was assigned, this phase is active, these attempts have already happened, this validation passed or failed, this approval is still missing, and this artifact is the final output.

**Orchestration** is the control logic that reads that lifecycle state and moves it forward. It decides whether to continue investigation, retry a failed step, delegate a subtask, pause for approval, run verification, publish an artifact, or stop because the budget is exhausted.

The distinction is easiest to see without focusing on any particular tool:

```text
User request
  |
  v
Lifecycle creates a task record
  - requirement
  - workspace / branch / environment identity
  - current phase
  - budgets and approval requirements
  - known evidence and artifacts
  |
  v
Orchestrator advances the task
  - choose next phase
  - choose responsible actor
  - choose what context and capabilities are active
  - interpret results
  - update lifecycle state
  - decide continue / retry / delegate / pause / finish
  |
  v
Verified artifact or stopped task
```

So lifecycle is mostly about **state and allowed transitions**. Orchestration is mostly about **decision and coordination**. As an overview, this layer tracks or decides:

| Area | Lifecycle records | Orchestration decides |
| --- | --- | --- |
| Task | requirement, owner, workspace, branch, environment | whether the task is actionable or needs clarification |
| Phase | `intake`, `setup`, `investigate`, `implement`, `verify`, `publish`, `done` | which phase should run now and when to transition |
| Actor | active agent, pending subagents, completed subtasks | whether to use the main agent, reviewer, specialist, or remote agent |
| Context | durable evidence, relevant artifacts, summarized prior results | what context should be loaded for the next loop step |
| Capability scope | allowed tools, approval requirements, policy status | which capabilities are visible or gated in this phase |
| Attempts | commands/actions tried, failures, retries, budgets | whether to retry, change strategy, delegate, pause, or stop |
| Verification | validation command, revision, result, review findings | whether evidence is sufficient for delivery |
| Completion | final artifact, PR URL or report, residual risk | whether to finish, publish, or return for more work |

Tools still appear in this story, but they are not the subject of this layer. A test runner, file editor, browser, or issue-tracker API is an operation. Lifecycle and orchestration determine **why that operation is being used now**, **what state changes after it returns**, and **whether the overall task should continue**.

For example, consider a coding task that eventually opens a pull request. The lifecycle might be:

```text
intake -> setup -> investigate -> implement -> verify -> publish -> done
```

The orchestrator is the component that enforces the movement between those states:

```text
if requirement is ambiguous:
    phase = "needs_clarification"
elif no reproduction or evidence exists:
    phase = "investigate"
elif candidate change exists but validation is missing:
    phase = "verify"
elif validation passed and approval exists:
    phase = "publish"
else:
    continue_current_phase_or_retry()
```

The important point is that the lifecycle state must survive beyond the model's immediate context window. A model can say "the tests passed," but the harness should record which validation ran, on which revision, with what result, and whether that evidence is sufficient for the next transition.

That is why a list of tools is not a workflow. A workflow needs durable state, transition rules, retry policy, delegation boundaries, approval gates, and completion criteria. Those are the lifecycle and orchestration concerns this post focuses on.

## 2. Lifecycle State Is Not Model Context

Before discussing orchestration, it helps to separate **context** from **lifecycle state**.

| Concern | Meaning | Example |
| --- | --- | --- |
| Context and memory, section 5 | Information selected for the model to reason over now | Relevant source snippets, summarized prior decisions, tool descriptions |
| Lifecycle state, section 6 | Operational truth the harness maintains to continue and coordinate work | Task status, pending subtask IDs, current branch, test outcome, retry count, artifact locations |
| Observability, section 7 | Record used to inspect behavior and performance | Span log of each call, latency, tokens, approvals, failures |

A task runner should not rely on a model remembering "tests passed on commit X" only because the statement appeared somewhere in conversation history. It should persist that operational fact in structured task state and decide whether to expose it to the model when needed.

Lifecycle state is partly predefined status and partly accumulated runtime evidence. The predefined part is the phase model, such as `investigate`, `implement`, `verify`, and `publish`. The accumulated part is everything learned while the task runs: which workspace was created, what files changed, which validation commands ran, which subtasks are pending, which approvals were granted, and where the final artifact lives.

A minimal state object for a coding workflow might be:

```python
from dataclasses import dataclass, field


@dataclass
class CodingTaskState:
    issue_url: str
    workspace_id: str
    branch: str
    phase: str = "investigate"
    attempt_count: int = 0
    completed_subtasks: list[str] = field(default_factory=list)
    pending_subtasks: list[str] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    last_test_command: str | None = None
    last_test_passed: bool = False
    validation_revision: str | None = None
    approval_to_publish: bool = False
    pull_request_url: str | None = None
    residual_risk: str | None = None
```

The model should usually not receive the entire state object. It receives a **projection**: the subset or summary needed for the next decision. The orchestrator owns the authoritative copy.

```text
Authoritative lifecycle state
  - full task record
  - all attempts and observations
  - approval and artifact status
        |
        | selected and summarized by the orchestrator
        v
Model-facing context
  - current phase
  - relevant evidence
  - active constraints
  - available capabilities
```

This answers an important practical question: if the full lifecycle state is not just dumped into the prompt, how does the agent know whether the task is complete? The answer is that completion is shared between the model and the harness:

| Responsibility | Example |
| --- | --- |
| The model can propose completion | "The patch is ready; focused tests pass; open a PR." |
| The orchestrator validates completion | Check that validation was recorded on the current revision and publication approval exists |
| The lifecycle state records completion | `phase = "done"`, `pull_request_url = ...`, validation evidence and residual risk stored |

So the model may know the current status through a compact state summary, but it is not the authority that decides the status. A model can say "done." The harness decides whether "done" is allowed.

Lifecycle updates also do not only come from one synchronous model loop. In real systems, state changes often arrive as events:

| Event source | Example event | Lifecycle update |
| --- | --- | --- |
| Model action | Agent proposes a patch or final answer | Record proposed action and next expected evidence |
| Tool observation | Test command exits `0` or `1` | Record command, revision, result, and failure text |
| Subagent result | Reviewer returns blocking issue | Mark subtask complete and return phase to `implement` or `verify` |
| Human input | User approves publication | Set approval flag and allow transition to `publish` |
| External system | CI webhook reports failure after PR creation | Reopen verification or attach failing check to task state |
| Timeout or crash | Agent loop stops before completion | Persist stopped state so the task can resume or fail cleanly |

In an interactive product such as Claude Code, some of these events may appear through tool results, hooks, subagent results, or user approvals. In OpenHands, the same idea appears as persisted conversation state, actions, observations, workspaces, and sub-agent delegation. In skill-centered assistants such as OpenClaw, messages, skills, scripts, and external automations still need to feed back into a task record if the system is expected to recover, audit, or continue long-running work.

The durable state can therefore be updated by parsing structured tool results, interpreting model outputs, receiving subagent artifacts, handling user approvals, or consuming asynchronous external callbacks. The common pattern is the same: events enter the harness, the orchestrator interprets them, and the lifecycle record changes.

## 3. Single-Agent Inner Loop

The simplest lifecycle pattern is a single agent alternating between model decisions and tool observations. The paper links this to ReAct and classifies tools such as Claude Code, Codex CLI, Gemini CLI, Aider, OpenCode, and SWE-agent primarily under this pattern.

```text
User task
   |
   v
Build context from instruction, state projection, tools, and current workspace
   |
   v
Model proposes action
   |
   +--> respond/finalize
   |
   +--> call permitted tool -> obtain observation -> update state -> repeat
```

A simplified control loop is:

```python
def run_single_agent(task, agent, tools, policy, state, max_steps=50):
    for step in range(max_steps):
        available_tools = policy.allowed_tools(state, tools)
        action = agent.next_action(task=task, state=state, tools=available_tools)

        if action.type == "finish":
            return validate_completion(action, state)

        policy.authorize(action, state)
        observation = available_tools[action.tool].invoke(action.arguments)
        state.record(action, observation)

        if observation.requires_human_input:
            return pause_for_review(state)

    return fail_with_budget_exhausted(state)
```

At this level, an important execution distinction is **stateless replay** versus **hybrid execution**.

In a stateless replay loop, the next model call is reconstructed from replayable records:

```text
system/developer instructions
+ user task
+ message history
+ prior tool calls and observations
+ summaries or retrieved workspace facts
-> next model call
```

The agent may still edit files or run commands, but the model-facing reasoning state is rebuilt from recorded history rather than hidden live memory. If the process restarts, the harness can replay or summarize the transcript and continue from the external workspace.

Hybrid execution combines replayable history with persistent runtime artifacts:

| Persistent artifact | What the next model call may see |
| --- | --- |
| Edited files on disk | Relevant snippets, changed-file list, or diff summary |
| Git branch or repository | Current branch, base revision, changed hunks |
| Test logs | Pass/fail status, stack trace, or concise failure summary |
| Shell session | Last command/output or running process status |
| Browser session | Current URL, screenshot, DOM text, or page summary |
| MCP or API session | Tool result summary, server availability, active capabilities |
| Subagent workspace | Returned artifact, review finding, or mergeable diff |

The persistent state is not automatically pasted into every prompt. The harness chooses a **projection** of that state for the next model call. For example:

```text
Persistent workspace/session state
        |
        | inspect, retrieve, summarize, or filter
        v
Model-facing context
  - current phase: verify
  - changed file: parser.py
  - last focused test failed
  - relevant stack trace
  - active constraint: do not publish yet
```

Most practical coding agents are hybrid. They keep a replayable conversation or trajectory, but the real continuity also lives in files, repositories, shells, browser sessions, MCP sessions, CI runs, and subagent outputs. This makes them useful for real work, but it also creates a lifecycle problem: the harness must know which persistent facts are authoritative, which ones should be shown to the model, and which ones should trigger a phase transition.

This is small, but the surrounding harness decisions determine reliability:

- Are tool outputs truncated or summarized before entering context?
- Does a failed edit return actionable feedback?
- Are test results recorded structurally?
- Is there a step, token, time, or cost budget?
- Does "done" require running tests or displaying a diff?
- Can the session recover after a process crash?

A strong single-agent loop is often sufficient for tightly scoped coding tasks. Multi-agent orchestration is not automatically better; it creates additional coordination and cost.

## 4. Multi-Agent Orchestration Patterns

Multi-agent orchestration is useful when responsibilities can be separated: parallel research, independent review, specialized domains, or deliberate planning and verification. It is not automatically better than a single loop. The benefit comes from cleaner division of responsibility; the cost is coordination overhead.

Every multi-agent design has to answer four questions:

| Question | Why it matters |
| --- | --- |
| Who owns the final answer or artifact? | Prevents every worker from optimizing for a different definition of done |
| How do agents communicate? | Determines whether workers report only to a coordinator or talk to each other |
| Do agents share a workspace? | Determines whether sandbox/worktree isolation is needed |
| Which tools can each role use? | Prevents reviewer, researcher, or external specialist agents from receiving unnecessary authority |

| Pattern | Control shape | Good fit | Typical cost or failure |
| --- | --- | --- | --- |
| Hierarchical supervisor | One coordinator delegates to specialists and synthesizes results | Research plus implementation plus review | Coordinator becomes a bottleneck or misroutes tasks |
| Team orchestration | Named peers communicate or own different components | Large feature with separable ownership | Conflicting changes and communication overhead |
| Graph composition | Explicit nodes and transitions encode possible paths | Durable workflows, approvals, retries, conditional validation | Graph complexity grows with exceptions |
| Fan-out and aggregation | Multiple agents attempt or inspect in parallel, then a judge combines | Search, alternative patches, review coverage | High token/compute cost and hard aggregation |
| Deterministic workflow with agent steps | Code determines sequence; LLMs handle bounded steps | Regulated or repeatable business process | Less flexible for unexpected exploration |

### Pattern Diagrams

#### Hierarchical Supervisor

One coordinator owns the task and delegates bounded subtasks. Workers do not need to communicate with each other; they report back to the coordinator.

```text
User / task runner
      |
      v
Coordinator / supervisor
  - decomposes task
  - assigns work
  - receives results
  - owns final synthesis
      |
      +--> Researcher
      +--> Implementer
      +--> Reviewer
      |
      v
Final artifact or next phase
```

This is the most common shape for coding-agent delegation because it preserves one owner for the final patch, PR, or answer.

#### Team Orchestration

Multiple agents are peers. They may own different components and communicate through messages, a task board, or shared artifacts.

```text
Team lead / shared task list
      |
      +--> Frontend agent <---- messages ----+
      |                                      |
      +--> Backend agent  <---- messages ----+
      |                                      |
      +--> Test agent     <---- messages ----+
      |
      v
Integrated result after coordination
```

This is useful when components can be owned independently, but it needs conflict control: clear file ownership, worktree isolation, dependency ordering, and merge review.

#### Graph Composition

The application defines explicit nodes and transitions. Some nodes use LLM agents; other nodes are deterministic checks, approvals, or tool calls.

```text
START
  |
  v
investigate --> implement --> verify
                    ^          |
                    |          v
                    +-- fail --+
                               |
                               v
                         approval_gate --> publish --> END
```

Graph composition is less conversational but more durable. It is a good fit when approvals, retries, and checkpoints matter.

#### Fan-Out and Aggregation

Several agents work on the same question or inspect the same artifact from different perspectives. A judge or coordinator combines the results.

```text
Problem or artifact
      |
      +--> Agent A: security lens
      +--> Agent B: performance lens
      +--> Agent C: test coverage lens
      |
      v
Aggregator / judge
  - compare findings
  - remove duplicates
  - resolve conflicts
  - produce final report
```

This pattern is good for broad review and competing hypotheses, but it is expensive and aggregation can become its own hard problem.

#### Deterministic Workflow with Agent Steps

Code owns the process order. LLMs are called only inside bounded steps.

```text
for each task:
    classify requirement
    run LLM planner
    run deterministic setup
    run LLM implementer
    run deterministic tests
    run LLM reviewer
    require approval
    publish artifact
```

This pattern is useful when the organization wants repeatability more than free-form autonomy. The agent is inside a workflow, not the workflow itself.

### Planner, Worker, Reviewer

A coding-agent workflow can separate responsibility without building an elaborate organization:

```text
Coordinator/planner
  - reads issue and repository orientation
  - chooses independent subtasks
       |
       +--> Worker: implement patch in sandbox or worktree
       |
       +--> Researcher: inspect API/docs or reproduce bug
       |
       v
Reviewer/verifier
  - reads diff and requirement
  - runs or assesses tests
  - identifies unresolved risk
       |
       v
Coordinator: revise or prepare delivery
```

This is usually a **hierarchical supervisor** pattern: the coordinator owns decomposition and final synthesis, while the worker and reviewer are scoped specialists. It can also be implemented as a graph if the phases are fixed, for example `plan -> implement -> review -> revise -> publish`. It becomes team orchestration only when the worker and reviewer are independent peers that coordinate with each other rather than only reporting back to the coordinator.

The important improvement is not that three model calls are inherently smarter than one. It is that each role can receive a smaller tool set, cleaner context, and an explicit completion test.

| Role | Typical sandbox relationship | Typical tool scope |
| --- | --- | --- |
| Planner | Usually read-only in the main workspace | issue/docs/repo search, no write or publish tools |
| Worker | Isolated worktree or sandbox if it may edit | file edit, shell/test, local browser if needed |
| Reviewer | Read-only view of diff or separate worktree | diff inspection, test execution, CI query, no publish |
| Coordinator | Owns lifecycle state and delivery gate | can request approval and publish only after validation |

### Subagents as Tools Versus Handoffs

Two common patterns are easy to confuse:

| Pattern | Who retains control? | When useful |
| --- | --- | --- |
| Subagent as a tool | The coordinator invokes a specialist and receives its result, remaining responsible for the final response | A scoped investigation or review that supports one main task |
| Handoff | Control of the conversation or task transfers to another agent | Routing a request to a specialist who should interact directly for subsequent turns |

Both may be implemented with ordinary function tools inside one application, with graph nodes, or through a remote protocol such as A2A when the specialist is independently deployed.

### How This Relates to Sandboxes and Tools

Multi-agent orchestration does not replace Part 1 or Part 2. It composes them.

| Harness layer | Multi-agent implication |
| --- | --- |
| Execution and sandbox, Part 1 | Each agent may share a workspace, receive a separate worktree, or run in a remote sandbox. The more agents can write, the more isolation matters. |
| Tools and protocols, Part 2 | Each role should receive only the tools it needs. A reviewer may need diff and test tools, not publish authority. A remote specialist may be reached through A2A, while its internal tools may be MCP-backed. |
| Lifecycle and orchestration, this post | The coordinator or workflow records subtasks, pending results, approvals, validation evidence, and merge/delivery decisions. |

A multi-agent coding workflow therefore has two resource-scoping axes:

```text
Role scope:      planner / worker / reviewer / remote specialist
Execution scope: shared workspace / isolated worktree / remote sandbox
Tool scope:      read-only / edit-test / review / publish / external system
```

Good orchestration keeps those scopes aligned. A planner with publish tools is too powerful. A worker editing the same files as another worker without isolation is fragile. A reviewer with only the model's summary and no diff/test access is weak.

The concrete products later in this post can be read through these patterns: Claude Code subagents look like scoped hierarchical delegation, OpenHands exposes sub-agent delegation through a tool-like boundary, LangGraph emphasizes graph composition, and ADK exposes deterministic workflow agents alongside dynamic routing.

## 5. From Issue to Pull Request

The full lifecycle pipeline is where tools and orchestration stop being abstractions and become software delivery. A coding agent should not be evaluated only by whether it can produce an edit. It must handle a durable task from requirement to reviewable output.

```text
Issue or user specification
        |
        v
Acquire workspace and branch
        |
        v
Investigate repository and reproduce problem
        |
        v
Plan and optionally delegate subtasks
        |
        v
Edit inside bounded execution environment
        |
        v
Run focused tests, then broader relevant validation
        |
        v
Review diff and summarize residual risk
        |
        v
Human approval or delivery policy gate
        |
        v
Commit, push branch, and open/update pull request
```

Each transition needs a state record and often a different set of tools:

| Phase | Relevant tools | Durable state or artifact |
| --- | --- | --- |
| Intake | issue tracker read, repository metadata | issue ID, acceptance criteria |
| Setup | sandbox/worktree creator, Git operations | workspace ID, base commit, branch |
| Investigation | read/search, shell, logs, documentation MCP | reproduction command and observed failure |
| Implementation | edit/apply patch, shell/test tools | diff and changed files |
| Verification | test runner, CI query, reviewer agent | validation results and review findings |
| Delivery | GitHub/GitLab PR tools after policy check | commit, branch, PR URL |

The output of a complete harness is not "the model said it fixed the bug." It is a traceable artifact associated with tests, policy decisions, and a delivery destination.

## 6. How Real Systems Use Tools and Orchestration

The products and frameworks below operate at different levels. Comparing them as if they were direct substitutes would be misleading.

| System | Architectural level | Tool approach | Orchestration approach | Best reason to study it here |
| --- | --- | --- | --- | --- |
| Claude Code | Integrated coding-agent product | Built-in development tools, permissions, hooks, MCP extensions | Single-agent loop plus subagents and broader agent-team features | Shows a product harness that directly edits and tests a repository |
| OpenHands | Open-source software-agent platform and SDK | Typed tools for Bash, editing, browser, MCP, workspaces | Conversation/agent loop plus sub-agent delegation | Shows an inspectable open-source coding harness |
| LangChain | High-level agent/component library | Tool abstractions and MCP adapters | Agent abstractions commonly built on LangGraph | Shows how tools enter an application framework |
| LangGraph | Low-level orchestration runtime | Uses tools/components supplied by the application | Stateful graphs, checkpoints, interrupts, subgraphs and multi-agent patterns | Shows explicit durable orchestration |
| OpenAI Agents SDK | Open-source application SDK | Function tools, hosted tools, MCP-backed tools | Manager-as-tool and handoff patterns, sessions, tracing | Shows compact multi-agent primitives in an SDK |
| Google ADK | Open-source agent development kit | Function and MCP toolsets | LLM routing, deterministic sequential/parallel/loop workflow agents, A2A integration | Shows MCP and A2A used at different boundaries |

### Common Tool Choices

For building real applications, the common tools fall into a few practical categories. They are often discussed together, but they solve different parts of the harness:

| Category | Common choices | What they are best at | What they do not remove |
| --- | --- | --- | --- |
| OpenAI-native agent SDK | OpenAI Agents SDK | Agent loop, function tools, hosted tools, MCP tools, handoffs, sessions, guardrails, tracing | Application-specific workflow design, domain validation, product lifecycle |
| Graph orchestration | LangGraph | Explicit state machines, checkpoints, retries, interrupts, subgraphs, durable multi-step workflows | Tool design, prompt quality, sandboxing, final artifact policy |
| High-level agent framework | LangChain | Model/tool abstractions, integrations, middleware, prebuilt agent construction | Low-level durable orchestration unless paired with LangGraph |
| Google/Gemini stack | Google ADK | Gemini-oriented agents, deterministic workflow agents, MCP tools, A2A remote-agent integration | Non-Google deployment choices and domain-specific safety logic |
| Role/task team frameworks | CrewAI, AutoGen | Fast prototyping of crews, conversational teams, role-based delegation, sequential or hierarchical work | Deterministic lifecycle gates, repository isolation, artifact verification |
| Microsoft application stack | Semantic Kernel Agent Framework, AutoGen | .NET/Python application integration, plugins, process orchestration, multi-agent collaboration | Product-specific task state, approval policy, environment hardening |
| Data/RAG-heavy agents | LlamaIndex | Retrieval, indexing, query engines, document/RAG workflows, data-backed multi-agent patterns | General-purpose coding workflow control |
| Protocol boundaries | MCP, A2A | MCP connects agents to tools and data; A2A connects independently deployed agents | The orchestrator that decides when to call which tool or remote agent |

The most important distinction is between an **agent runtime** and a **workflow runtime**.

An agent runtime, such as the OpenAI Agents SDK or the high-level LangChain agent API, usually gives you the inner loop:

```text
call model -> receive tool calls or final output
           -> execute tools
           -> append observations
           -> continue until final output or turn limit
```

A workflow runtime, such as LangGraph or ADK workflow agents, makes the outer lifecycle explicit:

```text
intake -> plan -> act -> validate -> retry or escalate -> publish
```

For a simple text-to-SQL application, an agent runtime may be enough: expose schema lookup and query execution as tools, require structured SQL output, validate SQL with a parser, and run a repair loop when execution fails. For a coding-agent product or a production automation, the outer workflow usually matters more: branch setup, approval gates, test evidence, retries, and delivery should be first-class state rather than hidden conversation text.

### Best Practices for Single-Agent Systems

A single-agent design should be the default baseline. It is easier to observe, cheaper to run, and less prone to coordination errors. A strong single-agent harness usually has:

| Practice | Why it matters |
| --- | --- |
| Keep the tool menu small and phase-appropriate | Fewer actions reduce routing mistakes and prompt bloat |
| Use structured outputs for decisions | The harness can validate `action`, `sql`, `patch_status`, or `next_step` instead of parsing prose |
| Put domain safety in code | SQL read-only checks, path allowlists, command policy, and schema validation should not depend only on prompts |
| Separate model memory from operational state | Conversation context can be summarized; task status, retries, approvals, and artifacts should be durable |
| Prefer deterministic gates around risky effects | Run validation before execution, approval before external writes, and tests before publish |
| Record traces and evidence | A useful result includes what tools ran, what changed, which validation passed, and what remains risky |

For example, a single-agent text-to-SQL loop should not trust the model's "safe query" claim. The harness should parse the SQL AST, reject non-read operations, cap rows, execute in a constrained database connection, and only then summarize results.

### Best Practices for Multi-Agent Systems

Multi-agent orchestration should be introduced only when it buys something concrete: parallelism, isolated context, specialized expertise, independent review, or integration with a separately operated service.

Good multi-agent harnesses make the following boundaries explicit:

| Boundary | Best practice |
| --- | --- |
| Ownership | One coordinator owns the final artifact and definition of done |
| Work scope | Give each worker a bounded task, expected output shape, and stop condition |
| Tool authority | Give specialists only the tools required for their role |
| Workspace | Use separate worktrees or sandboxes for parallel editing agents |
| Communication | Prefer structured reports or artifacts over long free-form chat between agents |
| Integration | Merge or synthesize through the coordinator, not through unmanaged peer edits |
| Verification | Use an independent reviewer or deterministic tests before publishing |
| Budget | Put turn, time, token, and cost limits on each delegated subtask |

A good rule is: **delegate responsibilities, not confusion**. If the main agent cannot state what the subagent should do, what files or systems it may touch, and how success will be judged, delegation usually adds noise.

The common multi-agent patterns map naturally to tool choices:

| Need | Practical tool choice |
| --- | --- |
| One assistant with tools and occasional specialist calls | OpenAI Agents SDK manager-as-tool pattern, LangChain agent, or CrewAI hierarchical process |
| Explicit retry/approval/checkpoint flow | LangGraph or ADK workflow agents |
| Independent remote specialist owned by another service/team | A2A, with local lifecycle state tracking task status and returned artifacts |
| Tool/data integration across many systems | MCP servers exposed to whichever agent runtime is used |
| RAG-heavy domain specialists | LlamaIndex tools or agents embedded under a coordinator |
| Microsoft/.NET enterprise app integration | Semantic Kernel Agent Framework or AutoGen |

In practice, many production systems combine these layers: an outer LangGraph or ADK workflow owns the lifecycle; inner OpenAI Agents SDK, LangChain, CrewAI, or LlamaIndex agents handle bounded reasoning steps; MCP exposes local and remote tools; A2A is reserved for remote agents that have their own runtime and lifecycle.

### Claude Code

Claude Code is a coding-agent product rather than only an orchestration library. In the lifecycle taxonomy, its ordinary interactive workflow is primarily a single-agent loop: inspect code, edit, run commands or tests, observe results, and continue.

Its extensibility also exposes the tool/lifecycle relationship:

| Feature | Layer significance |
| --- | --- |
| Built-in read, edit, search, and Bash capabilities | Core tool surface for repository work |
| MCP servers | Connect additional or custom tools and contextual systems through a standard capability boundary |
| Hooks such as pre- and post-tool events | Insert policy, validation, notifications, or additional context around actions |
| Subagents | Give a separate model context a scoped responsibility and optionally scoped tools |
| Worktree isolation for subagents | Keep parallel code changes separated until they are inspected or integrated |
| Persistent subagent memory and turn bounds | Lifecycle controls for repeated or long-running responsibility |

The Claude Code subagent documentation illustrates good capability scoping: a custom subagent can declare allowed or disallowed tools, MCP servers, hooks, memory scope, maximum turns, and `worktree` isolation. That is orchestration expressed through tool authority and execution boundaries.

Thus Claude Code is usable as a coding agent without installing custom integrations: its built-in tools cover normal file and terminal work. MCP is the usual extension boundary when it needs a new system-specific callable action, such as a Jira lookup, an internal service operation, or an organization-specific API. In Claude Agent SDK applications, Anthropic also documents custom tools as in-process MCP server tools.

For example, a reviewer subagent should normally need read/search, diff inspection, and test execution, but not direct authority to push a branch:

```yaml
---
name: patch-reviewer
description: Review a candidate implementation and validation evidence
tools: Read, Grep, Glob, Bash
maxTurns: 12
isolation: worktree
---

Inspect the proposed changes against the task requirements. Run focused
validation where useful. Report correctness issues and missing tests.
Do not publish changes.
```

The configuration is product-specific, but the harness idea is general: isolate responsibility and expose only the tools that serve it.

### OpenHands

[OpenHands](https://docs.openhands.dev/sdk/index) is an open-source platform and SDK for software agents. Its documentation describes a reasoning loop, conversation lifecycle and persistence, typed actions and observations, workspace abstraction, built-in Bash/edit/browser tools, MCP integration, custom SDK tools, and agent-server deployment.

Its tool system is particularly relevant to section 4. OpenHands includes built-in tools such as `BashTool` and `FileEditorTool`. For extensions, it provides two paths: a developer can implement an OpenHands-native tool by defining its action, observation, executor, and tool definition; or connect MCP servers for external tools reusable through a protocol. OpenHands documents MCP tools as dynamically discovered during agent initialization, with action schemas constructed from the MCP tool `inputSchema` for validation. Its workspace/runtime separation connects the tool layer to the execution environment described in Part 1.

For orchestration, OpenHands documents sub-agent delegation in which a main agent delegates independent work to subagents with their own conversation contexts and receives consolidated results. This fits tasks such as parallel repository inspection or specialized review:

```text
OpenHands main software agent
  - owns requested change and final synthesis
      |
      +--> subagent: locate implementation points
      +--> subagent: inspect related tests
      +--> subagent: review final patch
      |
      v
Workspace tools: Bash, edits, browser, MCP tools
      |
      v
Docker, Kubernetes, local, or configured remote workspace execution
```

OpenHands is therefore useful for studying a complete open-source harness: it is not merely a tool-calling library and not merely a sandbox provider.

### LangChain and LangGraph

LangChain and LangGraph should be separated conceptually:

| Library | Primary role |
| --- | --- |
| LangChain | Models, messages, tools, middleware, integrations, and high-level agent construction |
| LangGraph | Low-level runtime for long-running, stateful control flows expressed as graphs |

The official LangGraph docs state that LangChain agents use LangGraph underneath, while LangGraph can also be used without LangChain. This division maps closely to this post:

- LangChain exposes or adapts tools, including tools from MCP servers through `langchain-mcp-adapters`.
- LangGraph defines orchestration using graph state, checkpointing, interrupts, durable execution, and subgraphs.

For MCP, LangChain can discover tools across several servers and make them available to a constructed agent:

```python
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient


client = MultiServerMCPClient({
    "repo": {
        "transport": "stdio",
        "command": "python",
        "args": ["repo_tools_server.py"],
    },
    "issues": {
        "transport": "http",
        "url": "https://example.internal/mcp",
    },
})

tools = await client.get_tools()
agent = create_agent("provider:model", tools)
```

For orchestration, a LangGraph application can define separate nodes for investigation, implementation, test validation, approval, and publication. A checkpointer persists graph state so that a workflow can be interrupted for approval or resumed after failure.

```text
START -> investigate -> implement -> test
                                  |       |
                                  |       +-- failure -> implement
                                  v
                             review_gate -> publish_pr -> END
```

LangGraph also documents subgraphs for multi-agent systems. A specialist subgraph can be stateless, maintain state only during one invocation, or retain per-thread state. This makes the persistence decision explicit rather than leaving it hidden in conversation history.

### OpenAI Agents SDK

[OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) is an open-source SDK for building applications whose agents use tools, delegate work, retain session state, and emit traces. It supports MCP-backed tools and distinguishes two common orchestration forms:

| Pattern | Meaning |
| --- | --- |
| Agent as tool / manager pattern | A central agent calls specialized agents as tools and keeps ownership of the final interaction |
| Handoff | One agent transfers the conversation to another specialist |

That distinction is useful even outside this particular SDK. A coding coordinator typically wants a test-review specialist as a tool, because the coordinator must still assemble the patch and delivery summary. A support router may prefer a handoff because the specialist should take over the conversation.

The SDK is a good additional comparison because it provides orchestration primitives without prescribing a complete coding-agent product or repository lifecycle.

### Google Agent Development Kit

[Google Agent Development Kit (ADK)](https://adk.dev/) is useful in this discussion because its official documentation covers both protocols without merging their roles:

- MCP toolsets expose capabilities to an agent.
- A2A integrates independently deployed remote agents.
- `SequentialAgent`, `ParallelAgent`, and `LoopAgent` provide deterministic workflow control around LLM-powered subagents.
- An LLM agent can perform dynamic routing when a static workflow is not appropriate.

This makes ADK a clear example of layered harness architecture:

```text
Deterministic workflow agent
   |
   +--> LLM agent using MCP tools
   |
   +--> remote specialist through A2A
   |
   +--> validation agent
```

The orchestration decision is in the workflow or router. MCP and A2A serve different integration edges.

### Other Systems Worth Exploring

The survey's section 6 also names additional systems that are useful depending on the goal:

| System | Why it may be useful |
| --- | --- |
| Microsoft AutoGen | Multi-agent application framework centered on agent conversations and team patterns |
| Microsoft Semantic Kernel | Workflow-oriented agent and plugin integration in application stacks |
| DeepAgents | LangChain ecosystem implementation emphasizing planning, subagents, and longer-horizon work |
| Gemini CLI, OpenCode, Aider, SWE-agent, Codex CLI | Coding-agent inner loops useful for comparing tool surfaces and repository workflows |
| GitHub Agentic Workflows, Vibe Kanban, Symphony | Systems closer to task runners and full issue-to-pull-request lifecycle control |

The important choice is not which list is largest. It is whether the intended system is a complete coding product, an embeddable agent SDK, an orchestration runtime, a protocol boundary, or an execution environment.

## 7. A Concrete Coding-Harness Architecture

Combining the tool contracts from Part 2 with lifecycle control, a practical coding harness can use the following responsibilities:

```text
Task runner / orchestrator
  - owns issue state, branch, budgets, approval, delivery
  - checkpoints phase transitions and validation results
        |
        +--> Coordinator agent
        |      - small read/search/plan tool set
        |      - decides whether scoped delegation is useful
        |
        +--> Implementation subagent in worktree/sandbox
        |      - read, edit, run tests
        |
        +--> Review subagent
        |      - inspect diff, rerun validation, identify risks
        |
        +--> MCP servers
        |      - issue/PR system, documentation, CI status
        |
        +--> optional A2A remote agent
               - independently operated security/compliance review
```

A phase-based tool policy could be:

| Phase | Available actions | Completion condition |
| --- | --- | --- |
| Understand | Read issue, search repo, retrieve documentation | Reproduction or written implementation plan |
| Modify | Edit isolated workspace, run focused tests | Candidate diff and passing focused tests |
| Verify | Review diff, run relevant tests, query CI | Validation record and no unresolved blocking findings |
| Publish | Create commit/branch/PR only with permitted authority | PR URL or exported patch |

This design addresses several failure modes:

| Failure | Harness control |
| --- | --- |
| Tool overload confuses the model | Phase- and role-scoped tool menus |
| A subagent damages another agent's changes | Worktree or sandbox isolation |
| A remote integration changes unexpectedly | MCP discovery plus schema validation and tool policy |
| A specialist service takes a long time | A2A task status and artifact lifecycle |
| The model claims success without evidence | Structured validation gate before delivery |
| A run crashes mid-task | Persisted task state and replayable traces |
| The agent publishes too early | Human or policy approval for external writes |

## 8. Design Lessons

Tool contracts are necessary, but the lifecycle layer is what turns repeated tool use into a task that can pause, recover, delegate, verify, and finish.

The most practical conclusions are:

1. **Operational state must live outside the model's memory.** Branches, subtasks, test outcomes, retries, approvals, and artifacts should be durable harness facts.
2. **The phase of work should shape the action space.** The right tools for investigation are not necessarily the right tools for publication.
3. **Multi-agent is a design choice, not a default upgrade.** Delegation helps when tasks are parallelizable or roles need isolated context and authority; it also creates coordination cost and new failure modes.
4. **Protocols define edges, not the whole workflow.** MCP and A2A can make tools and remote agents interoperable, but the harness still decides when and why to use them.
5. **Recovery requires checkpoints and traces.** Long-running agent work needs enough durable state to resume after interruption and enough observability to explain what happened.
6. **A coding agent is finished only when it delivers a verified artifact.** A diff, commit, or pull request connected to validation evidence is stronger than an unverified "done" message.

Part 1 answered: **where can the agent safely act?** Part 2 answered: **what can it call, and through which protocol boundary?** This post answers: **how does repeated action become a completed workflow?**

## References

- [Agent Harness Engineering: A Survey of LLM Infrastructure](https://picrew.github.io/LLM-Harness/main.pdf)
- [Agent2Agent Protocol Documentation](https://a2a-protocol.org/latest/)
- [Claude Code: Extend Claude Code](https://code.claude.com/docs/en/features-overview)
- [Claude Code: Tools Reference](https://code.claude.com/docs/en/tools-reference)
- [Claude Code: Connect to Tools via MCP](https://code.claude.com/docs/en/mcp)
- [Claude Code: Create Custom Subagents](https://code.claude.com/docs/en/sub-agents)
- [Claude Code: Run Agent Teams](https://code.claude.com/docs/en/agent-teams)
- [Claude Code: Hooks Reference](https://code.claude.com/docs/en/hooks)
- [OpenClaw Capabilities Overview](https://docs.openclaw.ai/tools)
- [OpenClaw Skills](https://docs.openclaw.ai/tools/skills)
- [OpenHands Software Agent SDK](https://docs.openhands.dev/sdk/index)
- [OpenHands Tool System and MCP](https://docs.openhands.dev/sdk/arch/tool-system)
- [OpenHands Sub-Agent Delegation](https://docs.openhands.dev/sdk/guides/agent-delegation)
- [LangGraph Overview](https://docs.langchain.com/oss/python/langgraph)
- [LangGraph Subgraphs and Persistence](https://docs.langchain.com/oss/python/langgraph/use-subgraphs)
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)
- [OpenAI Agents SDK: Agents](https://openai.github.io/openai-agents-python/agents/)
- [Google Agent Development Kit](https://adk.dev/)
- [Google ADK: Introduction to A2A](https://adk.dev/a2a/intro/)
- [Google ADK: Workflow Agents](https://adk.dev/agents/workflow-agents/)
- [CrewAI Documentation](https://docs.crewai.com/)
- [Microsoft AutoGen](https://microsoft.github.io/autogen/stable/)
- [Semantic Kernel Agent Framework](https://learn.microsoft.com/en-us/semantic-kernel/frameworks/agent/)
- [LlamaIndex Multi-agent Patterns](https://developers.llamaindex.ai/python/framework/understanding/agent/multi_agent/)
