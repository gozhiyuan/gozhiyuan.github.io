---
layout: post
title: "Agent Harness Engineering Part 5: Real Agent Systems"
subtitle: OpenClaw, Hermes, Claude Code, LangGraph, Deep Agents, CrewAI, and long-running writing agents
categories: Large-Language-Model Agents Harness-Engineering
tags: [Blogs]
banner: "/assets/images/banners/yuanpang-wa-iceburg2.jpg"
---

# Agent Harness Engineering Part 5: Real Agent Systems

This is the final part of my notes on [Agent Harness Engineering: A Survey of LLM Infrastructure](https://picrew.github.io/LLM-Harness/main.pdf).

[Part 1]({% post_url 2026-05-23-harness-engineering-01 %}) covered **E: Execution Environment and Sandbox**. [Part 2]({% post_url 2026-05-23-harness-engineering-02 %}) covered **T: Tool Interface and Protocol Layer**. [Part 3]({% post_url 2026-05-25-harness-engineering-03 %}) covered **L: Lifecycle and Orchestration**. [Part 4]({% post_url 2026-05-27-harness-engineering-04 %}) covered **C: Context and Memory Management**.

This post asks a more applied question:

> If I want to build a long-running agent that can write a novel, a book-length technical manuscript, or a research survey, what can I learn from real agent projects?

The answer is not "pick the best framework." The answer is to understand what each project teaches about the harness:

- how the agent runs,
- what tools it can use,
- what memory survives,
- how work is delegated,
- how actions are verified,
- how humans stay in control,
- and how artifacts are produced.

The examples below are not a leaderboard. They are different answers to the same engineering problem.

## 1. Blog Summary

This post compares real agent systems through the harness lens from the previous four posts. The point is not to rank products. The point is to see which harness layer each system makes concrete:

| System | Main lesson |
| --- | --- |
| OpenClaw / Pi-style personal agents | Always-on gateway, channels, skills, sessions, local-first operation, and user-facing control plane |
| Hermes Agent | Persistent memory, repeat workflow learning, scheduled runs, and skill accumulation |
| Claude Code | Artifact engineering: repo context, file edits, shell commands, hooks, MCP, subagents, permissions, and validation |
| LangChain / LangGraph / Deep Agents | Programmable tool loops, durable orchestration, persistence, human-in-the-loop control, subagents, and filesystem-backed context |
| CrewAI | Separation between process state and collaborative agent teams through Flows and Crews |

The practical conclusion is that long-running agents are usually compositional:

```text
operator surface
  -> durable orchestrator
  -> specialist agents
  -> sandboxed tools
  -> artifact workspace
  -> memory and provenance store
  -> validators and human approval gates
```

For a long-running book, novel, or literature-review agent, I would start with a repo-backed manuscript runtime, then choose the framework based on the missing layer:

| Missing layer | Better starting point |
| --- | --- |
| Always-on personal entrypoint | OpenClaw-style gateway |
| Repeatable personal routines | Hermes-style skill and memory loop |
| File/shell/build work | Claude Code or OpenHands-style artifact harness |
| Durable state machine | LangGraph or Temporal |
| Role-based research/writing teams | CrewAI Flow plus Crews |
| Long-term project memory | Mem0, vector search, graph memory, or a custom memory service |

The hands-on example at the end turns this into a concrete manuscript workflow: sources, notes, outline, chapters, reviews, lifecycle state, validators, and human approval.

## 2. OpenClaw: Pi-Style Personal Agent Gateway

I am using "Pi-style" here in the practical sense of a personal-interface or personal-intelligence agent: an assistant that stays available, receives tasks from normal communication channels, and can act across the user's digital environment. OpenClaw is useful to study because it treats the agent as something that lives with you. It runs on your own device or server, talks through messaging channels, and exposes a gateway that manages sessions, tools, events, channels, and skills.

The official README describes it as a personal assistant that runs on your own devices, supports many channels such as WhatsApp, Telegram, Slack, Discord, Signal, iMessage, Matrix, Teams, and others, and provides an onboarding flow for the gateway, workspace, channels, and skills.

Minimal install path:

```bash
npm install -g openclaw@latest
openclaw onboard --install-daemon
openclaw gateway status
```

Foreground debug mode:

```bash
openclaw gateway stop
openclaw gateway --port 18789 --verbose
```

Send a test agent request:

```bash
openclaw agent \
  --message "Create a weekly research digest plan for my long-running book project." \
  --thinking high
```

For a writing agent, the OpenClaw-style lesson is the **outer operator loop**:

```text
Telegram / Slack / mobile message
  -> gateway
  -> route to the right agent workspace
  -> run tools or delegate
  -> persist session and artifacts
  -> notify the human
```

That is useful when the agent should keep running while I am not sitting in an IDE. Example tasks:

- "Every morning, summarize new papers on long-horizon agents."
- "Every Friday, propose three chapter improvements based on the last week's notes."
- "Watch my manuscript repo and tell me when citations are stale."
- "Prepare a revision plan, but do not edit the canonical draft without approval."

The security lesson is also direct. A personal gateway connected to real accounts has a wide blast radius. OpenClaw's README explicitly calls out pairing, allowlists, risky DM policies, and sandboxing non-main sessions. For writing agents, I would make the first version boring:

```yaml
agents:
  defaults:
    sandbox:
      mode: "non-main"

writing_project:
  allowed_actions:
    - read_project_notes
    - search_sources
    - write_draft_branch
    - create_review_report
  blocked_actions:
    - publish_without_human_approval
    - delete_source_cache
    - send_email_from_personal_account
```

The exact config shape depends on the project, but the policy idea matters more than the syntax: the always-on gateway should be narrow at first.

## 3. Hermes Agent: Persistent Skill and Memory Loop

Hermes Agent is useful because it emphasizes learning from repeated work. Its README describes a closed learning loop: agent-curated memory, autonomous skill creation after complex tasks, self-improving skills, past-conversation search, scheduled automations, subagents, and multiple terminal backends.

Install:

```bash
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
source ~/.bashrc
hermes
```

Basic setup:

```bash
hermes model
hermes tools
hermes setup
hermes gateway
```

The writing-agent use case for Hermes is not "write a perfect book once." It is better framed as:

> Learn the repeatable routines that my project keeps needing.

Examples:

- turn a messy reading note into a structured paper card,
- refresh a bibliography section,
- compare two chapter outlines,
- rewrite a section in the established voice,
- check whether a character timeline contradicts earlier chapters,
- run a nightly "what changed in the source world?" report,
- convert successful workflows into reusable skills.

A simple operator prompt:

```text
You are the research-maintenance agent for the manuscript repo.

Every run:
1. Read project_state.md.
2. Search the source cache and recent notes.
3. Produce one markdown artifact under reports/YYYY-MM-DD/.
4. Do not modify chapters unless the task explicitly says "edit draft".
5. When a routine succeeds twice, propose a skill file.
```

The harness lesson is that memory should be operational, not nostalgic. A long-running writing agent should not merely remember conversation vibes. It should accumulate reusable procedures:

```text
good search query patterns
accepted outline structure
venue-specific paper style
known weak arguments
chapter continuity constraints
reviewer preferences
build commands
source-verification rules
```

## 4. Claude Code: Artifact Engineering Harness

Claude Code is the most relevant project when the writing system has a repository behind it. Its docs describe a coding agent that reads a codebase, edits files, runs commands, integrates with development tools, supports persistent instructions through `CLAUDE.md`, can use MCP, hooks, subagents, worktrees, background agents, and scheduled tasks.

Install and start:

```bash
curl -fsSL https://claude.ai/install.sh | bash
cd manuscript-project
claude
```

For a book or research-paper repo, `CLAUDE.md` is the project memory that should be visible on every session. A minimal version:

```markdown
# Manuscript Agent Instructions

## Project

This repository contains a long-running manuscript project.

## Commands

- Build PDF: `make pdf`
- Check citations: `make citations`
- Check links: `make links`
- Run all validation: `make verify`

## Rules

- Never edit `main.tex` or files under `chapters/` without a task-specific plan.
- Put exploratory drafts under `drafts/YYYY-MM-DD/`.
- Every factual claim in a research chapter needs a citation key.
- Every final answer should mention which validation command ran.
```

Claude Code hooks are especially important because instructions are not enforcement. A hook can block or inspect high-impact actions before they run. A simple policy sketch:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "if": "Bash(rm *)",
            "command": "${CLAUDE_PROJECT_DIR}/.claude/hooks/block-destructive.sh"
          }
        ]
      }
    ]
  }
}
```

For long writing projects, I would use Claude Code for the deterministic artifact loop:

```text
read issue / plan
  -> edit Markdown or LaTeX
  -> run formatter
  -> run citation/link/PDF build
  -> produce diff
  -> summarize residual risk
```

Useful subagent roles:

| Subagent | Job |
| --- | --- |
| `source-checker` | Verify citations, quotes, URLs, and bibliography entries |
| `continuity-reviewer` | Check chapter claims, terminology, character state, timeline, and prior decisions |
| `latex-builder` | Fix LaTeX, figure, table, and build failures |
| `style-editor` | Improve prose while preserving claims and citations |
| `skeptical-reviewer` | Produce review comments without editing the draft |

The key point is that Claude Code is strongest when the writing task becomes inspectable files plus repeatable validation.

## 5. LangChain, LangGraph, and Deep Agents: Programmable Harness

LangChain's current docs define an agent as a model calling tools in a loop until the task is complete, and describe the harness as the model, prompt, tools, and middleware around that loop. LangGraph sits underneath as an orchestration runtime for durable execution, streaming, human-in-the-loop, and persistence. Deep Agents prepackages planning, subagents, filesystem tools, and context management on top of LangGraph.

Start with a small LangChain agent:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -qU langchain "langchain[openai]"
```

```python
from langchain.agents import create_agent


def paper_search(query: str) -> str:
    """Search the local paper index and return short results."""
    return f"Results for: {query}"


agent = create_agent(
    model="openai:gpt-5.5",
    tools=[paper_search],
    system_prompt=(
        "You are a research assistant. Return concise, source-grounded notes. "
        "Do not invent citations."
    ),
)

result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Find arguments for a section on context rot in long-running agents.",
    }]
})

print(result["messages"][-1].content_blocks)
```

For a deeper harness, use Deep Agents:

```bash
uv add deepagents
```

```python
from deepagents import create_deep_agent


def search_sources(query: str) -> str:
    """Search project sources and return concise notes with source ids."""
    return "source_id=demo-001; note=Long-context reliability can degrade with noisy context."


agent = create_deep_agent(
    model="openai:gpt-5.5",
    tools=[search_sources],
    system_prompt=(
        "You are the research coordinator for a long manuscript. "
        "Plan before drafting, write artifacts to the workspace, "
        "and preserve provenance for every factual claim."
    ),
)

result = agent.invoke({
    "messages": "Draft a one-page outline for the chapter on agent memory."
})
```

The reason I would reach for LangGraph or Deep Agents is ownership of the state machine. A book/research system needs durable state that is more specific than chat history:

```python
from dataclasses import dataclass, field


@dataclass
class ManuscriptState:
    project_id: str
    phase: str = "intake"
    topic: str = ""
    outline_path: str = "outline.md"
    source_index_path: str = "sources/index.jsonl"
    active_chapter: str | None = None
    open_questions: list[str] = field(default_factory=list)
    reviewer_findings: list[str] = field(default_factory=list)
    last_build_passed: bool = False
    human_approval_required: bool = True
```

Then the workflow becomes explicit:

```text
intake
  -> source_recall
  -> source_scoring
  -> outline_update
  -> section_draft
  -> critique
  -> revision
  -> build
  -> human_review
  -> publish_candidate
```

This is the layer where I would encode budgets, retries, approval gates, and "do not continue drafting until sources are verified."

## 6. CrewAI: Flows for Process, Crews for Teamwork

CrewAI is useful because its architecture makes a clean distinction:

- **Flows** manage state and execution order.
- **Crews** are groups of agents that collaborate on delegated work.

That maps naturally to writing:

```text
Flow:
  choose topic
  collect sources
  run research crew
  run writing crew
  run review crew
  build artifact
  ask for human approval

Crews:
  researcher
  writer
  reviewer
  editor
  formatter
```

Create a starter Flow:

```bash
crewai create flow manuscript-flow
cd manuscript_flow
crewai install
crewai run
```

A minimal `agents.yaml` for a research crew:

```yaml
researcher:
  role: >
    {topic} Senior Researcher
  goal: >
    Find credible sources, summarize the main claims, and preserve links.
  backstory: >
    You are careful about provenance. You prefer fewer high-quality sources
    over a long unverified bibliography.

reviewer:
  role: >
    Skeptical Manuscript Reviewer
  goal: >
    Find weak claims, missing evidence, and unclear structure.
  backstory: >
    You are direct and evidence-driven. You do not rewrite until the critique
    is complete.
```

A minimal `tasks.yaml`:

```yaml
research_task:
  description: >
    Research {topic}. Return a markdown brief with claims, source URLs,
    confidence notes, and open questions.
  expected_output: >
    A source-grounded markdown brief. Every factual claim should include a
    source URL or a TODO explaining why evidence is missing.
  agent: researcher
  output_file: output/research_brief.md

review_task:
  description: >
    Review the research brief for unsupported claims and missing angles.
  expected_output: >
    A markdown review with blocking issues, non-blocking suggestions, and
    recommended next steps.
  agent: reviewer
  output_file: output/review.md
```

The Flow should own the process state, not the agents:

```python
from pydantic import BaseModel
from crewai.flow import Flow, listen, start

from manuscript_flow.crews.content_crew.content_crew import ResearchCrew


class ManuscriptFlowState(BaseModel):
    topic: str = "Long-running AI agents for research writing"
    research_done: bool = False
    review_done: bool = False


class ManuscriptFlow(Flow[ManuscriptFlowState]):
    @start()
    def prepare(self):
        print(f"Topic: {self.state.topic}")

    @listen(prepare)
    def run_research(self):
        ResearchCrew().crew().kickoff(inputs={"topic": self.state.topic})
        self.state.research_done = True

    @listen(run_research)
    def summarize(self):
        print("Artifacts: output/research_brief.md and output/review.md")


def kickoff():
    ManuscriptFlow().kickoff()
```

CrewAI's lesson is readability. When the work has human-meaningful roles and process stages, a Flow plus Crew design can make the system easier to inspect than a single large agent loop.

## 7. Hands-On Long-Run Multi-Agent Example

A book-writing, novel-writing, or literature-review agent is a long-horizon system. It is not a prompt.

The output may look like text, but the work is closer to a software project:

| Writing task | Harness equivalent |
| --- | --- |
| Literature search | Retrieval tools, web/API connectors, citation database |
| Outline | Planning state and decomposition |
| Drafting | Artifact generation inside a workspace |
| Chapter continuity | Memory, outline graph, entity/style bible |
| Revision | Reviewer subagents and structured critique |
| Fact checking | External tools, provenance, source verification |
| Formatting | Deterministic build tools such as LaTeX, Pandoc, or Markdown |
| Publication | Final artifact pipeline, validation, human approval |

Victor Chen's [AutoResearch V2](https://victorchen96.github.io/blog_auto_research_v2.html) is a useful reference point because it treats paper writing as a pipeline: literature recall, scoring, classification, writing, experiments, and review. The interesting part is not only that agents produced long documents. It is that the pipeline records intermediate artifacts, uses specialized skills, and iterates through review.

That is the mental model I want for my own long-running writing agents:

```text
human direction
  -> durable project state
  -> retrieval and source intake
  -> outline and task graph
  -> chapter / section workers
  -> reviewer and fact-checker loops
  -> manuscript build
  -> provenance and evaluation report
  -> human approval
```

For a novel, replace "citation verification" with character continuity, timeline consistency, style constraints, and reader-response review. For a research paper, replace it with references, experiments, tables, venue style, and reviewer simulation. The harness shape is similar.

A useful way to generalize these systems is to treat long-form writing as a **manuscript runtime**. The runtime is not a single agent. It is a repo-backed workspace where sources, outlines, drafts, reviews, validation scripts, and decisions all have durable locations.

A minimal project layout might look like this:

```text
manuscript-agent/
  AGENTS.md or CLAUDE.md
  project_state.md
  outline.md
  sources/
    index.jsonl
    snapshots/
  chapters/
    01-introduction.md
    02-memory.md
  drafts/
  reviews/
  reports/
  skills/
    paper_card/
    section_review/
    citation_check/
  scripts/
    verify_sources.py
    build_pdf.sh
    check_continuity.py
  Makefile
```

This structure turns writing into an inspectable workflow:

| Directory or file | Harness role |
| --- | --- |
| `project_state.md` | Current phase, active chapter, open questions, and validation status |
| `sources/index.jsonl` | Source cards, citation metadata, reliability notes, and provenance |
| `outline.md` | Task graph for chapters, sections, claims, and missing evidence |
| `chapters/` | Canonical manuscript artifacts |
| `drafts/` | Experimental branches of prose that are not yet accepted |
| `reviews/` | Reviewer comments, critique reports, and revision plans |
| `skills/` | Repeatable agent procedures such as paper-card creation or citation checking |
| `scripts/` | Deterministic checks that should not depend on model judgment |

The first useful runtime only needs a small command surface:

```bash
make ingest        # add sources to sources/index.jsonl
make outline       # rebuild outline and task graph
make draft         # draft one selected section
make review        # run reviewer checks
make verify        # citations, links, build, continuity
```

The state record should be boring and explicit:

```yaml
project: long-running-writing-agent
mode: research_paper
phase: review
active_chapter: "02-memory"
source_policy: "no unsupported factual claims"
approval_required_for:
  - publishing
  - deleting sources
  - emailing external people
  - changing final chapters
last_validation:
  command: "make verify"
  passed: false
  report: "reports/2026-07-01-verify.md"
```

The orchestrator can be LangGraph, Deep Agents, CrewAI, Claude Code, Hermes, OpenClaw, or a smaller custom runner. The important part is the contract:

```text
The model may propose.
The harness records.
The tools execute.
The validators check.
The human approves.
The artifact pipeline publishes.
```

This pattern also makes the difference between genres clear. A research-paper agent needs citation provenance, experiment records, and reviewer simulation. A novel-writing agent needs character state, timeline consistency, world rules, voice constraints, and continuity review. Both still need a durable workspace, scoped agent roles, validation, and human approval gates.

## 8. Choosing the Best Framework and Starter Roadmap

The useful comparison is not "which agent is smartest?" It is "which part of the harness does this project make concrete?"

| Project | Strongest harness lesson | Best fit for long writing agents |
| --- | --- | --- |
| [OpenClaw](https://github.com/openclaw/openclaw) | Always-on personal gateway: channels, skills, sessions, local-first operation, and real account access | A personal assistant layer that accepts tasks from chat, schedules recurring work, and owns the outer control plane |
| [Hermes Agent](https://github.com/NousResearch/hermes-agent) | Self-improving skill and memory loop, cron, gateway, terminal backends, and cross-session recall | A persistent specialist that improves repeat workflows such as literature refresh, daily summaries, or manuscript maintenance |
| [Claude Code](https://code.claude.com/docs/en/overview) | Coding-agent harness: repo context, file edits, shell commands, hooks, MCP, subagents, worktrees, and permissions | A strong artifact engineer for code-backed writing projects: LaTeX builds, scripts, tests, site generation, and review automation |
| [LangChain](https://docs.langchain.com/oss/python/langchain/overview), [LangGraph](https://docs.langchain.com/oss/python/langgraph/overview), and [Deep Agents](https://github.com/langchain-ai/deepagents) | Programmable agent runtime: tools, middleware, durable execution, human-in-the-loop, persistence, subagents, filesystem, and context management | The framework layer for a custom book/research agent where I own the state machine and artifact schema |
| [CrewAI](https://docs.crewai.com/en/introduction) | Explicit separation between Flows and Crews: process state versus autonomous agent teams | A readable orchestration model for multi-role writing teams such as researcher, writer, reviewer, editor, and formatter |
| OpenAI Agents SDK | Python-first agent loop, tools, handoffs, sessions, guardrails, tracing, and sandbox agents | A good base when I want OpenAI-native primitives but still want to own the product harness |
| AutoGen | Multi-agent conversation and agent-as-tool patterns | Useful for multi-agent experiments, with roadmap caution because the original Microsoft project is now maintenance-mode |

The long-running writing use case probably needs pieces from more than one category:

```text
Gateway / operator surface:
  OpenClaw-style or Hermes-style always-on assistant

Artifact workspace:
  Git repo with Markdown, LaTeX, data, scripts, source cache, and build checks

Programmable orchestrator:
  LangGraph / Deep Agents / CrewAI Flow

Artifact engineer:
  Claude Code or another coding harness for deterministic editing, tests, and builds

Memory and provenance:
  project notes, citation DB, vector index, source snapshots, review logs, chapter state
```

My default choice for a serious long-running writing agent would be:

```text
LangGraph or Temporal for durable control flow
+ CrewAI-style role teams when the task naturally decomposes
+ Claude Code / OpenHands / sandbox agents for file and build work
+ Mem0 or a custom memory service for cross-project memory
+ deterministic validators for citations, links, builds, continuity, and publication gates
```

CrewAI is attractive when the system should read like a human organization chart. LangGraph is better when resumability, state inspection, and correction matter most. Claude Code is better when the task is mostly repo artifact work. OpenClaw or Hermes is better when the agent should stay available through normal communication channels and scheduled routines.

A public, reusable roadmap should start with supervised autonomy rather than a fully independent writer:

1. **Start with one workspace.** Store sources, outlines, chapters, reviews, and build scripts in Git.
2. **Define one source-card format.** Every source should have title, URL, author, date, summary, important claims, and reliability notes.
3. **Draft one section at a time.** Avoid asking the system to rewrite the entire manuscript until the section-level workflow is stable.
4. **Use at least two review passes.** One reviewer checks evidence or continuity. Another checks structure, style, and argument flow.
5. **Keep validation deterministic where possible.** Links, citations, PDF builds, duplicate references, missing source IDs, and file layout should be checked by scripts.
6. **Require human approval for irreversible steps.** Publishing, deleting source caches, emailing external people, or changing accepted chapters should not be autonomous by default.
7. **Extract skills only after repetition.** If a routine works twice, turn it into a skill, prompt, script, or workflow template.

The next layer can add more autonomy:

- always-on gateway access,
- scheduled literature refresh,
- multi-agent parallel chapter drafting,
- long-term memory across projects,
- automatic experiment generation,
- agent-simulated peer review,
- and publication automation.

This staged approach keeps the article's main point practical: long-running writing agents become useful when their harness turns messy creative or research work into small, reviewable, recoverable artifact updates.

## 9. Final Takeaway

The main lesson from these projects is that long-running agents are not defined by long prompts. They are defined by harnesses.

OpenClaw shows the always-on personal gateway. Hermes shows self-improving memory and skills. Claude Code shows artifact engineering with real tools, files, permissions, hooks, and validation. LangGraph and Deep Agents show how to program durable orchestration and context management. CrewAI shows how to express multi-agent work as process plus teams.

For writing books, novels, and research papers, the winning design is probably not one agent that "writes everything." It is a controlled manuscript runtime:

```text
memory + sources + outline + task graph + agent roles + review + build + approval
```

That is harness engineering applied to writing.

## References

- [OpenClaw GitHub README](https://github.com/openclaw/openclaw)
- [Hermes Agent GitHub README](https://github.com/NousResearch/hermes-agent)
- [Claude Code overview](https://code.claude.com/docs/en/overview)
- [Claude Code memory](https://code.claude.com/docs/en/memory)
- [Claude Code hooks reference](https://code.claude.com/docs/en/hooks)
- [LangChain agents documentation](https://docs.langchain.com/oss/python/langchain/agents)
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)
- [Deep Agents GitHub README](https://github.com/langchain-ai/deepagents)
- [CrewAI introduction](https://docs.crewai.com/en/introduction)
- [CrewAI quickstart](https://docs.crewai.com/en/quickstart)
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)
- [AutoGen GitHub README](https://github.com/microsoft/autogen)
- [Mem0 documentation](https://docs.mem0.ai/)
- [Temporal documentation](https://docs.temporal.io/)
- [AutoResearch V2](https://victorchen96.github.io/blog_auto_research_v2.html)
