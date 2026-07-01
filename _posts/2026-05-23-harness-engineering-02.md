---
layout: post
title: "Agent Harness Engineering Part 2: Tools and Protocols"
subtitle: MCP, A2A, tool schemas, host execution, and policy wrappers
categories: Large-Language-Model Agents Harness-Engineering
tags: [Blogs]
banner: "/assets/images/banners/yuanpang-wa-iceburg2.jpg"
---

# Agent Harness Engineering Part 2: Tools and Protocols

This post is the second part of my notes on [Agent Harness Engineering: A Survey of LLM Infrastructure](https://picrew.github.io/LLM-Harness/main.pdf). [Part 1]({% post_url 2026-05-23-harness-engineering-01 %}) discussed **E: Execution Environment and Sandbox**: where actions run and what physically contains them.

This post focuses on **T: Tool Interface and Protocol Layer** from the paper: how an agent discovers, represents, invokes, constrains, and observes external capabilities.

The tool/protocol layer is large enough on its own, so this post keeps lifecycle and orchestration mostly as contrast. The distinction still matters here: a tool interface gives an agent verbs such as "read file", "query issue tracker", or "run test". Orchestration decides which actor should use which verb, in what order, with which state, until the job is complete.

That distinction is particularly important when discussing two protocols mentioned in the paper:

| Protocol | Main boundary | What it makes interoperable | Relationship to orchestration |
| --- | --- | --- | --- |
| **MCP: Model Context Protocol** | Agent application to tool/context server | Tools, resources, and prompts | Supplies capabilities that one or many agents may use; it is not itself a multi-agent planner |
| **A2A: Agent2Agent Protocol** | Agent application to another independent agent service | Agent discovery, messages, long-running tasks, status, and artifacts | Provides a network boundary for delegated agent work; an orchestrator still decides when to delegate and how to combine results |

So MCP may appear inside a multi-agent system, but an MCP tool server is not automatically another agent. A2A is more directly about agent-to-agent collaboration, but A2A also does not decide the workflow by itself.

## 1. Tools: The Actions Available to the Workflow

With execution boundaries established in Part 1, the next layer is the operations an agent can call. The paper describes the tool interface layer as the boundary through which an agent discovers capabilities, represents callable affordances, and executes actions across heterogeneous runtime boundaries.

At a high level, a **tool** is an operation that lets the model obtain an external observation or create an external effect. The tool might read a file, execute a shell command, browse a page, call an application API, or ask another service for information.

This section has five parts:

1. **Tool taxonomy:** what counts as a tool, and what is only a protocol, skill, or delegation mechanism.
2. **Runtime exposure:** how the model is told which tools are available now.
3. **Provider/API mechanics:** how tool-call tokens become structured API and SDK objects.
4. **Host execution:** how the coding-agent product validates and runs the requested tool.
5. **Design controls:** built-in versus custom tools, tool menus, and policy wrappers.

### 1.1 Tool Taxonomy

The word "tool" is also used loosely in agent products. Function calls, Bash, skills, MCP, and A2A do not all mean the same thing:

| Item | What it actually is | Is it a tool? | Coding-agent example |
| --- | --- | --- | --- |
| Function or tool call | A model-facing structured invocation format, usually with a name, description, and JSON-shaped arguments | It is the common **interface used to call a tool** | `run_tests({"command": "pytest -q"})` |
| Built-in file/search/edit operation | A capability implemented by the agent host | Yes | `read_file`, `grep`, `apply_patch` |
| Bash or shell executor | One broad tool whose argument is a command; the shell then exposes many installed programs | Yes; `pytest` and `git` may be commands reached through the Bash tool rather than separate model tools | `bash({"command": "git diff --stat"})` |
| API wrapper | Application code that turns an external REST/GraphQL/SDK operation into a model-callable schema | Yes | `get_pull_request_status`, `comment_on_issue` |
| MCP server | A protocol-speaking server that publishes tools, resources, and prompts to an agent host | MCP itself is a protocol; its exposed operations are registered as tools by the host | GitHub MCP server exposes `get_issue` |
| Skill | A reusable package of instructions, workflow guidance, or supporting scripts loaded when relevant | Usually not an effectful tool by itself; it teaches the agent how to use tools for a task | A "release review" skill instructs the agent which checks and tools to run |
| Subagent | A separate agent context with a delegated responsibility and its own tools | Not inherently a primitive tool, although a coordinator can invoke it as one | `review_patch` delegates diff review to a reviewer agent |
| A2A remote agent | An independently deployed agent reached through a task/message/artifact protocol | A2A itself is a protocol for agent collaboration; an orchestrator may wrap delegation as a tool-like action | Delegate a security audit and receive a report artifact |

This means that a coding agent may have a visible tool named `Bash`; from that one tool it can run `rg`, `git`, `pytest`, or `npm test` if those commands exist in its execution environment. It may also have a directly typed `create_pull_request` tool backed by an API or an MCP server. A skill may tell it the correct release procedure, but the skill does not push the branch unless it invokes a real Git or pull-request tool.

### 1.2 Examples of Actual Tools

Once this distinction is clear, tools can be categorized by the effect they expose:

| Tool category | Example operation | Role in a coding workflow |
| --- | --- | --- |
| Read and search | `read_file`, `search_code`, read issue or docs | Understand the requirement and repository |
| Modify | `edit_file`, `apply_patch` | Produce candidate changes |
| Execute | `run_tests`, `run_command`, browser interaction | Reproduce failures and validate behavior |
| Integrate | MCP-backed issue tracker, CI, documentation, or database calls | Reach external systems through defined interfaces |
| Deliver | `push_branch`, `create_pull_request` | Publish a verified result under policy control |

These tools do not determine the overall workflow. For example, `run_tests` knows how to invoke a test command and return a result; orchestration decides whether the test is being run to reproduce a bug, verify a patch, gate publication, or diagnose a regression.

### 1.3 How the LLM Knows Which Tools Are Available

There are two different sources of tool ability:

1. **Training teaches general behavior.** A tool-capable model can learn patterns such as producing structured arguments, following tool results, or understanding common shell commands.
2. **The runtime harness grants concrete capabilities.** At the beginning of a turn or session, the application gives the model descriptions of the tools actually available now. This is what allows the model to call this repository's test runner, this GitHub integration, or this remote agent.

Training alone does not grant access. A model may know that GitHub issues exist, but it cannot read a private issue until the runtime exposes an authorized tool and executes the selected call.

| Capability form | How it is prepared outside the model | What is supplied to the model at runtime | Who executes it? |
| --- | --- | --- | --- |
| Application function tool | Developer writes a function and its schema, or an SDK derives the schema | Tool name, description, and argument schema in the model request | Agent application calls local code or a service API |
| Built-in coding tool | Product implements file editing, search, Bash, or browser control | Product-defined tool descriptions and permission rules | Coding-agent host and its sandbox/runtime |
| Bash tool | Product provisions a shell environment and exposes an executor | A tool such as `Bash(command: string)` plus execution constraints | Sandbox shell launches commands installed in that environment |
| API-backed tool | Developer wraps a service operation and configures credentials/policy | Structured operation such as `get_ci_status(run_id)` | Wrapper invokes external API |
| MCP tool | Host connects to an MCP server and discovers allowed tools, often through `tools/list` | Discovered name, description, and input schema for each selected MCP tool | MCP server performs its tool operation |
| Skill | Author writes instructions and possibly supporting assets/scripts; host indexes it | A description initially, then relevant instructions when selected by the user or model, depending on the product | Agent follows instructions and invokes its actual tools |
| A2A agent delegation | Remote agent publishes an Agent Card; orchestrator configures trust and routing | A delegation action or selected remote capability description | Remote agent runs its own lifecycle and returns messages/artifacts |

MCP and A2A therefore let a running system acquire access to new capabilities without changing the foundation model weights:

| Protocol | What the agent learns at runtime | How it can appear in the agent loop |
| --- | --- | --- |
| MCP | "This server offers a callable operation named `get_issue` with these arguments," or "this resource can be read" | The host registers the discovered operation in the model's available tool menu and routes a selected call to the MCP server |
| A2A | "This remote agent offers a capability or skill such as security review and accepts delegated tasks" | The orchestrator routes a subtask to that agent, possibly presenting `delegate_security_review(...)` to a coordinator as a tool-like action |

The word "learns" here means **receives a runtime capability description**, not **updates its neural parameters**. The LLM is told what capability is available in the active session; the host or orchestrator still owns access control, dispatch, and interpretation of the result.

Native function/tool calling makes the simplest case visible. The application supplies a schema to a model API, receives a proposed invocation, runs the corresponding implementation, and sends the result back as an observation:

```python
tools = [{
    "name": "run_tests",
    "description": "Run a test command in the isolated repository workspace.",
    "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"],
    },
}]

response = model.generate(task_context, tools=tools)

if response.tool_name == "run_tests":
    policy.authorize(response.arguments)
    result = sandbox.exec(response.arguments["command"])
    model.generate(task_context, tool_result=result)
```

In this discussion, the **host** is the coding-agent application itself: Claude Code, OpenHands, Aider, Codex CLI, or a custom agent runtime. The host owns the agent loop, state, permissions, tool registry, and execution runtime.

For each model step, the host builds an **active tool set**:

```text
Coding-agent host state:
  built-in tools         -> read, edit, search, Bash, browser
  configured MCP servers -> discovered external tools/resources
  selected skills        -> instructions, workflows, and possible tool restrictions
  active subagent role   -> role-specific prompt and tool permissions
  workflow phase         -> investigate, implement, verify, publish
  governance rules       -> allow/ask/deny, sandbox, network, approval policy
        |
        v
Active tool set for this model call:
  only the tools the model should be able to request now
```

This is what people sometimes mean by "hot loading" tools. The model may be trained to use tools in general, but concrete available tools are supplied by the host at runtime. The host may literally include schemas in the model request, or it may use provider features such as tool references, deferred loading, caching, or compacted descriptions. Architecturally, the point is the same: each model call has an active capability surface chosen by the host.

```text
For each agent step:
  1. Host loads current task/session state
  2. Host selects active tools from built-ins, MCP, skills, subagent role, and phase
  3. Host sends prompt/context plus active tool definitions or references
  4. Model returns text or structured tool calls
  5. Host validates permissions and executes allowed calls
  6. Host records observations and continues or stops
```

This also explains where governance lives. The model provider API can validate structured tool-call format, and the SDK can deserialize the response. But project policy is enforced by the coding-agent host before and after the model call:

| Governance moment | Example |
| --- | --- |
| Before the model call | Do not expose `create_pull_request` during investigation; expose only read/search/test tools to a reviewer subagent |
| After the model call | If the model requests `Bash(command="rm -rf .")`, deny it or ask the user even though `Bash` was visible |

Skills fit into this runtime-loading picture, but they are different from executable tools. A skill usually contributes instructions, procedures, examples, assets, or scripts. The host can index skill metadata and load the full skill instructions when the user invokes the skill or when the model/host selects it as relevant. Once active, the skill may guide the model toward certain built-in or MCP tools, and in some products it can restrict which tools are allowed in that skill context. But the skill itself does not perform the external effect unless the agent calls an actual tool such as Bash, Edit, browser, or an MCP operation.

```text
Skill selected:
  load skill instructions into context
  optionally narrow allowed tools
  model follows the workflow
  actual effects still happen through tools
```

### 1.4 Provider API, SDK, and Tool-Call Parsing

In modern tool-calling APIs, the same model request includes the available tool schemas. The model is trained and prompted to either produce normal text or emit a structured tool-call object whose `name` must match one of the supplied tool names.

The SDK is usually **not** scraping a dictionary out of ordinary assistant prose. The provider returns tool calls through a structured part of the API response. Depending on the API, that part may be named `tool_calls`, `function_call`, a `function_call` item in an `output` list, or a `tool_use` content block. The application reads those typed fields rather than searching the natural-language text for JSON.

There are three different layers here:

```text
Provider serving layer:
  turns model-generated tokens into provider API objects

Provider SDK:
  deserializes provider API JSON into language-native objects

Application or coding-agent host:
  validates policy, executes the tool, captures the result,
  and sends the tool result back to the model
```

At the provider layer, the model may be trained or prompted with an internal format such as:

```text
Available tools:
<tool>
name: run_tests
description: Run a test command in the isolated repository workspace.
schema: {"type":"object","properties":{"command":{"type":"string"}},"required":["command"]}
</tool>

When using a tool, emit:
<tool_use>
{"name":"run_tests","input":{"command":"..."}}
</tool_use>
```

At inference time, a simplified internal model output might look like:

```text
I will run the focused parser tests.
<tool_use>
{"name":"run_tests","input":{"command":"pytest tests/test_parser.py -q"}}
</tool_use>
```

or, in a provider-specific special-token format:

```text
<|tool_call_start|>
{"name":"run_tests","arguments":{"command":"pytest tests/test_parser.py -q"}}
<|tool_call_end|>
```

The provider serving layer parses this known channel, validates basic protocol shape, and converts it into the public API response. In production this should not be a fragile regex over arbitrary text. Providers usually rely on a combination of special tokens, chat templates, constrained decoding, streaming parsers, deterministic JSON/schema validation, and sometimes retry or repair logic.

For basic validation, another lightweight model is usually unnecessary. Deterministic code can check the objective properties:

| Validation check | Example |
| --- | --- |
| Tool name is offered | Reject `run_testz` if only `run_tests` was supplied |
| Arguments parse as JSON | Reject an unterminated JSON object |
| Required fields exist | `command` is required |
| Types match schema | `command` must be a string, not an object |
| Enum or bounds are respected | `mode` must be one of the allowed values |
| Tool-choice policy is respected | Reject tool calls if `tool_choice` says none |
| Parallel-call policy is respected | Reject or serialize multiple calls if parallel calls are disabled |

A separate light model may still be useful for semantic routing or safety review, such as deciding which of 500 tools to expose or whether a shell command is dangerous. But basic parsing and schema validation should be deterministic.

Conceptually, the request and response look like:

```json
{
  "input": "Run the focused parser tests.",
  "tools": [
    {
      "name": "run_tests",
      "description": "Run a test command in the isolated repository workspace.",
      "input_schema": {
        "type": "object",
        "properties": {"command": {"type": "string"}},
        "required": ["command"]
      }
    }
  ]
}
```

The raw API response is provider-specific, but conceptually the tool call is a typed response item, not prose:

```json
{
  "output": [
    {
      "type": "function_call",
      "call_id": "call_123",
      "name": "run_tests",
      "arguments": "{\"command\":\"pytest tests/test_parser.py -q\"}"
    }
  ]
}
```

An SDK or a thin adapter may convert that into a friendlier object:

```python
ToolCall(
    id="call_123",
    name="run_tests",
    arguments={"command": "pytest tests/test_parser.py -q"},
)
```

or, in a simplified teaching example:

```python
response.tool_name == "run_tests"
response.arguments == {"command": "pytest tests/test_parser.py -q"}
```

The model may also return normal text if no tool is needed. The application then checks that any requested tool name exists in the offered tool set and that the arguments satisfy the schema and policy.

Two details are easy to miss:

1. Tool arguments are often delivered as a JSON-encoded string inside the typed tool-call item. Your application or SDK still parses that argument string into a dictionary and validates it against the schema.
2. The model may return zero, one, or multiple tool calls. Production code should normally iterate over a list of calls, not assume exactly one.

For example, a raw response may contain several tool calls:

```json
[
  {
    "id": "call_1",
    "type": "function",
    "function": {
      "name": "read_file",
      "arguments": "{\"path\":\"src/parser.py\"}"
    }
  },
  {
    "id": "call_2",
    "type": "function",
    "function": {
      "name": "read_file",
      "arguments": "{\"path\":\"tests/test_parser.py\"}"
    }
  }
]
```

The host then loops over them:

```python
for call in response.tool_calls:
    name = call.name
    arguments = json.loads(call.arguments) if isinstance(call.arguments, str) else call.arguments

    if name not in offered_tools:
        reject(call)
        continue

    validate_json_schema(arguments, offered_tools[name].schema)
    policy.authorize(name, arguments)
    result = offered_tools[name].execute(arguments)
    tool_results.append({"call_id": call.id, "output": result})
```

The next model request includes the tool results, usually with the corresponding `call_id` or tool-use id, so the model can connect each result to the call that produced it.

The complete API/SDK/app workflow is:

```text
1. App sends messages + tool schemas to provider API
        |
        v
2. Provider renders internal prompt/chat template
        |
        v
3. Model emits either text or a known tool-call format
        |
        v
4. Provider serving layer parses and validates protocol shape
        |
        v
5. Provider API returns structured JSON response
        |
        v
6. SDK deserializes JSON into objects such as ToolCall
        |
        v
7. App validates application policy and executes the tool
        |
        v
8. App sends tool result back with the matching call id
        |
        v
9. Model continues with the observation
```

The boundaries matter:

| Layer | Responsibility |
| --- | --- |
| Model | Chooses whether to answer or request a tool, and emits the learned/internal tool-call format |
| Provider serving layer | Converts model output into public structured API fields and validates protocol/schema shape |
| SDK | Turns API JSON into convenient Python/TypeScript/etc. objects |
| Application host | Applies permissions and business policy, executes tools, truncates or summarizes observations, and manages the loop |

Provider-side validation and app-side validation are not substitutes. The provider may ensure that `{"command":"pytest -q"}` is a schema-valid argument for `run_tests`. The coding-agent host must still decide whether that command is allowed in this workspace, under this user's policy, with this timeout and sandbox.

There are older or custom setups where the model is asked to print JSON in plain text, such as:

```json
{"tool": "run_tests", "arguments": {"command": "pytest -q"}}
```

In that design, the application really does manually parse assistant text and recover from malformed JSON. Standard tool-calling APIs avoid much of that by returning tool calls as typed response objects, although the application must still validate names, schemas, permissions, and argument JSON.

There are systems that use a separate router model or classifier to choose which tools to reveal before the main model call. But after a tool has been supplied to a standard tool-calling model request, selecting `run_tests` is normally part of that same model generation, not a second LLM call.

The model chooses the call, but it does not implement `run_tests`, create the sandbox, hold API credentials, or decide the permissions. Those responsibilities remain in the harness.

### 1.5 Application Host Execution

Section 2.4 described the full provider/API/SDK path. Section 2.5 is the same workflow from the coding-agent product's point of view: once the provider SDK exposes a structured tool-call object, the coding-agent host decides whether and how to execute it.

For built-in coding tools, the product implements this harness side. The exact implementation language is product-specific and usually not important architecturally. It might be TypeScript, Python, Go, Rust, or a mixture. What matters is the control path:

```text
Provider API / SDK structured response:
  tool_use {
    name: "Bash",
    input: {"command": "pytest tests/test_parser.py -q"}
  }
        |
        v
Coding-agent host:
  reads/deserializes the tool-use object from the SDK/API response
  validates the command against permissions and sandbox policy
  chooses the target runtime/session
        |
        v
Tool executor:
  starts a subprocess or sends an exec request to a sandbox
  captures stdout, stderr, exit code, timeout, and resource errors
        |
        v
Agent loop:
  summarizes or truncates the observation
  records durable state
  sends the result back to the model for the next step
```

So when we say "the model uses Bash", the literal meaning is: the provider API returns a structured request for the host's Bash tool, derived from the model's tool-call generation. The host, not the model, reads that request and runs the command through an executor. For a local product that executor may spawn a local process with OS sandboxing. For a remote coding agent it may call a container, VM, or managed sandbox API.

The same pattern applies when the tool comes from elsewhere:

```text
Function tool: developer registers schema -> model selects it -> application executes code
Bash tool:     product registers Bash    -> model writes command -> sandbox executes process
MCP tool:      host discovers schema     -> model selects it -> MCP server executes operation
Skill:         host reveals guidance     -> model follows recipe -> actual tools execute effects
A2A agent:     orchestrator finds agent  -> delegates task -> remote lifecycle returns artifact
```

### 1.6 Built-In Tools Versus Custom Tools in Coding Agents

In products such as Claude Code, OpenHands, and OpenClaw, most routine software-engineering actions are already implemented by the product. The user normally does not need to write a tool for reading source files, editing text, or running terminal commands.

| Product | Built-in tool examples | How to add external or customized capabilities | What the model sees |
| --- | --- | --- | --- |
| Claude Code | Repository/file tools, search, edit/write operations, Bash, web-related tools, and agent features depending on configuration | Connect an MCP server; in Agent SDK applications, custom tools are exposed through an SDK MCP server | Built-in tool definitions plus allowed MCP tool definitions, such as `mcp__github__list_issues` |
| OpenHands | SDK tools including `BashTool` and `FileEditorTool`, with default tool presets for software-agent tasks | Configure MCP servers for external tools, or write a native custom SDK tool using an Action, Observation, Executor, and `ToolDefinition` | Generated schemas for native tools plus dynamically discovered MCP tool schemas |
| OpenClaw | Built-in tools for execution, files, web search/fetch, browser control, messaging, sessions, automation, and media depending on configuration | Install or build plugins for new runtime capabilities; add skills for reusable workflows around existing tools | Structured tool definitions from built-ins and plugins, plus skill instructions loaded into the prompt |

For example, both systems already know how to edit a local workspace and run a command. If an agent must query an internal bug tracker or call a company's deployment preview service, a developer can expose those operations through an MCP server:

```text
Claude Code or OpenHands
  - built in: read file, edit file, execute command
  - added through MCP: get_internal_ticket, create_preview_environment
        |
        v
MCP server owned by the developer or service team
  - defines names, descriptions, input schemas, and implementation
  - authenticates to the external service under configured policy
```

OpenHands additionally supports native custom SDK tools when a developer is building an OpenHands-based application and wants the tool logic to be part of that application rather than a separately reusable MCP integration:

```text
OpenHands native custom tool:
  Action       -> typed arguments from the model
  Executor     -> application-owned implementation
  Observation  -> typed result returned to the agent
```

The design choice is practical:

| Requirement | Prefer |
| --- | --- |
| Use the coding agent's normal editing and command abilities | Built-in tools |
| Add one integration that should work across MCP-capable hosts | MCP server |
| Add an application-specific OpenHands capability tightly coupled to its SDK state or workspace | Native OpenHands custom tool |
| Provide a repeated checklist or project procedure using existing tools | Skill, not a new executable tool |
| Delegate an entire responsibility to another independently running agent | Subagent mechanism or A2A, not a low-level tool integration |

Email access is a useful example because it shows where the labels can become confusing.

If I want Claude Code to access email, a skill is not the email API. The cleaner design is an email MCP server that exposes typed operations such as `search_messages`, `read_thread`, `create_draft`, or `send_message`. The skill can then describe the workflow and policy: summarize unread recruiter emails, draft replies but do not send, or require approval before sending. If I write the email MCP server myself, that is a custom integration, but from Claude Code's point of view it is still an MCP-provided tool rather than a native built-in tool.

OpenClaw is more skill-centered, but it is not "skills only." Its documentation separates tools, skills, and plugins: tools are typed callable actions, skills are `SKILL.md` instruction packs that teach the agent how and when to use tools, and plugins add runtime capabilities such as tools, providers, channels, hooks, and packaged skills. So an OpenClaw skill can make a workflow feel like one named capability, but the external effect still happens through some runtime surface: a built-in tool, an `exec` script, a channel tool, browser control, or a plugin-provided tool.

The same email example maps differently across the two products:

| Goal | Claude Code framing | OpenClaw framing |
| --- | --- | --- |
| Search the web | Built-in or MCP-backed web/search capability; skill may describe research procedure | Built-in web search/fetch tool; skill describes the research workflow |
| Read or draft email | Prefer email MCP server; skill wraps procedure and policy | Prefer plugin/tool or script-backed capability; skill wraps procedure and policy |
| Send a message | External-write tool or MCP operation, normally gated by approval | Built-in or plugin-provided channel/message tool, governed by tool policy |
| Reuse a repeated workflow | Skill using existing tools | Skill using existing tools |

The practical rule is: a **skill packages intent and procedure**; a **tool or plugin supplies an executable capability**; **MCP is one standardized way to expose such a capability across hosts**. A system may market or organize the experience around skills, but external side effects still require a callable runtime boundary somewhere.

A useful tool is more than a Python function made visible to an LLM. Its interface should tell the model enough to select it correctly and tell the harness enough to control it safely:

| Interface property | Why it matters |
| --- | --- |
| Clear name and purpose | The model must distinguish `search_repository` from `search_web` or `query_production_logs`. |
| Typed input schema | Structured arguments are easier to validate than prose commands. |
| Bounded output | Raw terminal logs or giant API responses consume context and hide the useful signal. |
| Failure semantics | The loop must distinguish invalid input, transient failure, denied permission, and completed action. |
| Side-effect classification | Reading a file and deleting a deployment should not share the same approval policy. |
| Session semantics | Some tools are stateless; others require a browser session, database transaction, shell session, or remote workspace. |
| Traceability | A future reviewer needs to know which tool changed which artifact and why. |

For a coding agent, a coarse Bash tool is powerful, because the shell already composes thousands of commands. It is also ambiguous and risky. A structured edit tool may be less general, but it can produce smaller diffs, easier validation, and better feedback when an edit fails.

This is a recurring harness tradeoff:

| Tool design | Advantage | Failure mode |
| --- | --- | --- |
| A few broad tools such as Bash and browser | Small menu, flexible composition | Harder to govern and harder for the model to use safely |
| Many narrow API tools | Typed effects and precise policy | Large tool menu increases selection confusion and prompt cost |
| Dynamically discovered tools | Expose only capabilities relevant to the step | Discovery and caching become part of the lifecycle |
| High-level workflow tools such as `create_pull_request` | Encapsulate repeated correct procedure | Can hide important intermediate decisions or grant a large effect at once |

### 1.7 Tool Menus Are Context

Tools consume context before they are ever called. Their names, descriptions, schemas, usage rules, and returned observations enter the model-facing information stream.

Suppose an enterprise agent can access 200 APIs. Sending all 200 schemas into every model call creates at least three problems:

1. More prompt tokens are spent on capabilities irrelevant to the current step.
2. Similar tools become easier to confuse, such as read versus update APIs or development versus production targets.
3. Sensitive capabilities appear in contexts where the agent never needed them.

A better approach is progressive disclosure:

```text
Small stable core tools:
  read, search, edit, run command, request approval, discover capability

Task-relevant tools loaded later:
  GitHub issue and PR tools for a coding task
  deployment status tools after validation
  incident tools only during an operational task
```

In other words, tool selection is partly a context-engineering problem and partly a governance problem. A harness should expose the least confusing useful action space for the current responsibility.

### 1.8 Tools Need Policy Wrappers

The same capability may need different policies depending on effect. The policy wrapper is usually static host-side code or configuration, but it is applied dynamically at two moments:

1. **Before the LLM call:** filter the registered tools into the active tool set that the model is allowed to see now.
2. **After the LLM call:** authorize, ask, trace, or deny the specific tool call and arguments the model requested.

So policy is not simply "hot loaded into the prompt." The enforceable policy remains in the coding-agent host, but it shapes the model request by deciding which tools are exposed.

```python
READ_ONLY = {"read_file", "search_code", "get_issue", "get_ci_status"}
WORKSPACE_WRITE = {"edit_file", "apply_patch", "run_tests"}
EXTERNAL_WRITE = {"push_branch", "create_pull_request", "comment_on_issue"}
HIGH_RISK = {"deploy_production", "delete_resource", "rotate_secret"}


def visible_before_model_call(tool_name, task_state, user_policy):
    if tool_name in HIGH_RISK:
        return False
    if tool_name in EXTERNAL_WRITE and task_state.phase != "publish":
        return False
    if tool_name in WORKSPACE_WRITE and not task_state.workspace_isolated:
        return False
    return user_policy.can_see(tool_name)


def authorize(tool_name, task_state, user_policy):
    if tool_name in READ_ONLY:
        return "allow"
    if tool_name in WORKSPACE_WRITE and task_state.workspace_isolated:
        return "allow_and_trace"
    if tool_name in EXTERNAL_WRITE:
        return "require_approval"
    if tool_name in HIGH_RISK:
        return "deny_unless_explicitly_delegated"
    return "deny_unknown_tool"
```

The runtime loop then applies the same policy in two places:

```python
# Dynamic pre-call filtering.
active_tools = [
    tool
    for tool in all_registered_tools
    if visible_before_model_call(tool.name, task_state, user_policy)
]

response = model.generate(context, tools=[tool.schema for tool in active_tools])

# Dynamic post-call authorization.
for call in response.tool_calls:
    decision = authorize(call.name, task_state, user_policy)

    if decision == "allow":
        execute(call)
    elif decision == "allow_and_trace":
        execute_and_log(call)
    elif decision == "require_approval":
        ask_user_for_approval(call)
    else:
        deny(call)
```

This matters because hiding a tool and authorizing a tool are different controls. During investigation, the host may hide `create_pull_request` so the model does not even consider publication. But the host still needs post-call checks: if `Bash` is visible and the model asks for a destructive command, the command can still be denied or sent to the user for approval.

Before sending tools to the model, a coding-agent host usually performs a set of readiness checks:

| Pre-call check | Example question |
| --- | --- |
| Tool registration | Does this tool have a name, description, schema, and executor? |
| Schema validity | Is the JSON schema valid and small enough to send or reference? |
| Runtime availability | Is the sandbox, shell session, browser, or MCP server connected? |
| Credential scope | Does the tool have only the credentials needed for this task? |
| Phase and role fit | Should this role see write tools, or only read/review tools? |
| User/project policy | Is this tool allowed, hidden, or ask-only for this workspace? |
| Context budget | Is the tool description worth including now, or should it be deferred? |
| Conflict and ambiguity | Are there duplicate tools with confusingly similar names? |

The exact implementation differs by product, but the principle remains: tool visibility, tool readiness, and tool authority are related but separate decisions.

## 2. Protocol Standards: Function Tools, MCP, and A2A

Early agent prototypes often define tools in the same process as their loop: a function table and a schema included in a model request. This remains useful for local tools. It becomes restrictive when tools are owned by other teams, available remotely, authenticated independently, or needed by several agent applications.

Protocol standards move those boundaries out of one agent implementation.

| Interface style | Provider of capability | Consumer sees | Suitable boundary |
| --- | --- | --- | --- |
| Native function/tool calling | Application code | Function schema supplied to a model API | Small application-owned tool set |
| REST/OpenAPI adapter | Existing service API | Generated or handwritten tool wrappers | Conventional services exposed to one application |
| MCP | MCP server | Discoverable tools, resources, prompts, and protocol lifecycle | Reusable agent-to-capability integration |
| A2A | Remote agent service | Agent card, tasks, status, messages, artifacts | Delegation to an independent agent system |

An application may use all of these at once. For example, a coding orchestrator can use local file functions, connect to a GitHub MCP server, and delegate security review to a remote A2A agent.

The short distinction is:

```text
MCP: agent host <-> tool or contextual resource
A2A: agent/orchestrator <-> independent remote agent
```

Both are protocols rather than model abilities. Both can add runtime capability to a harness. But only an MCP `tool` directly becomes an ordinary tool invocation in the agent's menu. An A2A interaction is a delegation lifecycle, although a coordinator application may hide that lifecycle behind a higher-level tool such as `ask_security_agent(...)`.

### 2.1 MCP: A Standard Tool and Resource Boundary

[Model Context Protocol](https://modelcontextprotocol.io/docs/learn/architecture) uses a host-client-server design:

```text
Agent host application
  - owns user interaction, model calls, policy, and orchestration
        |
        | one MCP client connection per configured server
        v
MCP server
  - declares tools, resources, and prompts
  - performs its owned operation when invoked
```

The host might be Claude Code, an IDE agent, a custom LangGraph application, OpenHands, Google ADK, or another agent product. The server might expose GitHub operations, a database, an internal documentation source, a browser, or a deployment service.

MCP's data layer is based on JSON-RPC. Its core server-facing primitives are:

| MCP primitive | Meaning | Example in a coding task |
| --- | --- | --- |
| `tools` | Callable actions | `get_issue`, `search_pull_requests`, `create_review_comment` |
| `resources` | Readable contextual data | repository conventions document or database schema |
| `prompts` | Reusable interaction templates | incident-triage or release-review template |
| Notifications and progress | Changes or progress in a session | updated tool list or long-running job progress |

Its commonly documented transports are:

| Transport | Shape | Common use |
| --- | --- | --- |
| `stdio` | The host launches and communicates with a local server process | Local filesystem or development integration |
| Streamable HTTP | The host connects to a server over HTTP, with optional streaming | Hosted tools managed by another service or team |

For remote MCP, the connection layer can look ordinary: TCP, TLS, and HTTP(S), just like many service APIs. The protocol distinction appears one layer above transport. Instead of an agent host needing a Notion-specific, GitHub-specific, or Slack-specific client, it can speak the MCP tool/resource contract to any compatible server.

For example, if Notion hosts an MCP server, the shape is:

```text
Claude Code or another MCP host
  |
  | MCP protocol messages over HTTP(S)
  | initialize, list tools, call tool, read resource
  v
Notion MCP server
  |
  | Notion-specific API or internal calls
  v
Authorized Notion workspace
```

Connection-wise, this may still be ordinary HTTPS. What makes it MCP is the standardized agent-facing contract carried over that connection: initialization, capability discovery, tool schemas, tool calls, resource reads, results, errors, notifications, and capability updates.

Authentication also belongs outside the model's raw reasoning. After OAuth or token setup, the host and MCP server can keep an authorized connection and invoke Notion tools without asking the user to re-authenticate on every call, until the token expires, permissions change, the server is disabled, or policy blocks the action. The LLM itself does not "have the auth"; it sees tool names, schemas, and observations. The host and server hold the authorized connection and enforce scopes, workspace permissions, and tool policy.

This means a Notion MCP server is different from the Notion REST API even if both use HTTPS:

| Boundary | Who it is for | Example shape |
| --- | --- | --- |
| Notion REST API | Notion-specific application clients | `POST /v1/search` with Notion-specific JSON |
| Notion MCP server | MCP-capable agent hosts | `tools/list`, then `tools/call` for `search_pages` or `update_page` |

The service-specific API remains behind the MCP server. MCP provides the agent-facing adapter layer.

Conceptually, tool discovery and execution look like:

```json
{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
```

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "tools": [
      {
        "name": "get_issue",
        "description": "Read one issue by repository and number.",
        "inputSchema": {
          "type": "object",
          "properties": {
            "repository": {"type": "string"},
            "number": {"type": "integer"}
          },
          "required": ["repository", "number"]
        }
      }
    ]
  }
}
```

The model does not need to know whether the tool ultimately calls a REST service, queries a database, or reads a local file. The host discovers a schema, presents an allowed tool to the model, validates a proposed invocation, dispatches it, and returns a bounded result.

#### What MCP Solves and What It Does Not

| MCP helps standardize | MCP does not automatically solve |
| --- | --- |
| Tool and resource discovery | Whether the model should call a tool at this step |
| Schema-shaped invocations and results | Whether the tool description is sufficiently clear |
| Local and remote server transport | Which user identity or permission should be granted |
| Reuse of an integration across agent hosts | Prompt injection or dangerous tool combinations |
| Connection and capability lifecycle | Task decomposition, retries, validation, or PR delivery |

This is why MCP belongs in the tool-interface layer. It is an important tool and context protocol, not a complete harness.

#### MCP Inside a Multi-Agent Harness

MCP becomes useful in orchestration when capabilities are scoped to responsibilities. A research subagent may receive web and documentation tools; an implementation subagent may receive repository edit and test tools; a reviewer may receive diff and CI tools but no write access.

```text
Coordinator
  |
  +-- Research subagent     -> docs MCP server, issue tracker MCP server
  |
  +-- Implementation agent  -> sandbox file tools, test runner
  |
  +-- Review subagent       -> diff tools, CI status MCP server
```

The existence of MCP servers does not create these agents or route tasks between them. The orchestrator does that. MCP makes each capability boundary portable and more consistently described.

### 2.2 A2A: A Standard Agent Delegation Boundary

[Agent2Agent Protocol (A2A)](https://a2a-protocol.org/latest/) is designed for communication between independent agent applications. In contrast to an MCP tool, a remote agent may plan internally, ask for more input, run for a long period, and return artifacts rather than a single function result.

The key concepts are:

| A2A concept | Purpose |
| --- | --- |
| Agent Card | Describes an agent's identity, endpoint, capabilities, skills, and interaction or authentication information |
| Message | Exchanges instructions or conversational information between client and remote agent |
| Task | Represents a stateful unit of delegated work |
| Status | Communicates whether work is running, waiting for input, complete, failed, or canceled |
| Artifact | Returns produced output, such as a report, patch, data object, or document |
| Streaming or notification | Reports progress from a long-running remote task |

The conceptual difference is:

```text
MCP:
  "Here is a typed capability. Call it and receive a tool result."

A2A:
  "Here is an independent agent. Give it a task, follow its status,
   exchange messages if needed, and receive produced artifacts."
```

#### Local Subagent Versus Remote A2A Agent

Not every subagent needs a network protocol. A subagent inside the same product can be represented as an internal tool invocation, a new model context, or a graph node.

| Case | Useful integration |
| --- | --- |
| A Claude Code subagent exploring another directory in the same project | Product-native subagent mechanism |
| A LangGraph supervisor invoking a specialist graph in the same deployment | Graph/subgraph invocation |
| A company-wide compliance agent owned and deployed by another team | A2A-style remote-agent boundary |
| A database query needed by any of those agents | MCP tool boundary |

Google's Agent Development Kit documentation makes this distinction explicitly: local subagents are internal components, whereas A2A is appropriate when a separate standalone agent service must be contacted over the network.

#### MCP and A2A Together

A realistic architecture can use MCP for tools below agents and A2A between agent services:

```text
Issue-to-PR orchestrator
  |
  | A2A task: analyze security impact
  v
Remote security-review agent
  |
  | MCP tool calls
  +--> vulnerability database server
  +--> policy document server

Issue-to-PR orchestrator
  |
  | local tools / MCP tools
  +--> repository workspace and test execution
  +--> GitHub issue and pull-request server
```

Protocols establish interoperable edges; orchestration assembles those edges into a workflow.

### 2.3 Tool Discovery, Selection, and Scalability

A demo agent can bind five tools at startup. A production system may need internal APIs, multiple repositories, browsers, deployment environments, external SaaS applications, and specialist agents. At that scale, the main challenge changes from "can it call a function?" to "can it expose the right capability, to the right agent, for the right step?"

#### Tool Selection Strategies

| Strategy | How it works | Benefit | Cost |
| --- | --- | --- | --- |
| Fixed tool set | Every turn includes the same small menu | Simple and predictable | Does not scale to broad capability coverage |
| Role-scoped tools | Each agent role sees only relevant tools | Better safety and selection quality | Requires responsibility design |
| Dynamic retrieval | Tool descriptions are searched and loaded when needed | Keeps active menu small | A poor retrieval decision can hide a needed tool |
| Hierarchical tools | A high-level tool invokes a lower-level workflow | Reduces planning burden | Hides internals and may enlarge side effects |
| Human-gated escalation | Sensitive tools appear only after approval or task transition | Controls authority | Interrupts autonomy for high-risk actions |

For a software task, a useful progression might be:

```text
Planning phase:       issue reader, repository search, documentation
Implementation phase: file edits, shell in sandbox, test runner
Review phase:         diff inspection, test results, reviewer subagent
Delivery phase:       branch push and pull-request creation after approval
```

The agent need not see production deployment deletion APIs while it is correcting a unit test.

#### Sessions Are Part of the Tool Contract

Some calls are naturally independent: fetch issue number 42 or read a file. Others have session state:

- a shell whose working directory and running process should persist,
- a browser retaining authentication and page navigation,
- an MCP server retaining an authenticated transaction or task context,
- a remote agent performing a multi-minute delegated job.

Session management affects both correctness and cost:

| Session approach | Benefit | Risk |
| --- | --- | --- |
| New tool session for every invocation | Easier cleanup and replay | Loses state and can repeat setup cost |
| Persistent session | Efficient continuity for browser, shell, or long-lived server | State can become stale or leak across tasks |
| Checkpointed session | Can resume after interruption | Requires consistent serialization and cleanup policy |

For example, the LangChain MCP adapter documentation says its multi-server client is stateless by default, but permits an explicit persistent MCP `ClientSession` when a server must maintain context across calls. That is not merely an SDK option: it is a lifecycle decision.

#### Training Versus Integration

The paper's tool-interface discussion also covers tool-augmented training. There are two separate ways to improve tool use:

| Approach | What changes | Example objective |
| --- | --- | --- |
| Model-side training or fine-tuning | The model becomes better at selecting APIs or forming arguments | Learn to invoke tools reliably across unfamiliar tasks |
| Harness-side interface design | Descriptions, schemas, routing, output filtering, permissions, and verification change | Make the current model succeed with a smaller and clearer action space |

These methods complement each other. A model trained for tool calls can still fail with ambiguous tools or irrelevant menus. A clear harness can improve a fixed model without retraining it, which is central to the paper's harness-engineering thesis.

## 3. Design Lessons

The practical conclusions for the tool and protocol layer are:

1. **A tool is a contract, not just a capability.** Its schema, side effects, outputs, permissions, and session behavior determine whether an agent can use it reliably.
2. **More tools can reduce capability in practice.** A smaller role- and phase-specific action space often improves selection, cost, and safety.
3. **MCP and A2A solve different boundaries.** MCP connects an agent host to tools and contextual resources. A2A connects independent agent services through task-oriented communication.
4. **Protocol interoperability does not remove host responsibility.** The harness still validates arguments, scopes authority, routes execution, records observations, and decides which capabilities are available.
5. **Training and interface design complement each other.** A model can learn better tool use, but clear schemas, concise menus, policy wrappers, and observable results can improve a fixed model immediately.

Part 1 answered: **where can the agent safely act?** This post answers: **what can it call, and through which protocol boundary?**

## References

- [Agent Harness Engineering: A Survey of LLM Infrastructure](https://picrew.github.io/LLM-Harness/main.pdf)
- [Model Context Protocol: Architecture Overview](https://modelcontextprotocol.io/docs/learn/architecture)
- [Agent2Agent Protocol Documentation](https://a2a-protocol.org/latest/)
- [A2A project repository](https://github.com/a2aproject/A2A)
- [Claude Code: Tools Reference](https://code.claude.com/docs/en/tools-reference)
- [Claude Code: Connect to Tools via MCP](https://code.claude.com/docs/en/mcp)
- [Claude Agent SDK: Custom Tools](https://code.claude.com/docs/en/agent-sdk/custom-tools)
- [Claude Code: Extend Claude with Skills](https://code.claude.com/docs/en/skills)
- [OpenHands Software Agent SDK](https://docs.openhands.dev/sdk/index)
- [OpenHands Custom Tools](https://docs.openhands.dev/sdk/guides/custom-tools)
- [OpenHands Tool System and MCP](https://docs.openhands.dev/sdk/arch/tool-system)
- [OpenClaw Capabilities Overview](https://docs.openclaw.ai/tools)
- [OpenClaw Skills](https://docs.openclaw.ai/tools/skills)
- [LangChain: Model Context Protocol](https://docs.langchain.com/oss/python/langchain/mcp)
- [OpenAI API: Function Calling](https://developers.openai.com/api/docs/guides/function-calling)
- [Google ADK: MCP Tools](https://adk.dev/tools-custom/mcp-tools/)
- [Google ADK: Introduction to A2A](https://adk.dev/a2a/intro/)
