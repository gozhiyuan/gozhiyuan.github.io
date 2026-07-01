---
layout: post
title: "Agent Harness Engineering Part 1: Execution Layer and Sandboxes"
subtitle: Docker, gVisor, Firecracker, managed sandboxes, and coding-agent runtimes
categories: Large-Language-Model Agents Harness-Engineering
tags: [Blogs]
banner: "/assets/images/banners/yuanpang-wa-iceburg2.jpg"
---

# Agent Harness Engineering Part 1: Execution Layer and Sandboxes

This post is the first part of my notes on [Agent Harness Engineering: A Survey of LLM Infrastructure](https://picrew.github.io/LLM-Harness/main.pdf).

The paper's main idea is useful: for long-running agents, the model is no longer the only bottleneck. Reliability also depends on the **harness** around the model: where it runs, which tools it can call, what state it sees, how it is observed, how outputs are verified, and how permissions are governed.

The survey organizes the harness into seven layers:

| Layer | Meaning |
| --- | --- |
| E: Execution | Where agent actions physically happen: shells, filesystems, browsers, containers, VMs, sandboxes. |
| T: Tooling | How tools are described, discovered, called, and constrained. |
| C: Context | What the model sees at each step, including retrieval, memory, and compaction. |
| L: Lifecycle | The loop that carries work across turns, tasks, retries, and subagents. |
| O: Observability | Traces, logs, costs, errors, and debugging signals. |
| V: Verification | Tests, judges, task-specific checks, and trajectory evaluation. |
| G: Governance | Permissions, audit, identity, policies, and blast-radius control. |

This post focuses on **E: the execution layer**.

## 1. Why the Execution Layer Matters

An LLM agent becomes dangerous and useful at the same moment: when its text output is turned into an action.

For a coding agent, actions include reading a repository, editing files, running tests, installing packages, launching a web server, using a browser, opening network connections, and sometimes pushing code. The execution layer decides where those actions happen and what their consequences can be.

The execution layer has three jobs.

**Security.** Run untrusted or model-generated code where it cannot freely damage the host, read secrets, or exfiltrate data.

**Reproducibility.** Make agent runs resettable. Benchmarks such as SWE-bench need every task to start from a known repository state and dependency environment.

**Liveness.** Let the agent work without asking for permission every few seconds. Anthropic reported that Claude Code sandboxing reduced permission prompts by 84% in internal use, because routine Bash commands could run inside pre-defined filesystem and network boundaries rather than requiring one approval per command.

The key point: sandboxing is not just a security feature. It is also an autonomy feature.

## 2. Three Boundaries, Not One

When people say "sandbox", they often mean several different things. I find it clearer to split the execution layer into three nested boundaries.

| Boundary | Question | Examples |
| --- | --- | --- |
| Environment boundary | What can the agent see? | mounted repo, allowed directories, network allowlist, available credentials |
| Tool boundary | What can the agent ask to do? | `Read`, `Edit`, `Bash`, browser, MCP tools, deny/ask/allow rules |
| Execution boundary | Where does code physically run? | Docker container, gVisor container, Firecracker microVM, OS sandbox, hosted VM |

These are complementary. A Docker container may protect the host better than a plain local shell, but it does not know whether `rm -rf src` is logically bad for the task. A permission system may block a dangerous tool call, but if you approve the wrong command, the process still runs with whatever OS access it has.

A production harness usually layers them:

1. The **tool policy** decides whether the agent can attempt an action.
2. The **environment policy** decides what files, networks, and credentials are visible.
3. The **execution substrate** enforces CPU, memory, process, filesystem, and kernel isolation.
4. The **verification layer** checks whether the action actually improved the task.

## 3. Underlying Sandbox Technologies

If I understand the current ecosystem correctly, most agent sandboxes are built from a small set of infrastructure primitives: Docker-style containers, gVisor, Firecracker microVMs, WebAssembly, and OS-level sandboxing.

| Technology | Isolation boundary | Startup and overhead | Compatibility | Security posture for agent code | Typical agent use |
| --- | --- | --- | --- | --- | --- |
| Docker / OCI container | Namespace and cgroup isolation; shares host kernel | Fast startup; low overhead | Excellent Linux/package/tool compatibility | Good for trusted or hardened single-tenant work; weaker boundary for hostile multi-tenant code unless further isolated | Local coding agents, CI, benchmark evaluation, custom dependency images |
| gVisor | User-space kernel between container process and host kernel | Container-like startup with additional syscall overhead | High, but some low-level workloads can be affected | Stronger than ordinary containers because less host-kernel surface is exposed | Managed agent sandboxes such as Modal |
| Firecracker microVM | Hardware-virtualized VM with its own guest kernel | Fast for a VM; more overhead than a container | Strong Linux environment compatibility through prepared images | Strong isolation suited to untrusted, multi-tenant code execution | Cloud agent sandboxes such as E2B |
| OS-level sandbox | Restricted host process using primitives such as Seatbelt or `bubblewrap` | Very fast; no remote environment boot | Uses the developer's existing local tools and filesystem | Useful for limiting local command access, but remains tied to the host environment | Local interactive agents such as Claude Code sandboxed Bash |
| WebAssembly | Capability-constrained module runtime | Extremely fast and lightweight | Limited for full software engineering environments and native dependencies | Strong capability-oriented isolation for supported workloads | Small code execution, plugins, deterministic snippets |

The table is a simplification: configuration matters as much as the technology name. A well-configured container can be safer than a poorly configured VM, and network/credential exposure can defeat an otherwise strong compute boundary.

### Docker and OCI Containers

Docker is the default mental model for many coding-agent environments. It uses Linux namespaces, cgroups, filesystem layers, and container images to create a process-level boundary.

Why it is popular:

- fast startup,
- huge image ecosystem,
- easy custom dependencies,
- good benchmark reproducibility,
- familiar developer workflow.

The weakness is that a normal Docker container still shares the host kernel. For trusted CI jobs this is often acceptable. For arbitrary AI-generated code in a multi-tenant service, it may not be enough unless hardened carefully.

A minimal hardened local pattern looks like this:

```bash
docker run --rm \
  --network none \
  --read-only \
  --cap-drop ALL \
  --pids-limit 256 \
  --memory 1g \
  --cpus 1 \
  --tmpfs /tmp:rw,noexec,nosuid,size=256m \
  -v "$PWD:/workspace:ro" \
  -w /workspace \
  python:3.12-slim \
  python -c "print('hello from a constrained container')"
```

This is not a universal secure sandbox, but it shows the harness-level knobs: network, filesystem, Linux capabilities, process count, memory, CPU, and mount mode.

### gVisor

gVisor adds a stronger isolation layer between the container and the host kernel. Instead of letting the container directly exercise the host kernel surface, gVisor provides a user-space kernel that intercepts and implements much of the Linux syscall interface.

This is attractive for agent workloads because the workload still looks like a container from the outside, but the host kernel exposure is reduced. Modal documents that its Sandboxes are built on gVisor and that default sandboxes cannot accept incoming network connections or access other Modal resources by default.

The tradeoff is compatibility and performance. Some low-level syscalls, unusual filesystems, or performance-sensitive workloads may behave differently than in a normal container.

### Firecracker MicroVMs

Firecracker is a lightweight virtual machine monitor originally built for serverless computing. It creates microVMs: small VMs with hardware virtualization isolation, but with startup and resource overhead closer to containers than traditional VMs.

For agent sandboxes, the appeal is simple:

- each run can get its own kernel boundary,
- the environment can still boot quickly,
- the provider can snapshot or template images,
- untrusted code is separated from the host more strongly than with plain containers.

E2B describes its sandboxes as fast Linux VMs, and its product page says each sandbox is powered by Firecracker. This is why E2B is often grouped with "microVM-based" agent sandbox providers.

### OS-Level Sandboxes

Some tools do not start a separate container or VM. They restrict commands on the user's existing machine using OS primitives.

Claude Code's sandboxed Bash tool is an example. Anthropic says it uses OS-level primitives such as Linux `bubblewrap` and macOS Seatbelt to enforce filesystem and network boundaries for Bash commands and their child processes. This is lighter than booting a container, but it is scoped: in Claude Code docs, sandboxing applies to Bash commands, while the broader permission system applies to Bash, Read, Edit, WebFetch, MCP, and other tools.

This distinction matters. A local OS sandbox can reduce approval fatigue for shell commands, but it is not the same thing as running the whole agent inside a disposable cloud VM.

### WebAssembly

WebAssembly is another useful substrate, especially for small deterministic code snippets or plugin systems. It can start very quickly and expose a capability-style interface. The limitation is ecosystem compatibility: many agent tasks want `apt`, `pip`, browsers, databases, native libraries, and long-running processes. That pushes most software-engineering agents toward containers or VMs.

## 4. Managed Sandbox Services

On top of these primitives, there is a growing layer of sandbox-as-a-service products. These are not just "Docker in the cloud". The product value is usually lifecycle management: create, exec, stream logs, upload files, expose ports, snapshot state, terminate, and scale to many concurrent sandboxes.

### How Does Code Get Into the Remote Sandbox?

A natural question is: when we call Modal, E2B, or Daytona, do we tar up all our scripts, upload them through an API, and run them remotely?

Sometimes, but that is only one transfer strategy. The more general abstraction is that the harness creates or attaches to a remote isolated environment, makes the relevant code and data available there, executes commands there, and returns observations to the agent loop.

```text
Your application / agent loop
        |
        | create sandbox through SDK/API
        v
Remote isolated environment
        |
        | obtain repository, scripts, data, and dependencies
        | execute commands and tests
        | produce stdout, stderr, exit codes, and diffs
        v
Your application / agent loop
```

The application can provision the sandbox contents in several ways:

| Method | How it works | Best for |
| --- | --- | --- |
| Prebuilt image or template | Dependencies and base tools are installed before a task starts | Repeated workloads, benchmarks, low startup latency |
| Git clone inside sandbox | The sandbox clones a repository directly, possibly at a pinned commit | Coding agents modifying a repository |
| File upload API | The orchestrator writes scripts, patches, config, or input files into the remote filesystem | Small generated files and user data |
| Persistent volume | A mounted storage volume is reused or shared across sandbox runs | Package caches, datasets, durable workspaces |
| Download from object storage or URL | The sandbox retrieves large artifacts after startup | Large datasets, model weights, archives |
| Tar or zip upload | The orchestrator packages many local files and extracts them remotely | A local working tree with uncommitted changes or bulk transfer |

For a coding-agent task, a typical flow is closer to cloning a repository and applying generated edits than re-uploading every script for every command:

```python
sandbox = provider.create(template="python-dev")

try:
    sandbox.exec("git clone https://github.com/org/repo /workspace/repo")
    sandbox.write_file("/workspace/repo/fix.patch", model_generated_patch)
    sandbox.exec("cd /workspace/repo && git apply fix.patch && pytest")

    diff = sandbox.exec("cd /workspace/repo && git diff")
    print(diff.stdout)
finally:
    sandbox.delete()
```

The commands run **inside the remote sandbox**, not on the machine running the model loop. The harness sends an action, receives command output or filesystem results, and uses those observations to decide the next action.

Provider APIs expose different choices:

| Service | Common ways to place code/data in the sandbox |
| --- | --- |
| Modal | Filesystem API, mounted Volumes or bucket mounts, image setup, commands that clone/download files |
| E2B | Custom templates, `files.write(...)`, and shell commands such as `git clone`; individual file transfer is part of its API |
| Daytona | Filesystem APIs, built-in Git operations, snapshots/images, and process execution inside a workspace |

A tarball is still useful when the input is a large local directory that is not represented by a remote Git commit:

```python
sandbox.upload_file("repo.tar.gz", "/tmp/repo.tar.gz")
sandbox.exec("mkdir -p /workspace && tar -xzf /tmp/repo.tar.gz -C /workspace")
sandbox.exec("cd /workspace/repo && pytest")
```

But the central idea is not "upload scripts." It is **remote controlled execution**: the agent's control loop remains in one place, while potentially unsafe shell and filesystem operations happen in a disposable or policy-constrained environment.

### If the Agent Changes Remote Files, How Do They Return?

Yes: when the execution environment is remote, edits made there do not automatically appear in a local checkout. If those edits should become part of the developer's working tree, the harness needs an explicit result-export or synchronization step.

In practice, synchronization usually happens after a useful checkpoint, such as after tests pass or when the agent proposes a final patch. Synchronizing after every command would add latency and create conflicts while the remote agent is still exploring.

```text
Local orchestrator                         Remote sandbox
------------------                         --------------
create or attach sandbox       --------->  clone repo / receive files
send agent tool actions        --------->  edit files, run tests
receive observations            <---------  stdout, stderr, test results
retrieve accepted changes      <---------  patch, files, commit, or archive
apply/review locally
```

Common result-export strategies are:

| Strategy | How remote changes return | Best for |
| --- | --- | --- |
| Read changed files | Use the provider filesystem API to download selected outputs | A few generated files or reports |
| Return a Git patch | Run `git diff --binary` remotely, download the patch, and apply it locally | Coding agents editing an existing checkout |
| Push a branch or open a pull request | Commit and push changes from the sandbox using scoped Git credentials | Hosted agents and collaborative code review |
| Download an archive | Package an output directory and retrieve it | Many generated files or build artifacts |
| Leave workspace remote until acceptance | Keep results in the cloud workspace and only export selected work | Hosted IDE or browser-based agents |

For software engineering tasks, returning a Git patch is often the cleanest design because it preserves reviewability and does not require blindly overwriting local files:

```python
sandbox = provider.create(template="python-dev")

try:
    sandbox.exec("git clone https://github.com/org/repo /workspace/repo")

    # The agent edits files remotely and verifies the change.
    sandbox.exec("cd /workspace/repo && pytest")

    # Export only the resulting change set.
    result = sandbox.exec("cd /workspace/repo && git diff --binary")
    patch = result.stdout

    # The local harness can show, approve, then apply this patch locally.
    local_workspace.apply_patch(patch)
finally:
    sandbox.delete()
```

Another common workflow treats Git as the synchronization protocol:

```text
Remote sandbox:  clone -> edit -> test -> commit -> push agent branch
Local checkout:  fetch branch -> inspect diff -> merge or cherry-pick
```

This distinction also explains why local and cloud coding agents feel different:

| Execution mode | Where edits occur | Is remote-to-local synchronization needed? |
| --- | --- | --- |
| Local Claude Code with sandboxed Bash | In the permitted local working directory | No: the edited files are already local |
| Claude Code on the web | In an isolated cloud session | Yes: changes need to be committed, patched, or otherwise exported |
| Claude Managed Agents with Daytona | In the assigned Daytona sandbox | Yes: the orchestrator must retrieve files/diffs or use a Git workflow |
| E2B or Modal coding-agent runtime | In the remote VM/container filesystem or mounted storage | Yes, unless the intended result remains entirely remote |

Therefore, a managed sandbox has two data paths: **input provisioning** sends the repository or task data into the environment, and **result export** returns accepted edits, patches, artifacts, or commits after execution.

### Are Managed Sandboxes Only for Coding Agents?

No. Coding agents are a prominent use case because terminal and filesystem execution are easy to expose through an SDK, but a managed sandbox can host any environment its image and APIs support: browsers, local web applications, virtual desktops, GUI programs, data-analysis kernels, test devices, or computer-use agents.

| Service | Shell and code execution | Browser or visual-agent support | Computer-use support |
| --- | --- | --- | --- |
| Modal | Yes | Can run Chromium, Playwright, web UIs, and VNC stacks inside a Sandbox | Modal documents running Anthropic's Computer Use demo in a Sandbox |
| E2B | Yes | Desktop sandboxes can host browsers and stream the display through VNC | E2B documents desktop agents that view screenshots, click, type, and scroll |
| Daytona | Yes | Its desktop environment can run browser and GUI applications | Daytona exposes Computer Use operations for mouse, keyboard, screenshots, recordings, display, and VNC |

A VLM-based computer-use loop has the same execution-layer idea as a coding agent, but the observation is a screenshot instead of terminal output:

```text
Remote desktop sandbox
  - browser and applications
  - virtual display
  - mouse and keyboard execution API
        |
        | screenshot
        v
VLM agent loop
  - inspect visual state
  - decide click/type/scroll action
        |
        | GUI action
        v
Remote desktop sandbox
```

The sandbox is the computer being operated. It protects the surrounding infrastructure from web content, downloaded files, GUI automation, or arbitrary code executed during the task.

### Where Does the Agent Loop Run?

A sandbox, an agent loop, and model inference are separate components:

| Component | Responsibility |
| --- | --- |
| Model inference | Interpret context or screenshots and select the next action |
| Agent loop / orchestrator | Maintain task state, call the model, validate tool calls, and route observations |
| Sandbox tool environment | Execute Bash commands, manipulate files, run browsers/desktops, and return results |

In many production designs, the loop stays outside the sandbox while only tools run inside it:

```text
Application / orchestration service
  - task state, policy, budgets, audit log
  - calls hosted LLM or VLM API
          |
          | approved command / file / GUI action
          v
Managed sandbox
  - repository or desktop
  - shell, browser, applications
  - screenshots, logs, files
          |
          | observation
          v
Application / orchestration service
```

For computer use, the orchestrator may request a screenshot from the sandbox, send that screenshot to a VLM, receive a `click`, `type`, or `scroll` decision, and forward the action back to the sandbox. In this architecture, the model does not need to be installed in the sandbox. The sandbox holds the environment the model controls.

Keeping the loop outside has practical advantages:

| Benefit | Why it matters |
| --- | --- |
| Secret isolation | Model API keys and control-plane credentials do not have to live in an environment executing risky tools |
| Policy enforcement | The orchestrator can reject or log actions before they affect the sandbox |
| Session recovery | State can persist even when a disposable sandbox is terminated and recreated |
| Backend portability | The same loop can direct Modal, E2B, Daytona, or an internal environment |
| Independent scaling | Model requests and execution environments can be allocated separately |

Anthropic documents this split for Claude Managed Agents self-hosted sandboxes: orchestration remains on Anthropic's side, while tool execution, files, and network egress move into the developer's infrastructure.

It is also possible to run the agent application inside the sandbox:

```text
Managed sandbox
  - agent process
  - repository or desktop
  - local services
  - browser and VNC stack
        |
        | model API call
        v
Hosted LLM / VLM inference service
```

This can be useful when a background agent needs low-latency access to a full development stack or desktop. Modal's official example runs Anthropic's Computer Use demo application and VNC desktop in a Modal Sandbox. The model inference can still be a remote API call: running the agent program inside the sandbox does not imply that the foundation model weights are hosted there.

The model itself runs inside the sandbox only when the developer explicitly deploys local inference, for example an open-source VLM on GPU infrastructure. That is possible, but it requires much more compute, image management, and security engineering than calling a hosted model API.

| Agent deployment | What is in the sandbox? | Where is the loop likely to run? | Where is model inference likely to run? |
| --- | --- | --- | --- |
| Coding tool runtime | Repo, shell, dependencies, tests | External orchestrator | Hosted model API |
| Browser or VLM computer-use runtime | Virtual desktop, browser, screenshot/action API | External orchestrator | Hosted multimodal API |
| Hosted development workspace | Repo, services, browser, VNC, possibly agent process | Inside workspace or split across services | Usually hosted model API |
| Fully self-hosted agent stack | Tools, agent process, and local model server | Inside controlled infrastructure | Local GPU model deployment |

### Modal

[Modal Sandboxes](https://modal.com/docs/guide/sandboxes) are secure containers for running untrusted user or agent code. Modal's docs explicitly mention LLM-generated code, arbitrary dependencies, checking out repositories, and running test suites as use cases. Its networking/security docs say Modal Sandboxes are built on gVisor.

Minimal Modal example:

```python
import modal

app = modal.App.lookup("agent-sandbox-demo", create_if_missing=True)
image = modal.Image.debian_slim().pip_install("pytest")

sb = modal.Sandbox.create(app=app, image=image, timeout=60)

try:
    proc = sb.exec("python", "-c", "print('hello from Modal sandbox')")
    print(proc.stdout.read())

    proc = sb.exec(
        "bash",
        "-lc",
        "python - <<'PY'\nimport sys\nprint(sys.version)\nPY",
        timeout=10,
    )
    print(proc.stdout.read())
finally:
    sb.terminate()
```

For an agent harness, the agent loop would not call `subprocess.run()` locally. It would call `sb.exec(...)`, capture stdout/stderr/exit code, and feed the observation back to the model.

Modal is not limited to code execution. Its documentation includes an Anthropic Computer Use demo that launches a desktop/VNC environment in a Sandbox and exposes ports for a Streamlit agent interface and a noVNC display.

### E2B

[E2B](https://e2b.dev/docs) provides secure cloud sandboxes for agents. The docs describe a sandbox as a fast, secure Linux VM created on demand, and the product page says E2B sandboxes are powered by Firecracker microVMs.

Minimal E2B example:

```python
from e2b import Sandbox

sbx = Sandbox.create()  # requires E2B_API_KEY

try:
    result = sbx.commands.run("python3 - <<'PY'\nprint('hello from E2B')\nPY")
    print(result.stdout)

    result = sbx.commands.run("mkdir -p /tmp/agent && echo ok > /tmp/agent/out.txt")
    print("exit:", result.exit_code)
finally:
    sbx.kill()
```

The interesting design point is the control plane. Your product can keep the LLM, planner, user session, and billing logic in your app, while code execution happens inside a short-lived remote VM.

E2B also provides Desktop sandboxes for computer-use agents. In this mode, the remote environment contains an Ubuntu desktop and applications; an agent can consume visual observations and issue mouse, keyboard, and scrolling actions while a user observes the desktop through VNC streaming.

### Daytona

[Daytona](https://www.daytona.io/docs/en/) is an open-source sandbox platform for AI-generated code. Its docs describe sandboxes as isolated environments with a dedicated kernel, filesystem, network stack, vCPU, RAM, and disk, with OCI/Docker compatibility. The SDK exposes filesystem, Git, process execution, code interpreter, and computer-use operations.

Minimal Daytona example:

```python
from daytona import Daytona

daytona = Daytona()  # uses DAYTONA_API_KEY or configured credentials
sandbox = daytona.create()

try:
    response = sandbox.process.exec("echo 'hello from Daytona'")
    print(response.result)

    response = sandbox.process.code_run("""
def greet(name):
    return f"hello, {name}"

print(greet("agent sandbox"))
""")
    print(response.result)
finally:
    sandbox.delete()
```

Daytona is especially relevant to Claude Managed Agents. Daytona's guide says that Anthropic runs the API, agent loop, and work queue; your orchestrator manages sandbox lifecycle; and Daytona provides the sandbox containers where filesystem and shell tools execute. In that design, Bash/read/write/edit/glob/grep happen inside Daytona, while web tools and MCP tools are dispatched by Anthropic server-side.

This is a good example of execution-layer separation: the model orchestration and the shell/filesystem execution do not have to live in the same infrastructure.

Daytona's broader platform also exposes Computer Use. The desktop runs in its sandbox, and SDK operations can control mouse/keyboard input, capture screenshots or recordings, and provide VNC access for visual inspection.

## 5. Benchmark and Framework Sandboxes

The execution layer is also central to evaluation frameworks and open-source coding agents.

### SWE-bench

SWE-bench evaluates whether an agent can fix real GitHub issues. The benchmark is only meaningful because each task can be executed in a controlled environment.

Conceptually, a SWE-bench task contains:

- a repository and commit,
- an issue statement,
- a patch generated by an agent,
- environment setup scripts,
- test scripts,
- `FAIL_TO_PASS` tests that should become passing,
- `PASS_TO_PASS` tests that should remain passing.

The harness builds Docker images and runs evaluation inside containers. The `TestSpec` object in the SWE-bench harness contains fields such as `repo_script_list`, `env_script_list`, `eval_script_list`, `FAIL_TO_PASS`, `PASS_TO_PASS`, and Docker image tags.

Typical evaluation flow:

```bash
python -m swebench.harness.run_evaluation \
  --dataset_name princeton-nlp/SWE-bench_Verified \
  --predictions_path predictions.jsonl \
  --max_workers 8 \
  --run_id my-agent-run
```

A prediction file is usually a JSONL file mapping an instance id to a generated patch:

```json
{"instance_id":"django__django-12345","model_patch":"diff --git a/..."}
```

The important point is that SWE-bench is not just a dataset. It is an execution harness: reset repo, apply patch, install deps, run tests, parse logs, report pass/fail.

#### Reproducibility Is Not the Same as Maximum Isolation

SWE-bench's default evaluation design relies on Docker containers. This is a sensible choice for the benchmark's primary objective: reproducibly checking whether a generated patch fixes a real issue in the expected dependency environment.

```text
SWE-bench instance
      |
      v
Repository-specific Docker image
      |
      | create clean container
      | apply generated patch
      | execute tests
      v
Resolved / not resolved
```

However, a model-generated patch is executable input. A patch can modify code that tests import and run. If evaluation is offered as a public service accepting arbitrary submissions, the threat model is no longer only "does the patch pass?" It also includes "could the patch attack the evaluator?"

| Concern | Docker evaluation container | Firecracker or gVisor outer sandbox |
| --- | --- | --- |
| Primary purpose | Recreate repository and test dependencies | Protect hosted infrastructure from untrusted execution |
| Kernel isolation on native Linux | Shares the host kernel | Adds a stronger kernel or syscall isolation boundary |
| Startup and operational complexity | Lower | Higher |
| Good use case | Local research runs and controlled evaluation | Public or multi-tenant agent/evaluation service |

Therefore, Docker is not a problem simply because SWE-bench uses it. Docker is doing the benchmark-reproducibility job. A security-sensitive hosted evaluator may add another isolation layer around that same benchmark container:

```text
Managed gVisor sandbox or Firecracker microVM
      |
      v
SWE-bench Docker evaluation environment
      |
      v
Apply patch and run official tests
```

The inner Docker image preserves comparable SWE-bench test conditions. The outer sandbox protects the hosting platform from hostile or accidentally harmful generated code. Even for local runs, the evaluation environment should avoid mounted credentials, Docker socket access, unnecessary network access, and unlimited CPU or memory.

### SWE-ReX

[SWE-ReX](https://github.com/SWE-agent/SWE-ReX) is an open-source runtime interface for sandboxed shell environments. It powers SWE-agent and is designed to decouple an agent's command-execution logic from the infrastructure where commands actually run. Its documented deployments include local/container and hosted backends such as Modal, AWS/Fargate, and Daytona.

It is useful to separate three related projects:

| Component | What it provides | What it does not provide |
| --- | --- | --- |
| SWE-bench | Repository/issue tasks and the official patch evaluation harness | An interactive agent runtime |
| SWE-agent or another coding agent | The reasoning loop that explores a repo, edits files, and proposes a patch | The benchmark definition or infrastructure portability layer |
| SWE-ReX | A portable runtime API for commands, sessions, and files on a selected backend | New benchmark tasks or an automatic guarantee of stronger security |

The flow for a SWE-bench run using an agent and SWE-ReX is:

```text
SWE-bench task: repository + issue statement
      |
      v
SWE-agent or your own coding agent
      |
      v
SWE-ReX runtime API
      |
      | Docker / Modal / Daytona / Fargate backend
      | inspect files, edit code, run tests
      v
Candidate patch
      |
      v
SWE-bench evaluation harness
      |
      v
Benchmark score
```

The task is still a SWE-bench task, and the final score is still determined by SWE-bench. SWE-ReX is the execution wrapper used while an interactive agent works on that task. It can also be used for developer-defined tasks unrelated to SWE-bench.

The portability benefit is that the agent can ask for shell execution through one interface while deployment can change independently:

| Development or deployment need | Possible SWE-ReX backend | Security consequence |
| --- | --- | --- |
| Fast local development | Local Docker | Retains ordinary Docker/container trust assumptions |
| Parallel cloud agent runs | Modal | Uses Modal's hosted sandbox boundary, documented as gVisor-based |
| Managed workspaces | Daytona | Uses Daytona-provided isolated workspaces; exact isolation depends on its deployment |
| Custom cloud operation | AWS/Fargate | Depends on the configured cloud runtime and network/credential policy |

SWE-ReX therefore does not automatically fix the Docker-versus-microVM concern. It makes the backend replaceable. A developer can use Docker during local development and choose a stronger hosted execution boundary for untrusted or large-scale workloads without rewriting the agent loop around a provider-specific API.

Minimal SWE-ReX pattern:

```python
import asyncio

from swerex.deployment.docker import DockerDeployment


async def main():
    deployment = DockerDeployment(image="python:3.12")
    await deployment.start()
    try:
        runtime = deployment.runtime
        result = await runtime.run_in_session("python -c \"print('hello from SWE-ReX')\"")
        print(result)
    finally:
        await deployment.stop()


asyncio.run(main())
```

The exact runtime call names may change across versions, but the architectural shape is stable:

1. create a deployment,
2. start a sandbox,
3. send commands through a runtime interface,
4. collect stdout/stderr/exit code,
5. stop and clean up.

That is the same shape a coding-agent harness needs.

#### Is SWE-ReX Only for Coding Agents?

SWE-ReX is most naturally used by coding agents because its common interface focuses on commands, persistent shell sessions, and filesystem operations. It is also suitable for terminal-driven debugging, data analysis, and security tasks where the relevant actions can be represented as commands and files.

| Capability | SWE-ReX common runtime interface |
| --- | --- |
| Run shell commands and tests | Yes |
| Keep an interactive shell/session alive | Yes |
| Transfer or edit files | Yes |
| Run a script that itself uses a headless browser | Possible, if the sandbox image contains the dependencies |
| First-class browser navigation and DOM observation tools | Not documented as part of the common runtime API |
| Screenshot-based mouse and keyboard computer use | Not documented as part of the common runtime API |

For example, a sandbox with Node and Playwright installed could execute a scripted browser test through a command:

```python
result = await runtime.run_in_session("node ./run_playwright_test.js")
```

But SWE-ReX does not itself define an agent interaction loop such as `observe screenshot -> click -> type -> scroll`. A product building browser-use or desktop-use agents needs an additional tool layer, or it needs to use backend-specific capabilities directly. Daytona may expose broader computer-use features as a platform, but a portable SWE-ReX shell interface should not be assumed to expose every provider-specific capability.

### OpenHands

[OpenHands](https://docs.openhands.dev/usage/runtimes/overview) belongs at a different architectural level than SWE-ReX. SWE-ReX is a runtime abstraction that a developer can use while implementing an agent. OpenHands is an open-source software-agent platform: it includes model integration, an agent loop, tool routing, conversation or task state, user interfaces and SDKs, and the sandbox in which actions execute.

That makes OpenHands conceptually closer to Claude Code than to Modal, Daytona, E2B, or SWE-ReX:

| System | Architectural role | Provides an agent loop? | Provides or chooses execution environments? |
| --- | --- | --- | --- |
| Claude Code | Integrated coding-agent product and SDK ecosystem | Yes | Yes: local sandboxing and hosted/managed deployment modes |
| OpenHands | Open-source software-engineering agent platform | Yes | Yes: Docker, Process, or Remote sandbox providers in V1 |
| SWE-ReX | Portable command/filesystem runtime library | No | Selects execution backends for an agent built elsewhere |
| Modal / E2B / Daytona | Managed execution infrastructure | No, unless an application is deployed into it | Supplies remote sandbox environments and lifecycle APIs |
| SWE-bench | Benchmark task and evaluation harness | No | Provides reproducible test execution for scoring |

The high-level OpenHands shape is:

```text
OpenHands application
  - LLM integration
  - agent loop and conversation state
  - tool selection and observations
  - UI, CLI, or SDK surface
        |
        v
OpenHands sandbox
  - repository filesystem
  - terminal commands
  - local development servers
  - browser tools and optional visual access
        |
        v
Docker / Process / Remote sandbox provider
```

This is why OpenHands needs both a model and a sandbox. The model decides which action to attempt; the sandbox is where that action affects files, commands, and applications.

#### Model Credentials

To operate as an agent, OpenHands must be connected to an LLM. For hosted models, this normally requires model-provider credentials. Its documentation says OpenHands can connect to LLMs supported through its model stack and documents providers including the OpenHands LLM provider, Anthropic, OpenAI, and local OpenAI-compatible endpoints.

| Deployment | Credential requirement |
| --- | --- |
| OpenHands with Anthropic, OpenAI, or another hosted provider | Provider API key configured in OpenHands |
| OpenHands using the OpenHands LLM provider | OpenHands LLM API key backed by OpenHands Cloud credits |
| Programmatic control of OpenHands Cloud | OpenHands Cloud API key, which is distinct from the LLM key |
| OpenHands with local Ollama | No paid external inference key; configuration may use a placeholder key such as `local-llm` |
| OpenHands with local vLLM or SGLang | The token configured for the local model endpoint, if one is enforced |

For a self-hosted instance using a hosted model, the conceptual configuration is:

```bash
export LLM_MODEL="anthropic/claude-sonnet-4-5-20250929"
export LLM_API_KEY="your-provider-api-key"
```

The sandbox does not replace the model service:

```text
LLM credentials -> authorize reasoning/inference calls
Sandbox         -> execute the selected tools in a bounded environment
```

#### OpenHands Sandbox Providers

OpenHands V1 calls the action environment a **sandbox**. It is where OpenHands runs commands, edits files, and starts servers while working on a task. Current documentation describes three sandbox providers:

| Provider | Where work executes | Isolation and use case |
| --- | --- | --- |
| Docker sandbox | Agent server and workspace run inside a Docker container | Recommended local option; provides isolation and reproducibility |
| Process sandbox | Agent server runs as a normal local process | Faster for controlled debugging, but explicitly unsafe because it has host-user access |
| Remote sandbox | Agent server runs in a remote execution environment | Used in OpenHands Cloud and advanced self-hosted/managed deployments |

In current deployments the provider may still be selected through the legacy `RUNTIME` variable:

```bash
export RUNTIME=docker   # default/recommended local sandbox
# export RUNTIME=process  # fast but no container isolation
# export RUNTIME=remote   # configured remote sandbox endpoint
```

For the local Docker sandbox, a repository can be mounted into the container so the OpenHands agent directly updates that workspace:

```bash
openhands serve --mount-cwd
```

or explicitly:

```bash
export SANDBOX_VOLUMES="$PWD:/workspace:rw"
openhands serve
```

This is local mounted-workspace behavior: if the container modifies `/workspace`, the permitted host repository changes immediately because it is a read-write bind mount. By contrast, an OpenHands remote sandbox needs the same result-export mechanisms described earlier: patch retrieval, branch/PR workflow, downloaded files, or provider-specific persistence.

For extra tools and dependencies, the sandbox image can also be customized:

```toml
[sandbox]
base_container_image = "my-openhands-sandbox:latest"
```

#### More Than Shell Execution: Browser-Enabled Workspaces

OpenHands is still primarily a software-development agent platform, not a universal visual-desktop automation platform. However, it exposes richer software-agent tooling than SWE-ReX's portable shell/filesystem runtime.

Its SDK Docker sandbox documentation includes browser-enabled workspaces. In that configuration, OpenHands can run a browser inside the isolated workspace and enable browser tools for the agent:

```python
agent = get_default_agent(
    llm=llm,
    cli_mode=False,  # enables browser tools
)
```

With a browser-enabled Docker workspace, OpenHands can use browser interaction during a development task, for example to inspect a locally launched web application. The documentation also describes VNC access so a developer can visually watch the browser operating inside the sandbox.

| Capability | SWE-ReX | OpenHands |
| --- | --- | --- |
| Execute shell commands and tests | Yes | Yes |
| Read/write source files | Yes | Yes |
| Provide a complete agent loop | No | Yes |
| Start and inspect an application server | Possible through commands | Integrated into agent workflows |
| Browser tools in the sandbox | Not a documented common runtime primitive | Documented in browser-enabled SDK sandbox setup |
| Human visual observation through VNC | Not a core runtime abstraction | Documented for browser-enabled Docker workspace |
| General-purpose desktop computer-use API | Not a core runtime abstraction | Not the primary platform abstraction; use specialized desktop sandbox infrastructure when needed |

The distinction is important. OpenHands can browse and interact with software during an engineering workflow. For a general VLM agent controlling arbitrary desktop applications through screenshot/mouse/keyboard primitives, Daytona Computer Use, E2B Desktop, or a purpose-built computer-use environment is a clearer substrate.

#### How an OpenHands Session Runs

A simplified local OpenHands task looks like:

```text
User asks OpenHands to fix or build something
        |
        v
OpenHands sends current context to configured LLM
        |
        v
LLM proposes a tool action
        |
        v
Docker sandbox executes:
  read/edit file, Bash command, run server, browser action
        |
        v
Observation returns to OpenHands and then to the LLM
        |
        v
Repeat until completion or user intervention
```

For local Docker with a mounted repository, approved source edits appear in the user's checkout. For Remote sandbox deployment, work is isolated remotely and must be exported or committed before it appears locally.

So the short characterization is:

> SWE-ReX is a lower-level portable execution interface. OpenHands is an open-source coding-agent harness, comparable in architectural level to Claude Code, whose sandbox provides the controlled workspace where its terminal, file, server, and browser actions occur.

## 6. Claude Code as a Harness System

OpenHands and Claude Code belong at the same architectural level: both are complete software-agent harnesses, not merely sandbox APIs. They combine a model-facing loop with tools, state, execution boundaries, and a user-facing workflow. The execution design differs depending on how Claude is deployed.

| Mode | Where actions execute | Main isolation mechanism | Where resulting files live |
| --- | --- | --- | --- |
| Local Claude Code | On the user's machine | Permissions plus OS-level sandboxing for Bash subprocesses | In the local permitted workspace |
| Claude Code on the web | In an isolated hosted environment | Anthropic-managed cloud isolation and scoped Git access | In the hosted session until exported through Git or another workflow |
| Claude Managed Agents cloud environment | In a per-session Anthropic-managed container | Managed cloud container configuration | In the cloud environment/output path |
| Claude Managed Agents self-hosted environment | In the customer's worker or sandbox provider | Customer-selected isolation, such as a container or managed sandbox | In customer-controlled infrastructure |

For the local product, Anthropic documents an important boundary:

- Permissions apply before tool execution across tools such as Bash, Read, Edit, WebFetch, and MCP.
- The OS-level sandbox applies to **Bash commands and their child processes**, using Seatbelt on macOS and `bubblewrap` on Linux.
- Built-in file tools such as Read, Edit, and Write are controlled by permissions rather than the Bash sandbox.
- Computer use on the local machine controls the actual desktop; it is not isolated by the Bash sandbox.

This is why the phrase "Claude Code has a sandbox" should not be read as "every possible agent action runs in a remote VM." Locally, it is a layered design: tool permissions plus OS isolation for command execution.

For hosted agents, the topology changes. Claude Managed Agents defines an environment where tool execution happens. By default this is an Anthropic-managed cloud container. With a self-hosted sandbox, Anthropic retains orchestration while the developer supplies the environment worker and determines filesystem mounts, network egress, runtime hardening, and result retrieval. Daytona documents one integration for this self-hosted path.

```text
Anthropic orchestration / model-facing agent loop
        |
        | tool execution work item
        v
Self-hosted environment worker
        |
        | start or attach sandbox
        v
Daytona / Modal / custom container or VM
  - files, commands, tests, network policy
        |
        | tool results and final outputs
        v
Anthropic orchestration
```

This does **not** mean that ordinary local Claude Code automatically sends work to Daytona. Daytona is relevant to the Managed Agents self-hosted environment design.

### A Local Configuration Pattern

The following illustrates a restrictive local Claude Code posture. The current working directory is writable by default for sandboxed Bash; additional write paths should be granted narrowly. `allowUnsandboxedCommands: false` disables the normal escape hatch that could otherwise ask to rerun a blocked command outside the sandbox.

```json
{
  "sandbox": {
    "enabled": true,
    "failIfUnavailable": true,
    "autoAllowBashIfSandboxed": true,
    "allowUnsandboxedCommands": false,
    "filesystem": {
      "allowWrite": ["/tmp/agent-work"],
      "denyRead": ["~/.ssh", "~/.aws", "~/.kube", "~/.config/gh"]
    },
    "network": {
      "allowedDomains": ["github.com", "registry.npmjs.org", "pypi.org"]
    }
  },
  "permissions": {
    "deny": [
      "Read(~/.ssh/**)",
      "Read(~/.aws/**)",
      "Read(~/.kube/**)",
      "Read(~/.config/gh/**)"
    ]
  }
}
```

Network policy needs care. A broad denied-domain rule can override an allowlist, so a policy that intends to permit package registries should not also deny every domain. In real deployments, test the configuration against the specific build tools the agent is allowed to use.

## 7. An Execution Adapter Is Not a Complete Harness

A custom agent can isolate provider-specific execution calls behind an interface. This is approximately the role SWE-ReX plays for command-oriented agents:

```python
from dataclasses import dataclass
from typing import Protocol


@dataclass
class ExecResult:
    stdout: str
    stderr: str
    exit_code: int


class Sandbox(Protocol):
    def exec(self, command: str, timeout: int = 60) -> ExecResult:
        ...

    def write_file(self, path: str, content: str) -> None:
        ...

    def read_file(self, path: str) -> str:
        ...

    def export_patch(self) -> str:
        ...

    def close(self) -> None:
        ...
```

But this adapter only represents the execution portion of the harness. A real loop should validate an action before sending it to the sandbox and should explicitly export accepted results:

```python
def run_agent_step(model, policy, sandbox: Sandbox, task: str, transcript: list[str]) -> str:
    action = model.next_action(build_prompt(task, transcript))
    policy.authorize(action)

    if action["type"] == "bash":
        result = sandbox.exec(action["command"], timeout=action.get("timeout", 60))
        observation = (
            f"exit={result.exit_code}\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )
    elif action["type"] == "write_file":
        sandbox.write_file(action["path"], action["content"])
        observation = f"wrote {action['path']}"
    else:
        raise ValueError(f"unsupported action: {action['type']}")

    transcript.append(f"action: {action}\nobservation: {observation}")
    return observation


def finalize_work(sandbox: Sandbox) -> str:
    return sandbox.export_patch()
```

This small example separates three concerns:

| Concern | Minimal code element | Real system responsibility |
| --- | --- | --- |
| Decision | `model.next_action(...)` | Prompt, context, memory, lifecycle |
| Control | `policy.authorize(...)` | Permissions, egress policy, approval, audit |
| Execution | `sandbox.exec(...)` | Docker, Modal, E2B, Daytona, SWE-ReX backend |
| Result transfer | `sandbox.export_patch()` | Review, Git branch, artifact retrieval, local sync |

It also highlights why a complete product such as OpenHands or Claude Code is larger than a sandbox SDK. The product must implement context management, tool definitions, permissions, observation handling, verification, result delivery, and a user workflow in addition to starting an isolated process.

## 8. Design Lessons

The execution layer is where an agent's proposed actions become real effects. The most useful design questions are practical:

| Question | Design consequence |
| --- | --- |
| Is code trusted, generated, or submitted by unknown users? | Choose an isolation boundary proportionate to the threat model; hosted untrusted execution usually needs more than default Docker. |
| Must each run be reproducible? | Use images, templates, pinned commits, resettable workspaces, and deterministic evaluation scripts. |
| Does the agent edit local files or remote files? | Define the result path in advance: mounted workspace, returned patch, artifact download, or Git branch/PR. |
| Which secrets and network endpoints are required? | Do not mount broad credentials; explicitly restrict reads, writes, and outbound access. |
| Does the task require browser or desktop interaction? | Select an environment that exposes screenshots and actions, not only shell execution. |
| How will a bad action be noticed? | Capture commands, outputs, diffs, resource consumption, network activity, approvals, and test results. |

Four distinctions summarize this layer:

1. **Reproducibility is not security.** Docker images are excellent reproducible testbeds; hostile multi-tenant execution may require gVisor, microVMs, or another stronger outer boundary.
2. **Permissions are not isolation.** A policy decides which actions should run; a sandbox limits damage if a permitted action is unsafe.
3. **Infrastructure is not the agent harness.** Modal, E2B, and Daytona supply execution environments; OpenHands and Claude Code integrate those kinds of environments into complete agent systems.
4. **Remote execution needs a return path.** An agent has not completed a software change until the patch, branch, artifact, or approved file update reaches the user or repository.

The next execution-layer design decision is not simply "which sandbox is fastest?" It is which combination of tool policy, environment boundary, runtime isolation, result export, and verification gives the agent enough freedom to be productive without giving failures an uncontrolled blast radius.

## References

- [Agent Harness Engineering: A Survey of LLM Infrastructure](https://picrew.github.io/LLM-Harness/main.pdf)
- [Modal Sandboxes](https://modal.com/docs/guide/sandboxes)
- [Modal Sandbox networking and security](https://modal.com/docs/guide/sandbox-networking)
- [Modal: Run Anthropic's computer use demo in a Sandbox](https://modal.com/docs/examples/anthropic_computer_use)
- [E2B documentation](https://e2b.dev/docs)
- [E2B product page](https://e2b.dev/)
- [E2B Computer Use](https://e2b.dev/docs/use-cases/computer-use)
- [Daytona documentation](https://www.daytona.io/docs/en/)
- [Daytona process and code execution](https://www.daytona.io/docs/en/process-code-execution/)
- [Daytona Computer Use](https://www.daytona.io/docs/en/computer-use/)
- [Claude Managed Agents on Daytona](https://www.daytona.io/docs/en/guides/claude/claude-managed-agents/)
- [Claude Agent SDK agent loop](https://platform.claude.com/docs/en/agent-sdk/agent-loop)
- [Claude Code sandboxing docs](https://code.claude.com/docs/en/sandboxing)
- [Claude Code permissions docs](https://code.claude.com/docs/en/permissions)
- [Anthropic: Beyond permission prompts](https://www.anthropic.com/engineering/claude-code-sandboxing)
- [Claude Managed Agents self-hosted sandboxes](https://platform.claude.com/docs/en/managed-agents/self-hosted-sandboxes)
- [Claude Managed Agents cloud environments](https://platform.claude.com/docs/en/managed-agents/environments)
- [Claude Managed Agents self-hosted sandbox security model](https://platform.claude.com/docs/en/managed-agents/self-hosted-sandboxes-security)
- [SWE-bench harness docs](https://www.swebench.com/SWE-bench/api/harness/)
- [SWE-bench evaluation guide](https://www.swebench.com/SWE-bench/guides/evaluation/)
- [SWE-ReX](https://github.com/SWE-agent/SWE-ReX)
- [SWE-ReX documentation](https://swe-rex.com/latest/)
- [SWE-ReX abstract runtime API](https://swe-rex.com/latest/api/runtimes/abstract/)
- [SWE-agent environments](https://swe-agent.com/latest/config/environments/)
- [OpenHands sandbox overview](https://docs.openhands.dev/usage/runtimes/overview)
- [OpenHands Docker sandbox](https://docs.openhands.dev/openhands/usage/runtimes/docker)
- [OpenHands Remote sandbox](https://docs.openhands.dev/usage/runtimes/remote)
- [OpenHands SDK Docker sandbox and browser tools](https://docs.openhands.dev/sdk/guides/agent-server/docker-sandbox)
- [OpenHands LLM overview](https://docs.openhands.dev/openhands/usage/llms)
- [OpenHands local LLMs](https://docs.openhands.dev/openhands/usage/llms/local-llms)
- [OpenHands Cloud API](https://docs.openhands.dev/openhands/usage/cloud/cloud-api)
- [Docker Engine security](https://docs.docker.com/engine/security/)
- [Firecracker design](https://github.com/firecracker-microvm/firecracker/blob/main/docs/design.md)
