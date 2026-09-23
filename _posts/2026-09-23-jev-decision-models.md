---
layout: post
title: "Jev, JEVfire, and Kev: What Does a Decision Model Actually Compute?"
subtitle: A worked token-scoring example, open implementations, serving-engine boundaries, and an RLCD hypothesis
categories: Large-Language-Model LLM-Inference
tags: [Blogs]
---

# Jev, JEVfire, and Kev: What Does a Decision Model Actually Compute?

I have been reading [nano-vLLM]({% post_url 2026-09-06-nanovllm-01 %}) and serving-engine internals, so Jev's launch made me ask a fairly specific question: **what has to happen between a model forward pass and a typed decision?** Could I put a small wrapper around vLLM or SGLang, or would I need to change the engine and train a new model?

The answer depends on the goal. An off-the-shelf language model can already score a finite set of choices without generating JSON. [JEVfire](https://github.com/kikoncuo/jevfire) demonstrates that path with vLLM. [Kev](https://github.com/jaredpalmer/kev) takes a different path: it trains a LoRA and a pointer head to read decisions from hidden states. Neither implementation reveals TypeSafe's unpublished Jev weights or its RLCD recipe. They reproduce useful *behaviors and interfaces*, not Jev itself.

This post follows one concrete request through both open implementations, surveys the other approaches, and ends with a testable hypothesis about [Reinforcement Learning for Calibrated Decisions (RLCD)](https://typesafe.ai/blog/introducing-system-one-models-and-jev). **All example probabilities below are invented to explain the computation; they are not model measurements.** Sources and the code paths I used are linked throughout. The landscape is changing quickly; this is a snapshot as of September 23, 2026.

## 1. The actual task: one state, several small decisions

Imagine a racing game whose physics engine reports:

```text
Tight bend ahead. Low grip. Lane clear. Boost charged.
```

We want two outputs:

| Question | Allowed answers |
| --- | --- |
| Maneuver? | `brake`, `coast`, `accelerate` |
| Use boost? | `true`, `false` |

We could ask a chat model to generate `{"maneuver":"brake","boost":false}`. That works, but the model must emit punctuation, field names, and values one token at a time. A decision interface instead asks for a probability distribution over the declared answers to each question. Application code then builds the object and enforces game rules. It cannot return an undeclared field, although it can still pick a **wrong allowed action**.

TypeSafe calls these question shapes [Choice, Score, and Noul](https://docs.typesafe.ai/introduction): an unordered choice, a rating on a rubric, and a yes/no probability. The questions are evaluated independently against the same state. This is a particularly good fit for routing, tagging, and short agent gates. It is a poor substitute for writing arbitrary text or for a move that requires long search. A probability of `0.90` is useful as a correctness estimate only if calibration has been established on comparable data.

## 2. JEVfire: score next-token labels on stock weights

[JEVfire's API](https://github.com/kikoncuo/jevfire/blob/main/README.md) accepts a text `context` and a flat schema of finite fields. Its FastAPI process is a **sidecar** in front of an already running vLLM server; it does not load a second model. Its [core.py](https://github.com/kikoncuo/jevfire/blob/main/jevfire/core.py) constructs prompts, asks for selected token log probabilities, normalizes them, and assembles the response.

Here is the request in the shape used by its racing example:

```json
{
  "context": "Tight bend ahead. Low grip. Lane clear. Boost charged.",
  "schema": {
    "maneuver": {
      "type": "enum",
      "description": "Brake for a tight bend with low grip.",
      "choices": ["brake", "coast", "accelerate"]
    },
    "boost": {
      "type": "boolean",
      "description": "Boost only on a clear straight with good grip."
    }
  }
}
```

**Step 1: turn answer values into single-token handles.** JEVfire checks its model's tokenizer at startup and keeps labels such as `A`, `B`, and `C` only if each is one distinct, round-tripping, non-special token. For `maneuver`, `A → brake`, `B → coast`, `C → accelerate`. In a *separate* prompt for `boost`, `A → true`, `B → false`. The labels are internal handles, not the values sent to the application. Checking the actual tokenizer matters: a seemingly short string need not be one token.

**Step 2: make one prompt per field.** The system instructions and race context are shared. Each prompt appends only its own field definition and label map. The boost question does not see the maneuver answer, or even the maneuver options. Both prompts stop exactly where the assistant would begin answering:

```text
[shared instructions and race context]
Selected field: maneuver. A=brake, B=coast, C=accelerate.
Assistant: [score the next token here]

[shared instructions and race context]
Selected field: boost. A=true, B=false.
Assistant: [score the next token here]
```

Those lines show the logic, not the repository's literal chat-template serialization. The real prompts include field descriptions and are converted to token IDs before the API call.

**Step 3: ask vLLM for one next-token position.** The interesting part of JEVfire's `_score_chunk` is its `/v1/completions` request:

```python
payload = {
    "model": self.model,
    "prompt": prompts,                # one token-ID prompt per field
    "max_tokens": 1,
    "temperature": 0,
    "logprobs": 0,
    "logprob_token_ids": token_ids,   # IDs for A, B, C
    "return_tokens_as_token_ids": True,
    "add_special_tokens": False,
}
```

For each field prompt, the model computes next-token logits over its vocabulary. vLLM returns the requested label-token log probabilities. The sampled output token is discarded. **Three options within a field are not three independent model requests.** One prompt has one readout position with three candidate logits. The two *fields* are separate prompts, submitted together so the engine can batch them. A single HTTP call does not imply a single GPU kernel launch or one forward pass for the entire request.

Suppose the returned log probabilities are:

| Field | `A` | `B` | `C` |
| --- | ---: | ---: | ---: |
| maneuver | -0.2 | -1.5 | -2.1 |
| boost | -1.8 | -0.4 | — |

**Step 4: normalize only over the allowed options.** JEVfire applies `softmax(score / T)` over the labels of each field. With `T=1`, the maneuver probabilities are approximately `0.70, 0.19, 0.11`; boost becomes `0.20, 0.80`. Python maps `A` back to `brake` and `B` back to `false` and constructs the JSON. The model never writes the JSON keys.

These numbers are **conditional on the offered menu**, not a demonstrated 70% chance that braking is objectively correct. If the prompt encourages an answer outside `A/B/C`, even a poor candidate can win after renormalization. JEVfire exposes candidate probability mass separately; a confidence threshold or an `unknown` option still needs evaluation on the actual workload. Its decoder treats a missing or non-finite requested score as an error rather than silently selecting a default. If a menu exceeds stock vLLM's 128 requested-token limit, JEVfire scores chunks of the *same full prompt* and merges the raw log probabilities before normalizing; its optional patch raises the cap. See its [implementation](https://github.com/kikoncuo/jevfire/blob/main/jevfire/core.py) and [vLLM's sampling parameters](https://docs.vllm.ai/en/latest/api/vllm/sampling_params/).

**Step 5: reuse shared context when possible.** The two prompts have an identical instruction/state prefix. vLLM's prefix cache can reuse completed matching cache blocks. JEVfire offers plain batching and variants that prefill one question first or align the shared prefix with a cache-block boundary. This is an engine optimization with workload-specific tradeoffs: alignment adds tokens and can even alter answers. It does not magically make two different question suffixes the same computation.

JEVfire reports a **496.9 ms median** for a synthetic 28-field fresh-prefix fixture versus **5,113.1 ms** for generating equivalent grammar-constrained JSON with the same 27B weights and thinking setting on its measured machine. That is a useful demonstration of avoiding sequential output tokens, not a general speed guarantee; the author provides the [methodology and raw results](https://github.com/kikoncuo/jevfire). Its browser Mario example is another implementation path with local WebLLM and a physics guard, not this CUDA/vLLM server.

## 3. Kev: train a model to read options from hidden states

The racing request could also be encoded as a state plus two typed questions in [Kev](https://github.com/jaredpalmer/kev). But the computation changes. A Kev checkpoint is a Qwen base **plus a trained LoRA adapter and a trained pointer head**. It reads internal hidden states instead of asking the language-model vocabulary head for the logit of the letter `A`.

In the attention-only Qwen3 implementation, the input is conceptually packed like this:

```text
<state> tight bend, low grip, lane clear, boost charged
<q> choose maneuver <opt> brake </opt> <opt> coast </opt>
    <opt> accelerate </opt> <decide>
<q> use boost? <opt> true </opt> <opt> false </opt> <decide>
```

Kev's branch attention mask lets each question see the shared state and *its own* options, but not another question. The pointer head projects the hidden state at each option's `</opt>` into a **key** and the hidden state at that question's `<decide>` into a **query**. Scaled dot products produce one score per option; softmax converts them to a distribution. In shorthand:

$$
s_i = \frac{(W_q h_{\mathrm{decide}})^\top(W_k h_{\mathrm{option},i})}{\sqrt{d}},\qquad p_i = \operatorname{softmax}(s)_i.
$$

This scores an option as a span represented at its boundary, so `accelerate`, a long department name, or a chess move need not be a single vocabulary token. The LoRA adapts the backbone's representations and the head learns the readout together; keeping only the head on an untouched base is a different model. Kev trains those parameters with cross-entropy on labeled decisions and can fit a separate temperature to a development set. Its [README](https://github.com/jaredpalmer/kev#how-it-works) and [model code](https://github.com/jaredpalmer/kev/blob/main/kev/model.py) document the details.

There is a version detail worth getting right. On attention-only Qwen3 bases, Kev can pack isolated questions into one block-causal sequence. The current Qwen3.5 models have recurrent Gated DeltaNet layers that do not obey the same attention mask, so Kev runs one state-plus-question row per question and reuses the state cache. Both designs keep questions isolated. Its earliest Qwen2.5-0.5B checkpoint and current Qwen3.5-0.8B/4B/9B family should not be described as the same exact execution path. Kev's published results include strong in-distribution numbers and explicit out-of-domain limitations; a calibrated development set is not proof of calibration on arbitrary new decisions.

## 4. What the other open projects actually reproduce

The name “Jev reproduction” covers several distinct experiments. The following is an implementation map, **not a quality ranking**:

| Approach | Representative implementation | What changes? | Main tradeoff |
| --- | --- | --- | --- |
| Prompt and score existing weights | [JEVfire](https://github.com/kikoncuo/jevfire), [SemIf](https://github.com/TheoLeeCJ/SemIf) | Prompt layout, choice-token scoring, API wrapper; no weight training | Fast route to a usable decision API; confidence needs workload calibration |
| Shared-prefix direct readout | [Qwen-2.5-1B-RLCD demo](https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD) | Reuse prefix KV, branch into fields, score candidate tokens | Efficient reuse, but option tokenization and continuation matter; the name alone does not establish RL training |
| Trained adapter and head | [Kev](https://github.com/jaredpalmer/kev), [Solomon](https://huggingface.co/DoccyHealth/Solomon) | Train a task readout and adapter, with different branching schemes | Better control over representation; need labeled data and a dedicated serving path |
| Fine-tuned decision model or classifier | [Decider](https://github.com/Mapika/decider), [Verdict](https://github.com/DINHCHUNG93/verdict-open-jev) | Fine-tune a language-model readout or train a classifier | Small or specialized runtime possible; task coverage depends on training |
| Reinforcement learning from outcome feedback | [eve-rlcd](https://github.com/anthony-maio/eve-rlcd) | Update a decision policy using sampled actions and their outcomes | Meaningful when only the chosen action is graded; an independent recipe, not TypeSafe's |

There are also grammar-constrained JSON generation and diffusion-language-model experiments. They can satisfy an output *format*, but their runtime and training claims need to be evaluated separately. Decider's [4B and 35B releases](https://github.com/Mapika/decider) describe supervised training, while its 2B v10 also reports a later calibration-aware RL stage: even checkpoints in one project can have different recipes. [Solomon's model card](https://huggingface.co/DoccyHealth/Solomon) describes a Qwen3.8-27B base, a question-side rank-64 LoRA, and trained answer heads; its evidence pointers and document specialization are additional features, not consequences of the Jev wire format. [Kev's repository](https://github.com/jaredpalmer/kev) includes its own suite, training recipe, and model cards. Comparing a self-reported accuracy on one project's data with another project's latency on another machine would be misleading.

For broader comparison, the independent [Decision Index reproduction kit](https://github.com/apolinario/decision-index) specifies **132,422 frozen requests across 37 benchmarks**, of which 19 form the five-area headline index. Unanswered requests count as wrong. It can run a full engine sweep on one RTX PRO 6000 GPU; this means **one GPU per run**, not that all competing models are evaluated simultaneously on one GPU. Its index covers knowledge, language, retrieval, tools, and judgment, so a strong result on a familiar classification dataset does not settle the harder generalization question. Hosted API round-trip latency and local GPU-only latency are different clocks.

## 5. Does vLLM or SGLang already implement this?

**They expose the necessary scoring operations.** vLLM has an official [`/generative_scoring` endpoint](https://docs.vllm.ai/en/latest/serving/online_serving/generative_scoring/) for a causal model. It appends each item to a query, reads specified next-token label probabilities, and optionally softmaxes over those labels. Its response returns the **first label's score for each item**, so a multi-field Jev-shaped API still needs an adapter to construct one prompt per question, handle all option distributions, types, failure modes, and output assembly. JEVfire instead calls `/v1/completions` with `logprob_token_ids` to get the individual requested scores. vLLM's unrelated `/classify` and `/score` endpoints serve different model/task interfaces; endpoint names alone do not make them Jev.

SGLang's documented [`/generate` inputs](https://github.com/sgl-project/sglang/blob/main/docs/docs/basic_usage/sampling_params.mdx) likewise support batched `input_ids`, `return_logprob`, and `token_ids_logprob`. An external adapter can build per-field prompts, request those scores, normalize the appropriate subset, and return the decision schema. I did **not** find a first-party, Jev-compatible `/v1/systemone` implementation in either project; the official support here is for the lower-level scoring primitives.

For a **JEVfire-style scorer**, my first implementation would therefore be a wrapper, without rewriting the scheduler, attention kernel, or model forward pass. Confirm token-ID logprob behavior on the exact pinned engine version and model before relying on it. A custom engine path becomes useful when the requirement changes: return raw logits without even sampling a throwaway token, hold reusable state handles across requests, fuse several question readouts, serve Kev's trained pointer head, support nonstandard branch attention, or optimize batching and cache lifetime. Those are distinct improvements and need measurements before becoming engine patches.

## 6. Why Jev attracted attention if ordinary models already work

JEVfire's result is instructive: **much of the immediate speedup is an interface and inference trick**, available with already trained weights. Even small fine-tunes can do well on their own distributions. That does not establish that a frozen model has Jev's claimed combination of broad task accuracy, honest probabilities, low latency, and low serving cost on new workloads.

Three product decisions also matter. First, a typed API turns uncertain model judgments into explicit program branches without asking an LLM to serialize the same decision as prose. Second, treating several *independent* questions over one state as a first-class workload creates opportunities for prefix reuse and parallel execution. Third, calibration can change what an application safely automates: a threshold of `p > 0.9` is valuable only if events scored around 0.9 really succeed around nine times out of ten on the target population. Format-valid output alone does not provide that property.

TypeSafe [states](https://typesafe.ai/blog/introducing-system-one-models-and-jev) that Jev uses a new model architecture, a parallel sampler, and RLCD. Its public docs describe isolated Choice/Score/Noul questions and probabilities, but **neither the weights, detailed architecture, nor the RLCD objective and training data are published**. I would treat the speed and calibration claims as hypotheses to test on my own workload, and the open projects as useful baselines rather than assume the proprietary advantage must or must not exist.

## 7. A bounded hypothesis about RLCD

What could “reinforcement learning for calibrated decisions” mean technically? The following is **my inference, not a description of Jev's internal training**.

Start with a policy $p_\theta(a\mid s,q,\mathcal A)$ over allowed actions for state $s$, question $q$, and menu $\mathcal A$. A pretrained model or a supervised decision fine-tune supplies semantic knowledge. Then train the decision readout on a mixture of tasks, including ambiguous and out-of-distribution cases, with feedback that rewards both correct choices and honest uncertainty. When the full correct label is available, one can directly optimize cross-entropy or a proper scoring rule such as the Brier score, and temperature-scale on held-out data. **There is no special need for RL merely because the output is a probability.**

RL becomes interesting if the system observes only the outcome of the action it actually took. If a support agent routes a ticket to Billing and later learns whether that route succeeded, it may never learn what would have happened under Infrastructure or Security. One possible bandit update samples $a\sim p_\theta$, observes correctness $c\in\{0,1\}$, and applies a policy-gradient estimator with a confidence-sensitive reward. The independent [eve-rlcd experiment](https://github.com/anthony-maio/eve-rlcd) uses $r=c-p_\theta(a)$ with REINFORCE and a group baseline; its author relates the expected gradient to the Brier objective under the stated feedback setup. That demonstrates **one plausible open mechanism**, not TypeSafe's recipe. Other possibilities include an outcome reward shaped by log score, ordinal scoring for Score questions, or a supervised-plus-RL mixture. None is confirmed for Jev.

The decisive experiments are empirical: reliability diagrams and ECE/Brier/log loss on held-out *new* task families; accuracy-versus-coverage when abstaining; option-order and wording sensitivity; distribution shift; and matched throughput/latency under concurrent requests. A softmax that sums to one is a probability distribution over offered options. It is not, by itself, calibrated knowledge of whether a selected answer is correct.

## 8. The follow-up I want to build

For a second post and a small open-source repository, I would instrument **[mini-SGLang](https://github.com/sgl-project/mini-sglang)** first. It already has an online serving path and a radix cache. In contrast, **[nano-vLLM](https://github.com/GeeeekExplorer/nano-vllm)** is a very readable offline engine whose model runner, logits, and sampler I have [already followed]({% post_url 2026-09-11-nanovllm-03 %}); it is an excellent second backend or a smaller offline teaching exercise. I would keep upstream repos as dependencies or pinned forks, with a focused patch only if the existing interface cannot expose the selected logits. The novel work should be a reproducible comparison, not just another renamed wrapper.

The prototype would expose `state + questions -> typed answers + per-option scores`. First, implement the JEVfire-style one-token-label baseline on a small Qwen model. Then add an engine-side scoring path that reads selected logits directly after prefill, and measure whether removing the one-token decode and HTTP hop actually helps. Compare separate prompts, batched prompts, and cached-prefix branches against constrained JSON on *identical* model weights and hardware. Keep accuracy, Brier/ECE, token counts, p50/p95 latency, throughput at concurrency, and cache hit rate separate.

For a game, I would use **chess puzzles before full chess play**. `python-chess` can produce the legal moves from a FEN and reject illegal output; the model scores those moves as choices, while a chess engine or labeled puzzle supplies a tactical reference. The position, side to move, and legal moves are explicit. Some positions have a large move menu; candidate-label limits, prompt length, and option-order effects become real engineering tests. Report top-1 legal-move/puzzle accuracy, forced-mate detection, and latency per position; do not market a next-token choice as a substitute for search. A simple live game can come later, with a legal-move guard and a clearly identified engine opponent.

My working conclusion is modest: **a useful decision service can start as a scoring adapter; a trained and well-calibrated decision model is a separate research problem.** The interesting follow-up is measuring exactly where the adapter stops being enough.
