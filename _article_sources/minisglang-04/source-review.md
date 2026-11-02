# Independent Part 4 source review

Reviewed the initial article and three SVGs against Mini-SGLang commit 9a91cfafe754aa85daee49998176275667eb58f2. Compared Part 3 addresses. Read only the two cited full-SGLang configuration files at f825d729363136a2d4a4b330fa694d0b37a878fa. No GPU/model execution or downloads.

## Actionable findings

1. **Wrong running-example capacity (material; parent correcting).** Four usable pages of two slots cap Engine.max_seq_len at eight. Scheduler._process_one_msg clamps six-token B to max_tokens=2. PrefillAdder._try_allocate_one also rejects output_len=4 after locking four matched tokens, because suffix2+output4=6 exceeds available4. Thus prefill plus two further decode forwards cannot be presented as the admitted four-page path. Expand to five usable pages and use free physical page [8,9] for two decodes, explicitly distinguishing Part 3 eviction. Correct five-page BF16 pool is [2,2,6,2,2,64], usable10240 bytes,total12288 bytes including dummy page. Sources: engine/engine.py Engine.__init__, scheduler/scheduler.py Scheduler._process_one_msg, scheduler/prefill.py PrefillAdder._try_allocate_one.
2. **Incomplete mathematical residual figure (minor).** engine-qwen3-shapes.svg draws the attention skip only. Add the post-attention residual bypass around MLP into the next/final addition if figure is called the mathematical residual path. Prose correctly explains deferred MLP residual addition. Source: models/qwen3.py Qwen3DecoderLayer.forward, Qwen3Model.forward; layers/norm.py RMSNormFused.forward.
3. **Minor arrow endpoint.** The first shape-flow arrow ends at decoder-stack border rather than RMSNorm box; extend to the box or use clear stack input convention.

## Sampled claims

| Section | Claim | Verdict | Source function / checked expression |
|---|---|---|---|
| 1 | Input token gather happens in _forward | correct | Scheduler._forward: batch.input_ids=self.token_pool[input_mapping] |
| 1 | Token and KV mappings differ | correct | _prepare_batch out_loc=page_table[input_mapping]; _forward writes token_pool[output_mapping] |
| 1 | Packed suffix has two rows for one request | correct | _make_positions and _make_input_tuple sum req.extend_len |
| 1 | Context model effective inputs exceed token IDs | correct | Qwen3ForCausalLM.forward; AttentionLayer.forward |
| 2 | Head width64 is supported | correct | RotaryEmbedding.__init__ assert head_size in [64,128,256,512] |
| 2 | QKV merged width512 at TP1 | correct | LinearQKVMerged.__init__: (4+2*2)*64 |
| 2 | Q/K norm then rotary; V unchanged | correct | AttentionLayer.forward |
| 2 | Fused residual additions happen at following norms | correct | Qwen3DecoderLayer.forward; RMSNormFused.forward |
| 2 | MLP gate_up1024 then512 then256 | correct | GatedMLP.__init__, forward |
| 2 | SiLU1*2 about1.4621 | correct | arithmetic; GatedMLP uses silu_and_mul |
| 3 | Store K/V before attention wrapper | correct | FlashInferBackend.forward |
| 3 | Pool layout K/V,layers,pages,page,heads,d | correct | MHAKVCache.__init__ |
| 3 | Four-page pool8192 usable10240 total | correct in isolation | MHAKVCache.__init__; engine adds1 dummy page |
| 3 | FI page_size1 virtual token pages | correct | _flatten_cache and FIMetadata.page_size |
| 3 | FI indices follow page_table through device_len | correct | FlashInferBackend.prepare_metadata |
| 3 | FI causal=True with suffix queries | correct | _initialize_metadata_once prefill.plan |
| 3 | Plan once per batch, reuse per layer | correct | metadata.initialized guard |
| 3 | Backend autoselection SM100/SM90/FI | correct | _adjust_config |
| 3 | FA uses page-table stride and division | correct | FlashAttentionBackend.prepare_metadata |
| 4 | Last packed rows chosen before head | correct | ParallelLMHead.forward; FIMetadata.get_last_indices |
| 4 | All-greedy shortcut argmax only if all greedy | correct | Sampler.prepare, sample |
| 4 | complete_one changes4,6 to6,7 | correct | Req.complete_one |
| 5 | Decode-only graphs, sampling outside capture | correct | GraphRunner.can_use_cuda_graph,_capture_graphs; Engine.forward_batch |
| 5 | Graph padding uses reserved request/page | correct | Engine.__init__; GraphRunner.pad_batch; _make_input_tuple |
| 5 | Replay refreshes buffers and backend plan | correct | GraphRunner.replay; FlashInferBackend.prepare_for_replay |
| 5 | Buckets1,2,4 then step8;160 or256 | correct | _determine_cuda_graph_bs |
| 6 | TP2 local Q2 KV1, merged width256 | correct | LinearQKVMerged.__init__ |
| 6 | Embedding reduce, output reduce, vocabulary gather | correct | VocabParallelEmbedding.forward,LinearOProj.forward,ParallelLMHead.forward |
| 6 | KV heads replicated when TP exceeds KV heads | correct | _shard_tensor; div_even allow_replicate |
| 6 | No explicit sampled token broadcast in engine/sampler | correct | Engine.forward_batch; Sampler.sample |
| 7 | Copy event precedes token_pool write | correct | Engine.forward_batch records event; Scheduler._forward next writes pool |
| 7 | Engine stream waits scheduler metadata stream | correct | Scheduler.overlap_loop |
| 8 | Speculation/disaggregation config surfaces exist | correct | speculative_hook handlers; handle_pd_disaggregation |
| 9 | Store/comm tests assertions; scheduler smoke text-only | correct | test_store_cache, test_comm.run, test_scheduler.main |

Counts: 34 sampled correct claims, 1 material incorrect composite running-example claim, 0 unsupported claims in sampled scope. The per-layer SVG is incomplete rather than numerically wrong. No major promised Part4 subsystem is absent: execution, attention read/write, TP, sampling, graph capture/replay, host/device feedback, and bounded full-SGLang comparison are covered.

## Readability (separate from semantics)

Opening, fixture labels, coordinate distinction, progressive engine path, and GPU verification boundary are clear. Source references are function-based and code excerpts match the opened implementation. Consider explaining once that a request row and a physical cache page are independent allocations. Tests remain a consolidated final section, appropriate for blog narration; no false GPU pass is claimed.

## Correction spot check

Parent changed article to five usable pages, ten usable slots, output allowance4, decode physical slots8/9, usable10240 and total12288 bytes. These changes resolve the material finding; final coordinate sentence updated too. Figure now has second residual bypass. One rendering detail communicated: initial arrow emitted before the opaque decoder-stack rectangle is partially hidden; emit it after rectangle. No remaining semantic blockers after corrected-draft spot check.
