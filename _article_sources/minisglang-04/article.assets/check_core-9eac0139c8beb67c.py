"""CPU contract checks against the actual minisgl.core module; no model execution."""
import platform
import torch
from minisgl.core import Batch, Context, Req, SamplingParams

print(f'Python {platform.python_version()}, torch {torch.__version__}, CPU')
req = Req(
    input_ids=torch.tensor([10, 11, 20, 21, 40, 41], dtype=torch.int32),
    table_idx=0, cached_len=4, output_len=4, uid=7,
    sampling_params=SamplingParams(), cache_handle=None,
)
assert (req.cached_len, req.device_len, req.max_device_len, req.extend_len) == (4, 6, 10, 2)
print('initial: cached=4, device=6, max_device=10, extend=2')
for token, expected in zip([50, 51, 52], [(6, 7), (7, 8), (8, 9)]):
    req.complete_one()
    assert (req.cached_len, req.device_len) == expected
    assert req.extend_len == 1
    assert req.can_decode
    req.append_host(torch.tensor([token], dtype=torch.int32))
    assert req.input_ids[-1].item() == token
    assert req.input_ids.is_cpu and req.input_ids.dtype == torch.int32
    assert len(req.input_ids) == req.device_len
    print(f'complete+append {token}: cached={req.cached_len}, device={req.device_len}, host_len={len(req.input_ids)}')
assert req.input_ids.tolist() == [10, 11, 20, 21, 40, 41, 50, 51, 52]

batch = Batch([req], phase='decode')
ctx = Context(page_size=2)
try:
    ctx.batch
except AssertionError as exc:
    assert str(exc) == 'No active batch in context'
else:
    raise AssertionError('empty Context.batch should fail')
with ctx.forward_batch(batch):
    assert ctx.batch is batch
assert ctx._batch is None
try:
    with ctx.forward_batch(batch):
        assert ctx.batch is batch
        raise ValueError('fixture-error')
except ValueError as exc:
    assert str(exc) == 'fixture-error'
assert ctx._batch is None
with ctx.forward_batch(batch):
    try:
        with ctx.forward_batch(batch):
            raise AssertionError('nested context unexpectedly entered')
    except AssertionError as exc:
        assert str(exc) == 'Nested forward_batch is not allowed'
    assert ctx.batch is batch
assert ctx._batch is None
print('Context: missing batch rejected; normal/exception cleanup passes; nesting rejected without corrupting outer batch')

cases = [
    ({}, True),
    ({'temperature': -1}, True),
    ({'temperature': 0.7}, False),
    ({'temperature': 0.7, 'top_k': 1}, True),
    ({'temperature': 0.0, 'top_p': 0.9}, False),
    ({'temperature': 0.7, 'top_k': 1, 'top_p': 0.9}, False),
    ({'temperature': 0.0, 'top_p': 1.000001}, False),
]
for kwargs, expected in cases:
    actual = SamplingParams(**kwargs).is_greedy
    assert actual is expected, (kwargs, actual, expected)
    print(f'is_greedy({kwargs}) = {actual}')
print('PASS: actual Req, Context and SamplingParams CPU contracts (illustrative input, no model/sample execution)')
