# quantcard

**Quantization report cards for PyTorch models headed to the edge.**

You trained a model and you want it smaller and faster on-device. torchao gives
you a dozen quantization configs; ExecuTorch consumes them; but nothing tells
you, for *your* model, what each config costs in accuracy and buys in size and
latency. `quantcard` answers that with one command:

```bash
pip install quantcard

quantcard report my_model.py
```

```markdown
| config       | top-1 acc | acc delta | weights  | size ratio | latency (ms) | speedup |
|--------------|-----------|-----------|----------|------------|--------------|---------|
| fp32         | 98.44%    | +0.00%    | 100 KiB  | 1.00x      | 0.05         | 1.00x   |
| int8wo       | 98.44%    | +0.00%    | 33 KiB   | 0.33x      | 0.05         | 1.05x   |
| int8da-int8w | 98.24%    | -0.20%    | 33 KiB   | 0.33x      | 0.06         | 0.90x   |
```

Your model file just needs two functions:

```python
# my_model.py
def get_model():          # -> torch.nn.Module (trained, ready for eval)
    ...

def get_eval_batches():   # -> iterable of (inputs, targets)
    ...
```

## Configs

`fp32` (baseline) · `int8wo` · `int8da-int8w` · `int4wo` · `int8da-int4w`
(the classic ExecuTorch 8da4w recipe) · `qat-int8da-int4w` (QAT prepare
step — what the fake-quant numerics cost before any training). Run
`quantcard report model.py --configs all` for everything.

## ExecuTorch export column

With the `executorch` package installed (`pip install quantcard[pte]`),
`--pte` adds a column with the exported `.pte` program size — and tells you
honestly when a config can't export yet:

```
| config | ... | .pte |
| fp32   | ... | 102 KiB |
| int8wo | ... | 32 KiB |
| int8da-int4w | ... | export failed (RuntimeError) |
```

## Status

Early alpha, deliberately narrow: torchao configs, CPU
accuracy/size/latency, `.pte` export size, markdown reports. Planned next:
robustness rows (input noise, quantized-vs-float disagreement rate),
QAT convert deltas after a training hook, and device latency via
ExecuTorch runners.

## Development

```bash
pip install -e .[dev]
pytest
```

## License

MIT
