# quantbench

**Quantization report cards for PyTorch models headed to the edge.**

You trained a model and you want it smaller and faster on-device. torchao gives
you a dozen quantization configs; ExecuTorch consumes them; but nothing tells
you, for *your* model, what each config costs in accuracy and buys in size and
latency. `quantbench` answers that with one command:

```bash
pip install quantbench

quantbench report my_model.py
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

## Status

Early alpha, deliberately narrow: torchao configs (`int8wo`,
`int8da-int8w`), CPU accuracy/size/latency, markdown reports. Planned next:
more torchao recipes (int4, QAT prepare/convert deltas), ExecuTorch `.pte`
export size + load check, robustness rows (input noise, quantized-vs-float
disagreement rate), and device latency via ExecuTorch runners.

## Development

```bash
pip install -e .[dev]
pytest
```

## License

MIT
