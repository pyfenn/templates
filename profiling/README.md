# Template profiling

Use `profile_template.py` to profile a template without adding profiler code to
the template itself.

List the available templates:

```bash
python profiling/profile_template.py --list
```

Profile end-to-end Python execution with `cProfile`:

```bash
python profiling/profile_template.py mlp-binary --backend cprofile
```

Profile PyTorch operators and memory:

```bash
python profiling/profile_template.py mlp-binary --backend torch
```

Arguments after `--` are forwarded to the template through `sys.argv`:

```bash
python profiling/profile_template.py <template> --backend cprofile -- <arguments>
```

The current templates read workload settings from `fenn.yaml`. Reduce settings
such as epochs or dataset size there before profiling, then restore the file
before committing.

Results are written to `profiling/results/<template>/` by default:

- `cprofile.prof`: machine-readable `cProfile` data
- `cprofile.txt`: functions sorted by cumulative Python time
- `torch-trace.json`: Chrome trace from `torch.profiler`
- `torch.txt`: operators sorted by device or CPU time

Install the selected template's dependencies before profiling it. Some
templates also download datasets, require credentials, or need interactive
input. Profile short, representative workloads and record the hardware,
dependency versions, and configuration values when comparing results.

Profiler output is environment-specific and is intentionally excluded from
version control.
