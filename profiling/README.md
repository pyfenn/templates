# Template profiling

Profile a template from the repository root:

```bash
python profiling/profile_template.py <template>
```

The command runs the template with Python's built-in `cProfile` and writes the
binary profile plus a cumulative-time report to
`profiling/results/<template>/`.

Install the template's dependencies first. For long-running templates, reduce
the workload in its `fenn.yaml` before profiling and restore it afterwards.
Generated profiles are environment-specific and are not committed.
