from __future__ import annotations

import argparse
import cProfile
import os
import pstats
import runpy
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = Path(__file__).resolve().parent / "results"


def discover_templates(repository_root: Path = REPOSITORY_ROOT) -> list[str]:
    """Return template directories that contain a main.py entry point."""
    return sorted(
        path.name
        for path in repository_root.iterdir()
        if path.is_dir() and (path / "main.py").is_file()
    )


def resolve_template(
    template_name: str, repository_root: Path = REPOSITORY_ROOT
) -> Path:
    """Resolve a template name without allowing paths outside the repository."""
    available = discover_templates(repository_root)
    if template_name not in available:
        choices = ", ".join(available)
        raise ValueError(f"Unknown template {template_name!r}. Available: {choices}")
    return repository_root / template_name


@contextmanager
def template_execution_context(
    template_dir: Path, template_args: Sequence[str]
) -> Iterator[Path]:
    """Run a template as if its main.py had been launched from its directory."""
    main_path = template_dir / "main.py"
    previous_cwd = Path.cwd()
    previous_argv = sys.argv[:]
    previous_path = sys.path[:]

    os.chdir(template_dir)
    sys.argv = [str(main_path), *template_args]
    sys.path.insert(0, str(template_dir))
    try:
        yield main_path
    finally:
        os.chdir(previous_cwd)
        sys.argv = previous_argv
        sys.path[:] = previous_path


def run_template(template_dir: Path, template_args: Sequence[str]) -> None:
    """Execute a template's main.py entry point."""
    with template_execution_context(template_dir, template_args) as main_path:
        runpy.run_path(str(main_path), run_name="__main__")


def profile_with_cprofile(
    template_dir: Path,
    output_dir: Path,
    template_args: Sequence[str],
    row_limit: int,
) -> tuple[Path, Path]:
    """Profile complete Python execution and write binary and text reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "cprofile.prof"
    table_path = output_dir / "cprofile.txt"
    profiler = cProfile.Profile()

    try:
        profiler.enable()
        run_template(template_dir, template_args)
    finally:
        profiler.disable()
        profiler.dump_stats(data_path)
        with table_path.open("w", encoding="utf-8") as stream:
            stats = pstats.Stats(profiler, stream=stream)
            stats.strip_dirs().sort_stats("cumulative").print_stats(row_limit)

    return data_path, table_path


def _torch_activities(torch_module):
    activities = [torch_module.profiler.ProfilerActivity.CPU]
    sort_key = "cpu_time_total"

    if torch_module.cuda.is_available():
        activities.append(torch_module.profiler.ProfilerActivity.CUDA)
        sort_key = "cuda_time_total"
    elif hasattr(torch_module, "xpu") and torch_module.xpu.is_available():
        activities.append(torch_module.profiler.ProfilerActivity.XPU)
        sort_key = "xpu_time_total"

    return activities, sort_key


def profile_with_torch(
    template_dir: Path,
    output_dir: Path,
    template_args: Sequence[str],
    row_limit: int,
    with_stack: bool,
) -> tuple[Path, Path]:
    """Profile PyTorch operators and write a Chrome trace and text report."""
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "The torch backend requires PyTorch. Install the selected "
            "template's dependencies first."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / "torch-trace.json"
    table_path = output_dir / "torch.txt"
    activities, sort_key = _torch_activities(torch)

    profiler = torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=with_stack,
    )
    try:
        with profiler:
            with torch.profiler.record_function(f"template:{template_dir.name}"):
                run_template(template_dir, template_args)
    finally:
        profiler.export_chrome_trace(str(trace_path))
        table = profiler.key_averages(group_by_input_shape=True).table(
            sort_by=sort_key,
            row_limit=row_limit,
        )
        table_path.write_text(table, encoding="utf-8")

    return trace_path, table_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile a fenn template without modifying its entry point."
    )
    parser.add_argument("template", nargs="?", help="Template directory to profile")
    parser.add_argument(
        "--backend",
        choices=("cprofile", "torch"),
        default="cprofile",
        help="Profiler backend (default: cprofile)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: profiling/results/<template>)",
    )
    parser.add_argument(
        "--row-limit",
        type=int,
        default=25,
        help="Maximum rows in the text report (default: 25)",
    )
    parser.add_argument(
        "--with-stack",
        action="store_true",
        help="Record source stacks with torch.profiler (higher overhead)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List templates with a main.py entry point and exit",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_args = list(argv) if argv is not None else sys.argv[1:]
    if "--" in raw_args:
        separator = raw_args.index("--")
        profiler_args = raw_args[:separator]
        template_args = raw_args[separator + 1 :]
    else:
        profiler_args = raw_args
        template_args = []

    args = build_parser().parse_args(profiler_args)
    available = discover_templates()

    if args.list:
        print("\n".join(available))
        return 0
    if args.template is None:
        raise SystemExit("a template is required unless --list is used")
    if args.row_limit < 1:
        raise SystemExit("--row-limit must be at least 1")

    try:
        template_dir = resolve_template(args.template)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    output_dir = args.output_dir or DEFAULT_OUTPUT_ROOT / args.template
    if args.backend == "cprofile":
        artifacts = profile_with_cprofile(
            template_dir,
            output_dir,
            template_args,
            args.row_limit,
        )
    else:
        artifacts = profile_with_torch(
            template_dir,
            output_dir,
            template_args,
            args.row_limit,
            args.with_stack,
        )

    for artifact in artifacts:
        print(artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
