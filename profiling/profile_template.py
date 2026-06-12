import argparse
import pstats
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile a fenn template.")
    parser.add_argument("template", help="Template directory to profile")
    parser.add_argument("--limit", type=int, default=25, help="Rows in the report")
    args = parser.parse_args()

    template = ROOT / args.template
    entrypoint = template / "main.py"
    if not entrypoint.is_file() or template.parent != ROOT:
        parser.error(f"unknown template: {args.template}")

    output_dir = ROOT / "profiling" / "results" / args.template
    output_dir.mkdir(parents=True, exist_ok=True)
    profile_path = output_dir / "cprofile.prof"
    report_path = output_dir / "cprofile.txt"

    subprocess.run(
        [
            sys.executable,
            "-m",
            "cProfile",
            "-o",
            str(profile_path),
            entrypoint.name,
        ],
        cwd=template,
        check=True,
    )

    with report_path.open("w", encoding="utf-8") as report:
        pstats.Stats(str(profile_path), stream=report).strip_dirs().sort_stats(
            "cumulative"
        ).print_stats(args.limit)

    print(f"Profile: {profile_path}")
    print(f"Report:  {report_path}")


if __name__ == "__main__":
    main()
