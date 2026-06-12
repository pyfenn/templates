from __future__ import annotations

import importlib.util
import io
import shutil
import unittest
import uuid
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

from profiling import profile_template


PROFILING_DIR = Path(__file__).resolve().parents[1]


class ProfileTemplateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = PROFILING_DIR / "tests" / ".tmp" / uuid.uuid4().hex
        self.root.mkdir(parents=True)
        self.addCleanup(shutil.rmtree, self.root, True)
        self.template_dir = self.root / "example"
        self.template_dir.mkdir()

    def test_discovers_only_directories_with_entry_points(self) -> None:
        (self.template_dir / "main.py").write_text("", encoding="utf-8")
        (self.root / "not-a-template").mkdir()
        (self.root / "file.txt").write_text("", encoding="utf-8")

        self.assertEqual(
            profile_template.discover_templates(self.root),
            ["example"],
        )

    def test_rejects_unknown_template(self) -> None:
        (self.template_dir / "main.py").write_text("", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "Unknown template"):
            profile_template.resolve_template("missing", self.root)

    def test_cprofile_writes_reports_and_forwards_arguments(self) -> None:
        (self.template_dir / "main.py").write_text(
            "from pathlib import Path\n"
            "import sys\n"
            "Path('arguments.txt').write_text('|'.join(sys.argv[1:]))\n"
            "sum(range(100))\n",
            encoding="utf-8",
        )
        output_dir = self.root / "results"

        data_path, table_path = profile_template.profile_with_cprofile(
            self.template_dir,
            output_dir,
            ["train.epochs=1"],
            row_limit=10,
        )

        self.assertTrue(data_path.is_file())
        self.assertIn("main.py", table_path.read_text(encoding="utf-8"))
        self.assertEqual(
            (self.template_dir / "arguments.txt").read_text(encoding="utf-8"),
            "train.epochs=1",
        )

    def test_cli_separates_profiler_options_from_template_arguments(self) -> None:
        with (
            mock.patch.object(
                profile_template, "discover_templates", return_value=["example"]
            ),
            mock.patch.object(
                profile_template,
                "resolve_template",
                return_value=self.template_dir,
            ),
            mock.patch.object(
                profile_template,
                "profile_with_cprofile",
                return_value=(Path("data"), Path("table")),
            ) as profile,
        ):
            with redirect_stdout(io.StringIO()):
                result = profile_template.main(
                    [
                        "example",
                        "--backend",
                        "cprofile",
                        "--",
                        "train.epochs=1",
                    ]
                )

        self.assertEqual(result, 0)
        self.assertEqual(profile.call_args.args[2], ["train.epochs=1"])

    @unittest.skipUnless(
        importlib.util.find_spec("torch"),
        "PyTorch is not installed",
    )
    def test_torch_profiler_writes_trace_and_report(self) -> None:
        (self.template_dir / "main.py").write_text(
            "import torch\n"
            "left = torch.ones((2, 2))\n"
            "right = torch.ones((2, 2))\n"
            "torch.matmul(left, right)\n",
            encoding="utf-8",
        )
        output_dir = self.root / "results"

        trace_path, table_path = profile_template.profile_with_torch(
            self.template_dir,
            output_dir,
            [],
            row_limit=10,
            with_stack=False,
        )

        self.assertTrue(trace_path.is_file())
        self.assertIn("aten::matmul", table_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
