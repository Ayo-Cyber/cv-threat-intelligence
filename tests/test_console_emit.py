"""cvti.console.emit is called correctly everywhere (regression guard).

The print -> logging conversion rewrote `print(..., file=sys.stderr)` to
`emit(..., file=sys.stderr)`, but emit takes `err=True`. Every VideoMAE
inference then raised TypeError inside a broad handler, so the fine-tuned model
— the headline of the demo — failed 249 times per run and said only
"[VideoAction error]". A silent failure inside an exception handler is exactly
what EP-01 exists to prevent, so this checks the whole class statically.
"""
import ast
import inspect
import pathlib
import unittest

from cvti.console import emit


class EmitCallSitesTest(unittest.TestCase):
    def test_every_call_site_matches_the_signature(self):
        allowed = set(inspect.signature(emit).parameters) - {"parts"}
        bad = []
        for path in sorted(pathlib.Path("cvti").rglob("*.py")):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                if not (isinstance(node.func, ast.Name) and node.func.id == "emit"):
                    continue
                for kw in node.keywords:
                    if kw.arg is not None and kw.arg not in allowed:
                        bad.append(f"{path}:{node.lineno} passes {kw.arg}=")
        self.assertEqual(bad, [], f"emit() called with unsupported keywords: {bad}")

    def test_emit_writes_to_stderr_when_asked(self):
        import io
        import sys
        buf, saved = io.StringIO(), sys.stderr
        sys.stderr = buf
        try:
            emit("problem", err=True)
        finally:
            sys.stderr = saved
        self.assertEqual(buf.getvalue(), "problem\n")


if __name__ == "__main__":
    unittest.main()
