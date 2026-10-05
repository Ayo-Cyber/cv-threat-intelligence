"""The PPE scanner must see people even when no frames are being published.

`latest_boxes` is the scanner's only source of person boxes. It used to be
written inside `if self.publisher is not None:`, so an engine started with
--no-publish-frames (the CI KPI harness, any headless run) handed the scanner
frames with nobody in them: "people_seen": 0 on a clip full of workers, and
no PPE violation ever (5 Oct 2026, KPI 12 run).
"""
from __future__ import annotations

import ast
import inspect
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class BoxesRecordedRegardlessOfPublisherTest(unittest.TestCase):
    def test_latest_boxes_is_not_guarded_by_the_publisher(self):
        import textwrap
        from cvti.serving.pipeline import MultiStreamPipeline
        src = textwrap.dedent(inspect.getsource(MultiStreamPipeline._route_to_queue))
        tree = ast.parse(src)
        guarded: list[list[str]] = []

        def visit(node, guards):
            for child in ast.iter_child_nodes(node):
                g = guards
                if isinstance(child, ast.If):
                    test_src = ast.unparse(child.test)
                    g = guards + ([test_src] if "publisher" in test_src else [])
                if isinstance(child, ast.Assign):
                    targets = " ".join(ast.unparse(t) for t in child.targets)
                    if "latest_boxes" in targets and g:
                        guarded.append(g)
                visit(child, g)

        visit(tree, [])
        self.assertEqual(guarded, [], "latest_boxes is written only when a publisher exists — "
                                      "the PPE scanner then never sees a person")
        self.assertIn("self.latest_boxes[frame.camera_id] = boxes", src)


if __name__ == "__main__":
    unittest.main()
