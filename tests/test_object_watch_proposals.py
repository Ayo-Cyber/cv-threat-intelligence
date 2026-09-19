import importlib
import importlib.util
import math
import subprocess
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import pytest

from cvti.object_watch.matcher import ObjectCandidate
from cvti.object_watch.proposals import (
    YoloWorldProposalProvider,
    merge_proposals,
    yolo_world_preflight,
)


def test_merge_proposals_nms_and_global_cap() -> None:
    existing = [ObjectCandidate((0, 0, 20, 20), "generic", 0.8)]
    proposed = [
        {"box": (1, 1, 19, 19), "phrase": "grounded", "score": 0.9},
        {"box": (30, 30, 40, 40), "phrase": "other", "score": 0.7},
    ]
    rows = merge_proposals(existing, proposed, max_candidates=2)
    assert [row.label_hint for row in rows] == ["grounded", "other"]


def test_missing_models_fail_offline_without_constructor(tmp_path: Path) -> None:
    readiness = yolo_world_preflight(tmp_path / "missing.pt", tmp_path / "missing-clip.pt")
    assert not readiness.ready
    assert "missing" in " ".join(readiness.reasons)


class _Tensor:
    def __init__(self, values):
        self.values = values

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class _InjectedModel:
    def __init__(self, results):
        self.results = results
        self.phrases = None
        self.predict_kwargs = None

    def configure_classes(self, phrases):
        self.phrases = phrases

    def predict(self, frame, **kwargs):
        self.predict_kwargs = kwargs
        return self.results


def _local_weights(tmp_path: Path) -> tuple[Path, Path]:
    import zipfile

    world = tmp_path / "world.pt"
    with zipfile.ZipFile(world, "w") as archive:
        archive.writestr("weights", b"local")
    clip = tmp_path / "clip.pt"
    clip.write_bytes(b"local")
    return world, clip


def test_provider_decodes_results_and_limits_without_set_classes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    world, clip = _local_weights(tmp_path)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    boxes = SimpleNamespace(
        xyxy=_Tensor([[20.2, 20.1, 30.4, 31.0], [0, 0, 10, 10],
                      [math.nan, 0, 2, 2], [1, 1, 3, 3]]),
        conf=_Tensor([0.4, 0.9, 0.8, math.nan]),
        cls=_Tensor([1, 0, 0, 1]),
    )
    fake = _InjectedModel([SimpleNamespace(boxes=boxes)])
    provider = YoloWorldProposalProvider(world, clip, model_factory=lambda *_: fake)

    rows = provider.propose(object(), ["red bag", "blue case", "red bag"],
                            limit_candidates=2)

    assert [(row.label_hint, row.confidence) for row in rows] == [
        ("red bag", 0.9), ("blue case", 0.4),
    ]
    assert fake.phrases == ("red bag", "blue case")
    assert fake.predict_kwargs == {"device": "cpu", "verbose": False}


def test_provider_rejects_invalid_prompts_and_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    world, clip = _local_weights(tmp_path)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    provider = YoloWorldProposalProvider(
        world, clip, model_factory=lambda *_: _InjectedModel([]),
    )
    with pytest.raises(ValueError, match="iterable of strings"):
        provider.propose(object(), "one prompt")
    with pytest.raises(ValueError, match="positive integer"):
        provider.propose(object(), ["bag"], limit_candidates=1.5)  # type: ignore[arg-type]


def test_default_loader_uses_only_local_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    world_path, clip_path = _local_weights(tmp_path)
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("subprocess used"))
    monkeypatch.setattr(subprocess, "check_call", lambda *a, **k: pytest.fail("pip used"))
    monkeypatch.setattr(urllib.request, "urlretrieve", lambda *a, **k: pytest.fail("download used"))

    class Feature:
        ndim = 2
        shape = (1, 2)

        def float(self): return self
        def norm(self, **kwargs): return FeatureNorm()
        def __truediv__(self, other): return self
        def unsqueeze(self, dim): return self

    class FeatureNorm:
        def __gt__(self, other): return Truth()

    class Truth:
        def all(self): return True
        def __bool__(self): return True

    class NoGrad:
        def __enter__(self): return None
        def __exit__(self, *args): return False

    class LocalClipModel:
        def eval(self): return self
        def encode_text(self, tokens): return Feature()

    class Tokens:
        def to(self, device): return self

    class LocalModel:
        def __init__(self):
            self.model = [SimpleNamespace(nc=0)]
            self.txt_feats = None
            self.names = {}

        def eval(self): return self

    class LocalWorld:
        def __init__(self, path):
            assert path == str(world_path.resolve())
            self.model = LocalModel()
            self.predictor = None

        def to(self, device): return self
        def predict(self, frame, **kwargs): return []

    fake_clip = SimpleNamespace(
        load=lambda path, **kwargs: (
            LocalClipModel(), None
        ) if path == str(clip_path.resolve()) else pytest.fail("non-local CLIP path"),
        tokenize=lambda texts, truncate: Tokens(),
    )
    fake_torch = SimpleNamespace(
        no_grad=lambda: NoGrad(), cat=lambda chunks, dim: chunks[0],
        isfinite=lambda value: Truth(),
    )
    real_import = importlib.import_module

    def fake_import(name):
        replacements = {"ultralytics": SimpleNamespace(YOLOWorld=LocalWorld),
                        "clip": fake_clip, "torch": fake_torch}
        return replacements[name] if name in replacements else real_import(name)

    monkeypatch.setattr(importlib, "import_module", fake_import)
    provider = YoloWorldProposalProvider(world_path, clip_path)
    assert provider.propose(object(), ["yellow toolbox"]) == ()
