"""FeedbackManager — orchestrates the reinforcement-training loop.

    python -m cvti.feedback.manager status   --db runs/live/events.db
    python -m cvti.feedback.manager calibrate --db runs/live/events.db   # online: write calibration.json
    python -m cvti.feedback.manager export    --db runs/live/events.db   # offline: build training set
    python -m cvti.feedback.manager retrain   --db runs/live/events.db [--run]
    python -m cvti.feedback.manager rollback

Online loop (no GPU): `calibrate` reads operator labels and writes calibration.json
next to the events DB; the running pipeline reloads it and stops paging on
chronically-wrong (camera, rule) pairs. Offline loop: `export` turns labels into a
dataset, `retrain` fine-tunes the VideoMAE on it and registers the checkpoint;
`rollback` reverts to the previous model.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from cvti.feedback.calibration import Calibration
from cvti.feedback.dataset import export_dataset
from cvti.feedback.registry import ModelRegistry
from cvti.feedback.store import FeedbackStore
from cvti.logging_setup import get_logger

log = get_logger(__name__)


class FeedbackManager:
    def __init__(self, db_path: str, *, calibration_path: str | None = None,
                 dataset_dir: str = "runs/feedback/dataset",
                 registry_path: str = "runs/models/registry.json") -> None:
        self.db_path = db_path
        self.store = FeedbackStore(db_path)
        self.calibration_path = calibration_path or str(Path(db_path).with_name("calibration.json"))
        self.dataset_dir = dataset_dir
        self.registry = ModelRegistry(registry_path)

    # ---- online ----
    def calibrate(self) -> dict:
        cal = Calibration.from_store(self.store)
        cal.save(self.calibration_path)
        return {"calibration_path": self.calibration_path,
                "overall_precision": cal.overall_precision(),
                "demoted": cal.demoted_keys(),
                "rules": len(cal.rules)}

    # ---- offline ----
    def export(self) -> dict:
        res = export_dataset(self.store, self.dataset_dir)
        return res.to_dict()

    def retrain(self, *, run: bool = False, epochs: int = 3, backend: str = "videomae") -> dict:
        exp = self.export()
        cmd = ["python", "-m", "cvti.training.video_finetune", "--backend", backend,
               "--data-root", self.dataset_dir, "--epochs", str(epochs)]
        result = {"dataset": exp, "command": " ".join(cmd), "ran": False}
        if exp["total"] == 0:
            result["note"] = "No labeled clips to train on yet — label some alerts first."
            return result
        if not run:
            result["note"] = "Dry run — pass --run to actually fine-tune (needs time / a GPU)."
            return result
        import subprocess
        proc = subprocess.run(cmd, capture_output=True, text=True)
        result["ran"] = True
        result["returncode"] = proc.returncode
        result["stdout_tail"] = proc.stdout[-800:]
        result["stderr_tail"] = proc.stderr[-800:]
        if proc.returncode == 0:
            out_model = "runs/video_finetune/videomae"
            entry = self.registry.register(out_model, created=time.time(),
                                           metrics={"epochs": epochs, "clips": exp["total"]})
            result["registered"] = entry
        return result

    def rollback(self) -> dict:
        prev = self.registry.rollback()
        return {"rolled_back_to": prev} if prev else {"note": "Need >=2 registered models to roll back."}

    # ---- status ----
    def status(self) -> dict:
        counts = self.store.counts()
        cal = Calibration.from_store(self.store)
        rules = [st.to_dict() for st in sorted(
            cal.rules.values(), key=lambda s: (s.precision if s.precision is not None else 1.0))]
        return {
            "db": self.db_path,
            "labels": counts,
            "overall_precision": cal.overall_precision(),
            "demoted": cal.demoted_keys(),
            "rules": rules,
            "model": {"active": self.registry.active(), "versions": len(self.registry.versions())},
        }


def main() -> None:
    p = argparse.ArgumentParser(description="CVTI feedback / reinforcement-training manager.")
    p.add_argument("cmd", choices=("status", "calibrate", "export", "retrain", "rollback"))
    p.add_argument("--db", default="runs/live/events.db")
    p.add_argument("--dataset-dir", default="runs/feedback/dataset")
    p.add_argument("--run", action="store_true", help="retrain: actually fine-tune (else dry run)")
    p.add_argument("--epochs", type=int, default=3)
    args = p.parse_args()

    from cvti.logging_setup import setup_logging
    setup_logging(component="argus-feedback")

    mgr = FeedbackManager(args.db, dataset_dir=args.dataset_dir)
    import json
    if args.cmd == "status":
        out = mgr.status()
    elif args.cmd == "calibrate":
        out = mgr.calibrate()
    elif args.cmd == "export":
        out = mgr.export()
    elif args.cmd == "retrain":
        out = mgr.retrain(run=args.run, epochs=args.epochs)
    else:
        out = mgr.rollback()
    log.info(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
