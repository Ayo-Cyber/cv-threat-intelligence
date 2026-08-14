"""Metrics for the two-stage evaluation.

Per clip we only care whether the system raised the threat at all — a clip is a
true positive if it's a threat clip and at least one alert fired on it. That
matches how an operator experiences it: one alert per incident is a catch, and
any alert on a normal clip is a false alarm.

Computing the SAME metrics before and after the gate is the point: the drop in
false positives (with recall held) is the value TrueSight adds.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class StageMetrics:
    stage: str
    tp: int = 0          # threat clip, alert raised
    fn: int = 0          # threat clip, nothing raised      (missed threat)
    fp: int = 0          # normal clip, alert raised        (false alarm)
    tn: int = 0          # normal clip, nothing raised
    alerts: int = 0      # total alerts (an operator sees these, not clips)

    @property
    def recall(self) -> float | None:
        d = self.tp + self.fn
        return self.tp / d if d else None

    @property
    def precision(self) -> float | None:
        d = self.tp + self.fp
        return self.tp / d if d else None

    @property
    def fpr(self) -> float | None:
        """Share of NORMAL clips that raised a false alarm."""
        d = self.fp + self.tn
        return self.fp / d if d else None

    @property
    def f1(self) -> float | None:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p and r and (p + r) else None

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(precision=_r(self.precision), recall=_r(self.recall),
                 fpr=_r(self.fpr), f1=_r(self.f1))
        return d


def _r(x: float | None) -> float | None:
    return round(x, 4) if x is not None else None


def score(clip_results: list[dict], stage: str, key: str) -> StageMetrics:
    """clip_results: [{'is_threat': bool, '<key>': <alert count>}, ...]"""
    m = StageMetrics(stage=stage)
    for r in clip_results:
        fired = (r.get(key) or 0) > 0
        m.alerts += r.get(key) or 0
        if r["is_threat"]:
            if fired:
                m.tp += 1
            else:
                m.fn += 1
        else:
            if fired:
                m.fp += 1
            else:
                m.tn += 1
    return m


def compare_stages(clip_results: list[dict]) -> dict:
    """Stage 1 (raw detector candidates) vs Stage 2 (TrueSight-confirmed)."""
    raw = score(clip_results, "raw_detectors", "candidates")
    gated = score(clip_results, "truesight_confirmed", "confirmed")

    suppressed = raw.alerts - gated.alerts
    fp_removed = raw.fp - gated.fp
    missed_cost = gated.fn - raw.fn        # threats the gate wrongly threw away

    return {
        "raw_detectors": raw.to_dict(),
        "truesight_confirmed": gated.to_dict(),
        "delta": {
            "alerts_suppressed": suppressed,
            "alerts_suppressed_pct": _r(suppressed / raw.alerts) if raw.alerts else None,
            "false_alarm_clips_removed": fp_removed,
            "precision_gain": _r((gated.precision or 0) - (raw.precision or 0))
            if raw.precision is not None and gated.precision is not None else None,
            "threats_lost_to_gate": missed_cost,
        },
    }


def render_report(summary: dict, dataset: dict, meta: dict) -> str:
    """A short markdown report — the thing you paste into a deck or a pitch."""
    raw = summary["raw_detectors"]
    gat = summary["truesight_confirmed"]
    d = summary["delta"]

    def pct(x):
        return "—" if x is None else f"{x*100:.1f}%"

    lines = [
        "# Argus — evaluation report", "",
        f"**Clips:** {dataset['total']} ({dataset['threat']} threat / {dataset['normal']} normal) "
        f"· sources: {', '.join(dataset['sources'])}",
        f"**Gate:** {meta.get('gate', '?')} · **model:** {meta.get('gate_model', '—')}", "",
        "## Headline", "",
        f"TrueSight suppressed **{d['alerts_suppressed']}** of {raw['alerts']} raw alerts "
        f"({pct(d['alerts_suppressed_pct'])}), removing **{d['false_alarm_clips_removed']}** "
        f"false-alarm clips.", "",
        f"Precision **{pct(raw['precision'])} → {pct(gat['precision'])}** "
        f"with recall **{pct(raw['recall'])} → {pct(gat['recall'])}**.", "",
        "## Both stages", "",
        "| | Raw detectors | + TrueSight |",
        "|---|---|---|",
        f"| Precision | {pct(raw['precision'])} | **{pct(gat['precision'])}** |",
        f"| Recall | {pct(raw['recall'])} | {pct(gat['recall'])} |",
        f"| False-alarm rate (normal clips) | {pct(raw['fpr'])} | **{pct(gat['fpr'])}** |",
        f"| Alerts raised | {raw['alerts']} | {gat['alerts']} |",
        f"| Missed threats | {raw['fn']} | {gat['fn']} |", "",
    ]
    if d["threats_lost_to_gate"]:
        lines += [f"> ⚠️ The gate rejected **{d['threats_lost_to_gate']}** real threat(s) the "
                  f"detectors caught — the cost of the precision gain.", ""]
    return "\n".join(lines)
