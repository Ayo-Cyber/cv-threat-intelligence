"""Metrics for the two-stage evaluation.

Per clip we only care whether the system raised the threat at all — a clip is a
true positive if it's a threat clip and at least one alert fired on it. That
matches how an operator experiences it: one alert per incident is a catch, and
any alert on a normal clip is a false alarm.

Computing the SAME metrics before and after the gate is the point: the drop in
false positives (with recall held) is the value TrueSight adds.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

# 95% two-sided normal quantile. Every rate below is a proportion measured on a
# few dozen clips, so the point estimate on its own overstates what we know:
# 9/9 fires caught is a 95% interval of 66–100%, not a promise of 100%.
Z_95 = 1.959963984540054


def wilson_interval(k: int, n: int, z: float = Z_95) -> tuple[float, float] | None:
    """95% Wilson score interval for k successes in n trials.

    Wilson rather than the textbook normal interval because our n is small and
    our proportions sit at the edges (0 misses, 100% recall), where the normal
    interval degenerates to zero width and claims certainty we do not have.
    """
    if n <= 0:
        return None
    phat = k / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
    # The interval always contains the estimate; clamp to phat as well as to
    # [0,1] so float error can't produce a bound of 0.9999999999999999 for 30/30.
    lo = min(max(0.0, center - half), phat)
    hi = max(min(1.0, center + half), phat)
    return (lo, hi)


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

    # Each rate is k/n over a different denominator; the interval is only
    # meaningful next to the n it was measured on, so both are carried.
    @property
    def recall_n(self) -> int:
        return self.tp + self.fn

    @property
    def precision_n(self) -> int:
        return self.tp + self.fp

    @property
    def fpr_n(self) -> int:
        return self.fp + self.tn

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(precision=_r(self.precision), recall=_r(self.recall),
                 fpr=_r(self.fpr), f1=_r(self.f1))
        d.update(precision_n=self.precision_n, recall_n=self.recall_n, fpr_n=self.fpr_n,
                 precision_ci=_ci(self.tp, self.precision_n),
                 recall_ci=_ci(self.tp, self.recall_n),
                 fpr_ci=_ci(self.fp, self.fpr_n))
        return d


def _r(x: float | None) -> float | None:
    return round(x, 4) if x is not None else None


def _ci(k: int, n: int) -> list[float] | None:
    iv = wilson_interval(k, n)
    return [round(iv[0], 4), round(iv[1], 4)] if iv else None


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

    def rate(stage: dict, key: str) -> str:
        """A rate is never quoted without its n and its interval — at these sample
        sizes the point estimate alone is the misleading part."""
        n = stage.get(f"{key}_n")
        ci = stage.get(f"{key}_ci")
        out = pct(stage.get(key))
        if n:
            out += f" (n={n}"
            if ci:
                out += f", 95% CI {pct(ci[0])}–{pct(ci[1])}"
            out += ")"
        return out

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
        "Every rate carries its sample size and 95% Wilson interval.", "",
        "| | Raw detectors | + TrueSight |",
        "|---|---|---|",
        f"| Precision | {rate(raw, 'precision')} | **{rate(gat, 'precision')}** |",
        f"| Recall | {rate(raw, 'recall')} | {rate(gat, 'recall')} |",
        f"| False-alarm rate (normal clips) | {rate(raw, 'fpr')} | **{rate(gat, 'fpr')}** |",
        f"| Alerts raised | {raw['alerts']} | {gat['alerts']} |",
        f"| Missed threats | {raw['fn']} | {gat['fn']} |", "",
    ]
    if d["threats_lost_to_gate"]:
        lines += [f"> ⚠️ The gate rejected **{d['threats_lost_to_gate']}** real threat(s) the "
                  f"detectors caught — the cost of the precision gain.", ""]
    return "\n".join(lines)
