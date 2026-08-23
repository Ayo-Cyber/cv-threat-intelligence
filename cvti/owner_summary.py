"""The weekly owner summary (EP-08-T2, USR-02/PD-06b).

The person who signs the cheque currently has no recurring contact with what
the product did. This is the fix: once a week, unprompted, a one-page summary
in business terms — incidents, outcomes, false alarms not chased, hours given
back — with month-over-month movement.

Every figure is a count of real rows in events.db / the suppression ledger
over a stated window; the money lines are those counts times rates the site
itself typed in, and are omitted when it hasn't. A number we invented is
worth nothing to the person deciding a renewal — so there are none.

Rendering reuses the incident-record PDF writer: the summary is a document a
buyer forwards, not a dashboard screenshot.
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

from cvti.logging_setup import get_logger

log = get_logger(__name__)

WEEK = 7 * 86400


def _window_stats(db_path: str | Path, start: float, end: float) -> dict:
    """Raw counts for one time window. Every key maps to one SQL count."""
    out = {"incidents": 0, "real": 0, "false_alarm": 0, "unverified": 0,
           "shown": 0, "noise_removed": 0, "active_days": 0,
           "by_camera": {}}
    try:
        con = sqlite3.connect(str(db_path))
        con.row_factory = sqlite3.Row
    except sqlite3.OperationalError:
        return out
    try:
        row = con.execute(
            "SELECT COUNT(*) n, "
            "SUM(CASE WHEN outcome='real' OR review='true' THEN 1 ELSE 0 END) r, "
            "SUM(CASE WHEN outcome='false_alarm' OR review='false' THEN 1 ELSE 0 END) f, "
            "SUM(COALESCE(unverified,0)) u "
            "FROM events WHERE ts >= ? AND ts < ? AND COALESCE(retracted,0)=0",
            (start, end)).fetchone()
        out.update(incidents=row["n"] or 0, real=row["r"] or 0,
                   false_alarm=row["f"] or 0, unverified=row["u"] or 0)
        for r in con.execute(
                "SELECT camera_id, COUNT(*) n FROM events WHERE ts >= ? AND ts < ? "
                "AND COALESCE(retracted,0)=0 GROUP BY camera_id ORDER BY n DESC",
                (start, end)):
            out["by_camera"][r["camera_id"]] = r["n"]
        day_a = time.strftime("%Y-%m-%d", time.localtime(start))
        day_b = time.strftime("%Y-%m-%d", time.localtime(end))
        try:
            row = con.execute(
                "SELECT SUM(shown) s, SUM(rejected+deduped) noise, COUNT(*) d "
                "FROM suppression_daily WHERE day >= ? AND day < ?",
                (day_a, day_b)).fetchone()
            out.update(shown=row["s"] or 0, noise_removed=row["noise"] or 0,
                       active_days=row["d"] or 0)
        except sqlite3.OperationalError:
            pass                          # ledger not created yet: a fresh site
    except sqlite3.OperationalError:
        log.warning("summary window query failed", exc_info=True)
    finally:
        con.close()
    return out


def compute_summary(db_path: str | Path, site_meta: dict, now: float | None = None) -> dict:
    """The figures for the report: this week, plus month-over-month movement."""
    now = now or time.time()
    this_week = _window_stats(db_path, now - WEEK, now)
    this_month = _window_stats(db_path, now - 4 * WEEK, now)
    prev_month = _window_stats(db_path, now - 8 * WEEK, now - 4 * WEEK)

    review_minutes = float(site_meta.get("review_minutes") or 2.0)
    hours_saved = round(this_week["noise_removed"] * review_minutes / 60.0, 1)
    money = {}
    guard_rate = float(site_meta.get("guard_hourly_cost") or 0)
    incident_value = float(site_meta.get("incident_value") or 0)
    if guard_rate > 0:
        money["attention_saved"] = round(hours_saved * guard_rate, 2)
    if incident_value > 0:
        money["incidents_value"] = round(this_week["real"] * incident_value, 2)

    def delta(key):
        a, b = prev_month.get(key, 0), this_month.get(key, 0)
        return {"prev": a, "now": b, "change": b - a}

    return {
        "site": site_meta.get("name") or "My Site",
        "window": {"from": time.strftime("%Y-%m-%d", time.localtime(now - WEEK)),
                   "to": time.strftime("%Y-%m-%d", time.localtime(now))},
        "week": this_week,
        "hours_saved": hours_saved,
        "money": money,
        "inputs": {"review_minutes": review_minutes,
                   "guard_hourly_cost": guard_rate, "incident_value": incident_value},
        "month_over_month": {k: delta(k) for k in
                             ("incidents", "real", "false_alarm", "noise_removed", "shown")},
        # Traceability: the exact provenance of every number above.
        "traceability": "every count above is a row count over events.db / the "
                        "suppression ledger for the stated window; money = counts x "
                        "the rates this site entered; nothing is modelled",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
    }


def _arrow(change: int) -> str:
    return "up " if change > 0 else "down " if change < 0 else "unchanged "


def render_pdf(summary: dict, dest: str | Path) -> Path:
    """One page a buyer can forward. Reuses the incident-record writer."""
    from cvti.incident_pdf import _PageWriter, _Pdf
    pdf = _Pdf()
    page = _PageWriter(pdf)
    w = summary["week"]

    page.text("ARGUS — WEEKLY SUMMARY", size=16, bold=True, gap=22)
    page.text(f"{summary['site']} — {summary['window']['from']} to "
              f"{summary['window']['to']} — generated {summary['generated_at']}",
              size=8, colour="0.45 0.48 0.55", gap=18)
    page.rule()

    page.text("This week", bold=True, gap=16)
    page.text(f"Confirmed incidents:  {w['incidents']}  "
              f"(real: {w['real']} - false alarms: {w['false_alarm']} - "
              f"unverified, reviewed by a person: {w['unverified']})")
    page.text(f"Alerts you were shown: {w['shown']}   -   "
              f"noise filtered before it reached anyone: {w['noise_removed']}")
    page.text(f"Attention given back: ~{summary['hours_saved']} hours "
              f"(at {summary['inputs']['review_minutes']:.0f} min per alert reviewed)")
    for label, amount in summary["money"].items():
        name = "Guard attention saved" if label == "attention_saved" else "Incident value protected"
        page.text(f"{name}: {amount:,.2f} (your rates x these counts)")
    if not summary["money"]:
        page.text("Money lines omitted: this site has not entered its rates. "
                  "(Settings -> Value)", size=9, colour="0.45 0.48 0.55")
    page.y -= 6

    page.text("Month over month (last 28 days vs the 28 before)", bold=True, gap=16)
    labels = {"incidents": "Confirmed incidents", "real": "Real incidents",
              "false_alarm": "False alarms", "noise_removed": "Noise filtered",
              "shown": "Alerts shown"}
    for key, lab in labels.items():
        d = summary["month_over_month"][key]
        page.text(f"{lab}:  {d['prev']} -> {d['now']}  ({_arrow(d['change'])}"
                  f"{abs(d['change'])})")
    page.y -= 6

    if w["by_camera"]:
        page.text("Where it happened", bold=True, gap=16)
        for cam, n in list(w["by_camera"].items())[:8]:
            page.text(f"{cam}:  {n} incident(s)")
        page.y -= 6

    page.rule()
    page.text(summary["traceability"], size=8, colour="0.45 0.48 0.55")
    if w["active_days"] < 7:
        page.text(f"Monitoring was active on {w['active_days']} of 7 days this week — "
                  f"figures cover only those days.", size=8, colour="0.55 0.35 0.05")

    page.done()
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(pdf.render())
    return dest


def weekly_summary(db_path: str | Path, site_meta: dict, out_dir: str | Path,
                   now: float | None = None) -> dict:
    """Build + persist one summary (json for machines, pdf for the buyer)."""
    s = compute_summary(db_path, site_meta, now=now)
    out = Path(out_dir) / "summaries"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"weekly-{s['window']['to']}"
    (out / f"{stem}.json").write_text(json.dumps(s, indent=2))
    pdf = render_pdf(s, out / f"{stem}.pdf")
    s["pdf"] = str(pdf)
    log.info("weekly summary written: %s", pdf)
    return s


def due(out_dir: str | Path, now: float | None = None) -> bool:
    """Monday 08:00+, at most once per ISO week. State survives restarts."""
    now = now or time.time()
    lt = time.localtime(now)
    if lt.tm_wday != 0 or lt.tm_hour < 8:
        return False
    state = Path(out_dir) / "summaries" / "state.json"
    try:
        last = json.loads(state.read_text()).get("last_week", "")
    except (OSError, ValueError):
        last = ""
    return last != time.strftime("%G-W%V", lt)


def mark_sent(out_dir: str | Path, now: float | None = None) -> None:
    state = Path(out_dir) / "summaries" / "state.json"
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps(
        {"last_week": time.strftime("%G-W%V", time.localtime(now or time.time()))}))
