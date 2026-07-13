# Handoffs

Point-in-time engineering handoffs — what was done in a work session, why, where
the code lives, how to run/verify it, known caveats, and what's next. Read the
newest one to get caught up; each is a snapshot, not a living document.

## Naming convention

```
YYYY-MM-DD-HHMM-<kebab-slug>.md
```

- **`YYYY-MM-DD-HHMM`** — date and 24h local time (WAT) the handoff was **written**,
  so anyone can see when it was authored and files sort oldest → newest.
- **`<kebab-slug>`** — a few words on the theme (e.g. `merge-serving-phases-1-4`).

Example: `2026-07-13-2044-merge-serving-phases-1-4.md`

## Rules

- **Append, don't edit.** A handoff reflects what was true when written. Start a
  new dated file for new work rather than rewriting an old one.
- Put the same date/time in the file's header block (see the template) so it's
  visible when reading the doc directly, not just in the filename.
- Keep the index below up to date (newest first).

## Header template

```markdown
# <Title>

> **Written:** YYYY-MM-DD HH:MM WAT (<Weekday>)
> **Author:** <name>
> **Branch:** <branch(es)>
> **Commit range:** <base>..<head>
```

## Index (newest first)

- [2026-07-13-2044 — Merge + multi-stream serving + plan.md Phases 1–4](2026-07-13-2044-merge-serving-phases-1-4.md)
