#!/usr/bin/env python3
"""Assemble README.md from the template plus generated result tables.

The template (``scripts/README_template.md``) holds all hand-written prose. The
results section is generated from ``results/raw/*.csv`` by
``build_results_section.py``, and the findings section is read from
``scripts/README_findings.md`` if present. This guarantees that no metric in
the README can drift from the committed experiment output.

Usage::

    python scripts/build_readme.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from build_results_section import EXPERIMENTS, load_raw, render, sensitivity_block  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "scripts" / "README_template.md"
FINDINGS = ROOT / "scripts" / "README_findings.md"
OUTPUT = ROOT / "README.md"

RESULTS_MARKER = "<!-- RESULTS_SECTION -->"
FINDINGS_MARKER = "<!-- FINDINGS_SECTION -->"


def build_results() -> str:
    """Render every experiment block, marking un-run experiments PENDING."""
    blocks = [
        sensitivity_block() if name == "sensitivity" else render(name, heading)
        for name, heading, _ in EXPERIMENTS
    ]
    return "\n\n".join(blocks)


def build_findings() -> str:
    """Return the hand-written findings section, or a placeholder if absent."""
    if FINDINGS.exists():
        return FINDINGS.read_text(encoding="utf-8")
    return (
        "## 11. Findings\n\n"
        "**PENDING** — written once all experiments have been executed.\n"
    )


def main() -> int:
    """Write README.md and report which experiments are still pending."""
    if not TEMPLATE.exists():
        print(f"template not found: {TEMPLATE}", file=sys.stderr)
        return 1
    text = TEMPLATE.read_text(encoding="utf-8")
    for marker in (RESULTS_MARKER, FINDINGS_MARKER):
        if marker not in text:
            print(f"template is missing marker {marker}", file=sys.stderr)
            return 1
    text = text.replace(RESULTS_MARKER, build_results())
    text = text.replace(FINDINGS_MARKER, build_findings())
    OUTPUT.write_text(text, encoding="utf-8")

    pending = [name for name, _, _ in EXPERIMENTS if load_raw(name) is None]
    print(f"wrote {OUTPUT.relative_to(ROOT)} ({len(text.splitlines())} lines)")
    if pending:
        print(f"PENDING experiments (marked as such in README): {', '.join(pending)}")
    else:
        print("all experiments have results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
