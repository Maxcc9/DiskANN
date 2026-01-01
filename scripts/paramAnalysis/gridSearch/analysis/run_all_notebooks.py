#!/usr/bin/env python3
"""Run all analysis notebooks in order and generate outputFiles/analyze/<report_prefix>/summary.md."""

from __future__ import annotations

import sys
from pathlib import Path
import os
from datetime import datetime

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def run_notebook(path: Path, timeout: int = 1200) -> None:
    with path.open("r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)
    ep = ExecutePreprocessor(timeout=timeout, kernel_name="python3")
    ep.preprocess(nb, {"metadata": {"path": str(path.parent)}})
    with path.open("w", encoding="utf-8") as f:
        nbformat.write(nb, f)


def latest_file(root: Path, pattern: str) -> str:
    files = sorted(root.glob(pattern))
    return str(files[-1]) if files else ""


def write_summary(report_dir: Path, analyze_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    figures = sorted((report_dir / "figures").glob("*.png"))
    tables = sorted((report_dir / "tables").glob("*.csv"))

    stats_csv = latest_file(analyze_dir, "collected_stats_*.csv")
    topk_csv = latest_file(analyze_dir, "collected_topk_*.csv")

    lines = []
    lines.append("# Analysis Summary\n")
    lines.append("\n")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("\n")
    lines.append("## Inputs\n")
    lines.append(f"- collected_stats: `{stats_csv}`\n" if stats_csv else "- collected_stats: (not found)\n")
    lines.append(f"- collected_topk: `{topk_csv}`\n" if topk_csv else "- collected_topk: (not found)\n")
    lines.append("\n")
    lines.append("## Figures\n")
    if figures:
        for fig in figures:
            lines.append(f"- `{fig}`\n")
    else:
        lines.append("- (none)\n")
    lines.append("\n")
    lines.append("## Tables\n")
    if tables:
        for tbl in tables:
            lines.append(f"- `{tbl}`\n")
    else:
        lines.append("- (none)\n")

    (report_dir / "summary.md").write_text("".join(lines), encoding="utf-8")


def main() -> int:
    analysis_dir = Path(__file__).resolve().parent
    analyze_dir = (analysis_dir / "../outputFiles/analyze").resolve()
    report_prefix = Path(os.environ.get("REPORT_PREFIX", "analysis_reports"))
    report_dir = (analyze_dir / report_prefix).resolve()

    notebooks = [
        "00_load_and_qc.ipynb",
        "01_basic_tradeoff_plots.ipynb",
        "02_bottleneck_attribution.ipynb",
        "03_graph_structure.ipynb",
        "04_surrogate_xgb.ipynb",
        "05_shap.ipynb",
        "06_worstcase_report.ipynb",
    ]

    for name in notebooks:
        path = analysis_dir / name
        if not path.exists():
            print(f"ERROR: missing notebook: {path}", file=sys.stderr)
            return 1
        print(f"Running: {path}")
        run_notebook(path)

    write_summary(report_dir, analyze_dir)
    print(f"Summary: {report_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
