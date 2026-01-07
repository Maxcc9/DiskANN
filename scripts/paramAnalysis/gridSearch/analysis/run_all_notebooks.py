#!/usr/bin/env python3
"""Run all analysis notebooks in order and generate outputFiles/analyze/<report_prefix>/summary.md."""

from __future__ import annotations

import sys
from pathlib import Path
import os
from datetime import datetime
import csv
import json

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


def read_csv_head(path: Path, max_rows: int = 5) -> tuple[list[dict[str, str]], list[str]]:
    if not path.exists():
        return [], []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for idx, row in enumerate(reader):
            if idx >= max_rows:
                break
            rows.append(row)
        return rows, reader.fieldnames or []


def save_config_json(report_dir: Path) -> None:
    """Save analysis parameters to config.json for future reference."""
    report_dir.mkdir(parents=True, exist_ok=True)
    
    # 定義所有分析參數及其預設值
    analysis_params = {
        'REPORT_PREFIX': 'analysis_reports',
        'COLLECT_PREFIX': 'REPORT_PREFIX',
        'FILTER_SEARCH_K': '10',
        'PLOT_MAX_POINTS': '20000',
        'PLOT_LOG_LATENCY': '1',
        'QC_RECALL_THRESHOLD': '0.7',
        'QC_RECALL_PCTL': '0',
        'QC_OUTLIER_Z': '4.0',
        'SHAP_MAX_SAMPLES': '2000',
        'MODEL_TEST_SIZE': '0.2',
        'MODEL_RANDOM_STATE': '42',
        'WORSTCASE_PCTL': '0.95',
        'WORSTCASE_MIN_COUNT': '10',
        'WORSTCASE_MAX_SAMPLES': '200',
        'BOTTLENECK_SHARE_THRESHOLD': '0.5',
    }
    
    config = {
        'generated_at': datetime.now().isoformat(),
        'parameters': {}
    }
    
    for key, default in analysis_params.items():
        value = os.environ.get(key, default)
        config['parameters'][key] = {
            'value': value,
            'source': 'environment' if key in os.environ else 'default'
        }
    
    config_path = report_dir / 'config.json'
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f'Saved config: {config_path}')


def write_summary(report_dir: Path, analyze_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    figures = sorted((report_dir / "figures").glob("*.png"))
    tables = sorted((report_dir / "tables").glob("*.csv"))
    tables_dir = report_dir / "tables"

    stats_csv = latest_file(analyze_dir, "collected_stats_*.csv")
    topk_csv = latest_file(analyze_dir, "collected_topk_*.csv")

    lines = []
    lines.append("# Analysis Summary\n")
    lines.append("\n")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    lines.append("\n")
    
    # 新增：參數紀錄段落
    lines.append("## Analysis Parameters\n")
    lines.append("\n")
    analysis_params = {
        'REPORT_PREFIX': 'analysis_reports',
        'COLLECT_PREFIX': 'REPORT_PREFIX',
        'FILTER_SEARCH_K': '10',
        'PLOT_MAX_POINTS': '20000',
        'PLOT_LOG_LATENCY': '1',
        'QC_RECALL_THRESHOLD': '0.7',
        'QC_RECALL_PCTL': '0',
        'QC_OUTLIER_Z': '4.0',
        'SHAP_MAX_SAMPLES': '2000',
        'MODEL_TEST_SIZE': '0.2',
        'MODEL_RANDOM_STATE': '42',
        'WORSTCASE_PCTL': '0.95',
        'WORSTCASE_MIN_COUNT': '10',
        'WORSTCASE_MAX_SAMPLES': '200',
        'BOTTLENECK_SHARE_THRESHOLD': '0.5',
    }
    for key, default in analysis_params.items():
        value = os.environ.get(key, default)
        source = '(env)' if key in os.environ else '(default)'
        lines.append(f"- `{key}`: `{value}` {source}\n")
    lines.append("\n")
    
    lines.append("## Inputs\n")
    lines.append(f"- collected_stats: `{stats_csv}`\n" if stats_csv else "- collected_stats: (not found)\n")
    lines.append(f"- collected_topk: `{topk_csv}`\n" if topk_csv else "- collected_topk: (not found)\n")
    lines.append("\n")
    lines.append("## QC\n")
    qc_summary = tables_dir / "qc_summary.csv"
    qc_rows, _ = read_csv_head(qc_summary, max_rows=1)
    if qc_rows:
        row = qc_rows[0]
        lines.append("- qc_summary:\n")
        keys = [
            "total_rows",
            "filtered_rows",
            "excluded_rows",
            "low_recall_threshold",
        ]
        for key in keys:
            if key in row:
                lines.append(f"  - {key}: {row[key]}\n")
    else:
        lines.append("- qc_summary: (not found)\n")

    lines.append("\n")
    lines.append("## Tradeoff\n")
    for name in [
        "pareto_recall_latency.csv",
        "tradeoff_by_params.csv",
    ]:
        path = tables_dir / name
        if path.exists():
            lines.append(f"- {name}: `{path}`\n")
    corr_files = sorted(tables_dir.glob("latency_correlation_*.csv"))
    if corr_files:
        corr_rows, _ = read_csv_head(corr_files[-1], max_rows=5)
        lines.append(f"- latency_correlation: `{corr_files[-1]}`\n")
        for row in corr_rows:
            feat = row.get("feature", "")
            spearman = row.get("spearman", "")
            count = row.get("count", "")
            lines.append(f"  - {feat}: spearman={spearman} count={count}\n")

    lines.append("\n")
    lines.append("## Bottleneck Analysis\n")
    bottleneck_summary = sorted(tables_dir.glob("bottleneck_summary_*.csv"))
    if bottleneck_summary:
        rows, _ = read_csv_head(bottleneck_summary[-1], max_rows=1)
        if rows:
            lines.append(f"See: `{bottleneck_summary[-1].name}`\n")
    
    lines.append("\n")
    lines.append("## Surrogate Model Performance\n")
    model_metrics = tables_dir / "model_metrics.csv"
    if model_metrics.exists():
        rows, _ = read_csv_head(model_metrics, max_rows=1)
        if rows:
            lines.append(f"See: `{model_metrics.name}`\n")
    
    lines.append("\n")
    lines.append("## Feature Importance (SHAP)\n")
    shap_files = sorted(tables_dir.glob("shap_mean_abs_latency_p99_us.csv"))
    if shap_files:
        rows, _ = read_csv_head(shap_files[0], max_rows=3)
        lines.append(f"Top features affecting latency_p99_us:\n")
        for row in rows:
            feature = row.get("feature", "")
            value = row.get("mean_abs", "")
            if feature:
                lines.append(f"  - {feature}: {value}\n")

    lines.append("\n")
    lines.append("## Worst-Case Analysis\n")
    worstcase_summary = tables_dir / "worstcase_summary.csv"
    if worstcase_summary.exists():
        rows, _ = read_csv_head(worstcase_summary, max_rows=1)
        if rows:
            row = rows[0]
            lines.append(f"- Worst-case threshold: {row.get('threshold','')} (rate: {row.get('worstcase_rate','')})\n")
    worstcase_model = tables_dir / "worstcase_model_metrics.csv"
    if worstcase_model.exists():
        rows, _ = read_csv_head(worstcase_model, max_rows=1)
        if rows:
            row = rows[0]
            lines.append(f"- Model ROC AUC: {row.get('roc_auc','')} (Accuracy: {row.get('accuracy','')})\n")
    
    lines.append("\n")
    lines.append("## Output Files\n")
    lines.append(f"All detailed results are in: `tables/` and `figures/`\n")
    lines.append(f"Key tables:\n")
    key_tables = [
        'filtered_stats.csv',
        'pareto_recall_latency.csv',
        'tradeoff_by_params.csv',
        'qc_summary.csv',
        'bottleneck_summary_latency_p99_us.csv',
        'worstcase_summary.csv',
    ]
    for table_name in key_tables:
        table_path = tables_dir / table_name
        if table_path.exists():
            lines.append(f"  - `{table_name}`\n")

    (report_dir / "summary.md").write_text("".join(lines), encoding="utf-8")


def main() -> int:
    analysis_dir = Path(__file__).resolve().parent
    analyze_dir = (analysis_dir / "../outputFiles/analyze").resolve()
    report_prefix = Path(os.environ.get("REPORT_PREFIX", "analysis_reports"))
    collect_prefix = Path(os.environ.get("COLLECT_PREFIX", str(report_prefix)))
    report_dir = (analyze_dir / report_prefix).resolve()
    collect_dir = (analyze_dir / collect_prefix).resolve()

    stats_csv = latest_file(collect_dir, "collected_stats_*.csv")
    topk_csv = latest_file(collect_dir, "collected_topk_*.csv")
    if not stats_csv or not topk_csv:
        print("ERROR: collected_stats/topk not found in:", collect_dir, file=sys.stderr)
        print("Hint: run `python collect.py` or set COLLECT_PREFIX to the correct folder.", file=sys.stderr)
        return 1

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

    # 先保存配置，再生成摘要
    save_config_json(report_dir)
    write_summary(report_dir, collect_dir)
    print(f"Summary: {report_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
