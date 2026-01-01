# Param Analysis Research Plan

This document consolidates the research workflow and the hard constraints for analysis code generation
based on the CSV outputs produced by `collect.py`.

## 0) Global Rules (Must Follow)

Source of truth
- CSV outputs from `collect.py` are the only authoritative inputs.
- Raw logs are optional and must not be parsed unless explicitly requested.

No silent assumptions
- All thresholds, filters, and ratios must be parameterized via config/constants.
- No hard-coded magic numbers inside analysis code.

Strict execution order
- Layer 2 (ML / surrogate modeling) must NOT run unless Layer 1 checks pass.
- Violations must stop the pipeline and emit a clear message.

## 1) Inputs

### 1.1 Required
- `outputFiles/analyze/collected_stats_{search_dir}_{timestamp}.csv`
- `outputFiles/analyze/collected_topk_{search_dir}_{timestamp}.csv`

### 1.2 Optional (only if present)
- `outputFiles/search/*/*_expanded_nodes.csv`
- `outputFiles/search/*/*_iostat.log`
- `outputFiles/search/*/*_topk{K}_neighbors.csv`

## 2) Configuration (Must Externalize)

The analysis must define a config object (YAML/JSON/Python constants), including at least:

```
recall_threshold: 0.7
degenerate_beam_factor: 2.0
min_valid_queries: null   # if null, infer from dataset
latency_target_quantile: p99
io_bound_threshold: 0.6   # for io_share_p99
random_seed: 42
```

All analyses must reference these values, not inline literals.

## 3) Stage A – Data Quality & Reproducibility (Gate 1)

### 3.1 Key Integrity

For both CSVs:
- Primary key: `run_prefix`
- Enforce:
  - `run_prefix` uniqueness
  - all build/search parameters constant within a `run_prefix`
  - `num_queries` consistent across runs (or explicitly flagged)

Failure action:
- Stop pipeline with descriptive error.

### 3.2 Degenerate Configuration Handling

For each run:
- Exclude if: `search_L < search_K`
- Flag (do not exclude) if: `search_L < degenerate_beam_factor * search_W`

Recall handling:
- Runs with `recall_mean < recall_threshold` must be:
  - either excluded from latency comparisons, or
  - analyzed in a separate low-recall group

This logic must be implemented explicitly.

### 3.3 System Load / Drift Control

Use the following only as quality-control signals:
- `iostat_%util_mean`
- `iostat_aqu-sz_mean`
- `iostat_r_await_mean`

Rules:
- Do NOT treat iostat metrics as primary outcomes.
- Either filter overloaded runs, or include them as covariates.
- Decision must be logged.

## 4) Stage B – Research Target Definition (Single Primary Focus)

Assume exactly one primary research objective:
- Predictability and drivers of tail latency (p99 / p999) in SSD-based ANN search.

All other analyses are secondary and must support this goal.

## 5) Stage C – Derived Feature Construction (Mandatory)

### 5.1 Normalization
- `log_latency_p99 = log(latency_p99_us)`
- Preserve `dataset_size`, `vector_dim` as covariates.

### 5.2 Mechanism-Driven Ratios

(All divisions must guard against zero or missing values.)

- `io_share_p99    = io_us_p99   / max(latency_p99_us, 1)`
- `cpu_share_p99   = cpu_us_p99  / max(latency_p99_us, 1)`
- `sort_share_p99  = sort_us_p99 / max(latency_p99_us, 1)`
- `io_cost_p99     = io_us_p99 / max(ios_p99, 1)`
- `io_per_hop_p99  = ios_p99  / max(hop_p99, 1)`
- `compares_per_visited = compares_mean / max(visited_mean, 1)`
- `hot_top1  = node_counts_top1_share`
- `hot_top10 = node_counts_top10_share`

If any derived value is NaN or infinite:
- Flag run
- Exclude from modeling

## 6) Stage D – Layer 1 Analysis (Mechanism Validation)

### 6.1 Required Plots (Must All Exist)

Recall vs Tail Latency
- x: `recall_mean`
- y: `latency_p99_us`
- color: one of `{search_W, search_L, search_T}`

I/O vs Tail Latency
- x: `ios_p99` or `io_us_p99`
- y: `latency_p99_us`

Queueing Amplification
- x: `queue_depth_p99` or `iostat_aqu-sz_mean`
- y: `latency_p99_us`
- color: `search_T`

IO-bound vs CPU-bound Scatter
- x: `io_share_p99`
- y: `cpu_share_p99`

Graph Structure to Worst-case
- x: `out_degree_p99` or `hot_top10`
- y: `latency_p99_us` or `expanded_revisit_ratio`

### 6.2 Gate Condition for Layer 2

Layer 2 may proceed only if:
- At least one clear monotonic or clustered trend is observable in Layer 1.

Otherwise:
- Skip surrogate modeling
- Emit message: `Layer 2 skipped: no stable mechanism observed`

## 7) Stage E – Layer 2 Analysis (Surrogate + SHAP)

### 7.1 Models
- Model A: regression → `log_latency_p99`
- Model B: regression → `ios_p99`
- Model C: classification → IO-bound vs CPU-bound (`io_share_p99 >= io_bound_threshold`)

### 7.2 Feature Sets
- S: build/search params + dataset meta
- M: S + graph/expanded/node_counts
- L: M + iostat summaries

Train/evaluate separately.

### 7.3 SHAP Usage Rules

SHAP is for trend validation, not causal claims.

Required outputs:
- Global feature importance (top 10)
- Local SHAP for top 5 worst-case runs

## 8) Stage F – Worst-case Reporting

### 8.1 Baseline Definition
- Worst-case = `latency_p99_us` (primary)
- Optional: `latency_p999_us`, `latency_max_us`

### 8.2 Reporting
- MAE / median absolute error on holdout set
- Table of worst runs:
  - `recall_mean`
  - `latency_p99_us`
  - `ios_p99`
  - `queue_depth_p99`
  - `io_share_p99`
  - `thread_util_p99`

## 9) Expected Code Outputs

```
analysis/
  config.yaml
  qc.py
  features.py
  plots_layer1.py
  model_layer2.py
  shap_analysis.py
reports/
  figures/
  tables/
```

Each script must be runnable independently.

## 10) Explicit Non-Goals

Do NOT invent new metrics.
Do NOT claim causality from ML models.
Do NOT mix build and search effects without grouping.
Do NOT proceed silently on failed gates.

## Appendix: Original Workflow Overview (Human-Oriented Summary)

This appendix provides a concise, human-readable overview for planning and reporting.

### A) Research Questions and Core KPIs

Pick one primary question and treat others as secondary.

1) Tail latency predictability
- KPI: `latency_p99_us` (primary), `latency_p999_us` or `latency_max_us` (secondary)
- Supporting: `ios_p99`, `queue_depth_p99`, `io_us_p99`, `cpu_us_p99`

2) Bottleneck attribution
- KPI: `io_us_*`, `cpu_us_*`, `sort_us_*` (p99 and mean)
- Supporting: `queue_depth_*`, `thread_util_*`

3) Graph structure drives worst-case
- KPI: `latency_p99_us`, `expanded_revisit_ratio`, `expanded_steps_p90`
- Supporting: `out_degree_p99`, `out_degree_max`, `node_counts_top10_share`

### B) Minimal Deliverables (Fast Path)

1) Table of worst runs:
- `recall_mean`, `latency_p99_us`, `ios_p99`, `queue_depth_p99`, `io_share_p99`, `thread_util_p99`

2) Two core plots:
- `ios_p99` vs `latency_p99_us` (color by `search_T`)
- `queue_depth_p99` vs `latency_p99_us` (color by `search_T`)

3) Small XGBoost:
- target: `log(latency_p99_us)`
- features: build/search + `expanded_revisit_ratio` + `node_counts_top10_share` + `iostat_aqu-sz_mean`

### C) Suggested Project Layout

```
analysis/
  00_load_and_qc.ipynb
  01_basic_tradeoff_plots.ipynb
  02_bottleneck_attribution.ipynb
  03_graph_structure.ipynb
  04_surrogate_xgb.ipynb
  05_shap.ipynb
  06_worstcase_report.ipynb
reports/
  figures/
  tables/
```
