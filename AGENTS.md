# Repository Guidelines

## Project Structure & Module Organization
- `src/` C++ core (graph builders, search, IO); mirror headers in `include/`.
- `apps/` CLI binaries (build/search); artifacts under `build/apps/`.
- `python/src/diskannpy/` PyBind wrapper; tests in `python/tests/`.
- `tests/` Boost unit tests; datasets under `test_data/`, golden outputs in `test_run/`.
- `scripts/`, `workflows/` reproducible scenarios (SSD, filtered, dynamic, param analysis grid search).
- `scripts/paramAnalysis/gridSearch/` grid-search tooling for build/search batches and offline analysis.
- Output naming: search artifacts use prefix `S{search_id}_{index_tag}_W{W}_L{L}_K{K}_cache{cache}_T{threads}`.
- Top-K analysis outputs: `*_topk{K}_nodes.txt`, `*_topk{K}_neighbors.csv`.
- Aggregation outputs: `outputFiles/analyze/collected_stats_{search_dir}_{timestamp}.csv` and `outputFiles/analyze/collected_topk_{search_dir}_{timestamp}.csv`.
- Analysis reports: `outputFiles/analyze/<REPORT_PREFIX>/figures/`, `outputFiles/analyze/<REPORT_PREFIX>/tables/`, `outputFiles/analyze/<REPORT_PREFIX>/summary.md`.
- Notebook helper: `scripts/paramAnalysis/gridSearch/analysis.ipynb`.
- Batch tooling supports `EXPERIMENT_TAG` to create per-run subfolders under `outputFiles/build` and `outputFiles/search`.
- `build_batch.sh` and `search_batch.sh` require named args (`--build-csv`, `--search-csv`, `--dataset`, `--max-parallel`).
- `rust/` Rust crates; follow Cargo workflows.

## Build, Test, and Development Commands
- Configure: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release` (MKL/OpenMP on path).
- Build all: `cmake --build build --target all -- -j`.
- C++ tests: `cmake --build build --target diskann_unit_tests` then `ctest --test-dir build --output-on-failure`.
- Python: `pip install -e .[dev]` then `python -m unittest discover python/tests`.
- End-to-end smoke: `./unit_tester.sh build ./catalog.txt ./tmp_workdir`.
- Rust: `cd rust && cargo build` (or `-r`) and `cargo test`.

## Coding Style & Naming Conventions
- C++: `.clang-format` (Microsoft style, 4 spaces, deterministic includes). Namespace `diskann`, files/functions `snake_case`, types/configs `CamelCase`.
- Python: `black`/`isort` per `pyproject.toml`; add type hints for public APIs.
- Rust: `cargo fmt` and `cargo clippy`.
- CLI/tool filenames mirror command names (e.g., `build_disk_index.cpp`).
- Offline analysis helper: `apps/dump_disk_neighbors.cpp` (dump neighbor lists for node ids).

## Testing Guidelines
- Add Boost tests as separate `.cpp` with `BOOST_AUTO_TEST_SUITE`; register in `tests/CMakeLists.txt`.
- Cover build-time and query-time paths; check budget/latency logging.
- Python tests `test_*.py` with `unittest` + `numpy.testing`.
- Rust tests live in each crate; keep `rust/cmd_drivers/` examples current.
- Update `workflows/*.md` when adding scenarios; stash catalogs in `test_run/`.
- For param analysis updates, keep `workflows/param_analysis_gridsearch.md` in sync.

## Commit & Pull Request Guidelines
- Commit format: `Type: short imperative summary (#issue)` (e.g., `Fix: clamp PQ chunk count (#654)`).
- Each commit should build and pass tests independently; squash WIP.
- PRs: describe scenario, list verification commands, link issues/workflow docs; attach screenshots or benchmark deltas for search-quality changes and note MKL/dataset needs.

## Security & Configuration Tips
- Review `SECURITY.md` before reporting vulnerabilities; scrub sensitive dataset paths from logs.
- Document new env knobs (e.g., `OMP_PATH`, memory budgets) in the relevant workflow guide with conservative defaults.

## 回應語言
- 用繁體中文回覆使用者

## AGENTS.md 編寫建議
- 用簡短標題與條列，保持可掃描性。
- 只寫專案通用規範與可操作指令，避免一次性細節。
- 避免敏感資訊（私人路徑、憑證、內網網址）。
- 內容需與現況同步，新增工具/流程時一併更新。
