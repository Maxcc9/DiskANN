# Repository Guidelines

## Project Structure & Module Organization
- `src/` C++ core (graph builders, search, IO); mirror headers in `include/`.
- `apps/` CLI binaries (build/search); artifacts under `build/apps/`.
- `python/src/diskannpy/` PyBind wrapper; tests in `python/tests/`.
- `tests/` Boost unit tests; datasets under `test_data/`, golden outputs in `test_run/`.
- `scripts/`, `workflows/` reproducible scenarios (SSD, filtered, dynamic).
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

## Testing Guidelines
- Add Boost tests as separate `.cpp` with `BOOST_AUTO_TEST_SUITE`; register in `tests/CMakeLists.txt`.
- Cover build-time and query-time paths; check budget/latency logging.
- Python tests `test_*.py` with `unittest` + `numpy.testing`.
- Rust tests live in each crate; keep `rust/cmd_drivers/` examples current.
- Update `workflows/*.md` when adding scenarios; stash catalogs in `test_run/`.

## Commit & Pull Request Guidelines
- Commit format: `Type: short imperative summary (#issue)` (e.g., `Fix: clamp PQ chunk count (#654)`).
- Each commit should build and pass tests independently; squash WIP.
- PRs: describe scenario, list verification commands, link issues/workflow docs; attach screenshots or benchmark deltas for search-quality changes and note MKL/dataset needs.

## Security & Configuration Tips
- Review `SECURITY.md` before reporting vulnerabilities; scrub sensitive dataset paths from logs.
- Document new env knobs (e.g., `OMP_PATH`, memory budgets) in the relevant workflow guide with conservative defaults.

## 回應語言
- 用繁體中文回覆使用者