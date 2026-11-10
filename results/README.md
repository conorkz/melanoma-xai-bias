# Benchmark results

This directory is populated by `scripts/collect_results.py`. It stays empty
until the benchmark runs finish. Raw run outputs live under `outputs/runs/` and
are gitignored.

## Files produced after Stage 1

- `rq1_stage1_runs.csv`: one row per run, including seed and bootstrap interval.
- `rq1_stage1.csv`: mean and standard deviation across seeds, one row per configuration.
- `rq1_stage1.md`: markdown version of the benchmark table.

Command:

```bash
python scripts/collect_results.py \
  --runs-dir outputs/runs --run-prefix stage1 \
  --out-dir results --prefix rq1_stage1
```

## Files produced after Stage 2

- `rq1_stage2_runs.csv`
- `rq1_stage2.csv`
- `rq1_stage2.md`

Command:

```bash
python scripts/collect_results.py \
  --runs-dir outputs/runs --run-prefix stage2 \
  --out-dir results --prefix rq1_stage2
```

## The memo

`RESULTS_MEMO_TEMPLATE.md` in the project root is the template. After both
tables exist:

```bash
cp RESULTS_MEMO_TEMPLATE.md RESULTS_MEMO.md
```

Then paste the two tables into the memo and fill in the interpretation.
