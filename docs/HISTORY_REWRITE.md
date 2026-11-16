# History Rewrite of 2026-11-16

On 2026-11-16 the commit history of this repository was rewritten once, at the author's request, for
two reasons:

1. Twenty commits carried a `Co-Authored-By` line. The author is solely accountable for this
   repository, so those lines were removed.
2. The author and committer name was changed from `olagoke-dsae` to `aolagoke-dsae`, the author's
   GitHub login, and the author's personal email address was replaced by the private address
   GitHub provides for that account.

## What changed and what did not

| | |
| :--- | :--- |
| File contents of every commit | unchanged (all 29 file trees are identical) |
| Author date and committer date of every commit | unchanged |
| Commit messages | unchanged, except for the removed line in 20 of them |
| Commit IDs | all 29 changed, because an ID is a hash of the message, names, email and parent |

Verification performed after the rewrite: the tree hash, author date and committer date of
each commit were compared with the original, and the content difference between the original and
rewritten heads is zero lines.

## Why this matters for the held-out evaluation

`docs/ANALYSIS_PLAN_H1_H3.md` was committed before the analysis code and before the results, and
the commit IDs were cited as evidence of that order. The IDs have changed; the order and the dates
have not:

| Step | Original ID | Current ID | Commit date |
| :--- | :--- | :--- | :--- |
| Analysis plan approved | `23bf83d` | `6c8e730` | 2026-11-15 19:46 |
| Analysis code | `65f4de2` | `292c883` | 2026-11-15 20:19 |
| Results, as produced | `079e855` | `601a320` | 2026-11-15 20:47 |

Commit dates are recorded by the author's machine and are not independent proof. The original
commits were pushed to GitHub before the rewrite: the plan and the code at 20:31 on 2026-11-15,
before the evaluation started at 20:32. A complete copy of the original history is kept by the
author outside the repository (`backups/original-history-2026-11-16.bundle`, ignored by git) and
can be inspected with `git bundle verify` and `git clone`.

IDs cited in tracked files were updated to the current IDs in the commit that added this document.
The run record `dashboards/data/locked_test_run.json` and the constant `ANALYSIS_PLAN_COMMIT` in
`notebooks/02_cqr_calibration_analysis.py` now hold the current ID of the plan commit; no result
was recomputed.

## Old and new commit IDs

| Original | Current | Commit date | Subject |
| :--- | :--- | :--- | :--- |
| `b418ecf` | `49af7c9` | 2026-11-15 01:53 | feat(sprint-1): complete US-1.1 config, dual Delta/Parquet store, and task runners |
| `356b69e` | `932be82` | 2026-11-15 02:08 | feat(sprint-1): complete US-1.2 Bronze snapshot provenance (Gate G1) and EDA diagnostics |
| `85b3a99` | `67f51f9` | 2026-11-15 02:13 | style(eda): split DE-LU and FR carbon intensity distributions into 1x2 panels |
| `59a51d0` | `eeb516d` | 2026-11-15 02:30 | feat(sprint-1): complete US-1.3 leakage-free features (Gate G2), Split-CQR baseline (90.1% coverage), and fan charts |
| `4f00112` | `5eebd49` | 2026-11-15 03:07 | feat(sprint-2): complete US-2.1 Unity Catalog Feature Store materialization with local fallback |
| `8d96cb1` | `f8bcd08` | 2026-11-15 08:38 | fix(provenance): record exact SHA-256 checksum in bronze_snapshot README |
| `9e1076c` | `729dd32` | 2026-11-15 09:49 | feat(sprint-2): complete US-2.2 Asymmetric Split CQR, MLflow Registry @challenger, and @champion Gate G3 |
| `ba1adc9` | `a159bf7` | 2026-11-15 09:49 | feat(sprint-2): complete US-2.2 Asymmetric Split CQR, MLflow Registry @challenger, and @champion Gate G3 |
| `dc33e7d` | `be2d8bb` | 2026-11-15 09:55 | feat(sprint-2): complete US-2.3 Spark UDF batch scoring, DAB jobs, and GitHub Actions CI |
| `8c8ff28` | `f5da353` | 2026-11-15 12:39 | fix(A): make CI and local test runners trustworthy; add review issue register |
| `7fc6b7f` | `2b57ad8` | 2026-11-15 12:51 | fix(B): reproducible Bronze snapshot, pinned provenance, hour-exact lags |
| `269bf86` | `fc65be5` | 2026-11-15 13:01 | fix(C): purge split boundaries, make Gate G2 falsifiable, seal EDA from test |
| `48c446a` | `20b14fc` | 2026-11-15 13:14 | fix(D): per-horizon CQR with scheduled recalibration and a change-based target |
| `3d003c9` | `a34d8ee` | 2026-11-15 14:31 | test(A-D): close audit gaps, finish Phase D docs, regenerate outputs |
| `590859a` | `0ce762a` | 2026-11-15 14:46 | chore(D): confirm and freeze model selection, repeatable breakage audit, run-time fix |
| `648fd80` | `cae0ba3` | 2026-11-15 15:04 | fix(E): promotion gate re-scores models itself and judges per zone |
| `23bf83d` | `6c8e730` | 2026-11-15 19:46 | docs(F): approved locked-test analysis plan for H1-H3; Phase E done |
| `65f4de2` | `292c883` | 2026-11-15 20:19 | feat(F): locked-test analysis code for H1-H3 (not yet executed on the test) |
| `079e855` | `601a320` | 2026-11-15 20:47 | data(F): locked-test results for the frozen model, as produced |
| `d3d2434` | `7127c90` | 2026-11-15 20:58 | docs(F): locked-test results write-up, register and disclosure |
| `d2845b1` | `2e2479b` | 2026-11-15 21:21 | feat(G): Serverless bundle, job entry points and scheduled recalibration |
| `e8f467e` | `2964d07` | 2026-11-15 22:32 | fix(G15): select the MLflow experiment explicitly in Databricks jobs; workspace results |
| `94083cf` | `0e9486f` | 2026-11-15 23:24 | docs(G): inference job verified on the workspace; checklist steps 7-11 pass |
| `279dea1` | `84f730f` | 2026-11-15 23:25 | docs(G): clarify what the feature-table merge has and has not exercised |
| `056db2b` | `bf22b0f` | 2026-11-15 23:30 | docs(G): checklist step 12 passes; gate verified against an existing champion |
| `a4e6ac3` | `355ec8c` | 2026-11-15 23:34 | docs(G): second inference run leaves the registry in its operating state |
| `2da4336` | `ec99624` | 2026-11-15 23:40 | docs: root README with results, limits, disclosure and data-licence warning |
| `83f3c3f` | `be9def4` | 2026-11-15 23:49 | docs: record verification state after the README (audit 59 of 59) |
| `a82cebe` | `99c994a` | 2026-11-16 14:23 | feat(US-3.2): drift and coverage monitoring |
