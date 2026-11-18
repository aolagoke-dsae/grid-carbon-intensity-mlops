# Slide data and chart code

Everything needed to check, or redraw, the charts and figures shown in the project's presentation.

| File | What it is |
| :--- | :--- |
| `export_deck_data.py` | Computes every number the slides show, from the project's result tables, the committed snapshot and the findings register |
| `deck_data.json` | Those numbers, as last exported. Open it to check a slide without running anything |
| `build_deck.py` | Draws the slides from `deck_data.json` as an editable PowerPoint file with native charts |

The PowerPoint files themselves are not in the repository. `build_deck.py` writes them to
`dashboards/design/`, which Git ignores.

## Re-running it

From the repository root:

| Step | Windows | Linux or macOS |
| :--- | :--- | :--- |
| Recompute the slide data | `.\scripts\dev.ps1 slides-data` | `make slides-data` |
| Recompute it and build both decks | `.\scripts\dev.ps1 slides` | `make slides` |

Building the decks needs two extra packages: `pip install -e ".[slides]"` (python-pptx and segno).
Recomputing the data needs only the project's own environment.

In PowerPoint, right-click any chart and choose Edit Data to see the numbers it was drawn from.

## How it stays current

- **A test fails when the data is out of date.** `tests/test_slides_data.py` runs with every test
  run and in CI. `deck_data.json` records a SHA-256 of each file it was built from; the test
  compares those with the files as they are now, and re-reads the counts and quoted figures. If
  anything differs it names what changed and the command to run.
- **Refreshing the result tables refreshes the slide data.** The `dashboards` task rewrites the
  tables in `dashboards/data/` and then runs the export.
- **Nothing commits itself.** After a refresh, `deck_data.json` shows up as a changed file to
  review and commit, and the decks are rebuilt with the `slides` task.

What the test does not see: it does not recompute the figures, so a change to the package's code
that leaves every input file unchanged is not flagged here. The package's own tests cover that code.

## Where each chart's numbers come from

`deck_data.json` is grouped by slide. The sources are:

| Group in `deck_data.json` | Shown on | Source |
| :--- | :--- | :--- |
| `fan` | Title and stakeholder slides: one real forecast | `dashboards/data/forecast_history.csv` |
| `problem` | Motivation: average summer and winter days | The committed snapshot, read by `ingest_bronze_snapshot` |
| `template` | Data introduction, exploration charts, baseline and model comparison tables | The snapshot before the held-out period; `locked_test_calibration.csv`; the model-size table in `docs/REVIEW_ISSUES.md` |
| `method` | The four-way split and model settings | `chronological_four_way_split`, `CarbonMLOpsConfig`, `DEFAULT_LGBM_PARAMS` |
| `trust` | Calibration: coverage overall, by lead time and week by week | `locked_test_calibration.csv`, `gold_monitoring.csv` |
| `hypotheses` | Hypothesis outcomes | `hypothesis_results.csv` |
| `value` | Business value: tonnes avoided and bad days | `window_strategy_summary.csv` |
| `limits` | Monitoring: skill and value week by week | `gold_monitoring.csv`, `window_value_weekly.csv` |
| `summary` | The three answers | `locked_test_calibration.csv` |
| `backup` | Backup slides: skill by lead time, coverage by quartile and month, value by window length | `locked_test_calibration.csv`, `window_strategy_summary.csv`, `docs/REVIEW_ISSUES.md` |
| `mlops` | Counts of tests and deliberate breakages | pytest; `scripts/mutation_audit.py` |
| `facts` | Figures quoted in slide text | `docs/REVIEW_ISSUES.md`; the model-size table |
| `sources` | Not shown: the fingerprint of each input file | |

Two figures are typed into the export and not computed: the champion model version and the number
of recalibration runs on the workspace, both recorded in `docs/DATABRICKS_VERIFICATION.md`.
