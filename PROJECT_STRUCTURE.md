# Project structure

This directory is the canonical working project. Run commands from here.

## Active project content

```text
wind_turbine_predictive_maintenance/
|-- README.md                    Main project overview and run instructions
|-- PROJECT_STRUCTURE.md         This folder map
|-- RESEARCH_PLAN.md             Research questions and work packages
|-- *.py                         Pipeline and experiment source code
|-- tests/                       Regression and evaluation tests
|-- data/
|   `-- CARE_To_Compare/         Unzipped real CARE dataset (~20 GB)
`-- results/
    |-- *.csv, *.png             Legacy pre-correction synthetic outputs
    |-- synthetic/               Reserved for corrected synthetic runs
    `-- care/                    Saved CARE Farm A/B CSV and PNG outputs
```

The extended multi-farm benchmark is implemented by `care_benchmark.py` and
`care_results_analysis.py`. Its current full outputs are under
`results/care_benchmark/full_2026-07-23/`.

The Python files remain together at the project root because they import one
another directly. The result folders contain the current saved evidence, while
new script executions may initially write regenerated outputs to the working
directory.

## Material outside the active project

The sibling directory `../_extras_archive/` is preserved for reference but is
not part of the active project:

- `zip_snapshots/` contains downloaded and historical ZIP archives.
- `docs/` contains personal project stories and report-planning material.
- `older_docs/` contains superseded root documentation and the old
  `for ourself` packaging.
- `handoff_notes/` contains the unverified handoff document reviewed on
  2026-07-23.
- `generated_cache/` contains Python bytecode cache files.

Nothing was deleted during organization. Do not edit or cite archived copies
unless they are deliberately restored into the active project.
