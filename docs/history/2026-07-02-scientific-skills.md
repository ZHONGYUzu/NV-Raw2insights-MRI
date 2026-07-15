# Daily Log: Scientific Agent Skills Setup

> Historical environment/setup record. Skill availability can change, so the
> installed-skill list below is evidence for 2026-07-02, not a current runtime
> inventory.

Date: 2026-07-02

## Context

Today we extended the local Codex environment with scientific-agent skills for the custom CINE MRI reconstruction and evaluation workflow.

The skills were installed from:

```text
K-Dense-AI/scientific-agent-skills
```

The installer cloned the GitHub repository, discovered `149` available skills, and selected the requested `9`. Installation completed successfully with project scope.

## Installation Command

```bash
npx skills add K-Dense-AI/scientific-agent-skills \
  --skill scientific-visualization \
  --skill matplotlib \
  --skill statistical-analysis \
  --skill optimize-for-gpu \
  --skill get-available-resources \
  --skill scientific-writing \
  --skill citation-management \
  --skill paper-lookup \
  --skill xlsx
```

## Confirmed Installation Result

- Source: `https://github.com/K-Dense-AI/scientific-agent-skills.git`
- Available skills discovered: `149`
- Skills selected and installed: `9`
- Installation scope: project
- Project skill root: `.agents/skills/`
- Agent targets selected by the installer: `13`, including Codex
- Installer status: `Installation complete`

The project-scoped skill directories are:

```text
.agents/skills/citation-management
.agents/skills/get-available-resources
.agents/skills/matplotlib
.agents/skills/optimize-for-gpu
.agents/skills/paper-lookup
.agents/skills/scientific-visualization
.agents/skills/scientific-writing
.agents/skills/statistical-analysis
.agents/skills/xlsx
```

## Installed Skills

| Skill | Intended use in this project |
| --- | --- |
| `scientific-visualization` | Design clear reconstruction, ground-truth, error-map, and acceleration-comparison figures. |
| `matplotlib` | Implement publication-quality plots, metric distributions, image grids, and CINE visualizations. |
| `statistical-analysis` | Compare reconstruction metrics across accelerations, subjects, slices, and cardiac frames. |
| `optimize-for-gpu` | Profile and improve GPU utilization for training and inference where appropriate. |
| `get-available-resources` | Inspect available compute resources before selecting expensive experiments. |
| `scientific-writing` | Draft experiment descriptions, methods, results, limitations, and research notes. |
| `citation-management` | Organize and format references used in reports and papers. |
| `paper-lookup` | Find relevant MRI reconstruction and deep-learning literature. |
| `xlsx` | Export and review experiment metrics in spreadsheet form. |

## Additional Skill Installed

The installer offered the optional `find-skills` helper, and the prompt was accepted. It was installed separately from `vercel-labs/skills` with global user scope:

```text
~/.agents/skills/find-skills
```

This helper is intended to discover and suggest additional skills. It is not part of the nine project-scoped K-Dense scientific skills.

## Security Assessment Report

The installation transcript reported the following automated assessments:

| Skill | Gen | Socket | Snyk |
| --- | --- | ---: | --- |
| `citation-management` | Safe | 0 alerts | Medium risk |
| `get-available-resources` | Safe | 0 alerts | Low risk |
| `matplotlib` | Safe | 0 alerts | Low risk |
| `optimize-for-gpu` | Safe | 0 alerts | Low risk |
| `paper-lookup` | Safe | 0 alerts | Medium risk |
| `scientific-visualization` | Safe | 0 alerts | Low risk |
| `scientific-writing` | Safe | 1 alert | Medium risk |
| `statistical-analysis` | Safe | 0 alerts | Low risk |
| `xlsx` | Medium risk | 0 alerts | Low risk |
| `find-skills` | Safe | 0 alerts | Medium risk |

These are automated installer reports, not a manual security audit. The installer explicitly warns that skills run with full agent permissions, so skill instructions and bundled scripts should be reviewed before first use. In particular, review `scientific-writing` because Socket reported one alert, and review `xlsx` because the Gen assessment reported medium risk.

## Planned Use For The CINE MRI Workflow

The new skills can support the current project in four main areas:

1. Evaluate acc8, acc16, and acc24 reconstructions using full-size frame-level metrics.
2. Produce consistent GT, reconstruction, absolute-error, boxplot, violin, and temporal visualizations.
3. Summarize results statistically by acceleration, subject, slice, and cardiac frame.
4. Turn experiment outputs into traceable tables, figures, citations, and scientific prose.

## Current Experiment Context

The active research goal remains reconstructing custom CINE 2D MRI data with NV-Raw2Insights-MRI and evaluating predictions against normalized fully sampled ground truth.

The established experiment mapping is:

| Experiment | Acceleration | Dataset root | Output root |
| --- | ---: | --- | --- |
| R1 | 8 | `dataset/CustomCINEDataR1` | `output/CustomCINEOutputR1` |
| R2 | 16 | `dataset/CustomCINEDataR2` | `output/CustomCINEOutputR2` |
| R3 | 24 | `dataset/CustomCINEDataR3` | `output/CustomCINEOutputR3` |

Server-only inference and other GPU-heavy work should still be run on the server. Local Codex work should favor code inspection, documentation, lightweight validation, and plotting against already-produced results.

## Session Note

The transcript confirms that all nine requested scientific skills and the optional `find-skills` helper were installed successfully. A new Codex session or skill-list refresh may still be required before newly installed skills appear in the active session. No inference, evaluation, statistical analysis, or figure generation was performed as part of this setup entry.

## Next Steps

- Confirm the new skills appear in a fresh Codex session.
- Review each installed `SKILL.md` and any bundled scripts before first use.
- Investigate the reported `scientific-writing` Socket alert and the `xlsx` Gen medium-risk rating.
- Validate the R1, R2, and R3 result folders before analysis.
- Run full-frame metric evaluation across the selected subjects.
- Generate comparison figures and spreadsheet summaries.
- Record normalization assumptions and statistical methods alongside reported results.
