# Historical Notes And Run Logs

Files in this directory preserve dated evidence, implementation investigations,
and completed experiment records. They are not authoritative instructions for
new runs unless the active SOP explicitly links to them.

| File | Date/status | Scope |
| --- | --- | --- |
| [`2026-08-05-server-run-audit.md`](2026-08-05-server-run-audit.md) | 2026-08-05 snapshot | Historical server outputs and metric gaps; not live status. |
| [`2026-08-11-vista-mask-generalization-run-plan.md`](2026-08-11-vista-mask-generalization-run-plan.md) | 2026-08-11 plan | Raw-VISTA seed experiments; later seed-8 record in [SERVER_WORKFLOW.md](../../SERVER_WORKFLOW.md). |
| [`2026-09-08-local-assets-cleanup.md`](2026-09-08-local-assets-cleanup.md) | Local cleanup | Retained documents, ignored reading/results material, and active work exclusions. |
| [`2026-06-23-external-sensitivity-map-analysis.md`](2026-06-23-external-sensitivity-map-analysis.md) | Superseded analysis | Pre-implementation review that concluded external sensitivity maps were not yet wired into inference. |
| [`2026-06-23-smap-acc-acs.md`](2026-06-23-smap-acc-acs.md) | Historical development log | External-map implementation and the 132×176 CINE1 experiment plan. |
| [`2026-07-02-scientific-skills.md`](2026-07-02-scientific-skills.md) | Historical setup log | Scientific-agent skill installation evidence. |
| [`2026-07-14-h5-dmap-noacs-acc8-inference.md`](2026-07-14-h5-dmap-noacs-acc8-inference.md) | Completed run | 10-case external-`dMap`, no-forced-ACS acc8 inference with FE 192. |

The apparent sensitivity-map conflict is chronological: the June 23 analysis
captured the missing interface before it was implemented; later code and the
July 14–15 run confirm that inference support exists now.
