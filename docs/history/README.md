# Historical Notes And Run Logs

Files in this directory preserve dated evidence, implementation investigations,
and completed experiment records. They are not authoritative instructions for
new runs unless the active SOP explicitly links to them.

| File | Date/status | Scope |
| --- | --- | --- |
| [`2026-06-23-external-sensitivity-map-analysis.md`](2026-06-23-external-sensitivity-map-analysis.md) | Superseded analysis | Pre-implementation review that concluded external sensitivity maps were not yet wired into inference. |
| [`2026-06-23-smap-acc-acs.md`](2026-06-23-smap-acc-acs.md) | Historical development log | External-map implementation and the 132×176 CINE1 experiment plan. |
| [`2026-07-02-scientific-skills.md`](2026-07-02-scientific-skills.md) | Historical setup log | Scientific-agent skill installation evidence. |
| [`2026-07-14-h5-dmap-noacs-acc8-inference.md`](2026-07-14-h5-dmap-noacs-acc8-inference.md) | Completed run | 10-case external-`dMap`, no-forced-ACS acc8 inference with FE 192. |

The apparent sensitivity-map conflict is chronological: the June 23 analysis
captured the missing interface before it was implemented; later code and the
July 14–15 run confirm that inference support exists now.
