# GPT-5.5 × LexBench overnight stability campaign

Status: complete. All retained reports are standalone Helm `HDOC/1.0` artifacts; machine-readable
evidence lives beside each report.

| Stage | Question | Decision | Report |
| --- | --- | --- | --- |
| 1 | Is there a general backend quality winner after counterbalanced repeats? | No winner established: Lexmount 82/128, Local 80/128, clustered 95% CI crosses both directions. | [report](stage1/report.html) |
| 2 | Are selected site-level directions reproducible, and what accompanies them? | 16/18 holdout directions retained; effects are bidirectional and site-specific. | [report](stage2/report.html) |
| 3 | Is quota 64 real, and what concurrency is formally supported? | Balanced 64 is attainable; task-shaped 28/36 failed twice; c32 is the highest complete formal point. | [report](stage3/report.html) |
| 4 | What can the campaign safely claim and what changes the next decision? | General accuracy is near parity; site stability and runner resource displacement are the supported Lexmount benefits. | [final synthesis](stage4/report.html) |

## Current operating boundary

- Use `c32` as the default backed by complete quality, lifecycle, and resource evidence.
- Before a formal `c64` run, require two consecutive successful `28 EN + 36 ZH` raw admission
  preflights or provider-side reservation/allocation evidence.
- Treat stable site directions as routing/fallback hypotheses until a fresh, unselected holdout confirms them.

## Fixed experiment identity

- Runner: `lexmount/browseruse-agent-bench@bce2c2a17dc2bcf3062b56df4946230c94426cd6`
- Agent: `browser-use` + `gpt-5.5`
- Judge: `gpt-5.4`, per-task threshold stepwise
- Task sample: `experiments/gpt55-lexbench/task_sets/capacity64.txt`
- Task-set SHA-256: `8806a9686ed3b2a46f20da479ac07ceb31ee5f452aba5b18b471ae8801650047`

Raw trajectories and screenshots remain on the run host. The repository contains only summaries,
session probes, comparison outputs, manifests, self-audits, and reports.
