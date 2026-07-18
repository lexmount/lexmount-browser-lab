# Controlled Browser Parity Suite

This is a deterministic, action-bearing compatibility suite for the existing
WebVoyager-trained policy contract: `observe -> fill -> click -> observe -> answer`.
It is intentionally narrower than WebVoyager or LexBench. Its purpose is to
isolate browser-session and DOM/action compatibility from changing public sites,
egress IPs, and LLM judging.

The task page must be served from one public URL reachable by both backends.
Render the same task manifest for both arms:

```bash
python experiments/controlled-browser-parity/generate_tasks.py \
  --base-url https://your-public-fixture.example/ \
  --output /data/wf/sxh/controlled-parity/tasks-50.jsonl
```

Run `training/scripts/webvoyager_posttrain_eval.py run` with `--judge exact`.
The task's `expected_answer.must_include` list is scored locally; no external
judge is used. Run the availability probe first, then compare only the shared
available tasks and preserve the normal run manifests and raw trajectories.

The suite supports a conclusion about this browser/tool contract only. It must
not be presented as a WebVoyager, LexBench, or general browser-agent quality
score.
