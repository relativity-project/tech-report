"""rl_timing.py TRAIN.log...: mean per-step phase times of Miles' RL runs over their final 15 steps.

Miles logs, per step, perf/rollout_time (generation), perf/log_probs_time, perf/actor_train_time (the optimizer
steps) and perf/update_weights_time (the weight transfer); the report's RL section quotes their means.
"""
import re
import statistics
import sys

KEYS = ["rollout_time", "log_probs_time", "actor_train_time", "update_weights_time", "step_time"]
for path in sys.argv[1:]:
    text = re.sub(r"\x1b\[[0-9;]*m", "", open(path, errors="replace").read())
    per_step = {}
    for m in re.finditer(r"perf (\d+): (\{[^}]*\})", text):
        per_step[int(m[1])] = per_step.get(int(m[1]), "") + m[2]  # the rollout executor's and the trainer's lines
    last = sorted(per_step)[-15:]
    means = {}
    for k in KEYS:
        vals = [float(v[0]) for s in last if (v := re.findall(rf"'perf/{k}': ([0-9.]+)", per_step[s]))]
        means[k] = round(statistics.mean(vals), 1) if vals else None
    print(path, f"steps {last[0] + 1}..{last[-1] + 1}" if last else "no steps", means)
