#!/usr/bin/env bash
set -u
PY=/home/khm/.conda/envs/dggt/bin/python
cd /home/khm/3D_4D/dnpc
LOG=outputs/stage0/run_log.txt
: > "$LOG"
run() { echo "=== $* ===" | tee -a "$LOG"; $PY scripts/stage0_probe.py "$@" 2>&1 \
        | grep -vE "UserWarning|warnings.warn|TORCH_CUDA" | tee -a "$LOG"; }

for s in staircase whiteroom complete_kitchen; do
  run --dataset nrgbd --scene "$s" --tag main --frame-stride 2 --iters-per-frame 20 --checkpoint-every 25
done
run --dataset tum --scene rgbd_dataset_freiburg3_long_office_household --tag main \
    --frame-stride 2 --iters-per-frame 20 --checkpoint-every 25
run --dataset tum --scene rgbd_dataset_freiburg2_xyz --tag main \
    --frame-stride 4 --iters-per-frame 20 --checkpoint-every 25

for k in 5 10 20 50; do
  run --dataset nrgbd --scene staircase --tag "age$k" --frame-stride 2 \
      --iters-per-frame 20 --freeze-age-k "$k" --checkpoint-every 0
done
echo "ALL_DONE" | tee -a "$LOG"
