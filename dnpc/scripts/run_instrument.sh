#!/usr/bin/env bash
# Frame-stride identification across scenes and datasets: does the null replicate?
set -u
PY=/home/khm/.conda/envs/dggt/bin/python
cd /home/khm/3D_4D/dnpc
LOG=outputs/stage0/instrument_log.txt
: > "$LOG"
go() { echo "=== $* ===" | tee -a "$LOG"; $PY scripts/stage0_probe.py "$@" 2>&1 \
       | grep -vE "UserWarning|warnings.warn|TORCH_CUDA" | tee -a "$LOG"; }

for s in whiteroom complete_kitchen; do
  for st in 2 4 8; do
    go --dataset nrgbd --scene "$s" --tag "str$st" --frame-stride $st \
       --iters-per-frame 20 --replay-horizon 30 --checkpoint-every 12
  done
done
for st in 2 4 8; do
  go --dataset tum --scene rgbd_dataset_freiburg3_long_office_household --tag "str$st" \
     --frame-stride $st --iters-per-frame 20 --replay-horizon 30 --checkpoint-every 12
done
echo "ALL_DONE" | tee -a "$LOG"
