#!/usr/bin/env bash
# Sequential n=10 reruns of all chained-15 panels needed for the
# ArXiv-ready paper draft. Total wall-time depends on hardware; designed
# to run unattended in the background.
#
# Order (most-headline first so a partial run still gives the most
# important update): manifold replay, hippocampal K=50, online EWC, LwF,
# PackNet, HAT.
#
# Run with:
#   bash experiments/run_n10_paper.sh > outputs/run_n10_paper_master.log 2>&1 &

set -u  # don't set -e — we want to keep going if one panel fails

cd "$(dirname "$0")/.."
mkdir -p outputs

run_panel() {
    local mod="$1"
    local logname="$2"
    local logpath="outputs/${logname}"
    echo "================================================================="
    echo "[run_n10_paper] starting ${mod}  ->  ${logpath}  at $(date -Is)"
    echo "================================================================="
    python3 -m "experiments.${mod}" > "${logpath}" 2>&1
    local rc=$?
    echo "[run_n10_paper] finished ${mod} with rc=${rc} at $(date -Is)"
    return $rc
}

run_panel bench_manifold_replay_n10        bench_chained_15task_n10_MANIFOLD_REPLAY.log
run_panel bench_hippo_k50_n10              bench_chained_15task_n10_HIPPO_K50.log
run_panel bench_online_ewc_chained_15_n10  bench_chained_15task_n10_ONLINE_EWC.log
run_panel bench_lwf_chained_15_n10         bench_chained_15task_n10_LWF.log
run_panel bench_packnet_chained_15_n10     bench_chained_15task_n10_PACKNET.log
run_panel bench_hat_chained_15_n10         bench_chained_15task_n10_HAT.log

echo
echo "[run_n10_paper] all panels done at $(date -Is)"
