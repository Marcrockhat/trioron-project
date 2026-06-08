#!/usr/bin/env bash
# Per-axis ablation on the manifold-replay headline config (grown_uncapped_dream).
# Each axis on its own, plus Axis 5 + 5.5 combined (plastic needs Axis 5 to
# grow branches first; "plastic only" is effectively no-axes).
#
# n=3 seeds {0,1,2}, full epoch budget. Compare against the baseline at
# outputs/bench_chained_15task_n3_MANIFOLD_REPLAY_baseline_multiseed.csv.
#
# Wall-clock ~12-15 min per condition × 4 conditions = ~50-60 min total.
#
# Run with:
#   bash experiments/run_per_axis_ablation.sh \
#       > outputs/run_per_axis_ablation.log 2>&1 &

set -u

cd "$(dirname "$0")/.."
mkdir -p outputs

run_condition() {
    local name="$1"
    shift
    local logname="outputs/bench_chained_15task_n3_MANIFOLD_REPLAY_${name}.log"
    local csvname="bench_chained_15task_n3_MANIFOLD_REPLAY_${name}.csv"
    echo "================================================================="
    echo "[per_axis_ablation] starting ${name} at $(date -Is)"
    echo "  env: $*"
    echo "================================================================="
    env "$@" python3 -m experiments.bench_chained_15task \
        --seeds 0,1,2 \
        --arms grown_uncapped_dream \
        --csv "${csvname}" \
        > "${logname}" 2>&1
    local rc=$?
    echo "[per_axis_ablation] finished ${name} with rc=${rc} at $(date -Is)"
}

# Axis 3 only — insert_layer at startup.
run_condition "axis3_only" \
    TRIORON_MANIFOLD_REPLAY=1 \
    TRIORON_AXIS_3=1

# Axis 4 only — axonal_gain modulation per task.
run_condition "axis4_only" \
    TRIORON_MANIFOLD_REPLAY=1 \
    TRIORON_AXIS_4=1

# Axis 5 only — dendritic branch growth, no plastic rewiring.
run_condition "axis5_only" \
    TRIORON_MANIFOLD_REPLAY=1 \
    TRIORON_AXIS_5=1

# Axis 5 + 5.5 — dendritic growth plus plastic rewiring after each grow.
run_condition "axis5_plus_plastic" \
    TRIORON_MANIFOLD_REPLAY=1 \
    TRIORON_AXIS_5=1 \
    TRIORON_PLASTIC=1 \
    TRIORON_PLASTIC_REWIRE_STEPS=50

echo
echo "[per_axis_ablation] all conditions done at $(date -Is)"
