#!/usr/bin/env bash
# Pairwise interaction ablation on the manifold-replay headline config.
# Locates the destructive -0.043 interaction in the all-axes arm. Per-axis
# singles already established: Axis3 -0.036, Axis4 -0.059, Axis5+5.5 -0.028.
# Linear sum -0.123 vs observed all-axes -0.166 → -0.043 excess.
#
# Three pairs to localize where the interaction lives:
#   3+4         : both linear axes together (no Axis 5 / plastic)
#   3+5+5.5     : insert_layer interacting with dendritic growth on the new layer
#   4+5+5.5     : axonal_gain modulation interacting with plastic rewiring
#
# n=3 seeds {0,1,2}, full epoch budget. ~30-45 min total.
#
# Run with:
#   bash experiments/run_pairwise_ablation.sh \
#       > outputs/run_pairwise_ablation.log 2>&1 &

set -u

cd "$(dirname "$0")/.."
mkdir -p outputs

run_condition() {
    local name="$1"
    shift
    local logname="outputs/bench_chained_15task_n3_MANIFOLD_REPLAY_${name}.log"
    local csvname="bench_chained_15task_n3_MANIFOLD_REPLAY_${name}.csv"
    echo "================================================================="
    echo "[pairwise] starting ${name} at $(date -Is)"
    echo "  env: $*"
    echo "================================================================="
    env "$@" python3 -m experiments.bench_chained_15task \
        --seeds 0,1,2 \
        --arms grown_uncapped_dream \
        --csv "${csvname}" \
        > "${logname}" 2>&1
    local rc=$?
    echo "[pairwise] finished ${name} with rc=${rc} at $(date -Is)"
}

# Pair: Axis 3 + Axis 4 (both linear axes, no Axis 5 / plastic).
run_condition "axis3_plus_4" \
    TRIORON_MANIFOLD_REPLAY=1 \
    TRIORON_AXIS_3=1 \
    TRIORON_AXIS_4=1

# Pair: Axis 3 + Axis 5 + plastic (insert_layer × dendritic growth on it).
run_condition "axis3_plus_5_plastic" \
    TRIORON_MANIFOLD_REPLAY=1 \
    TRIORON_AXIS_3=1 \
    TRIORON_AXIS_5=1 \
    TRIORON_PLASTIC=1 \
    TRIORON_PLASTIC_REWIRE_STEPS=50

# Pair: Axis 4 + Axis 5 + plastic (axonal_gain × plastic rewiring).
run_condition "axis4_plus_5_plastic" \
    TRIORON_MANIFOLD_REPLAY=1 \
    TRIORON_AXIS_4=1 \
    TRIORON_AXIS_5=1 \
    TRIORON_PLASTIC=1 \
    TRIORON_PLASTIC_REWIRE_STEPS=50

echo
echo "[pairwise] all conditions done at $(date -Is)"
