# Trioron — Step 1: TrioronLayer

Foundational unit for the dynamic architecture. See `trioron_blueprint.md` for the full design.

## Setup (WSL)

```bash
# 1. Move this folder into your WSL filesystem (NOT /mnt/c — that's slow)
mv /mnt/c/path/to/outputs ~/trioron-project
cd ~/trioron-project

# 2. (Recommended) use a venv so you don't pollute system Python
python3 -m venv .venv
source .venv/bin/activate

# 3. Install
pip install -r requirements.txt
```

Torch CPU wheel is ~750 MB. First install is the slow part; after that you're done.

## Run the tests

```bash
python3 test_node.py        # step 1: TrioronLayer (21 tests)
python3 test_network.py     # step 2: TrioronNetwork (12 tests)
```

Expected on each: all PASS, 0 FAIL.

## Run the step-2 experiment

```bash
python3 experiments/continual_2task.py
```

Trains a 3-layer Trioron network on synthetic Task A, anchors, then trains
on Task B with and without EWC. Prints a comparison table at the end. The
WITH-EWC line should show lower Task-A loss than the NO-EWC control —
that's the multi-layer continual-learning evidence.

## Layout

```
trioron-project/
├── trioron_blueprint.md             # full design doc
├── README.md                        # this file
├── requirements.txt
├── trioron/
│   ├── __init__.py
│   ├── node.py                      # TrioronLayer
│   └── network.py                   # TrioronNetwork
├── experiments/
│   ├── __init__.py
│   └── continual_2task.py           # step 2 experiment
├── test_node.py                     # step 1 tests
└── test_network.py                  # step 2 tests
```

## Status

- [x] Step 1: TrioronLayer + tests
- [~] Step 2: TrioronNetwork + 2-task continual-learning verification (awaiting verification)
- [ ] Step 3: Scripted incubation environment
- [ ] Step 4: Three-condition growth trigger
- [ ] Step 5: Cellular division routine
- [ ] Step 6: Pruning loop
- [ ] Step 7: Hard ceilings (VRAM + time)
- [ ] Step 8: Benchmark vs same-param fixed MLP (falsification gate)
- [ ] Step 9 (deferred): Language adapter (cloud only)
