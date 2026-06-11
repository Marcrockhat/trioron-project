"""Unit test: the sweep reproduces Rocky's chicken/dog worked example exactly
(s028 design §3, verified spec). Run: python3 test_feed.py  (or pytest)."""
from collections import Counter

from feed import CHICKEN, DOG, EMPTY, accumulate, votes_at

CLASSES = {"chicken": CHICKEN, "dog": DOG}
N_FEATURES = 4


def test_worked_example():
    # v=0: f1 'ext' contains 0 → votes both; f0/f2/f3 empty
    assert votes_at(CLASSES, N_FEATURES, 0) == Counter({EMPTY: 3, "chicken": 1, "dog": 1})
    # v=1: f0∈[1,5] → chicken; rest empty
    assert votes_at(CLASSES, N_FEATURES, 1) == Counter({EMPTY: 3, "chicken": 1})
    # v=8: f0∈[6,10] → dog
    assert votes_at(CLASSES, N_FEATURES, 8) == Counter({EMPTY: 3, "dog": 1})
    # v=35: dog f2∈[30,50] & f3∈[35,80]; chicken f3∈[30,40]
    assert votes_at(CLASSES, N_FEATURES, 35) == Counter({EMPTY: 2, "dog": 2, "chicken": 1})
    # v=90: f2∈[80,100] → chicken
    assert votes_at(CLASSES, N_FEATURES, 90) == Counter({EMPTY: 3, "chicken": 1})
    # v=1000: the other 'ext' extreme → votes both again
    assert votes_at(CLASSES, N_FEATURES, 1000) == Counter({EMPTY: 3, "chicken": 1, "dog": 1})


def test_period_accumulation():
    total = accumulate(CLASSES, N_FEATURES)
    # chicken: f0 5 + f1 2 + f2 21 + f3 11 quanta of slicing
    assert total["chicken"] == 5 + 2 + 21 + 11
    # dog: f0 5 + f1 2 + f2 21 + f3 46
    assert total["dog"] == 5 + 2 + 21 + 46
    # every (v, feature) pair that slices nothing is an empty vote
    assert total[EMPTY] == 4 * 1001 - (5 + 5) - 2 - (21 + 21) - (11 + 46 - 6)


if __name__ == "__main__":
    test_worked_example()
    test_period_accumulation()
    print("feed: all assertions pass (chicken/dog worked example reproduced)")
