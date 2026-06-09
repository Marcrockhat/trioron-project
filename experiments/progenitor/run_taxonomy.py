"""Run the in-place council decision on the 10-species / 4-feature taxonomy.

Same living-organism loop as ``step3c_council_decides.main`` (the 6-animal probe) —
only the council's I/O width changes: 7-d input (log-weight, log-height, onehot
cover[4], eggs) → 10 outputs (incl. turtle). Tests whether the council's autonomous
type+count decision generalizes from the 1-feature toy to a richer tabular taxonomy
where more than one species can be frustrated.

Run: python3 -m experiments.progenitor.run_taxonomy
"""
from __future__ import annotations

from .data_taxonomy import make_taxonomy, bayes_accuracy, bayes_per_class
from .step3c_council_decides import run_decision


def main() -> None:
    train_d = make_taxonomy(seed=0, n_per_class=256)
    test_d = make_taxonomy(seed=1, n_per_class=512)
    n_in = test_d.x.shape[1]          # 7
    n_out = len(test_d.names)          # 10
    run_decision(train_d, test_d, bayes_per_class(test_d),
                 n_in=n_in, n_out=n_out, bayes_overall=bayes_accuracy(test_d))


if __name__ == "__main__":
    main()
