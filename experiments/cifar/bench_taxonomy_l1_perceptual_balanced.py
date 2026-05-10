"""L1 — 4-way perceptual taxonomy with balanced fine-class counts.

Removes the class-imbalance contamination of the 16-class run by
subsampling each macrocluster to 2 fine classes, so the 4 perceptual
labels each have 1000 train / 200 test images. With balanced labels:
  * macro-avg = micro-avg (no Simpson's-paradox confusion)
  * always-predict-largest baseline = 0.25 (vs 0.5625 in imbalanced)
  * dynamic range above chance is 0.75 (vs 0.4375 imbalanced)
so any genuine fit advantage of the trioron-discovered taxonomy
should now be visible as a wide margin above 0.25.

The 8 fine classes were picked to keep one fine class from each *sub-
cluster* of central-object discovered at k=8 of the dendrogram, so the
balanced subset still spans the natural perceptual diversity.
"""
from __future__ import annotations
import sys
import os

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Override the macrocluster definitions before bench_taxonomy_l1_perceptual
# imports them. We rebind the dict in that module's namespace.
from experiments.cifar import bench_taxonomy_l1_perceptual as base


BALANCED_PERCEPTUAL_GROUPS = {
    0: {"label": "compact-object",
        "names": ["chair", "cup"]},
    1: {"label": "central-object",
        "names": ["wolf", "motorcycle"]},          # mid-mammal + vehicle
    2: {"label": "horizontal-landscape",
        "names": ["dolphin", "mountain"]},
    3: {"label": "vertical-landscape",
        "names": ["oak_tree", "castle"]},
}


def main(argv=None) -> int:
    base.PERCEPTUAL_GROUPS = BALANCED_PERCEPTUAL_GROUPS
    if argv is None:
        argv = sys.argv[1:]
    # Default the output path to a balanced-tagged file so we don't
    # clobber the imbalanced donor.
    if not any(a.startswith("--out-path") for a in argv):
        argv = list(argv) + [
            "--out-path",
            "outputs/cifar_taxonomy/donor_l1_perceptual_4way_balanced.pt",
        ]
    return base.main(argv)


if __name__ == "__main__":
    sys.exit(main())
