"""CIFAR-100 sense-conductor experiment.

trioron sees only sense readings, never pixels. Each primitive donor is
trained on one sense's output; the conductor (multi-branch organism)
fuses readings from multiple senses into a final classification.
"""
