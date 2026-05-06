"""Drawing live-learn module for the HF Space demo.

Pretrained 5-digit (0-4) donor that classifies user-drawn 28x28 images
and can be extended to new digit classes (5-9) at runtime via the
trioron's existing extend() API.
"""
from .data import (  # noqa: F401
    sketch_to_tensor,
    load_mnist_subset,
)
from .state import (  # noqa: F401
    DrawingSession,
    PRETRAIN_CLASSES,
    EXTEND_CLASSES,
    TEACH_THRESHOLD,
)
from .predict import predict, teach  # noqa: F401
