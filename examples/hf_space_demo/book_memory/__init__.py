"""Trioron-as-compressed-book-memory module for the HF Space demo.

Re-export the small public API so the Space's app.py imports cleanly:

    from book_memory import BookMemoryHead, build_head_from_ckpt, pool_query_embeddings
"""
from .model import (  # noqa: F401
    BookMemory,
    BookMemoryHead,
    SoftPromptHead,
    build_head_from_ckpt,
    pool_query_embeddings,
)
from .entity_archive import (  # noqa: F401
    EntityArchive,
    build_archive,
    extract_pairs,
    is_entity_answer,
    pool_question,
)
