"""Local object watchlist support."""

from cvti.object_watch.store import (
    EmbeddingRecord,
    ObjectExample,
    ObjectTarget,
    add_example,
    load_embeddings,
    load_targets,
    save_target,
    targets_needing_reembed,
    write_embedding,
)

__all__ = [
    "EmbeddingRecord",
    "ObjectExample",
    "ObjectTarget",
    "add_example",
    "load_embeddings",
    "load_targets",
    "save_target",
    "targets_needing_reembed",
    "write_embedding",
]
