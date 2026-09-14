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
from cvti.object_watch.embeddings import (
    EmbeddingBackend,
    HashEmbeddingBackend,
    embed_examples,
    load_embedding_backend,
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
    "EmbeddingBackend",
    "HashEmbeddingBackend",
    "embed_examples",
    "load_embedding_backend",
]
