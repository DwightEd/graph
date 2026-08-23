"""Bound the lifetime of one sample's loaded attention tensors."""

from contextlib import contextmanager


@contextmanager
def loaded_attention(sample):
    """Load one sample and release its attention on every exit path."""

    try:
        yield sample.attention()
    finally:
        sample.release_attention()
