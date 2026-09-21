"""Install arbitrary representation operations for the duration of a model call."""

from collections import defaultdict
from contextlib import ExitStack, contextmanager
from functools import partial

from .operations import apply_operations


@contextmanager
def intervene(model, operations):
    """Group by representation/layer; preserve the order within each site."""
    sites = defaultdict(list)
    for operation in operations:
        target = operation.target
        for layer in target.layers or (None,):
            sites[target.representation, layer].append(operation)
    with ExitStack() as stack:
        for (name, layer), selected in sites.items():
            transform = partial(apply_operations, operations=selected)
            stack.enter_context(model.bind(name, layer, transform, edit=True))
        yield
