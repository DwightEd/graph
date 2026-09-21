"""Install arbitrary representation operations for the duration of a model call."""

from collections import defaultdict
from contextlib import ExitStack, contextmanager
from functools import partial

from .operations import ReplaceSource, apply_operations


@contextmanager
def intervene(model, operations):
    """Group by site: routing edits precede source replacements and head readout hooks."""
    sites = defaultdict(list)
    for operation in operations:
        target = operation.target
        for layer in target.layers or (None,):
            sites[target.representation, layer].append(operation)
    with ExitStack() as stack:
        for (name, layer), selected in sites.items():
            replacements = [op for op in selected if isinstance(op, ReplaceSource)]
            edits = [op for op in selected if not isinstance(op, ReplaceSource)]
            transform = partial(apply_operations, operations=edits)
            if replacements:
                stack.enter_context(model.bind_attention(layer, transform, True, replacements))
            else:
                stack.enter_context(model.bind(name, layer, transform, edit=True))
        yield
